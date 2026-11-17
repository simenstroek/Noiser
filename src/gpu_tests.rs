//! Tests that run the generated WGSL on a real GPU: tiling seams, continuity and output range
//! for every noise kind. They skip (pass with a note) when no GPU adapter is available.

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::{Graph, NodeKind, NoiseKind, NoiseParams, WorleyFeature};
use eframe::wgpu;

/// GPU tests run one at a time. Several tests creating Vulkan devices and waiting on them at once
/// intermittently deadlocked the whole suite (the stuck tests were always GPU tests).
static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Held until the test drops the device, so the next GPU test starts afterwards.
    _serial: std::sync::MutexGuard<'static, ()>,
}

pub(crate) fn gpu() -> Option<Gpu> {
    // A test that panicked while holding the lock poisons it; the lock has no data to protect.
    let serial = GPU_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(Gpu { device, queue, _serial: serial })
}

/// Evaluates the graph's density at each point.
fn eval(gpu: &Gpu, graph: &Graph, points: &[[f32; 3]]) -> Vec<f32> {
    let c = codegen::compile(graph).unwrap();
    let pts: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    eval_expr(gpu, &c, "graph_density(q.xyz)", &pts)
}

/// Evaluates a WGSL f32 expression of `q` (the point, a vec4f) next to the compiled graph.
fn eval_expr(gpu: &Gpu, c: &codegen::Compiled, expr: &str, points: &[[f32; 4]]) -> Vec<f32> {
    use wgpu::util::DeviceExt;
    let src = format!(
        "{NOISE_LIB}\n{}\n{EVAL_BINDINGS}
@group(0) @binding(4) var<storage, read> pts: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x < arrayLength(&out)) {{ let q = pts[id.x]; out[id.x] = {expr}; }}
}}",
        c.wgsl
    );
    let d = &gpu.device;
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    // Explicit layout: an automatic one drops bindings the graph doesn't use (e.g. blue noise).
    let entry_layout = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry_layout(0, wgpu::BufferBindingType::Uniform),
            entry_layout(1, ro),
            entry_layout(3, ro),
            entry_layout(4, ro),
            entry_layout(5, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init = |contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage });
    let uniforms = init(bytemuck::bytes_of(&[0u32; 4]), wgpu::BufferUsages::UNIFORM);
    let params = init(bytemuck::cast_slice(&c.params), wgpu::BufferUsages::STORAGE);
    let blue = init(bytemuck::cast_slice(&crate::bluenoise::tables()), wgpu::BufferUsages::STORAGE);
    let pts = init(bytemuck::cast_slice(points), wgpu::BufferUsages::STORAGE);
    let size = (points.len() * 4) as u64;
    let out = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let read = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
    }
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[entry(0, &uniforms), entry(1, &params), entry(3, &blue), entry(4, &pts), entry(5, &out)],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
    gpu.queue.submit([enc.finish()]);
    read.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let view = read.get_mapped_range(..).unwrap();
    bytemuck::cast_slice(&view).to_vec()
}

fn noise_graph(p: NoiseParams) -> Graph {
    let mut g = Graph::default();
    let n = g.add(NodeKind::Noise(p), [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(n, o, 0);
    g
}

/// Deterministic points in [0, 1)³.
fn points(count: usize, seed: u32) -> Vec<[f32; 3]> {
    let mut s = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..count).map(|_| [next(), next(), next()]).collect()
}

/// Every kind (and Worley feature), and whether its output is continuous.
fn cases() -> Vec<(String, NoiseParams, bool)> {
    let mut v = Vec::new();
    for kind in NoiseKind::ALL {
        let scale = if kind == NoiseKind::Blue { 8.0 } else { 4.0 };
        let features: &[WorleyFeature] = if kind == NoiseKind::Worley {
            &[WorleyFeature::F1, WorleyFeature::F2, WorleyFeature::F2MinusF1, WorleyFeature::CellValue]
        } else {
            &[WorleyFeature::F1]
        };
        for &feature in features {
            // Fractal Worley exercises the octave loop; a flat cell value is never continuous.
            let continuous = matches!(kind, NoiseKind::Perlin | NoiseKind::Simplex | NoiseKind::Worley) && feature != WorleyFeature::CellValue;
            let p = NoiseParams { kind, feature, fractal: true, scale, octaves: 3, ..Default::default() };
            v.push((format!("{} {feature:?}", kind.name()), p, continuous));
        }
    }
    v.push(("Worley F1 single octave".into(), NoiseParams { kind: NoiseKind::Worley, ..Default::default() }, true));
    v.push(("Perlin distorted".into(), NoiseParams { distortion: 0.7, octaves: 3, ..Default::default() }, true));
    v
}

#[test]
fn every_noise_tiles() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let base = points(4096, 1);
    for (name, params, continuous) in cases() {
        let g = noise_graph(params);
        let a = eval(&gpu, &g, &base);
        for axis in 0..3 {
            let shifted: Vec<_> = base.iter().map(|p| { let mut q = *p; q[axis] += 1.0; q }).collect();
            let b = eval(&gpu, &g, &shifted);
            let diffs: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect();
            if continuous {
                let worst = diffs.iter().cloned().fold(0.0, f32::max);
                assert!(worst < 2e-3, "{name}: axis {axis} seam, max diff {worst}");
            } else {
                // Cell edges can land on different sides after the shift's rounding; allow a few.
                let bad = diffs.iter().filter(|d| **d > 1e-4).count();
                assert!(bad * 200 < diffs.len(), "{name}: axis {axis}, {bad} of {} points differ", diffs.len());
            }
        }
    }
}

#[test]
fn continuous_noises_have_no_jumps() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    // Half the points sit just before the tile edge, so the step crosses x = 1.
    let mut base = points(4096, 2);
    for p in base.iter_mut().step_by(2) {
        p[0] = 1.0 - 2e-5;
    }
    let step: Vec<_> = base.iter().map(|p| [p[0] + 1e-4, p[1] + 0.7e-4, p[2] + 0.3e-4]).collect();
    for (name, params, continuous) in cases() {
        if !continuous {
            continue;
        }
        let g = noise_graph(params);
        let (a, b) = (eval(&gpu, &g, &base), eval(&gpu, &g, &step));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(worst < 0.02, "{name}: jump of {worst} over a 1e-4 step");
    }
}

/// Writes target/noise_sheet.pgm: every case side by side over 2×2 tiles (z = 0.5), each normalized.
#[test]
#[ignore = "visual check; writes target/noise_sheet.pgm"]
fn noise_contact_sheet() {
    let gpu = gpu().expect("needs a GPU");
    let n = 160;
    let grid: Vec<_> = (0..n * n).map(|i| [2.0 * (i % n) as f32 / n as f32, 2.0 * (i / n) as f32 / n as f32, 0.5]).collect();
    let cases = cases();
    let w = n * cases.len();
    let mut img = vec![0u8; w * n];
    for (c, (name, params, _)) in cases.into_iter().enumerate() {
        let v = eval(&gpu, &noise_graph(params), &grid);
        let lo = v.iter().cloned().fold(f32::MAX, f32::min);
        let hi = v.iter().cloned().fold(f32::MIN, f32::max);
        eprintln!("{c}: {name}");
        for (i, x) in v.iter().enumerate() {
            img[(i / n) * w + c * n + i % n] = (255.0 * (x - lo) / (hi - lo).max(1e-9)) as u8;
        }
    }
    let mut out = format!("P5 {w} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/target/noise_sheet.pgm"), out).unwrap();
}

/// GPU time of the 3D view (1000×1000 px) and the 512² slice for a graph file, median of 30 frames.
/// `cargo test --release bench_preview -- --ignored --nocapture` (NOISER_BENCH_GRAPH=path to pick a file,
/// default SimpleTestGraph.noisegraph in the repo root; NOISER_BENCH_STEP=step safety, default 0.6).
#[test]
#[ignore = "benchmark"]
fn bench_preview() {
    use crate::view;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    let desc = wgpu::DeviceDescriptor { required_features: wgpu::Features::TIMESTAMP_QUERY, ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc)).unwrap();
    eprintln!("adapter: {}", adapter.get_info().name);

    let path = std::env::var("NOISER_BENCH_GRAPH")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/SimpleTestGraph.noisegraph").into());
    let graph = Graph::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let c = codegen::compile(&graph).unwrap();
    // NOISER_BENCH_BUST=1: a unique dummy function so the driver's shader cache can't be used.
    let mut wgsl = c.wgsl.clone();
    if std::env::var("NOISER_BENCH_BUST").is_ok() {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1_000_000_007;
        wgsl = wgsl.replace("fn graph_density(p: vec3f) -> f32 { return ", &format!("fn graph_density(p: vec3f) -> f32 {{ return {n}e-30 + "));
    }
    let t0 = std::time::Instant::now();
    let mut gpu = view::build(&device, &queue, wgpu::TextureFormat::Bgra8Unorm, &wgsl);
    eprintln!("backend {:?}: pipeline build {:.0} ms", adapter.get_info().backend, t0.elapsed().as_secs_f32() * 1e3);

    let size = [1000u32, 1000];
    let step: f32 = std::env::var("NOISER_BENCH_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.6);
    for (mode, name) in [(0u32, "solid"), (1, "cloud")] {
        // Same camera and preview defaults as the app.
        let (yaw, pitch, dist) = (0.6f32, 0.35f32, 3.5f32);
        let pos = [dist * pitch.cos() * yaw.sin(), dist * pitch.sin(), dist * pitch.cos() * yaw.cos()];
        let l = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let fwd = [-pos[0] / l, -pos[1] / l, -pos[2] / l];
        let rl = (fwd[2] * fwd[2] + fwd[0] * fwd[0]).sqrt();
        let right = [-fwd[2] / rl, 0.0, fwd[0] / rl];
        let up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
        let push = view::push_range(graph.density_range(), 0.25);
        let u = view::Uniforms {
            cam_pos: pos, aspect: 1.0, cam_fwd: fwd, amount: 0.25, cam_right: right, step_scale: step, cam_up: up, seed: 0,
            tile_size: 1.0, prim_size: 1.0, pixel: 2.0 / size[1] as f32, cloud_density: 8.0, mode, prim: 0,
            slice_axis: 2, slice_mode: 0, slice_pos: 0.5, slice_aspect: 1.0, slice_tiles: 1.0, slice_res: view::SLICE_RES,
            push_min: push.0, push_max: push.1, _pad: [0.0; 2], mesh_origin: [0.0; 3], mesh_voxel: 1.0,
        };
        let mut samples = Vec::new();
        for _ in 0..30 {
            gpu.upload(&queue, &u, &c.params);
            let mut enc = device.create_command_encoder(&Default::default());
            gpu.render_3d(&mut enc, size);
            gpu.compute_slice(&queue, &mut enc);
            queue.submit([enc.finish()]);
            let t = gpu.timer.as_mut().unwrap();
            t.collect(&device); // starts the map
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            t.collect(&device); // reads it
            samples.push(t.ms);
        }
        let median = |i: usize| {
            let mut v: Vec<f32> = samples[5..].iter().map(|s| s[i]).collect(); // skip warm-up
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        eprintln!("{name:5}: 3D {:7.2} ms  slice {:6.2} ms", median(0), median(1));
    }
}

/// Density = mean(x, y, z), so every exported voxel has a known value.
fn position_graph() -> Graph {
    let mut g = Graph::default();
    let p = g.add(NodeKind::Position, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(p, o, 0);
    g
}

#[test]
fn export_evaluates_texel_centres() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let ev = crate::export::Evaluator::new(&gpu.device, &gpu.queue, &position_graph(), 0).unwrap();
    let n = 20;
    let v = ev.eval(n, [3, 5, 7], [4, 3, 2], false, None);
    // Box value at local (x, y, z) = 1, 2, 1, x fastest.
    let (x, y, z) = (3 + 1, 5 + 2, 7 + 1);
    let want = ((x as f32 + 0.5) + (y as f32 + 0.5) + (z as f32 + 0.5)) / (3.0 * n as f32);
    assert!((v[(1 * 3 + 2) * 4 + 1] - want).abs() < 1e-6);
}

#[test]
fn slice_grid_png_places_slices_and_levels() {
    use crate::export::{self, Format, Kind, Layout, Levels, Output, Settings};
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let g = position_graph(); // unbounded range, so 16-bit falls back to data min/max
    let ev = export::Evaluator::new(&gpu.device, &gpu.queue, &g, 0).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    let s = Settings { kind: Kind::SliceGrid, output: Output::Density, res: 8, format: Format::Png16, layout: Layout::Square, levels: Levels::GraphRange, ..Default::default() };
    let png = dir.join("grid.png");
    export::run(&ev, &s, g.density_range(), &png, &Default::default()).unwrap();
    let img = image::open(&png).unwrap().into_luma16();
    assert_eq!(img.dimensions(), (24, 24)); // 8 slices in a 3×3 grid of 8×8 tiles
    // Slice z = 4 sits in tile (1, 1); check texel (2, 3) in it.
    let v = |x: f32, y: f32, z: f32| (x + y + z + 1.5) / 24.0;
    let (lo, hi) = (v(0.0, 0.0, 0.0), v(7.0, 7.0, 7.0));
    let want = ((v(2.0, 3.0, 4.0) - lo) / (hi - lo) * 65535.0).round() as i32;
    let got = img.get_pixel(8 + 2, 8 + 3).0[0] as i32;
    assert!((got - want).abs() <= 1, "got {got}, want {want}");
    // The unused ninth tile stays black.
    assert_eq!(img.get_pixel(16 + 4, 16 + 4).0[0], 0);

    // EXR and 2D variants write without error.
    let exr = Settings { format: Format::Exr32, output: Output::Normal, ..s.clone() };
    export::run(&ev, &exr, g.density_range(), &dir.join("grid.exr"), &Default::default()).unwrap();
    let flat = Settings { kind: Kind::Image2d, res: 32, format: Format::Png8, ..s };
    export::run(&ev, &flat, g.density_range(), &dir.join("flat.png"), &Default::default()).unwrap();
    assert_eq!(image::open(dir.join("flat.png")).unwrap().into_luma8().dimensions(), (32, 32));
}

#[test]
fn cache_bake_matches_full_evaluation() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let g = noise_graph(NoiseParams { kind: NoiseKind::Worley, fractal: true, octaves: 2, ..Default::default() });
    let c = codegen::compile(&g).unwrap();
    let res = 64u32; // 64 × 4 bytes = one 256-byte row, so the readback needs no padding
    let tex = crate::cache::texture(&gpu.device, res);
    crate::cache::Baker::new(&gpu.device, &gpu.queue, &c.wgsl).bake(&tex, res, &c.params, 0);
    let bytes = (res * res * res * 4) as u64;
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(res * 4), rows_per_image: Some(res) } },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
    );
    gpu.queue.submit([enc.finish()]);
    read.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let baked: Vec<f32> = bytemuck::cast_slice(&read.get_mapped_range(..).unwrap()).to_vec();
    // The export evaluator samples the same texel centres, x fastest, like the texture rows.
    let ev = crate::export::Evaluator::new(&gpu.device, &gpu.queue, &g, 0).unwrap();
    let want = ev.eval(res, [0; 3], [res; 3], false, None);
    let worst = baked.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(worst < 1e-5, "bake differs from full evaluation by {worst}");
}

/// Writes target/check.vdb (20³, density = mean of the texel centre) for checking in OpenVDB.
#[test]
#[ignore = "writes target/check.vdb"]
fn write_check_vdb() {
    let gpu = gpu().unwrap();
    let ev = crate::export::Evaluator::new(&gpu.device, &gpu.queue, &position_graph(), 0).unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("check.vdb");
    crate::vdb::write(&path, 20, |o, s| ev.eval(20, o, s, false, None), |_| {}).unwrap();
}

#[test]
fn noises_stay_in_signed_range() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let pts = points(65536, 3);
    for kind in NoiseKind::ALL {
        let scale = if kind == NoiseKind::Blue { 8.0 } else { 4.0 };
        let v = eval(&gpu, &noise_graph(NoiseParams { kind, scale, octaves: 1, ..Default::default() }), &pts);
        let lo = v.iter().cloned().fold(f32::MAX, f32::min);
        let hi = v.iter().cloned().fold(f32::MIN, f32::max);
        eprintln!("{:16} min {lo:+.3} max {hi:+.3}", kind.name());
        assert!(lo >= -1.05 && hi <= 1.05, "{}: range {lo}..{hi}", kind.name());
        assert!(hi - lo > 1.0, "{}: range {lo}..{hi} is too narrow", kind.name());
    }
}

/// Writes `target/blender/hash.json`: hash01 for a spread of cells and seeds, so
/// `blender/check_hash.py` can confirm Blender's White Noise node gives the same values.
#[test]
fn hash_vectors_for_blender() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let c = codegen::compile(&noise_graph(NoiseParams::default())).unwrap();
    let mut cases = Vec::new();
    for (i, p) in points(200, 11).into_iter().enumerate() {
        let cell = p.map(|v| ((v - 0.5) * 2000.0).floor()); // negatives, zero, larger values
        let seed = [0.0, 1.0, 7.0, 100.0, (0x51ED27 + 3) as f32][i % 5];
        cases.push([cell[0], cell[1], cell[2], seed]);
    }
    cases.push([0.0, 0.0, 0.0, 0.0]);
    let ch: Vec<Vec<f32>> = (0..3).map(|k| eval_expr(&gpu, &c, &format!("hash01(vec3i(q.xyz), u32(q.w))[{k}]"), &cases)).collect();
    let rows: Vec<_> = cases.iter().enumerate().map(|(i, q)| serde_json::json!({"cell": &q[..3], "seed": q[3], "h": [ch[0][i], ch[1][i], ch[2][i]]})).collect();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("blender");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hash.json"), serde_json::to_string(&rows).unwrap()).unwrap();
    // Sanity: uniform-ish in [0, 1).
    let mean = ch[0].iter().sum::<f32>() / ch[0].len() as f32;
    assert!(ch.iter().flatten().all(|h| (0.0..=1.0).contains(h)) && (0.35..0.65).contains(&mean), "mean {mean}");
}

/// Writes `target/blender/*.noiser` through the real export path, for `blender/check_file.py`.
#[test]
fn noiser_files_for_blender() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("blender");
    std::fs::create_dir_all(&dir).unwrap();
    let perlin = |p: NoiseParams| noise_graph(NoiseParams { kind: NoiseKind::Perlin, ..p });
    let mut chain = perlin(NoiseParams { scale: 3.0, octaves: 2, ..Default::default() });
    let n = chain.find(|k| matches!(k, NodeKind::Noise(_))).unwrap();
    let o = chain.find(|k| *k == NodeKind::OutDensity).unwrap();
    let m = chain.add(NodeKind::Math { op: crate::graph::MathOp::Divide, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    let k = chain.add(NodeKind::Math { op: crate::graph::MathOp::Multiply, a: 0.0, b: 0.8, c: 0.5 }, [0.0; 2]);
    chain.connect(n, m, 0); // noise / 0 (safe divide) feeds nothing useful; max keeps it in the graph
    chain.connect(n, k, 0);
    let mx = chain.add(NodeKind::Math { op: crate::graph::MathOp::Max, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    chain.connect(k, mx, 0);
    chain.connect(m, mx, 1);
    chain.connect(mx, o, 0);
    let cases: Vec<(&str, Graph)> = vec![
        ("perlin_default", Graph::default_graph()),
        ("perlin_octaves_normalized", perlin(NoiseParams { scale: 5.0, octaves: 4, lacunarity: 2.0, roughness: 0.6, normalize: true, ..Default::default() })),
        ("perlin_distorted", perlin(NoiseParams { scale: 3.0, octaves: 2, distortion: 0.5, seed: Some(9), ..Default::default() })),
        ("perlin_not_tiling", perlin(NoiseParams { scale: 2.5, octaves: 3, lacunarity: 2.3, ..Default::default() })),
        ("perlin_math_chain", chain),
    ];
    for (name, g) in cases {
        let ev = crate::export::Evaluator::new(&gpu.device, &gpu.queue, &g, 4).unwrap();
        let s = crate::noiser_file::file(&g, 4, |pts| ev.eval_points(pts)).unwrap();
        std::fs::write(dir.join(format!("{name}.noiser")), s).unwrap();
    }
}

fn single_node_graph(kind: NodeKind) -> Graph {
    let mut g = Graph::default();
    let n = g.add(kind, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(n, o, 0);
    g
}

fn near(got: f32, want: f32, what: &str) {
    assert!((got - want).abs() < 1e-4, "{what}: got {got}, want {want}");
}

#[test]
fn new_math_ops_match_reference() {
    use crate::graph::MathOp::*;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let at = [[0.3, 0.6, 0.9]];
    let math = |op, a, b, c| eval(&gpu, &single_node_graph(NodeKind::Math { op, a, b, c }), &at)[0];
    near(math(Power, 2.0, 3.0, 0.5), 8.0, "2^3");
    near(math(Power, -2.0, 3.0, 0.5), -8.0, "(-2)^3");
    near(math(Power, -2.0, 2.0, 0.5), 4.0, "(-2)^2");
    near(math(Power, -2.0, 0.5, 0.5), 0.0, "(-2)^0.5 is 0, not NaN");
    near(math(Power, 0.0, 0.0, 0.5), 1.0, "0^0");
    near(math(Power, 0.0, 2.0, 0.5), 0.0, "0^2");
    // Blender's polynomial smooth min: 1 - h³k/6 with h = (0.5 - 0.2) / 0.5.
    near(math(SmoothMin, 1.0, 1.2, 0.5), 1.0 - 0.6f32.powi(3) * 0.5 / 6.0, "smooth min");
    near(math(SmoothMax, 1.0, 1.2, 0.5), 1.2 + 0.6f32.powi(3) * 0.5 / 6.0, "smooth max");
    near(math(SmoothMin, 0.0, 5.0, 0.5), 0.0, "smooth min, far apart");
    near(math(SmoothMin, 1.0, 1.2, 0.0), 1.0, "smooth min with k = 0 is min");
    // The blend width can come from a link too.
    let mut g = Graph::default();
    let w = g.add(NodeKind::Float(0.5), [0.0; 2]);
    let m = g.add(NodeKind::Math { op: SmoothMax, a: 1.0, b: 1.2, c: 0.0 }, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(w, m, 2);
    g.connect(m, o, 0);
    near(eval(&gpu, &g, &at)[0], 1.2 + 0.6f32.powi(3) * 0.5 / 6.0, "linked blend width");
    // Dot product of two vectors.
    let mut g = Graph::default();
    let a = g.add(NodeKind::Vec3([1.0, 2.0, 3.0]), [0.0; 2]);
    let b = g.add(NodeKind::Vec3([4.0, 5.0, 6.0]), [0.0; 2]);
    let d = g.add(NodeKind::Math { op: Dot, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(a, d, 0);
    g.connect(b, d, 1);
    g.connect(d, o, 0);
    near(eval(&gpu, &g, &at)[0], 32.0, "dot");
    // A vector with a float favors the vector: max((1, 2, 6), 3) = (3, 3, 6), whose average is 4
    // (taking the vector's average first would give max(3, 3) = 3).
    let mut g = Graph::default();
    let v = g.add(NodeKind::Vec3([1.0, 2.0, 6.0]), [0.0; 2]);
    let m = g.add(NodeKind::Math { op: Max, a: 0.0, b: 3.0, c: 0.5 }, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(v, m, 0);
    g.connect(m, o, 0);
    near(eval(&gpu, &g, &at)[0], 4.0, "float with vector stays a vector");
}

#[test]
fn remap_matches_reference() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let at = [[0.3, 0.6, 0.9]];
    let remap = |x: f32, vals: [f32; 5], clamp: bool| {
        let mut g = Graph::default();
        let f = g.add(NodeKind::Float(x), [0.0; 2]);
        let r = g.add(NodeKind::Remap { vals, clamp }, [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(f, r, 0);
        g.connect(r, o, 0);
        eval(&gpu, &g, &at)[0]
    };
    near(remap(0.25, [0.0, 0.0, 1.0, 10.0, 20.0], false), 12.5, "linear");
    near(remap(2.0, [0.0, 0.0, 1.0, 10.0, 20.0], false), 30.0, "extrapolates");
    near(remap(2.0, [0.0, 0.0, 1.0, 10.0, 20.0], true), 20.0, "clamped high");
    near(remap(-1.0, [0.0, 0.0, 1.0, 10.0, 20.0], true), 10.0, "clamped low");
    near(remap(0.25, [0.0, 0.0, 1.0, 20.0, 10.0], false), 17.5, "reversed output");
    near(remap(2.0, [0.0, 0.0, 1.0, 20.0, 10.0], true), 10.0, "clamped on a reversed output");
    near(remap(0.7, [0.0, 1.0, 1.0, 10.0, 20.0], false), 10.0, "empty input range gives the output minimum");
}

#[test]
fn grid_spans_minus_one_to_one_and_tiles() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    for corners in [false, true] {
        for counts in [[8.0, 8.0, 8.0], [8.0, 4.0, 2.0]] {
            let g = single_node_graph(NodeKind::Grid { counts, corners });
            let centre = [0.5 / counts[0], 0.5 / counts[1], 0.5 / counts[2]];
            let corner = [1.0 / counts[0], 2.0 / counts[1], 1.0 / counts[2]];
            let v = eval(&gpu, &g, &[centre, corner]);
            let (at_centre, at_corner) = if corners { (1.0, -1.0) } else { (-1.0, 1.0) };
            near(v[0], at_centre, "cell centre");
            near(v[1], at_corner, "cell corner");
            let pts = points(20000, 5);
            let v = eval(&gpu, &g, &pts);
            let (lo, hi) = (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max));
            assert!(lo >= -1.0001 && hi <= 1.0001, "range {lo}..{hi}");
            assert!(lo < -0.9 && hi > 0.9, "range {lo}..{hi} doesn't fill -1..1");
            // One tile later it repeats, on every axis.
            for axis in 0..3 {
                let moved: Vec<[f32; 3]> = pts.iter().map(|p| { let mut q = *p; q[axis] += 1.0; q }).collect();
                let w = eval(&gpu, &g, &moved);
                let worst = v.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                assert!(worst < 2e-3, "axis {axis}: seam error {worst}");
            }
        }
    }
}

fn bricks_graph(f: impl FnOnce(&mut crate::graph::PatternParams)) -> Graph {
    let mut p = crate::graph::PatternParams::default();
    f(&mut p);
    single_node_graph(NodeKind::Pattern(p))
}

#[test]
fn bricks_distance_without_offset_is_the_grid_distance() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let pts = points(5000, 9);
    let bricks = bricks_graph(|p| {
        p.mode2d = false;
        p.steps = 1;
        p.counts = [8.0, 4.0, 2.0];
    });
    let grid = single_node_graph(NodeKind::Grid { counts: [8.0, 4.0, 2.0], corners: false });
    let (a, b) = (eval(&gpu, &bricks, &pts), eval(&gpu, &grid, &pts));
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(worst < 1e-4, "bricks (no row offset) differ from the grid by {worst}");
}

#[test]
fn bricks_rows_are_offset_so_joints_do_not_line_up() {
    use crate::graph::PatternOutput::Shape;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    // 8 bricks per row and 16 rows: a brick is 1/8 wide and 1/16 tall. z does not matter in 2D.
    let (x, z) = (0.5 / 8.0, 0.3);
    let row = |k: f32| [x, (k + 0.5) / 16.0, z];
    let at_centre_of_row0 = [row(0.0), row(1.0), row(2.0), row(3.0)];
    let v = eval(&gpu, &bricks_graph(|p| p.output = Shape), &at_centre_of_row0);
    near(v[0], -1.0, "centre of a brick in row 0");
    near(v[1], 1.0, "the same x one row up is a vertical joint (mortar)");
    near(v[2], -1.0, "two rows up repeats row 0");
    near(v[3], 1.0, "three rows up is a joint again");
    // Without the offset every row lines up.
    let v = eval(&gpu, &bricks_graph(|p| { p.output = Shape; p.steps = 1; }), &at_centre_of_row0);
    assert!(v.iter().all(|x| (x + 1.0).abs() < 1e-4), "steps 1 should stack the bricks: {v:?}");
    // Thirds: row 1 is shifted by a third of a brick, so its centre sits a third along.
    // (Distance, since Shape is -1 across the whole brick face.)
    let thirds = bricks_graph(|p| { p.steps = 3; p.counts[1] = 12.0; });
    let v = eval(&gpu, &thirds, &[[x + 1.0 / 3.0 / 8.0, 1.5 / 12.0, z], [x, 1.5 / 12.0, z]]);
    near(v[0], -1.0, "centre of a brick in a row shifted by a third");
    assert!(v[1] > -0.9, "the unshifted x is off-centre in that row, got {}", v[1]);
}

#[test]
fn bricks_stay_in_range_fill_it_and_tile() {
    use crate::graph::PatternOutput::{CellValue, Distance, Shape};
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let pts = points(20000, 5);
    for (steps, counts) in [(2, [8.0, 16.0, 8.0]), (3, [6.0, 12.0, 4.0]), (1, [5.0, 7.0, 3.0])] {
        for output in [Distance, Shape, CellValue] {
            for mode2d in [true, false] {
                let g = bricks_graph(|p| { p.steps = steps; p.counts = counts; p.output = output; p.mode2d = mode2d; });
                let what = format!("steps {steps} {output:?} 2d={mode2d}");
                let v = eval(&gpu, &g, &pts);
                assert!(v.iter().all(|x| x.is_finite()), "{what}: NaN or infinity");
                let (lo, hi) = (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max));
                assert!(lo >= -1.0001 && hi <= 1.0001, "{what}: range {lo}..{hi}");
                if output == CellValue {
                    // One random value per brick: a small wall only has a few dozen of them.
                    assert!(hi - lo > 1.2, "{what}: values only span {lo}..{hi}");
                } else {
                    assert!(lo < -0.9 && hi > 0.9, "{what}: range {lo}..{hi} does not fill -1..1");
                }
                for axis in 0..3 {
                    let moved: Vec<[f32; 3]> = pts.iter().map(|p| { let mut q = *p; q[axis] += 1.0; q }).collect();
                    let w = eval(&gpu, &g, &moved);
                    // A point within float error of a brick edge may fall either side, so allow a few.
                    let off = v.iter().zip(&w).filter(|(a, b)| (*a - *b).abs() > 2e-3).count();
                    assert!(off <= 3, "{what}, axis {axis}: {off} of {} points differ one tile later", pts.len());
                }
            }
        }
    }
}

#[test]
fn bricks_with_no_bevel_or_no_gap_are_finite() {
    use crate::graph::PatternOutput::Shape;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let pts = points(20000, 6);
    // Bevel 0 makes the ramp's low and high ends equal, which would be a NaN in smoothstep.
    let v = eval(&gpu, &bricks_graph(|p| { p.output = Shape; p.bevel = 0.0; }), &pts);
    assert!(v.iter().all(|x| x.is_finite() && x.abs() <= 1.0001), "bevel 0 gave a bad value");
    let hard = v.iter().filter(|x| x.abs() > 0.999).count();
    assert!(hard as f32 > 0.99 * v.len() as f32, "a hard edge should give only -1 or +1, got {hard} of {}", v.len());
    assert!(v.iter().any(|x| *x > 0.999) && v.iter().any(|x| *x < -0.999));
    // No gap and no bevel: the mortar has no width, so +1 is only reached on the border itself.
    let v = eval(&gpu, &bricks_graph(|p| { p.output = Shape; p.bevel = 0.0; p.gap = 0.0; }), &pts);
    assert!(v.iter().all(|x| x.is_finite() && x.abs() <= 1.0001), "gap 0 gave a bad value");
}

#[test]
fn bricks_cell_value_is_flat_per_brick_and_agrees_across_the_tile_edge() {
    use crate::graph::PatternOutput::CellValue;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let g = bricks_graph(|p| p.output = CellValue); // 8 per row, 16 rows, running bond
    let row0 = |x: f32, y: f32| [x, y, 0.3];
    // Two points in one row-0 brick (x in [0, 1/8), y in [0, 1/16)), and one in the next brick.
    let v = eval(&gpu, &g, &[row0(0.02, 0.01), row0(0.11, 0.05), row0(0.15, 0.01)]);
    near(v[0], v[1], "one brick has one value");
    assert!((v[0] - v[2]).abs() > 1e-3, "neighbouring bricks should differ: {v:?}");
    // In row 1 the bricks are shifted half a brick, so one straddles the tile edge: x = 0.99 and
    // x = 0.01 are in the same brick and must hash the same id.
    let y1 = 1.5 / 16.0;
    let v = eval(&gpu, &g, &[[0.99, y1, 0.3], [0.01, y1, 0.3], [0.5, y1, 0.3]]);
    near(v[0], v[1], "the brick across the tile edge has one value on both sides");
    assert!((v[0] - v[2]).abs() > 1e-3, "a brick across the tile has another value: {v:?}");
    // The brick above (row 2) is a different brick from the one in row 0 at the same x.
    let v = eval(&gpu, &g, &[row0(0.02, 0.01), row0(0.02, 2.5 / 16.0)]);
    assert!((v[0] - v[1]).abs() > 1e-3, "bricks in different rows should differ: {v:?}");
}

#[test]
fn bricks_2d_extrudes_along_the_chosen_axis() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let pts = points(4000, 8);
    for axis in 0..3u32 {
        let g = bricks_graph(|p| p.axis = axis);
        let v = eval(&gpu, &g, &pts);
        let shift = |a: usize| -> f32 {
            let moved: Vec<[f32; 3]> = pts.iter().map(|p| { let mut q = *p; q[a] += 0.37; q }).collect();
            let w = eval(&gpu, &g, &moved);
            v.iter().zip(&w).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
        };
        assert!(shift(axis as usize) < 1e-4, "axis {axis}: the pattern should be constant along the extrusion axis");
        for other in (0..3).filter(|a| *a != axis as usize) {
            assert!(shift(other) > 0.1, "axis {axis}: the pattern should change along axis {other}");
        }
    }
}

/// Writes target/pattern_*.png (a slice through each pattern) for looking at.
#[test]
#[ignore = "visual check; writes target/pattern_*.png"]
fn render_pattern_images() {
    use crate::export::{self, Format, Kind, Settings};
    use crate::graph::PatternOutput::{CellValue, Distance, Shape};
    let gpu = gpu().expect("needs a GPU");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    let s = Settings { kind: Kind::Image2d, res: 384, format: Format::Png8, ..Default::default() };
    let graphs = [
        ("bricks_shape", bricks_graph(|p| p.output = Shape)),
        ("bricks_distance", bricks_graph(|p| p.output = Distance)),
        ("bricks_cells", bricks_graph(|p| p.output = CellValue)),
        ("bricks_thirds", bricks_graph(|p| { p.output = Shape; p.steps = 3; p.counts = [6.0, 24.0, 8.0]; })),
        ("grid_centres", single_node_graph(NodeKind::Grid { counts: [8.0; 3], corners: false })),
        ("grid_corners", single_node_graph(NodeKind::Grid { counts: [8.0; 3], corners: true })),
    ];
    for (name, g) in graphs {
        let ev = export::Evaluator::new(&gpu.device, &gpu.queue, &g, 0).unwrap();
        export::run(&ev, &s, g.density_range(), &dir.join(format!("pattern_{name}.png")), &Default::default()).unwrap();
    }
}

// ---- mesh export --------------------------------------------------------------------

mod mesh_export_tests {
    use super::*;
    use crate::mesh::{Normalization, shapes};
    use crate::meshexport::{Displace, Format, Job, MeshEvaluator, Settings};
    use crate::meshsdf::MeshSdf;
    use crate::surface::RawMesh;
    use std::sync::atomic::AtomicU32;

    /// A radius-1 icosphere baked and uploaded, with an evaluator for `graph`.
    fn evaluator(gpu: &Gpu, graph: &Graph, seed: u32) -> (MeshSdf, MeshEvaluator, wgpu::Texture) {
        let sdf = MeshSdf::build(&shapes::icosphere(1.0, 3), 64, 0.6, &AtomicU32::new(0));
        let (texture, view) = sdf.upload(&gpu.device, &gpu.queue);
        let ev = MeshEvaluator::new(&gpu.device, &gpu.queue, graph, seed, view, sdf.origin, sdf.voxel).unwrap();
        (sdf, ev, texture)
    }

    fn job<'a>(ev: &'a MeshEvaluator, d: Displace, reach: f32, res: u32) -> Job<'a> {
        Job { ev, d, lo: [-1.0; 3], hi: [1.0; 3], reach, norm: Normalization::IDENTITY, settings: Settings { res, format: Format::Obj, cull: 0.0 } }
    }

    fn radius(p: [f32; 3]) -> f32 {
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    /// (edges, edges that are not used exactly once in each direction, edges used only one way).
    fn edge_stats(m: &RawMesh) -> (usize, usize, usize) {
        let mut edges: std::collections::HashMap<(u32, u32), (u32, u32)> = std::collections::HashMap::new();
        for t in &m.triangles {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                let e = edges.entry((a.min(b), a.max(b))).or_default();
                if a < b { e.0 += 1 } else { e.1 += 1 }
            }
        }
        let bad = edges.values().filter(|&&c| c != (1, 1)).count();
        let open = edges.values().filter(|&&c| c.0 == 0 || c.1 == 0).count();
        (edges.len(), bad, open)
    }

    fn is_closed(m: &RawMesh) -> bool {
        edge_stats(m).1 == 0
    }

    #[test]
    fn without_displacement_the_export_is_the_mesh() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let (_sdf, ev, _tex) = evaluator(&gpu, &single_node_graph(NodeKind::Float(0.0)), 0);
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0 };
        let j = job(&ev, d, 0.0, 64);
        let m = j.extract(&Default::default());
        assert!(m.triangles.len() > 5000, "{} triangles", m.triangles.len());
        assert!(is_closed(&m), "not a closed manifold");
        let exact = 4.0 / 3.0 * std::f32::consts::PI;
        assert!((m.signed_volume() - exact).abs() < 0.06 * exact, "volume {}", m.signed_volume());
        let worst = m.positions.iter().map(|p| (radius(*p) - 1.0).abs()).fold(0.0, f32::max);
        assert!(worst < 0.03, "a vertex is {worst} off the sphere");
    }

    #[test]
    fn every_vertex_lies_on_the_zero_level_of_the_displaced_field() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let noise = noise_graph(NoiseParams { scale: 3.0, octaves: 3, ..Default::default() });
        let (sdf, ev, _tex) = evaluator(&gpu, &noise, 0);
        let d = Displace { amount: 0.12, tile_size: 1.0, prim_size: 1.0 };
        let j = job(&ev, d, 0.12, 96);
        let h = j.grid().h;
        let m = j.extract(&Default::default());
        // Surface nets can leave a few non-manifold edges where a cell's corner signs are ambiguous,
        // but never holes.
        let (edges, bad, open) = edge_stats(&m);
        assert_eq!(open, 0, "the surface has holes");
        assert!(bad * 1000 < edges, "{bad} of {edges} edges are non-manifold");
        let rs: Vec<f32> = m.positions.iter().map(|p| radius(*p)).collect();
        let (lo, hi) = (rs.iter().cloned().fold(f32::MAX, f32::min), rs.iter().cloned().fold(f32::MIN, f32::max));
        assert!(hi - lo > 0.05, "the noise should visibly shape the sphere: radius {lo}..{hi}");
        assert!(lo > 0.8 && hi < 1.2, "displacement of 0.12 moved the surface to radius {lo}..{hi}");
        // The graph's density at each vertex, from the texture-export evaluator, and the same field on the CPU.
        let reference = crate::export::Evaluator::new(&gpu.device, &gpu.queue, &noise, 0).unwrap();
        let sample: Vec<[f32; 3]> = m.positions.iter().step_by(7).copied().collect();
        let density = reference.eval_points(&sample);
        let worst = sample.iter().zip(&density).map(|(p, rho)| (sdf.sample(*p) - d.amount * rho).abs()).fold(0.0, f32::max);
        assert!(worst < 0.7 * h, "a vertex is off the surface by {worst} (sample spacing {h})");
    }

    #[test]
    fn shape_size_and_normalization_are_undone_in_the_written_file() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let (_sdf, ev, _tex) = evaluator(&gpu, &single_node_graph(NodeKind::Float(0.0)), 0);
        // Shape size 2 doubles the sphere in the preview; the mesh's own units are 3 per normalized
        // unit and centred at x = 5, so the file should hold a sphere of radius 2 / 2 × 3 = 3 there.
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 2.0 };
        let mut j = job(&ev, d, 0.0, 64);
        j.hi = [1.0; 3];
        j.norm = Normalization { center: [5.0, 0.0, 0.0], scale: 3.0 };
        let dir = std::env::temp_dir().join("noiser-mesh-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sphere.obj");
        let summary = j.run(&path, &Default::default()).unwrap();
        assert!(summary.contains("triangles"), "{summary}");
        let back = crate::mesh::Mesh::load_obj(&path).unwrap();
        let worst = back.positions.iter().map(|p| (radius([p[0] - 5.0, p[1], p[2]]) - 3.0).abs()).fold(0.0, f32::max);
        assert!(worst < 0.15, "the exported sphere is {worst} off radius 3 around (5, 0, 0)");
        // A PLY of the same job is written too.
        j.settings.format = Format::Ply;
        j.run(&dir.join("sphere.ply"), &Default::default()).unwrap();
        assert!(std::fs::metadata(dir.join("sphere.ply")).unwrap().len() > 1000);
    }

    #[test]
    fn seeds_change_the_result_and_repeat() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let noise = noise_graph(NoiseParams { scale: 3.0, octaves: 2, ..Default::default() });
        let d = Displace { amount: 0.1, tile_size: 1.0, prim_size: 1.0 };
        let run = |seed: u32| {
            let (_sdf, ev, _tex) = evaluator(&gpu, &noise, seed);
            job(&ev, d, 0.1, 48).extract(&Default::default())
        };
        let (a, a_again, b) = (run(0), run(0), run(1));
        assert_eq!(a.positions, a_again.positions, "the same seed should give the same mesh");
        let differs = a.positions.len() != b.positions.len() || a.positions.iter().zip(&b.positions).any(|(p, q)| p != q);
        assert!(differs, "another seed should give another mesh");
    }
}

/// Exports a real mesh with a real graph: `NOISER_TEST_MESH=file.obj cargo test export_a_real_mesh -- --ignored --nocapture`.
/// Optional: NOISER_TEST_GRAPH (default SimpleTestGraph.noisegraph), NOISER_TEST_RES (256),
/// NOISER_TEST_AMOUNT (0.25), NOISER_TEST_SEED (0). Writes target/<mesh name>_displaced.obj.
#[test]
#[ignore = "needs NOISER_TEST_MESH"]
fn export_a_real_mesh() {
    use crate::meshexport::{Displace, Format, Job, MeshEvaluator, Settings};
    let env = |k: &str| std::env::var(k).ok();
    let mesh_path = std::path::PathBuf::from(env("NOISER_TEST_MESH").expect("set NOISER_TEST_MESH"));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let graph = Graph::from_json(&std::fs::read_to_string(env("NOISER_TEST_GRAPH").map_or(root.join("SimpleTestGraph.noisegraph"), Into::into)).unwrap()).unwrap();
    let res: u32 = env("NOISER_TEST_RES").map_or(256, |v| v.parse().unwrap());
    let amount: f32 = env("NOISER_TEST_AMOUNT").map_or(0.25, |v| v.parse().unwrap());
    let seed: u32 = env("NOISER_TEST_SEED").map_or(0, |v| v.parse().unwrap());
    let gpu = gpu().expect("needs a GPU");
    let t = std::time::Instant::now();
    let (mesh, norm) = crate::mesh::Mesh::load_obj(&mesh_path).unwrap().normalized();
    eprintln!("loaded {} vertices, {} triangles in {:?}", mesh.positions.len(), mesh.triangles.len(), t.elapsed());
    let t = std::time::Instant::now();
    let pad = 0.5;
    let sdf = crate::meshsdf::MeshSdf::build(&mesh, 128, pad, &Default::default());
    eprintln!("distance grid {:?} in {:?}", sdf.dims, t.elapsed());
    let (_tex, view) = sdf.upload(&gpu.device, &gpu.queue);
    let ev = MeshEvaluator::new(&gpu.device, &gpu.queue, &graph, seed, view, sdf.origin, sdf.voxel).unwrap();
    let push = crate::view::push_range(graph.density_range(), amount);
    let reach = push.0.abs().max(push.1.abs()).min(pad - 0.05);
    let (lo, hi) = mesh.bounds();
    let settings = Settings { res, format: Format::Obj, cull: 0.01 };
    let job = Job { ev: &ev, d: Displace { amount, tile_size: 1.0, prim_size: 1.0 }, lo, hi, reach, norm, settings };
    eprintln!("export grid {:?}", job.grid().dims);
    let t = std::time::Instant::now();
    let out = root.join("target").join(format!("{}_displaced.obj", mesh_path.file_stem().unwrap().to_string_lossy()));
    let summary = job.run(&out, &Default::default()).unwrap();
    eprintln!("{summary}\nexport took {:?}", t.elapsed());
}
