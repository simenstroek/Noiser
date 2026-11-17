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
    let mut gpu = view::build(&device, &queue, wgpu::TextureFormat::Bgra8Unorm, &wgsl, false, false);
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
        let push = view::push_range(graph.density_range(), 0.25, (1.0, 1.0));
        let u = view::Uniforms {
            cam_pos: pos, aspect: 1.0, cam_fwd: fwd, amount: 0.25, cam_right: right, step_scale: step, cam_up: up, seed: 0,
            tile_size: 1.0, prim_size: 1.0, pixel: 2.0 / size[1] as f32, cloud_density: 8.0, mode, prim: 0,
            slice_axis: 2, slice_mode: 0, slice_pos: 0.5, slice_aspect: 1.0, slice_tiles: 1.0, slice_res: view::SLICE_RES,
            push_min: push.0, push_max: push.1, _pad: [0.0; 2], mesh_origin: [0.0; 3], mesh_voxel: 1.0, mesh_lo: [0.0; 3], mask_n: 0, mesh_hi: [0.0; 3], _pad_mask: 0.0, mask: [[0.0; 4]; 12],
            look: [[0.8, 0.78, 0.74, 0.0], [0.5, 0.0, 0.0, 0.0]],
            light: crate::Lighting::default().pack(),
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
    near(math(Floor, -2.5, 9.0, 0.5), -3.0, "floor of a negative");
    near(math(Fract, -2.25, 9.0, 0.5), 0.75, "fraction of a negative");
    near(math(Sine, std::f32::consts::FRAC_PI_2, 9.0, 0.5), 1.0, "sine");
    near(math(Sqrt, 9.0, 0.0, 0.5), 3.0, "square root");
    near(math(Sqrt, -4.0, 0.0, 0.5), 0.0, "square root of a negative is 0");
    near(math(Modulo, -1.0, 3.0, 0.5), 2.0, "modulo takes the sign of the divisor");
    near(math(Modulo, 7.5, 2.0, 0.5), 1.5, "modulo");
    near(math(Modulo, 7.5, 0.0, 0.5), 0.0, "modulo by zero");
    near(math(Greater, 2.0, 1.0, 0.5), 1.0, "greater");
    near(math(Greater, 1.0, 1.0, 0.5), 0.0, "equal is not greater");
    near(math(Absolute, -2.5, 9.0, 0.5), 2.5, "absolute of a negative");
    near(math(Absolute, 2.5, 9.0, 0.5), 2.5, "absolute of a positive");
    near(math(Ridge, 0.25, 9.0, 0.5), 0.75, "ridge");
    near(math(Ridge, -0.25, 9.0, 0.5), 0.75, "ridge is symmetric");
    near(math(Ridge, 3.0, 9.0, 0.5), -2.0, "ridge below zero past 1");
    // On a vector the unary ops act per component: |(−1, 2, −6)| averages to 3.
    let mut g = Graph::default();
    let v = g.add(NodeKind::Vec3([-1.0, 2.0, -6.0]), [0.0; 2]);
    let m = g.add(NodeKind::Math { op: Absolute, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    let o = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(v, m, 0);
    g.connect(m, o, 0);
    near(eval(&gpu, &g, &at)[0], 3.0, "absolute of a vector");
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
fn random_graphs_build_and_evaluate_on_the_gpu() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rnd = move |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let at = [[0.3, 0.6, 0.9], [0.1, 0.2, 0.3]];
    let rounds: usize = std::env::var("NOISER_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    for round in 0..rounds {
        let g = codegen::random_graph(&mut rnd);
        let v = eval(&gpu, &g, &at); // panics if the driver rejects the shader
        assert_eq!(v.len(), 2, "round {round}");
    }
}

#[test]
fn mix_matches_reference() {
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let mix = |vals: [f32; 3], clamp: bool| {
        let mut g = Graph::default();
        let m = g.add(NodeKind::Mix { vals, clamp }, [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(m, o, 0);
        eval(&gpu, &g, &[[0.3, 0.6, 0.9]])[0]
    };
    near(mix([0.25, 2.0, 6.0], true), 3.0, "a quarter of the way");
    near(mix([1.5, 2.0, 6.0], false), 8.0, "extrapolates without the clamp");
    near(mix([1.5, 2.0, 6.0], true), 6.0, "clamped at B");
    near(mix([-1.0, 2.0, 6.0], true), 2.0, "clamped at A");
}

#[test]
fn curve_matches_reference() {
    use crate::graph::CurveMode::*;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let at = [[0.3, 0.6, 0.9]];
    // Deliberately unsorted: the node sorts them.
    let pts = vec![[1.0, 0.0], [0.0, 1.0], [0.5, 3.0]];
    let curve = |x: f32, mode| {
        let mut g = Graph::default();
        let f = g.add(NodeKind::Float(x), [0.0; 2]);
        let c = g.add(NodeKind::Curve { points: pts.clone(), mode }, [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(f, c, 0);
        g.connect(c, o, 0);
        eval(&gpu, &g, &at)[0]
    };
    near(curve(0.25, Linear), 2.0, "linear first segment");
    near(curve(0.75, Linear), 1.5, "linear second segment");
    near(curve(-1.0, Linear), 1.0, "holds the first value");
    near(curve(2.0, Linear), 0.0, "holds the last value");
    near(curve(0.25, Smooth), 2.0, "smoothstep at the midpoint");
    near(curve(0.125, Smooth), 1.0 + 2.0 * (3.0 * 0.0625 - 2.0 * 0.015625), "smoothstep off the midpoint");
    near(curve(0.25, Step), 1.0, "step holds the left value");
    near(curve(0.75, Step), 3.0, "step second segment");
    // Cubic: slopes 4 (end), 0 (a peak), -6 (end); hand-computed Hermite values.
    near(curve(0.25, Cubic), 2.25, "cubic first segment");
    near(curve(0.75, Cubic), 1.875, "cubic second segment");
    near(curve(0.5, Cubic), 3.0, "cubic passes through a point");
    for i in 0..=40 {
        let v = curve(i as f32 / 40.0, Cubic);
        assert!((-1e-4..=3.0001).contains(&v), "cubic overshoots its points at {i}: {v}");
    }
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
    use crate::graph::PatternOutput;
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
        ("checker", single_node_graph(NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Checker)))),
        ("stripes_square", single_node_graph(NodeKind::Pattern({ let mut p = crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Stripes); p.variant = 3; p.dir = [1.0, 1.0, 0.0]; p.counts[0] = 6.0; p.width = 0.4; p }))),
        ("dimples_golf", single_node_graph(NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Dimples)))),
        ("gyroid_shell", single_node_graph(NodeKind::Pattern({ let mut p = crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Gyroid); p.output = PatternOutput::Shape; p.counts = [3.0; 3]; p }))),
        ("honeycomb", single_node_graph(NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Honeycomb)))),
        ("honeycomb_cells", single_node_graph(NodeKind::Pattern({ let mut p = crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Honeycomb); p.output = PatternOutput::CellValue; p }))),
        ("menger", single_node_graph(NodeKind::Pattern({ let mut p = crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Fractal); p.iterations = 4; p }))),
        ("sierpinski", single_node_graph(NodeKind::Pattern({ let mut p = crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Fractal); p.variant = 1; p.iterations = 4; p }))),
        ("truchet_arcs", single_node_graph(NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Truchet)))),
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
        let mesh = shapes::icosphere(1.0, 3);
        let bounds = mesh.bounds();
        let sdf = MeshSdf::build(&mesh, 64, 0.6, &AtomicU32::new(0));
        let (texture, view) = sdf.upload(&gpu.device, &gpu.queue);
        let ev = MeshEvaluator::new(&gpu.device, &gpu.queue, graph, seed, view, sdf.origin, sdf.voxel, bounds, true).unwrap();
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
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
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
        let d = Displace { amount: 0.12, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
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
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 2.0, mask: crate::meshmask::PackedMask::NONE };
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

    /// The mask value at each exported vertex of a unit sphere. With a constant density of 1 the
    /// surface moves outward by `amount * mask`, so mask = (radius - 1) / amount.
    fn probe_mask(gpu: &Gpu, terms: Vec<crate::meshmask::MaskTerm>) -> Vec<([f32; 3], f32)> {
        let amount = 0.1;
        let mask = crate::meshmask::MaskSettings { terms }.pack();
        let (_sdf, ev, _tex) = evaluator(gpu, &single_node_graph(NodeKind::Float(1.0)), 0);
        let d = Displace { amount, tile_size: 1.0, prim_size: 1.0, mask };
        let j = job(&ev, d, 0.3, 80);
        let m = j.extract(&Default::default());
        assert!(!m.positions.is_empty());
        m.positions.iter().map(|p| (*p, (radius(*p) - 1.0) / amount)).collect()
    }

    #[test]
    fn mask_terms_vary_the_displacement_as_described() {
        use crate::meshmask::{MaskKind, MaskTerm};
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let mean = |v: &[([f32; 3], f32)]| v.iter().map(|(_, m)| m).sum::<f32>() / v.len() as f32;
        let (lo, hi) = shapes::icosphere(1.0, 3).bounds();

        // A constant is just a strength: weight 2 moves everything out by 0.2.
        let mut c = MaskTerm::new(MaskKind::Constant);
        c.weight = 2.0;
        let v = probe_mask(&gpu, vec![c]);
        assert!(v.iter().all(|(_, m)| (m - 2.0).abs() < 0.25), "constant 2: {:?}", (mean(&v)));

        // A ramp up the Y axis: 0 at the bottom of the mesh, 1 at the top.
        let ramp = MaskTerm::new(MaskKind::Position);
        let v = probe_mask(&gpu, vec![ramp]);
        let worst = v.iter().map(|(p, m)| {
            let s = p.map(|c| c / radius(*p)); // the surface point this vertex came from
            (m - ((s[1] - lo[1]) / (hi[1] - lo[1])).clamp(0.0, 1.0)).abs()
        }).fold(0.0, f32::max);
        assert!(worst < 0.15, "position ramp is off by {worst}");
        let end = |keep: &dyn Fn(f32) -> bool| mean(&v.iter().cloned().filter(|(p, _)| keep(p[1])).collect::<Vec<_>>());
        let (bottom, top) = (end(&|y| y < -0.85), end(&|y| y > 0.85));
        assert!(bottom < 0.1 && top > 0.85, "bottom {bottom}, top {top}");

        // The same ramp the other way (negative direction) and inverted both flip it.
        let mut down = MaskTerm::new(MaskKind::Position);
        down.negative = true;
        let v_down = probe_mask(&gpu, vec![down]);
        assert!((mean(&v_down) - 0.5).abs() < 0.1 && mean(&v_down.iter().filter(|(p, _)| p[1] > 0.8).map(|(p, m)| (*p, *m)).collect::<Vec<_>>()) < 0.1);

        // Normal direction: facing up counts fully, the equator and below not at all.
        let up = MaskTerm::new(MaskKind::Normal);
        let v = probe_mask(&gpu, vec![up]);
        let worst = v.iter().map(|(p, m)| (m - (p[1] / radius(*p)).clamp(0.0, 1.0)).abs()).fold(0.0, f32::max);
        assert!(worst < 0.15, "normal term is off by {worst}");

        // Curvature: a unit sphere has mean curvature 1 everywhere, which is 0.75 of the way from -2 to 2.
        let v = probe_mask(&gpu, vec![MaskTerm::new(MaskKind::Curvature)]);
        assert!((mean(&v) - 0.75).abs() < 0.12, "curvature mean {}", mean(&v));

        // Occlusion: a convex sphere is fully open.
        let v = probe_mask(&gpu, vec![MaskTerm::new(MaskKind::Occlusion)]);
        assert!(mean(&v) > 0.9, "occlusion mean {}", mean(&v));

        // Thickness: the solid behind a unit sphere's surface is its diameter, 2, which is half of 0 to 4.
        let mut t = MaskTerm::new(MaskKind::Thickness);
        t.hi = 4.0;
        let v = probe_mask(&gpu, vec![t]);
        assert!((mean(&v) - 0.5).abs() < 0.1, "thickness mean {}", mean(&v));

        // Terms add: a constant 0.5 plus an inverted ramp.
        let mut half = MaskTerm::new(MaskKind::Constant);
        half.weight = 0.5;
        let mut inv = MaskTerm::new(MaskKind::Position);
        inv.invert = true;
        inv.weight = 0.5;
        let v = probe_mask(&gpu, vec![half, inv]);
        assert!((mean(&v.iter().filter(|(p, _)| p[1] < -0.8).map(|(p, m)| (*p, *m)).collect::<Vec<_>>()) - 1.0).abs() < 0.1);
        assert!((mean(&v.iter().filter(|(p, _)| p[1] > 0.8).map(|(p, m)| (*p, *m)).collect::<Vec<_>>()) - 0.5).abs() < 0.1);
    }

    #[test]
    fn seeds_change_the_result_and_repeat() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let noise = noise_graph(NoiseParams { scale: 3.0, octaves: 2, ..Default::default() });
        let d = Displace { amount: 0.1, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
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
    let ev = MeshEvaluator::new(&gpu.device, &gpu.queue, &graph, seed, view, sdf.origin, sdf.voxel, mesh.bounds(), false).unwrap();
    let push = crate::view::push_range(graph.density_range(), amount, (1.0, 1.0));
    let reach = push.0.abs().max(push.1.abs()).min(pad - 0.05);
    let (lo, hi) = mesh.bounds();
    let settings = Settings { res, format: Format::Obj, cull: 0.01 };
    let job = Job { ev: &ev, d: Displace { amount, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE }, lo, hi, reach, norm, settings };
    eprintln!("export grid {:?}", job.grid().dims);
    let t = std::time::Instant::now();
    let out = root.join("target").join(format!("{}_displaced.obj", mesh_path.file_stem().unwrap().to_string_lossy()));
    let summary = job.run(&out, &Default::default()).unwrap();
    eprintln!("{summary}\nexport took {:?}", t.elapsed());
}

// ---- batch pipeline ---------------------------------------------------------------

mod batch_tests {
    use super::*;
    use crate::mesh::shapes;
    use crate::meshexport::{Displace, FileJob, Format, SeedMode, Settings};
    use crate::surface::RawMesh;

    fn write_sphere_obj(path: &std::path::Path, radius: f32) {
        let m = shapes::icosphere(radius, 2);
        let raw = RawMesh { normals: vec![[0.0, 0.0, 1.0]; m.positions.len()], positions: m.positions, triangles: m.triangles };
        raw.write_obj(path).unwrap();
    }

    #[test]
    fn seed_modes_are_distinct_repeatable_and_exact() {
        assert_eq!((0..4).map(|i| SeedMode::Same.seed_for(7, i)).collect::<Vec<_>>(), [7, 7, 7, 7]);
        assert_eq!((0..4).map(|i| SeedMode::Increment.seed_for(7, i)).collect::<Vec<_>>(), [7, 8, 9, 10]);
        let random: Vec<u32> = (0..50).map(|i| SeedMode::Random.seed_for(7, i)).collect();
        assert_eq!(random, (0..50).map(|i| SeedMode::Random.seed_for(7, i)).collect::<Vec<_>>());
        let unique: std::collections::HashSet<_> = random.iter().collect();
        assert_eq!(unique.len(), 50, "random seeds should differ");
        assert!(random.iter().all(|s| *s < (1 << 24)));
        assert_ne!(SeedMode::Random.seed_for(8, 0), random[0], "another global seed gives other seeds");
    }

    #[test]
    fn a_batch_gives_each_mesh_its_own_seed_and_repeats() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let dir = std::env::temp_dir().join("noiser-batch-test");
        std::fs::create_dir_all(&dir).unwrap();
        // Two different meshes (sizes and positions differ; normalization makes the noise match).
        write_sphere_obj(&dir.join("small.obj"), 1.0);
        write_sphere_obj(&dir.join("big.obj"), 3.0);
        let noise = noise_graph(NoiseParams { scale: 3.0, octaves: 2, ..Default::default() });
        let run = |input: &str, seed: u32, out: &str| {
            let job = FileJob {
                device: &gpu.device,
                queue: &gpu.queue,
                graph: &noise,
                seed,
                d: Displace { amount: 0.1, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE },
                settings: Settings { res: 48, format: Format::Obj, cull: 0.0 },
                grid_res: 48,
                grid_pad: 0.5,
            };
            let out = dir.join(out);
            job.run(&dir.join(input), &out, &Default::default()).unwrap();
            crate::mesh::Mesh::load_obj(&out).unwrap()
        };
        let a = run("small.obj", 0, "a.obj");
        let a2 = run("small.obj", 0, "a2.obj");
        let b = run("small.obj", 1, "b.obj");
        assert_eq!(a.positions, a2.positions, "the same seed repeats");
        assert!(a.positions.len() != b.positions.len() || a.positions != b.positions, "another seed differs");
        // The big mesh is the same shape scaled up: same noise relative to the mesh, so the same
        // topology, and every vertex at three times the position.
        let big = run("big.obj", 0, "c.obj");
        assert_eq!(big.positions.len(), a.positions.len(), "scaled copies should displace identically");
        let worst = a.positions.iter().zip(&big.positions).map(|(p, q)| (0..3).map(|k| (p[k] * 3.0 - q[k]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
        assert!(worst < 0.02, "scaled mesh differs by {worst}");
        // A missing file is an error, not a panic.
        let job = FileJob { device: &gpu.device, queue: &gpu.queue, graph: &noise, seed: 0, d: Displace { amount: 0.1, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE }, settings: Settings::default(), grid_res: 32, grid_pad: 0.5 };
        assert!(job.run(&dir.join("nope.obj"), &dir.join("x.obj"), &Default::default()).is_err());
    }
}

// ---- more patterns: checker, stripes, dimples, gyroid ---------------------------------

mod pattern_family_tests {
    use super::*;
    use crate::graph::{PatternKind, PatternOutput, PatternParams};

    fn pattern(kind: PatternKind, f: impl FnOnce(&mut PatternParams)) -> Graph {
        let mut p = PatternParams::for_kind(kind);
        f(&mut p);
        single_node_graph(NodeKind::Pattern(p))
    }

    /// Range inside [-1, 1] and (optionally) filling it, no NaN, and the same values one tile later
    /// on every axis (a few points right on an edge may fall either side).
    fn check(gpu: &Gpu, g: &Graph, what: &str, fills: bool) {
        let pts = points(20000, 11);
        let v = eval(gpu, g, &pts);
        assert!(v.iter().all(|x| x.is_finite()), "{what}: NaN or infinity");
        let (lo, hi) = (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max));
        assert!(lo >= -1.0001 && hi <= 1.0001, "{what}: range {lo}..{hi}");
        if fills {
            assert!(lo < -0.9 && hi > 0.9, "{what}: range {lo}..{hi} does not fill -1..1");
        }
        for axis in 0..3 {
            let moved: Vec<[f32; 3]> = pts.iter().map(|p| { let mut q = *p; q[axis] += 1.0; q }).collect();
            let w = eval(gpu, g, &moved);
            let off = v.iter().zip(&w).filter(|(a, b)| (*a - *b).abs() > 2e-3).count();
            assert!(off <= 4, "{what}, axis {axis}: {off} of {} points differ one tile later", pts.len());
        }
    }

    #[test]
    fn checker_stripes_dimples_and_gyroid_stay_in_range_and_tile() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        for (soft, flat) in [(0.0, true), (0.3, true), (0.05, false), (1.0, false)] {
            check(&gpu, &pattern(Checker, |p| { p.softness = soft; p.mode2d = flat; }), &format!("checker soft {soft} 2d={flat}"), soft <= 0.3); // very soft edges rarely reach +-0.9 in random samples
        }
        for wave in 0..4 {
            for dir in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, -1.0]] {
                check(&gpu, &pattern(Stripes, |p| { p.variant = wave; p.dir = dir; p.counts[0] = 5.0; }), &format!("stripes wave {wave} dir {dir:?}"), true);
            }
        }
        for lattice in 0..3 {
            check(&gpu, &pattern(Dimples, |p| { p.variant = lattice; p.output = PatternOutput::Distance; }), &format!("dimple distance, lattice {lattice}"), true);
            for profile in 0..4 {
                check(&gpu, &pattern(Dimples, |p| { p.variant = lattice; p.profile = profile; p.output = PatternOutput::Shape; }), &format!("dimples lattice {lattice} profile {profile}"), true);
            }
        }
        for surface in 0..3 {
            for output in [PatternOutput::Distance, PatternOutput::Shape] {
                check(&gpu, &pattern(Gyroid, |p| { p.variant = surface; p.output = output; }), &format!("gyroid {surface} {output:?}"), true);
            }
        }
        for variant in 0..2 {
            for iterations in [1, 2, 4, 6] {
                for counts in [[1.0; 3], [2.0, 3.0, 1.0]] {
                    check(&gpu, &pattern(Fractal, |p| { p.variant = variant; p.iterations = iterations; p.counts = counts; }), &format!("fractal {variant} x{iterations} {counts:?}"), false);
                }
            }
        }
        for variant in 0..3 {
            for flat in [true, false] {
                check(&gpu, &pattern(Truchet, |p| { p.variant = variant; p.mode2d = flat; }), &format!("truchet {variant} 2d={flat}"), true);
            }
        }
    }

    #[test]
    fn fractals_and_truchet_tiles_have_their_known_points() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        // A Menger sponge: the middle of the cell and of a face are holes, a corner is solid, at any depth.
        for iterations in 1..=5 {
            let g = pattern(Fractal, |p| { p.variant = 0; p.iterations = iterations; });
            let v = eval(&gpu, &g, &[[0.5, 0.5, 0.5], [0.0005, 0.0005, 0.0005], [0.5, 0.5, 0.02]]); // the corner point is in the corner cube at every depth
            assert!(v[0] > 0.99, "depth {iterations}: the middle should be empty, got {}", v[0]);
            assert!(v[1] < 0.0, "depth {iterations}: a corner should be solid, got {}", v[1]);
            assert!(v[2] > 0.99, "depth {iterations}: the middle of a face is the mouth of a through-hole, got {}", v[2]);
        }
        // More iterations carve out more: the share of solid space shrinks (a sponge keeps 20/27 of its cube each time).
        let solid_share = |iterations: u32| {
            let v = eval(&gpu, &pattern(Fractal, |p| { p.variant = 0; p.iterations = iterations; }), &points(30000, 6));
            v.iter().filter(|x| **x < 0.0).count() as f32 / v.len() as f32
        };
        let (s1, s2, s3) = (solid_share(1), solid_share(2), solid_share(3));
        assert!(s1 > s2 && s2 > s3, "solid share should fall with depth: {s1} {s2} {s3}");
        assert!((s1 - 20.0 / 27.0).abs() < 0.03, "one level keeps about 20/27 of the cube, got {s1}");

        // Truchet arcs pass through the midpoint of every cell edge whichever way the cell is flipped,
        // and are empty at the cell centre (width 0.3, so the line is 0.15 from its centre line).
        let n = 6.0;
        let arcs = pattern(Truchet, |_| {});
        let mut pts = Vec::new();
        for (i, j) in [(0.0, 0.0), (1.0, 3.0), (4.0, 2.0), (5.0, 5.0)] {
            for (dx, dy) in [(0.5, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 1.0)] {
                pts.push([(i + dx) / n, (j + dy) / n, 0.37]);
            }
            pts.push([(i + 0.5) / n, (j + 0.5) / n, 0.37]);
        }
        let v = eval(&gpu, &arcs, &pts);
        for (k, chunk) in v.chunks(5).enumerate() {
            for e in 0..4 {
                near(chunk[e], -1.0, &format!("cell {k}: an arc should cross the edge midpoint {e}"));
            }
            near(chunk[4], 1.0, &format!("cell {k}: the centre is off the arcs"));
        }
        // Diagonals cross at the centre of every cell and are empty at the midpoint of an edge.
        let diagonals = pattern(Truchet, |p| p.variant = 1);
        let v = eval(&gpu, &diagonals, &[[0.5 / n, 0.5 / n, 0.0], [3.5 / n, 2.5 / n, 0.0], [0.5 / n, 0.0, 0.0]]);
        near(v[0], -1.0, "diagonal through a cell centre");
        near(v[1], -1.0, "diagonal through another cell centre");
        near(v[2], 1.0, "no diagonal at an edge midpoint");
        // Same seed, same tiles; another seed flips some cells.
        let seeded = |seed: u32| eval(&gpu, &pattern(Truchet, |p| p.seed = Some(seed)), &points(3000, 2));
        assert_eq!(seeded(1), seeded(1));
        assert!(seeded(1).iter().zip(seeded(2)).any(|(a, b)| (a - b).abs() > 0.5), "another seed should change the layout");
    }

    #[test]
    fn truchet_pipes_pass_through_every_face_centre() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let n = 4.0;
        let pipes = pattern(Truchet, |p| { p.variant = 2; p.counts = [n; 3]; });
        let mut pts = Vec::new();
        for (i, j, k) in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (3.0, 1.0, 2.0)] {
            for (dx, dy, dz) in [(0.0, 0.5, 0.5), (1.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 0.0), (0.5, 0.5, 1.0)] {
                pts.push([(i + dx) / n, (j + dy) / n, (k + dz) / n]);
            }
        }
        for (i, v) in eval(&gpu, &pipes, &pts).iter().enumerate() {
            assert!(*v < -0.99, "face centre {i} should be inside a pipe, got {v}");
        }
        let share = |v: Vec<f32>| v.iter().filter(|x| **x < 0.0).count() as f32 / v.len() as f32;
        let s = share(eval(&gpu, &pipes, &points(4000, 3)));
        assert!(s > 0.05 && s < 0.4, "pipes should fill part of the volume, got {s}");
    }

    #[test]
    fn checker_cells_alternate_and_stripes_follow_their_wave() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        // 8 cells per tile: cell (i, j) has its centre at ((i + 0.5) / 8, (j + 0.5) / 8).
        let c = |i: f32, j: f32, k: f32| [(i + 0.5) / 8.0, (j + 0.5) / 8.0, (k + 0.5) / 8.0];
        let v = eval(&gpu, &pattern(Checker, |_| {}), &[c(0.0, 0.0, 0.3), c(1.0, 0.0, 0.7), c(1.0, 1.0, 0.1), c(2.0, 1.0, 0.9)]);
        for (got, want, what) in [(v[0], 1.0, "cell (0,0)"), (v[1], -1.0, "cell (1,0)"), (v[2], 1.0, "cell (1,1)"), (v[3], -1.0, "cell (2,1)")] {
            near(got, want, what);
        }
        // In 3D the third axis flips it too.
        let v = eval(&gpu, &pattern(Checker, |p| p.mode2d = false), &[c(0.0, 0.0, 0.0), c(0.0, 0.0, 1.0), c(1.0, 0.0, 1.0)]);
        near(v[0], 1.0, "3D cell (0,0,0)");
        near(v[1], -1.0, "3D cell (0,0,1)");
        near(v[2], 1.0, "3D cell (1,0,1)");

        // 4 stripes along X. A quarter of the way through a period: sine peaks, triangle is 0, sawtooth -0.5.
        let at = |phase: f32| [phase / 4.0, 0.4, 0.6];
        let stripe = |wave: u32, f: &dyn Fn(&mut PatternParams)| {
            let g = pattern(Stripes, |p| { p.variant = wave; p.counts[0] = 4.0; f(p) });
            eval(&gpu, &g, &[at(0.25), at(0.5), at(0.75)])
        };
        let s = stripe(0, &|_| {});
        near(s[0], 1.0, "sine at 1/4");
        near(s[1], 0.0, "sine at 1/2");
        near(s[2], -1.0, "sine at 3/4");
        let t = stripe(1, &|_| {});
        near(t[0], 0.0, "triangle at 1/4");
        near(t[1], 1.0, "triangle at 1/2");
        let w = stripe(2, &|_| {});
        near(w[0], -0.5, "sawtooth at 1/4");
        near(w[2], 0.5, "sawtooth at 3/4");
        // Square with a duty of 0.5 and hard edges: +1 around the middle of the period, -1 at its ends.
        let q = stripe(3, &|p| { p.width = 0.5; p.softness = 0.0; });
        near(q[1], 1.0, "square at 1/2");
        let ends = eval(&gpu, &pattern(Stripes, |p| { p.variant = 3; p.counts[0] = 4.0; p.width = 0.5; p.softness = 0.0; }), &[at(0.05), at(0.95)]);
        near(ends[0], -1.0, "square near the start of a period");
        near(ends[1], -1.0, "square near the end of a period");
        // A diagonal runs along (1, 1, 0): moving along (1, -1, 0) changes nothing.
        let d = pattern(Stripes, |p| { p.variant = 0; p.dir = [1.0, 1.0, 0.0]; p.counts[0] = 3.0; });
        let v = eval(&gpu, &d, &[[0.3, 0.2, 0.1], [0.4, 0.1, 0.1], [0.5, 0.0, 0.9]]);
        near(v[0], v[1], "along the stripe");
        near(v[0], v[2], "along the stripe, another z");
    }

    #[test]
    fn dimple_lattices_have_their_dimples_where_they_should() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        // 4 cubic cells per tile: cell edge 0.25. Lattice points: cubic at cell centres, body-centred
        // at the corners and centres, face-centred at the corners and face centres.
        let a = 0.25;
        let cases: [(u32, Vec<[f32; 3]>); 3] = [
            (0, vec![[0.5 * a, 0.5 * a, 0.5 * a], [1.5 * a, 2.5 * a, 0.5 * a]]),
            (1, vec![[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a], [a, 2.0 * a, 0.0]]),
            (2, vec![[0.0, 0.0, 0.0], [0.0, 0.5 * a, 0.5 * a], [0.5 * a, 0.0, 0.5 * a], [a, 0.5 * a, 0.5 * a]]),
        ];
        for (lattice, centres) in cases {
            let distance = eval(&gpu, &pattern(Dimples, |p| { p.variant = lattice; p.output = PatternOutput::Distance; }), &centres);
            let shape = eval(&gpu, &pattern(Dimples, |p| { p.variant = lattice; p.output = PatternOutput::Shape; p.radius = 0.9; }), &centres);
            for k in 0..centres.len() {
                near(distance[k], -1.0, &format!("lattice {lattice}: distance at point {k}"));
                near(shape[k], -1.0, &format!("lattice {lattice}: dimple depth at point {k}"));
            }
        }
        // Far from every dimple, with a radius below 1, the surface is flat: a cubic cell's corner.
        let flat = eval(&gpu, &pattern(Dimples, |p| { p.variant = 0; p.output = PatternOutput::Shape; p.radius = 0.8; }), &[[0.0, 0.0, 0.0], [a, a, a]]);
        near(flat[0], 1.0, "between cubic dimples");
        near(flat[1], 1.0, "between cubic dimples, another corner");
        // And a radius of 0.5 leaves flat ground a quarter of the way to the next point.
        let half = eval(&gpu, &pattern(Dimples, |p| { p.variant = 0; p.output = PatternOutput::Shape; p.radius = 0.5; p.profile = 3; }), &[[0.5 * a + 0.3 * a, 0.5 * a, 0.5 * a], [0.5 * a + 0.1 * a, 0.5 * a, 0.5 * a]]);
        near(half[0], 1.0, "outside a flat-bottomed dimple of radius 0.5");
        near(half[1], -1.0, "inside a flat-bottomed dimple of radius 0.5");

        // A cubic lattice's distance is the Grid node's cell-centre distance, anisotropic counts included.
        let pts = points(4000, 3);
        let cubic = pattern(Dimples, |p| { p.counts = [8.0, 4.0, 2.0]; p.variant = 0; p.output = PatternOutput::Distance; });
        let grid = single_node_graph(NodeKind::Grid { counts: [8.0, 4.0, 2.0], corners: false });
        let (x, y) = (eval(&gpu, &cubic, &pts), eval(&gpu, &grid, &pts));
        let worst = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(worst < 1e-4, "cubic dimple distance differs from Grid by {worst}");
        // A golf ball (body-centred, sphere cap, radius 0.9) is mostly flat ground between pits.
        let golf = eval(&gpu, &pattern(Dimples, |_| {}), &points(20000, 4));
        let flat_share = golf.iter().filter(|v| **v > 0.999).count() as f32 / golf.len() as f32;
        assert!((0.25..0.7).contains(&flat_share), "flat share {flat_share}");
    }

    #[test]
    fn honeycomb_cells_have_centres_walls_and_flat_values() {
        use PatternKind::*;
        use PatternOutput::{CellValue, Distance, Shape};
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        for variant in 0..4 {
            for output in [Distance, Shape, CellValue] {
                let g = pattern(Honeycomb, |p| { p.variant = variant; p.mode2d = variant == 0 || variant == 3; p.output = output; if variant > 0 { p.counts = [4.0; 3]; } });
                check(&gpu, &g, &format!("honeycomb {variant} {output:?}"), output != CellValue);
            }
        }
        // Triangles, 4 x 4 counts: X = 8 q.x, Y = 4 q.y. The up triangle (0,0) (2,0) (1,1) has its
        // incentre at (1, 1/(1+sqrt 2)) and its walls along Y = 0 and X + Y = 2.
        let tri = pattern(Honeycomb, |p| { p.variant = 3; p.counts = [4.0; 3]; p.output = Distance; });
        let inc = 1.0 / (1.0 + 2f32.sqrt());
        let v = eval(&gpu, &tri, &[[1.0 / 8.0, inc / 4.0, 0.3], [0.2, 0.0, 0.3], [(2.0 - 0.5) / 8.0, 0.5 / 4.0, 0.3]]);
        near(v[0], 1.0, "triangle incentre");
        near(v[1], -1.0, "triangle wall Y = 0");
        near(v[2], -1.0, "triangle wall X + Y = 2");
        // Lattice points are cell centres (depth 1); the midpoint between two neighbouring points is
        // on the wall between their cells (depth 0). counts are cells per tile; the points sit at
        // (cell + b) / counts.
        let probes: [(u32, [f32; 3], [f32; 3], [f32; 3]); 3] = [
            // hexagons, 7 x 4: points (0,0) and (1/2,1/2) of a cell, extruded along z
            (0, [7.0, 4.0, 4.0], [0.0, 0.0, 0.3], [0.5 / 7.0, 0.0, 0.3]),
            // body-centred, 4 per side: (0,0,0) and (1/2,1/2,1/2) of a cell
            (1, [4.0; 3], [0.0; 3], [0.25 * 0.25, 0.25 * 0.25, 0.25 * 0.25]),
            // face-centred, 4 per side: (0,0,0) and (0,1/2,1/2) of a cell
            (2, [4.0; 3], [0.0; 3], [0.0, 0.25 * 0.25, 0.25 * 0.25]),
        ];
        for (variant, counts, centre, wall) in probes {
            let make = |output: PatternOutput| pattern(Honeycomb, |p| { p.variant = variant; p.mode2d = variant == 0; p.counts = counts; p.output = output; p.gap = 0.1; p.bevel = 0.1; });
            let d = eval(&gpu, &make(Distance), &[centre, wall]);
            let sh = eval(&gpu, &make(Shape), &[centre, wall]);
            near(d[0], 1.0, &format!("variant {variant}: a lattice point is a cell centre"));
            near(d[1], -1.0, &format!("variant {variant}: halfway between neighbours is a wall"));
            near(sh[0], -1.0, &format!("variant {variant}: inside a cell"));
            near(sh[1], 1.0, &format!("variant {variant}: on a wall"));
        }
        // Hexagons are extruded: nothing changes along z.
        let hex = pattern(Honeycomb, |_| {});
        let v = eval(&gpu, &hex, &[[0.13, 0.21, 0.1], [0.13, 0.21, 0.6], [0.13, 0.21, 0.95]]);
        near(v[0], v[1], "hexagons along z");
        near(v[0], v[2], "hexagons along z, again");
        // Cell values are flat inside a cell and differ between cells.
        let cells = pattern(Honeycomb, |p| p.output = CellValue);
        let v = eval(&gpu, &cells, &[[0.0, 0.0, 0.3], [0.01, 0.01, 0.3], [0.5 / 7.0, 0.5 / 4.0, 0.3], [1.0 / 7.0, 0.0, 0.3]]);
        near(v[0], v[1], "two points in one cell");
        assert!((v[0] - v[2]).abs() > 1e-3 && (v[0] - v[3]).abs() > 1e-3, "neighbouring cells should differ: {v:?}");
    }

    #[test]
    fn gyroid_surfaces_have_their_known_values() {
        use PatternKind::*;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let origin = [[0.0, 0.0, 0.0]];
        // At the origin: the gyroid is 0, Schwarz P is at its maximum 3, the diamond is 0.
        for (surface, want) in [(0, 0.0), (1, 1.0), (2, 0.0)] {
            let v = eval(&gpu, &pattern(Gyroid, |p| p.variant = surface), &origin);
            near(v[0], want, &format!("surface {surface} at the origin"));
        }
        // A shell is -1 where the field is 0 (the origin for the gyroid) and +1 where it is largest.
        let shell = pattern(Gyroid, |p| { p.output = PatternOutput::Shape; p.width = 0.2; });
        near(eval(&gpu, &shell, &origin)[0], -1.0, "gyroid shell at its surface");
        let p_shell = pattern(Gyroid, |p| { p.variant = 1; p.output = PatternOutput::Shape; p.width = 0.2; });
        near(eval(&gpu, &p_shell, &origin)[0], 1.0, "Schwarz P shell where the field peaks");
        // The field really reaches 1 (the normalization constants are the true extremes).
        for surface in 0..3 {
            let v = eval(&gpu, &pattern(Gyroid, |p| { p.variant = surface; p.counts = [1.0; 3]; }), &points(40000, 8));
            let hi = v.iter().cloned().fold(f32::MIN, f32::max);
            let lo = v.iter().cloned().fold(f32::MAX, f32::min);
            assert!(hi > 0.97 && lo < -0.97 && hi <= 1.0001 && lo >= -1.0001, "surface {surface}: {lo}..{hi}");
        }
    }
}

/// Prints the 99.9th percentile and maximum of each Worley feature for every metric and jitter, which
/// is where `worley_scale` in noise.wgsl comes from: `cargo test measure_worley_ranges -- --ignored --nocapture`.
#[test]
#[ignore = "measurement"]
fn measure_worley_ranges() {
    let gpu = gpu().expect("needs a GPU");
    let c = codegen::compile(&noise_graph(NoiseParams { kind: NoiseKind::Worley, jitter: 0.5, ..Default::default() })).unwrap();
    let pts: Vec<[f32; 4]> = points(200_000, 21).into_iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    let stat = |expr: &str| {
        let mut v = eval_expr(&gpu, &c, expr, &pts);
        v.sort_by(|a, b| a.total_cmp(b));
        (v[v.len() * 999 / 1000], v[v.len() - 1])
    };
    for metric in 0..3 {
        for jitter in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let f1 = stat(&format!("cells(q.xyz * 7.3, 0, 5u, false, {metric}u, {jitter:?}).x"));
            let f2 = stat(&format!("cells(q.xyz * 7.3, 0, 5u, false, {metric}u, {jitter:?}).y"));
            let d = stat(&format!("cells(q.xyz * 7.3, 0, 5u, false, {metric}u, {jitter:?}).y - cells(q.xyz * 7.3, 0, 5u, false, {metric}u, {jitter:?}).x"));
            let w = if metric == 0 { stat(&format!("cell_wall_distance(q.xyz * 7.3, 0, 5u, {jitter:?})")) } else { (0.0, 0.0) };
            eprintln!("metric {metric} jitter {jitter:4}: F1 {:.3}/{:.3}  F2 {:.3}/{:.3}  F2-F1 {:.3}/{:.3}  wall {:.3}/{:.3}   (99.9th / max)", f1.0, f1.1, f2.0, f2.1, d.0, d.1, w.0, w.1);
        }
    }
}

// ---- Worley variants ----------------------------------------------------------------

mod worley_variant_tests {
    use super::*;
    use crate::graph::WorleyMetric;

    fn worley(f: impl FnOnce(&mut NoiseParams)) -> Graph {
        let mut p = NoiseParams { kind: NoiseKind::Worley, scale: 4.0, octaves: 1, ..Default::default() };
        f(&mut p);
        noise_graph(p)
    }

    #[test]
    fn every_metric_jitter_and_feature_stays_in_range_and_tiles() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let pts = points(20000, 13);
        for metric in [WorleyMetric::Euclidean, WorleyMetric::Manhattan, WorleyMetric::Chebyshev] {
            for jitter in [0.0, 0.3, 0.7, 1.0] {
                for feature in [WorleyFeature::F1, WorleyFeature::F2, WorleyFeature::F2MinusF1, WorleyFeature::Edges] {
                    if feature == WorleyFeature::Edges && metric != WorleyMetric::Euclidean {
                        continue; // the cell edge distance is always Euclidean
                    }
                    let what = format!("{metric:?} jitter {jitter} {feature:?}");
                    let g = worley(|p| { p.metric = metric; p.jitter = jitter; p.feature = feature; });
                    let v = eval(&gpu, &g, &pts);
                    assert!(v.iter().all(|x| x.is_finite()), "{what}: NaN");
                    let (lo, hi) = (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max));
                    assert!(lo >= -1.0001 && hi <= 1.0001, "{what}: range {lo}..{hi}");
                    assert!(hi - lo > 0.8, "{what}: range {lo}..{hi} is too narrow");
                    for axis in 0..3 {
                        let moved: Vec<[f32; 3]> = pts.iter().map(|p| { let mut q = *p; q[axis] += 1.0; q }).collect();
                        let w = eval(&gpu, &g, &moved);
                        let off = v.iter().zip(&w).filter(|(a, b)| (*a - *b).abs() > 2e-3).count();
                        assert!(off <= 4, "{what}, axis {axis}: {off} points differ one tile later");
                    }
                }
            }
        }
    }

    #[test]
    fn distance_measures_and_cell_edges_match_hand_computed_values() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let c = codegen::compile(&worley(|p| p.jitter = 0.5)).unwrap(); // a graph that includes the general Worley
        // With jitter 0 every feature point is the middle of its cell, so the point nearest (0.75, 0.55, 0.62)
        // is (0.5, 0.5, 0.5): the offset is (0.25, 0.05, 0.12).
        let at = [[0.75, 0.55, 0.62, 0.0]];
        let raw = |metric: u32| eval_expr(&gpu, &c, &format!("cells(q.xyz, 0, 1u, false, {metric}u, 0.0).x"), &at)[0];
        near(raw(0), (0.25f32 * 0.25 + 0.05 * 0.05 + 0.12 * 0.12).sqrt(), "Euclidean");
        near(raw(1), 0.25 + 0.05 + 0.12, "Manhattan");
        near(raw(2), 0.25, "Chebyshev");
        // Cell edges of a perfect lattice: 0 at the wall between two cells, 0.5 (cell units) at a centre.
        let wall = |p: [f32; 3]| eval_expr(&gpu, &c, "cell_wall_distance(q.xyz, 0, 1u, 0.0)", &[[p[0], p[1], p[2], 0.0]])[0];
        near(wall([0.5, 0.5, 0.5]), 0.5, "wall distance at a cell centre");
        near(wall([0.999, 0.5, 0.5]), 0.001, "wall distance next to a wall");
        near(wall([0.8, 0.5, 0.5]), 0.2, "wall distance 0.2 from a wall");
        // Through the node: a lattice's F1 is -1 at a feature point, and its edges are -1 on a wall, +1 at a centre.
        let lattice = |feature: WorleyFeature, metric: WorleyMetric| worley(|p| { p.jitter = 0.0; p.feature = feature; p.metric = metric; });
        near(eval(&gpu, &lattice(WorleyFeature::F1, WorleyMetric::Euclidean), &[[0.5 / 4.0, 0.5 / 4.0, 0.5 / 4.0]])[0], -1.0, "F1 at a feature point");
        let edges = lattice(WorleyFeature::Edges, WorleyMetric::Euclidean);
        let v = eval(&gpu, &edges, &[[0.5 / 4.0, 0.5 / 4.0, 0.5 / 4.0], [1.0 / 4.0, 0.5 / 4.0, 0.5 / 4.0]]);
        near(v[0], 1.0, "edges at a cell centre");
        near(v[1], -1.0, "edges on a wall");
    }

    /// The Worley that shipped before jitter, metrics and edges, kept here under other names: with jitter 1
    /// and the Euclidean metric the new code must give exactly the same values for every original feature.
    const LEGACY_WORLEY: &str = "
fn cells_legacy(x: vec3f, period: i32, seed: u32, want_value: bool) -> vec3f {
    let i = wrap_cell(vec3i(floor(x)), period);
    let f = fract(x);
    var f1 = 1e9;
    var f2 = 1e9;
    var nearest = i;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let o = vec3i(xx, y, z);
                let c = wrap_near(i + o, period);
                let d = vec3f(o) + hash01(c, seed) - f;
                let d2 = dot(d, d);
                if (d2 < f1) { f2 = f1; f1 = d2; nearest = c; } else if (d2 < f2) { f2 = d2; }
            }
        }
    }
    var val = 0.0;
    if (want_value) { val = hash01(nearest, seed + 0x51ED27u).x * 2.0 - 1.0; }
    return vec3f(sqrt(f1), sqrt(f2), val);
}
fn worley_legacy(x: vec3f, period: i32, seed: u32, feature: u32) -> f32 {
    let c = cells_legacy(x, period, seed, feature == 3u);
    switch feature {
        case 1u: { return clamp(c.y / 1.4 * 2.0 - 1.0, -1.0, 1.0); }
        case 2u: { return clamp((c.y - c.x) * 2.0 - 1.0, -1.0, 1.0); }
        case 3u: { return c.z; }
        default: { return clamp(c.x * 2.0 - 1.0, -1.0, 1.0); }
    }
}
";

    #[test]
    fn the_general_cells_loop_agrees_with_the_classic_one() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let c = codegen::compile(&worley(|p| p.jitter = 0.5)).unwrap();
        let pts: Vec<[f32; 4]> = points(20000, 19).into_iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        for component in ["x", "y", "z"] {
            let general = eval_expr(&gpu, &c, &format!("cells(q.xyz * 5.3, 0, 2u, true, 0u, 1.0).{component}"), &pts);
            let classic = eval_expr(&gpu, &c, &format!("cells_classic(q.xyz * 5.3, 0, 2u, true).{component}"), &pts);
            assert!(general.iter().zip(&classic).all(|(a, b)| a.to_bits() == b.to_bits()), "component {component} differs");
        }
    }

    #[test]
    fn the_default_worley_is_bit_for_bit_what_it_was() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let mut c = codegen::compile(&worley(|p| p.jitter = 0.5)).unwrap();
        c.wgsl = format!("{LEGACY_WORLEY}
{}", c.wgsl);
        let pts: Vec<[f32; 4]> = points(20000, 17).into_iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        for feature in 0..4u32 {
            for (period, scale) in [(0, 3.7), (4, 4.0), (6, 6.0)] {
                let new = eval_expr(&gpu, &c, &format!("worley(q.xyz * {scale:?}, {period}, 9u, {feature}u, 0u, 1.0)"), &pts);
                let old = eval_expr(&gpu, &c, &format!("worley_legacy(q.xyz * {scale:?}, {period}, 9u, {feature}u)"), &pts);
                assert!(new.iter().zip(&old).all(|(a, b)| a.to_bits() == b.to_bits()), "feature {feature}, period {period}: the default Worley changed");
            }
        }
    }
}
