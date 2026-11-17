//! Tests that run the generated WGSL on a real GPU: tiling seams, continuity and output range
//! for every noise kind. They skip (pass with a note) when no GPU adapter is available.

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::{Graph, NodeKind, NoiseKind, NoiseParams, WorleyFeature};
use eframe::wgpu;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(Gpu { device, queue })
}

/// Evaluates the graph's density at each point.
fn eval(gpu: &Gpu, graph: &Graph, points: &[[f32; 3]]) -> Vec<f32> {
    use wgpu::util::DeviceExt;
    let c = codegen::compile(graph).unwrap();
    let src = format!(
        "{NOISE_LIB}\n{}\n{EVAL_BINDINGS}
@group(0) @binding(4) var<storage, read> pts: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x < arrayLength(&out)) {{ out[id.x] = graph_density(pts[id.x].xyz); }}
}}",
        c.wgsl
    );
    let d = &gpu.device;
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init = |contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage });
    let uniforms = init(bytemuck::bytes_of(&[0u32; 4]), wgpu::BufferUsages::UNIFORM);
    let params = init(bytemuck::cast_slice(&c.params), wgpu::BufferUsages::STORAGE);
    let blue = init(bytemuck::cast_slice(&crate::bluenoise::tables()), wgpu::BufferUsages::STORAGE);
    let pts: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    let pts = init(bytemuck::cast_slice(&pts), wgpu::BufferUsages::STORAGE);
    let size = (points.len() * 4) as u64;
    let out = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let read = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
    }
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
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
        let continuous = matches!(kind, NoiseKind::Perlin | NoiseKind::Simplex | NoiseKind::Worley);
        let features: &[WorleyFeature] =
            if kind == NoiseKind::Worley { &[WorleyFeature::F1, WorleyFeature::F2, WorleyFeature::F2MinusF1] } else { &[WorleyFeature::F1] };
        for &feature in features {
            let p = NoiseParams { kind, feature, scale, octaves: 3, ..Default::default() };
            v.push((format!("{} {feature:?}", kind.name()), p, continuous));
        }
    }
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
