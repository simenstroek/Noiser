//! Wind erosion on the GPU: the Erosion node's second kind. A saltation model on the height grid. Each iteration, every
//! cell that the wind reaches lifts a little material (most on the windward face, a little from flat ground); the
//! material hops downwind by a fixed distance; where the wind is blocked (in the lee of higher ground: the shadow of an
//! upwind ridge at the shadow angle) and on falling ground it settles. Dunes build up behind crests, windward faces
//! wear, and a few thermal steps keep slip faces at the angle of repose. Everything that is still in the air at the end
//! settles where it is. Own shader and bind group, so it does not touch the hydraulic passes' binding budget.
//! See docs/terrain-maps-plan.md.

use crate::export::Progress;
use crate::landscape_erosion::{Erosion, ErosionResult};
use eframe::wgpu;
use std::sync::atomic::Ordering;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    reach_steps: u32,
    dx: f32,
    hop_cells: f32,
    dir_x: f32,
    dir_y: f32,
    lift: f32,
    deposit_shadow: f32,
    deposit_open: f32,
    shadow_slope: f32,
    max_step: f32,
    talus_h: f32,
    rate: f32,
    step_cells: f32,
    pad: [f32; 2],
}

const SHADER: &str = "
struct P { n: u32, reach_steps: u32, dx: f32, hop_cells: f32, dir_x: f32, dir_y: f32, lift: f32, deposit_shadow: f32, deposit_open: f32, shadow_slope: f32, max_step: f32, talus_h: f32, rate: f32, step_cells: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_mid: array<f32>;
@group(0) @binding(4) var<storage, read_write> db: array<f32>;
@group(0) @binding(5) var<storage, read_write> carried: array<f32>;

fn cell(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.n) - 1)) * p.n + u32(clamp(x, 0, i32(p.n) - 1));
}

fn bilinear_b(q: vec2f) -> f32 {
    let f = clamp(q, vec2f(0.0), vec2f(f32(p.n) - 1.001));
    let i = vec2i(f);
    let a = f - vec2f(i);
    let top = mix(b[cell(i.x, i.y)], b[cell(i.x + 1, i.y)], a.x);
    let bottom = mix(b[cell(i.x, i.y + 1)], b[cell(i.x + 1, i.y + 1)], a.x);
    return mix(top, bottom, a.y);
}

// What lifts, what settles, from the ground and the wind that reaches it.
@compute @workgroup_size(8, 8)
fn wind_calc(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let h = b[i];
    let dir = vec2f(p.dir_x, p.dir_y);
    let here = vec2f(f32(x), f32(y));
    // In the lee of higher ground? March upwind: a point further up the wind shades this one when it stands higher than
    // the shadow line, which falls off at the shadow angle.
    var shade = 0.0;
    for (var k = 1u; k <= p.reach_steps; k = k + 1u) {
        let dist_cells = f32(k) * p.step_cells;
        let up = bilinear_b(here - dir * dist_cells);
        let depth = (up - p.shadow_slope * dist_cells * p.dx) - h;
        shade = max(shade, clamp(depth / (2.0 * p.dx), 0.0, 1.0));
    }
    let gx = (b[cell(x + 1, y)] - b[cell(x - 1, y)]) / (2.0 * p.dx);
    let gy = (b[cell(x, y + 1)] - b[cell(x, y - 1)]) / (2.0 * p.dx);
    // Positive where the ground rises downwind (the windward face), negative on the lee face.
    let along = gx * dir.x + gy * dir.y;
    let exposure = 1.0 - shade;
    let lift = min(exposure * p.lift * clamp(0.4 + 2.0 * along, 0.0, 1.6), p.max_step);
    let held = c[i] + lift;
    let rate = clamp(mix(p.deposit_open, p.deposit_shadow, shade) + clamp(-along, 0.0, 1.0) * 0.4, 0.0, 1.0);
    let dep = held * rate;
    db[i] = dep - lift;
    c_mid[i] = held - dep;
}

// The airborne material hops downwind (gathered from where it came from); the ground takes the change.
@compute @workgroup_size(8, 8)
fn wind_move(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let src = vec2f(f32(x), f32(y)) - vec2f(p.dir_x, p.dir_y) * p.hop_cells;
    var value = 0.0;
    // Nothing blows in from beyond the upwind border.
    if (src.x >= 0.0 && src.y >= 0.0 && src.x <= f32(p.n) - 1.001 && src.y <= f32(p.n) - 1.001) {
        let q = vec2i(src);
        let a = src - vec2f(q);
        let top = mix(c_mid[cell(q.x, q.y)], c_mid[cell(q.x + 1, q.y)], a.x);
        let bottom = mix(c_mid[cell(q.x, q.y + 1)], c_mid[cell(q.x + 1, q.y + 1)], a.x);
        value = mix(top, bottom, a.y);
    }
    c[i] = value;
    carried[i] = carried[i] + value;
    b[i] = b[i] + db[i];
}

fn give(from_h: f32, to_h: f32) -> f32 {
    return max(0.0, from_h - to_h - p.talus_h) * p.rate * 0.25;
}

@compute @workgroup_size(8, 8)
fn thermal_calc(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let h = b[i];
    var delta = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let dx = select(0, select(-1, 1, k == 1), k < 2);
        let dy = select(0, select(-1, 1, k == 3), k >= 2);
        let j = cell(x + dx, y + dy);
        if (j == i) { continue; }
        delta = delta - give(h, b[j]) + give(b[j], h);
    }
    db[i] = delta;
}

@compute @workgroup_size(8, 8)
fn thermal_apply(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let i = cell(i32(id.x), i32(id.y));
    b[i] = b[i] + db[i];
}

// Whatever is still in the air settles where it is.
@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let i = cell(i32(id.x), i32(id.y));
    b[i] = b[i] + c[i];
    c[i] = 0.0;
}
";

/// Erodes `heights` (`n` x `n` metres, rows along y) with wind on a grid whose cells are `cell_m` metres apart. The
/// result's `flow` is the material carried through each cell over the run.
pub fn erode_wind(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], n: usize, cell_m: f32, e: &Erosion, progress: &Progress) -> Result<ErosionResult, String> {
    if heights.len() != n * n {
        return Err("the grid has the wrong size".into());
    }
    if n > crate::landscape_erosion::max_resolution(device) {
        return Err(format!("{n} x {n} needs buffers larger than this GPU allows; use a smaller working resolution"));
    }
    let cells = (n * n) as u64;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let zeroed = || {
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cells * 4, usage: storage, mapped_at_creation: false });
        queue.write_buffer(&buf, 0, &vec![0u8; (cells * 4) as usize]);
        buf
    };
    let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(heights), usage: storage });
    let (c, c_mid, db, carried) = (zeroed(), zeroed(), zeroed(), zeroed());
    let (sin, cos) = e.wind_deg.to_radians().sin_cos();
    let hop_cells = (e.wind_hop_m / cell_m).clamp(1.0, 64.0);
    // The shadow march covers about three hops upwind, in at most 32 steps.
    let reach_cells = hop_cells * 3.0;
    let reach_steps = (reach_cells.ceil() as u32).clamp(1, 32);
    let params = Params {
        n: n as u32,
        reach_steps,
        dx: cell_m,
        hop_cells,
        dir_x: cos,
        dir_y: sin,
        lift: e.wind_lift,
        deposit_shadow: 0.7,
        deposit_open: e.wind_deposit,
        shadow_slope: e.wind_shadow_deg.to_radians().tan(),
        max_step: e.wind_lift * 2.0,
        talus_h: e.talus_deg.to_radians().tan() * cell_m,
        rate: 1.0,
        step_cells: reach_cells / reach_steps as f32,
        pad: [0.0; 2],
    };
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("wind erosion"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let storage_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
    let mut entries = vec![wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }];
    entries.extend((1..6).map(storage_entry));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("wind erosion"), entries: &entries });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
    let (wind_calc, wind_move, thermal_calc, thermal_apply, finish) = (pipeline("wind_calc"), pipeline("wind_move"), pipeline("thermal_calc"), pipeline("thermal_apply"), pipeline("finish"));
    let bufs = [&uniform, &b, &c, &c_mid, &db, &carried];
    let group_entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(k, buf)| wgpu::BindGroupEntry { binding: k as u32, resource: buf.as_entire_binding() }).collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &group_entries });
    let groups = (n as u32).div_ceil(8);
    // The last fifth of the iterations has no new lift... (they settle what is airborne), then the thermal steps.
    let wind_its = e.wind_iterations;
    let total = wind_its + e.thermal_iterations;
    let mut done = 0u32;
    while done < total {
        let batch = (total - done).min(20);
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &group, &[]);
            for it in done..done + batch {
                if it < wind_its {
                    for pipe in [&wind_calc, &wind_move] {
                        pass.set_pipeline(pipe);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                    if it + 1 == wind_its {
                        pass.set_pipeline(&finish);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                } else {
                    for pipe in [&thermal_calc, &thermal_apply] {
                        pass.set_pipeline(pipe);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                }
            }
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
        done += batch;
        progress.store((done * 1000 / total.max(1)) as u32, Ordering::Relaxed);
    }
    let read_back = |src: &wgpu::Buffer| -> Result<Vec<f32>, String> {
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cells * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &read, 0, cells * 4);
        queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map wind erosion result"));
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
        let view = read.get_mapped_range(..).map_err(|e| format!("{e:?}"))?;
        Ok(bytemuck::cast_slice(&view).to_vec())
    };
    // The airborne material per cell is sand in metres of depth per hop; scale it like the water flow (depth x distance).
    let flow = read_back(&carried)?.into_iter().map(|v| v * cell_m).collect();
    Ok(ErosionResult { heights: read_back(&b)?, flow })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_tests::gpu;

    fn small() -> Erosion {
        Erosion { kind: crate::landscape_erosion::ErosionKind::Wind, wind_iterations: 60, thermal_iterations: 5, ..Default::default() }
    }

    #[test]
    fn wind_moves_material_downwind_and_keeps_most_of_it() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let n = 128;
        // A flat plain with one round hill in the middle.
        let hill: Vec<f32> = (0..n * n)
            .map(|k| {
                let (x, y) = ((k % n) as f32 - 64.0, (k / n) as f32 - 64.0);
                20.0 * (-(x * x + y * y) / 200.0).exp()
            })
            .collect();
        let e = Erosion { wind_deg: 0.0, ..small() };
        let r = erode_wind(&gpu.device, &gpu.queue, &hill, n, 4.0, &e, &Default::default()).unwrap();
        assert!(r.heights.iter().all(|v| v.is_finite()));
        let sum = |v: &[f32]| v.iter().sum::<f32>();
        let kept = sum(&r.heights) / sum(&hill);
        assert!(kept > 0.6, "most material stays on the terrain (the rest blows off the downwind edge): kept {kept}");
        // The hill's windward half (x < 64) loses, the lee side (x > 64) gains.
        let half = |h: &[f32], east: bool| -> f32 { (0..n * n).filter(|k| ((k % n) > 64) == east).map(|k| h[k]).sum() };
        assert!(half(&r.heights, false) < half(&hill, false), "windward side wears");
        assert!(half(&r.heights, true) > half(&hill, true) * 0.98 - 1.0, "the lee side does not lose as much as the windward side");
        let moved: f32 = r.heights.iter().zip(&hill).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved > 10.0, "something moved: {moved}");
        // The same input gives the same result.
        let again = erode_wind(&gpu.device, &gpu.queue, &hill, n, 4.0, &e, &Default::default()).unwrap();
        assert_eq!(r.heights, again.heights);
        // Reversing the wind mirrors where material goes.
        let west = erode_wind(&gpu.device, &gpu.queue, &hill, n, 4.0, &Erosion { wind_deg: 180.0, ..e.clone() }, &Default::default()).unwrap();
        assert!(half(&west.heights, true) < half(&hill, true), "with the wind from the other side the east wears");
    }
}
