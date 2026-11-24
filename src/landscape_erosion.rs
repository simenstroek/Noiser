//! Hydraulic and thermal erosion of a heightfield on the GPU.
//!
//! Hydraulic: the pipe model of Mei, Decaudin and Hu, "Fast Hydraulic Erosion Simulation and Visualization on GPU"
//! (2007). Rain adds water; water flows between neighbouring cells through four pipes driven by the height difference;
//! the flow's speed and the slope give a carrying capacity; water erodes the ground while it carries less than that and
//! drops sediment while it carries more; sediment travels with the water; water evaporates. Heights are in metres, so
//! the settings behave the same at any resolution as long as the cell size is given.
//!
//! Thermal: material slides down to lower neighbours wherever the slope is steeper than the angle of repose.
//!
//! Each iteration is a few compute passes over storage buffers. A pass writes only its own cell, or buffers that its
//! neighbours do not read in the same pass, so there are no races.

use crate::export::Progress;
use eframe::wgpu;
use std::sync::atomic::Ordering;
use wgpu::util::DeviceExt;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Erosion {
    /// Working grid side in samples (the preview and the exports use this grid).
    pub resolution: usize,
    /// Hydraulic iterations (a fifth more follow with no rain so the sediment settles).
    pub iterations: u32,
    /// Rain in metres of water per second (each iteration is a fixed time step).
    pub rain: f32,
    /// Carrying capacity: how much sediment fast water on a slope can carry.
    pub capacity: f32,
    /// How quickly water picks sediment up, 0 to 1.
    pub dissolve: f32,
    /// How quickly water drops sediment, 0 to 1.
    pub deposit: f32,
    /// Evaporation per second.
    pub evaporation: f32,
    /// Thermal iterations, and the angle of repose in degrees.
    pub thermal_iterations: u32,
    pub talus_deg: f32,
}

impl Default for Erosion {
    fn default() -> Self {
        Erosion { resolution: 1024, iterations: 1000, rain: 0.04, capacity: 0.05, dissolve: 0.05, deposit: 0.05, evaporation: 0.05, thermal_iterations: 20, talus_deg: 35.0 }
    }
}

/// Resolutions the settings offer.
pub const RESOLUTIONS: [usize; 4] = [512, 1024, 2048, 4096];

/// The largest grid this GPU can hold: the biggest buffer is the flux, four floats a cell.
pub fn max_resolution(device: &wgpu::Device) -> usize {
    let limit = (device.limits().max_storage_buffer_binding_size as u64).min(device.limits().max_buffer_size);
    RESOLUTIONS.iter().rev().copied().find(|n| (*n as u64) * (*n as u64) * 16 <= limit).unwrap_or(512)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    pad: u32,
    dx: f32,
    dt: f32,
    g: f32,
    rain: f32,
    kc: f32,
    ks: f32,
    kd: f32,
    ke: f32,
    min_tilt: f32,
    max_step: f32,
    talus_h: f32,
    rate: f32,
    damp: f32,
    pad2: f32,
}

const SHADER: &str = "
struct P { n: u32, pad: u32, dx: f32, dt: f32, g: f32, rain: f32, kc: f32, ks: f32, kd: f32, ke: f32, min_tilt: f32, max_step: f32, talus_h: f32, rate: f32, damp: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_mid: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<storage, read_write> flux: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> vel: array<vec2f>;

fn cell(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.n) - 1)) * p.n + u32(clamp(x, 0, i32(p.n) - 1));
}

// The four outflows of each cell from the height differences of ground plus water.
@compute @workgroup_size(8, 8)
fn flux_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let h = b[i] + d[i];
    let f = flux[i];
    let k = p.dt * p.g * p.dx;
    let l = max(0.0, f.x * p.damp + k * (h - (b[cell(x - 1, y)] + d[cell(x - 1, y)])));
    let r = max(0.0, f.y * p.damp + k * (h - (b[cell(x + 1, y)] + d[cell(x + 1, y)])));
    let t = max(0.0, f.z * p.damp + k * (h - (b[cell(x, y - 1)] + d[cell(x, y - 1)])));
    let u = max(0.0, f.w * p.damp + k * (h - (b[cell(x, y + 1)] + d[cell(x, y + 1)])));
    let total = l + r + t + u;
    // A cell cannot send out more water than it has.
    let scale = select(1.0, min(1.0, d[i] * p.dx * p.dx / (total * p.dt)), total > 0.0);
    flux[i] = vec4f(l, r, t, u) * scale;
}

// New water depth from the flows in and out, and the water's velocity.
@compute @workgroup_size(8, 8)
fn water_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let f = flux[i];
    let from_l = flux[cell(x - 1, y)].y; let from_r = flux[cell(x + 1, y)].x;
    let from_t = flux[cell(x, y - 1)].w; let from_b = flux[cell(x, y + 1)].z;
    // At the border the neighbour is the cell itself: nothing flows in from outside.
    let edge_l = x == 0; let edge_r = x == i32(p.n) - 1; let edge_t = y == 0; let edge_b = y == i32(p.n) - 1;
    let in_l = select(from_l, 0.0, edge_l); let in_r = select(from_r, 0.0, edge_r);
    let in_t = select(from_t, 0.0, edge_t); let in_b = select(from_b, 0.0, edge_b);
    let out_l = select(f.x, 0.0, edge_l); let out_r = select(f.y, 0.0, edge_r);
    let out_t = select(f.z, 0.0, edge_t); let out_b = select(f.w, 0.0, edge_b);
    let old = d[i];
    let now = max(0.0, old + p.dt * (in_l + in_r + in_t + in_b - out_l - out_r - out_t - out_b) / (p.dx * p.dx));
    d[i] = now;
    let mean = max(0.5 * (old + now), 0.02);
    var v = vec2f(0.5 * (in_l - out_l + out_r - in_r), 0.5 * (in_t - out_t + out_b - in_b)) / (p.dx * mean);
    // Water never moves a whole cell or more in one step (the sediment would skip cells and be lost), nor faster than a torrent.
    let vmax = min(0.9 * p.dx / p.dt, 5.0);
    let speed = length(v);
    if (speed > vmax) { v = v * (vmax / speed); }
    vel[i] = v;
}

// Erode or deposit by comparing the carrying capacity with the sediment the water holds.
@compute @workgroup_size(8, 8)
fn erode_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    // A Sobel gradient: the plain central difference cannot see a height pattern that alternates cell by cell.
    let gx = ((b[cell(x + 1, y - 1)] + 2.0 * b[cell(x + 1, y)] + b[cell(x + 1, y + 1)]) - (b[cell(x - 1, y - 1)] + 2.0 * b[cell(x - 1, y)] + b[cell(x - 1, y + 1)])) / (8.0 * p.dx);
    let gy = ((b[cell(x - 1, y + 1)] + 2.0 * b[cell(x, y + 1)] + b[cell(x + 1, y + 1)]) - (b[cell(x - 1, y - 1)] + 2.0 * b[cell(x, y - 1)] + b[cell(x + 1, y - 1)])) / (8.0 * p.dx);
    let tilt = max(length(vec2f(gx, gy)) / sqrt(1.0 + gx * gx + gy * gy), p.min_tilt);
    // Little water, little erosion: the capacity fades out below a few centimetres of depth.
    let wet = clamp(d[i] * 20.0, 0.0, 1.0);
    let capacity = p.kc * tilt * length(vel[i]) * wet;
    let held = s[i];
    if (capacity > held) {
        let amount = min(p.ks * (capacity - held), p.max_step);
        db[i] = -amount;
        s_mid[i] = held + amount;
    } else {
        let amount = p.kd * (held - capacity);
        db[i] = amount;
        s_mid[i] = held - amount;
    }
}

// Sediment moves with the water (looked up where the water came from), the ground takes the change, water evaporates
// and rain falls (here, not in the flux pass: that pass reads its neighbours' water).
@compute @workgroup_size(8, 8)
fn transport_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let x = i32(id.x); let y = i32(id.y);
    let i = cell(x, y);
    let src = vec2f(f32(x), f32(y)) - vel[i] * p.dt / p.dx;
    let fx = clamp(src.x, 0.0, f32(p.n) - 1.001); let fy = clamp(src.y, 0.0, f32(p.n) - 1.001);
    let x0 = i32(fx); let y0 = i32(fy);
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    let top = mix(s_mid[cell(x0, y0)], s_mid[cell(x0 + 1, y0)], ax);
    let bottom = mix(s_mid[cell(x0, y0 + 1)], s_mid[cell(x0 + 1, y0 + 1)], ax);
    s[i] = mix(top, bottom, ay);
    b[i] = b[i] + db[i];
    d[i] = d[i] * (1.0 - p.ke * p.dt) + p.rain * p.dt;
}

// Thermal erosion: what each cell gives its lower neighbours and takes from its higher ones, in one go.
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

// Whatever sediment is still in the water goes back onto the ground.
@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let i = cell(i32(id.x), i32(id.y));
    b[i] = b[i] + s[i];
    s[i] = 0.0;
}
";

/// Erodes `heights` (`n` x `n` metres, rows along y) on a grid whose cells are `cell_m` metres apart.
pub fn erode(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], n: usize, cell_m: f32, e: &Erosion, progress: &Progress) -> Result<Vec<f32>, String> {
    if heights.len() != n * n {
        return Err("the grid has the wrong size".into());
    }
    if n > max_resolution(device) {
        return Err(format!("{n} x {n} needs buffers larger than this GPU allows; use a smaller working resolution"));
    }
    let cells = (n * n) as u64;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let buffer = |per_cell: u64| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cells * per_cell * 4, usage: storage, mapped_at_creation: false });
    let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(heights), usage: storage });
    let (d, s, s_mid, db, flux, vel) = (buffer(1), buffer(1), buffer(1), buffer(1), buffer(4), buffer(2));
    for buf in [&d, &s, &s_mid, &db, &flux, &vel] {
        queue.write_buffer(buf, 0, &vec![0u8; (buf.size()) as usize]);
    }
    let base = Params {
        n: n as u32,
        pad: 0,
        dx: cell_m,
        dt: 0.1,
        g: 9.81,
        rain: e.rain,
        kc: e.capacity,
        ks: e.dissolve,
        kd: e.deposit,
        ke: e.evaporation,
        min_tilt: 0.05,
        max_step: 0.2,
        talus_h: e.talus_deg.to_radians().tan() * cell_m,
        rate: 1.0,
        damp: 0.995,
        pad2: 0.0,
    };
    // The last fifth of the iterations has no rain and evaporates faster, so the sediment settles.
    let settle = Params { rain: 0.0, ke: e.evaporation * 4.0, ..base };
    let uniform = |p: &Params| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(p), usage: wgpu::BufferUsages::UNIFORM });
    let (u_rain, u_settle) = (uniform(&base), uniform(&settle));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("erosion"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let storage_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
    let mut entries = vec![wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }];
    entries.extend((1..8).map(storage_entry));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
    let (flux_pass, water_pass, erode_pass, transport_pass, thermal_calc, thermal_apply, finish) = (pipeline("flux_pass"), pipeline("water_pass"), pipeline("erode_pass"), pipeline("transport_pass"), pipeline("thermal_calc"), pipeline("thermal_apply"), pipeline("finish"));
    let bind = |u: &wgpu::Buffer| {
        let bufs = [u, &b, &d, &s, &s_mid, &db, &flux, &vel];
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(k, buf)| wgpu::BindGroupEntry { binding: k as u32, resource: buf.as_entire_binding() }).collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &entries })
    };
    let (g_rain, g_settle) = (bind(&u_rain), bind(&u_settle));
    let groups = (n as u32).div_ceil(8);
    let hydraulic = e.iterations;
    let settling = e.iterations / 5;
    let total = hydraulic + settling + e.thermal_iterations;
    let mut done = 0u32;
    while done < total {
        let batch = (total - done).min(20);
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for it in done..done + batch {
                let thermal = it >= hydraulic + settling;
                if thermal {
                    pass.set_bind_group(0, &g_rain, &[]);
                    for pipe in [&thermal_calc, &thermal_apply] {
                        pass.set_pipeline(pipe);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                } else {
                    pass.set_bind_group(0, if it < hydraulic { &g_rain } else { &g_settle }, &[]);
                    for pipe in [&flux_pass, &water_pass, &erode_pass, &transport_pass] {
                        pass.set_pipeline(pipe);
                        pass.dispatch_workgroups(groups, groups, 1);
                    }
                }
                // The sediment goes back on the ground when the water part is over.
                if it + 1 == hydraulic + settling {
                    pass.set_bind_group(0, &g_rain, &[]);
                    pass.set_pipeline(&finish);
                    pass.dispatch_workgroups(groups, groups, 1);
                }
            }
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
        done += batch;
        progress.store((done * 1000 / total.max(1)) as u32, Ordering::Relaxed);
    }
    let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cells * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&b, 0, &read, 0, cells * 4);
    queue.submit([enc.finish()]);
    read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map erosion result"));
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
    let view = read.get_mapped_range(..).map_err(|e| format!("{e:?}"))?;
    Ok(bytemuck::cast_slice(&view).to_vec())
}
