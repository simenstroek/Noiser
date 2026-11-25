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

/// The settings of an Erosion node (Landscape graph). The grid it erodes on is the terrain's working resolution.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Erosion {
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
    /// Run again by itself when what feeds it changes (otherwise it keeps its last result, marked out of date, until
    /// Run is pressed).
    pub auto: bool,
    /// The preset these settings were loaded from or saved as (shown on the node; empty for none).
    pub preset: String,
}

impl Default for Erosion {
    fn default() -> Self {
        Erosion { iterations: 1000, rain: 0.04, capacity: 0.05, dissolve: 0.05, deposit: 0.05, evaporation: 0.05, thermal_iterations: 20, talus_deg: 35.0, auto: true, preset: String::new() }
    }
}

/// Erosion presets: `.erosion` files (JSON) in the library's Erosion Presets folder, or anywhere. A preset holds how the
/// erosion behaves: not whether the node runs by itself, nor the preset's own name (the file's name is that).
impl Erosion {
    /// Only what shapes the result (for presets, and for the key of a cached run).
    fn simulation(&self) -> Erosion {
        Erosion { auto: true, preset: String::new(), ..self.clone() }
    }

    pub fn to_preset(&self) -> String {
        let mut settings = serde_json::to_value(self.simulation()).unwrap_or_default();
        if let Some(map) = settings.as_object_mut() {
            map.remove("auto");
            map.remove("preset");
        }
        serde_json::to_string_pretty(&serde_json::json!({ "noiser_erosion": 1, "settings": settings })).unwrap_or_default()
    }

    /// These settings with a preset's applied (settings it leaves out take their defaults), named `name`; whether the
    /// node runs by itself stays as it was.
    pub fn with_preset(&self, text: &str, name: &str) -> Result<Erosion, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let settings = v.get("settings").filter(|_| v.get("noiser_erosion").is_some()).ok_or("not an erosion preset")?;
        let preset: Erosion = serde_json::from_value(settings.clone()).map_err(|e| e.to_string())?;
        Ok(Erosion { auto: self.auto, preset: name.to_string(), ..preset })
    }

    /// These settings back at their defaults (whether the node runs by itself stays).
    pub fn reset(&self) -> Erosion {
        Erosion { auto: self.auto, ..Erosion::default() }
    }

    /// The same erosion as `other`: the settings that shape the result are equal.
    pub fn same_as(&self, other: &Erosion) -> bool {
        self.simulation() == other.simulation()
    }

    /// A hash of the settings that shape the result.
    pub fn key(&self) -> String {
        serde_json::to_string(&self.simulation()).unwrap_or_default()
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn a_preset_round_trips_without_the_node_s_own_settings() {
        let mine = Erosion { iterations: 3000, rain: 0.1, talus_deg: 30.0, auto: false, preset: "old".into(), ..Default::default() };
        let text = mine.to_preset();
        assert!(!text.contains("auto") && !text.contains("preset"), "a preset holds only how the erosion behaves: {text}");
        let other = Erosion { auto: false, ..Default::default() };
        let loaded = other.with_preset(&text, "dunes").unwrap();
        assert!(loaded.same_as(&mine) && !loaded.auto && loaded.preset == "dunes");
        assert!(other.with_preset("{\"graph\": 1}", "x").is_err(), "a file that is not a preset is refused");
    }

    #[test]
    fn a_preset_missing_settings_gets_the_defaults_and_reset_keeps_auto() {
        let base = Erosion { iterations: 7, auto: false, ..Default::default() };
        let loaded = base.with_preset("{\"noiser_erosion\": 1, \"settings\": {\"rain\": 0.2, \"resolution\": 4096}}", "p").unwrap();
        assert_eq!((loaded.rain, loaded.iterations), (0.2, Erosion::default().iterations));
        let reset = base.reset();
        assert!(reset.same_as(&Erosion::default()) && !reset.auto);
    }
}
/// The Erosion node's settings: its preset (pick from the library, save, load, reset), whether it runs by itself, and
/// the simulation settings. Whether its result is up to date, and Run, are shown by the Landscape tab beside this.
pub fn node_ui(ui: &mut eframe::egui::Ui, e: &mut Erosion) {
    use eframe::egui;
    let d = Erosion::default();
    let before = e.clone();
    ui.horizontal(|ui| {
        let shown = if !e.preset.is_empty() { e.preset.clone() } else if e.same_as(&d) { "default".to_string() } else { "custom".to_string() };
        let mut pick = None;
        egui::ComboBox::from_id_salt("erosion preset").selected_text(shown).width(130.0).show_ui(ui, |ui| {
            // Read when the list opens, not every frame: the library may be on a network drive.
            let presets = library_presets();
            if presets.is_empty() {
                ui.weak(if crate::library::root().is_some() { "No presets in the library yet" } else { "No library folder set (Preferences)" });
            }
            for (name, path) in presets {
                if ui.selectable_label(false, name).clicked() {
                    pick = Some(path);
                }
            }
        }).response.on_hover_text("Erosion presets in the library's Erosion Presets folder");
        if crate::icons::button(ui, "open", "Load erosion preset", None, true).on_hover_text("Load an erosion preset file from anywhere").clicked() {
            pick = crate::library::dialog(crate::library::Kind::Erosion, rfd::FileDialog::new().add_filter("Erosion preset", &["erosion"])).pick_file();
        }
        if let Some(path) = pick {
            let name = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            if let Ok(loaded) = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|text| e.with_preset(&text, &name)) {
                *e = loaded;
            }
        }
        if crate::icons::button(ui, "save", "Save erosion preset", None, true).on_hover_text("Save these settings as a preset (in the library's Erosion Presets folder unless you pick another)").clicked() {
            if let Some(dir) = crate::library::dir(crate::library::Kind::Erosion) {
                let _ = std::fs::create_dir_all(dir);
            }
            let start = if e.preset.is_empty() { "erosion".to_string() } else { e.preset.clone() };
            let dialog = rfd::FileDialog::new().add_filter("Erosion preset", &["erosion"]).set_file_name(format!("{start}.erosion"));
            if let Some(path) = crate::library::dialog(crate::library::Kind::Erosion, dialog).save_file()
                && std::fs::write(&path, e.to_preset()).is_ok() {
                    e.preset = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
                }
        }
        if crate::icons::button(ui, "reset", "Reset erosion", None, !e.same_as(&d)).on_hover_text("Put every erosion setting back to its default").clicked() {
            *e = e.reset();
        }
    });
    ui.checkbox(&mut e.auto, "run by itself").on_hover_text("Run again when what feeds this node changes. Off: the last result stays (marked out of date) until you press Run.");
    crate::row(ui, &mut e.iterations, d.iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=20000).speed(10.0).prefix("iterations ")));
    crate::row(ui, &mut e.rain, d.rain, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("rain ").suffix(" m/s")));
    crate::row(ui, &mut e.capacity, d.capacity, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("capacity ")).on_hover_text("How much sediment fast water on a slope can carry: more means deeper channels."));
    crate::row(ui, &mut e.dissolve, d.dissolve, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("dissolve ")).on_hover_text("How quickly water picks sediment up."));
    crate::row(ui, &mut e.deposit, d.deposit, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("deposit ")).on_hover_text("How quickly water drops sediment."));
    crate::row(ui, &mut e.evaporation, d.evaporation, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("evaporation ")));
    crate::row(ui, &mut e.thermal_iterations, d.thermal_iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=5000).speed(1.0).prefix("thermal iterations ")));
    crate::row(ui, &mut e.talus_deg, d.talus_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=89.0).speed(0.5).prefix("repose ").suffix(" deg")));
    // Changed by hand: no longer the preset it was.
    if !e.same_as(&before) && e.preset == before.preset {
        e.preset.clear();
    }
    ui.weak("Outputs: the eroded height (in the input's units), then flow, deposition and wear (0 to 1), for layers, masks and Scatter.");
}

/// The presets in the library's folder, by name, sorted.
pub fn library_presets() -> Vec<(String, std::path::PathBuf)> {
    let Some(dir) = crate::library::dir(crate::library::Kind::Erosion) else { return Vec::new() };
    let mut out: Vec<(String, std::path::PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("erosion")))
        .map(|p| (p.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned()), p))
        .collect();
    out.sort_by_key(|(name, _)| name.to_lowercase());
    out
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

/// How much water passed through each cell: a pass of its own (its own bind group), so the hydraulic passes stay
/// within the default limit of 8 storage buffers.
const FLOW_SHADER: &str = "
struct P { n: u32, pad: u32, dx: f32, dt: f32, g: f32, rain: f32, kc: f32, ks: f32, kd: f32, ke: f32, min_tilt: f32, max_step: f32, talus_h: f32, rate: f32, damp: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> d: array<f32>;
@group(0) @binding(2) var<storage, read> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> flow: array<f32>;

@compute @workgroup_size(8, 8)
fn flow_pass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.n || id.y >= p.n) { return; }
    let i = id.y * p.n + id.x;
    flow[i] = flow[i] + d[i] * length(vel[i]) * p.dt;
}
";

/// What an erosion run gives: the eroded heights (metres) and the water that flowed through each cell over the run
/// (metres of depth times metres travelled; see `ErosionMaps` for the 0..1 maps).
pub struct ErosionResult {
    pub heights: Vec<f32>,
    pub flow: Vec<f32>,
}

/// Erodes `heights` (`n` x `n` metres, rows along y) on a grid whose cells are `cell_m` metres apart.
pub fn erode(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], n: usize, cell_m: f32, e: &Erosion, progress: &Progress) -> Result<ErosionResult, String> {
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
    // The flow pass: water depth and velocity read, flow added up.
    let flow = buffer(1);
    queue.write_buffer(&flow, 0, &vec![0u8; flow.size() as usize]);
    let flow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("erosion flow"), source: wgpu::ShaderSource::Wgsl(FLOW_SHADER.into()) });
    let read_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
    let flow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("erosion flow"), entries: &[entries[0], read_entry(1), read_entry(2), storage_entry(3)] });
    let flow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("erosion flow"), bind_group_layouts: &[Some(&flow_bgl)], immediate_size: 0 });
    let flow_pass = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("flow_pass"), layout: Some(&flow_layout), module: &flow_module, entry_point: Some("flow_pass"), compilation_options: Default::default(), cache: None });
    let g_flow = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion flow"),
        layout: &flow_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u_rain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: flow.as_entire_binding() },
        ],
    });
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
                    // Flow is counted while it rains (the settling steps would only add still water).
                    if it < hydraulic {
                        pass.set_bind_group(0, &g_flow, &[]);
                        pass.set_pipeline(&flow_pass);
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
    let read_back = |src: &wgpu::Buffer| -> Result<Vec<f32>, String> {
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cells * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &read, 0, cells * 4);
        queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map erosion result"));
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
        let view = read.get_mapped_range(..).map_err(|e| format!("{e:?}"))?;
        Ok(bytemuck::cast_slice(&view).to_vec())
    };
    Ok(ErosionResult { heights: read_back(&b)?, flow: read_back(&flow)? })
}

/// The 0..1 maps an erosion run leaves besides the height: where water ran (flow, on a log scale, since rivers carry
/// orders of magnitude more than slopes), where material was laid down (deposition) and where it was taken away
/// (wear). Deposition and wear come from comparing the heights before and after, so they cover thermal erosion too
/// (scree fans and their sources).
pub struct ErosionMaps {
    pub flow: Vec<f32>,
    pub deposition: Vec<f32>,
    pub wear: Vec<f32>,
}

impl ErosionMaps {
    pub fn new(before: &[f32], result: &ErosionResult) -> ErosionMaps {
        let change: Vec<f32> = result.heights.iter().zip(before).map(|(a, b)| a - b).collect();
        // Scaled by a high percentile rather than the maximum, so a few extreme cells don't make the rest of the map dark.
        let scale_by = |values: Vec<f32>| -> Vec<f32> {
            let top = percentile(&values, 0.99).max(1e-6);
            values.iter().map(|v| (v / top).clamp(0.0, 1.0)).collect()
        };
        let flow: Vec<f32> = result.flow.iter().map(|f| (1.0 + f.max(0.0) * 100.0).ln()).collect();
        ErosionMaps {
            flow: scale_by(flow),
            deposition: scale_by(change.iter().map(|c| c.max(0.0)).collect()),
            wear: scale_by(change.iter().map(|c| (-c).max(0.0)).collect()),
        }
    }
}

/// The value below which `fraction` of the positive values lie (0 when there are none).
fn percentile(values: &[f32], fraction: f32) -> f32 {
    let mut positive: Vec<f32> = values.iter().copied().filter(|v| *v > 0.0).collect();
    if positive.is_empty() {
        return 0.0;
    }
    let k = ((positive.len() - 1) as f32 * fraction) as usize;
    *positive.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}
