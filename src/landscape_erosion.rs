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

/// Which kind of erosion a node does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ErosionKind {
    /// Rain, flowing water and the sediment it carries, then thermal slides (`landscape_erosion`).
    #[default]
    Water,
    /// Saltation by wind: windward faces wear, dunes build in the lee (`landscape_wind`).
    Wind,
    /// Learned detail of real terrain, added to a coarser copy of the input (`landscape_detail`); shown as its own node,
    /// Learned Detail.
    Detail,
    /// A Gaussian blur: softens hard edges (before Learned Detail, say); shown as its own node, Blur.
    Blur,
}

/// One stage of Learned Detail.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DetailStage {
    pub on: bool,
    /// Sampling steps (more: slower; 8 looked the same as 32 in tests).
    pub steps: u32,
    /// How strongly the chosen terrain type is pushed away from an average one (1: the model as trained; more: the
    /// types differ more and the detail is stronger; each step then costs twice).
    pub guidance: f32,
    /// How much of this stage's added detail is kept: 0 none, 1 all, above 1 strengthened.
    pub amount: f32,
    /// How closely the input's shape is kept: 1 exactly; lower lets the model reshape it (landforms stage only).
    pub adherence: f32,
}

impl Default for DetailStage {
    fn default() -> Self {
        DetailStage { on: true, steps: 8, guidance: 2.0, amount: 1.0, adherence: 1.0 }
    }
}

/// The settings of an Erosion node (Landscape graph). The grid it erodes on is the terrain's working resolution.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Erosion {
    pub kind: ErosionKind,
    /// Wind: the direction it blows towards, degrees counter-clockwise from the +x side (east) of the terrain.
    pub wind_deg: f32,
    /// Wind iterations (each lifts, carries one hop and settles); thermal iterations follow.
    pub wind_iterations: u32,
    /// Wind: metres lifted per iteration from ground the wind reaches.
    pub wind_lift: f32,
    /// Wind: how far material hops each iteration, in metres.
    pub wind_hop_m: f32,
    /// Wind: the angle of the shadow behind higher ground, in degrees: material settles in it.
    pub wind_shadow_deg: f32,
    /// Wind: the share of airborne material that settles per hop on open ground, 0 to 1.
    pub wind_deposit: f32,
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
    /// Detail: the kind of real terrain whose detail is learned (one of `landscape_detail::TYPES`).
    pub detail_type: String,
    /// Detail: the stages, coarse to fine: 16x to 4x (landforms), 4x to the grid (fine detail), sharpen (the finest
    /// grain, on the grid itself).
    pub detail_stages: [DetailStage; 3],
    /// Blur: the Gaussian's radius (standard deviation) in metres.
    pub blur_m: f32,
    /// Run again by itself when what feeds it changes (otherwise it keeps its last result, marked out of date, until
    /// Run is pressed).
    pub auto: bool,
    /// The preset these settings were loaded from or saved as (shown on the node; empty for none).
    pub preset: String,
}

impl Default for Erosion {
    fn default() -> Self {
        Erosion { kind: ErosionKind::Water, wind_deg: 0.0, wind_iterations: 150, wind_lift: 0.04, wind_hop_m: 24.0, wind_shadow_deg: 12.0, wind_deposit: 0.15, iterations: 1000, rain: 0.04, capacity: 0.05, dissolve: 0.05, deposit: 0.05, evaporation: 0.05, thermal_iterations: 20, talus_deg: 35.0, detail_type: "hills".into(), detail_stages: Default::default(), blur_m: 10.0, auto: true, preset: String::new() }
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
    fn blur_softens_a_step_and_keeps_flat_ground() {
        let Some(gpu) = crate::gpu_tests::gpu() else { return eprintln!("no GPU adapter; skipping") };
        let n = 64;
        let step: Vec<f32> = (0..n * n).map(|k| if k % n < n / 2 { 0.0 } else { 10.0 }).collect();
        let e = Erosion { kind: ErosionKind::Blur, blur_m: 4.0, ..Default::default() };
        let r = erode(&gpu.device, &gpu.queue, &step, n, 1.0, &e, &Default::default()).unwrap();
        let row = &r.heights[10 * n..11 * n];
        assert!((row[0] - 0.0).abs() < 1e-4 && (row[n - 1] - 10.0).abs() < 1e-4, "far from the step stays put");
        assert!(row[n / 2 - 2] > 1.0 && row[n / 2 + 1] < 9.0, "the step is softened: {:?}", &row[n / 2 - 3..n / 2 + 3]);
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
    if e.kind == ErosionKind::Detail {
        return detail_ui(ui, e);
    }
    if e.kind == ErosionKind::Blur {
        return blur_ui(ui, e);
    }
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
    crate::row(ui, &mut e.kind, d.kind, |ui, v| {
        egui::ComboBox::from_id_salt("erosion kind").selected_text(match v { ErosionKind::Water => "Water", ErosionKind::Wind => "Wind", ErosionKind::Detail => "Learned Detail", ErosionKind::Blur => "Blur" }).show_ui(ui, |ui| {
            ui.selectable_value(v, ErosionKind::Water, "Water: rain, rivers and slides");
            ui.selectable_value(v, ErosionKind::Wind, "Wind: dunes and wear");
        });
    });
    ui.checkbox(&mut e.auto, "run by itself").on_hover_text("Run again when what feeds this node changes. Off: the last result stays (marked out of date) until you press Run.");
    if e.kind == ErosionKind::Wind {
        crate::row(ui, &mut e.wind_deg, d.wind_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(1.0).prefix("blows towards ").suffix("°")).on_hover_text("Counter-clockwise from east (the +x side of the terrain): 90 blows towards north."));
        crate::row(ui, &mut e.wind_iterations, d.wind_iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=20000).speed(5.0).prefix("iterations ")));
        crate::row(ui, &mut e.wind_lift, d.wind_lift, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("lift ").suffix(" m")).on_hover_text("How much ground the wind picks up per iteration where it reaches it."));
        crate::row(ui, &mut e.wind_hop_m, d.wind_hop_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=500.0).speed(0.5).prefix("hop ").suffix(" m")).on_hover_text("How far the material travels per iteration: longer hops make longer dunes."));
        crate::row(ui, &mut e.wind_shadow_deg, d.wind_shadow_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=60.0).speed(0.2).prefix("shadow ").suffix("°")).on_hover_text("The angle of the sheltered zone behind higher ground, where material settles."));
        crate::row(ui, &mut e.wind_deposit, d.wind_deposit, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("settling ")).on_hover_text("The share of airborne material that settles per hop on open ground."));
        crate::row(ui, &mut e.thermal_iterations, d.thermal_iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=5000).speed(1.0).prefix("slide iterations ")).on_hover_text("Steps that let slopes steeper than the angle of repose slip."));
        crate::row(ui, &mut e.talus_deg, d.talus_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=89.0).speed(0.5).prefix("repose ").suffix(" deg")));
    } else {
    crate::row(ui, &mut e.iterations, d.iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=20000).speed(10.0).prefix("iterations ")));
    crate::row(ui, &mut e.rain, d.rain, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("rain ").suffix(" m/s")));
    crate::row(ui, &mut e.capacity, d.capacity, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("capacity ")).on_hover_text("How much sediment fast water on a slope can carry: more means deeper channels."));
    crate::row(ui, &mut e.dissolve, d.dissolve, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("dissolve ")).on_hover_text("How quickly water picks sediment up."));
    crate::row(ui, &mut e.deposit, d.deposit, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("deposit ")).on_hover_text("How quickly water drops sediment."));
    crate::row(ui, &mut e.evaporation, d.evaporation, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("evaporation ")));
    crate::row(ui, &mut e.thermal_iterations, d.thermal_iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=5000).speed(1.0).prefix("thermal iterations ")));
    crate::row(ui, &mut e.talus_deg, d.talus_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=89.0).speed(0.5).prefix("repose ").suffix(" deg")));
    }
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
// The diagonal pipes: outflows towards (-1,-1), (+1,-1), (-1,+1) and (+1,+1). With them water spreads about equally in
// every direction: four pipes alone cut straight, axis-aligned gullies.
@group(0) @binding(8) var<storage, read_write> flux2: array<vec4f>;

const DIAG: f32 = 0.70710678;

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
    let k2 = k * DIAG;
    let f2 = flux2[i];
    let tl = max(0.0, f2.x * p.damp + k2 * (h - (b[cell(x - 1, y - 1)] + d[cell(x - 1, y - 1)])));
    let tr = max(0.0, f2.y * p.damp + k2 * (h - (b[cell(x + 1, y - 1)] + d[cell(x + 1, y - 1)])));
    let bl = max(0.0, f2.z * p.damp + k2 * (h - (b[cell(x - 1, y + 1)] + d[cell(x - 1, y + 1)])));
    let br = max(0.0, f2.w * p.damp + k2 * (h - (b[cell(x + 1, y + 1)] + d[cell(x + 1, y + 1)])));
    let total = l + r + t + u + tl + tr + bl + br;
    // A cell cannot send out more water than it has.
    let scale = select(1.0, min(1.0, d[i] * p.dx * p.dx / (total * p.dt)), total > 0.0);
    flux[i] = vec4f(l, r, t, u) * scale;
    flux2[i] = vec4f(tl, tr, bl, br) * scale;
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
    // The diagonals: out of the cell towards a corner that exists, and in from the corner cells.
    let ok_l = x > 0; let ok_r = x < i32(p.n) - 1; let ok_t = y > 0; let ok_b = y < i32(p.n) - 1;
    let g2 = flux2[i];
    let o_tl = select(0.0, g2.x, ok_l && ok_t); let o_tr = select(0.0, g2.y, ok_r && ok_t);
    let o_bl = select(0.0, g2.z, ok_l && ok_b); let o_br = select(0.0, g2.w, ok_r && ok_b);
    let i_tl = select(0.0, flux2[cell(x - 1, y - 1)].w, ok_l && ok_t); let i_tr = select(0.0, flux2[cell(x + 1, y - 1)].z, ok_r && ok_t);
    let i_bl = select(0.0, flux2[cell(x - 1, y + 1)].y, ok_l && ok_b); let i_br = select(0.0, flux2[cell(x + 1, y + 1)].x, ok_r && ok_b);
    let now = max(0.0, old + p.dt * (in_l + in_r + in_t + in_b - out_l - out_r - out_t - out_b + i_tl + i_tr + i_bl + i_br - o_tl - o_tr - o_bl - o_br) / (p.dx * p.dx));
    d[i] = now;
    let mean = max(0.5 * (old + now), 0.02);
    let vx = 0.5 * (in_l - out_l + out_r - in_r) + 0.5 * DIAG * (o_tr + o_br - o_tl - o_bl + i_tl + i_bl - i_tr - i_br);
    let vy = 0.5 * (in_t - out_t + out_b - in_b) + 0.5 * DIAG * (o_bl + o_br - o_tl - o_tr + i_tl + i_tr - i_bl - i_br);
    var v = vec2f(vx, vy) / (p.dx * mean);
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

/// The Blur node's settings: the radius, and whether it runs by itself.
fn blur_ui(ui: &mut eframe::egui::Ui, e: &mut Erosion) {
    use eframe::egui;
    let d = Erosion::default();
    ui.checkbox(&mut e.auto, "run by itself").on_hover_text("Run again when what feeds this node changes. Off: the last result stays (marked out of date) until you press Run.");
    crate::row(ui, &mut e.blur_m, d.blur_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=100.0).speed(0.2).prefix("radius ").suffix(" m")).on_hover_text("The Gaussian's radius (standard deviation) in metres: hard edges and steps soften over about twice this."));
}

/// The Learned Detail node's settings: the terrain type, then each stage on or off with its own steps, guidance and amount.
fn detail_ui(ui: &mut eframe::egui::Ui, e: &mut Erosion) {
    use eframe::egui;
    let d = Erosion::default();
    ui.checkbox(&mut e.auto, "run by itself").on_hover_text("Run again when what feeds this node changes. Off: the last result stays (marked out of date) until you press Run.");
    crate::row(ui, &mut e.detail_type, d.detail_type.clone(), |ui, v| {
        egui::ComboBox::from_id_salt("detail type").selected_text(v.clone()).show_ui(ui, |ui| {
            for t in crate::landscape_detail::TYPES {
                ui.selectable_value(v, t.to_string(), t);
            }
        }).response.on_hover_text("The kind of real terrain whose detail is added.");
    });
    let names = [("Landforms: 16x to 4x", "Reads the input 16 times coarser and adds ridges, gullies and slopes back to 4 times coarser. Trained for 4 m output cells."),
                 ("Fine detail: 4x to the grid", "Reads the input (or the stage before) 4 times coarser and adds the finest detail back to the working grid. Trained for 1 m cells (a 1 km terrain at 1024)."),
                 ("Sharpen: finest grain", "Adds the finest grain on the working grid itself (no coarser copy), to the stages above or to the input. Trained for 1 m cells. Experimental: an early model.")];
    for (i, (name, tip)) in names.iter().enumerate() {
        let (s, ds) = (&mut e.detail_stages[i], d.detail_stages[i]);
        ui.separator();
        ui.checkbox(&mut s.on, *name).on_hover_text(*tip);
        ui.add_enabled_ui(s.on, |ui| {
            ui.push_id(i, |ui| {
                crate::row(ui, &mut s.steps, ds.steps, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1..=64).speed(0.2).prefix("steps ")).on_hover_text("Sampling steps: more is slower."));
                crate::row(ui, &mut s.guidance, ds.guidance, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=8.0).speed(0.01).prefix("guidance ")).on_hover_text("How strongly the terrain type shows: 1 is the model as trained; higher pushes it away from an average terrain, so the types look more different and the detail stronger. Other than 1, each step costs twice as long."));
                crate::row(ui, &mut s.amount, ds.amount, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=3.0).speed(0.01).prefix("amount ")).on_hover_text("How much of this stage's detail is kept: 0 none, 1 all, above 1 strengthened."));
                if i == 0 {
                    crate::row(ui, &mut s.adherence, ds.adherence, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("adherence ")).on_hover_text("How closely the input's shape is followed: 1 keeps it exactly; lower lets the model reshape it into landforms of the chosen type. Around 0.9 the shapes stay and the ground gets rougher; at 0.75 and below, hills up to about a kilometre across are replaced. Below 1 the result no longer matches the input exactly."));
                }
            });
        });
    }
}

/// Erodes `heights` (`n` x `n` metres, rows along y) on a grid whose cells are `cell_m` metres apart.
pub fn erode(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], n: usize, cell_m: f32, e: &Erosion, progress: &Progress) -> Result<ErosionResult, String> {
    if e.kind == ErosionKind::Wind {
        return crate::landscape_wind::erode_wind(device, queue, heights, n, cell_m, e, progress);
    }
    if e.kind == ErosionKind::Blur {
        let sigma = e.blur_m / cell_m;
        let heights = if sigma < 0.1 || heights.len() != n * n { heights.to_vec() } else { crate::landscape_detail::gauss(heights, n, n, sigma) };
        return Ok(ErosionResult { heights, flow: vec![0.0; n * n] });
    }
    if e.kind == ErosionKind::Detail {
        return crate::landscape_detail::detail(device, queue, heights, n, cell_m, e, 0, progress);
    }
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
    let (d, s, s_mid, db, flux, vel, flux2) = (buffer(1), buffer(1), buffer(1), buffer(1), buffer(4), buffer(2), buffer(4));
    for buf in [&d, &s, &s_mid, &db, &flux, &vel, &flux2] {
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
    entries.extend((1..9).map(storage_entry));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
    let (flux_pass, water_pass, erode_pass, transport_pass, thermal_calc, thermal_apply, finish) = (pipeline("flux_pass"), pipeline("water_pass"), pipeline("erode_pass"), pipeline("transport_pass"), pipeline("thermal_calc"), pipeline("thermal_apply"), pipeline("finish"));
    let bind = |u: &wgpu::Buffer| {
        let bufs = [u, &b, &d, &s, &s_mid, &db, &flux, &vel, &flux2];
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
