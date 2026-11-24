//! The Landscape tab, phase 0: a noise graph evaluated over a square of terrain, drawn in 3D and exported as a
//! heightmap for Unreal's Landscape import. The graph is a normal noise graph (its density output is the height), so
//! every noise, pattern and math node works as a terrain generator; the simulation nodes (erosion and so on) need
//! their own grid passes and come later (docs/landscape-plan.md).
//!
//! The terrain is one tile of the graph, `x` and `y` in 0..1, with the height up; the world size in metres only scales
//! the export and the view.

use crate::editor::Editor;
use crate::export::{Evaluator, Progress};
use crate::graph::{Graph, NodeKind, NoiseParams};
use eframe::egui;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// Samples across an Unreal landscape export: 64 components of 63 quads, plus one (a valid landscape size).
pub const UNREAL_RES: usize = 4033;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    pub seed: u32,
    /// Side of the square terrain in metres.
    pub world_m: f32,
    /// Height range in metres.
    pub height_m: f32,
    /// Stretch the graph's output to fill 0..`height_m` (otherwise the output times `height_m`, around 0).
    pub fit: bool,
    pub erosion: crate::landscape_erosion::Erosion,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { seed: 0, world_m: 2000.0, height_m: 400.0, fit: true, erosion: Default::default() }
    }
}

/// The starting graph: position, fractal noise, density output.
pub fn default_graph() -> Graph {
    let mut g = Graph::default();
    let pos = g.add(NodeKind::position(), [0.0, 0.0]);
    let noise = g.add(NodeKind::Noise(NoiseParams { scale: 3.0, octaves: 6, ..Default::default() }), [200.0, 0.0]);
    let out = g.add(NodeKind::OutDensity, [400.0, 0.0]);
    g.connect(pos, noise, 0);
    g.connect(noise, out, 0);
    g
}

/// The graph's output over an `n` x `n` grid across the tile (both edges included), rows along y, x fastest.
pub fn sample(ev: &Evaluator, n: usize, progress: &Progress) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * n);
    let rows_per_call = (262_000 / n).max(1);
    let step = 1.0 / (n - 1).max(1) as f32;
    let mut y = 0;
    while y < n {
        let rows = rows_per_call.min(n - y);
        let pts: Vec<[f32; 3]> = (y..y + rows).flat_map(|j| (0..n).map(move |i| [i as f32 * step, j as f32 * step, 0.0])).collect();
        out.extend(ev.eval_points(&pts));
        y += rows;
        progress.store((y * 1000 / n) as u32, Ordering::Relaxed);
    }
    out
}

/// The side of a square grid with `len` values.
fn side(len: usize) -> usize {
    (len as f64).sqrt().round() as usize
}

/// A grid value (`n` x `n`, rows along y) at `x`, `y` across the terrain (0 to 1), by bilinear interpolation.
pub fn bilinear(grid: &[f32], n: usize, x: f32, y: f32) -> f32 {
    let top = (n - 1) as f32 - 1e-3;
    let (gx, gy) = ((x * (n - 1) as f32).clamp(0.0, top.max(0.0)), (y * (n - 1) as f32).clamp(0.0, top.max(0.0)));
    let (i, j, fx, fy) = (gx as usize, gy as usize, gx.fract(), gy.fract());
    let (i1, j1) = ((i + 1).min(n - 1), (j + 1).min(n - 1));
    (grid[j * n + i] * (1.0 - fx) + grid[j * n + i1] * fx) * (1.0 - fy) + (grid[j1 * n + i] * (1.0 - fx) + grid[j1 * n + i1] * fx) * fy
}

/// The grid as an `m` x `m` one, `m` at most `max`, for drawing.
fn shrink(grid: &[f32], n: usize, max: usize) -> (Vec<f32>, usize) {
    let m = n.min(max);
    if m == n {
        return (grid.to_vec(), n);
    }
    (((0..m * m).map(|k| bilinear(grid, n, (k % m) as f32 / (m - 1) as f32, (k / m) as f32 / (m - 1) as f32))).collect(), m)
}

/// How the graph's values become heights in metres: `value * k + offset`.
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub k: f32,
    pub offset: f32,
}

impl Mapping {
    /// With `fit`, the lowest of `values` becomes 0 and the highest `height_m`; otherwise `value * height_m`.
    pub fn from_values(values: &[f32], s: &Settings) -> Mapping {
        if s.fit {
            let (lo, hi) = values.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            let k = if hi > lo { s.height_m / (hi - lo) } else { 0.0 };
            Mapping { k, offset: -lo * k }
        } else {
            Mapping { k: s.height_m, offset: 0.0 }
        }
    }

    pub fn apply(&self, v: f32) -> f32 {
        v * self.k + self.offset
    }
}

/// Heights in metres for the graph's values.
pub fn to_metres(values: &[f32], s: &Settings) -> Vec<f32> {
    let m = Mapping::from_values(values, s);
    values.iter().map(|v| m.apply(*v)).collect()
}

/// An Unreal landscape's heights: 16-bit samples (32768 is the actor's own height), and the actor's Z scale and
/// height that map them back to metres. `step_cm` is the height of one step.
pub struct UnrealHeights {
    pub samples: Vec<u16>,
    pub z_scale: f32,
    pub centre_cm: f32,
    pub step_cm: f32,
    pub range_m: (f32, f32),
}

/// Unreal reads a height as `(v - 32768) / 128` units times the actor's Z scale (LandscapeDataAccess.h), so the scale is
/// chosen to make the 16 bits span exactly the terrain's height range, centred on the actor.
pub fn encode_unreal(metres: &[f32]) -> UnrealHeights {
    let (lo, hi) = metres.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let hi = if hi - lo < 1e-4 { lo + 1.0 } else { hi };
    let (span_cm, centre_cm) = ((hi - lo) * 100.0, (lo + hi) * 50.0);
    let z_scale = span_cm / 512.0;
    let step_cm = z_scale / 128.0;
    let samples = metres.iter().map(|m| (32768.0 + (m * 100.0 - centre_cm) / step_cm).round().clamp(0.0, 65535.0) as u16).collect();
    UnrealHeights { samples, z_scale, centre_cm, step_cm, range_m: (lo, hi) }
}

/// Writes `<path>` (16-bit grayscale PNG) and `<path>.json` with what to enter in Unreal's import dialog.
pub fn write_unreal(path: &Path, n: usize, world_m: f32, h: &UnrealHeights) -> Result<String, String> {
    let img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(n as u32, n as u32, h.samples.clone()).ok_or("bad image size")?;
    img.save(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let xy = world_m * 100.0 / (n - 1) as f32;
    let side = std::path::PathBuf::from(format!("{}.json", path.display()));
    let json = serde_json::json!({
        "resolution": n,
        "size_m": world_m,
        "height_range_m": [h.range_m.0, h.range_m.1],
        "step_cm": h.step_cm,
        "unreal_import": {
            "location_cm": [0.0, 0.0, h.centre_cm],
            "scale": [xy, xy, h.z_scale],
            "note": "Landscape mode > Import from File: this heightmap, this resolution; set the scale and location above. 16 bits cover the height range in steps of step_cm."
        }
    });
    std::fs::write(&side, serde_json::to_string_pretty(&json).unwrap_or_default()).map_err(|e| format!("{}: {e}", side.display()))?;
    Ok(format!("Wrote {} ({n} x {n}, {:.2} cm per step) and {}", path.display(), h.step_cm, side.file_name().unwrap_or_default().to_string_lossy()))
}

/// The terrain from a camera by ray marching the height grid: metres in, a picture out.
fn render(metres: &[f32], n: usize, world_m: f32, (yaw, pitch, zoom): (f32, f32, f32), (w, h): (usize, usize)) -> egui::ColorImage {
    use rayon::prelude::*;
    let (lo, hi) = metres.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let range = (hi - lo).max(1e-3);
    let cell = world_m / (n - 1) as f32;
    let at = |x: f32, y: f32| -> f32 {
        let (gx, gy) = ((x / cell).clamp(0.0, (n - 1) as f32 - 1e-3), (y / cell).clamp(0.0, (n - 1) as f32 - 1e-3));
        let (i, j, fx, fy) = (gx as usize, gy as usize, gx.fract(), gy.fract());
        let m = |i: usize, j: usize| metres[j * n + i];
        (m(i, j) * (1.0 - fx) + m(i + 1, j) * fx) * (1.0 - fy) + (m(i, j + 1) * (1.0 - fx) + m(i + 1, j + 1) * fx) * fy
    };
    // Z is up. The camera orbits the middle of the terrain.
    let target = [world_m * 0.5, world_m * 0.5, (lo + hi) * 0.5];
    let dist = world_m * 1.3 / zoom.max(0.1);
    let eye = [target[0] + dist * pitch.cos() * yaw.cos(), target[1] + dist * pitch.cos() * yaw.sin(), target[2] + dist * pitch.sin()];
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let fwd = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let right = norm([fwd[1], -fwd[0], 0.0]);
    let up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
    let tan_half = (40.0f32.to_radians() * 0.5).tan();
    let light = norm([0.5, 0.35, 0.8]);
    let mut pixels = vec![egui::Color32::BLACK; w * h];
    pixels.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        for (px, out) in row.iter_mut().enumerate() {
            let (sx, sy) = ((px as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (py as f32 + 0.5) / h as f32 * 2.0);
            let (cx, cy) = (sx * tan_half * w as f32 / h as f32, sy * tan_half);
            let d = norm([fwd[0] + right[0] * cx + up[0] * cy, fwd[1] + right[1] * cx + up[1] * cy, fwd[2] + right[2] * cx + up[2] * cy]);
            // Sky: lighter at the horizon.
            let t = (d[2] * 2.0).clamp(0.0, 1.0);
            let sky = [150.0 + (70.0 - 150.0) * t, 180.0 + (110.0 - 180.0) * t, 215.0 + (170.0 - 215.0) * t];
            *out = egui::Color32::from_rgb(sky[0] as u8, sky[1] as u8, sky[2] as u8);
            // March from the camera until the ray dips below the surface.
            let step = cell * 0.9;
            let (mut t, mut prev) = (0.0f32, 0.0f32);
            let max_t = dist * 3.0;
            while t < max_t {
                let p = [eye[0] + d[0] * t, eye[1] + d[1] * t, eye[2] + d[2] * t];
                if p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= world_m && p[1] <= world_m {
                    if p[2] < at(p[0], p[1]) {
                        // Refine between the last two steps.
                        let (mut a, mut b) = (prev, t);
                        for _ in 0..5 {
                            let m = (a + b) * 0.5;
                            let q = [eye[0] + d[0] * m, eye[1] + d[1] * m, eye[2] + d[2] * m];
                            if q[2] < at(q[0], q[1]) { b = m } else { a = m }
                        }
                        let q = [eye[0] + d[0] * b, eye[1] + d[1] * b, eye[2] + d[2] * b];
                        let (gx, gy) = ((at(q[0] + cell, q[1]) - at(q[0] - cell, q[1])) / (2.0 * cell), (at(q[0], q[1] + cell) - at(q[0], q[1] - cell)) / (2.0 * cell));
                        let nrm = norm([-gx, -gy, 1.0]);
                        let diffuse = (nrm[0] * light[0] + nrm[1] * light[1] + nrm[2] * light[2]).max(0.0);
                        let height = ((q[2] - lo) / range).clamp(0.0, 1.0);
                        // Low ground green, higher ground rock, the top snow; steep slopes always rock.
                        let green = [70.0, 105.0, 60.0];
                        let rock = [125.0, 115.0, 100.0];
                        let snow = [235.0, 238.0, 245.0];
                        let blend = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
                        let low = blend(green, rock, ((height - 0.25) * 3.0).clamp(0.0, 1.0));
                        let base = blend(low, snow, ((height - 0.78) * 8.0).clamp(0.0, 1.0));
                        let base = blend(base, rock, ((1.0 - nrm[2]) * 3.5 - 0.35).clamp(0.0, 1.0) * (1.0 - ((height - 0.78) * 8.0).clamp(0.0, 0.6)));
                        let shade = 0.25 + 0.85 * diffuse;
                        *out = egui::Color32::from_rgb((base[0] * shade).min(255.0) as u8, (base[1] * shade).min(255.0) as u8, (base[2] * shade).min(255.0) as u8);
                        break;
                    }
                }
                prev = t;
                t += step;
            }
        }
    });
    egui::ColorImage { size: [w, h], source_size: egui::vec2(w as f32, h as f32), pixels }
}

/// What a finished job hands back.
enum Done {
    Values(Vec<f32>),
    Eroded(Vec<f32>, u64),
    Message(String),
}

/// A background job: sampling the graph (preview or export) and what to do with the result.
enum Job {
    /// The graph's values on the working grid.
    Preview(mpsc::Receiver<Result<Vec<f32>, String>>),
    /// The eroded heights, and the key (see `erosion_key`) they belong to.
    Erode(mpsc::Receiver<Result<Vec<f32>, String>>, Progress, u64),
    Export(mpsc::Receiver<Result<String, String>>, Progress),
    /// The chunk export, with its log.
    Chunks(mpsc::Receiver<Result<String, String>>, Progress, std::sync::Arc<std::sync::Mutex<Vec<String>>>),
}

pub struct LandscapeWorkspace {
    pub graph: Graph,
    pub editor: Editor,
    pub path: Option<PathBuf>,
    pub settings: Settings,
    /// The graph's values on the working grid, and the key (graph, seed, resolution) they were sampled for.
    values: Option<Vec<f32>>,
    sampled_key: Option<u64>,
    /// The eroded heights in metres and the key of everything they were made from; out of date when it differs.
    eroded: Option<(u64, Vec<f32>)>,
    job: Option<Job>,
    status: String,
    pub chunks: crate::landscape_chunks::Options,
    chunk_log: Vec<String>,
    view: (f32, f32, f32),
    picture: Option<(u64, egui::TextureHandle)>,
}

impl Default for LandscapeWorkspace {
    fn default() -> Self {
        LandscapeWorkspace { graph: default_graph(), editor: Editor::default(), path: None, settings: Settings::default(), values: None, sampled_key: None, eroded: None, job: None, status: String::new(), chunks: Default::default(), chunk_log: Vec::new(), view: (0.8, 0.6, 1.0), picture: None }
    }
}

impl LandscapeWorkspace {
    pub fn restore(&mut self, graph: Graph, settings: Settings, path: Option<PathBuf>) {
        (self.graph, self.settings, self.path) = (graph, settings, path);
    }

    /// What the sampled values depend on: the graph, the seed and the working resolution.
    fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.graph.to_json(), self.settings.seed, self.settings.erosion.resolution).hash(&mut h);
        h.finish()
    }

    /// What eroded heights depend on: the sampled values, how they become metres, and the erosion settings.
    fn erosion_key(&self) -> u64 {
        let s = &self.settings;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.key(), s.world_m.to_bits(), s.height_m.to_bits(), s.fit, serde_json::to_string(&s.erosion).unwrap_or_default()).hash(&mut h);
        h.finish()
    }

    /// The eroded heights when they are up to date.
    fn current_eroded(&self) -> Option<&Vec<f32>> {
        self.eroded.as_ref().filter(|(k, _)| *k == self.erosion_key()).map(|(_, e)| e)
    }

    fn open(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Noise graph", &["noisegraph", "json"]).pick_file() else { return };
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Graph::from_json(&t)) {
            Ok(g) => {
                self.graph = g;
                self.path = Some(path);
            }
            Err(e) => self.status = format!("could not open: {e}"),
        }
    }

    fn save(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Noise graph", &["noisegraph"]).set_file_name("terrain.noisegraph").save_file() else { return };
        self.status = match std::fs::write(&path, self.graph.to_json()) {
            Ok(()) => {
                self.path = Some(path.clone());
                format!("Saved {}", path.display())
            }
            Err(e) => format!("could not save: {e}"),
        };
    }
}

impl crate::App {
    pub(crate) fn landscape_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let ls = &mut self.landscape;

        // The background job: its result.
        let mut finished = None;
        match &ls.job {
            Some(Job::Preview(rx)) => match rx.try_recv() {
                Ok(r) => finished = Some(r.map(Done::Values)),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the sampler stopped unexpectedly".into())),
            },
            Some(Job::Erode(rx, _, key)) => match rx.try_recv() {
                Ok(r) => finished = Some(r.map(|e| Done::Eroded(e, *key))),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the erosion stopped unexpectedly".into())),
            },
            Some(Job::Chunks(rx, _, log)) => {
                ls.chunk_log = log.lock().unwrap().clone();
                match rx.try_recv() {
                    Ok(r) => finished = Some(r.map(Done::Message)),
                    Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
                    Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the chunk export stopped unexpectedly".into())),
                }
            }
            Some(Job::Export(rx, _)) => match rx.try_recv() {
                Ok(r) => finished = Some(r.map(Done::Message)),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the export stopped unexpectedly".into())),
            },
            None => {}
        }
        if let Some(r) = finished {
            ls.job = None;
            match r {
                Ok(Done::Values(v)) => ls.values = Some(v),
                Ok(Done::Eroded(e, key)) => {
                    ls.eroded = Some((key, e));
                    ls.status = "Erosion done".into();
                }
                Ok(Done::Message(m)) => ls.status = m,
                Err(e) => {
                    ls.status = e;
                    ls.values = None;
                }
            }
        }
        // Sample the preview again when the graph or seed changed and nothing else is running.
        let key = ls.key();
        if ls.job.is_none() && ls.sampled_key != Some(key) {
            ls.sampled_key = Some(key);
            let (graph, seed, resolution) = (ls.graph.clone(), ls.settings.seed, ls.settings.erosion.resolution);
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let r = Evaluator::new(&device, &queue, &graph, seed).map(|ev| sample(&ev, resolution, &Progress::default()));
                let _ = tx.send(r);
            });
            ls.job = Some(Job::Preview(rx));
        }

        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let ls = &mut self.landscape;
        egui::Panel::right("landscape props").default_size(320.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").on_hover_text("Opens a noise graph as the terrain.").clicked() {
                    ls.open();
                }
                if ui.button("Save…").clicked() {
                    ls.save();
                }
                if ui.button("New").clicked() {
                    ls.graph = default_graph();
                    ls.path = None;
                }
            });
            ui.separator();
            let d = Settings::default();
            crate::row(ui, &mut ls.settings.seed, d.seed, |ui, v| _ = ui.add(egui::DragValue::new(v).prefix("seed ")));
            crate::row(ui, &mut ls.settings.world_m, d.world_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(10.0..=100000.0).speed(10.0).prefix("size ").suffix(" m")).on_hover_text("Side of the square terrain."));
            crate::row(ui, &mut ls.settings.height_m, d.height_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=20000.0).speed(1.0).prefix("height ").suffix(" m")));
            crate::row(ui, &mut ls.settings.fit, d.fit, |ui, v| _ = ui.checkbox(v, "fit the height range").on_hover_text("On: the lowest point is 0 and the highest is the height. Off: the graph's output times the height, around 0."));
            ui.weak("The terrain is one tile of the graph (x and y from 0 to 1) with the density output as height, so any noise graph works. Tiling graphs make a terrain whose edges match.");
            let running = matches!(ls.job, Some(Job::Export(..) | Job::Chunks(..) | Job::Erode(..)));
            if ui.add_enabled(!running && ls.values.is_some(), egui::Button::new("Export heightmap for Unreal…")).on_hover_text(format!("A 16-bit PNG at {UNREAL_RES} x {UNREAL_RES} (a valid landscape size) and a .json with the import settings.")).clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("16-bit PNG", &["png"]).set_file_name("heightmap.png").save_file() {
                let (graph, s) = (ls.graph.clone(), ls.settings.clone());
                let eroded = ls.current_eroded().cloned();
                let (tx, rx) = mpsc::channel();
                let progress = Progress::default();
                let p = progress.clone();
                let (device, queue) = (device.clone(), queue.clone());
                std::thread::spawn(move || {
                    let r = match eroded {
                        // The eroded grid, resampled to the landscape size.
                        Some(e) => {
                            let n = side(e.len());
                            let step = 1.0 / (UNREAL_RES - 1) as f32;
                            let metres: Vec<f32> = (0..UNREAL_RES * UNREAL_RES).map(|k| bilinear(&e, n, (k % UNREAL_RES) as f32 * step, (k / UNREAL_RES) as f32 * step)).collect();
                            write_unreal(&path, UNREAL_RES, s.world_m, &encode_unreal(&metres))
                        }
                        None => Evaluator::new(&device, &queue, &graph, s.seed).and_then(|ev| {
                            let metres = to_metres(&sample(&ev, UNREAL_RES, &p), &s);
                            write_unreal(&path, UNREAL_RES, s.world_m, &encode_unreal(&metres))
                        }),
                    };
                    let _ = tx.send(r);
                });
                ls.job = Some(Job::Export(rx, progress));
                ls.status = "Exporting…".into();
            }
            if let Some(Job::Export(_, p)) = &ls.job {
                ui.add(egui::ProgressBar::new(p.load(Ordering::Relaxed) as f32 / 1000.0).show_percentage());
            }
            if let Some(Job::Chunks(_, p, _)) = &ls.job {
                ui.add(egui::ProgressBar::new(p.load(Ordering::Relaxed) as f32 / 1000.0).show_percentage());
            }
            egui::CollapsingHeader::new("Erosion").default_open(true).show(ui, |ui| {
                let d = crate::landscape_erosion::Erosion::default();
                let max = crate::landscape_erosion::max_resolution(&device);
                let up_to_date = ls.current_eroded().is_some();
                let e = &mut ls.settings.erosion;
                crate::row(ui, &mut e.resolution, d.resolution, |ui, v| {
                    for r in crate::landscape_erosion::RESOLUTIONS.into_iter().filter(|r| *r <= max) {
                        ui.selectable_value(v, r, format!("{r}"));
                    }
                });
                crate::row(ui, &mut e.iterations, d.iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=20000).speed(10.0).prefix("iterations ")));
                crate::row(ui, &mut e.rain, d.rain, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("rain ").suffix(" m/s")));
                crate::row(ui, &mut e.capacity, d.capacity, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("capacity ")).on_hover_text("How much sediment fast water on a slope can carry: more means deeper channels."));
                crate::row(ui, &mut e.dissolve, d.dissolve, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("dissolve ")).on_hover_text("How quickly water picks sediment up."));
                crate::row(ui, &mut e.deposit, d.deposit, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005).prefix("deposit ")).on_hover_text("How quickly water drops sediment."));
                crate::row(ui, &mut e.evaporation, d.evaporation, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.001).prefix("evaporation ")));
                crate::row(ui, &mut e.thermal_iterations, d.thermal_iterations, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0..=5000).speed(1.0).prefix("thermal iterations ")));
                crate::row(ui, &mut e.talus_deg, d.talus_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=89.0).speed(0.5).prefix("repose ").suffix(" deg")));
                ui.horizontal(|ui| {
                    if ui.add_enabled(!running && ls.values.is_some(), egui::Button::new("Run erosion")).on_hover_text("Erodes the terrain at the working resolution. The preview and both exports use the result until the graph or a setting changes.").clicked() {
                        let s = ls.settings.clone();
                        let values = ls.values.clone().unwrap_or_default();
                        let (n, key) = (side(values.len()), ls.erosion_key());
                        let metres = to_metres(&values, &s);
                        let cell = s.world_m / (n.max(2) - 1) as f32;
                        let (tx, rx) = mpsc::channel();
                        let progress = Progress::default();
                        let p = progress.clone();
                        let (device, queue) = (device.clone(), queue.clone());
                        std::thread::spawn(move || {
                            let _ = tx.send(crate::landscape_erosion::erode(&device, &queue, &metres, n, cell, &s.erosion, &p));
                        });
                        ls.job = Some(Job::Erode(rx, progress, key));
                        ls.status = "Eroding...".into();
                    }
                    if ui.add_enabled(ls.eroded.is_some(), egui::Button::new("Clear")).clicked() {
                        ls.eroded = None;
                    }
                });
                if let Some(Job::Erode(_, p, _)) = &ls.job {
                    ui.add(egui::ProgressBar::new(p.load(Ordering::Relaxed) as f32 / 1000.0).show_percentage());
                }
                ui.weak(match (ls.eroded.is_some(), up_to_date) {
                    (true, true) => "Eroded: the preview and the exports use it.",
                    (true, false) => "Out of date: the graph or a setting changed. Run again, or the exports use the plain graph.",
                    _ => "Not eroded. The working resolution is the grid the preview and the erosion use.",
                });
            });
            egui::CollapsingHeader::new("Nanite chunks").default_open(true).show(ui, |ui| {
                let d = crate::landscape_chunks::Options::default();
                let o = &mut ls.chunks;
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.text_edit_singleline(&mut o.name);
                });
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut o.folder).hint_text("folder").desired_width(190.0));
                    if ui.button("Browse…").clicked() && let Some(f) = rfd::FileDialog::new().pick_folder() {
                        o.folder = f.display().to_string();
                    }
                });
                crate::row(ui, &mut o.chunk_m, d.chunk_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(10.0..=20000.0).speed(5.0).prefix("chunk ").suffix(" m")));
                crate::row(ui, &mut o.budget, d.budget, |ui, v| _ = ui.add(egui::DragValue::new(v).range(2000..=4_000_000).speed(1000.0).prefix("triangles ").suffix(" per chunk")));
                crate::row(ui, &mut o.tex, d.tex, |ui, v| {
                    for t in [1024usize, 2048, 4096] {
                        ui.selectable_value(v, t, format!("{t} px"));
                    }
                });
                crate::row(ui, &mut o.ao_m, d.ao_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=500.0).prefix("AO reach ").suffix(" m")));
                crate::row(ui, &mut o.glb, d.glb, |ui, v| {
                    ui.selectable_value(v, true, "GLB");
                    ui.selectable_value(v, false, "OBJ");
                });
                ui.checkbox(&mut o.green_up, "normal map green up (OpenGL)");
                ui.checkbox(&mut o.pack_udim, "also pack into a UDIM tileset").on_hover_text("Runs UDIM pack into a UDIM folder: one combined mesh, one material, the maps as .1001, .1002 ...");
                let plan = crate::landscape_chunks::plan(ls.settings.world_m, o.chunk_m, o.budget);
                ui.weak(format!("{0} x {0} chunks of {1} m, {2} x {2} quads (about {3} triangles) each; tiles 1001 to {4}.", plan.chunks, plan.chunk_m, plan.quads, 2 * plan.quads * plan.quads, crate::udim::udim(plan.chunks * plan.chunks - 1)));
                if ui.add_enabled(!running && !o.folder.is_empty(), egui::Button::new("Export chunks")).on_hover_text("Writes every chunk's mesh (centimetres, the chunk's corner at its origin), its normal, occlusion and curvature maps, and a manifest with each chunk's location.").clicked() {
                    let (graph, s, opts) = (ls.graph.clone(), ls.settings.clone(), o.clone());
                    let eroded = ls.current_eroded().cloned();
                    let (tx, rx) = mpsc::channel();
                    let (progress, log) = (Progress::default(), std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new())));
                    let (p, l) = (progress.clone(), log.clone());
                    let (device, queue) = (device.clone(), queue.clone());
                    std::thread::spawn(move || {
                        let r = Evaluator::new(&device, &queue, &graph, s.seed).and_then(|ev| {
                            let grid = eroded.as_ref().map(|e| (e.as_slice(), side(e.len())));
                            crate::landscape_chunks::export(&ev, grid, &s, &opts, &p, &mut |m| l.lock().unwrap().push(m.to_string()))
                        });
                        let _ = tx.send(r);
                    });
                    ls.job = Some(Job::Chunks(rx, progress, log));
                    ls.status = "Exporting chunks…".into();
                }
                egui::ScrollArea::vertical().id_salt("chunk log").max_height(120.0).stick_to_bottom(true).show(ui, |ui| {
                    for l in &ls.chunk_log {
                        ui.weak(l);
                    }
                });
            });
            if !ls.status.is_empty() {
                ui.label(&ls.status);
            }
            ui.separator();
            // The selected node's settings, as in the noise graph tab.
            let names = ls.graph.declared_names();
            if let [id] = ls.editor.selected.iter().copied().collect::<Vec<_>>().as_slice()
                && let Some(node) = ls.graph.nodes.get_mut(id) {
                ui.heading(crate::editor::title(&node.kind));
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| crate::node_props(ui, &mut node.kind, &mut node.pins, &names));
            }
        });

        // The terrain, then the graph.
        let (settings, view) = (ls.settings.clone(), ls.view);
        let mut new_view = view;
        egui::Panel::top("landscape view").resizable(true).default_size(380.0).show(ui, |ui| {
            let size = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            if response.dragged() {
                let delta = response.drag_delta();
                new_view.0 -= delta.x * 0.01;
                new_view.1 = (new_view.1 + delta.y * 0.01).clamp(0.05, 1.5);
            }
            if response.hovered() {
                new_view.2 = (new_view.2 * (1.0 + ui.input(|i| i.smooth_scroll_delta.y) * 0.002)).clamp(0.3, 6.0);
            }
            if let Some(values) = &ls.values {
                let (w, h) = (((rect.width() * 0.6) as usize).clamp(64, 900), ((rect.height() * 0.6) as usize).clamp(48, 600));
                let eroded_key = ls.eroded.as_ref().map(|(k, _)| *k).filter(|k| *k == ls.erosion_key());
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                (ls.sampled_key, eroded_key, settings.world_m.to_bits(), settings.height_m.to_bits(), settings.fit, new_view.0.to_bits(), new_view.1.to_bits(), new_view.2.to_bits(), w, h).hash(&mut hasher);
                let picture_key = hasher.finish();
                if ls.picture.as_ref().is_none_or(|(k, _)| *k != picture_key) {
                    // The eroded terrain when it is up to date, otherwise the graph's; drawn from at most 513 x 513 samples.
                    let metres = match (&ls.eroded, eroded_key) {
                        (Some((_, e)), Some(_)) => e.clone(),
                        _ => to_metres(values, &settings),
                    };
                    let (grid, m) = shrink(&metres, side(metres.len()), 513);
                    let image = render(&grid, m, settings.world_m, new_view, (w, h));
                    ls.picture = Some((picture_key, ctx.load_texture("landscape view", image, egui::TextureOptions::LINEAR)));
                }
                if let Some((_, tex)) = &ls.picture {
                    ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                }
            } else {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(30));
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, if matches!(ls.job, Some(Job::Preview(_))) { "Sampling…" } else { "No terrain: the graph has no density output, or has an error" }, egui::FontId::proportional(14.0), egui::Color32::GRAY);
            }
            ui.painter().text(rect.left_bottom() + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "drag to orbit, scroll to zoom", egui::FontId::proportional(11.0), egui::Color32::from_white_alpha(120));
        });
        ls.view = new_view;
        let tiling = ls.graph.tiling();
        ls.editor.show(ui, &mut ls.graph, &tiling);
        ls.graph.prune_edges();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heights_encode_to_the_full_16_bits_and_decode_within_a_step() {
        let metres = [0.0, 100.0, 250.0, 400.0];
        let h = encode_unreal(&metres);
        assert_eq!((h.samples[0], h.samples[3]), (0, 65535));
        for (m, v) in metres.iter().zip(&h.samples) {
            let decoded_cm = h.centre_cm + (*v as f32 - 32768.0) / 128.0 * h.z_scale;
            assert!((decoded_cm - m * 100.0).abs() <= h.step_cm, "{m} m decoded as {decoded_cm} cm");
        }
        // A flat terrain still encodes (the range is widened to avoid dividing by zero).
        assert!(encode_unreal(&[5.0; 4]).step_cm > 0.0);
    }

    #[test]
    fn the_height_range_fits_or_scales() {
        let s = Settings { height_m: 100.0, ..Default::default() };
        assert_eq!(to_metres(&[-1.0, 0.0, 1.0], &s), [0.0, 50.0, 100.0]);
        let s = Settings { fit: false, height_m: 100.0, ..Default::default() };
        assert_eq!(to_metres(&[-1.0, 0.5], &s), [-100.0, 50.0]);
    }

    #[test]
    fn the_heightmap_png_is_16_bit_and_comes_with_import_settings() {
        let n = 33;
        let metres: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 * 10.0).collect();
        let h = encode_unreal(&metres);
        let path = std::env::temp_dir().join("noiser_landscape_test.png");
        write_unreal(&path, n, 1000.0, &h).unwrap();
        let back = image::open(&path).unwrap().to_luma16();
        assert_eq!((back.width(), back.get_pixel(0, 0).0[0], back.get_pixel(32, 5).0[0]), (n as u32, 0, 65535));
        let side: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{}.json", path.display())).unwrap()).unwrap();
        assert_eq!(side["resolution"], 33);
        // 1000 m over 32 quads: 3125 cm per quad.
        assert!((side["unreal_import"]["scale"][0].as_f64().unwrap() - 3125.0).abs() < 1e-3);
    }

    #[test]
    fn a_valid_unreal_size() {
        // 64 components of 63 quads (LandscapeConfigHelper.cpp lists 7, 15, 31, 63, 127, 255 quads per section).
        assert_eq!(UNREAL_RES, 63 * 64 + 1);
    }
}
