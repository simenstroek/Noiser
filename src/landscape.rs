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
use crate::landscape_masks::{Brush, Mask, Tool};
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

/// The names of the masks, for the Mask node's menu (set by the Landscape tab every frame).
static MASK_NAMES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub fn mask_names() -> Vec<String> {
    MASK_NAMES.lock().map(|n| n.clone()).unwrap_or_default()
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

use self::side as side_of;

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

/// The evaluator for the terrain: Noise Graph nodes inlined, the slope scale set from the size and height, the masks in.
pub(crate) fn build_evaluator(device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue, graph: &Graph, s: &Settings, masks: &std::collections::BTreeMap<String, Vec<f32>>, base: &Path) -> Result<Evaluator, String> {
    let mut g = crate::meshgraph::lowered_mesh(graph, base)?;
    for n in g.nodes.values_mut() {
        if let NodeKind::Slope { scale } = &mut n.kind {
            *scale = s.height_m / s.world_m.max(1.0);
        }
    }
    Evaluator::with_masks(device, queue, &g, s.seed, masks)
}

/// The map: the terrain from above, lit from the north-west (`m` x `m` metres, `world_m` across).
fn hillshade(metres: &[f32], m: usize, world_m: f32) -> egui::ColorImage {
    let cell = world_m / (m - 1) as f32;
    let (lo, hi) = metres.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let at = |x: usize, y: usize| metres[y.min(m - 1) * m + x.min(m - 1)];
    let pixels = (0..m * m)
        .map(|k| {
            let (x, y) = (k % m, k / m);
            let (dx, dy) = ((at(x + 1, y) - at(x.saturating_sub(1), y)) / (2.0 * cell), (at(x, y + 1) - at(x, y.saturating_sub(1))) / (2.0 * cell));
            let shade = ((dx * 0.5 + dy * 0.5 + 0.7) / (dx * dx + dy * dy + 1.0).sqrt()).clamp(0.0, 1.0);
            let h = ((at(x, y) - lo) / (hi - lo).max(1e-3)).clamp(0.0, 1.0);
            let v = (60.0 + 150.0 * shade) * (0.75 + 0.25 * h);
            egui::Color32::from_rgb(v as u8, (v * 0.97) as u8, (v * 0.9) as u8)
        })
        .collect();
    egui::ColorImage { size: [m, m], source_size: egui::vec2(m as f32, m as f32), pixels }
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
    /// The painted masks (read by Mask nodes), the selected one, the brush and its tool.
    pub masks: Vec<Mask>,
    mask_selected: usize,
    /// Bumped when a mask edit ends (the terrain is sampled again), and on every dab (the overlay is drawn again).
    mask_rev: u64,
    mask_live: u64,
    brush: Brush,
    tool: Tool,
    /// The top view shows the map (for painting) instead of the 3D terrain.
    map_mode: bool,
    stroke_from: Option<(f32, f32)>,
    new_mask: String,
    overlay: Option<((u64, usize), egui::TextureHandle)>,
    map_picture: Option<(u64, egui::TextureHandle)>,
}

impl Default for LandscapeWorkspace {
    fn default() -> Self {
        LandscapeWorkspace { graph: default_graph(), editor: Editor::default(), path: None, settings: Settings::default(), values: None, sampled_key: None, eroded: None, job: None, status: String::new(), chunks: Default::default(), chunk_log: Vec::new(), view: (0.8, 0.6, 1.0), picture: None, masks: Vec::new(), mask_selected: 0, mask_rev: 0, mask_live: 0, brush: Brush::default(), tool: Tool::Paint, map_mode: false, stroke_from: None, new_mask: String::new(), overlay: None, map_picture: None }
    }
}

impl LandscapeWorkspace {
    pub fn restore(&mut self, graph: Graph, settings: Settings, path: Option<PathBuf>) {
        (self.graph, self.settings, self.path) = (graph, settings, path);
    }

    /// Opens the top view as the map, with a mask to paint on if there is none (`--landscape-map`).
    pub fn show_map(&mut self, on: bool) {
        self.map_mode = on;
        if on && self.masks.is_empty() {
            self.masks.push(Mask::new("mask"));
        }
    }

    /// The whole project as JSON: the graph, the settings and the masks (16-bit PNGs, base64).
    pub fn to_project(&self) -> String {
        let masks: Vec<_> = self.masks.iter().map(|m| serde_json::json!({ "name": m.name, "png": crate::landscape_masks::base64_encode(&m.to_png()) })).collect();
        serde_json::json!({ "noiser_landscape": 1, "graph": self.graph.to_json(), "settings": self.settings, "masks": masks }).to_string()
    }

    /// Replaces the project with one written by `to_project`.
    pub fn from_project(&mut self, text: &str) -> Result<(), String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let graph = Graph::from_json(v.get("graph").and_then(|g| g.as_str()).ok_or("not a landscape project")?)?;
        let settings = v.get("settings").and_then(|s| serde_json::from_value(s.clone()).ok()).unwrap_or_default();
        let masks = v.get("masks").and_then(|m| m.as_array()).map(|a| a.iter().filter_map(|m| Mask::from_png(m.get("name")?.as_str()?, &crate::landscape_masks::base64_decode(m.get("png")?.as_str()?)?)).collect()).unwrap_or_default();
        (self.graph, self.settings, self.masks) = (graph, settings, masks);
        (self.mask_selected, self.mask_rev) = (0, self.mask_rev + 1);
        Ok(())
    }

    fn mask_map(&self) -> std::collections::BTreeMap<String, Vec<f32>> {
        self.masks.iter().map(|m| (m.name.clone(), m.data.clone())).collect()
    }

    /// Where a Noise Graph node's relative file names are looked for: the project's folder.
    fn base_dir(&self) -> PathBuf {
        self.path.as_ref().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
    }

    /// What the sampled values depend on: the graph, the seed and the working resolution.
    fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.graph.to_json(), self.settings.seed, self.settings.erosion.resolution, self.mask_rev).hash(&mut h);
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

    /// A landscape project (graph, settings, masks) or a plain noise graph (the masks stay).
    fn open(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Landscape project or noise graph", &["landscape", "noisegraph", "json"]).pick_file() else { return };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => return self.status = format!("could not open: {e}"),
        };
        let result = if path.extension().is_some_and(|e| e == "landscape") { self.from_project(&text) } else { Graph::from_json(&text).map(|g| self.graph = g) };
        match result {
            Ok(()) => self.path = Some(path),
            Err(e) => self.status = format!("could not open: {e}"),
        }
    }

    fn save(&mut self, project: bool) {
        let (title, ext) = if project { ("Landscape project", "landscape") } else { ("Noise graph", "noisegraph") };
        let Some(path) = rfd::FileDialog::new().add_filter(title, &[ext]).set_file_name(format!("terrain.{ext}")).save_file() else { return };
        let text = if project { self.to_project() } else { self.graph.to_json() };
        self.status = match std::fs::write(&path, text) {
            Ok(()) => {
                self.path = Some(path.clone());
                format!("Saved {}", path.display())
            }
            Err(e) => format!("could not save: {e}"),
        };
    }
}

// The tab's colours: the earth tone of the tab itself, and one for each part of the panel.
const TERRAIN: egui::Color32 = egui::Color32::from_rgb(200, 170, 115);
const MASK_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 135, 85);
const WATER_COLOR: egui::Color32 = egui::Color32::from_rgb(95, 165, 230);
const CHUNK_COLOR: egui::Color32 = egui::Color32::from_rgb(115, 205, 150);

/// A collapsible section of the side panel: an icon and a title in the section's colour.
fn section(ui: &mut egui::Ui, key: &str, icon: &str, title: &str, color: egui::Color32, open: bool, body: impl FnOnce(&mut egui::Ui)) {
    egui_extras::install_image_loaders(ui.ctx());
    let id = ui.make_persistent_id(("landscape section", key));
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, open)
        .show_header(ui, |ui| {
            ui.add(crate::icons::image(icon, 16.0, color));
            ui.label(egui::RichText::new(title).strong().color(color));
        })
        .body(body);
}

/// A button with an icon in front of its text (the icon in `tint` while enabled).
fn action(ui: &mut egui::Ui, icon: &str, text: &str, tint: egui::Color32, enabled: bool) -> egui::Response {
    egui_extras::install_image_loaders(ui.ctx());
    let tint = if enabled { tint } else { ui.visuals().weak_text_color() };
    ui.add_enabled(enabled, egui::Button::image_and_text(crate::icons::image(icon, 16.0, tint), text))
}

/// The 3D / Map switch in the corner of the top view (drawn last so nothing covers it).
fn view_toggle(ui: &mut egui::Ui, rect: egui::Rect, map_mode: &mut bool) {
    ui.put(egui::Rect::from_min_size(rect.min + egui::vec2(8.0, 6.0), egui::vec2(60.0, 26.0)), |ui: &mut egui::Ui| {
        ui.horizontal(|ui| {
            if crate::icons::tinted_button(ui, "view_3d", "3D view", Some(!*map_mode), true, Some(TERRAIN)).on_hover_text("The terrain in 3D: drag to orbit, scroll to zoom").clicked() {
                *map_mode = false;
            }
            if crate::icons::tinted_button(ui, "map", "Map view", Some(*map_mode), true, Some(MASK_COLOR)).on_hover_text("The terrain from above, for painting masks").clicked() {
                *map_mode = true;
            }
        })
        .response
    });
}

impl LandscapeWorkspace {
    /// The map view: the terrain from above with the selected mask in red over it, and the brush.
    fn map_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, rect: egui::Rect, response: &egui::Response, settings: &Settings) {
        let side = rect.width().min(rect.height());
        let map = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(24));
        let whole = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        if let Some(values) = &self.values {
            let eroded_key = self.eroded.as_ref().map(|(k, _)| *k).filter(|k| *k == self.erosion_key());
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (self.sampled_key, eroded_key, settings.world_m.to_bits(), settings.height_m.to_bits(), settings.fit).hash(&mut hasher);
            let key = hasher.finish();
            if self.map_picture.as_ref().is_none_or(|(k, _)| *k != key) {
                let metres = match (&self.eroded, eroded_key) {
                    (Some((_, e)), Some(_)) => e.clone(),
                    _ => to_metres(values, settings),
                };
                let (grid, m) = shrink(&metres, side_of(metres.len()), 512);
                self.map_picture = Some((key, ctx.load_texture("landscape map", hillshade(&grid, m, settings.world_m), egui::TextureOptions::LINEAR)));
            }
            if let Some((_, tex)) = &self.map_picture {
                ui.painter().image(tex.id(), map, whole, egui::Color32::WHITE);
            }
        }
        if let Some(mask) = self.masks.get(self.mask_selected) {
            let key = (self.mask_live, self.mask_selected);
            if self.overlay.as_ref().is_none_or(|(k, _)| *k != key) {
                const N: usize = 512;
                let res = crate::codegen::MASK_RES;
                let pixels = (0..N * N).map(|k| bilinear(&mask.data, res, (k % N) as f32 / (N - 1) as f32, (k / N) as f32 / (N - 1) as f32)).map(|v| egui::Color32::from_rgba_unmultiplied(255, 60, 40, (v * 170.0) as u8)).collect();
                let image = egui::ColorImage { size: [N, N], source_size: egui::vec2(N as f32, N as f32), pixels };
                self.overlay = Some((key, ctx.load_texture("landscape mask", image, egui::TextureOptions::LINEAR)));
            }
            if let Some((_, tex)) = &self.overlay {
                ui.painter().image(tex.id(), map, whole, egui::Color32::WHITE);
            }
            // Painting: Shift erases.
            let tool = if ui.input(|i| i.modifiers.shift) { Tool::Erase } else { self.tool };
            let norm = |p: egui::Pos2| ((p.x - map.left()) / side, (p.y - map.top()) / side);
            let inside = |n: (f32, f32)| (0.0..=1.0).contains(&n.0) && (0.0..=1.0).contains(&n.1);
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(n) = response.interact_pointer_pos().map(norm).filter(|n| inside(*n)) {
                    let from = self.stroke_from.unwrap_or(n);
                    let mask = &mut self.masks[self.mask_selected];
                    mask.stroke(from, n, &self.brush, tool);
                    self.stroke_from = Some(n);
                    self.mask_live += 1;
                }
            } else if response.clicked_by(egui::PointerButton::Primary) {
                if let Some(n) = response.interact_pointer_pos().map(norm).filter(|n| inside(*n)) {
                    self.masks[self.mask_selected].dab(n.0, n.1, &self.brush, tool);
                    (self.mask_live, self.mask_rev) = (self.mask_live + 1, self.mask_rev + 1);
                }
            }
            if response.drag_stopped() && self.stroke_from.take().is_some() {
                self.mask_rev += 1;
            }
            if let Some(p) = response.hover_pos() {
                ui.painter().circle_stroke(p, self.brush.radius * side, egui::Stroke::new(1.5, egui::Color32::WHITE));
                self.brush.radius = (self.brush.radius * (1.0 + ui.input(|i| i.smooth_scroll_delta.y) * 0.002)).clamp(0.002, 0.25);
            }
        } else {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Add a mask in the Masks panel, then paint here", egui::FontId::proportional(14.0), egui::Color32::GRAY);
        }
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
        if let Ok(mut names) = MASK_NAMES.lock() {
            *names = ls.masks.iter().map(|m| m.name.clone()).collect();
        }
        ls.editor.landscape_mode = true;
        // Sample the preview again when the graph, the seed or a mask changed and nothing else is running.
        let key = ls.key();
        if ls.job.is_none() && ls.sampled_key != Some(key) {
            ls.sampled_key = Some(key);
            let (graph, settings, masks, base) = (ls.graph.clone(), ls.settings.clone(), ls.mask_map(), ls.base_dir());
            let resolution = settings.erosion.resolution;
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let r = build_evaluator(&device, &queue, &graph, &settings, &masks, &base).map(|ev| sample(&ev, resolution, &Progress::default()));
                let _ = tx.send(r);
            });
            ls.job = Some(Job::Preview(rx));
        }

        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let ls = &mut self.landscape;
        egui::Panel::right("landscape props").default_size(320.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                if crate::icons::button(ui, "open", "Open", None, true).on_hover_text("Open a landscape project (graph, settings and masks) or a noise graph as the terrain").clicked() {
                    ls.open();
                }
                if crate::icons::button(ui, "save", "Save", None, true).on_hover_text("Save the project: the graph, the settings and the painted masks, in one file").clicked() {
                    ls.save(true);
                }
                if crate::icons::button(ui, "graph", "Save graph", None, true).on_hover_text("Save the graph alone as a noise graph file").clicked() {
                    ls.save(false);
                }
                if crate::icons::button(ui, "new", "New", None, true).on_hover_text("A new landscape: the starting graph and no masks").clicked() {
                    ls.graph = default_graph();
                    ls.path = None;
                    ls.masks.clear();
                    ls.mask_rev += 1;
                }
            });
            ui.separator();
            let running = matches!(ls.job, Some(Job::Export(..) | Job::Chunks(..) | Job::Erode(..)));
            section(ui, "terrain", "height", "Terrain", TERRAIN, true, |ui| {
            let d = Settings::default();
            crate::row(ui, &mut ls.settings.seed, d.seed, |ui, v| _ = ui.add(egui::DragValue::new(v).prefix("seed ")));
            crate::row(ui, &mut ls.settings.world_m, d.world_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(10.0..=100000.0).speed(10.0).prefix("size ").suffix(" m")).on_hover_text("Side of the square terrain."));
            crate::row(ui, &mut ls.settings.height_m, d.height_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=20000.0).speed(1.0).prefix("height ").suffix(" m")));
            crate::row(ui, &mut ls.settings.fit, d.fit, |ui, v| _ = ui.checkbox(v, "fit the height range").on_hover_text("On: the lowest point is 0 and the highest is the height. Off: the graph's output times the height, around 0."));
            ui.weak("The terrain is one tile of the graph (x and y from 0 to 1) with the density output as height, so any noise graph works. Tiling graphs make a terrain whose edges match.");
            if action(ui, "export", "Export heightmap for Unreal…", TERRAIN, !running && ls.values.is_some()).on_hover_text(format!("A 16-bit PNG at {UNREAL_RES} x {UNREAL_RES} (a valid landscape size) and a .json with the import settings.")).clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("16-bit PNG", &["png"]).set_file_name("heightmap.png").save_file() {
                let (graph, s) = (ls.graph.clone(), ls.settings.clone());
                let (masks, base) = (ls.mask_map(), ls.base_dir());
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
                        None => build_evaluator(&device, &queue, &graph, &s, &masks, &base).and_then(|ev| {
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
            });
            section(ui, "masks", "masks", "Masks", MASK_COLOR, true, |ui| {
                ui.weak("Paint masks on the map (the top view's Map button), then read them in the graph with a Mask node: shape a canyon or a ridge, or choose where a noise graph applies. Shift erases; scroll changes the brush size.");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut ls.new_mask).hint_text("new mask").desired_width(120.0));
                    let name = ls.new_mask.trim().to_string();
                    if action(ui, "add", "Add", MASK_COLOR, !name.is_empty() && !ls.masks.iter().any(|m| m.name == name)).on_hover_text("Add a mask with this name").clicked() {
                        ls.masks.push(Mask::new(&name));
                        (ls.mask_selected, ls.map_mode) = (ls.masks.len() - 1, true);
                        ls.mask_live += 1;
                        ls.new_mask.clear();
                    }
                });
                let mut pick = None;
                for (i, m) in ls.masks.iter().enumerate() {
                    if ui.selectable_label(i == ls.mask_selected, &m.name).clicked() {
                        pick = Some(i);
                    }
                }
                if let Some(i) = pick {
                    (ls.mask_selected, ls.map_mode) = (i, true);
                    ls.mask_live += 1;
                }
                let (mut rename, mut delete, mut edited) = (None, false, false);
                let others: Vec<String> = ls.masks.iter().enumerate().filter(|(i, _)| *i != ls.mask_selected).map(|(_, m)| m.name.clone()).collect();
                if let Some(m) = ls.masks.get_mut(ls.mask_selected) {
                    ui.horizontal(|ui| {
                        let old = m.name.clone();
                        let mut name = m.name.clone();
                        if ui.add(egui::TextEdit::singleline(&mut name).desired_width(120.0)).changed() && !name.is_empty() && !others.contains(&name) {
                            rename = Some((old, name.clone()));
                            m.name = name;
                        }
                        if crate::icons::button(ui, "delete", "Delete mask", None, true).on_hover_text("Delete the selected mask").clicked() {
                            delete = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        if crate::icons::tinted_button(ui, "clear_mask", "Clear", None, true, Some(MASK_COLOR)).on_hover_text("Empty the mask").clicked() {
                            m.fill(0.0);
                            edited = true;
                        }
                        if crate::icons::tinted_button(ui, "fill", "Fill", None, true, Some(MASK_COLOR)).on_hover_text("Fill the whole mask").clicked() {
                            m.fill(1.0);
                            edited = true;
                        }
                        if crate::icons::tinted_button(ui, "invert", "Invert", None, true, Some(MASK_COLOR)).on_hover_text("Swap painted and empty").clicked() {
                            m.invert();
                            edited = true;
                        }
                        if crate::icons::tinted_button(ui, "blur", "Blur", None, true, Some(MASK_COLOR)).on_hover_text("Soften the whole mask (a few metres at a time on a 2 km terrain)").clicked() {
                            m.blur(6);
                            edited = true;
                        }
                    });
                }
                if let Some((old, new)) = rename {
                    // Mask nodes that read the old name follow the rename.
                    for node in ls.graph.nodes.values_mut() {
                        if let NodeKind::Mask { name } = &mut node.kind {
                            if *name == old {
                                *name = new.clone();
                            }
                        }
                    }
                }
                if delete && ls.mask_selected < ls.masks.len() {
                    ls.masks.remove(ls.mask_selected);
                    ls.mask_selected = ls.mask_selected.saturating_sub(1);
                    edited = true;
                }
                if edited {
                    ls.mask_rev += 1;
                    ls.mask_live += 1;
                }
                ui.horizontal(|ui| {
                    for (tool, icon, name, hint) in [(Tool::Paint, "brush", "Paint", "Paint the mask in"), (Tool::Erase, "eraser", "Erase", "Take the mask out (Shift does this while held)"), (Tool::Smooth, "smooth", "Smooth", "Soften the mask under the brush")] {
                        if crate::icons::tinted_button(ui, icon, name, Some(ls.tool == tool), true, Some(MASK_COLOR)).on_hover_text(format!("{name}: {hint}")).clicked() {
                            ls.tool = tool;
                        }
                    }
                });
                let world = ls.settings.world_m;
                let mut radius_m = ls.brush.radius * world;
                if ui.add(egui::DragValue::new(&mut radius_m).range(1.0..=world * 0.5).speed(1.0).prefix("radius ").suffix(" m")).changed() {
                    ls.brush.radius = radius_m / world.max(1.0);
                }
                ui.add(egui::DragValue::new(&mut ls.brush.strength).range(0.01..=1.0).speed(0.01).prefix("strength "));
                ui.add(egui::DragValue::new(&mut ls.brush.hardness).range(0.0..=1.0).speed(0.01).prefix("hardness "));
            });
            section(ui, "erosion", "erosion", "Erosion", WATER_COLOR, true, |ui| {
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
                    if action(ui, "play", "Run erosion", WATER_COLOR, !running && ls.values.is_some()).on_hover_text("Erodes the terrain at the working resolution. The preview and both exports use the result until the graph or a setting changes.").clicked() {
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
                    if crate::icons::button(ui, "delete", "Clear erosion", None, ls.eroded.is_some()).on_hover_text("Drop the eroded terrain and use the plain graph again").clicked() {
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
            section(ui, "chunks", "chunks", "Nanite chunks", CHUNK_COLOR, true, |ui| {
                let d = crate::landscape_chunks::Options::default();
                let o = &mut ls.chunks;
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.text_edit_singleline(&mut o.name);
                });
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut o.folder).hint_text("folder").desired_width(190.0));
                    if crate::icons::button(ui, "open", "Browse", None, true).on_hover_text("Choose the folder").clicked() && let Some(f) = rfd::FileDialog::new().pick_folder() {
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
                if action(ui, "chunks", "Export chunks", CHUNK_COLOR, !running && !o.folder.is_empty()).on_hover_text("Writes every chunk's mesh (centimetres, the chunk's corner at its origin), its normal, occlusion and curvature maps, and a manifest with each chunk's location.").clicked() {
                    let (graph, s, opts) = (ls.graph.clone(), ls.settings.clone(), o.clone());
                    let (masks, base) = (ls.mask_map(), ls.base_dir());
                    let eroded = ls.current_eroded().cloned();
                    let (tx, rx) = mpsc::channel();
                    let (progress, log) = (Progress::default(), std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new())));
                    let (p, l) = (progress.clone(), log.clone());
                    let (device, queue) = (device.clone(), queue.clone());
                    std::thread::spawn(move || {
                        let r = build_evaluator(&device, &queue, &graph, &s, &masks, &base).and_then(|ev| {
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
            if ls.map_mode {
                ls.map_ui(ui, &ctx, rect, &response, &settings);
                view_toggle(ui, rect, &mut ls.map_mode);
                return;
            }
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
            view_toggle(ui, rect, &mut ls.map_mode);
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
    fn a_project_keeps_the_graph_the_settings_and_the_masks() {
        let mut a = LandscapeWorkspace::default();
        a.settings.world_m = 3000.0;
        let mut m = Mask::new("canyon");
        m.dab(0.4, 0.4, &Brush::default(), Tool::Paint);
        a.masks.push(m);
        let mut b = LandscapeWorkspace::default();
        b.from_project(&a.to_project()).unwrap();
        assert_eq!((b.graph.to_json(), b.settings.world_m, b.masks.len(), b.masks[0].name.as_str()), (a.graph.to_json(), 3000.0, 1, "canyon"));
        assert!(a.masks[0].data.iter().zip(&b.masks[0].data).all(|(x, y)| (x - y).abs() < 2e-5));
        assert!(b.from_project("{\"nothing\": 1}").is_err());
    }

    #[test]
    fn every_icon_the_tab_draws_exists() {
        for name in ["height", "masks", "erosion", "chunks", "graph", "map", "view_3d", "brush", "eraser", "smooth", "blur", "invert", "fill", "clear_mask", "play", "delete", "add", "export", "open", "save", "new", "landscape"] {
            assert!(crate::icons::source(name).is_some(), "no icon called {name}");
        }
    }

    #[test]
    fn a_valid_unreal_size() {
        // 64 components of 63 quads (LandscapeConfigHelper.cpp lists 7, 15, 31, 63, 127, 255 quads per section).
        assert_eq!(UNREAL_RES, 63 * 64 + 1);
    }
}
