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
    /// The height of water (a river, a lake, the sea) in metres: drawn in the previews, written into the heightmap's sidecar and
    /// the chunk manifest, and made into a water surface by the Unreal import script.
    pub water_m: Option<f32>,
    /// The working grid's side in samples: the preview samples the graph on it and Erosion nodes erode on it.
    pub resolution: usize,
    /// Erosion settings from before erosion was a node: read from old projects (see `migrate_erosion`), never written.
    #[serde(rename = "erosion", skip_serializing)]
    pub legacy_erosion: Option<serde_json::Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { seed: 0, world_m: 2000.0, height_m: 400.0, fit: true, water_m: None, resolution: 1024, legacy_erosion: None }
    }
}

/// The names of the masks, for the Mask node's menu (set by the Landscape tab every frame).
static MASK_NAMES: std::sync::Mutex<Vec<(String, crate::terrain_maps::Source)>> = std::sync::Mutex::new(Vec::new());

pub fn mask_names() -> Vec<(String, crate::terrain_maps::Source)> {
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

/// Samples a side of the grid the 3D view draws: the GPU marches it, so it can be finer than a CPU picture could.
const VIEW_SAMPLES: usize = 1025;

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
    pub water_m: Option<f32>,
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
    UnrealHeights { samples, z_scale, centre_cm, step_cm, range_m: (lo, hi), water_m: None }
}

/// `encode_unreal`, with the height of the water to write into the sidecar.
pub fn encode_unreal_with(metres: &[f32], water_m: Option<f32>) -> UnrealHeights {
    UnrealHeights { water_m, ..encode_unreal(metres) }
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
        "water_level_m": h.water_m,
        "unreal_import": {
            "location_cm": [0.0, 0.0, h.centre_cm],
            "scale": [xy, xy, h.z_scale],
            "note": "Landscape mode > Import from File: this heightmap, this resolution; set the scale and location above. 16 bits cover the height range in steps of step_cm."
        }
    });
    std::fs::write(&side, serde_json::to_string_pretty(&json).unwrap_or_default()).map_err(|e| format!("{}: {e}", side.display()))?;
    Ok(format!("Wrote {} ({n} x {n}, {:.2} cm per step) and {}", path.display(), h.step_cm, side.file_name().unwrap_or_default().to_string_lossy()))
}

/// Where a ray `eye + t·d` (t ≥ 0) is inside the box `lo..hi`: `(t_in, t_out)`, or `None` when it misses.
#[cfg(test)]
fn clip_ray(eye: [f32; 3], d: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> Option<(f32, f32)> {
    let (mut t_in, mut t_out) = (0.0f32, f32::MAX);
    for a in 0..3 {
        if d[a].abs() < 1e-9 {
            if eye[a] < lo[a] || eye[a] > hi[a] {
                return None;
            }
            continue;
        }
        let (t0, t1) = ((lo[a] - eye[a]) / d[a], (hi[a] - eye[a]) / d[a]);
        t_in = t_in.max(t0.min(t1));
        t_out = t_out.min(t0.max(t1));
    }
    (t_in <= t_out).then_some((t_in, t_out))
}

/// The terrain from a camera by ray marching the height grid on the CPU: metres in, a picture out. The view draws on
/// the GPU (`landscape_gpu`); this is the reference the tests hold it to. `min_step_cells` is the smallest step; without
/// `slope_steps` every step is the smallest one (for checking that the big steps miss nothing).
#[cfg(test)]
fn render_steps(metres: &[f32], n: usize, world_m: f32, water: Option<f32>, (yaw, pitch, zoom): (f32, f32, f32), (w, h): (usize, usize), min_step_cells: f32, slope_steps: bool) -> egui::ColorImage {
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
    // Water: a flat surface at its level wherever the ground is below it.
    let level = water.unwrap_or(f32::MIN);
    let surface = |x: f32, y: f32| at(x, y).max(level);
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
    // The box the surface (ground or water) lives in, and its steepest slope (metres up per metre across).
    let (floor, top) = (lo.min(level.max(lo)) - 1.0, hi.max(level) + 1.0);
    let steepest = (0..n * n)
        .map(|k| {
            let (i, j) = (k % n, k / n);
            let h = metres[k].max(level);
            let dx = if i + 1 < n { (metres[k + 1].max(level) - h).abs() } else { 0.0 };
            let dy = if j + 1 < n { (metres[k + n].max(level) - h).abs() } else { 0.0 };
            dx.max(dy)
        })
        .fold(0.0f32, f32::max)
        / cell
        * 1.5; // diagonal and interpolation margin
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
            // March only where the ray is inside the terrain's box (sky pixels and rays past the edge cost nothing),
            // until it dips below the surface.
            let Some((t_in, t_out)) = clip_ray(eye, d, [0.0, 0.0, floor], [world_m, world_m, top]) else { continue };
            let min_step = cell * min_step_cells;
            // The surface rises at most `steepest` metres per metre, so a ray `gap` above it can safely go
            // `gap / (|dz| + steepest * horizontal)` before it could touch it: big steps high up, small ones near it.
            let closing = if slope_steps { d[2].abs() + steepest * (d[0] * d[0] + d[1] * d[1]).sqrt() } else { f32::INFINITY };
            let (mut t, mut prev) = (t_in, t_in);
            while t <= t_out {
                let p = [eye[0] + d[0] * t, eye[1] + d[1] * t, eye[2] + d[2] * t];
                let gap = p[2] - surface(p[0], p[1]);
                {
                    if gap < 0.0 {
                        // Refine between the last two steps.
                        let (mut a, mut b) = (prev, t);
                        for _ in 0..5 {
                            let m = (a + b) * 0.5;
                            let q = [eye[0] + d[0] * m, eye[1] + d[1] * m, eye[2] + d[2] * m];
                            if q[2] < surface(q[0], q[1]) { b = m } else { a = m }
                        }
                        let q = [eye[0] + d[0] * b, eye[1] + d[1] * b, eye[2] + d[2] * b];
                        let (gx, gy) = ((surface(q[0] + cell, q[1]) - surface(q[0] - cell, q[1])) / (2.0 * cell), (surface(q[0], q[1] + cell) - surface(q[0], q[1] - cell)) / (2.0 * cell));
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
                        let base = if at(q[0], q[1]) < level { [45.0, 95.0, 165.0] } else { base };
                        *out = egui::Color32::from_rgb((base[0] * shade).min(255.0) as u8, (base[1] * shade).min(255.0) as u8, (base[2] * shade).min(255.0) as u8);
                        break;
                    }
                }
                prev = t;
                t += (gap / closing.max(1e-6)).max(min_step);
            }
        }
    });
    egui::ColorImage { size: [w, h], source_size: egui::vec2(w as f32, h as f32), pixels }
}

/// Projects from before erosion was a node kept its settings (and the working resolution) in `settings.erosion`, and
/// eroded only when Run was pressed. Their resolution moves to `settings.resolution`, and when their graph has no Erosion
/// node, one with those settings goes in front of the output, muted: the terrain looks as the plain graph did, and
/// unmuting it erodes as before.
fn migrate_erosion(graph: &mut Graph, settings: &mut Settings) {
    let Some(old) = settings.legacy_erosion.take() else { return };
    if let Some(r) = old.get("resolution").and_then(|r| r.as_u64()) {
        settings.resolution = r as usize;
    }
    if graph.nodes.values().any(|n| matches!(n.kind, NodeKind::Erosion(_))) {
        return;
    }
    let Some((out, out_node)) = graph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutDensity)).map(|(id, n)| (*id, n.pos)) else { return };
    let Some((src, port)) = graph.edges.iter().find(|e| e.to == out && e.port == 0).map(|e| (e.from, e.from_port)) else { return };
    let params: crate::landscape_erosion::Erosion = serde_json::from_value(old).unwrap_or_default();
    let node = graph.add(NodeKind::Erosion(params), [out_node[0] - 160.0, out_node[1]]);
    graph.nodes.get_mut(&node).expect("just added").muted = true;
    graph.connect_from(src, port, node, 0);
    graph.connect(node, out, 0);
}

/// The evaluator for the terrain graph with its Erosion nodes run (or their cached results used) and replaced by the
/// maps they made: what the preview, the exports and the headless export all evaluate. Returns how each Erosion node's
/// result stood.
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_evaluator(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    graph: &Graph,
    s: &Settings,
    maps: &crate::terrain_maps::TerrainMaps,
    maps_key: u64,
    base: &Path,
    cache: &crate::landscape_terrain::ErosionCache,
    run: &std::collections::BTreeSet<crate::graph::NodeId>,
    progress: &Progress,
) -> Result<(Evaluator, std::collections::HashMap<crate::graph::NodeId, crate::landscape_terrain::State>), String> {
    let (ev, prepared) = terrain_parts(device, queue, graph, s, maps, maps_key, base, cache, run, progress)?;
    Ok((ev, prepared.states))
}

/// `terrain_evaluator`, with the prepared graph and maps too (the layers are sampled from them).
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_parts(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    graph: &Graph,
    s: &Settings,
    maps: &crate::terrain_maps::TerrainMaps,
    maps_key: u64,
    base: &Path,
    cache: &crate::landscape_terrain::ErosionCache,
    run: &std::collections::BTreeSet<crate::graph::NodeId>,
    progress: &Progress,
) -> Result<(Evaluator, crate::landscape_terrain::Prepared), String> {
    let prepared = crate::landscape_terrain::prepare(device, queue, graph, s, maps, maps_key, base, cache, run, progress)?;
    Ok((build_evaluator(device, queue, &prepared.graph, s, &prepared.maps, base)?, prepared))
}

/// Writes the Unreal heightmap at `path` (the Erosion nodes run or come from the cache), and one weightmap per Layer
/// node beside it, listed in the sidecar. Shared by the app's export and the headless one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_terrain(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    graph: &Graph,
    s: &Settings,
    maps: &crate::terrain_maps::TerrainMaps,
    maps_key: u64,
    base: &Path,
    cache: &crate::landscape_terrain::ErosionCache,
    run: &std::collections::BTreeSet<crate::graph::NodeId>,
    progress: &Progress,
    path: &Path,
) -> Result<String, String> {
    let (ev, prepared) = terrain_parts(device, queue, graph, s, maps, maps_key, base, cache, run, progress)?;
    let metres = to_metres(&sample(&ev, UNREAL_RES, progress), s);
    let mut message = write_unreal(path, UNREAL_RES, s.world_m, &encode_unreal_with(&metres, s.water_m))?;
    if crate::landscape_layers::has_layers(&prepared.graph) {
        let layers = crate::landscape_layers::sample_layers(device, queue, &prepared, s, base, UNREAL_RES, progress)?;
        let files = crate::landscape_layers::write_weightmaps(path, UNREAL_RES, &layers)?;
        add_layers_to_sidecar(path, &files)?;
        message += &format!("; {} weightmaps ({})", files.len(), files.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));
    }
    Ok(message)
}

/// Adds the weightmaps (layer name, file name) to the heightmap's sidecar.
fn add_layers_to_sidecar(heightmap: &Path, files: &[(String, String)]) -> Result<(), String> {
    let side = PathBuf::from(format!("{}.json", heightmap.display()));
    let text = std::fs::read_to_string(&side).map_err(|e| format!("{}: {e}", side.display()))?;
    let mut json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    json["layers"] = files.iter().map(|(name, file)| serde_json::json!({ "name": name, "file": file })).collect();
    json["layers_note"] = "8-bit grayscale weightmaps, one per layer, same size as the heightmap; they add up to 255 in every cell. Landscape mode > Import from File > Paint layers: one file per layer.".into();
    std::fs::write(&side, serde_json::to_string_pretty(&json).unwrap_or_default()).map_err(|e| format!("{}: {e}", side.display()))
}

/// The evaluator for the terrain: Noise Graph nodes inlined, the slope scale set from the size and height, the terrain
/// maps in.
pub(crate) fn build_evaluator(device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue, graph: &Graph, s: &Settings, masks: &crate::terrain_maps::TerrainMaps, base: &Path) -> Result<Evaluator, String> {
    let mut g = crate::meshgraph::lowered_mesh(graph, base)?;
    for n in g.nodes.values_mut() {
        if let NodeKind::Slope { scale } = &mut n.kind {
            *scale = s.height_m / s.world_m.max(1.0);
        }
    }
    Evaluator::with_maps(device, queue, &g, s.seed, masks)
}

/// The map: the terrain from above, lit from the north-west (`m` x `m` metres, `world_m` across).
fn hillshade(metres: &[f32], m: usize, world_m: f32, water: Option<f32>) -> egui::ColorImage {
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
            if water.is_some_and(|w| at(x, y) < w) {
                return egui::Color32::from_rgb((v * 0.3) as u8, (v * 0.55) as u8, (v * 0.95) as u8);
            }
            egui::Color32::from_rgb(v as u8, (v * 0.97) as u8, (v * 0.9) as u8)
        })
        .collect();
    egui::ColorImage { size: [m, m], source_size: egui::vec2(m as f32, m as f32), pixels }
}

/// RGBA8 colours on an `n` x `n` grid, taken (nearest) onto an `m` x `m` grid across the same area.
fn resample_colours(rgba: &[u8], n: usize, m: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * m * 4);
    for y in 0..m {
        for x in 0..m {
            let (sx, sy) = ((x * (n - 1) + (m - 1) / 2) / (m - 1).max(1), (y * (n - 1) + (m - 1) / 2) / (m - 1).max(1));
            let k = (sy.min(n - 1) * n + sx.min(n - 1)) * 4;
            out.extend_from_slice(&rgba[k..k + 4]);
        }
    }
    out
}

thread_local! {
    /// The height setting in metres, for node panels that show what an image becomes (set each frame by the tab).
    pub(crate) static HEIGHT_FOR_UI: std::cell::Cell<f32> = const { std::cell::Cell::new(400.0) };
}

/// What a finished job hands back.
enum Done {
    /// The sampled values and how each Erosion node's result stood.
    Values(Vec<f32>, std::collections::HashMap<crate::graph::NodeId, crate::landscape_terrain::State>, Option<Vec<crate::landscape_layers::Layer>>),
    Message(String),
}

/// A background job: sampling the graph (preview or export) and what to do with the result.
enum Job {
    /// The graph's values on the working grid (Erosion nodes run first when they need to), with its progress.
    Preview(mpsc::Receiver<Result<Done, String>>, Progress),
    Export(mpsc::Receiver<Result<String, String>>, Progress),
}

pub struct LandscapeWorkspace {
    pub graph: Graph,
    pub editor: Editor,
    pub path: Option<PathBuf>,
    pub settings: Settings,
    /// The graph's values on the working grid, and the key (graph, seed, resolution) they were sampled for.
    values: Option<Vec<f32>>,
    sampled_key: Option<u64>,
    /// Erosion nodes' results, kept between runs, how each stood after the last one, and those to run on demand.
    erosion_cache: crate::landscape_terrain::ErosionCache,
    erosion_states: std::collections::HashMap<crate::graph::NodeId, crate::landscape_terrain::State>,
    erosion_run: std::collections::BTreeSet<crate::graph::NodeId>,
    /// The paint layers' weights on the working grid (when the graph has Layer nodes), and a counter for the 3D view.
    layers: Option<Vec<crate::landscape_layers::Layer>>,
    layers_rev: u64,
    /// A change waiting to be sampled, and when it was made (sampling waits a moment when there is erosion).
    pending: Option<(u64, std::time::Instant)>,
    job: Option<Job>,
    status: String,
    /// A heightmap being exported for the Mesh graph: when it is written, a graph for it opens there.
    to_mesh: Option<PathBuf>,
    view: (f32, f32, f32),
    /// The heights the 3D view draws (at most VIEW_SAMPLES a side, in metres), kept and uploaded once per terrain.
    view_grid: Option<crate::landscape_gpu::Grid>,
    /// Undo and redo of the graph.
    history: crate::history::History,
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
        LandscapeWorkspace { graph: default_graph(), editor: Editor::default(), path: None, settings: Settings::default(), values: None, sampled_key: None, erosion_cache: Default::default(), erosion_states: Default::default(), erosion_run: Default::default(), layers: None, layers_rev: 0, pending: None, job: None, status: String::new(), to_mesh: None, view: (0.8, 0.6, 1.0), view_grid: None, history: crate::history::History::new(&default_graph()), masks: Vec::new(), mask_selected: 0, mask_rev: 0, mask_live: 0, brush: Brush::default(), tool: Tool::Paint, map_mode: false, stroke_from: None, new_mask: String::new(), overlay: None, map_picture: None }
    }
}

impl LandscapeWorkspace {
    /// Writes the heightmap (the eroded terrain when there is one) to `path` on a thread.
    fn start_export(&mut self, path: PathBuf, device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue) {
        let ls = self;
        let (graph, s) = (ls.graph.clone(), ls.settings.clone());
        let (masks, base) = (ls.mask_map(), ls.base_dir());
        let (maps_key, cache, run) = (ls.mask_rev, ls.erosion_cache.clone(), ls.erosion_run.clone());
        let (tx, rx) = mpsc::channel();
        let progress = Progress::default();
        let p = progress.clone();
        let (device, queue) = (device.clone(), queue.clone());
        std::thread::spawn(move || {
            // Erosion nodes run (or come from the cache) at the working resolution; what follows them is sampled at
            // the Unreal size.
            let r = export_terrain(&device, &queue, &graph, &s, &masks, maps_key, &base, &cache, &run, &p, &path);
            let _ = tx.send(r);
        });
        ls.job = Some(Job::Export(rx, progress));
        ls.status = "Exporting…".into();
    }

    pub fn restore(&mut self, graph: Graph, settings: Settings, path: Option<PathBuf>) {
        (self.graph, self.settings, self.path) = (graph, settings, path);
        self.history = crate::history::History::new(&self.graph);
    }

    /// Undo (or redo) the last settled change to the graph.
    pub fn undo(&mut self, redo: bool) {
        let g = if redo { self.history.redo(&self.graph) } else { self.history.undo(&self.graph) };
        if let Some(g) = g {
            self.graph = g;
            self.editor.selected.retain(|id| self.graph.nodes.contains_key(id));
        }
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Applies an erosion preset file to the selected Erosion node (the library's double-click).
    pub fn apply_erosion_preset(&mut self, path: &Path) {
        let name = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        let selected: Vec<_> = self.editor.selected.iter().copied().collect();
        let Some(id) = selected.into_iter().find(|id| self.graph.nodes.get(id).is_some_and(|n| matches!(n.kind, NodeKind::Erosion(_)))) else {
            self.status = format!("Select an Erosion node to apply the preset {name} to");
            return;
        };
        let Some(NodeKind::Erosion(e)) = self.graph.nodes.get_mut(&id).map(|n| &mut n.kind) else { unreachable!() };
        match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|text| e.with_preset(&text, &name)) {
            Ok(loaded) => {
                *e = loaded;
                self.status = format!("Erosion preset {name} applied");
            }
            Err(err) => self.status = format!("could not load the preset {}: {err}", path.display()),
        }
    }

    /// Lays the graph out in columns, inputs on the left.
    pub fn arrange(&mut self) {
        crate::editor::arrange(&mut self.graph);
        self.editor.frame_next = true;
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
        let mut settings: Settings = v.get("settings").and_then(|s| serde_json::from_value(s.clone()).ok()).unwrap_or_default();
        let masks = v.get("masks").and_then(|m| m.as_array()).map(|a| a.iter().filter_map(|m| Mask::from_png(m.get("name")?.as_str()?, &crate::landscape_masks::base64_decode(m.get("png")?.as_str()?)?)).collect()).unwrap_or_default();
        let mut graph = graph;
        migrate_erosion(&mut graph, &mut settings);
        (self.graph, self.settings, self.masks) = (graph, settings, masks);
        (self.mask_selected, self.mask_rev) = (0, self.mask_rev + 1);
        self.history = crate::history::History::new(&self.graph);
        Ok(())
    }

    /// The terrain maps the graph can read: the painted masks (and, as later phases add them, the other maps).
    fn mask_map(&self) -> crate::terrain_maps::TerrainMaps {
        let mut maps = crate::terrain_maps::TerrainMaps::default();
        for m in &self.masks {
            maps.insert(m.name.clone(), crate::codegen::MASK_RES, std::sync::Arc::new(m.data.clone()), crate::terrain_maps::Source::Painted);
        }
        maps
    }

    /// Where a Noise Graph node's relative file names are looked for: the project's folder.
    fn base_dir(&self) -> PathBuf {
        self.path.as_ref().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
    }

    /// What the sampled values depend on: the graph, the seed, the working resolution, the masks, Erosion nodes asked
    /// to run, and (when there is erosion, which works in metres) the terrain's size and height.
    fn key(&self) -> u64 {
        let s = &self.settings;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.graph.to_json(), s.seed, s.resolution, self.mask_rev, &self.erosion_run, crate::landscape_heightmap::files_stamp(&self.graph, &self.base_dir())).hash(&mut h);
        if crate::landscape_terrain::has_erosion(&self.graph) {
            (s.world_m.to_bits(), s.height_m.to_bits()).hash(&mut h);
        }
        h.finish()
    }

    /// A landscape project (graph, settings, masks) or a plain noise graph (the masks stay).
    pub fn open(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Landscape project or noise graph", &["landscape", "noisegraph", "json"]).pick_file() else { return };
        self.open_path(path);
    }

    /// Opens a landscape project, or a noise graph as the terrain.
    pub fn open_path(&mut self, path: PathBuf) {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => return self.status = format!("could not open: {e}"),
        };
        let result = if path.extension().is_some_and(|e| e == "landscape") {
            self.from_project(&text)
        } else {
            Graph::from_json(&text).map(|g| {
                self.history = crate::history::History::new(&g);
                self.graph = g;
            })
        };
        match result {
            Ok(()) => self.path = Some(path),
            Err(e) => self.status = format!("could not open: {e}"),
        }
    }

    /// A new landscape: the starting graph and no masks.
    pub fn new_document(&mut self) {
        self.graph = default_graph();
        self.history = crate::history::History::new(&self.graph);
        self.path = None;
        self.masks.clear();
        self.mask_rev += 1;
    }

    /// Saves the project (graph, settings and masks in one file): to the project it came from, or (with `ask`, or when
    /// it has no project file yet) to one the user picks.
    pub fn save_project(&mut self, ask: bool) {
        match self.path.clone().filter(|p| !ask && p.extension().is_some_and(|e| e == "landscape")) {
            Some(path) => self.write(&path, true),
            None => self.save(true),
        }
    }

    /// Saves the graph alone as a noise graph file the user picks.
    pub fn save_graph(&mut self) {
        self.save(false);
    }

    fn save(&mut self, project: bool) {
        let (title, ext) = if project { ("Landscape project", "landscape") } else { ("Noise graph", "noisegraph") };
        let Some(path) = rfd::FileDialog::new().add_filter(title, &[ext]).set_file_name(format!("terrain.{ext}")).save_file() else { return };
        self.write(&path, project);
    }

    fn write(&mut self, path: &Path, project: bool) {
        let path = path.to_path_buf();
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

fn ls_default_water() -> f32 {
    5.0
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
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (self.sampled_key, settings.world_m.to_bits(), settings.height_m.to_bits(), settings.fit).hash(&mut hasher);
            let key = hasher.finish();
            if self.map_picture.as_ref().is_none_or(|(k, _)| *k != key) {
                let metres = to_metres(values, settings);
                let (grid, m) = shrink(&metres, side_of(metres.len()), 512);
                self.map_picture = Some((key, ctx.load_texture("landscape map", hillshade(&grid, m, settings.world_m, settings.water_m), egui::TextureOptions::LINEAR)));
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
    /// The File and Edit menus while the Landscape tab is shown: they act on the landscape project.
    pub(crate) fn landscape_menus(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            if crate::icons::menu_item(ui, "new", "New landscape", "Ctrl+N").clicked() {
                self.landscape.new_document();
            }
            if crate::icons::menu_item(ui, "open", "Open…", "Ctrl+O").on_hover_text("A landscape project, or a noise graph as the terrain").clicked() {
                self.landscape.open();
            }
            if crate::icons::menu_item(ui, "save", "Save", "Ctrl+S").on_hover_text("The project: the graph, the settings and the painted masks, in one file").clicked() {
                self.landscape.save_project(false);
            }
            if crate::icons::menu_item(ui, "save_as", "Save As…", "Ctrl+Shift+S").clicked() {
                self.landscape.save_project(true);
            }
            if crate::icons::menu_item(ui, "graph", "Save graph as noise graph…", "").clicked() {
                self.landscape.save_graph();
            }
        });
        ui.menu_button("Edit", |ui| {
            if crate::icons::menu_item_enabled(ui, "undo", "Undo", "Ctrl+Z", self.landscape.can_undo()).clicked() {
                self.landscape.undo(false);
            }
            if crate::icons::menu_item_enabled(ui, "redo", "Redo", "Ctrl+Y", self.landscape.can_redo()).clicked() {
                self.landscape.undo(true);
            }
            ui.separator();
            if crate::icons::menu_item(ui, "arrange", "Arrange nodes", "").on_hover_text("Lay the graph out in columns, inputs on the left").clicked() {
                self.landscape.arrange();
            }
            if crate::icons::menu_item(ui, "settings", "Preferences…", "").clicked() {
                self.prefs_open = true;
                ui.close();
            }
        });
    }

    pub(crate) fn landscape_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // New, open and save act on the landscape here.
        let (save_as, save, open, new) = ui.input(|i| (crate::keymap::pressed(i, crate::keymap::Action::SaveAs), crate::keymap::pressed(i, crate::keymap::Action::Save), crate::keymap::pressed(i, crate::keymap::Action::Open), crate::keymap::pressed(i, crate::keymap::Action::New)));
        if save_as {
            self.landscape.save_project(true);
        } else if save {
            self.landscape.save_project(false);
        } else if open {
            self.landscape.open();
        } else if new {
            self.landscape.new_document();
        }
        // Undo and redo of the graph, unless a text field has the keyboard.
        if !ctx.egui_wants_keyboard_input() {
            let (undo, redo) = ui.input(|i| (crate::keymap::pressed(i, crate::keymap::Action::Undo), crate::keymap::pressed(i, crate::keymap::Action::Redo)));
            if redo {
                self.landscape.undo(true);
            } else if undo {
                self.landscape.undo(false);
            }
        }
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let ls = &mut self.landscape;

        // The background job: its result.
        let mut finished = None;
        match &ls.job {
            Some(Job::Preview(rx, _)) => match rx.try_recv() {
                Ok(r) => finished = Some(r),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the sampler stopped unexpectedly".into())),
            },
            Some(Job::Export(rx, _)) => match rx.try_recv() {
                Ok(r) => finished = Some(r.map(Done::Message)),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the export stopped unexpectedly".into())),
            },
            None => {}
        }
        let mut to_mesh = None;
        if let Some(r) = finished {
            ls.job = None;
            if r.is_err() {
                ls.to_mesh = None;
            }
            match r {
                Ok(Done::Values(v, states, layers)) => {
                    ls.values = Some(v);
                    ls.layers = layers;
                    ls.layers_rev += 1;
                    if states.values().any(|s| *s == crate::landscape_terrain::State::Ran) {
                        ls.status = "Erosion done".into();
                    }
                    ls.erosion_states = states;
                    ls.erosion_run.clear();
                }
                Ok(Done::Message(m)) => {
                    ls.status = m;
                    to_mesh = ls.to_mesh.take();
                }
                Err(e) => {
                    ls.status = e;
                    ls.values = None;
                }
            }
        }
        if let Some(png) = to_mesh {
            self.mesh_ws.open_graph(crate::landscape_field::terrain_graph(&png, self.landscape.settings.world_m), png.with_file_name("terrain.meshgraph"));
            self.workspace = crate::meshworkspace::Workspace::Mesh;
            self.landscape.status = format!("Opened a mesh graph for {} in the Mesh graph tab", png.display());
            return;
        }
        let ls = &mut self.landscape;
        if let Ok(mut names) = MASK_NAMES.lock() {
            *names = ls.masks.iter().map(|m| (m.name.clone(), crate::terrain_maps::Source::Painted)).collect();
        }
        ls.editor.landscape_mode = true;
        let working = matches!(ls.job, Some(Job::Preview(..)));
        ls.editor.badges = ls.graph.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::Erosion(_))).map(|(id, _)| {
            let text = match ls.erosion_states.get(id) {
                _ if working => "running…",
                Some(crate::landscape_terrain::State::OutOfDate) => "out of date",
                Some(_) => "up to date",
                None => "not run yet",
            };
            (*id, text.to_string())
        }).collect();
        // Sample the preview again when the graph, the seed or a mask changed and nothing else is running. With erosion
        // in the graph (seconds of work) it waits until the graph has been still for a moment.
        let key = ls.key();
        if ls.sampled_key != Some(key) && ls.pending.is_none_or(|(k, _)| k != key) {
            ls.pending = Some((key, std::time::Instant::now()));
        }
        let settled = ls.pending.is_some_and(|(_, t)| !crate::landscape_terrain::has_erosion(&ls.graph) || t.elapsed() > std::time::Duration::from_millis(400));
        if ls.sampled_key != Some(key) && !settled {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if ls.job.is_none() && ls.sampled_key != Some(key) && settled {
            ls.sampled_key = Some(key);
            ls.pending = None;
            let (graph, settings, masks, base) = (ls.graph.clone(), ls.settings.clone(), ls.mask_map(), ls.base_dir());
            let (maps_key, cache, run) = (ls.mask_rev, ls.erosion_cache.clone(), ls.erosion_run.clone());
            let (tx, rx) = mpsc::channel();
            let progress = Progress::default();
            let p = progress.clone();
            std::thread::spawn(move || {
                let r = terrain_parts(&device, &queue, &graph, &settings, &masks, maps_key, &base, &cache, &run, &p).and_then(|(ev, prepared)| {
                    let values = sample(&ev, settings.resolution, &Progress::default());
                    let layers = if crate::landscape_layers::has_layers(&prepared.graph) {
                        Some(crate::landscape_layers::sample_layers(&device, &queue, &prepared, &settings, &base, settings.resolution, &Progress::default())?)
                    } else {
                        None
                    };
                    Ok(Done::Values(values, prepared.states, layers))
                });
                let _ = tx.send(r);
            });
            ls.job = Some(Job::Preview(rx, progress));
        }

        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let format = self.rs.target_format;
        let ls = &mut self.landscape;
        egui::Panel::right("landscape props").default_size(320.0).show(ui, |ui| egui::ScrollArea::vertical().id_salt("landscape props scroll").auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal(|ui| {
                if crate::icons::button(ui, "open", "Open", None, true).on_hover_text("Open a landscape project (graph, settings and masks) or a noise graph as the terrain").clicked() {
                    ls.open();
                }
                if crate::icons::button(ui, "save", "Save", None, true).on_hover_text("Save the project (Ctrl+S): the graph, the settings and the painted masks, in one file").clicked() {
                    ls.save_project(false);
                }
                if crate::icons::button(ui, "graph", "Save graph", None, true).on_hover_text("Save the graph alone as a noise graph file").clicked() {
                    ls.save_graph();
                }
                if crate::icons::button(ui, "new", "New", None, true).on_hover_text("A new landscape (Ctrl+N): the starting graph and no masks").clicked() {
                    ls.new_document();
                }
            });
            ui.separator();
            let running = matches!(ls.job, Some(Job::Export(..)));
            section(ui, "terrain", "height", "Terrain", TERRAIN, true, |ui| {
            let d = Settings::default();
            crate::row(ui, &mut ls.settings.seed, d.seed, |ui, v| _ = ui.add(egui::DragValue::new(v).prefix("seed ")));
            crate::row(ui, &mut ls.settings.world_m, d.world_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(10.0..=100000.0).speed(10.0).prefix("size ").suffix(" m")).on_hover_text("Side of the square terrain."));
            crate::row(ui, &mut ls.settings.height_m, d.height_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=20000.0).speed(1.0).prefix("height ").suffix(" m")));
            crate::row(ui, &mut ls.settings.water_m, d.water_m, |ui, v| {
                ui.horizontal(|ui| {
                    let mut on = v.is_some();
                    if ui.checkbox(&mut on, "water at").on_hover_text("A river, lake or sea: the ground below this height is water. It is drawn blue here and in the map, and the Unreal import script lays a water surface at this height.").changed() {
                        *v = on.then_some(ls_default_water());
                    }
                    if let Some(w) = v {
                        ui.add(egui::DragValue::new(w).speed(0.1).suffix(" m"));
                    }
                });
            });
            crate::row(ui, &mut ls.settings.fit, d.fit, |ui, v| _ = ui.checkbox(v, "fit the height range").on_hover_text("On: the lowest point is 0 and the highest is the height. Off: the graph's output times the height, around 0."));
            ui.weak("The terrain is one tile of the graph (x and y from 0 to 1) with the density output as height, so any noise graph works. Tiling graphs make a terrain whose edges match.");
            if action(ui, "export", "Export heightmap for Unreal…", TERRAIN, !running && ls.values.is_some()).on_hover_text(format!("A 16-bit PNG at {UNREAL_RES} x {UNREAL_RES} (a valid landscape size) and a .json with the import settings.")).clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("16-bit PNG", &["png"]).set_file_name("heightmap.png").save_file() {
                ls.start_export(path, &device, &queue);
            }
            if action(ui, "mesh_graph", "Open as Nanite chunks in the Mesh graph…", TERRAIN, !running && ls.values.is_some()).on_hover_text("Exports the heightmap into a folder you choose and opens a Mesh graph there that cuts it into Nanite chunks (planar UVs, normal / occlusion / curvature maps). Press Run in the Mesh graph tab; terrain_chunks.json and tools/unreal_import_terrain.py bring the result into Unreal.").clicked()
                && let Some(dir) = rfd::FileDialog::new().pick_folder() {
                let png = dir.join("heightmap.png");
                ls.to_mesh = Some(png.clone());
                ls.start_export(png, &device, &queue);
            }
            if let Some(Job::Export(_, p)) = &ls.job {
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
            section(ui, "grid", "erosion", "Working grid", WATER_COLOR, true, |ui| {
                let max = crate::landscape_erosion::max_resolution(&device);
                let default = Settings::default().resolution;
                crate::row(ui, &mut ls.settings.resolution, default, |ui, v| {
                    for r in crate::landscape_erosion::RESOLUTIONS.into_iter().filter(|r| *r <= max) {
                        ui.selectable_value(v, r, format!("{r}"));
                    }
                });
                ui.weak("The grid the preview samples the graph on and Erosion nodes erode on. Erosion is a node: add one in the graph (right-click, Erosion).");
                if let Some(Job::Preview(_, p)) = &ls.job
                    && crate::landscape_terrain::has_erosion(&ls.graph) {
                        ui.add(egui::ProgressBar::new(p.load(Ordering::Relaxed) as f32 / 1000.0).show_percentage().text("eroding"));
                    }
            });
            if !ls.status.is_empty() {
                ui.label(&ls.status);
            }
            ui.separator();
            // The selected node's settings, as in the noise graph tab.
            let names = ls.graph.declared_names();
            let base_for_ui = ls.base_dir();
            if let [id] = ls.editor.selected.iter().copied().collect::<Vec<_>>().as_slice()
                && let Some(node) = ls.graph.nodes.get_mut(id) {
                ui.heading(crate::editor::title(&node.kind));
                if let NodeKind::Erosion(e) = &node.kind {
                    let state = ls.erosion_states.get(id).copied();
                    let working = matches!(ls.job, Some(Job::Preview(..)));
                    ui.horizontal(|ui| {
                        ui.label(match (state, working) {
                            (_, true) => "Running…",
                            (Some(crate::landscape_terrain::State::OutOfDate), _) => "Out of date",
                            (Some(_), _) => "Up to date",
                            (None, _) => "Not run yet",
                        });
                        if ui.add_enabled(!working, egui::Button::new("Run")).on_hover_text("Run this erosion now, even when it does not run by itself").clicked() {
                            ls.erosion_run.insert(*id);
                        }
                    });
                    if !e.auto {
                        ui.weak("Runs only when you press Run.");
                    }
                }
                HEIGHT_FOR_UI.with(|h| h.set(ls.settings.height_m));
                crate::meshgraph_ui::set_base(base_for_ui.clone());
                crate::node_props(ui, &mut node.kind, &mut node.pins, &names);
            }
        }));

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
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                (ls.sampled_key, ls.layers_rev, settings.world_m.to_bits(), settings.height_m.to_bits(), settings.fit, settings.water_m.map(f32::to_bits)).hash(&mut hasher);
                let grid_key = hasher.finish();
                // The terrain (with its erosion, which is part of the graph), at most VIEW_SAMPLES a side: worked out
                // (and uploaded to the GPU) once per terrain, not on every camera move.
                if ls.view_grid.as_ref().is_none_or(|g| g.key != grid_key) {
                    let metres = to_metres(values, &settings);
                    let (grid, m) = shrink(&metres, side(metres.len()), VIEW_SAMPLES);
                    let mut view_grid = crate::landscape_gpu::Grid::new(grid_key, grid, m, settings.world_m, settings.water_m);
                    if let Some(layers) = ls.layers.as_ref().filter(|l| !l.is_empty()) {
                        view_grid = view_grid.with_colours(resample_colours(&crate::landscape_layers::colours(layers), side(layers[0].weights.len()), m));
                    }
                    ls.view_grid = Some(view_grid);
                }
                let grid = ls.view_grid.as_ref().expect("just made");
                // Drawn on the GPU at the panel's full resolution, straight into egui's pass for this rectangle.
                let uniforms = crate::landscape_gpu::ViewUniforms::new(grid, settings.world_m, settings.water_m, new_view, rect.width() / rect.height().max(1.0), format.is_srgb(), crate::landscape_gpu::MIN_STEP_CELLS, true);
                let callback = crate::landscape_gpu::ViewCallback { uniforms, key: grid.key, heights: grid.heights.clone(), colours: grid.colours.clone(), n: grid.n, format };
                ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(rect, callback));
            } else {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(30));
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, if matches!(ls.job, Some(Job::Preview(..))) { "Sampling…" } else { "No terrain: the graph has no density output, or has an error" }, egui::FontId::proportional(14.0), egui::Color32::GRAY);
            }
            ui.painter().text(rect.left_bottom() + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "drag to orbit, scroll to zoom", egui::FontId::proportional(11.0), egui::Color32::from_white_alpha(120));
            view_toggle(ui, rect, &mut ls.map_mode);
        });
        ls.view = new_view;
        let tiling = ls.graph.tiling();
        ls.editor.show(ui, &mut ls.graph, &tiling);
        ls.graph.prune_edges();
        // One undo step per settled edit (a whole drag is one step).
        let settled = !ui.input(|i| i.pointer.any_down());
        ls.history.commit(&ls.graph, settled);
    }
}

/// `noiser --preview P.landscape OUT.png`: the terrain (its Erosion nodes run) as a map from above and as a 3D view side
/// by side, painted in the layers' colours when the graph has Layer nodes.
pub fn preview_project(project: &Path, size: u32) -> Result<image::RgbImage, String> {
    let text = std::fs::read_to_string(project).map_err(|e| format!("{}: {e}", project.display()))?;
    let mut ls = LandscapeWorkspace::default();
    ls.from_project(&text)?;
    ls.path = Some(project.to_path_buf());
    let (device, queue) = crate::meshgraph::headless_gpu()?;
    let (s, base, masks) = (ls.settings.clone(), ls.base_dir(), ls.mask_map());
    let run: std::collections::BTreeSet<_> = ls.graph.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::Erosion(_))).map(|(id, _)| *id).collect();
    let progress = Progress::default();
    let (ev, prepared) = terrain_parts(&device, &queue, &ls.graph, &s, &masks, ls.mask_rev, &base, &ls.erosion_cache, &run, &progress)?;
    let n = s.resolution.max(2);
    let metres = to_metres(&sample(&ev, n, &progress), &s);
    let layers = if crate::landscape_layers::has_layers(&prepared.graph) { Some(crate::landscape_layers::sample_layers(&device, &queue, &prepared, &s, &base, n, &progress)?) } else { None };
    let size = size.clamp(128, 2048);
    // The 3D view at the size's height; the map beside it.
    let (grid, m) = shrink(&metres, n, VIEW_SAMPLES);
    let mut view_grid = crate::landscape_gpu::Grid::new(1, grid, m, s.world_m, s.water_m);
    if let Some(layers) = layers.as_ref().filter(|l| !l.is_empty()) {
        view_grid = view_grid.with_colours(resample_colours(&crate::landscape_layers::colours(layers), n, m));
    }
    let uniforms = crate::landscape_gpu::ViewUniforms::new(&view_grid, s.world_m, s.water_m, (0.8, 0.6, 1.0), 1.0, false, crate::landscape_gpu::MIN_STEP_CELLS, true);
    let pixels = crate::landscape_gpu::render_image(&device, &queue, &view_grid, &uniforms, (size, size));
    let map = hillshade(&metres, n, s.world_m, s.water_m);
    let mut img = image::RgbImage::new(size * 2, size);
    for (k, p) in pixels.iter().enumerate() {
        img.put_pixel((k as u32) % size + size, (k as u32) / size, image::Rgb([p[0], p[1], p[2]]));
    }
    let tint = layers.as_ref().filter(|l| !l.is_empty()).map(|l| crate::landscape_layers::colours(l));
    for y in 0..size {
        for x in 0..size {
            let (sx, sy) = ((x as usize * (n - 1)) / (size as usize - 1), (y as usize * (n - 1)) / (size as usize - 1));
            let k = sy * n + sx;
            let c = map.pixels[k];
            let mut rgb = [c.r() as f32, c.g() as f32, c.b() as f32];
            if let Some(t) = &tint {
                // Hillshade brightness, layer colours as the hue.
                let lum = (rgb[0] + rgb[1] + rgb[2]) / (3.0 * 190.0);
                rgb = [t[k * 4] as f32 * lum, t[k * 4 + 1] as f32 * lum, t[k * 4 + 2] as f32 * lum];
            }
            img.put_pixel(x, y, image::Rgb(rgb.map(|v| v.clamp(0.0, 255.0) as u8)));
        }
    }
    Ok(img)
}

/// Exports a landscape project's heightmap for Unreal without a window (`noiser --headless --landscape-export P.landscape
/// OUT.png [--erode]`): the graph sampled at the Unreal size, its Erosion nodes run first. `erode` also switches on Erosion
/// nodes that are muted (as projects from before erosion was a node open), so their erosion is applied as it was then.
pub fn export_headless(project: &Path, out: &Path, erode: bool, log: &mut dyn FnMut(&str)) -> Result<String, String> {
    let text = std::fs::read_to_string(project).map_err(|e| format!("{}: {e}", project.display()))?;
    let mut ls = LandscapeWorkspace::default();
    if project.extension().is_some_and(|e| e == "landscape") {
        ls.from_project(&text)?;
    } else {
        ls.graph = Graph::from_json(&text)?;
    }
    ls.path = Some(project.to_path_buf());
    let (device, queue) = crate::meshgraph::headless_gpu()?;
    if erode {
        for n in ls.graph.nodes.values_mut().filter(|n| matches!(n.kind, NodeKind::Erosion(_))) {
            n.muted = false;
        }
    }
    let (s, masks, base) = (ls.settings.clone(), ls.mask_map(), ls.base_dir());
    // Every Erosion node runs: nothing is cached in a fresh process.
    let run: std::collections::BTreeSet<_> = ls.graph.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::Erosion(_))).map(|(id, _)| *id).collect();
    if !run.is_empty() {
        log(&format!("eroding on the {0} x {0} working grid ({1} Erosion node(s))", s.resolution, run.len()));
    }
    let progress = Progress::default();
    export_terrain(&device, &queue, &ls.graph, &s, &masks, ls.mask_rev, &base, &ls.erosion_cache, &run, &progress, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_old_project_gets_a_muted_erosion_node_with_its_settings() {
        let mut graph = default_graph();
        let mut settings = Settings { legacy_erosion: Some(serde_json::json!({"resolution": 2048, "iterations": 77})), ..Default::default() };
        migrate_erosion(&mut graph, &mut settings);
        assert_eq!(settings.resolution, 2048);
        let (id, node) = graph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::Erosion(_))).expect("an Erosion node");
        let NodeKind::Erosion(e) = &node.kind else { unreachable!() };
        assert_eq!(e.iterations, 77);
        assert!(node.muted, "muted: the terrain looks as it did");
        let out = graph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutDensity)).map(|(i, _)| *i).unwrap();
        assert!(graph.edges.iter().any(|e| e.to == out && e.port == 0 && e.from == *id), "in front of the output");
        // Migrating twice changes nothing.
        let count = graph.nodes.len();
        migrate_erosion(&mut graph, &mut settings);
        assert_eq!(graph.nodes.len(), count);
    }

    #[test]
    fn the_3d_view_s_big_steps_miss_nothing_the_small_ones_hit() {
        // Rolling hills with sharp spikes and a lake: thin peaks are what a too-big step would skip.
        let n = 129;
        let metres: Vec<f32> = (0..n * n)
            .map(|k| {
                let (x, y) = ((k % n) as f32 / (n - 1) as f32, (k / n) as f32 / (n - 1) as f32);
                let hills = 60.0 * (x * 9.0).sin() * (y * 7.0).cos() + 80.0;
                let spike = if (k % n) % 17 == 5 && (k / n) % 13 == 3 { 150.0 } else { 0.0 };
                hills + spike
            })
            .collect();
        for view in [(0.8, 0.6, 1.0), (2.4, 0.15, 2.5), (-1.0, 1.3, 0.4)] {
            for water in [None, Some(70.0)] {
                // Truth: a march in twentieths of a cell. Steps of 0.9 cell (the old fixed march) graze past some of the
                // thin spikes; the slope-safe steps must not miss more than that.
                let truth = render_steps(&metres, n, 1000.0, water, view, (160, 100), 0.05, false);
                let fixed = render_steps(&metres, n, 1000.0, water, view, (160, 100), 0.9, false);
                let fast = render_steps(&metres, n, 1000.0, water, view, (160, 100), 0.9, true);
                // A hit a fraction of a step off shades a touch differently; a missed peak shows sky or far ground.
                let far = |a: &egui::Color32, b: &egui::Color32| (0..3).any(|c| (a.to_array()[c] as i32 - b.to_array()[c] as i32).abs() > 40);
                let wrong = |image: &egui::ColorImage| image.pixels.iter().zip(&truth.pixels).filter(|(a, b)| far(a, b)).count();
                let (fixed_wrong, fast_wrong) = (wrong(&fixed), wrong(&fast));
                assert!(fast_wrong <= fixed_wrong + fast.pixels.len() / 400, "view {view:?}, water {water:?}: {fast_wrong} pixels wrong, the fixed march {fixed_wrong}");
                assert!(fast.pixels.iter().any(|p| p.b() < 200 && p.g() > p.b()), "some ground is drawn");
            }
        }
    }

    #[test]
    fn the_gpu_view_draws_what_the_cpu_renderer_draws() {
        let Some(gpu) = crate::gpu_tests::gpu() else { return eprintln!("no GPU adapter; skipping") };
        let n = 129;
        let metres: Vec<f32> = (0..n * n)
            .map(|k| {
                let (x, y) = ((k % n) as f32 / (n - 1) as f32, (k / n) as f32 / (n - 1) as f32);
                60.0 * (x * 9.0).sin() * (y * 7.0).cos() + 80.0 + 25.0 * (x * 23.0).cos() * (y * 19.0).sin()
            })
            .collect();
        let (w, h) = (160usize, 100usize);
        for view in [(0.8, 0.6, 1.0), (2.4, 0.15, 2.5), (-1.0, 1.3, 0.4)] {
            for water in [None, Some(70.0)] {
                let grid = crate::landscape_gpu::Grid::new(1, metres.clone(), n, 1000.0, water);
                let uniforms = crate::landscape_gpu::ViewUniforms::new(&grid, 1000.0, water, view, w as f32 / h as f32, false, crate::landscape_gpu::MIN_STEP_CELLS, true);
                let gpu_pixels = crate::landscape_gpu::render_image(&gpu.device, &gpu.queue, &grid, &uniforms, (w as u32, h as u32));
                let cpu = render_steps(&metres, n, 1000.0, water, view, (w, h), crate::landscape_gpu::MIN_STEP_CELLS, true);
                // Rounding (the CPU truncates to bytes) and single-precision differences shift a few silhouette pixels;
                // anything else would differ a lot.
                let far = gpu_pixels.iter().zip(&cpu.pixels).filter(|(g, c)| (0..3).any(|k| (g[k] as i32 - c.to_array()[k] as i32).abs() > 40)).count();
                assert!(far <= w * h / 100, "view {view:?}, water {water:?}: {far} of {} pixels differ", w * h);
                assert!(gpu_pixels.iter().any(|p| p[1] > p[2] && p[2] < 200), "some ground is drawn");
            }
        }
    }

    #[test]
    fn graph_edits_undo_and_redo_and_a_new_document_starts_fresh() {
        let mut ls = LandscapeWorkspace::default();
        let before = ls.graph.clone();
        ls.graph.add(crate::graph::NodeKind::Separate, [0.0; 2]);
        ls.history.commit(&ls.graph, true);
        let after = ls.graph.clone();
        assert!(ls.can_undo());
        ls.undo(false);
        assert!(ls.graph == before, "undo puts the graph back");
        ls.undo(true);
        assert!(ls.graph == after, "redo brings the edit back");
        ls.new_document();
        assert!(!ls.can_undo() && !ls.can_redo(), "a new landscape has no history");
    }

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
        for name in ["height", "masks", "erosion", "chunks", "graph", "map", "view_3d", "brush", "eraser", "smooth", "blur", "invert", "fill", "clear_mask", "play", "delete", "add", "export", "open", "save", "new", "landscape", "mesh_graph"] {
            assert!(crate::icons::source(name).is_some(), "no icon called {name}");
        }
    }

    #[test]
    fn the_eiffel_project_in_the_library_opens() {
        let path = std::path::Path::new("H:/My Drive/NoiserLibrary/Terrains/EiffelLandscape.landscape");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let mut ls = LandscapeWorkspace::default();
        ls.from_project(&text).unwrap();
        assert_eq!((ls.settings.world_m, ls.settings.water_m, ls.masks.len()), (2000.0, Some(6.5), 3));
    }

    #[test]
    fn a_valid_unreal_size() {
        // 64 components of 63 quads (LandscapeConfigHelper.cpp lists 7, 15, 31, 63, 127, 255 quads per section).
        assert_eq!(UNREAL_RES, 63 * 64 + 1);
    }
}
