//! The Stamp node of the Landscape graph: an `.sdf` shape pressed into the terrain. The shape is traced from above (along
//! its -Y, since `.sdf` files are Y-up and the terrain is Z-up) into a height and a coverage map over the terrain's
//! working grid, at the node's place, size and rotation. The graph prepares it like the other grid nodes (see
//! `landscape_terrain`): the node becomes Map reads and a little arithmetic: Add stacks the shape on the input, Max
//! lets it rise out of the ground where it covers, Cut carves it out. Caves and overhangs stay in the Mesh graph.
//! See docs/terrain-maps-plan.md.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StampMode {
    /// The shape's height is added to the terrain: it stands on the ground.
    #[default]
    Add,
    /// Where the shape covers, the terrain is at least as high as the shape's own height.
    Max,
    /// The shape's height is taken away: a crater or a quarry.
    Cut,
}

impl StampMode {
    pub const ALL: [StampMode; 3] = [StampMode::Add, StampMode::Max, StampMode::Cut];

    pub fn label(self) -> &'static str {
        match self {
            StampMode::Add => "Add (stands on the ground)",
            StampMode::Max => "Max (rises out of the ground)",
            StampMode::Cut => "Cut (carves into the ground)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StampParams {
    /// The `.sdf` file: relative to the project's folder, or absolute.
    pub file: String,
    /// The widest side of the shape's box in metres (the shape is scaled to it).
    pub size_m: f32,
    /// Stretches the shape's height (1 keeps its proportions).
    pub height_factor: f32,
    /// Where the middle of the shape sits on the terrain (x and y across it, 0 to 1).
    pub center: [f32; 2],
    /// Turns the shape about the vertical axis, in degrees.
    pub rotation_deg: f32,
    pub mode: StampMode,
}

impl Default for StampParams {
    fn default() -> Self {
        StampParams { file: String::new(), size_m: 200.0, height_factor: 1.0, center: [0.5, 0.5], rotation_deg: 0.0, mode: StampMode::Add }
    }
}

/// A shape traced onto the terrain grid: heights in metres above the shape's lowest point, and 1 where the shape is.
#[derive(Debug)]
pub struct Baked {
    pub side: usize,
    pub height_m: Vec<f32>,
    pub cover: Vec<f32>,
}

fn resolve(base: &Path, file: &str) -> PathBuf {
    let p = Path::new(file);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

/// Traces the shape onto a `side` x `side` grid over a terrain `world_m` across.
pub fn bake_uncached(p: &StampParams, base: &Path, side: usize, world_m: f32) -> Result<Baked, String> {
    use rayon::prelude::*;
    if p.file.trim().is_empty() {
        return Err("no .sdf file chosen".into());
    }
    let sdf = crate::sdf_file::SdfFile::load(&resolve(base, &p.file))?;
    let fields = crate::sdf_model::layer_fields(&sdf);
    let scene = crate::sdf_model::file_scene(&fields, None);
    let (lo, hi) = scene.bounds.ok_or("the .sdf file has nothing to draw")?;
    let dist = |q: [f64; 3]| fields.iter().filter(|f| !f.hidden).filter_map(|f| f.distance(q)).fold(f64::MAX, f64::min);
    let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let widest = ext.iter().cloned().fold(0.0, f64::max).max(1e-9);
    // Metres per unit of the file.
    let k = p.size_m.max(0.01) as f64 / widest;
    let centre = [(lo[0] + hi[0]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let half_diag = 0.5 * (ext[0] * ext[0] + ext[2] * ext[2]).sqrt() * k;
    let (cx, cy) = (p.center[0] as f64 * world_m as f64, p.center[1] as f64 * world_m as f64);
    let (sin, cos) = (p.rotation_deg as f64).to_radians().sin_cos();
    let cell = world_m as f64 / (side - 1) as f64;
    let eps = widest * 1e-3;
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..side)
        .into_par_iter()
        .map(|j| {
            let mut height = vec![0.0f32; side];
            let mut cover = vec![0.0f32; side];
            let y = j as f64 * cell - cy;
            if y.abs() <= half_diag + cell {
                for i in 0..side {
                    let x = i as f64 * cell - cx;
                    if x.abs() > half_diag + cell {
                        continue;
                    }
                    // Into the shape's frame: turn back, divide by the scale; terrain y runs to -z so nothing mirrors.
                    let (rx, ry) = (x * cos + y * sin, -x * sin + y * cos);
                    let (sx, sz) = (centre[0] + rx / k, centre[1] - ry / k);
                    if sx < lo[0] - eps || sx > hi[0] + eps || sz < lo[2] - eps || sz > hi[2] + eps {
                        continue;
                    }
                    // Down from the top of the box until the surface.
                    let mut t = 0.0;
                    let top = hi[1] + eps;
                    let depth = top - lo[1] + eps;
                    for _ in 0..160 {
                        let d = dist([sx, top - t, sz]);
                        if d < eps {
                            height[i] = (((top - t) - lo[1]).max(0.0) * k) as f32;
                            cover[i] = 1.0;
                            break;
                        }
                        t += d.max(eps * 0.5);
                        if t > depth {
                            break;
                        }
                    }
                }
            }
            (height, cover)
        })
        .collect();
    let (mut height_m, mut cover) = (Vec::with_capacity(side * side), Vec::with_capacity(side * side));
    for (h, c) in rows {
        height_m.extend(h);
        cover.extend(c);
    }
    let factor = p.height_factor;
    height_m.iter_mut().for_each(|h| *h *= factor);
    Ok(Baked { side, height_m, cover })
}

/// The last few bakes, so the preview does not trace the shape again for every edit elsewhere in the graph.
static CACHE: OnceLock<Mutex<Vec<(u64, Arc<Baked>)>>> = OnceLock::new();

/// The key of a bake: the settings, the file's stamp (size and time), the grid and the terrain's size.
fn key(p: &StampParams, base: &Path, side: usize, world_m: f32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (serde_json::to_string(p).unwrap_or_default(), side, world_m.to_bits()).hash(&mut h);
    if let Ok(m) = std::fs::metadata(resolve(base, &p.file)) {
        (m.len(), m.modified().ok()).hash(&mut h);
    }
    h.finish()
}

/// `bake_uncached`, remembering the last 8.
pub fn bake(p: &StampParams, base: &Path, side: usize, world_m: f32) -> Result<(u64, Arc<Baked>), String> {
    let key = key(p, base, side, world_m);
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Some(b) = cache.lock().map_err(|_| "the stamp cache is unusable")?.iter().find(|(k, _)| *k == key) {
        return Ok((key, b.1.clone()));
    }
    let baked = Arc::new(bake_uncached(p, base, side, world_m)?);
    let mut list = cache.lock().map_err(|_| "the stamp cache is unusable")?;
    list.push((key, baked.clone()));
    if list.len() > 8 {
        list.remove(0);
    }
    Ok((key, baked))
}

/// A stamp of the `.sdf` files the graph's Stamp nodes read (size and modified time), so an edited shape is seen.
pub fn files_stamp(graph: &crate::graph::Graph, base: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for n in graph.nodes.values() {
        if let crate::graph::NodeKind::Stamp(p) = &n.kind
            && let Ok(m) = std::fs::metadata(resolve(base, &p.file)) {
            (m.len(), m.modified().ok()).hash(&mut h);
        }
    }
    h.finish()
}

/// A stamp's settings.
pub fn node_ui(ui: &mut eframe::egui::Ui, p: &mut StampParams) {
    use eframe::egui;
    ui.weak("An .sdf shape pressed into the terrain, seen from above: its height over its lowest point, at this place, size and turn. Wire the terrain's height in. Caves and overhangs belong in the Mesh graph.");
    crate::meshgraph_ui::path_row(ui, "stamp path", &mut p.file, "shape .sdf", Some(("SDF shape", &["sdf"])));
    let d = StampParams::default();
    crate::row(ui, &mut p.mode, d.mode, |ui, v| {
        egui::ComboBox::from_id_salt("stamp mode").selected_text(v.label()).show_ui(ui, |ui| {
            for m in StampMode::ALL {
                ui.selectable_value(v, m, m.label());
            }
        });
    });
    crate::row(ui, &mut p.size_m, d.size_m, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.1..=100_000.0).speed(1.0).prefix("size ").suffix(" m")).on_hover_text("The widest side of the shape's box."));
    crate::row(ui, &mut p.height_factor, d.height_factor, |ui, v| _ = ui.add(egui::DragValue::new(v).range(0.0..=100.0).speed(0.01).prefix("height × ")).on_hover_text("1 keeps the shape's proportions; more makes it taller."));
    crate::row(ui, &mut p.center, d.center, |ui, v| {
        ui.add(egui::DragValue::new(&mut v[0]).range(-0.5..=1.5).speed(0.002).prefix("x "));
        ui.add(egui::DragValue::new(&mut v[1]).range(-0.5..=1.5).speed(0.002).prefix("y "));
    });
    crate::row(ui, &mut p.rotation_deg, d.rotation_deg, |ui, v| _ = ui.add(egui::DragValue::new(v).speed(0.5).suffix("°").prefix("turn ")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mushroom() -> StampParams {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("SDFs").join("Mushroom.sdf");
        StampParams { file: path.to_string_lossy().into_owned(), size_m: 100.0, ..Default::default() }
    }

    #[test]
    fn a_shape_is_traced_into_a_footprint_of_the_right_size_at_the_right_place() {
        let (side, world) = (101usize, 1000.0f32);
        let p = StampParams { center: [0.25, 0.75], ..mushroom() };
        let b = bake_uncached(&p, Path::new("."), side, world).unwrap();
        let covered: Vec<usize> = (0..side * side).filter(|k| b.cover[*k] > 0.5).collect();
        assert!(!covered.is_empty(), "the shape lands somewhere");
        let (xs, ys): (Vec<usize>, Vec<usize>) = (covered.iter().map(|k| k % side).collect(), covered.iter().map(|k| k / side).collect());
        let (x0, x1, y0, y1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap(), *ys.iter().min().unwrap(), *ys.iter().max().unwrap());
        let (cx, cy) = ((x0 + x1) as f32 / 2.0 / (side - 1) as f32, (y0 + y1) as f32 / 2.0 / (side - 1) as f32);
        assert!((cx - 0.25).abs() < 0.03 && (cy - 0.75).abs() < 0.03, "centred at ({cx}, {cy})");
        // No side is wider than the size asked for (100 m on a 10 m grid: 10 cells, plus a cell of slack).
        assert!(x1 - x0 <= 11 && y1 - y0 <= 11, "{} x {} cells", x1 - x0 + 1, y1 - y0 + 1);
        assert!(b.height_m.iter().cloned().fold(0.0, f32::max) > 1.0, "it has height");
        assert!(b.height_m.iter().zip(&b.cover).all(|(h, c)| *c > 0.5 || *h == 0.0), "no height outside the footprint");
    }

    #[test]
    fn the_height_factor_stretches_and_rotation_moves_an_off_centre_shape() {
        let (side, world) = (101usize, 1000.0f32);
        let flat = bake_uncached(&mushroom(), Path::new("."), side, world).unwrap();
        let tall = bake_uncached(&StampParams { height_factor: 2.0, ..mushroom() }, Path::new("."), side, world).unwrap();
        let top = |b: &Baked| b.height_m.iter().cloned().fold(0.0, f32::max);
        assert!((top(&tall) / top(&flat) - 2.0).abs() < 0.01);
        assert_eq!(flat.cover, tall.cover);
        // A missing file is an error.
        assert!(bake_uncached(&StampParams { file: "gone.sdf".into(), ..mushroom() }, Path::new("."), side, world).is_err());
    }

    #[test]
    fn the_cache_returns_the_same_bake_until_the_settings_change() {
        let p = mushroom();
        let (k1, a) = bake(&p, Path::new("."), 41, 500.0).unwrap();
        let (k2, b) = bake(&p, Path::new("."), 41, 500.0).unwrap();
        assert!(k1 == k2 && Arc::ptr_eq(&a, &b));
        let (k3, _) = bake(&StampParams { size_m: 50.0, ..p }, Path::new("."), 41, 500.0).unwrap();
        assert_ne!(k1, k3);
    }
}
