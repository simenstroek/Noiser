//! The Mesh graph tab: its own graph and node editor, a toolbar (new, open, save, run) and a panel with the selected
//! node's settings and the log of the last run. Running happens on a worker thread; see `meshgraph::Runner`.

use crate::graph::{Graph, NodeKind};
use crate::keymap::Action;
use crate::meshgraph::{MeshCache, Runner};
use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

/// Which graph the window is showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Workspace {
    #[default]
    Noise,
    Mesh,
    Composer,
    Landscape,
}

impl Workspace {
    pub const ALL: [Workspace; 4] = [Workspace::Composer, Workspace::Noise, Workspace::Mesh, Workspace::Landscape];

    pub fn label(self) -> &'static str {
        match self {
            Workspace::Noise => "Noise graph",
            Workspace::Mesh => "Mesh graph",
            Workspace::Composer => "SDF Composer",
            Workspace::Landscape => "Landscape",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Workspace::Noise => "noise",
            Workspace::Mesh => "mesh_graph",
            Workspace::Composer => "shape_cube",
            Workspace::Landscape => "landscape",
        }
    }

    /// The colour of the tab and of the outline around the tab's contents.
    pub fn color(self) -> egui::Color32 {
        match self {
            Workspace::Noise => egui::Color32::from_rgb(90, 190, 120),
            Workspace::Mesh => egui::Color32::from_rgb(100, 150, 240),
            Workspace::Composer => egui::Color32::from_rgb(235, 150, 70),
            Workspace::Landscape => egui::Color32::from_rgb(190, 160, 110),
        }
    }
}

/// A picture request for the ray marching worker: the graph (and what it was made from), the view and the size.
struct RayRequest {
    graph: Graph,
    key: u64,
    base: PathBuf,
    seed: u32,
    view: [f32; 3],
    size: u32,
}

/// What the worker sends back: the picture (RGBA bytes) for a request, or why there is none.
struct RayResult {
    size: u32,
    rgba: Result<Vec<u8>, String>,
}

/// Draws the field of the first Remesh by ray marching on its own thread: it keeps the compiled field for as long as
/// the graph does not change and answers only the newest request.
struct RayWorker {
    tx: mpsc::Sender<RayRequest>,
    rx: mpsc::Receiver<RayResult>,
    /// Requests sent and not yet answered.
    outstanding: usize,
}

/// The orthographic camera that matches the software view (`Preview::paint`) for a field with these bounds.
fn ray_camera(lo: [f32; 3], hi: [f32; 3], view: [f32; 3], size: u32) -> crate::meshexport::MarchCamera {
    let [yaw, pitch, zoom] = view;
    let centre: [f32; 3] = std::array::from_fn(|a| 0.5 * (lo[a] + hi[a]));
    let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max).max(1e-9) * 0.5;
    let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());
    // The rows of the software view's rotation: image x, image y and towards the viewer, in the field's space.
    let (x, y, z) = ([cy, 0.0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]);
    let half = radius / (0.9 * zoom);
    let step = 2.0 * half / size as f32;
    let at = |a: usize| centre[a] + z[a] * 2.0 * radius + x[a] * (-half + 0.5 * step) + y[a] * (half - 0.5 * step);
    let light: [f32; 3] = std::array::from_fn(|a| x[a] * 0.35 + y[a] * 0.55 + z[a] * 0.75);
    let l = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt();
    crate::meshexport::MarchCamera {
        origin: [at(0), at(1), at(2)],
        step_x: x.map(|c| c * step),
        step_y: y.map(|c| -c * step),
        direction: z.map(|c| -c),
        light: light.map(|c| c / l),
        t_max: 4.0 * radius,
        hit: 0.0008 * radius,
        gradient_step: 0.002 * radius,
    }
}

impl RayWorker {
    fn spawn(device: eframe::wgpu::Device, queue: eframe::wgpu::Queue) -> RayWorker {
        let (tx, requests) = mpsc::channel::<RayRequest>();
        let (answers, rx) = mpsc::channel::<RayResult>();
        std::thread::spawn(move || {
            let mut cache = MeshCache::default();
            let mut current: Option<(u64, crate::meshgraph::FieldView)> = None;
            while let Ok(mut req) = requests.recv() {
                let mut skipped = 0;
                while let Ok(newer) = requests.try_recv() {
                    req = newer;
                    skipped += 1;
                }
                // Answer once per request so the sender can count, newest picture last.
                for _ in 0..skipped {
                    let _ = answers.send(RayResult { size: 0, rgba: Err(String::new()) });
                }
                if current.as_ref().map(|c| c.0) != Some(req.key) {
                    current = None;
                    let mut quiet = |_: &str| {};
                    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        Runner { device: &device, queue: &queue, cache: &mut cache, base: req.base.clone(), seed: req.seed, progress: Default::default(), log: &mut quiet, batch_file: None, max_triangles: None }.field_view(&req.graph)
                    }));
                    match built {
                        Ok(Ok(view)) => current = Some((req.key, view)),
                        Ok(Err(e)) => {
                            let _ = answers.send(RayResult { size: req.size, rgba: Err(e) });
                            continue;
                        }
                        Err(_) => {
                            let _ = answers.send(RayResult { size: req.size, rgba: Err("the field could not be built".into()) });
                            continue;
                        }
                    }
                }
                let Some((_, view)) = &current else { continue };
                let (lo, hi) = view.bounds();
                let cam = ray_camera(lo, hi, req.view, req.size);
                let pixels = view.march(&cam, req.size, req.size);
                let rgba = pixels.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)).collect();
                let _ = answers.send(RayResult { size: req.size, rgba: Ok(rgba) });
            }
        });
        RayWorker { tx, rx, outstanding: 0 }
    }
}

struct RunJob {
    progress: crate::meshexport::Progress,
    /// How many of the `total` files are finished (a batch run has more than one).
    finished: Arc<std::sync::atomic::AtomicUsize>,
    total: usize,
    log: Arc<Mutex<Vec<String>>>,
    done: mpsc::Receiver<Result<Vec<PathBuf>, String>>,
}

pub struct MeshWorkspace {
    pub graph: Graph,
    pub editor: crate::editor::Editor,
    pub path: Option<PathBuf>,
    /// The graph as last saved, to tell whether there is anything to save.
    saved: Graph,
    cache: Arc<Mutex<MeshCache>>,
    job: Option<RunJob>,
    log: Vec<String>,
    written: Vec<PathBuf>,
    pub seed: u32,
    /// How a batch ('Run on files') picks the seed of each mesh from `seed`.
    pub seed_mode: crate::meshexport::SeedMode,
    pub udim: crate::udim::Tool,
    /// Start a run on the first frame (for `--run-mesh-graph`).
    pub autorun: bool,
    /// The first mesh the last run wrote, for the view above the graph.
    preview: Option<Preview>,
    /// Show the preview's UV layout instead of the shaded mesh.
    pub show_uv: bool,
    /// The images the last run wrote, as thumbnails under the view.
    thumbnails: Vec<(String, egui::TextureHandle)>,
    /// The map files behind the thumbnails, the one shown full size (and its texture), and that view's zoom and pan.
    map_files: Vec<PathBuf>,
    pub show_map: bool,
    map_index: usize,
    map_tex: Option<(usize, egui::TextureHandle)>,
    map_zoom: f32,
    map_pan: egui::Vec2,
    /// The ray-marched view of the field beside the software view, its worker, last picture, last error, and what was
    /// last requested (graph key, view, size) with when the graph key last changed.
    pub field_view: bool,
    /// The mesh view's surface: 0 baked normal and occlusion lit (default), 1 smooth grey, 2 and up map k - 2 unlit.
    surface: usize,
    surface_tex: Option<egui::TextureHandle>,
    surface_key: Option<(i32, i32, i32, usize, usize)>,
    ray: Option<RayWorker>,
    ray_tex: Option<egui::TextureHandle>,
    ray_error: Option<String>,
    ray_sent: Option<(u64, [i32; 3], u32)>,
    ray_key: (u64, Option<std::time::Instant>),
    /// Redo the quick preview a moment after the graph (or the noise graph it uses) changes.
    pub auto_preview: bool,
    /// What the last preview was made from, and since when the graph has differed from it.
    preview_key: u64,
    changed_since: Option<std::time::Instant>,
    history: crate::history::History,
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

/// A mesh to look at: positions centred and scaled to a unit radius, with triangles.
struct Preview {
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
    name: String,
    /// The texture coordinates of each triangle's corners (V up) and the island each triangle is in; empty without UVs.
    uvs: Vec<[[f32; 2]; 3]>,
    island: Vec<u32>,
    /// Smooth vertex normals (area weighted), for shading.
    normals: Vec<[f32; 3]>,
    /// The baked maps of the run (RGB, at most 1024 px) for the textured views, and whether the normal map has green up.
    maps: Vec<MapImage>,
    green_up: bool,
}

/// A baked map kept in memory for the textured mesh view.
struct MapImage {
    name: String,
    size: [usize; 2],
    rgb: Vec<u8>,
}

impl MapImage {
    fn load(path: &std::path::Path) -> Option<MapImage> {
        let img = image::open(path).ok()?;
        let img = if img.width().max(img.height()) > 1024 { img.thumbnail(1024, 1024) } else { img }.to_rgb8();
        let stem = path.file_stem()?.to_string_lossy().into_owned();
        Some(MapImage { name: stem, size: [img.width() as usize, img.height() as usize], rgb: img.into_raw() })
    }

    /// Bilinear sample at (u, v) with v down, wrapping outside 0 to 1, as 0 to 1 floats.
    fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let (w, h) = (self.size[0], self.size[1]);
        let (x, y) = (u * w as f32 - 0.5, v * h as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let px = |i: f32, j: f32| {
            let (i, j) = ((i as i64).clamp(0, w as i64 - 1) as usize, (j as i64).clamp(0, h as i64 - 1) as usize);
            let at = (j * w + i) * 3;
            [self.rgb[at] as f32, self.rgb[at + 1] as f32, self.rgb[at + 2] as f32]
        };
        let (a, b, c, d) = (px(x0, y0), px(x0 + 1.0, y0), px(x0, y0 + 1.0), px(x0 + 1.0, y0 + 1.0));
        std::array::from_fn(|k| ((a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy) / 255.0)
    }

    fn is(&self, what: &str) -> bool {
        self.name.to_lowercase().contains(&what.to_lowercase())
    }
}

impl Preview {
    fn load(path: &std::path::Path) -> Result<Preview, String> {
        let mesh = crate::meshformats::load(path)?;
        let (lo, hi) = mesh.bounds();
        let centre = [0, 1, 2].map(|a| 0.5 * (lo[a] + hi[a]));
        let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max).max(1e-9) * 0.5;
        let positions: Vec<[f32; 3]> = mesh.positions.iter().map(|p| [0, 1, 2].map(|a| (p[a] - centre[a]) / radius)).collect();
        let island = if mesh.uvs.len() == mesh.triangles.len() && !mesh.uvs.is_empty() { crate::uvtransfer::islands(&mesh) } else { Vec::new() };
        let mut normals = vec![[0.0f32; 3]; positions.len()];
        for t in &mesh.triangles {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            for &i in t {
                for k in 0..3 {
                    normals[i as usize][k] += n[k];
                }
            }
        }
        for n in &mut normals {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
            *n = [n[0] / l, n[1] / l, n[2] / l];
        }
        Ok(Preview { normals, positions, triangles: mesh.triangles, name: path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()), uvs: if island.is_empty() { Vec::new() } else { mesh.uvs }, island, maps: Vec::new(), green_up: false })
    }

    /// The UV layout: the 0 to 1 square with every triangle, filled by island.
    fn paint_uv(&self, painter: &egui::Painter, rect: egui::Rect) {
        let side = (rect.width().min(rect.height()) - 24.0).max(50.0);
        let square = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
        painter.rect_filled(square, 0.0, egui::Color32::from_gray(40));
        let at = |uv: [f32; 2]| egui::pos2(square.left() + uv[0] * side, square.bottom() - uv[1] * side);
        let mut mesh = egui::epaint::Mesh::default();
        let mut lines = Vec::new();
        for (t, uv) in self.uvs.iter().enumerate() {
            let h = self.island[t].wrapping_mul(2_654_435_761);
            let colour = egui::Color32::from_rgb(90 + (h & 0x7f) as u8, 90 + ((h >> 8) & 0x7f) as u8, 90 + ((h >> 16) & 0x7f) as u8);
            let p = uv.map(at);
            let base = mesh.vertices.len() as u32;
            for q in p {
                mesh.vertices.push(egui::epaint::Vertex { pos: q, uv: egui::epaint::WHITE_UV, color: colour });
            }
            mesh.indices.extend([base, base + 1, base + 2]);
            lines.push(egui::Shape::closed_line(p.to_vec(), egui::Stroke::new(0.5, egui::Color32::from_black_alpha(140))));
        }
        painter.add(egui::Shape::mesh(mesh));
        if self.uvs.len() <= 60_000 {
            painter.extend(lines);
        }
        painter.rect_stroke(square, 0.0, egui::Stroke::new(1.0, egui::Color32::from_gray(140)), egui::StrokeKind::Outside);
    }

    /// The mesh as a picture of `size` pixels, rasterized in software with the baked maps: `surface` 0 lights the
    /// normal map and multiplies the occlusion map, 1 and up show map `surface - 1` as it is, without light.
    /// None when there is nothing to texture (no UVs or maps).
    fn render(&self, size: usize, yaw: f32, pitch: f32, zoom: f32, surface: usize) -> Option<egui::ColorImage> {
        if self.uvs.is_empty() || self.maps.is_empty() || size < 8 {
            return None;
        }
        let normal_map = self.maps.iter().find(|m| m.is("normal"));
        let occlusion = self.maps.iter().find(|m| m.is("occlusion"));
        let unlit = surface.checked_sub(1).and_then(|k| self.maps.get(k));
        let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());
        let view = |p: [f32; 3]| {
            let x = cy * p[0] + sy * p[2];
            let z = -sy * p[0] + cy * p[2];
            [x, cp * p[1] - sp * z, sp * p[1] + cp * z]
        };
        let v: Vec<[f32; 3]> = self.positions.iter().map(|p| view(*p)).collect();
        let vn: Vec<[f32; 3]> = self.normals.iter().map(|n| view(*n)).collect();
        let scale = 0.45 * size as f32 * zoom;
        let c = size as f32 / 2.0;
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let unit = |a: [f32; 3]| {
            let l = dot(a, a).sqrt().max(1e-20);
            [a[0] / l, a[1] / l, a[2] / l]
        };
        let light = unit([0.35, 0.55, 0.75]);
        let mut depth = vec![f32::MIN; size * size];
        let mut pixels = vec![egui::Color32::TRANSPARENT; size * size];
        for (t, tri) in self.triangles.iter().enumerate() {
            let p = tri.map(|k| v[k as usize]);
            let s = p.map(|q| [c + q[0] * scale, c - q[1] * scale]);
            let area = (s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) - (s[2][0] - s[0][0]) * (s[1][1] - s[0][1]);
            if area.abs() < 1e-9 {
                continue;
            }
            let face = unit(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            let back = face[2] < 0.0;
            // Per-triangle tangent frame from the texture coordinates (v down, as the baker's tangents are).
            let uv = self.uvs[t].map(|q| [q[0], 1.0 - q[1]]);
            let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
            let (du1, dv1, du2, dv2) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1], uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
            let det = du1 * dv2 - du2 * dv1;
            let (tangent, bitangent) = if det.abs() > 1e-12 {
                let f = 1.0 / det;
                (std::array::from_fn(|a| f * (dv2 * e1[a] - dv1 * e2[a])), std::array::from_fn(|a| f * (-du2 * e1[a] + du1 * e2[a])))
            } else {
                ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            };
            let (min_x, max_x) = (s.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize, (s.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(size));
            let (min_y, max_y) = (s.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize, (s.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(size));
            for y in min_y..max_y {
                for x in min_x..max_x {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let b1 = ((px - s[0][0]) * (s[2][1] - s[0][1]) - (s[2][0] - s[0][0]) * (py - s[0][1])) / area;
                    let b2 = ((s[1][0] - s[0][0]) * (py - s[0][1]) - (px - s[0][0]) * (s[1][1] - s[0][1])) / area;
                    let b0 = 1.0 - b1 - b2;
                    if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                        continue;
                    }
                    let z = b0 * p[0][2] + b1 * p[1][2] + b2 * p[2][2];
                    if z <= depth[y * size + x] {
                        continue;
                    }
                    depth[y * size + x] = z;
                    let colour = if back {
                        [0.43, 0.1, 0.1]
                    } else {
                        let w = [b0, b1, b2];
                        let (tu, tv) = (b0 * uv[0][0] + b1 * uv[1][0] + b2 * uv[2][0], b0 * uv[0][1] + b1 * uv[1][1] + b2 * uv[2][1]);
                        if let Some(m) = unlit {
                            m.sample(tu, tv)
                        } else {
                            let n_low = unit(std::array::from_fn(|a| w[0] * vn[tri[0] as usize][a] + w[1] * vn[tri[1] as usize][a] + w[2] * vn[tri[2] as usize][a]));
                            let n = match normal_map {
                                Some(m) => {
                                    let tx = m.sample(tu, tv);
                                    let (x, mut y, zz) = (2.0 * tx[0] - 1.0, 2.0 * tx[1] - 1.0, 2.0 * tx[2] - 1.0);
                                    if self.green_up {
                                        y = -y;
                                    }
                                    let tt = unit(sub(tangent, std::array::from_fn(|a| n_low[a] * dot(tangent, n_low))));
                                    let bb = {
                                        let b = cross(n_low, tt);
                                        if dot(b, bitangent) < 0.0 { [-b[0], -b[1], -b[2]] } else { b }
                                    };
                                    unit(std::array::from_fn(|a| tt[a] * x + bb[a] * y + n_low[a] * zz))
                                }
                                None => n_low,
                            };
                            let lit = 0.2 + 0.8 * dot(n, light).max(0.0);
                            let ao = occlusion.map_or(1.0, |m| m.sample(tu, tv)[0]);
                            let k = lit * (0.25 + 0.75 * ao);
                            [0.804 * k, 0.765 * k, 0.686 * k]
                        }
                    };
                    pixels[y * size + x] = egui::Color32::from_rgb((colour[0].clamp(0.0, 1.0) * 255.0) as u8, (colour[1].clamp(0.0, 1.0) * 255.0) as u8, (colour[2].clamp(0.0, 1.0) * 255.0) as u8);
                }
            }
        }
        Some(egui::ColorImage { size: [size, size], source_size: egui::vec2(size as f32, size as f32), pixels })
    }

    /// Smooth-shaded triangles, far ones first, as one egui mesh in `rect`.
    fn paint(&self, painter: &egui::Painter, rect: egui::Rect, yaw: f32, pitch: f32, zoom: f32) {
        const LIMIT: usize = 150_000;
        if self.triangles.len() > LIMIT {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, format!("{} triangles: too many to preview here", self.triangles.len()), egui::FontId::proportional(14.0), egui::Color32::GRAY);
            return;
        }
        let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());
        // Rotate about the vertical axis, then tilt; the camera looks down -z, z points at the viewer.
        let view = |p: [f32; 3]| {
            let x = cy * p[0] + sy * p[2];
            let z = -sy * p[0] + cy * p[2];
            [x, cp * p[1] - sp * z, sp * p[1] + cp * z]
        };
        let v: Vec<[f32; 3]> = self.positions.iter().map(|p| view(*p)).collect();
        let vn: Vec<[f32; 3]> = self.normals.iter().map(|n| view(*n)).collect();
        let scale = 0.45 * rect.width().min(rect.height()) * zoom;
        let to_screen = |p: [f32; 3]| egui::pos2(rect.center().x + p[0] * scale, rect.center().y - p[1] * scale);
        let light = {
            let l = [0.35f32, 0.55, 0.75];
            let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
            [l[0] / n, l[1] / n, l[2] / n]
        };
        let mut order: Vec<(f32, usize)> = self.triangles.iter().enumerate().map(|(i, t)| (t.iter().map(|&k| v[k as usize][2]).sum::<f32>(), i)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut mesh = egui::epaint::Mesh::default();
        for (_, i) in order {
            let t = self.triangles[i];
            let [a, b, c] = t.map(|k| v[k as usize]);
            let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
            // Faces turned away show dark red, which also exposes inside-out or folded triangles; the rest is shaded
            // per vertex from averaged normals.
            let back = n[2] < 0.0;
            let _ = len;
            let base = mesh.vertices.len() as u32;
            for (j, p) in [a, b, c].into_iter().enumerate() {
                let colour = if back {
                    egui::Color32::from_rgb(110, 25, 25)
                } else {
                    let q = vn[t[j] as usize];
                    let facing = q[0] * light[0] + q[1] * light[1] + q[2] * light[2];
                    let k = 0.2 + 0.8 * facing.max(0.0);
                    egui::Color32::from_rgb((205.0 * k) as u8, (195.0 * k) as u8, (175.0 * k) as u8)
                };
                mesh.vertices.push(egui::epaint::Vertex { pos: to_screen(p), uv: egui::epaint::WHITE_UV, color: colour });
            }
            mesh.indices.extend([base, base + 1, base + 2]);
        }
        painter.add(egui::Shape::mesh(mesh));
    }
}

impl Default for MeshWorkspace {
    fn default() -> Self {
        let graph = Self::example();
        let mut editor = crate::editor::Editor::default();
        editor.mesh_mode = true;
        editor.frame_next = true;
        MeshWorkspace { saved: graph.clone(), graph, editor, path: None, cache: Arc::new(Mutex::new(MeshCache::default())), job: None, log: Vec::new(), written: Vec::new(), seed: 0, seed_mode: Default::default(), udim: Default::default(), autorun: false, preview: None, show_uv: false, thumbnails: Vec::new(), map_files: Vec::new(), show_map: false, map_index: 0, map_tex: None, map_zoom: 1.0, map_pan: egui::Vec2::ZERO, field_view: true, surface: 0, surface_tex: None, surface_key: None, ray: None, ray_tex: None, ray_error: None, ray_sent: None, ray_key: (0, None), auto_preview: false, preview_key: 0, changed_since: None, history: crate::history::History::new(&Graph::default()), yaw: 0.6, pitch: 0.35, zoom: 1.0 }
    }
}

impl MeshWorkspace {
    /// A starting graph: a mesh, a noise graph added to it, remeshed, unwrapped and baked, with its three outputs.
    pub fn example() -> Graph {
        use crate::graph::MathOp;
        let mut g = Graph::default();
        let mesh = g.add(NodeKind::LoadMesh(Default::default()), [0.0, 0.0]);
        let noise = g.add(NodeKind::NoiseGraph(Default::default()), [0.0, 140.0]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [200.0, 60.0]);
        let remesh = g.add(NodeKind::Remesh { triangles: 20_000, adaptivity: 1.0 }, [400.0, 60.0]);
        let uv = g.add(NodeKind::AutoUv(Default::default()), [600.0, 60.0]);
        let bake = g.add(NodeKind::Bake(Default::default()), [800.0, 100.0]);
        let out_mesh = g.add(NodeKind::OutMesh(Default::default()), [1000.0, 0.0]);
        let out_image = g.add(NodeKind::OutImage(Default::default()), [1000.0, 100.0]);
        g.connect(mesh, add, 0);
        g.connect(noise, add, 1);
        g.connect(add, remesh, 0);
        g.connect(remesh, uv, 0);
        g.connect(uv, bake, 0);
        g.connect(add, bake, 1);
        g.connect(uv, out_mesh, 0);
        g.connect_from(bake, 0, out_image, 0);
        g
    }

    /// The graph left at the last exit; it counts as unsaved unless it was read from a file unchanged.
    pub fn restore(&mut self, graph: Graph, path: Option<PathBuf>) {
        let saved = path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| Graph::from_json(&s).ok());
        self.graph = graph;
        self.saved = saved.unwrap_or_else(|| self.graph.clone());
        self.path = path;
        self.history = crate::history::History::new(&self.graph);
        self.editor.frame_next = true;
    }

    /// A file dropped on the window: a mesh becomes a Load Mesh node, a noise graph a Noise Graph node, a mesh graph is
    /// opened (after asking when there are unsaved changes).
    pub fn drop_file(&mut self, path: PathBuf) {
        self.drop_file_at(path, None);
    }

    /// Like `drop_file`, with the node placed under a screen position when there is one.
    pub fn drop_file_at(&mut self, path: PathBuf, screen: Option<egui::Pos2>) {
        let ext = path.extension().map_or(String::new(), |e| e.to_string_lossy().to_lowercase());
        let at = match screen {
            Some(p) => self.editor.screen_to_graph(p),
            None => {
                // Below the lowest node, at the left, so it does not land on top of anything.
                let y = self.graph.nodes.values().map(|n| n.pos[1]).fold(0.0, f32::max);
                [0.0, y + 120.0]
            }
        };
        let text = path.to_string_lossy().into_owned();
        match ext.as_str() {
            "meshgraph" => {
                if self.may_discard()
                    && let Ok(g) = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
                        self.set(g, Some(path));
                    }
            }
            "noisegraph" => {
                let embedded = std::fs::read_to_string(&path).unwrap_or_default();
                self.graph.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { file: text, embedded, ..Default::default() }), at);
            }
            "sdf" => {
                // A live Load SDF node. Beside a terrain it starts at the terrain's middle, a twentieth of its side wide.
                let mut p = crate::sdf_node::LoadSdfParams { file: text, ..Default::default() };
                let base = self.base();
                let terrain = self.graph.nodes.values().find_map(|n| match &n.kind {
                    NodeKind::LoadMesh(m) if crate::landscape_field::is_heightmap(&m.path) => crate::landscape_field::extent(&crate::meshgraph::resolve_in(&base, &m.path), m.unit_cm),
                    _ => None,
                });
                if let Some((side, z)) = terrain {
                    p.z_up = true;
                    p.snap = true;
                    p.size = side / 20.0;
                    p.offset = [side / 2.0, side / 2.0, z];
                }
                self.graph.add(NodeKind::LoadSdf(p), at);
            }
            "png" => {
                self.graph.add(NodeKind::LoadMesh(crate::meshgraph::LoadMeshParams { path: text, unit_cm: 1.0, ..Default::default() }), at);
            }
            _ if crate::meshformats::is_mesh_file(&path) => {
                self.graph.add(NodeKind::LoadMesh(crate::meshgraph::LoadMeshParams { path: text, ..Default::default() }), at);
            }
            _ => {}
        }
    }

    pub fn has_preview(&self) -> bool {
        self.preview.is_some()
    }

    pub fn dirty(&self) -> bool {
        self.graph != self.saved
    }

    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    fn set(&mut self, graph: Graph, path: Option<PathBuf>) {
        self.saved = graph.clone();
        self.history = crate::history::History::new(&graph);
        self.graph = graph;
        self.path = path;
        self.editor.selected.clear();
        self.editor.frame_next = true;
    }

    /// Whether replacing the graph may go ahead: nothing unsaved, or the user agrees to discard it.
    fn may_discard(&self) -> bool {
        !self.dirty()
            || rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Warning)
                .set_title("Discard changes?")
                .set_description("The mesh graph has unsaved changes. Discard them?")
                .set_buttons(rfd::MessageButtons::YesNo)
                .show()
                == rfd::MessageDialogResult::Yes
    }

    /// The folder relative paths in the graph belong to: the graph file's, or the working folder.
    pub fn base(&self) -> PathBuf {
        if let Some(dir) = self.path.as_ref().and_then(|p| p.parent()) {
            return dir.to_path_buf();
        }
        // An unsaved graph: next to its first mesh file, so results do not land in whatever folder the app started in.
        self.graph
            .nodes
            .values()
            .find_map(|n| match &n.kind {
                NodeKind::LoadMesh(p) if !p.path.is_empty() && p.path != "{batch}" => std::path::Path::new(&p.path).parent().map(|d| d.to_path_buf()),
                _ => None,
            })
            .filter(|d| d.is_dir())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn undo(&mut self, redo: bool) {
        let g = if redo { self.history.redo(&self.graph) } else { self.history.undo(&self.graph) };
        if let Some(g) = g {
            self.graph = g;
            self.editor.selected.retain(|id| self.graph.nodes.contains_key(id));
        }
    }

    pub fn open_graph(&mut self, graph: Graph, path: PathBuf) {
        self.set(graph, Some(path));
    }

    /// Writes the graph, first refreshing every noise graph node's embedded copy from its file, so the mesh graph
    /// carries what it was saved with (the file stays the source of truth).
    pub fn save(&mut self, ask: bool) -> Result<(), String> {
        let path = match (&self.path, ask) {
            (Some(p), false) => p.clone(),
            _ => match crate::library::dialog(crate::library::Kind::MeshGraph, rfd::FileDialog::new().add_filter("Mesh graph", &["meshgraph"]).set_file_name("mesh.meshgraph")).save_file() {
                Some(p) => p,
                None => return Ok(()),
            },
        };
        let base = path.parent().map_or(PathBuf::from("."), |p| p.to_path_buf());
        for n in self.graph.nodes.values_mut() {
            if let NodeKind::NoiseGraph(p) = &mut n.kind
                && let Ok(text) = std::fs::read_to_string(crate::meshgraph::resolve_in(&base, &p.file)) {
                    p.embedded = text;
                }
        }
        let text = self.graph.to_json();
        crate::keep_previous_version(&path, &text);
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        self.saved = self.graph.clone();
        crate::library::request_thumbnail(&path);
        self.path = Some(path);
        Ok(())
    }

    pub fn open(&mut self) -> Result<(), String> {
        let Some(path) = crate::library::dialog(crate::library::Kind::MeshGraph, rfd::FileDialog::new().add_filter("Mesh graph", &["meshgraph"])).pick_file() else { return Ok(()) };
        let g = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display())).and_then(|s| Graph::from_json(&s))?;
        self.set(g, Some(path));
        Ok(())
    }

    /// A quick look: the shape the first Remesh gives, at most 6000 triangles, written to a temporary file and shown.
    fn start_preview(&mut self, graph: Graph, device: eframe::wgpu::Device, queue: eframe::wgpu::Queue) {
        let progress = crate::meshexport::Progress::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let (cache, seed, p, l, base) = (self.cache.clone(), self.seed, progress.clone(), log.clone(), self.base());
        std::thread::spawn(move || {
            let mut cache = cache.lock().unwrap();
            let mut note = |s: &str| l.lock().unwrap().push(s.to_string());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mesh = Runner { device: &device, queue: &queue, cache: &mut cache, base, seed, progress: p, log: &mut note, batch_file: None, max_triangles: Some(6000) }.preview(&graph)?;
                let path = std::env::temp_dir().join("noiser-mesh-preview.obj");
                mesh.write_obj(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                Ok(vec![path])
            }));
            let _ = tx.send(result.unwrap_or_else(|_| Err("the preview crashed".into())));
        });
        crate::meshgraph::CANCEL.store(false, std::sync::atomic::Ordering::Relaxed);
        self.log.clear();
        self.written.clear();
        self.thumbnails.clear();
        self.map_files.clear();
        self.map_tex = None;
        self.job = Some(RunJob { progress, finished: Default::default(), total: 1, log, done: rx });
    }

    fn start(&mut self, graph: Graph, device: eframe::wgpu::Device, queue: eframe::wgpu::Queue, files: Vec<Option<PathBuf>>) {
        let progress = crate::meshexport::Progress::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let (cache, seed, mode, p, l) = (self.cache.clone(), self.seed, self.seed_mode, progress.clone(), log.clone());
        let base = self.base();
        let total = files.len();
        let finished = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fin = finished.clone();
        std::thread::spawn(move || {
            let mut cache = cache.lock().unwrap();
            let (mut written, mut failures) = (Vec::new(), Vec::new());
            let t0 = std::time::Instant::now();
            for (i, file) in files.into_iter().enumerate() {
                let label = file.as_ref().map_or(String::new(), |f| format!("[{}/{total}] {}: ", i + 1, f.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())));
                let mut note = |s: &str| l.lock().unwrap().push(format!("{:>6.1} s  {label}{s}", t0.elapsed().as_secs_f32()));
                if mode != crate::meshexport::SeedMode::Same {
                    note(&format!("seed {}", mode.seed_for(seed, i)));
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Runner { device: &device, queue: &queue, cache: &mut cache, base: base.clone(), seed: mode.seed_for(seed, i), progress: p.clone(), log: &mut note, batch_file: file, max_triangles: None }.run(&graph)));
                match result.unwrap_or_else(|_| Err("the run crashed".into())) {
                    Ok(f) => written.extend(f),
                    Err(e) => {
                        l.lock().unwrap().push(format!("{label}FAILED: {e}"));
                        failures.push(e);
                    }
                }
                fin.store(i + 1, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = tx.send(if failures.is_empty() { Ok(written) } else { Err(format!("{} of {total} run(s) failed: {}", failures.len(), failures[0])) });
        });
        crate::meshgraph::CANCEL.store(false, std::sync::atomic::Ordering::Relaxed);
        self.log.clear();
        self.written.clear();
        self.job = Some(RunJob { progress, finished, total, log, done: rx });
    }
}

impl crate::App {
    /// The mesh graph as a run should see it: a Noise Graph node that names the noise graph open in the Noise graph tab
    /// uses that graph as it is on screen, saved or not, so the noise can be tuned and the result previewed straight away.
    fn graph_for_run(&self) -> Graph {
        let mut g = self.mesh_ws.graph.clone();
        if let Some(open) = &self.path {
            let base = self.mesh_ws.base();
            let same = |a: &std::path::Path, b: &std::path::Path| a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y);
            let text = self.graph.to_json();
            for n in g.nodes.values_mut() {
                if let NodeKind::NoiseGraph(p) = &mut n.kind
                    && !p.detached && !p.file.is_empty() && same(&crate::meshgraph::resolve_in(&base, &p.file), open) {
                        p.embedded = text.clone();
                        p.detached = true;
                    }
            }
        }
        g
    }

    /// Saves the mesh graph. A noise graph that is open and changed in the Noise graph tab, and that this mesh graph
    /// uses, is saved first, so the copy embedded in the mesh graph is the one just edited.
    pub(crate) fn save_mesh_graph(&mut self, ask: bool) {
        let uses_open = self.path.as_ref().is_some_and(|open| {
            let base = self.mesh_ws.base();
            let same = |a: &std::path::Path, b: &std::path::Path| a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y);
            self.mesh_ws.graph.nodes.values().any(|n| matches!(&n.kind, NodeKind::NoiseGraph(p) if !p.file.is_empty() && same(&crate::meshgraph::resolve_in(&base, &p.file), open)))
        });
        if uses_open && self.graph != self.saved {
            self.save(false);
        }
        if let Err(e) = self.mesh_ws.save(ask) {
            self.status = e;
        }
    }

    /// The whole Mesh graph tab (the menu bar above it is the app's).
    /// The tab strip under the menu bar, and a outline in the selected tab's colour around everything below it.
    pub(crate) fn workspace_tabs(&mut self, ui: &mut egui::Ui) {
        const TAB_CORNER: u8 = 7;
        let mut selected_span = None;
        let frame = egui::Frame::NONE.fill(ui.visuals().panel_fill).inner_margin(egui::Margin { left: 8, right: 8, top: 4, bottom: 0 });
        egui::Panel::top("workspace tabs").frame(frame).show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.horizontal(|ui| {
                for workspace in Workspace::ALL {
                    let is_selected = self.workspace == workspace;
                    let color = workspace.color();
                    let text_color = if is_selected { ui.visuals().strong_text_color() } else { ui.visuals().text_color() };
                    let galley = ui.painter().layout_no_wrap(workspace.label().to_string(), egui::FontId::proportional(14.0), text_color);
                    let (rect, response) = ui.allocate_exact_size(galley.size() + egui::vec2(28.0 + 22.0, 10.0), egui::Sense::click());
                    let (fill, stroke) = if is_selected {
                        (ui.visuals().panel_fill, egui::Stroke::new(2.0, color))
                    } else if response.hovered() {
                        (color.gamma_multiply(0.18), egui::Stroke::new(1.0, color.gamma_multiply(0.8)))
                    } else {
                        (ui.visuals().faint_bg_color, egui::Stroke::new(1.0, color.gamma_multiply(0.45)))
                    };
                    let corners = egui::CornerRadius { nw: TAB_CORNER, ne: TAB_CORNER, sw: 0, se: 0 };
                    ui.painter().rect(rect, corners, fill, stroke, egui::StrokeKind::Inside);
                    egui_extras::install_image_loaders(ui.ctx());
                    crate::icons::image(workspace.icon(), 16.0, if is_selected { color } else { color.gamma_multiply(0.8) }).paint_at(ui, egui::Rect::from_center_size(egui::pos2(rect.left() + 18.0, rect.center().y), egui::vec2(16.0, 16.0)));
                    ui.painter().galley(egui::pos2(rect.left() + 32.0, rect.center().y - galley.size().y * 0.5), galley, text_color);
                    if is_selected {
                        selected_span = Some((rect.left() + 1.0, rect.right() - 1.0));
                    } else if response.clicked() {
                        self.workspace = workspace;
                    }
                }
            });
        });
        // Drawn on its own layer so it covers whatever the workspace shows, and open under the selected tab so the tab
        // and its contents read as one piece.
        let content = ui.available_rect_before_wrap().shrink(1.0);
        let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("workspace outline"));
        let stroke = egui::Stroke::new(2.0, self.workspace.color());
        let (gap_left, gap_right) = selected_span.unwrap_or((content.left(), content.left()));
        ui.ctx().layer_painter(layer).add(egui::Shape::line(
            vec![
                egui::pos2(gap_right, content.top()),
                content.right_top(),
                content.right_bottom(),
                content.left_bottom(),
                content.left_top(),
                egui::pos2(gap_left, content.top()),
            ],
            stroke,
        ));
    }

    pub(crate) fn mesh_workspace_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // The worker's progress and result.
        let mut finished = None;
        if let Some(job) = &self.mesh_ws.job {
            let lines = job.log.lock().unwrap().clone();
            self.mesh_ws.log = lines;
            match job.done.try_recv() {
                Ok(r) => finished = Some(r),
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => finished = Some(Err("the run stopped unexpectedly".into())),
            }
        }
        if let Some(r) = finished {
            self.mesh_ws.job = None;
            match r {
                Ok(files) if files.len() == 1 && files[0].file_name().is_some_and(|n| n == "noiser-mesh-preview.obj") => {
                    self.status = "Mesh graph preview ready (nothing written to your folders)".into();
                    self.mesh_ws.preview = Preview::load(&files[0]).ok();
                }
                Ok(files) => {
                    self.status = format!("Mesh graph: {} file(s) written", files.len());
                    self.mesh_ws.preview = files.iter().find(|f| f.extension().is_some_and(|e| e == "obj" || e == "ply" || e == "glb")).and_then(|f| Preview::load(f).ok());
                    self.mesh_ws.map_files = files.iter().filter(|f| f.extension().is_some_and(|e| e == "png")).take(8).cloned().collect();
                    self.mesh_ws.surface = 0;
                    self.mesh_ws.surface_key = None;
                    self.mesh_ws.map_tex = None;
                    self.mesh_ws.map_index = 0;
                    if let Some(p) = &mut self.mesh_ws.preview {
                        p.maps = self.mesh_ws.map_files.iter().filter_map(|f| MapImage::load(f)).collect();
                        p.green_up = self.mesh_ws.graph.nodes.values().any(|n| matches!(&n.kind, NodeKind::Bake(b) if b.green_up));
                    }
                    self.mesh_ws.thumbnails = files
                        .iter()
                        .filter(|f| f.extension().is_some_and(|e| e == "png"))
                        .take(8)
                        .filter_map(|f| {
                            let img = image::open(f).ok()?.thumbnail(256, 256).to_rgba8();
                            let size = [img.width() as usize, img.height() as usize];
                            let tex = ctx.load_texture(f.to_string_lossy(), egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::LINEAR);
                            Some((f.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()), tex))
                        })
                        .collect();
                    self.mesh_ws.written = files;
                }
                Err(e) => {
                    self.mesh_ws.log.push(format!("FAILED: {e}"));
                    self.status = format!("Mesh graph failed: {e}");
                }
            }
        }
        if self.mesh_ws.autorun && !self.mesh_ws.running() {
            self.mesh_ws.autorun = false;
            let (d, q, g) = (self.rs.device.clone(), self.rs.queue.clone(), self.graph_for_run());
            self.mesh_ws.start(g, d, q, vec![None]);
        }
        let base = self.mesh_ws.base();
        crate::meshgraph_ui::set_base(base.clone());
        // A noise graph file that was just chosen gets its embedded copy right away.
        for (id, n) in self.mesh_ws.graph.nodes.iter_mut() {
            if let NodeKind::NoiseGraph(p) = &mut n.kind
                && p.embedded.is_empty() && !p.file.is_empty()
                    && let Ok(text) = std::fs::read_to_string(crate::meshgraph::resolve_in(&base, &p.file)) {
                        // Not a change the user made: the saved copy gets it too, so the graph does not look edited.
                        if let Some(NodeKind::NoiseGraph(saved)) = self.mesh_ws.saved.nodes.get_mut(id).map(|n| &mut n.kind)
                            && saved.file == p.file && saved.embedded.is_empty() {
                                saved.embedded = text.clone();
                            }
                        p.embedded = text;
                    }
        }
        // Auto preview: when what the preview is made from has been still for a moment, make it again.
        if self.mesh_ws.auto_preview && !self.mesh_ws.running() {
            use std::hash::{Hash, Hasher};
            let g = self.graph_for_run();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            g.to_json().hash(&mut h);
            self.mesh_ws.seed.hash(&mut h);
            let key = h.finish();
            if key != self.mesh_ws.preview_key {
                let since = *self.mesh_ws.changed_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > std::time::Duration::from_millis(1000) && !ui.input(|i| i.pointer.any_down()) {
                    self.mesh_ws.preview_key = key;
                    self.mesh_ws.changed_since = None;
                    let (d, q) = (self.rs.device.clone(), self.rs.queue.clone());
                    self.mesh_ws.start_preview(g, d, q);
                } else {
                    ctx.request_repaint_after(std::time::Duration::from_millis(300));
                }
            } else {
                self.mesh_ws.changed_since = None;
            }
        }
        // Ctrl+Z and Ctrl+Y (or Ctrl+Shift+Z) act on this graph here, unless a text field has the keyboard.
        if !ctx.egui_wants_keyboard_input() {
            let (undo, redo) = ui.input(|i| (crate::keymap::pressed(i, Action::Undo), crate::keymap::pressed(i, Action::Redo)));
            if redo {
                self.mesh_ws.undo(true);
            } else if undo {
                self.mesh_ws.undo(false);
            }
        }
        // Saving acts on this graph here.
        let (save_as, save) = ui.input(|i| (crate::keymap::pressed(i, Action::SaveAs), crate::keymap::pressed(i, Action::Save)));
        if save_as {
            self.save_mesh_graph(true);
        } else if save {
            self.save_mesh_graph(false);
        }
        egui::Panel::right("mesh props").default_size(340.0).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if crate::icons::button(ui, "new", "New", None, true).on_hover_text("New mesh graph").clicked() && self.mesh_ws.may_discard() {
                    self.mesh_ws.set(MeshWorkspace::example(), None);
                }
                if crate::icons::button(ui, "open", "Open…", None, true).on_hover_text("Open a mesh graph… (Ctrl+O)").clicked() && self.mesh_ws.may_discard()
                    && let Err(e) = self.mesh_ws.open() {
                        self.status = e;
                    }
                if crate::icons::button(ui, "save", "Save", None, true).on_hover_text("Save (Ctrl+S). Saves the mesh graph, refreshes the embedded copy of every noise graph it uses, and saves the open noise graph too if it is one of them.").clicked() {
                    self.save_mesh_graph(false);
                }
                if crate::icons::button(ui, "save_as", "Save as…", None, true).on_hover_text("Save as…").clicked() {
                    self.save_mesh_graph(true);
                }
            });
            ui.weak(match (&self.mesh_ws.path, self.mesh_ws.dirty()) {
                (Some(p), true) => format!("{} (unsaved changes)", p.display()),
                (Some(p), false) => p.display().to_string(),
                (None, _) => "unsaved mesh graph".into(),
            });
            ui.separator();
            ui.horizontal(|ui| {
                let running = self.mesh_ws.running();
                if crate::icons::tinted_button(ui, "play", "Run", None, !running, Some(egui::Color32::from_rgb(110, 200, 120))).on_hover_text("Runs every Remesh, Auto UVs and Bake that feeds an output node and writes the files.").clicked() {
                    let (d, q, g) = (self.rs.device.clone(), self.rs.queue.clone(), self.graph_for_run());
                    self.mesh_ws.start(g, d, q, vec![None]);
                }
                if crate::icons::button(ui, "preview", "Preview", None, !running).on_hover_text("A quick look at the shape the first Remesh makes (6,000 triangles at most), without UVs or maps: for tuning the noise graphs.").clicked() {
                    let (d, q, g) = (self.rs.device.clone(), self.rs.queue.clone(), self.graph_for_run());
                    self.mesh_ws.start_preview(g, d, q);
                }
                if crate::icons::tinted_button(ui, "run_files", "Run on files…", None, !running, Some(egui::Color32::from_rgb(110, 200, 120))).on_hover_text("Runs the graph once for each mesh file you pick: a Load Mesh node whose path is {batch} reads that file, and {name} in an output name becomes its name.").clicked()
                    && let Some(files) = rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply", "stl", "gltf", "glb", "fbx", "sdf"]).pick_files() {
                        let (d, q, g) = (self.rs.device.clone(), self.rs.queue.clone(), self.graph_for_run());
                        self.mesh_ws.start(g, d, q, files.into_iter().map(Some).collect());
                    }
                if crate::icons::button(ui, "udim", "UDIM pack…", None, true).on_hover_text("After a batch: packs many meshes and their maps into one UDIM tileset with one combined mesh.").clicked() {
                    self.mesh_ws.udim.open = true;
                }
                if crate::icons::tinted_button(ui, "stop", "Cancel", None, running, Some(egui::Color32::from_rgb(230, 110, 100))).on_hover_text("Stops the run before its next node (a node that is already working finishes first).").clicked() {
                    crate::meshgraph::CANCEL.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.mesh_ws.seed).prefix("seed "));
                egui::ComboBox::from_id_salt("batch seed mode").selected_text(match self.mesh_ws.seed_mode {
                    crate::meshexport::SeedMode::Same => "same seed",
                    crate::meshexport::SeedMode::Increment => "seed + 1 each",
                    crate::meshexport::SeedMode::Random => "random each",
                }).show_ui(ui, |ui| {
                    use crate::meshexport::SeedMode::*;
                    ui.selectable_value(&mut self.mesh_ws.seed_mode, Same, "Same seed");
                    ui.selectable_value(&mut self.mesh_ws.seed_mode, Increment, "Seed + 1 each").on_hover_text("Run on files: the first mesh uses the seed, the next one seed + 1, and so on.");
                    ui.selectable_value(&mut self.mesh_ws.seed_mode, Random, "Random each").on_hover_text("Run on files: a different seed per mesh, repeatable from the seed.");
                });
            });
            self.mesh_ws.udim.window(ui.ctx());
            ui.checkbox(&mut self.mesh_ws.field_view, "Field view (ray marched, beside the mesh)").on_hover_text("Draws the field of the first Remesh directly on the GPU, without making a mesh: it follows the graph as you edit it and turns with the mesh view.");
            ui.checkbox(&mut self.mesh_ws.auto_preview, "Preview again after changes").on_hover_text("A second after the graph, or the noise graph open in the Noise tab, stops changing, the quick preview runs again.");
            if let Some(job) = &self.mesh_ws.job {
                let p = job.progress.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                ui.add(egui::ProgressBar::new(p).show_percentage());
                // A batch: the files done plus the share of the current one, as a fraction of all of them (an estimate: files differ in size).
                if job.total > 1 {
                    let done = job.finished.load(std::sync::atomic::Ordering::Relaxed);
                    let all = ((done as f32 + p) / job.total as f32).min(1.0);
                    ui.add(egui::ProgressBar::new(all).fill(egui::Color32::from_rgb(220, 175, 40)).text(format!("batch: file {} of {}  ({:.0}%)", (done + 1).min(job.total), job.total, all * 100.0)));
                }
            }
            egui::CollapsingHeader::new("Log").default_open(true).show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(140.0).stick_to_bottom(true).show(ui, |ui| {
                    for l in &self.mesh_ws.log {
                        ui.weak(l);
                    }
                });
            });
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let sel: Vec<_> = self.mesh_ws.editor.selected.iter().copied().collect();
                match sel.as_slice() {
                    [id] if self.mesh_ws.graph.nodes.contains_key(id) => {
                        let names = self.mesh_ws.graph.declared_names();
                        let node = self.mesh_ws.graph.nodes.get_mut(id).unwrap();
                        ui.heading(crate::editor::title(&node.kind));
                        let mut label = node.label.clone().unwrap_or_default();
                        if ui.add(egui::TextEdit::singleline(&mut label).hint_text("Label (shown on the node)").desired_width(f32::INFINITY)).changed() {
                            node.label = Some(label).filter(|l| !l.is_empty());
                        }
                        crate::node_props(ui, &mut node.kind, &mut node.pins, &names);
                    }
                    [] => _ = ui.weak("Select a node to edit it. Double-click a Noise Graph node to open that graph in the Noise graph tab."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
            });
        });
        self.mesh_ws.graph.prune_edges();
        let settled = !ui.input(|i| i.pointer.any_down());
        self.mesh_ws.history.commit(&self.mesh_ws.graph, settled);
        egui::Panel::bottom("mesh graph editor").resizable(true).default_size(330.0).show(ui, |ui| {
            self.mesh_ws.editor.show(ui, &mut self.mesh_ws.graph, &Default::default());
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            crate::library::set_view_rect(rect);
            let resp = ui.interact(rect, ui.scope_id().with("mesh preview"), egui::Sense::click_and_drag());
            let d = resp.drag_delta();
            let map_view = self.mesh_ws.show_map && !self.mesh_ws.map_files.is_empty();
            if map_view {
                self.mesh_ws.map_pan += d;
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    self.mesh_ws.map_zoom = (self.mesh_ws.map_zoom * (1.0 + scroll * 0.002)).clamp(0.1, 32.0);
                }
                if resp.double_clicked() {
                    (self.mesh_ws.map_zoom, self.mesh_ws.map_pan) = (1.0, egui::Vec2::ZERO);
                }
            } else {
                self.mesh_ws.yaw -= d.x * 0.01;
                self.mesh_ws.pitch = (self.mesh_ws.pitch + d.y * 0.01).clamp(-1.5, 1.5);
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    self.mesh_ws.zoom = (self.mesh_ws.zoom * (1.0 + scroll * 0.002)).clamp(0.2, 8.0);
                }
                if resp.double_clicked() {
                    (self.mesh_ws.yaw, self.mesh_ws.pitch, self.mesh_ws.zoom) = (0.6, 0.35, 1.0);
                }
            }
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_gray(28));
            // The maps the run wrote, in a strip along the bottom.
            let mut view_rect = rect;
            if !self.mesh_ws.thumbnails.is_empty() {
                let strip = 120.0;
                view_rect.max.y -= strip;
                let mut x = rect.left() + 8.0;
                for (name, tex) in &self.mesh_ws.thumbnails {
                    let img = egui::Rect::from_min_size(egui::pos2(x, rect.bottom() - strip + 4.0), egui::vec2(96.0, 96.0));
                    painter.image(tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    let hit = ui.interact(img, ui.scope_id().with(("map thumbnail", x as i32)), egui::Sense::click()).on_hover_text("Click to look at this map full size");
                    let k = ((x - rect.left() - 8.0) / 104.0).round() as usize;
                    if hit.clicked() {
                        self.mesh_ws.map_index = k.min(self.mesh_ws.map_files.len().saturating_sub(1));
                        self.mesh_ws.show_map = true;
                        self.mesh_ws.show_uv = false;
                    }
                    if map_view && k == self.mesh_ws.map_index {
                        painter.rect_stroke(img.expand(2.0), 0.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 190, 255)), egui::StrokeKind::Outside);
                    }
                    painter.text(img.left_bottom() + egui::vec2(0.0, 2.0), egui::Align2::LEFT_TOP, name, egui::FontId::proportional(10.0), egui::Color32::from_gray(170));
                    x += 104.0;
                }
            }
            // The selected map at full size, loaded on first use.
            if map_view {
                let i = self.mesh_ws.map_index.min(self.mesh_ws.map_files.len() - 1);
                if self.mesh_ws.map_tex.as_ref().map(|(k, _)| *k) != Some(i) {
                    let path = self.mesh_ws.map_files[i].clone();
                    self.mesh_ws.map_tex = image::open(&path).ok().map(|img| {
                        let img = if img.width().max(img.height()) > 4096 { img.thumbnail(4096, 4096) } else { img }.to_rgba8();
                        let size = [img.width() as usize, img.height() as usize];
                        (i, ctx.load_texture(path.to_string_lossy(), egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::NEAREST))
                    });
                }
                if let Some((_, tex)) = &self.mesh_ws.map_tex {
                    let side = (view_rect.width().min(view_rect.height()) - 16.0).max(50.0) * self.mesh_ws.map_zoom;
                    let img = egui::Rect::from_center_size(view_rect.center() + self.mesh_ws.map_pan, egui::vec2(side, side));
                    let clipped = ui.painter_at(view_rect);
                    clipped.image(tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    let name = self.mesh_ws.map_files[i].file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
                    painter.text(rect.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, format!("{name} ({}x{}). Zoom {:.0}%. Drag to pan, scroll to zoom, double-click to fit.", tex.size()[0], tex.size()[1], self.mesh_ws.map_zoom * 100.0), egui::FontId::proportional(12.0), egui::Color32::from_gray(160));
                } else {
                    painter.text(view_rect.center(), egui::Align2::CENTER_CENTER, "Could not read this map", egui::FontId::proportional(14.0), egui::Color32::from_gray(150));
                }
            }
            // The ray-marched field, to the left of the software view.
            let show_field = self.mesh_ws.field_view && !map_view && !self.mesh_ws.show_uv;
            let mut view_rect = view_rect;
            let mut field_rect = None;
            if show_field {
                let halves = view_rect.width() / 2.0;
                field_rect = Some(egui::Rect::from_min_size(view_rect.min, egui::vec2(halves, view_rect.height())));
                view_rect.min.x += halves;
                let side = (halves.min(view_rect.height()) as u32 / 8 * 8).clamp(128, 640);
                if self.mesh_ws.ray.is_none() {
                    self.mesh_ws.ray = Some(RayWorker::spawn(self.rs.device.clone(), self.rs.queue.clone()));
                }
                // What the picture is made from: the graph as a run would see it, and the seed.
                let graph = self.graph_for_run();
                let key = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    graph.to_json().hash(&mut h);
                    self.mesh_ws.seed.hash(&mut h);
                    h.finish()
                };
                let ws = &mut self.mesh_ws;
                if key != ws.ray_key.0 {
                    ws.ray_key = (key, Some(std::time::Instant::now()));
                }
                // A changed graph is drawn again once it has been still for a moment (compiling takes a while); a new
                // view or size is drawn at once.
                let settled = ws.ray_key.1.is_none_or(|t| t.elapsed() > std::time::Duration::from_millis(400));
                let quant = [(ws.yaw * 100.0) as i32, (ws.pitch * 100.0) as i32, (ws.zoom * 100.0) as i32];
                let first = ws.ray_sent.is_none();
                let changed = ws.ray_sent.is_none_or(|(k, v, s)| k != ws.ray_key.0 || v != quant || s != side);
                if changed && (settled || first) && !ui.input(|i| i.pointer.any_down() && ws.ray_sent.is_some_and(|(k, _, _)| k != ws.ray_key.0)) {
                    if let Some(w) = &mut ws.ray {
                        let req = RayRequest { graph, key, base: ws.path.as_ref().and_then(|p| p.parent()).map_or(PathBuf::from("."), |p| p.to_path_buf()), seed: ws.seed, view: [ws.yaw, ws.pitch, ws.zoom], size: side };
                        if w.tx.send(req).is_ok() {
                            w.outstanding += 1;
                        }
                    }
                    ws.ray_sent = Some((key, quant, side));
                    ws.ray_key.1 = None;
                } else if changed {
                    ctx.request_repaint_after(std::time::Duration::from_millis(150));
                }
                if let Some(w) = &mut ws.ray {
                    while let Ok(r) = w.rx.try_recv() {
                        w.outstanding = w.outstanding.saturating_sub(1);
                        match r.rgba {
                            Ok(rgba) => {
                                let image = egui::ColorImage::from_rgba_unmultiplied([r.size as usize, r.size as usize], &rgba);
                                ws.ray_tex = Some(ctx.load_texture("field view", image, egui::TextureOptions::LINEAR));
                                ws.ray_error = None;
                            }
                            Err(e) if e.is_empty() => {}
                            Err(e) => ws.ray_error = Some(e),
                        }
                    }
                    if w.outstanding > 0 {
                        ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    }
                }
                let fr = field_rect.unwrap();
                match (&ws.ray_tex, &ws.ray_error) {
                    (_, Some(e)) => _ = painter.text(fr.center(), egui::Align2::CENTER_CENTER, format!("Field view: {e}"), egui::FontId::proportional(12.0), egui::Color32::from_rgb(230, 150, 90)),
                    (Some(tex), None) => {
                        let s = fr.width().min(fr.height());
                        let img = egui::Rect::from_center_size(fr.center(), egui::vec2(s, s));
                        painter.image(tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    }
                    _ => _ = painter.text(fr.center(), egui::Align2::CENTER_CENTER, "Field view: working...", egui::FontId::proportional(12.0), egui::Color32::from_gray(130)),
                }
                painter.line_segment([egui::pos2(view_rect.min.x, view_rect.min.y), egui::pos2(view_rect.min.x, view_rect.max.y)], egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
                painter.text(fr.left_top() + egui::vec2(8.0, 22.0), egui::Align2::LEFT_TOP, "Field (ray marched)", egui::FontId::proportional(11.0), egui::Color32::from_gray(120));
            }
            let preview = self.mesh_ws.preview.take();
            match &preview {
                _ if map_view => {}
                Some(p) => {
                    let uv = self.mesh_ws.show_uv && !p.uvs.is_empty();
                    if uv {
                        p.paint_uv(&painter, view_rect);
                    } else {
                        let ws = &mut self.mesh_ws;
                        let side = (view_rect.width().min(view_rect.height()) as usize).clamp(64, 800);
                        let textured = ws.surface != 1 && !p.maps.is_empty() && !p.uvs.is_empty() && p.triangles.len() <= 150_000;
                        let key = (((ws.yaw * 100.0) as i32), ((ws.pitch * 100.0) as i32), ((ws.zoom * 100.0) as i32), ws.surface, side);
                        if textured {
                            if (ws.surface_key != Some(key) || ws.surface_tex.is_none())
                                && let Some(img) = p.render(side, ws.yaw, ws.pitch, ws.zoom, ws.surface) {
                                    ws.surface_tex = Some(ctx.load_texture("mesh surface", img, egui::TextureOptions::LINEAR));
                                    ws.surface_key = Some(key);
                                }
                            if let Some(tex) = &ws.surface_tex {
                                let img = egui::Rect::from_center_size(view_rect.center(), egui::vec2(side as f32, side as f32));
                                painter.image(tex.id(), img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                            }
                        } else {
                            p.paint(&painter, view_rect, ws.yaw, ws.pitch, ws.zoom);
                        }
                    }
                    let hint = if uv { "UV layout" } else { "Drag to turn, scroll to zoom, double-click to reset." };
                    painter.text(rect.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, format!("{}: {} triangles. {hint}", p.name, p.triangles.len()), egui::FontId::proportional(12.0), egui::Color32::from_gray(160));
                }
                None => _ = painter.text(view_rect.center(), egui::Align2::CENTER_CENTER, "Run the graph to see its mesh here", egui::FontId::proportional(14.0), egui::Color32::from_gray(120)),
            }
            self.mesh_ws.preview = preview;
            let has_maps = !self.mesh_ws.map_files.is_empty();
            let has_uv = self.mesh_ws.preview.as_ref().is_some_and(|p| !p.uvs.is_empty());
            let surface_names: Vec<String> = self.mesh_ws.preview.as_ref().map(|p| p.maps.iter().map(|m| m.name.clone()).collect()).unwrap_or_default();
            if !surface_names.is_empty() && !map_view && !self.mesh_ws.show_uv {
                let at = egui::Rect::from_min_size(egui::pos2(rect.right() - 260.0, rect.top() + 30.0), egui::vec2(256.0, 22.0));
                ui.scope_builder(egui::UiBuilder::new().max_rect(at), |ui| {
                    let current = match self.mesh_ws.surface {
                        0 => "Baked normal and occlusion".to_string(),
                        1 => "Smooth grey".to_string(),
                        k => format!("{} (unlit)", surface_names.get(k - 2).cloned().unwrap_or_default()),
                    };
                    egui::ComboBox::from_id_salt("mesh surface").selected_text(current).width(240.0).show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.mesh_ws.surface, 0, "Baked normal and occlusion");
                        ui.selectable_value(&mut self.mesh_ws.surface, 1, "Smooth grey");
                        for (k, n) in surface_names.iter().enumerate() {
                            ui.selectable_value(&mut self.mesh_ws.surface, k + 2, format!("{n} (unlit)"));
                        }
                    }).response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::ComboBox, true, "Mesh surface"));
                });
            }
            if has_maps || has_uv {
                let corner = egui::Rect::from_min_size(egui::pos2(rect.right() - 260.0, rect.top() + 4.0), egui::vec2(256.0, 22.0));
                ui.scope_builder(egui::UiBuilder::new().max_rect(corner), |ui| {
                    ui.horizontal(|ui| {
                        let mut mode = if map_view { 2 } else if self.mesh_ws.show_uv { 1 } else { 0 };
                        ui.selectable_value(&mut mode, 0, "Shaded");
                        if has_uv {
                            ui.selectable_value(&mut mode, 1, "UV layout");
                        }
                        if has_maps {
                            ui.selectable_value(&mut mode, 2, "Maps");
                        }
                        (self.mesh_ws.show_uv, self.mesh_ws.show_map) = (mode == 1, mode == 2);
                    });
                });
            }
        });
        // Double-clicking a Noise Graph node goes to that graph.
        if let Some(id) = self.mesh_ws.editor.opened.take()
            && let Some(NodeKind::NoiseGraph(p)) = self.mesh_ws.graph.nodes.get(&id).map(|n| n.kind.clone()) {
                if p.file.is_empty() {
                    self.status = "This Noise Graph node has no file chosen yet".into();
                } else {
                    self.workspace = Workspace::Noise;
                    self.request(crate::Pending::Path(crate::meshgraph::resolve_in(&self.mesh_ws.base(), &p.file)));
                }
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_sdf_lands_in_the_middle_of_the_terrain() {
        let dir = std::env::temp_dir().join("noiser-drop-sdf-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let metres: Vec<f32> = (0..33 * 33).map(|k| (k % 33) as f32).collect();
        let png = dir.join("heightmap.png");
        crate::landscape::write_unreal(&png, 33, 2000.0, &crate::landscape::encode_unreal(&metres)).unwrap();
        let mut ws = MeshWorkspace::default();
        ws.graph.add(NodeKind::LoadMesh(crate::meshgraph::LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 1.0, ..Default::default() }), [0.0; 2]);
        ws.drop_file(dir.join("rock.sdf"));
        let NodeKind::LoadSdf(p) = ws.graph.nodes.values().last().unwrap().kind.clone() else { panic!("a Load SDF node") };
        // Centimetres: 2 km is 200,000 units; a twentieth of it wide, in the middle, at the middle of the heights (16 m).
        assert!((p.size - 10_000.0).abs() < 1.0 && (p.offset[0] - 100_000.0).abs() < 1.0 && (p.offset[2] - 1_600.0).abs() < 2.0, "{p:?}");
    }

    #[test]
    fn dropped_files_become_nodes() {
        let dir = std::env::temp_dir().join("noiser-drop-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("rock.noisegraph"), Graph::default_graph().to_json()).unwrap();
        let mut ws = MeshWorkspace::default();
        let before = ws.graph.nodes.len();
        ws.drop_file(dir.join("rock.noisegraph"));
        ws.drop_file(dir.join("ball.obj"));
        ws.drop_file(dir.join("notes.txt")); // ignored
        assert_eq!(ws.graph.nodes.len(), before + 2);
        let kinds: Vec<_> = ws.graph.nodes.values().rev().take(2).map(|n| n.kind.clone()).collect();
        assert!(matches!(&kinds[0], NodeKind::LoadMesh(p) if p.path.ends_with("ball.obj")));
        assert!(matches!(&kinds[1], NodeKind::NoiseGraph(p) if p.file.ends_with("rock.noisegraph") && !p.embedded.is_empty()));
    }

    #[test]
    fn saving_over_a_graph_file_keeps_the_old_version_in_a_history_folder() {
        let dir = std::env::temp_dir().join("noiser-history-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("rock.noisegraph");
        crate::keep_previous_version(&file, "first"); // nothing to keep yet
        assert!(!dir.join("rock.noisegraph.history").exists());
        std::fs::write(&file, "first").unwrap();
        crate::keep_previous_version(&file, "first"); // unchanged: nothing to keep
        assert!(!dir.join("rock.noisegraph.history").exists());
        crate::keep_previous_version(&file, "second");
        let kept: Vec<_> = std::fs::read_dir(dir.join("rock.noisegraph.history")).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read_to_string(kept[0].path()).unwrap(), "first");
    }
}
