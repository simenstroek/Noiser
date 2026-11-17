//! The mesh side of the app: importing an OBJ, baking its distance grid on a worker thread, putting
//! the grid on the GPU for the preview, and the details-panel controls. (Exporting the displaced
//! mesh is in meshexport.rs.)

use crate::mesh::{Mesh, Normalization};
use crate::meshexport::{self, Displace, FileJob, Format, Job, MeshEvaluator, SeedMode};
use crate::meshsdf::MeshSdf;
use crate::{App, view};
use eframe::{egui, wgpu};
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, mpsc};

pub const DEFAULT_RES: u32 = 128;
pub const DEFAULT_PAD: f32 = 0.5;

/// The mesh's index in the preview's shape list (after the primitives); raymarch.wgsl switches on it.
pub const MESH_PRIM: u32 = 7;

/// A loaded mesh and its distance grid, on the CPU (for export) and on the GPU (for the preview).
pub struct MeshState {
    pub name: String,
    /// The welded mesh scaled to a bounding sphere of radius 1, and how to get back to object space.
    pub normalized: Arc<Mesh>,
    pub norm: Normalization,
    /// The normalized mesh's bounds.
    pub bounds: (crate::mesh::V3, crate::mesh::V3),
    pub sdf: Arc<MeshSdf>,
    pub view: wgpu::TextureView,
    _texture: wgpu::Texture,
    /// The settings the grid was baked with.
    pub res: u32,
    pub pad: f32,
}

/// A distance grid being baked on a worker thread.
pub struct MeshJob {
    label: String,
    done: mpsc::Receiver<Result<Baked, String>>,
}

struct Baked {
    name: String,
    normalized: Arc<Mesh>,
    norm: Normalization,
    sdf: MeshSdf,
    res: u32,
    pad: f32,
    /// A newly loaded file (show it) rather than a rebake of the current one.
    new_file: bool,
}

impl App {
    /// Loads an OBJ and bakes its distance grid, without blocking the UI.
    pub fn start_mesh_load(&mut self, path: PathBuf) {
        let (res, pad) = (self.mesh_res, self.mesh_pad);
        let name = path.file_stem().map_or("mesh".into(), |n| n.to_string_lossy().into_owned());
        let (tx, rx) = mpsc::channel();
        let label = format!("Loading {name}…");
        std::thread::spawn(move || {
            let baked = crate::meshformats::load(&path).map(|mesh| {
                let (normalized, norm) = mesh.normalized();
                let sdf = MeshSdf::build(&normalized, res, pad, &AtomicU32::new(0));
                Baked { name, normalized: Arc::new(normalized), norm, sdf, res, pad, new_file: true }
            });
            let _ = tx.send(baked);
        });
        self.mesh_job = Some(MeshJob { label, done: rx });
    }

    /// Bakes the current mesh again at the resolution and padding now set in the panel.
    pub fn start_mesh_rebake(&mut self) {
        let Some(m) = &self.mesh else { return };
        let (res, pad) = (self.mesh_res, self.mesh_pad);
        let (name, normalized, norm) = (m.name.clone(), m.normalized.clone(), m.norm);
        let (tx, rx) = mpsc::channel();
        let label = format!("Rebuilding the distance grid of {name}…");
        std::thread::spawn(move || {
            let sdf = MeshSdf::build(&normalized, res, pad, &AtomicU32::new(0));
            let _ = tx.send(Ok(Baked { name, normalized, norm, sdf, res, pad, new_file: false }));
        });
        self.mesh_job = Some(MeshJob { label, done: rx });
    }

    /// Takes a finished bake and puts its grid on the GPU.
    pub fn poll_mesh_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.mesh_job else { return };
        let baked = match job.done.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Err(mpsc::TryRecvError::Disconnected) => Err("the mesh worker stopped unexpectedly".into()),
        };
        self.mesh_job = None;
        let b = match baked {
            Ok(b) => b,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        let [nx, ny, nz] = b.sdf.dims;
        let (texture, view) = b.sdf.upload(&self.rs.device, &self.rs.queue);
        self.status = format!("{}: {} triangles, distance grid {nx}×{ny}×{nz}", b.name, b.normalized.triangles.len());
        let bounds = b.normalized.bounds();
        self.mesh = Some(MeshState { name: b.name, normalized: b.normalized, norm: b.norm, bounds, sdf: Arc::new(b.sdf), view, _texture: texture, res: b.res, pad: b.pad });
        self.mesh_gen += 1;
        if b.new_file {
            self.pv.prim = MESH_PRIM;
        }
    }

    /// The Mesh section of the details panel.
    pub fn mesh_props(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Mesh");
        ui.horizontal(|ui| {
            if ui.button("Import mesh (OBJ)…").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Mesh (OBJ, PLY, STL)", &["obj", "ply", "stl"]).pick_file() {
                    self.start_mesh_load(p);
                }
            }
            if ui.button("Batch…").on_hover_text("Shape many meshes with this graph").clicked() {
                self.batch_open = true;
            }
        });
        if let Some(job) = &self.mesh_job {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(&job.label);
            });
        }
        let Some(m) = &self.mesh else {
            ui.weak("Import a low-resolution mesh to shape it with the graph. It shows up as the Mesh shape in the 3D view.");
            return;
        };
        let reach = view::push_range(self.graph.density_range(), self.pv.amount, self.mesh_mask.range());
        let reach = reach.0.abs().max(reach.1.abs());
        let (name, tris, dims, grid_res, grid_pad) = (m.name.clone(), m.normalized.triangles.len(), m.sdf.dims, m.res, m.pad);
        ui.label(format!("{name}: {tris} triangles"));
        ui.weak(format!("distance grid {}×{}×{}", dims[0], dims[1], dims[2]));
        if self.mesh.as_ref().is_some_and(|m| m.sdf.open) {
            ui.weak("Open surface (terrain, sheet): the side its normals face is outside. Raise the grid resolution for fine detail.");
        }
        crate::row(ui, &mut self.mesh_res, DEFAULT_RES, |ui, v| _ = ui.add(egui::Slider::new(v, 32..=384).text("grid resolution")));
        crate::row(ui, &mut self.mesh_pad, DEFAULT_PAD, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=2.0).text("grid padding")));
        if (self.mesh_res, self.mesh_pad) != (grid_res, grid_pad) && self.mesh_job.is_none() && ui.button("Rebuild distance grid").clicked() {
            self.start_mesh_rebake();
        }
        // The displaced surface can only move as far as the grid extends past the mesh.
        if reach > grid_pad - 0.05 {
            ui.colored_label(crate::editor::ORANGE, format!("Displacement reaches {reach:.2} but the grid only extends {grid_pad:.2} past the mesh: the shape will be clipped. Raise the padding."));
        }
        self.mask_ui(ui);
        if ui.button("Export displaced mesh…").clicked() {
            self.mesh_export_open = true;
        }
        ui.weak("Mesh units are scaled so the mesh fits a sphere of radius 1; the export scales back."); 
    }
}

/// A mesh export running on a worker thread.
pub struct MeshExportJob {
    progress: meshexport::Progress,
    done: mpsc::Receiver<Result<String, String>>,
}

impl App {
    /// How far the current displacement can move the surface, limited to what the distance grid covers.
    fn displacement_reach(&self, grid_pad: f32) -> f32 {
        meshexport::reach(&self.graph, self.pv.amount, self.mesh_mask.range(), grid_pad)
    }

    /// The Export displaced mesh window.
    pub fn mesh_export_window(&mut self, ctx: &egui::Context) {
        if let Some(job) = &self.mesh_export_job {
            match job.done.try_recv() {
                Ok(r) => {
                    self.status = r.unwrap_or_else(|e| format!("Mesh export failed: {e}"));
                    self.mesh_export_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.mesh_export_job = None,
            }
        }
        if !self.mesh_export_open {
            return;
        }
        let Some(m) = &self.mesh else {
            // Nothing to export yet; a mesh may still be loading.
            self.mesh_export_open = self.mesh_job.is_some() && self.mesh_export_open;
            return;
        };
        let (lo, hi) = m.normalized.bounds();
        let displace = Displace { amount: self.pv.amount, tile_size: self.pv.tile_size, prim_size: self.pv.prim_size, mask: self.mesh_mask.pack() };
        let reach = self.displacement_reach(m.pad);
        let name = m.name.clone();
        let mut open = true;
        let mut start = false;
        egui::Window::new("Export displaced mesh").open(&mut open).resizable(false).show(ctx, |ui| {
            let d = meshexport::Settings::default();
            let s = &mut self.mesh_export;
            crate::row(ui, &mut s.res, d.res, |ui, v| {
                ui.add(egui::DragValue::new(v).range(32..=1024).speed(4.0).suffix(" samples")).on_hover_text("Samples along the longest side of the exported region. The mesh gets as much detail as this allows.");
            });
            crate::row(ui, &mut s.format, d.format, |ui, v| {
                ui.selectable_value(v, Format::Obj, "OBJ");
                ui.selectable_value(v, Format::Ply, "PLY (binary)").on_hover_text("Much smaller and faster to write than OBJ for big meshes.");
            });
            crate::row(ui, &mut s.cull, d.cull, |ui, v| {
                ui.add(egui::Slider::new(v, 0.0..=0.25).fixed_decimals(3).text("remove small parts"))
                    .on_hover_text("Drops separate pieces whose surface area is below this fraction of the largest piece's.\nSeveral big pieces are all kept; 0 keeps everything.");
            });
            let grid = meshexport::Grid::plan(lo, hi, displace, reach, s.res);
            let [nx, ny, nz] = grid.dims;
            ui.weak(format!("{nx}×{ny}×{nz} samples ({:.0} million)", grid.samples() as f64 / 1e6));
            if grid.samples() > 400_000_000 {
                ui.colored_label(crate::editor::ORANGE, "⚠ This is a lot of samples: it will take a while and the result will be a huge mesh.");
            }
            ui.weak(format!("Displacement {:.2} × the graph, tile size {:.2}, shape size {:.2}, seed {}: the same as the 3D view.", displace.amount, displace.tile_size, displace.prim_size, self.pv.seed));
            ui.separator();
            match &self.mesh_export_job {
                Some(job) => {
                    let p = job.progress.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                    ui.add(egui::ProgressBar::new(p).show_percentage());
                }
                None => start = ui.button("Export…").clicked(),
            }
        });
        self.mesh_export_open = open;
        if start {
            self.start_mesh_export(&name);
        }
    }

    fn start_mesh_export(&mut self, name: &str) {
        let Some(m) = &self.mesh else { return };
        let settings = self.mesh_export.clone();
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Mesh", &[settings.format.extension()])
            .set_file_name(format!("{name}_displaced.{}", settings.format.extension()))
            .save_file()
        else {
            return;
        };
        let (lo, hi) = m.normalized.bounds();
        let displace = Displace { amount: self.pv.amount, tile_size: self.pv.tile_size, prim_size: self.pv.prim_size, mask: self.mesh_mask.pack() };
        let reach = self.displacement_reach(m.pad);
        let (graph, seed, norm) = (self.graph.clone(), self.pv.seed, m.norm);
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let (view, origin, voxel) = (m.view.clone(), m.sdf.origin, m.sdf.voxel);
        let progress = meshexport::Progress::default();
        let p = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let r = MeshEvaluator::new(&device, &queue, &graph, seed, view, origin, voxel, (lo, hi), displace.mask.count > 0).and_then(|ev| {
                let job = Job { ev: &ev, d: displace, lo, hi, reach, norm, settings };
                job.run(&path, &p)
            });
            let _ = tx.send(r);
        });
        self.mesh_export_job = Some(MeshExportJob { progress, done: rx });
    }
}

/// A batch running on a worker thread.
pub struct BatchJob {
    /// How many files are finished, how many there are, and how far the current one is (thousandths).
    index: Arc<std::sync::atomic::AtomicU32>,
    total: usize,
    file_progress: meshexport::Progress,
    name: Arc<std::sync::Mutex<String>>,
    done: mpsc::Receiver<String>,
}

/// `dir/stem.ext`, or `dir/stem_2.ext` and so on if that exists, so a batch never overwrites anything.
pub(crate) fn unique_path(dir: &std::path::Path, stem: &str, ext: &str) -> PathBuf {
    let mut path = dir.join(format!("{stem}.{ext}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem}_{n}.{ext}"));
        n += 1;
    }
    path
}

impl App {
    /// The Batch window: one graph, many meshes.
    pub fn batch_window(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        if let Some(job) = &self.batch_job {
            match job.done.try_recv() {
                Ok(summary) => {
                    self.status = summary;
                    self.batch_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => self.batch_job = None,
            }
        }
        if !self.batch_open {
            return;
        }
        let mut open = true;
        let mut start = false;
        egui::Window::new("Batch displace meshes").open(&mut open).resizable(false).show(ctx, |ui| {
            ui.label("Shapes every mesh with the same graph and writes a high-resolution mesh for each.");
            ui.horizontal(|ui| {
                if ui.button("Add meshes (OBJ)…").clicked() {
                    if let Some(files) = rfd::FileDialog::new().add_filter("Mesh (OBJ, PLY, STL)", &["obj", "ply", "stl"]).pick_files() {
                        for f in files {
                            if !self.batch_files.contains(&f) {
                                self.batch_files.push(f);
                            }
                        }
                    }
                }
                if ui.add_enabled(!self.batch_files.is_empty(), egui::Button::new("Clear")).clicked() {
                    self.batch_files.clear();
                }
            });
            egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                let mut remove = None;
                for (i, f) in self.batch_files.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui.small_button("×").on_hover_text("Remove from the batch").clicked() {
                            remove = Some(i);
                        }
                        ui.label(f.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())).on_hover_text(f.display().to_string());
                    });
                }
                if let Some(i) = remove {
                    self.batch_files.remove(i);
                }
                if self.batch_files.is_empty() {
                    ui.weak("No meshes yet.");
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Output folder…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.batch_dir = Some(dir);
                    }
                }
                match &self.batch_dir {
                    Some(d) => ui.label(d.display().to_string()),
                    None => ui.weak("none chosen"),
                };
            });
            ui.separator();
            crate::row(ui, &mut self.batch_seeds, SeedMode::Same, |ui, v| {
                ui.selectable_value(v, SeedMode::Same, "Same seed");
                ui.selectable_value(v, SeedMode::Increment, "Seed + 1 each");
                ui.selectable_value(v, SeedMode::Random, "Random each");
            });
            crate::row(ui, &mut self.batch_amount_var, 0.0, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).custom_formatter(|x, _| format!("±{:.0}%", x * 100.0)).text("vary displacement")));
            let base = self.pv.seed;
            ui.weak(match self.batch_seeds {
                SeedMode::Same => format!("Every mesh uses the global seed ({base})."),
                SeedMode::Increment => format!("The first mesh uses the global seed ({base}), the next {}, and so on.", base + 1),
                SeedMode::Random => format!("Each mesh gets a different seed, repeatable from the global seed ({base}). Seeds are in the file names."),
            });
            ui.weak("Displacement, tile size, shape size and the graph come from the current 3D view; samples, format and small-part removal from Export displaced mesh; the distance grid from the Mesh section. Existing files are never overwritten.");
            ui.separator();
            match &self.batch_job {
                Some(job) => {
                    let done = job.index.load(Ordering::Relaxed) as f32;
                    let p = (done + job.file_progress.load(Ordering::Relaxed) as f32 / 1000.0) / job.total.max(1) as f32;
                    ui.label(job.name.lock().map(|n| n.clone()).unwrap_or_default());
                    ui.add(egui::ProgressBar::new(p.min(1.0)).show_percentage());
                }
                None => {
                    let ready = !self.batch_files.is_empty() && self.batch_dir.is_some();
                    start = ui.add_enabled(ready, egui::Button::new(format!("Start ({} meshes)", self.batch_files.len()))).clicked();
                }
            }
        });
        self.batch_open = open;
        if start {
            self.start_batch();
        }
    }

    pub fn start_batch(&mut self) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let Some(dir) = self.batch_dir.clone() else { return };
        let files = self.batch_files.clone();
        let total = files.len();
        let (graph, base, mode, var) = (self.graph.clone(), self.pv.seed, self.batch_seeds, self.batch_amount_var);
        let d = Displace { amount: self.pv.amount, tile_size: self.pv.tile_size, prim_size: self.pv.prim_size, mask: self.mesh_mask.pack() };
        let settings = self.mesh_export.clone();
        let (grid_res, grid_pad) = (self.mesh_res, self.mesh_pad);
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let index = Arc::new(AtomicU32::new(0));
        let file_progress = meshexport::Progress::default();
        let name = Arc::new(std::sync::Mutex::new(String::new()));
        let (tx, rx) = mpsc::channel();
        let (idx, fp, nm) = (index.clone(), file_progress.clone(), name.clone());
        std::thread::spawn(move || {
            let mut written = 0;
            let mut failed: Vec<String> = Vec::new();
            for (i, input) in files.iter().enumerate() {
                idx.store(i as u32, Ordering::Relaxed);
                fp.store(0, Ordering::Relaxed);
                let stem = input.file_stem().map_or("mesh".into(), |s| s.to_string_lossy().into_owned());
                if let Ok(mut n) = nm.lock() {
                    *n = format!("{stem} ({} of {total})", i + 1);
                }
                let seed = mode.seed_for(base, i);
                let label = if mode == SeedMode::Same { stem.clone() } else { format!("{stem}_seed{seed}") };
                let output = unique_path(&dir, &format!("{label}_displaced"), settings.format.extension());
                let d = Displace { amount: d.amount * meshexport::amount_factor(base, i, var), ..d };
                let job = FileJob { device: &device, queue: &queue, graph: &graph, seed, d, settings: settings.clone(), grid_res, grid_pad };
                // One bad file (or a GPU error inside it) must not stop the rest.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(input, &output, &fp)));
                match result {
                    Ok(Ok(_)) => written += 1,
                    Ok(Err(e)) => failed.push(format!("{stem}: {e}")),
                    Err(_) => failed.push(format!("{stem}: the export crashed")),
                }
            }
            idx.store(total as u32, Ordering::Relaxed);
            let mut summary = format!("Batch finished: {written} of {total} meshes written to {}", dir.display());
            if !failed.is_empty() {
                summary += &format!(". Failed: {}", failed.join("; "));
            }
            let _ = tx.send(summary);
        });
        self.batch_job = Some(BatchJob { index, total, file_progress, name, done: rx });
    }
}

impl App {
    /// The Displacement mask section: terms that vary how strongly the noise displaces the mesh.
    fn mask_ui(&mut self, ui: &mut egui::Ui) {
        use crate::meshmask::{MAX_TERMS, MaskKind, MaskTerm};
        egui::CollapsingHeader::new("Displacement mask").default_open(!self.mesh_mask.terms.is_empty()).show(ui, |ui| {
            ui.weak("Varies how strongly the noise displaces each part of the mesh. With no terms every part moves equally. Strength is the sum of weight × term; every term is measured on the mesh itself. Applies to the Mesh shape.");
            ui.add_enabled(!self.mesh_mask.terms.is_empty(), egui::Checkbox::new(&mut self.pv.show_mask, "Show mask on the mesh"))
                .on_hover_text("Colours the surface by mask strength: blue 0, green 0.5, red 1; magenta above 1, black below 0. The preview only.");
            let mut remove = None;
            for (i, t) in self.mesh_mask.terms.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let before = t.kind;
                        egui::ComboBox::from_id_salt(("mask kind", i)).selected_text(t.kind.name()).show_ui(ui, |ui| {
                            for k in MaskKind::ALL {
                                ui.selectable_value(&mut t.kind, k, k.name());
                            }
                        });
                        if t.kind != before {
                            (t.lo, t.hi) = t.kind.default_range();
                        }
                        if ui.small_button("×").on_hover_text("Remove this term").clicked() {
                            remove = Some(i);
                        }
                    });
                    crate::row(ui, &mut t.weight, 1.0, |ui, v| _ = ui.add(egui::Slider::new(v, -2.0..=2.0).text("weight")));
                    if matches!(t.kind, MaskKind::Position | MaskKind::Normal) {
                        let word = if t.kind == MaskKind::Position { "along" } else { "facing" };
                        crate::row(ui, &mut t.axis, 1, |ui, v| {
                            ui.label(word);
                            for (a, name) in [(0, "X"), (1, "Y"), (2, "Z")] {
                                ui.selectable_value(v, a, name);
                            }
                        });
                        crate::row(ui, &mut t.negative, false, |ui, v| _ = ui.checkbox(v, "negative direction"));
                    }
                    if t.kind != MaskKind::Constant {
                        let (dlo, dhi) = t.kind.default_range();
                        let speed = if t.kind == MaskKind::Curvature { 0.05 } else { 0.01 };
                        crate::row(ui, &mut (t.lo, t.hi), (dlo, dhi), |ui, v| {
                            ui.add(egui::DragValue::new(&mut v.0).speed(speed).prefix("0 at "));
                            ui.add(egui::DragValue::new(&mut v.1).speed(speed).prefix("1 at "));
                        });
                        crate::row(ui, &mut t.invert, false, |ui, v| _ = ui.checkbox(v, "invert"));
                    }
                    ui.weak(t.kind.help());
                });
            }
            if let Some(i) = remove {
                self.mesh_mask.terms.remove(i);
            }
            if self.mesh_mask.terms.len() < MAX_TERMS {
                ui.menu_button("Add term", |ui| {
                    for k in MaskKind::ALL {
                        if ui.button(k.name()).on_hover_text(k.help()).clicked() {
                            self.mesh_mask.terms.push(MaskTerm::new(k));
                            ui.close();
                        }
                    }
                });
            }
            if self.mesh_mask.range() != (1.0, 1.0) {
                let (lo, hi) = self.mesh_mask.range();
                ui.weak(format!("Strength ranges from {lo:.2} to {hi:.2}."));
            }
        });
    }
}
