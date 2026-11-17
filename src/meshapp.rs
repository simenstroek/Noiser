//! The mesh side of the app: importing an OBJ, baking its distance grid on a worker thread, putting
//! the grid on the GPU for the preview, and the details-panel controls. (Exporting the displaced
//! mesh is in meshexport.rs.)

use crate::mesh::{Mesh, Normalization};
use crate::meshexport::{self, Displace, Format, Job, MeshEvaluator};
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
            let baked = Mesh::load_obj(&path).map(|mesh| {
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
        self.mesh = Some(MeshState { name: b.name, normalized: b.normalized, norm: b.norm, sdf: Arc::new(b.sdf), view, _texture: texture, res: b.res, pad: b.pad });
        self.mesh_gen += 1;
        if b.new_file {
            self.pv.prim = MESH_PRIM;
        }
    }

    /// The Mesh section of the details panel.
    pub fn mesh_props(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Mesh");
        if ui.button("Import mesh (OBJ)…").clicked() {
            if let Some(p) = rfd::FileDialog::new().add_filter("OBJ mesh", &["obj"]).pick_file() {
                self.start_mesh_load(p);
            }
        }
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
        let reach = view::push_range(self.graph.density_range(), self.pv.amount);
        let reach = reach.0.abs().max(reach.1.abs());
        let (name, tris, dims, grid_res, grid_pad) = (m.name.clone(), m.normalized.triangles.len(), m.sdf.dims, m.res, m.pad);
        ui.label(format!("{name}: {tris} triangles"));
        ui.weak(format!("distance grid {}×{}×{}", dims[0], dims[1], dims[2]));
        crate::row(ui, &mut self.mesh_res, DEFAULT_RES, |ui, v| _ = ui.add(egui::Slider::new(v, 32..=384).text("grid resolution")));
        crate::row(ui, &mut self.mesh_pad, DEFAULT_PAD, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=2.0).text("grid padding")));
        if (self.mesh_res, self.mesh_pad) != (grid_res, grid_pad) && self.mesh_job.is_none() && ui.button("Rebuild distance grid").clicked() {
            self.start_mesh_rebake();
        }
        // The displaced surface can only move as far as the grid extends past the mesh.
        if reach > grid_pad - 0.05 {
            ui.colored_label(crate::editor::ORANGE, format!("Displacement reaches {reach:.2} but the grid only extends {grid_pad:.2} past the mesh: the shape will be clipped. Raise the padding."));
        }
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
        let push = view::push_range(self.graph.density_range(), self.pv.amount);
        push.0.abs().max(push.1.abs()).min((grid_pad - 0.05).max(0.0))
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
            self.mesh_export_open = false;
            return;
        };
        let (lo, hi) = m.normalized.bounds();
        let displace = Displace { amount: self.pv.amount, tile_size: self.pv.tile_size, prim_size: self.pv.prim_size };
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
        let displace = Displace { amount: self.pv.amount, tile_size: self.pv.tile_size, prim_size: self.pv.prim_size };
        let reach = self.displacement_reach(m.pad);
        let (graph, seed, norm) = (self.graph.clone(), self.pv.seed, m.norm);
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let (view, origin, voxel) = (m.view.clone(), m.sdf.origin, m.sdf.voxel);
        let progress = meshexport::Progress::default();
        let p = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let r = MeshEvaluator::new(&device, &queue, &graph, seed, view, origin, voxel).and_then(|ev| {
                let job = Job { ev: &ev, d: displace, lo, hi, reach, norm, settings };
                job.run(&path, &p)
            });
            let _ = tx.send(r);
        });
        self.mesh_export_job = Some(MeshExportJob { progress, done: rx });
    }
}
