// No console window behind the release build on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backup;
mod bluenoise;
mod cache;
mod codegen;
mod editor;
mod export;
mod graph;
mod history;
mod mesh;
mod meshapp;
mod meshmask;
mod meshexport;
mod meshsdf;
mod surface;
mod noiser_file;
#[cfg(test)]
mod gpu_tests;
mod vdb;
mod view;

use eframe::{egui, egui_wgpu};
use graph::{Graph, MathOp, NodeKind, Tiling};
use view::{MAX_PARAMS, Uniforms};

/// Primitive names, in the order `unit_prim` in raymarch.wgsl switches on.
const PRIMS: [&str; 7] = ["Sphere", "Box", "Round box", "Torus", "Cylinder", "Capsule", "Octahedron"];

/// Settings for the 3D and slice views (not part of the graph file).
#[derive(Clone, PartialEq)]
struct Preview {
    amount: f32,
    tile_size: f32,
    seed: u32,
    step_scale: f32,
    max_steps: u32,
    /// Fraction of the view's pixels each side the 3D view is rendered at (viewport only).
    render_scale: f32,
    /// Viewport material (never exported).
    color: [f32; 3],
    specular: f32,
    roughness: f32,
    shadows: bool,
    prim: u32,
    prim_size: f32,
    cloud: bool,
    cloud_density: f32,
    slice_axis: u32,
    slice_pos: f32,
    heatmap: bool,
    tiles_2x2: bool,
}

impl Default for Preview {
    fn default() -> Self {
        Self {
            amount: 0.25,
            tile_size: 1.0,
            seed: 0,
            // Overshoot is caught and refined, so this only needs to avoid stepping through thin features.
            step_scale: 0.9,
            max_steps: 300,
            render_scale: 1.0,
            color: [0.8, 0.78, 0.74],
            specular: 0.0,
            roughness: 0.5,
            shadows: false,
            prim: 0,
            prim_size: 1.0,
            cloud: false,
            cloud_density: 8.0,
            slice_axis: 2,
            slice_pos: 0.5,
            heatmap: false,
            tiles_2x2: false,
        }
    }
}

struct App {
    rs: egui_wgpu::RenderState,
    graph: Graph,
    editor: editor::Editor,
    /// File the graph was last opened from / saved to.
    path: Option<std::path::PathBuf>,
    status: String,
    /// WGSL of the installed preview pipeline, and the graph parameters laid out for it.
    built: Option<(String, Vec<f32>)>,
    /// A pipeline being built on a worker thread, for this WGSL.
    compiling: Option<(String, std::sync::mpsc::Receiver<view::Gpu>)>,
    yaw: f32,
    pitch: f32,
    dist: f32,
    pv: Preview,
    /// This run's backup session; None if the backup folder couldn't be created.
    session: Option<backup::Session>,
    /// Latest backup from sessions that didn't exit cleanly, waiting for the user to restore or discard it.
    recovery: Option<(backup::Backup, Vec<std::path::PathBuf>)>,
    /// The graph as last opened, saved or created. Backups only run while the graph differs from it,
    /// so an untouched session never shadows a crashed session's real work.
    saved: Graph,
    export_open: bool,
    export: export::Settings,
    export_job: Option<ExportJob>,
    caches: cache::Caches,
    /// The loaded mesh and its distance grid, and a bake running on a worker thread.
    mesh: Option<meshapp::MeshState>,
    mesh_job: Option<meshapp::MeshJob>,
    /// Bumped whenever the grid on the GPU changes, so a freshly built pipeline knows to rebind it.
    mesh_gen: u64,
    /// Grid settings for the next bake (see the Mesh section of the panel).
    mesh_res: u32,
    mesh_pad: f32,
    mesh_export_open: bool,
    mesh_export: meshexport::Settings,
    mesh_export_job: Option<meshapp::MeshExportJob>,
    batch_open: bool,
    batch_files: Vec<std::path::PathBuf>,
    batch_dir: Option<std::path::PathBuf>,
    batch_seeds: meshexport::SeedMode,
    batch_job: Option<meshapp::BatchJob>,
    /// How strongly the noise displaces each part of a mesh (see meshmask.rs).
    mesh_mask: meshmask::MaskSettings,
    history: history::History,
    /// Most recently opened or saved graph files, newest first (kept across runs).
    recent: Vec<std::path::PathBuf>,
    shortcuts_open: bool,
    prefs: Prefs,
    prefs_open: bool,
}

/// An export running on a worker thread.
struct ExportJob {
    progress: export::Progress,
    done: std::sync::mpsc::Receiver<Result<String, String>>,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        // `noiser path.noisegraph` opens that graph.
        // `noiser [graph.noisegraph] [--mesh file.obj] [--open-mesh-export] [--batch-out DIR --batch-file FILE ...] [--mask KIND ...]`
        let mut args = std::env::args().skip(1);
        let (mut graph_arg, mut mesh_arg, mut open_mesh_export) = (None, None, false);
        let (mut batch_dir, mut batch_files): (Option<std::path::PathBuf>, Vec<std::path::PathBuf>) = (None, Vec::new());
        let mut mask_terms = Vec::new();
        while let Some(a) = args.next() {
            if a == "--mesh" {
                mesh_arg = args.next().map(std::path::PathBuf::from);
            } else if a == "--open-mesh-export" {
                open_mesh_export = true;
            } else if a == "--mask" {
                // A displacement-mask term of the named kind with its default settings.
                let name = args.next().unwrap_or_default().to_lowercase();
                mask_terms.extend(meshmask::MaskKind::ALL.iter().find(|k| k.name().to_lowercase().starts_with(&name)).map(|k| meshmask::MaskTerm::new(*k)));
            } else if a == "--batch-out" {
                batch_dir = args.next().map(std::path::PathBuf::from);
            } else if a == "--batch-file" {
                batch_files.extend(args.next().map(std::path::PathBuf::from));
            } else if graph_arg.is_none() {
                graph_arg = Some(std::path::PathBuf::from(a));
            }
        }
        let (graph, path, status) = match graph_arg {
            Some(p) => match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
                Ok(g) => (g, Some(p.clone()), format!("Opened {}", p.display())),
                Err(e) => (Graph::default_graph(), None, format!("Could not open {}: {e}", p.display())),
            },
            None => (Graph::default_graph(), None, String::new()),
        };
        let prefs: Prefs = cc.storage.and_then(|s| s.get_string(PREFS_KEY)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let prefs = prefs.sane();
        cc.egui_ctx.set_zoom_factor(prefs.ui_scale);
        let mut editor = editor::Editor::default();
        // Wire-drop menu pick counts survive restarts.
        if let Some(picks) = cc.storage.and_then(|s| s.get_string(PICKS_KEY)).and_then(|s| serde_json::from_str(&s).ok()) {
            editor.picks = picks;
        }
        // Look for crashed sessions before starting ours.
        let root = backup::root();
        let recovery = root.as_deref().and_then(|r| {
            let dirs = backup::crashed(r);
            match backup::latest(&dirs) {
                Some(b) => Some((b, dirs)),
                None => {
                    backup::remove(&dirs); // crashed before anything was backed up
                    None
                }
            }
        });
        let session = root.and_then(|r| backup::Session::start(&r).ok());
        let saved_for_history = graph.clone();
        let mut app = Self {
            rs: cc.wgpu_render_state.clone().expect("Noiser needs the wgpu renderer"),
            saved: graph.clone(),
            graph,
            editor,
            path,
            status,
            built: None,
            compiling: None,
            yaw: 0.6,
            pitch: 0.35,
            dist: 3.5,
            pv: Preview::default(),
            session,
            recovery,
            export_open: false,
            export: export::Settings::default(),
            export_job: None,
            caches: cache::Caches::default(),
            mesh: None,
            mesh_job: None,
            mesh_gen: 0,
            mesh_res: meshapp::DEFAULT_RES,
            mesh_pad: meshapp::DEFAULT_PAD,
            mesh_export_open: false,
            mesh_export: meshexport::Settings::default(),
            mesh_export_job: None,
            batch_open: false,
            batch_files: Vec::new(),
            batch_dir: None,
            batch_seeds: meshexport::SeedMode::Same,
            batch_job: None,
            mesh_mask: meshmask::MaskSettings::default(),
            history: history::History::new(&saved_for_history),
            shortcuts_open: false,
            prefs: prefs.clone(),
            prefs_open: false,
            recent: cc.storage.and_then(|s| s.get_string(RECENT_KEY)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default(),
        };
        if let Some(p) = mesh_arg {
            app.start_mesh_load(p);
        }
        app.mesh_mask.terms = mask_terms;
        app.mesh_export_open = open_mesh_export; // for looking at the dialog without clicking
        // `--batch-out DIR --batch-file A.obj --batch-file B.obj` starts a batch right away.
        if batch_dir.is_some() && !batch_files.is_empty() {
            app.batch_dir = batch_dir;
            app.batch_files = batch_files;
            app.batch_open = true;
            app.start_batch();
        }
        app
    }

    fn export_window(&mut self, ctx: &egui::Context) {
        use export::{Format, Kind, Layout, Levels, Output};
        // Finish a running export.
        if let Some(job) = &self.export_job {
            match job.done.try_recv() {
                Ok(r) => {
                    self.status = r.unwrap_or_else(|e| format!("Export failed: {e}"));
                    self.export_job = None;
                }
                Err(_) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            }
        }
        if !self.export_open {
            return;
        }
        let mut open = true;
        let mut start = false;
        egui::Window::new("Export").open(&mut open).resizable(false).show(ctx, |ui| {
            let s = &mut self.export;
            let d = export::Settings::default();
            row(ui, &mut s.kind, d.kind, |ui, v| {
                ui.selectable_value(v, Kind::Image2d, "2D image");
                ui.selectable_value(v, Kind::SliceGrid, "3D slice grid");
                ui.selectable_value(v, Kind::Vdb, "3D OpenVDB");
            });
            row(ui, &mut s.output, d.output, |ui, v| {
                ui.selectable_value(v, Output::Density, "Density");
                ui.selectable_value(v, Output::Normal, "Normal");
            });
            let res_default = if s.kind == Kind::Image2d { 1024 } else { 128 };
            row(ui, &mut s.res, res_default, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1..=16384).suffix(" px").speed(4.0)).on_hover_text("Texels per tile side"));
            if s.kind != Kind::Vdb {
                row(ui, &mut s.format, d.format, |ui, v| {
                    ui.selectable_value(v, Format::Png8, "PNG 8-bit");
                    ui.selectable_value(v, Format::Png16, "PNG 16-bit");
                    ui.selectable_value(v, Format::Exr32, "EXR 32-bit");
                });
                if s.format != Format::Exr32 && s.output == Output::Density {
                    row(ui, &mut s.levels, d.levels, |ui, v| {
                        ui.selectable_value(v, Levels::GraphRange, "Graph range").on_hover_text("0–1 covers the graph's guaranteed range; the same for every export");
                        ui.selectable_value(v, Levels::DataMinMax, "Data min/max").on_hover_text("0–1 covers this export's own values");
                    });
                }
            }
            if s.kind == Kind::SliceGrid {
                row(ui, &mut s.layout, d.layout, |ui, v| {
                    ui.selectable_value(v, Layout::Square, "Square grid");
                    ui.selectable_value(v, Layout::Vertical, "Vertical strip");
                });
            }
            match s.kind {
                Kind::Image2d => _ = ui.weak(format!("Slice {} = {:.3}, from the slice view.", ["X", "Y", "Z"][self.pv.slice_axis as usize], self.pv.slice_pos)),
                Kind::Vdb => _ = ui.weak("Float grid \"density\" with raw signed values, 1 tile = 1 unit."),
                Kind::SliceGrid => {}
            }
            if s.format == Format::Exr32 && s.kind != Kind::Vdb {
                ui.weak("EXR keeps raw signed values.");
            }
            let (file, _) = s.estimate();
            let dims = s.image_size().map_or(format!("{0}×{0}×{0} voxels", s.res), |(w, h)| format!("{w}×{h} px"));
            ui.label(format!("{dims}, about {:.1} MB", file as f64 / (1 << 20) as f64));
            for w in s.warnings() {
                ui.colored_label(editor::ORANGE, format!("⚠ {w}"));
            }
            ui.separator();
            match &self.export_job {
                Some(job) => {
                    let p = job.progress.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                    ui.add(egui::ProgressBar::new(p).show_percentage());
                }
                None => start = ui.button("Export…").clicked(),
            }
        });
        self.export_open = open;
        if start {
            self.start_export();
        }
    }

    fn start_export(&mut self) {
        let mut s = self.export.clone();
        s.seed = self.pv.seed;
        s.axis = self.pv.slice_axis;
        s.pos = self.pv.slice_pos;
        let name = self.path.as_ref().and_then(|p| p.file_stem()).map_or("noise".into(), |n| n.to_string_lossy().into_owned());
        let Some(path) = rfd::FileDialog::new().add_filter("Export", &[s.extension()]).set_file_name(format!("{name}.{}", s.extension())).save_file() else { return };
        // Export always evaluates the full graph: no preview, no caches. Compiling happens on the
        // worker thread too, since the driver can take seconds.
        let graph = self.graph.clone();
        let range = graph.density_range();
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let progress = export::Progress::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let p = progress.clone();
        std::thread::spawn(move || {
            let r = export::Evaluator::new(&device, &queue, &graph, s.seed).and_then(|ev| export::run(&ev, &s, range, &path, &p));
            let _ = tx.send(r);
        });
        self.export_job = Some(ExportJob { progress, done: rx });
    }

    /// Writes the `.noiser` math-function file. Its test vectors come from the GPU, so this runs
    /// on a worker thread like texture export.
    fn export_function(&mut self) {
        let name = self.path.as_ref().and_then(|p| p.file_stem()).map_or("noise".into(), |n| n.to_string_lossy().into_owned());
        let Some(path) = rfd::FileDialog::new().add_filter("Noiser math function", &["noiser"]).set_file_name(format!("{name}.noiser")).save_file() else { return };
        let (graph, seed) = (self.graph.clone(), self.pv.seed);
        let (device, queue) = (self.rs.device.clone(), self.rs.queue.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = export::Evaluator::new(&device, &queue, &graph, seed)
                .and_then(|ev| noiser_file::file(&graph, seed, |pts| ev.eval_points(pts)))
                .and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()))
                .map(|()| format!("Exported {}", path.display()));
            let _ = tx.send(r);
        });
        self.export_job = Some(ExportJob { progress: export::Progress::default(), done: rx });
    }

    /// Asks whether to restore the backup of a session that didn't exit cleanly.
    fn recovery_prompt(&mut self, ctx: &egui::Context) {
        let Some((b, _)) = &self.recovery else { return };
        let ago = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()).saturating_sub(b.saved_unix);
        let ago = match ago {
            0..60 => format!("{ago} s ago"),
            60..3600 => format!("{} min ago", ago / 60),
            _ => format!("{} h ago", ago / 3600),
        };
        let file = b.path.as_ref().map_or("an unsaved graph".to_string(), |p| p.display().to_string());
        let mut choice = None;
        egui::Modal::new(egui::Id::unique("recovery")).show(ctx, |ui| {
            ui.heading("Noiser didn't close properly");
            ui.label(format!("A backup of {file} from {ago} is available."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Restore backup").clicked() {
                    choice = Some(true);
                }
                if ui.button("Discard").clicked() {
                    choice = Some(false);
                }
            });
        });
        let Some(restore) = choice else { return };
        let (b, dirs) = self.recovery.take().unwrap();
        if restore {
            self.graph = b.graph;
            self.path = b.path;
            self.editor.selected.clear();
            self.editor.preview = None;
            self.status = "Restored from backup (not saved yet)".into();
        }
        backup::remove(&dirs);
    }

    /// `range` is the density range of the graph being shown (the preview node's, when previewing).
    fn uniforms(&self, view3d: egui::Rect, slice: egui::Rect, range: (f32, f32)) -> Uniforms {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let pos = [self.dist * cp * sy, self.dist * sp, self.dist * cp * cy];
        let fwd = norm([-pos[0], -pos[1], -pos[2]]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let pv = &self.pv;
        let aspect = |r: egui::Rect| r.width() / r.height().max(1.0);
        // How far the surface can move: amount × density range. Kept finite for the shader;
        // 1e6 effectively disables the shell skip for unbounded graphs.
        // The displacement mask only applies to a mesh.
        let mesh_on = pv.prim == meshapp::MESH_PRIM && self.mesh.is_some();
        let mask = if mesh_on { self.mesh_mask.pack() } else { meshmask::PackedMask::NONE };
        let push = view::push_range(range, pv.amount, mask.range);
        let (mesh_origin, mesh_voxel) = self.mesh.as_ref().map_or(([0.0; 3], 1.0), |m| (m.sdf.origin, m.sdf.voxel));
        let (mesh_lo, mesh_hi) = self.mesh.as_ref().map_or(([0.0; 3], [0.0; 3]), |m| m.bounds);
        Uniforms {
            cam_pos: pos,
            aspect: aspect(view3d),
            cam_fwd: fwd,
            amount: pv.amount,
            cam_right: right,
            step_scale: pv.step_scale,
            cam_up: up,
            seed: pv.seed,
            tile_size: pv.tile_size,
            prim_size: pv.prim_size,
            // Logical points are fine here: only the ratio to the view height matters.
            pixel: 2.0 / (view3d.height() * pv.render_scale).max(1.0),
            cloud_density: pv.cloud_density,
            mode: pv.cloud as u32,
            prim: if pv.prim == meshapp::MESH_PRIM && self.mesh.is_none() { 0 } else { pv.prim },
            slice_axis: pv.slice_axis,
            slice_mode: pv.heatmap as u32,
            slice_pos: pv.slice_pos,
            slice_aspect: aspect(slice),
            slice_tiles: if pv.tiles_2x2 { 2.0 } else { 1.0 },
            slice_res: view::SLICE_RES,
            push_min: push.0,
            push_max: push.1,
            _pad: [pv.max_steps as f32, 0.0],
            mesh_origin,
            mesh_voxel,
            mesh_lo,
            mask_n: mask.count,
            mesh_hi,
            _pad_mask: 0.0,
            mask: std::array::from_fn(|i| mask.terms[i]),
            look: [[pv.color[0], pv.color[1], pv.color[2], pv.specular], [pv.roughness, pv.shadows as u32 as f32, 0.0, 0.0]],
        }
    }

    fn undo(&mut self, redo: bool) {
        let g = if redo { self.history.redo(&self.graph) } else { self.history.undo(&self.graph) };
        if let Some(g) = g {
            self.graph = g;
            self.editor.selected.retain(|id| self.graph.nodes.contains_key(id));
        }
    }

    fn file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new().add_filter("Noiser graph", &["noisegraph"])
    }

    fn open(&mut self) {
        let Some(path) = Self::file_dialog().pick_file() else { return };
        self.open_path(path);
    }

    fn open_path(&mut self, path: std::path::PathBuf) {
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
            Ok(g) => {
                self.saved = g.clone();
                self.graph = g;
                self.editor.selected.clear();
                self.editor.preview = None;
                self.status = format!("Opened {}", path.display());
                push_recent(&mut self.recent, path.clone());
                self.path = Some(path);
            }
            Err(e) => self.status = format!("Could not open {}: {e}", path.display()),
        }
    }

    fn save(&mut self, ask: bool) {
        let path = match (&self.path, ask) {
            (Some(p), false) => p.clone(),
            _ => match Self::file_dialog().set_file_name("untitled.noisegraph").save_file() {
                Some(p) => p,
                None => return,
            },
        };
        match std::fs::write(&path, self.graph.to_json()) {
            Ok(()) => {
                self.saved = self.graph.clone();
                self.status = format!("Saved {}", path.display());
                push_recent(&mut self.recent, path.clone());
                self.path = Some(path);
            }
            Err(e) => self.status = format!("Could not save {}: {e}", path.display()),
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Small icon button that puts `value` back to `default`; disabled while it already is.
fn reset<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T) {
    let icon = egui::Image::new(egui::include_image!("../assets/icons/restart-alt.svg"))
        .fit_to_exact_size(egui::vec2(14.0, 14.0))
        .tint(ui.visuals().text_color());
    let btn = ui.add_enabled(*value != default, egui::Button::image(icon).frame(false));
    if btn.on_hover_text("Reset to default").clicked() {
        *value = default;
    }
}

/// A row with a reset button in front of the widget.
fn row<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, default: T, widget: impl FnOnce(&mut egui::Ui, &mut T)) {
    ui.horizontal(|ui| {
        reset(ui, value, default);
        widget(ui, value);
    });
}

fn preview_props(ui: &mut egui::Ui, pv: &mut Preview, mesh_loaded: bool) {
    let d = Preview::default();
    ui.heading("3D view");
    row(ui, &mut pv.cloud, d.cloud, |ui, v| {
        ui.selectable_value(v, false, "Solid");
        ui.selectable_value(v, true, "Cloud");
    });
    row(ui, &mut pv.prim, d.prim, |ui, v| {
        let shown = PRIMS.get(*v as usize).copied().unwrap_or("Mesh");
        egui::ComboBox::from_label("shape").selected_text(shown).show_ui(ui, |ui| {
            for (i, name) in PRIMS.iter().enumerate() {
                ui.selectable_value(v, i as u32, *name);
            }
            if mesh_loaded {
                ui.selectable_value(v, meshapp::MESH_PRIM, "Mesh");
            }
        });
    });
    row(ui, &mut pv.prim_size, d.prim_size, |ui, v| _ = ui.add(egui::Slider::new(v, 0.25..=2.0).text("shape size")));
    row(ui, &mut pv.amount, d.amount, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("displacement")));
    row(ui, &mut pv.tile_size, d.tile_size, |ui, v| _ = ui.add(egui::Slider::new(v, 0.25..=4.0).logarithmic(true).text("tile size")));
    row(ui, &mut pv.seed, d.seed, |ui, v| _ = ui.add(egui::Slider::new(v, 0..=100).text("global seed")));
    if pv.cloud {
        row(ui, &mut pv.cloud_density, d.cloud_density, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=64.0).logarithmic(true).text("cloud density")));
    } else {
        row(ui, &mut pv.step_scale, d.step_scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=1.0).text("step safety")));
        row(ui, &mut pv.render_scale, d.render_scale, |ui, v| {
            ui.horizontal(|ui| {
                ui.label("render resolution");
                for (s, label) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                    ui.selectable_value(v, s, label);
                }
            });
        });
        row(ui, &mut pv.max_steps, d.max_steps, |ui, v| _ = ui.add(egui::Slider::new(v, 32..=600).logarithmic(true).text("max march steps")));
        ui.separator();
        ui.heading("Material (viewport only)");
        row(ui, &mut pv.color, d.color, |ui, v| _ = ui.color_edit_button_rgb(v));
        row(ui, &mut pv.specular, d.specular, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("specular")));
        row(ui, &mut pv.roughness, d.roughness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.05..=1.0).text("roughness")));
        row(ui, &mut pv.shadows, d.shadows, |ui, v| _ = ui.checkbox(v, "soft shadows"));
    }
    ui.separator();
    ui.heading("Slice view");
    row(ui, &mut pv.slice_axis, d.slice_axis, |ui, v| {
        ui.label("axis");
        for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
            ui.selectable_value(v, i as u32, *name);
        }
    });
    row(ui, &mut pv.slice_pos, d.slice_pos, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("position")));
    row(ui, &mut pv.heatmap, d.heatmap, |ui, v| {
        ui.selectable_value(v, false, "Grayscale");
        ui.selectable_value(v, true, "Heatmap");
    });
    row(ui, &mut pv.tiles_2x2, d.tiles_2x2, |ui, v| _ = ui.checkbox(v, "show 2×2 tiles"));
}

/// The per-node cache controls: preview-only, so they sit apart from the node's own parameters.
fn cache_props(ui: &mut egui::Ui, graph: &mut Graph, id: graph::NodeId, tiling: &std::collections::BTreeMap<graph::NodeId, Tiling>, total_bytes: u64) {
    ui.separator();
    let eligible = graph.cache_eligibility(id, tiling);
    let auto = {
        // Resolution "Auto" would pick, whatever is stored now.
        let mut probe = graph.clone();
        probe.nodes.get_mut(&id).unwrap().cache = Some(0);
        probe.cache_resolution(id)
    };
    let res = graph.cache_resolution(id);
    let node = graph.nodes.get_mut(&id).unwrap();
    let mut on = node.cache.is_some();
    ui.horizontal(|ui| {
        reset(ui, &mut node.cache, None);
        let r = ui.add_enabled(eligible.is_ok(), egui::Checkbox::new(&mut on, "Cache in preview"));
        let tip = match &eligible {
            Ok(()) => "Bakes everything upstream of this node into a 3D texture so the preview stays fast.\nExport always uses the full graph.".to_string(),
            Err(why) => format!("Can't cache: {why}"),
        };
        r.on_hover_text(tip);
    });
    if on != node.cache.is_some() {
        node.cache = on.then_some(0);
    }
    if let (Some(cur), Ok(())) = (node.cache, &eligible) {
        let label = |r: u32| if r == 0 { format!("Auto ({auto}³)") } else { format!("{r}³") };
        let mut sel = cur;
        ui.horizontal(|ui| {
            reset(ui, &mut sel, 0);
            egui::ComboBox::from_label("resolution").selected_text(label(sel)).show_ui(ui, |ui| {
                for r in [0, 32, 64, 128, 256, 512, 1024] {
                    ui.selectable_value(&mut sel, r, label(r));
                }
            });
        });
        node.cache = Some(sel);
        let mb = |b: u64| b as f64 / (1 << 20) as f64;
        ui.weak(format!("{:.0} MB for this cache · {:.0} MB for all", mb((res as u64).pow(3) * 4), mb(total_bytes)));
        if res >= 1024 {
            ui.colored_label(editor::ORANGE, "⚠ 1024³ needs 4 GB of GPU memory");
        }
    }
    if let Err(why) = &eligible {
        ui.weak(format!("Not cacheable: {why}"));
    }
}

/// Properties of one node.
fn node_props(ui: &mut egui::Ui, kind: &mut NodeKind) {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => {
            let is_vec = matches!(kind, NodeKind::Vec3(_));
            ui.horizontal(|ui| {
                if ui.selectable_label(!is_vec, "Float").clicked() && is_vec {
                    if let NodeKind::Vec3(v) = kind {
                        *kind = NodeKind::Float(v[0]);
                    }
                }
                if ui.selectable_label(is_vec, "Vector").clicked() && !is_vec {
                    if let NodeKind::Float(v) = kind {
                        *kind = NodeKind::Vec3([*v; 3]);
                    }
                }
            });
            let drag = |ui: &mut egui::Ui, x: &mut f32| _ = ui.add(egui::DragValue::new(x).speed(0.01));
            match kind {
                NodeKind::Float(v) => row(ui, v, 1.0, drag),
                NodeKind::Vec3(v) => row(ui, v, [1.0; 3], |ui, v| v.iter_mut().for_each(|x| drag(ui, x))),
                _ => {}
            }
        }
        NodeKind::Math { op, .. } => {
            row(ui, op, MathOp::Add, |ui, op| {
                egui::ComboBox::from_label("operation").selected_text(op.label()).show_ui(ui, |ui| {
                    for o in MathOp::ALL {
                        ui.selectable_value(op, o, o.label());
                    }
                });
            });
            if *op == MathOp::Ridge {
                ui.weak("1 − |a|. Ridge of noise gives ridged noise; Absolute gives billow.");
            }
        }
        NodeKind::Remap { clamp, .. } => {
            row(ui, clamp, false, |ui, c| _ = ui.checkbox(c, "clamp to the output range"));
            ui.weak("The input values sit on the node.");
        }
        NodeKind::Curve { points, mode } => {
            ui.horizontal(|ui| {
                ui.label("Interpolation");
                for m in graph::CurveMode::ALL {
                    ui.selectable_value(mode, m, m.label());
                }
            });
            let mut remove = None;
            for (i, q) in points.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut q[0]).speed(0.01).prefix("x "));
                    ui.add(egui::DragValue::new(&mut q[1]).speed(0.01).prefix("y "));
                    if ui.small_button("x").on_hover_text("Remove point").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove.filter(|_| points.len() > 1) {
                points.remove(i);
            }
            if points.len() < graph::MAX_CURVE_POINTS && ui.button("Add point").clicked() {
                let x = points.iter().map(|q| q[0]).fold(f32::MIN, f32::max) + 0.25;
                let y = points.last().map_or(0.0, |q| q[1]);
                points.push([x, y]);
            }
            ui.weak("Points are sorted by x when evaluated. Before the first point and after the last, the output holds the end value.");
        }
        NodeKind::Pattern(p) => pattern_props(ui, p),
        NodeKind::Grid { counts, corners } => {
            row(ui, counts, [8.0; 3], |ui, v| {
                for (c, label) in v.iter_mut().zip(["X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).range(1.0..=64.0).speed(0.1).prefix(label));
                }
            });
            row(ui, corners, false, |ui, v| {
                ui.selectable_value(v, false, "Cell centres");
                ui.selectable_value(v, true, "Cell corners");
            });
            ui.weak("Cells per tile on each axis; whole numbers tile. Outputs the distance to the nearest cell centre (or corner): -1 there, +1 as far from it as the grid gets.");
        }
        NodeKind::Normal { flip } => row(ui, flip, false, |ui, f| _ = ui.checkbox(f, "flip")),
        NodeKind::Noise(n) => {
            let d = graph::NoiseParams::default();
            // Changing the type keeps the other settings; it has no reset since it defines the node.
            egui::ComboBox::from_label("type").selected_text(n.kind.name()).show_ui(ui, |ui| {
                for k in graph::NoiseKind::ALL {
                    ui.selectable_value(&mut n.kind, k, k.name());
                }
            });
            if n.kind == graph::NoiseKind::Worley {
                row(ui, &mut n.feature, d.feature, |ui, v| {
                    use graph::WorleyFeature::*;
                    for (f, name) in [(F1, "F1"), (F2, "F2"), (F2MinusF1, "F2 − F1"), (CellValue, "Cell value"), (Edges, "Cell edges")] {
                        ui.selectable_value(v, f, name);
                    }
                });
                row(ui, &mut n.jitter, d.jitter, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("jitter")).on_hover_text("0 puts every point in the middle of its cell (a perfect lattice), 1 anywhere in it"));
                if n.feature != graph::WorleyFeature::Edges {
                    row(ui, &mut n.metric, d.metric, |ui, v| {
                        use graph::WorleyMetric::*;
                        ui.selectable_value(v, Euclidean, "Euclidean");
                        ui.selectable_value(v, Manhattan, "Manhattan").on_hover_text("Diamond-shaped cells");
                        ui.selectable_value(v, Chebyshev, "Chebyshev").on_hover_text("Square cells");
                    });
                }
                row(ui, &mut n.fractal, d.fractal, |ui, v| _ = ui.checkbox(v, "fractal (octaves)"));
            }
            row(ui, &mut n.scale, d.scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=32.0).text("scale (cells/tile)")));
            row(ui, &mut n.normalize, d.normalize, |ui, v| _ = ui.checkbox(v, "normalize 0-1"));
            if n.uses_octaves() {
                row(ui, &mut n.octaves, d.octaves, |ui, v| _ = ui.add(egui::Slider::new(v, 1..=8).text("octaves")));
                row(ui, &mut n.lacunarity, d.lacunarity, |ui, v| _ = ui.add(egui::Slider::new(v, 1.0..=4.0).text("lacunarity")));
                row(ui, &mut n.roughness, d.roughness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("roughness")));
            }
            row(ui, &mut n.distortion, d.distortion, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=2.0).text("distortion")));
            row(ui, &mut n.seed, d.seed, |ui, seed| {
                let mut locked = seed.is_some();
                ui.checkbox(&mut locked, "seed override");
                if locked {
                    ui.add(egui::DragValue::new(seed.get_or_insert(0)));
                } else {
                    *seed = None;
                }
            });
        }
        NodeKind::Position => _ = ui.label("World position. Feeds noise inputs by default."),
        NodeKind::OutDensity | NodeKind::OutNormal => {}
    }
}

const PICKS_KEY: &str = "node_picks";
const SHORTCUTS: [(&str, &str); 10] = [
    ("Ctrl+Z", "Undo"),
    ("Ctrl+Y / Ctrl+Shift+Z", "Redo"),
    ("Ctrl+D", "Duplicate the selected nodes at the cursor (graph editor)"),
    ("Delete / Backspace", "Delete the selected nodes (graph editor)"),
    ("Right-click canvas", "Add node menu; type to search, Enter adds the first match"),
    ("Drag from a pin to empty space", "Add a node already wired up"),
    ("Drag on empty canvas", "Box select (Shift/Ctrl adds)"),
    ("Middle-drag", "Pan the graph editor"),
    ("Mouse wheel", "Zoom the graph editor or the 3D view; moves the slice through the tile"),
    ("Drag in the 3D view", "Orbit (wheel zooms)"),
];
const RECENT_KEY: &str = "recent_files";
const PREFS_KEY: &str = "preferences";

/// Settings that belong to the user, not to a graph; saved with the app state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Prefs {
    /// Scale of the whole interface (egui zoom factor).
    ui_scale: f32,
    /// Rolling crash-recovery backups of the open graph.
    backups: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { ui_scale: 1.0, backups: true }
    }
}

impl Prefs {
    fn sane(mut self) -> Self {
        self.ui_scale = if self.ui_scale.is_finite() { self.ui_scale.clamp(0.75, 2.0) } else { 1.0 };
        self
    }
}

/// Moves `path` to the front of the recent list, without duplicates, at most 10 long.
fn push_recent(list: &mut Vec<std::path::PathBuf>, path: std::path::PathBuf) {
    list.retain(|p| *p != path);
    list.insert(0, path);
    list.truncate(10);
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(s) = serde_json::to_string(&self.editor.picks) {
            storage.set_string(PICKS_KEY, s);
        }
        if let Ok(s) = serde_json::to_string(&self.recent) {
            storage.set_string(RECENT_KEY, s);
        }
        if let Ok(s) = serde_json::to_string(&self.prefs) {
            storage.set_string(PREFS_KEY, s);
        }
    }

    fn on_exit(&mut self) {
        // A clean exit leaves nothing to recover. A pending prompt keeps its crashed sessions for next time.
        if let Some(s) = self.session.take() {
            s.end();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let cpu_start = std::time::Instant::now();
        self.recovery_prompt(&ui.ctx().clone());
        self.poll_mesh_job(&ui.ctx().clone());
        self.mesh_export_window(&ui.ctx().clone());
        self.batch_window(&ui.ctx().clone());
        egui::Window::new("Keyboard shortcuts").open(&mut self.shortcuts_open).collapsible(false).resizable(false).show(ui.ctx(), |ui| {
            egui::Grid::new("shortcuts").striped(true).show(ui, |ui| {
                for (k, what) in SHORTCUTS {
                    ui.monospace(k);
                    ui.label(what);
                    ui.end_row();
                }
            });
        });
        let ctx = ui.ctx().clone();
        egui::Window::new("Preferences").open(&mut self.prefs_open).collapsible(false).resizable(false).show(&ctx, |ui| {
            let r = ui.add(egui::Slider::new(&mut self.prefs.ui_scale, 0.75..=2.0).text("Interface scale"));
            r.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Interface scale"));
            // Applied on release: rescaling under the dragged slider would move it.
            if r.changed() && !r.dragged() || r.drag_stopped() {
                ctx.set_zoom_factor(self.prefs.ui_scale);
            }
            ui.checkbox(&mut self.prefs.backups, "Keep rolling backups for crash recovery");
            if ui.button("Reset to defaults").clicked() {
                self.prefs = Prefs::default();
                ctx.set_zoom_factor(1.0);
            }
        });
        self.export_window(&ui.ctx().clone());
        if let Some(s) = self.session.as_mut().filter(|_| self.graph != self.saved && self.prefs.backups) {
            if s.backup(&self.graph, self.path.as_deref()) {
                ui.ctx().request_repaint_after(backup::INTERVAL); // flush the held-back change
            }
        }
        // Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z), unless a text field has the keyboard.
        if !ui.ctx().egui_wants_keyboard_input() {
            let (z, y, shift) = ui.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::Z), i.modifiers.command && i.key_pressed(egui::Key::Y), i.modifiers.shift));
            if y || (z && shift) {
                self.undo(true);
            } else if z {
                self.undo(false);
            }
        }
        let tiling = self.graph.tiling();

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                        self.undo(false);
                    }
                    if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                        self.undo(true);
                    }
                    ui.separator();
                    if ui.button("Preferences…").clicked() {
                        self.prefs_open = true;
                        ui.close();
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.graph = Graph::default_graph();
                        self.saved = self.graph.clone();
                        self.editor.selected.clear();
                        self.editor.preview = None;
                        self.path = None;
                    }
                    if ui.button("Open…").clicked() {
                        self.open();
                    }
                    ui.add_enabled_ui(!self.recent.is_empty(), |ui| {
                        ui.menu_button("Open recent", |ui| {
                            let mut pick = None;
                            for p in &self.recent {
                                let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                                if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                                    pick = Some(p.clone());
                                    ui.close();
                                }
                            }
                            if let Some(p) = pick {
                                self.open_path(p);
                            }
                        });
                    });
                    if ui.button("Save").clicked() {
                        self.save(false);
                    }
                    if ui.button("Save As…").clicked() {
                        self.save(true);
                    }
                    ui.separator();
                    if ui.button("Import mesh (OBJ)…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("OBJ mesh", &["obj"]).pick_file() {
                            self.start_mesh_load(p);
                        }
                    }
                    if ui.add_enabled(self.mesh.is_some(), egui::Button::new("Export displaced mesh…")).clicked() {
                        self.mesh_export_open = true;
                    }
                    if ui.button("Batch displace meshes…").clicked() {
                        self.batch_open = true;
                    }
                    ui.separator();
                    if ui.button("Export…").clicked() {
                        self.export_open = true;
                    }
                    if ui.add_enabled(self.export_job.is_none(), egui::Button::new("Export Math Function (.noiser)…")).clicked() {
                        self.export_function();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard shortcuts").clicked() {
                        self.shortcuts_open = true;
                        ui.close();
                    }
                });
                ui.weak(&self.status);
            });
        });
        egui::Panel::right("props").default_size(320.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                preview_props(ui, &mut self.pv, self.mesh.is_some());
                self.mesh_props(ui);
                ui.separator();
                let sel: Vec<_> = self.editor.selected.iter().copied().collect();
                match sel.as_slice() {
                    [id] if self.graph.nodes.contains_key(id) => {
                        let node = self.graph.nodes.get_mut(id).unwrap();
                        ui.heading(editor::title(&node.kind));
                        node_props(ui, &mut node.kind);
                        // Below the controls: appearing above them mid-drag shifted the slider out from
                        // under the cursor and ended the drag.
                        if let Some(Tiling::Broken(why)) = tiling.get(id) {
                            ui.colored_label(editor::ORANGE, format!("Does not tile: {why}"));
                        }
                        cache_props(ui, &mut self.graph, *id, &tiling, self.caches.bytes());
                    }
                    [] => _ = ui.weak("Select a node to edit it."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
            });
        });
        self.graph.prune_edges(); // a Math node switched away from smooth min/max loses its third input
        egui::Panel::bottom("graph").resizable(true).default_size(330.0).show(ui, |ui| {
            self.editor.show(ui, &mut self.graph, &tiling);
        });
        // One undo step per finished gesture: wait until the pointer is released.
        let settled = !ui.input(|i| i.pointer.any_down());
        self.history.commit(&self.graph, settled);
        // The views show the Density output, or the previewed node wired into it.
        let shown = match self.editor.preview {
            Some(id) => std::borrow::Cow::Owned(self.graph.previewing(id)),
            None => std::borrow::Cow::Borrowed(&self.graph),
        };
        let range = shown.density_range();
        // Node caches: bake what changed (after this frame's edits), then compile against them.
        let tiling_now = self.graph.tiling();
        self.caches.update(&self.rs.device, &self.rs.queue, &self.graph, &tiling_now, self.pv.seed);
        if self.caches.pending {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
        }
        let compiled = match codegen::compile_with(&shown, &self.caches.slots) {
            Ok(c) if c.params.len() > MAX_PARAMS => Err(format!("graph uses {} parameters (max {MAX_PARAMS})", c.params.len())),
            r => r,
        };
        // Pipelines compile on a worker thread (the driver can take seconds); until a new one is
        // ready the previous pipeline keeps drawing with the parameters that belong to it.
        if let Some((wgsl, rx)) = &self.compiling {
            match rx.try_recv() {
                Ok(gpu) => {
                    self.rs.renderer.write().callback_resources.insert(gpu);
                    self.built = Some((wgsl.clone(), Vec::new()));
                    self.compiling = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ui.ctx().request_repaint_after(std::time::Duration::from_millis(50)),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.compiling = None,
            }
        }
        // The mesh displacement mask is only compiled when it is in use; the pipeline key says so.
        let mask_on = self.pv.prim == meshapp::MESH_PRIM && self.mesh.is_some() && !self.mesh_mask.terms.is_empty();
        if let Ok(c) = &compiled {
            let key = format!("{}{}", if mask_on { "//mask\n" } else { "" }, c.wgsl);
            let built = self.built.as_ref().is_some_and(|b| b.0 == key);
            let queued = self.compiling.as_ref().is_some_and(|p| p.0 == key);
            if !built && !queued {
                let (tx, rx) = std::sync::mpsc::channel();
                let (device, queue, format, wgsl) = (self.rs.device.clone(), self.rs.queue.clone(), self.rs.target_format, c.wgsl.clone());
                std::thread::spawn(move || _ = tx.send(view::build(&device, &queue, format, &wgsl, mask_on)));
                self.compiling = Some((key.clone(), rx)); // a superseded build just finishes and is dropped
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
            }
            if let Some(b) = self.built.as_mut().filter(|b| b.0 == key) {
                b.1 = c.params.clone();
            }
        }
        // (Re)bind cache textures when they changed or a freshly built pipeline doesn't have them yet.
        if let Some(gpu) = self.rs.renderer.write().callback_resources.get_mut::<view::Gpu>() {
            if gpu.cache_gen != self.caches.generation {
                gpu.set_caches(&self.caches.views(), self.caches.generation);
            }
            if gpu.mesh_gen != self.mesh_gen {
                gpu.set_mesh(self.mesh.as_ref().map(|m| &m.view), self.mesh_gen);
            }
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let full = ui.available_rect_before_wrap();
            let (r3d, rslice) = full.split_left_right_at_fraction(0.5);
            let resp = ui.interact(r3d, ui.scope_id().with("view3d"), egui::Sense::drag());
            let d = resp.drag_delta();
            self.yaw -= d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(-1.5, 1.5);
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() {
                self.dist = (self.dist * (1.0 - scroll * 0.002)).clamp(1.2, 15.0);
            }
            // Scrolling over the slice moves it through the tile.
            if ui.rect_contains_pointer(rslice) {
                self.pv.slice_pos = (self.pv.slice_pos + scroll * 0.0005).clamp(0.0, 1.0);
            }
            match compiled {
                Ok(_) => {
                    let uniforms = self.uniforms(r3d, rslice, range);
                    let painter = ui.painter();
                    let px = (r3d.size() * ui.ctx().pixels_per_point()).round();
                    // Draw with whichever pipeline is installed, using its own parameter layout.
                    // Empty params: installed for a graph that has since changed again; wait for its successor.
                    if let Some((_, params)) = self.built.as_ref().filter(|b| !b.1.is_empty()) {
                        let scale = self.pv.render_scale;
                        let cb3d = view::View3d { uniforms, params: params.clone(), size: [(px.x * scale) as u32, (px.y * scale) as u32] };
                        painter.add(egui_wgpu::Callback::new_paint_callback(r3d, cb3d));
                        let cbs = view::SliceView { uniforms, params: params.clone() };
                        painter.add(egui_wgpu::Callback::new_paint_callback(rslice, cbs));
                    }
                    let busy = match (self.compiling.is_some(), self.caches.pending) {
                        (true, _) => Some("Compiling shader…"),
                        (false, true) => Some("Baking cache…"),
                        _ => None,
                    };
                    if let Some(msg) = busy {
                        let at = r3d.center_bottom() - egui::vec2(0.0, 14.0);
                        painter.text(at, egui::Align2::CENTER_BOTTOM, msg, egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 210, 120));
                    }
                    let axis = ["X", "Y", "Z"][self.pv.slice_axis as usize];
                    let mut label = format!("{axis} = {:.3}", self.pv.slice_pos);
                    if let Some(node) = self.editor.preview.and_then(|id| self.graph.nodes.get(&id)) {
                        label = format!("{label}\nPreviewing {}", editor::title(&node.kind));
                    }
                    let font = egui::FontId::monospace(12.0);
                    let grey = egui::Color32::from_gray(200);
                    painter.text(rslice.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, label, font.clone(), grey);
                    // GPU times come back a few frames late; CPU time is this frame's UI + graph work.
                    let gpu_ms = self.rs.renderer.read().callback_resources.get::<view::Gpu>().and_then(|g| g.timer.as_ref().map(|t| t.ms));
                    let timing = match gpu_ms {
                        Some([a, b]) => format!("GPU 3D {a:.2} ms · slice {b:.2} ms · CPU {:.2} ms\n{}×{} px", cpu_start.elapsed().as_secs_f32() * 1e3, px.x, px.y),
                        None => format!("GPU timing unavailable · CPU {:.2} ms", cpu_start.elapsed().as_secs_f32() * 1e3),
                    };
                    painter.text(r3d.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, timing, font, grey);
                }
                Err(e) => {
                    ui.painter().rect_filled(full, 0.0, egui::Color32::from_gray(30));
                    ui.painter().text(full.center(), egui::Align2::CENTER_CENTER, e, egui::FontId::proportional(16.0), egui::Color32::RED);
                }
            }
        });
    }
}


fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: view::request_timestamps(Default::default()),
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]).with_title("Noiser"),
        ..Default::default()
    };
    eframe::run_native("Noiser", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

/// A row of cell counts for the axes a pattern uses, labelled by what they count.
fn pattern_counts(ui: &mut egui::Ui, p: &mut graph::PatternParams, labels: [&str; 3]) {
    let d = graph::PatternParams::for_kind(p.kind);
    let used = p.used_axes();
    let names = ["X", "Y", "Z"];
    row(ui, &mut p.counts, d.counts, |ui, c| {
        for a in used {
            ui.add(egui::DragValue::new(&mut c[a]).range(1.0..=64.0).speed(0.1).prefix(format!("{} ({}) ", labels[a], names[a])));
        }
    });
}

fn pattern_2d(ui: &mut egui::Ui, p: &mut graph::PatternParams) {
    let d = graph::PatternParams::for_kind(p.kind);
    row(ui, &mut p.mode2d, d.mode2d, |ui, v| _ = ui.checkbox(v, "2D (extruded)"));
    if p.mode2d {
        row(ui, &mut p.axis, d.axis, |ui, v| {
            ui.label("extrude along");
            for (a, name) in [(0, "X"), (1, "Y"), (2, "Z")] {
                ui.selectable_value(v, a, name);
            }
        });
    }
}

/// The controls of a Pattern node: the kind, then the settings that kind has.
fn pattern_props(ui: &mut egui::Ui, p: &mut graph::PatternParams) {
    use graph::{PatternKind, PatternOutput::*};
    let before = p.kind;
    egui::ComboBox::from_label("pattern").selected_text(p.kind.name()).show_ui(ui, |ui| {
        for k in PatternKind::ALL {
            ui.selectable_value(&mut p.kind, k, k.name());
        }
    });
    if p.kind != before {
        // A new kind starts from its own defaults, keeping only the seed override.
        let seed = p.seed;
        *p = graph::PatternParams::for_kind(p.kind);
        p.seed = seed;
    }
    let d = graph::PatternParams::for_kind(p.kind);
    match p.kind {
        PatternKind::Bricks => {
            pattern_2d(ui, p);
            let (row_axis, stack_axis, layer_axis) = p.axes();
            let names = ["X", "Y", "Z"];
            row(ui, &mut p.counts, d.counts, |ui, c| {
                let mut count = |ui: &mut egui::Ui, axis: usize, what: &str| {
                    ui.add(egui::DragValue::new(&mut c[axis]).range(1.0..=64.0).speed(0.1).prefix(format!("{what} ({}) ", names[axis])));
                };
                count(ui, row_axis, "per row");
                count(ui, stack_axis, "rows");
                if let Some(l) = layer_axis {
                    count(ui, l, "layers");
                }
            });
            row(ui, &mut p.steps, d.steps, |ui, v| {
                ui.add(egui::Slider::new(v, 1..=4).text("row offset steps"))
                    .on_hover_text("1 stacks the bricks in a grid, 2 is a running bond (each row shifted half a brick), 3 shifts by thirds.\nThe number of rows must be a multiple of this to tile.");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Shape");
                ui.selectable_value(v, CellValue, "Cell value");
            });
            match p.output {
                Distance => _ = ui.weak("Distance to the centre of the brick: -1 at the centre, +1 at the corners."),
                Shape => {
                    row(ui, &mut p.gap, d.gap, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).text("mortar gap")));
                    row(ui, &mut p.bevel, d.bevel, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.5).text("edge softness")));
                    ui.weak("-1 on the brick face, +1 in the mortar.");
                }
                CellValue => {
                    pattern_seed(ui, p, &d);
                    ui.weak("A flat random value per brick.");
                }
            }
        }
        PatternKind::Dimples => {
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Cubic");
                ui.selectable_value(v, 1, "Body-centred");
                ui.selectable_value(v, 2, "Face-centred");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Dimples");
            });
            if p.output == Shape {
                row(ui, &mut p.radius, d.radius, |ui, v| _ = ui.add(egui::Slider::new(v, 0.1..=1.2).text("radius")).on_hover_text("1 makes neighbouring dimples touch"));
                row(ui, &mut p.profile, d.profile, |ui, v| {
                    ui.selectable_value(v, 0, "Sphere cap");
                    ui.selectable_value(v, 1, "Cone");
                    ui.selectable_value(v, 2, "Smooth bowl");
                    ui.selectable_value(v, 3, "Flat disc");
                });
                ui.weak("-1 at the centre of a dimple, +1 on the flat between them. A body-centred or face-centred lattice with a radius near 0.9 is a golf ball.");
            } else {
                ui.weak("Distance to the nearest lattice point: -1 at a point, +1 as far from any as the lattice gets.");
            }
        }
        PatternKind::Checker => {
            pattern_2d(ui, p);
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            ui.weak("+1 and -1 cells. Every used count must be even to tile.");
        }
        PatternKind::Stripes => {
            row(ui, &mut p.counts[0], d.counts[0], |ui, v| _ = ui.add(egui::DragValue::new(v).range(1.0..=64.0).speed(0.1).prefix("stripes per tile ")));
            row(ui, &mut p.dir, d.dir, |ui, v| {
                for (c, name) in v.iter_mut().zip(["X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(c).range(-8.0..=8.0).speed(0.1).prefix(name));
                }
            });
            ui.weak("Direction the stripes run along; whole numbers tile, (1, 1, 0) is a diagonal.");
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Sine");
                ui.selectable_value(v, 1, "Triangle");
                ui.selectable_value(v, 2, "Sawtooth");
                ui.selectable_value(v, 3, "Square");
            });
            if p.variant == 3 {
                row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("width")));
                row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            }
        }
        PatternKind::Fractal => {
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Menger sponge");
                ui.selectable_value(v, 1, "Sierpinski tetrahedron");
            });
            row(ui, &mut p.iterations, d.iterations, |ui, v| _ = ui.add(egui::Slider::new(v, 1..=6).text("iterations")));
            ui.weak("-1 in the solid, +1 in the empty space, from a distance estimate. Changing the iterations does not recompile the shader.");
        }
        PatternKind::Truchet => {
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Arcs");
                ui.selectable_value(v, 1, "Diagonals");
                ui.selectable_value(v, 2, "Pipes (3D)");
            });
            if p.variant < 2 {
                pattern_2d(ui, p);
            }
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("line width")));
            row(ui, &mut p.softness, d.softness, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
            pattern_seed(ui, p, &d);
            ui.weak("-1 on the line, +1 away from it. Each cell is flipped at random; the lines join across cells.");
        }
        PatternKind::Honeycomb => {
            let before = p.variant;
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Hexagons (2D)");
                ui.selectable_value(v, 1, "Body-centred cells");
                ui.selectable_value(v, 2, "Face-centred cells");
                ui.selectable_value(v, 3, "Triangles (2D)");
            });
            if p.variant != before {
                p.mode2d = p.variant == 0 || p.variant == 3; // hexagons and triangles are flat; the lattice cells are 3D
            }
            if p.variant == 0 || p.variant == 3 {
                row(ui, &mut p.axis, d.axis, |ui, v| {
                    ui.label("extrude along");
                    for (a, name) in [(0, "X"), (1, "Y"), (2, "Z")] {
                        ui.selectable_value(v, a, name);
                    }
                });
            }
            pattern_counts(ui, p, ["cells", "cells", "cells"]);
            if p.variant == 0 {
                ui.weak("Hexagons are regular when the cell counts are in the ratio 1 : 0.577: 7 × 4, 12 × 7, 26 × 15.");
            }
            if p.variant == 3 {
                ui.weak("Triangles are equilateral when the second count is 1.155 × the first and even: 6 × 7 is close, 12 × 14, 26 × 30.");
            }
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Distance");
                ui.selectable_value(v, Shape, "Walls");
                ui.selectable_value(v, CellValue, "Cell value");
            });
            match p.output {
                Distance => _ = ui.weak("-1 at a wall, +1 at the centre of a cell."),
                Shape => {
                    row(ui, &mut p.gap, d.gap, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("wall width")));
                    row(ui, &mut p.bevel, d.bevel, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=1.0).text("edge softness")));
                    ui.weak("+1 on the walls, -1 inside the cells.");
                }
                CellValue => {
                    pattern_seed(ui, p, &d);
                    ui.weak("A flat random value per cell.");
                }
            }
        }
        PatternKind::Gyroid => {
            pattern_counts(ui, p, ["periods", "periods", "periods"]);
            row(ui, &mut p.variant, d.variant, |ui, v| {
                ui.selectable_value(v, 0, "Gyroid");
                ui.selectable_value(v, 1, "Schwarz P");
                ui.selectable_value(v, 2, "Diamond");
            });
            row(ui, &mut p.output, d.output, |ui, v| {
                ui.selectable_value(v, Distance, "Field");
                ui.selectable_value(v, Shape, "Shell");
            });
            if p.output == Shape {
                row(ui, &mut p.width, d.width, |ui, v| _ = ui.add(egui::Slider::new(v, 0.0..=0.95).text("thickness")));
                ui.weak("-1 in a thin sheet around the surface, +1 away from it.");
            } else {
                ui.weak("The field itself, scaled to -1..1; its zero level is the surface.");
            }
        }
    }
}

fn pattern_seed(ui: &mut egui::Ui, p: &mut graph::PatternParams, d: &graph::PatternParams) {
    row(ui, &mut p.seed, d.seed, |ui, seed| {
        let mut locked = seed.is_some();
        ui.checkbox(&mut locked, "seed override");
        if locked {
            ui.add(egui::DragValue::new(seed.get_or_insert(0)));
        } else {
            *seed = None;
        }
    });
}

#[cfg(test)]
mod recent_tests {
    use super::push_recent;
    use std::path::PathBuf;

    #[test]
    fn recent_list_is_newest_first_unique_and_capped() {
        let mut l = Vec::new();
        for i in 0..12 {
            push_recent(&mut l, PathBuf::from(format!("{i}.noisegraph")));
        }
        push_recent(&mut l, PathBuf::from("5.noisegraph"));
        assert_eq!(l.len(), 10);
        assert_eq!(l[0], PathBuf::from("5.noisegraph"));
        assert_eq!(l.iter().filter(|p| **p == PathBuf::from("5.noisegraph")).count(), 1);
    }
}

#[cfg(test)]
mod prefs_tests {
    use super::Prefs;

    #[test]
    fn prefs_fill_missing_fields_and_clamp_the_scale() {
        let p: Prefs = serde_json::from_str("{}").unwrap();
        assert_eq!(p, Prefs::default());
        let p: Prefs = serde_json::from_str(r#"{"ui_scale": 9.0}"#).unwrap();
        assert_eq!(p.sane().ui_scale, 2.0);
        assert_eq!(Prefs { ui_scale: f32::NAN, backups: true }.sane().ui_scale, 1.0);
    }
}
