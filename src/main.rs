// No console window behind the release build on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent_cli;
mod backup;
mod bluenoise;
mod cache;
mod codegen;
mod editor;
mod export;
mod graph;
mod graph_params;
mod headless;
mod history;
mod mesh;
mod meshapp;
mod meshmask;
mod meshexport;
mod meshquality;
mod remesh;
mod uvgen;
mod bake;
mod meshgraph;
mod meshgraph_ui;
mod meshworkspace;
mod landscape;
mod landscape_field;
mod landscape_gpu;
mod terrain_maps;
mod scatter;
mod landscape_masks;
mod landscape_erosion;
mod udim;
mod uvtransfer;
mod meshformats;
mod meshsdf;
mod shaderexport;
mod surface;
mod noiser_file;
mod props;
mod composer;
mod library;
mod prefs;
mod icons;
mod composer_gpu;
mod composer_reference;
mod keymap;
use keymap::{Action, pressed};
mod composer_render;
mod sdf_file;
mod sdf_mesh;
mod sdf_node;
mod sdf_model;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod gpu_terrain_tests;
#[cfg(test)]
mod demo_scene;
mod vdb;
mod view;

use eframe::{egui, egui_wgpu};
use graph::{Graph, Tiling};
use prefs::Prefs;
use props::*;
use view::{MAX_PARAMS, Uniforms};

/// Primitive names, in the order `unit_prim` in raymarch.wgsl switches on.
const PRIMS: [&str; 7] = ["Sphere", "Box", "Round box", "Torus", "Cylinder", "Capsule", "Octahedron"];

/// Viewport lighting: a sun, the sky light and the background gradient. Never exported.
#[derive(Clone, Copy, PartialEq)]
pub struct Lighting {
    /// Degrees around the up axis, and above the horizon.
    pub azimuth: f32,
    pub elevation: f32,
    pub sun: f32,
    pub sun_color: [f32; 3],
    pub ambient: f32,
    pub sky_top: [f32; 3],
    pub sky_bottom: [f32; 3],
}

impl Default for Lighting {
    /// The look the viewport had before lighting was adjustable: a sun at (0.6, 0.8, 0.51).
    fn default() -> Self {
        Lighting { azimuth: 49.6, elevation: 45.4, sun: 1.0, sun_color: [1.0; 3], ambient: 1.0, sky_top: [0.18, 0.19, 0.22], sky_bottom: [0.10, 0.10, 0.12] }
    }
}

impl Lighting {
    pub const PRESETS: [(&'static str, fn() -> Lighting); 4] = [
        ("Studio", Lighting::default),
        ("Sunset", || Lighting { azimuth: 75.0, elevation: 10.0, sun: 1.6, sun_color: [1.0, 0.6, 0.35], ambient: 0.7, sky_top: [0.12, 0.14, 0.26], sky_bottom: [0.45, 0.25, 0.18] }),
        ("Overcast", || Lighting { azimuth: 49.6, elevation: 70.0, sun: 0.35, sun_color: [0.9, 0.95, 1.0], ambient: 2.0, sky_top: [0.42, 0.44, 0.48], sky_bottom: [0.30, 0.31, 0.34] }),
        ("Night", || Lighting { azimuth: 200.0, elevation: 35.0, sun: 0.45, sun_color: [0.55, 0.65, 1.0], ambient: 0.5, sky_top: [0.02, 0.03, 0.07], sky_bottom: [0.01, 0.01, 0.03] }),
    ];

    /// The uniform layout: [sun direction, sun strength], [sun colour, ambient], [sky top], [sky bottom].
    pub fn pack(&self) -> [[f32; 4]; 4] {
        let (az, el) = (self.azimuth.to_radians(), self.elevation.to_radians());
        let dir = [az.sin() * el.cos(), el.sin(), az.cos() * el.cos()];
        [
            [dir[0], dir[1], dir[2], self.sun],
            [self.sun_color[0], self.sun_color[1], self.sun_color[2], self.ambient],
            [self.sky_top[0], self.sky_top[1], self.sky_top[2], 0.0],
            [self.sky_bottom[0], self.sky_bottom[1], self.sky_bottom[2], 0.0],
        ]
    }
}

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
    /// Colour the mesh by its displacement mask (viewport only).
    show_mask: bool,
    /// Viewport lighting (never exported).
    lighting: Lighting,
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
            show_mask: false,
            lighting: Lighting::default(),
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
    /// Which graph the window shows, and the mesh graph's state.
    workspace: meshworkspace::Workspace,
    mesh_ws: meshworkspace::MeshWorkspace,
    landscape: landscape::LandscapeWorkspace,
    composer: composer::Composer,
    /// File the graph was last opened from / saved to.
    path: Option<std::path::PathBuf>,
    status: String,
    /// WGSL of the installed preview pipeline, and the graph parameters laid out for it.
    built: Option<(String, Vec<f32>)>,
    /// A pipeline being built on a worker thread, for this WGSL.
    compiling: Option<(String, std::sync::mpsc::Receiver<view::Gpu>)>,
    /// Pipeline key whose build panicked (a shader the driver or wgpu rejected); not retried until it changes.
    failed_build: Option<String>,
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
    /// How the separate parts of an imported mesh are combined (see `meshsdf::Combine`).
    mesh_combine: meshsdf::Combine,
    mesh_export_open: bool,
    mesh_export: meshexport::Settings,
    mesh_export_job: Option<meshapp::MeshExportJob>,
    batch_open: bool,
    batch_files: Vec<std::path::PathBuf>,
    batch_dir: Option<std::path::PathBuf>,
    batch_seeds: meshexport::SeedMode,
    /// Batch: each mesh's displacement amount varies by up to this fraction.
    batch_amount_var: f32,
    /// Mesh: tile size and displacement are in the mesh file's units instead of relative to the mesh (see `App::detail_scale`).
    mesh_object_units: bool,
    batch_job: Option<meshapp::BatchJob>,
    /// How strongly the noise displaces each part of a mesh (see meshmask.rs).
    mesh_mask: meshmask::MaskSettings,
    history: history::History,
    /// Most recently opened or saved graph files, newest first (kept across runs).
    recent: Vec<std::path::PathBuf>,
    shortcuts_open: bool,
    prefs: Prefs,
    /// The preferences the window and the app were last brought in line with.
    applied_prefs: Option<Prefs>,
    prefs_open: bool,
    prefs_tab: prefs::PrefsTab,
    drawer: library::Drawer,
    /// Thumbnails waiting for a screenshot: the file and the view to cut out of it.
    thumbnails: Vec<(std::path::PathBuf, egui::Rect, std::sync::mpsc::Receiver<std::sync::Arc<egui::ColorImage>>)>,
    debug_thumbnail: Option<std::path::PathBuf>,
    /// Which tab of the side panel is showing.
    side_tab: SideTab,
    /// A New / Open that is waiting for the user to decide about unsaved changes.
    pending: Option<Pending>,
    about_open: bool,
    exposed_open: bool,
    /// Last window title sent, so it is only sent when it changes.
    title: String,
    /// The window was asked to close with unsaved changes; waiting for Save / Discard / Cancel.
    close_prompt: bool,
    /// Set once the user chose to discard (or saved), so the close goes through.
    close_ok: bool,
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
        let mut mesh_tab = false;
        let mut landscape_tab = false;
        let mut landscape_map = false;
        let mut landscape_arg: Option<std::path::PathBuf> = None;
        let mut library_root_arg: Option<String> = None;
        let mut show_library = false;
        let mut library_tab: Option<library::Kind> = None;
        let mut thumbnail_arg: Option<std::path::PathBuf> = None;
        let mut show_prefs: Option<prefs::PrefsTab> = None;
        let mut auto_preview = false;
        let mut uv_view = false;
        let mut map_view = false;
        let (mut mesh_graph_arg, mut run_mesh_graph): (Option<std::path::PathBuf>, bool) = (None, false);
        let mut sdf_arg: Option<std::path::PathBuf> = None;
        let (mut batch_dir, mut batch_files): (Option<std::path::PathBuf>, Vec<std::path::PathBuf>) = (None, Vec::new());
        let mut mask_terms = Vec::new();
        let mut show_mask = false;
        let mut lighting_arg: Option<String> = None;
        let mut select_arg: Option<u32> = None;
        let mut reference_arg: Option<std::path::PathBuf> = None;
        let (mut shape_arg, mut cloud_arg, mut amount_arg): (Option<String>, bool, Option<f32>) = (None, false, None);
        let mut template_arg: Option<String> = None;
        while let Some(a) = args.next() {
            if a == "--mesh" {
                mesh_arg = args.next().map(std::path::PathBuf::from);
            } else if a == "--template" {
                template_arg = args.next();
            } else if a == "--reference" {
                reference_arg = args.next().map(std::path::PathBuf::from);
            } else if a == "--select" {
                select_arg = args.next().and_then(|n| n.parse::<u32>().ok());
            } else if a == "--shape" {
                shape_arg = args.next();
            } else if a == "--cloud" {
                cloud_arg = true;
            } else if a == "--amount" {
                amount_arg = args.next().and_then(|v| v.parse().ok());
            } else if a == "--lighting" {
                lighting_arg = args.next();
            } else if a == "--show-mask" {
                show_mask = true;
            } else if a == "--open-mesh-export" {
                open_mesh_export = true;
            } else if a == "--library-root" {
                library_root_arg = args.next();
            } else if a == "--thumbnail-of" {
                // For checking the picture a save into the library would write: asks for it once the view is up.
                thumbnail_arg = args.next().map(std::path::PathBuf::from);
            } else if a == "--library-tab" {
                library_tab = match args.next().unwrap_or_default().as_str() {
                    "noises" => Some(library::Kind::Noise),
                    "meshgraphs" => Some(library::Kind::MeshGraph),
                    _ => None,
                };
                show_library = true;
            } else if a == "--library" {
                show_library = true;
            } else if a == "--prefs" {
                show_prefs = Some(match args.next().unwrap_or_default().as_str() {
                    "keybinds" => prefs::PrefsTab::Keybinds,
                    "performance" => prefs::PrefsTab::Performance,
                    "paths" => prefs::PrefsTab::FilePaths,
                    "style" => prefs::PrefsTab::Style,
                    _ => prefs::PrefsTab::General,
                });
            } else if a == "--mesh-tab" {
                mesh_tab = true;
            } else if a == "--landscape-tab" {
                landscape_tab = true;
            } else if a == "--landscape-map" {
                (landscape_tab, landscape_map) = (true, true);
            } else if a == "--window" {
                args.next(); // read below
            } else if a == "--mesh-graph" {
                mesh_graph_arg = args.next().map(std::path::PathBuf::from);
            } else if a == "--run-mesh-graph" {
                run_mesh_graph = true;
            } else if a == "--auto-preview" {
                auto_preview = true;
            } else if a == "--uv-view" {
                uv_view = true;
            } else if a == "--map-view" {
                map_view = true;
            } else if a == "--mask" {
                // A displacement-mask term of the named kind with its default settings.
                let name = args.next().unwrap_or_default().to_lowercase();
                mask_terms.extend(meshmask::MaskKind::ALL.iter().find(|k| k.name().to_lowercase().starts_with(&name)).map(|k| meshmask::MaskTerm::new(*k)));
            } else if a == "--batch-out" {
                batch_dir = args.next().map(std::path::PathBuf::from);
            } else if a == "--batch-file" {
                batch_files.extend(args.next().map(std::path::PathBuf::from));
            } else if a.to_lowercase().ends_with(".sdf") {
                // An SDF Modeler file given as a plain file name opens in the SDF Composer tab.
                sdf_arg = Some(std::path::PathBuf::from(a));
            } else if a.to_lowercase().ends_with(".landscape") {
                landscape_arg = Some(std::path::PathBuf::from(a));
            } else if a.to_lowercase().ends_with(".meshgraph") {
                // A mesh graph given as a plain file name ("open with") opens in the Mesh graph tab.
                mesh_graph_arg = Some(std::path::PathBuf::from(a));
            } else if graph_arg.is_none() {
                graph_arg = Some(std::path::PathBuf::from(a));
            }
        }
        let template = template_arg.as_deref().and_then(|n| Graph::TEMPLATES.iter().find(|t| t.eq_ignore_ascii_case(n)));
        let (graph, path, status) = match graph_arg {
            Some(p) => match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
                Ok(g) => (g, Some(p.clone()), format!("Opened {}", p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()))),
                Err(e) => (Graph::default_graph(), None, format!("Could not open {}: {e}", p.display())),
            },
            None => (template.map_or_else(Graph::default_graph, |t| Graph::template(t)), None, String::new()),
        };
        let prefs: Prefs = cc.storage.and_then(|s| s.get_string(PREFS_KEY)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let mut prefs = prefs.sane();
        // Started with a file, mesh or template: the getting-started window is not needed.
        if std::env::args().nth(1).is_some() {
            prefs.seen_about = true;
        }
        if let Some(root) = library_root_arg {
            prefs.library_root = root;
        }
        cc.egui_ctx.set_zoom_factor(prefs.ui_scale);
        let mut editor = editor::Editor::default();
        editor.frame_next = true;
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
            workspace: Default::default(),
            mesh_ws: Default::default(),
            landscape: Default::default(),
            composer: Default::default(),
            path,
            status,
            built: None,
            compiling: None,
            failed_build: None,
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
            mesh_combine: meshsdf::Combine::default(),
            mesh_export_open: false,
            mesh_export: meshexport::Settings::default(),
            mesh_export_job: None,
            batch_open: false,
            batch_files: Vec::new(),
            batch_dir: None,
            batch_seeds: meshexport::SeedMode::Same,
            batch_amount_var: 0.0,
            mesh_object_units: false,
            batch_job: None,
            mesh_mask: meshmask::MaskSettings::default(),
            history: history::History::new(&saved_for_history),
            shortcuts_open: false,
            prefs: prefs.clone(),
            applied_prefs: None,
            prefs_open: false,
            prefs_tab: Default::default(),
            drawer: Default::default(),
            thumbnails: Vec::new(),
            debug_thumbnail: None,
            side_tab: SideTab::default(),
            pending: None,
            about_open: false,
            exposed_open: false,
            title: String::new(),
            close_prompt: false,
            close_ok: false,
            recent: cc.storage.and_then(|s| s.get_string(RECENT_KEY)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default(),
        };
        if let Some(p) = sdf_arg {
            app.composer.open_path(p);
            app.workspace = meshworkspace::Workspace::Composer;
        } else if let Some(json) = cc.storage.and_then(|s| s.get_string(COMPOSER_KEY)).filter(|j| !j.is_empty()) {
            // The composition as it was left, an unsaved one included.
            let path = cc.storage.and_then(|s| s.get_string(COMPOSER_PATH_KEY)).filter(|p| !p.is_empty()).map(std::path::PathBuf::from);
            let modified = cc.storage.and_then(|s| s.get_string(COMPOSER_MODIFIED_KEY)).as_deref() == Some("1");
            app.composer.restore(&json, path, modified);
        }
        if let Some(json) = cc.storage.and_then(|s| s.get_string(COMPOSER_REFERENCE_KEY)).filter(|j| !j.is_empty()) {
            app.composer.restore_reference(&json);
        }
        let graph_from_arg = mesh_graph_arg.is_some();
        app.drawer.open = show_library;
        if let Some(kind) = library_tab {
            app.drawer.set_tab(kind);
        }
        app.debug_thumbnail = thumbnail_arg;
        if let Some(tab) = show_prefs {
            app.prefs_open = true;
            app.prefs_tab = tab;
        }
        if mesh_tab || mesh_graph_arg.is_some() {
            app.workspace = meshworkspace::Workspace::Mesh;
        }
        let landscape_from_arg = landscape_arg.is_some();
        if let Some(p) = landscape_arg {
            app.landscape.open_path(p);
            landscape_tab = true;
        }
        if landscape_tab {
            app.workspace = meshworkspace::Workspace::Landscape;
            app.landscape.show_map(landscape_map);
        }
        if let Some(p) = mesh_graph_arg {
            match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
                Ok(g) => app.mesh_ws.open_graph(g, p),
                Err(e) => app.status = format!("Could not open {}: {e}", p.display()),
            }
        }
        app.mesh_ws.autorun = run_mesh_graph;
        app.mesh_ws.auto_preview = auto_preview;
        app.mesh_ws.show_uv = uv_view;
        app.mesh_ws.show_map = map_view;
        if let Some(p) = mesh_arg {
            app.start_mesh_load(p);
        }
        // The mesh graph as it was left (an unsaved one included), with the file it belongs to.
        // (a graph named on the command line wins)
        if let Some(g) = cc.storage.and_then(|s| s.get_string(MESH_GRAPH_KEY)).and_then(|s| Graph::from_json(&s).ok()).filter(|_| !graph_from_arg) {
            let path = cc.storage.and_then(|s| s.get_string(MESH_GRAPH_PATH_KEY)).filter(|p| !p.is_empty()).map(std::path::PathBuf::from);
            app.mesh_ws.restore(g, path);
        }
        // The landscape as it was left (a project named on the command line wins).
        if landscape_from_arg {
        } else if let Some(project) = cc.storage.and_then(|s| s.get_string(LANDSCAPE_PROJECT_KEY)) {
            let _ = app.landscape.from_project(&project);
        } else if let Some(g) = cc.storage.and_then(|s| s.get_string(LANDSCAPE_GRAPH_KEY)).and_then(|s| Graph::from_json(&s).ok()) {
            let settings = cc.storage.and_then(|s| s.get_string(LANDSCAPE_SETTINGS_KEY)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            app.landscape.restore(g, settings, None);
        }
        app.mesh_mask.terms = mask_terms;
        app.pv.show_mask = show_mask;
        if let Some(i) = shape_arg.and_then(|n| PRIMS.iter().position(|p| p.eq_ignore_ascii_case(&n))) {
            app.pv.prim = i as u32;
        }
        app.pv.cloud = cloud_arg;
        if let Some(a) = amount_arg {
            app.pv.amount = a;
        }
        if let Some(path) = reference_arg {
            app.composer.load_reference(path);
            app.workspace = meshworkspace::Workspace::Composer;
        }
        if let Some(n) = select_arg.filter(|_| app.workspace == meshworkspace::Workspace::Composer) {
            // `--select N` selects the Nth top-level shape of the composition.
            app.composer.select(vec![n as usize]);
        }
        if let Some(n) = select_arg.filter(|n| app.workspace == meshworkspace::Workspace::Mesh && app.mesh_ws.graph.nodes.contains_key(n)) {
            app.mesh_ws.editor.selected.insert(n);
        }
        if let Some(n) = select_arg.filter(|n| app.graph.nodes.contains_key(n)) {
            app.editor.selected.insert(n);
        }
        if let Some(p) = lighting_arg.and_then(|n| Lighting::PRESETS.iter().find(|(l, _)| l.eq_ignore_ascii_case(&n))) {
            app.pv.lighting = (p.1)();
        }
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
                ui.selectable_value(v, Kind::Raw3d, "3D raw floats");
            });
            row(ui, &mut s.output, d.output, |ui, v| {
                ui.selectable_value(v, Output::Density, "Density");
                ui.selectable_value(v, Output::Normal, "Normal");
            });
            let res_default = if s.kind == Kind::Image2d { 1024 } else { 128 };
            row(ui, &mut s.res, res_default, |ui, v| _ = ui.add(egui::DragValue::new(v).range(1..=16384).suffix(" px").speed(4.0)).on_hover_text("Texels per tile side"));
            if !s.is_volume() {
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
                Kind::Raw3d => _ = ui.weak("Raw 32-bit floats, x fastest, then y, then z (a .raw.txt note is written next to it)."),
                Kind::SliceGrid => {}
            }
            if s.format == Format::Exr32 && !s.is_volume() {
                ui.weak("EXR keeps raw signed values.");
            }
            let (file, _) = s.estimate();
            let dims = s.image_size().map_or(format!("{0}×{0}×{0} voxels", s.res), |(w, h)| format!("{w}×{h} px"));
            ui.label(format!("{dims}, about {:.1} MB", file as f64 / (1 << 20) as f64));
            for w in s.warnings() {
                ui.colored_label(editor::ORANGE, format!("Warning: {w}"));
            }
            if let Some(why) = self.graph.tiling_problem_upstream() {
                ui.colored_label(editor::ORANGE, format!("The result does not tile: {why}"));
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
        let Some(path) = library::dialog(library::Kind::Bake, rfd::FileDialog::new().add_filter("Export", &[s.extension()]).set_file_name(format!("{name}.{}", s.extension()))).save_file() else { return };
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
                self.editor.frame_next = true;
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
        let k = if mesh_on { self.detail_scale() } else { 1.0 };
        let push = view::push_range(range, pv.amount * k, mask.range);
        let (mesh_origin, mesh_voxel) = self.mesh.as_ref().map_or(([0.0; 3], 1.0), |m| (m.sdf.origin, m.sdf.voxel));
        let (mesh_lo, mesh_hi) = self.mesh.as_ref().map_or(([0.0; 3], [0.0; 3]), |m| m.bounds);
        Uniforms {
            cam_pos: pos,
            aspect: aspect(view3d),
            cam_fwd: fwd,
            amount: pv.amount * k,
            cam_right: right,
            step_scale: pv.step_scale,
            cam_up: up,
            seed: pv.seed,
            tile_size: pv.tile_size * k,
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
            light: pv.lighting.pack(),
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
        library::dialog(library::Kind::Noise, rfd::FileDialog::new().add_filter("Noiser graph", &["noisegraph"]))
    }

    fn open(&mut self) {
        self.request(Pending::OpenDialog);
    }

    /// Does `action` now, or first asks what to do with unsaved changes.
    fn request(&mut self, action: Pending) {
        if self.graph != self.saved {
            self.pending = Some(action);
        } else {
            self.run_pending(action);
        }
    }

    fn run_pending(&mut self, action: Pending) {
        match action {
            Pending::New => self.start_fresh(Graph::default_graph()),
            Pending::Template(name) => self.start_fresh(Graph::template(name)),
            Pending::OpenDialog => {
                if let Some(path) = Self::file_dialog().pick_file() {
                    self.open_path(path);
                }
            }
            Pending::Path(path) => self.open_path(path),
        }
    }

    /// Replaces the graph with a new, unsaved one.
    fn start_fresh(&mut self, graph: Graph) {
        self.graph = graph;
        self.saved = self.graph.clone();
        self.editor.selected.clear();
        self.editor.frame_next = true;
        self.editor.preview = None;
        self.path = None;
    }

    fn open_path(&mut self, path: std::path::PathBuf) {
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
            Ok(g) => {
                self.saved = g.clone();
                self.graph = g;
                self.editor.selected.clear();
                self.editor.frame_next = true;
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
        let text = self.graph.to_json();
        keep_previous_version(&path, &text);
        match std::fs::write(&path, text) {
            Ok(()) => {
                self.saved = self.graph.clone();
                self.status = format!("Saved {}", path.display());
                library::request_thumbnail(&path);
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

/// Before a graph file is overwritten with different content, its old content goes to `<name>.history/` with a
/// timestamp, so a noise graph a mesh graph uses can always be taken back (graphs are small; the newest 100 are kept).
fn keep_previous_version(path: &std::path::Path, new_text: &str) {
    let Ok(old) = std::fs::read_to_string(path) else { return };
    if old == new_text {
        return;
    }
    let (Some(name), Some(dir)) = (path.file_name(), path.parent()) else { return };
    let history = dir.join(format!("{}.history", name.to_string_lossy()));
    if std::fs::create_dir_all(&history).is_err() {
        return;
    }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let ext = path.extension().map_or("noisegraph".to_string(), |e| e.to_string_lossy().into_owned());
    let _ = std::fs::write(history.join(format!("{stamp}.{ext}")), old);
    // Old versions beyond the newest 100 go.
    if let Ok(entries) = std::fs::read_dir(&history) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        for old in files.iter().take(files.len().saturating_sub(100)) {
            let _ = std::fs::remove_file(old);
        }
    }
}

const PICKS_KEY: &str = "node_picks";
const MESH_GRAPH_KEY: &str = "mesh_graph";
const COMPOSER_KEY: &str = "composer_document";
const COMPOSER_PATH_KEY: &str = "composer_path";
const COMPOSER_MODIFIED_KEY: &str = "composer_modified";
const COMPOSER_REFERENCE_KEY: &str = "composer_reference";
const MESH_GRAPH_PATH_KEY: &str = "mesh_graph_path";
const LANDSCAPE_GRAPH_KEY: &str = "landscape_graph";
const LANDSCAPE_SETTINGS_KEY: &str = "landscape_settings";
const LANDSCAPE_PROJECT_KEY: &str = "landscape_project";
/// The shortcuts that cannot be remapped; the rest come from the keymap.
const SHORTCUTS: [(&str, &str); 10] = [
    ("Drop a file on the window", "A .noisegraph opens as the graph, an .obj loads as the mesh"),
    ("Drop a lone node on a wire", "Splices the node into the wire (graph editor)"),
    ("Ctrl + right-drag", "Cut the wires the line crosses (graph editor)"),
    ("Ctrl+C / Ctrl+V", "Copy the selection, paste at the cursor; works between two Noiser windows"),
    ("Right-click canvas", "Add node menu; type to search, Enter adds the first match"),
    ("Drag from a pin to empty space", "Add a node already wired up"),
    ("Drag on empty canvas", "Box select (Shift/Ctrl adds)"),
    ("Middle-drag", "Pan the graph editor"),
    ("Mouse wheel", "Zoom the graph editor or the 3D view; moves the slice through the tile"),
    ("Drag in the 3D view", "Orbit (wheel zooms, double-click resets the camera)"),
];
const RECENT_KEY: &str = "recent_files";
const PREFS_KEY: &str = "preferences";

/// What to do once unsaved changes have been dealt with.
#[derive(Clone, Debug)]
enum Pending {
    New,
    Template(&'static str),
    OpenDialog,
    Path(std::path::PathBuf),
}

/// Moves `path` to the front of the recent list, without duplicates, at most 10 long.
fn push_recent(list: &mut Vec<std::path::PathBuf>, path: std::path::PathBuf) {
    list.retain(|p| *p != path);
    list.insert(0, path);
    list.truncate(10);
}

impl App {
    /// Brings the window and the global settings in line with the preferences, once after each change.
    fn apply_prefs(&mut self, ctx: &egui::Context) {
        if self.applied_prefs.as_ref() == Some(&self.prefs) {
            return;
        }
        let before = self.applied_prefs.replace(self.prefs.clone());
        let changed = |f: &dyn Fn(&Prefs) -> bool| before.as_ref().is_none_or(f);
        library::set_root(self.prefs.library_path());
        library::set_thumbnails(self.prefs.thumbnails);
        sdf_mesh::set_resolution(self.prefs.sdf_mesh_resolution as usize);
        if changed(&|b| b.theme != self.prefs.theme || b.accent != self.prefs.accent) {
            ctx.set_theme(self.prefs.theme.preference());
            ctx.set_visuals_of(egui::Theme::Dark, prefs::visuals(true, self.prefs.accent));
            ctx.set_visuals_of(egui::Theme::Light, prefs::visuals(false, self.prefs.accent));
        }
        if changed(&|b| b.render_scale != self.prefs.render_scale) {
            self.pv.render_scale = self.prefs.render_scale;
        }
        if changed(&|b| b.max_steps != self.prefs.max_steps) {
            self.pv.max_steps = self.prefs.max_steps;
        }
        if changed(&|b| b.composer_render_scale != self.prefs.composer_render_scale || b.composer_max_steps != self.prefs.composer_max_steps) {
            self.composer.set_render_quality(self.prefs.composer_render_scale, self.prefs.composer_max_steps);
        }
        if changed(&|b| b.export_samples != self.prefs.export_samples) {
            self.composer.set_export_samples(self.prefs.export_samples as usize);
        }
    }

    /// Writes the pictures asked for by files saved into the library, from screenshots of the window.
    fn collect_thumbnails(&mut self, ctx: &egui::Context) {
        if self.debug_thumbnail.is_some() {
            let ready = self.workspace != meshworkspace::Workspace::Mesh || self.mesh_ws.has_preview();
            if ready && library::view_rect().is_some() && ctx.input(|i| i.time) > 4.0 {
                if let Some(file) = self.debug_thumbnail.take() {
                    library::request_thumbnail(&file);
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(500));
            }
        }
        for file in library::take_requests() {
            if let Some(rect) = library::view_rect() {
                let (tx, rx) = std::sync::mpsc::channel();
                self.thumbnails.push((file, rect.shrink(2.5), rx));
                library::set_capturing(true);
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |image| {
                    let _ = tx.send(image);
                })));
                ctx.request_repaint();
            }
        }
        let mut waiting = Vec::new();
        for (file, rect, rx) in std::mem::take(&mut self.thumbnails) {
            match rx.try_recv() {
                Ok(shot) => {
                    if let Err(e) = library::save_thumbnail(&shot, rect, ctx.pixels_per_point(), &library::thumbnail_path(&file)) {
                        self.status = format!("No picture saved for {}: {e}", file.display());
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => waiting.push((file, rect, rx)),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
        if waiting.is_empty() {
            library::set_capturing(false);
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        self.thumbnails = waiting;
    }

    /// The library drawer (a key, `Tab` by default), and what dropping one of its cards does in the open tab.
    fn library_drawer(&mut self, ui: &mut egui::Ui) {
        use meshworkspace::Workspace;
        let ctx = ui.ctx().clone();
        if !ctx.egui_wants_keyboard_input() && ctx.input_mut(|i| keymap::consume(i, keymap::Action::Library)) {
            self.drawer.open = !self.drawer.open;
        }
        if !self.drawer.open {
            return;
        }
        let hint = match (self.workspace, self.drawer.tab()) {
            (Workspace::Composer, library::Kind::Sdf) => "Drag an SDF into the view to add it to the composition.",
            (Workspace::Noise, library::Kind::Sdf) => "Drag an SDF onto the view to use it as the mesh.",
            (Workspace::Mesh, library::Kind::MeshGraph) => "Double-click to open a mesh graph.",
            (Workspace::Mesh, _) => "Drag a file into the graph to make a node from it.",
            _ => "Double-click to open it.",
        };
        let outer = ui.max_rect();
        let action = self.drawer.show(ui, hint);
        let keep_out = egui::Rect::from_min_max(egui::pos2(outer.left(), ui.available_rect_before_wrap().bottom()), outer.right_bottom());
        match action {
            Some(library::Action::ChooseFolder) => {
                self.prefs_open = true;
                self.prefs_tab = prefs::PrefsTab::FilePaths;
            }
            Some(library::Action::Open(path, kind)) => self.open_library_item(path, kind),
            None => {}
        }
        if let Some((item, at)) = library::Drawer::dropped(&ctx, keep_out) {
            self.drop_library_item(item, at);
        }
    }

    fn open_library_item(&mut self, path: std::path::PathBuf, kind: library::Kind) {
        use meshworkspace::Workspace;
        match kind {
            library::Kind::Sdf => {
                self.workspace = Workspace::Composer;
                self.composer.open_request(path);
            }
            library::Kind::Noise => {
                self.workspace = Workspace::Noise;
                self.request(Pending::Path(path));
            }
            library::Kind::MeshGraph => {
                self.workspace = Workspace::Mesh;
                self.mesh_ws.drop_file(path);
            }
            library::Kind::Erosion => {
                self.workspace = Workspace::Landscape;
                self.landscape.load_erosion_preset(&path);
            }
            library::Kind::Bake | library::Kind::Udim => {}
        }
    }

    fn drop_library_item(&mut self, item: library::LibraryItem, at: egui::Pos2) {
        use meshworkspace::Workspace;
        match (self.workspace, item.kind) {
            (Workspace::Composer, library::Kind::Sdf) => {
                if let Err(e) = self.composer.insert_file(&item.path) {
                    self.status = e;
                }
            }
            (Workspace::Noise, library::Kind::Sdf) => self.start_mesh_load(item.path),
            (Workspace::Noise, library::Kind::Noise) => self.request(Pending::Path(item.path)),
            (Workspace::Mesh, _) => self.mesh_ws.drop_file_at(item.path, Some(at)),
            _ => {}
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(s) = serde_json::to_string(&self.editor.picks) {
            storage.set_string(PICKS_KEY, s);
        }
        match self.composer.storage() {
            Some((json, path, modified)) => {
                storage.set_string(COMPOSER_KEY, json);
                storage.set_string(COMPOSER_PATH_KEY, path);
                storage.set_string(COMPOSER_MODIFIED_KEY, if modified { "1" } else { "0" }.into());
            }
            None => storage.set_string(COMPOSER_KEY, String::new()),
        }
        storage.set_string(COMPOSER_REFERENCE_KEY, self.composer.reference_storage());
        storage.set_string(MESH_GRAPH_KEY, self.mesh_ws.graph.to_json());
        storage.set_string(LANDSCAPE_PROJECT_KEY, self.landscape.to_project());
        storage.set_string(MESH_GRAPH_PATH_KEY, self.mesh_ws.path.as_ref().map_or(String::new(), |p| p.to_string_lossy().into_owned()));
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
        keymap::activate(&self.prefs.keymap);
        if !self.prefs_open {
            keymap::set_capturing(false);
        }
        self.collect_thumbnails(&ui.ctx().clone());
        self.recovery_prompt(&ui.ctx().clone());
        self.poll_mesh_job(&ui.ctx().clone());
        self.mesh_export_window(&ui.ctx().clone());
        self.batch_window(&ui.ctx().clone());
        egui::Window::new("Keyboard shortcuts").open(&mut self.shortcuts_open).collapsible(false).resizable(false).show(ui.ctx(), |ui| {
            egui::Grid::new("shortcuts").striped(true).show(ui, |ui| {
                for action in Action::ALL {
                    ui.monospace(self.prefs.keymap.text(action));
                    ui.label(action.label());
                    ui.end_row();
                }
                for (k, what) in SHORTCUTS {
                    ui.monospace(k);
                    ui.label(what);
                    ui.end_row();
                }
            });
        });
        let ctx = ui.ctx().clone();
        // Files dropped on the window: .obj becomes the mesh, anything else is opened as a graph.
        for f in ctx.input(|i| i.raw.dropped_files.clone()) {
            { let p = f.path().to_path_buf();
                if self.workspace == meshworkspace::Workspace::Composer {
                    // The composer takes .sdf files from the window itself.
                } else if self.workspace == meshworkspace::Workspace::Mesh {
                    self.mesh_ws.drop_file(p);
                } else if meshformats::is_mesh_file(&p) {
                    self.start_mesh_load(p);
                } else {
                    self.request(Pending::Path(p));
                }
            }
        }
        self.editor.snap = self.prefs.snap_nodes;
        sdf_node::set_base(sdf_node::NOISE_GRAPH_BASE, self.path.as_deref().and_then(|p| p.parent()));
        sdf_node::set_base(sdf_node::MESH_GRAPH_BASE, self.mesh_ws.path.as_deref().and_then(|p| p.parent()));
        // The first time Noiser runs, the getting-started window opens by itself.
        if !self.prefs.seen_about {
            self.prefs.seen_about = true;
            self.about_open = true;
        }
        // Window title: the file and an asterisk for unsaved changes.
        let dirty = self.graph != self.saved;
        // The tab shown decides whose file it names.
        let named = |path: &Option<std::path::PathBuf>, d: bool| match (path, d) {
            (None, false) => "Noiser".to_string(),
            (p, d) => format!("Noiser - {}{}", p.as_ref().and_then(|p| p.file_name()).map_or("untitled".into(), |n| n.to_string_lossy()), if d { " *" } else { "" }),
        };
        let title = match self.workspace {
            meshworkspace::Workspace::Composer => format!("Noiser - {}", self.composer.title()),
            meshworkspace::Workspace::Mesh => named(&self.mesh_ws.path, self.mesh_ws.dirty()),
            meshworkspace::Workspace::Landscape => named(&self.landscape.path, false),
            meshworkspace::Workspace::Noise => named(&self.path, dirty),
        };
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
        // Closing with unsaved changes asks first.
        let any_dirty = dirty || self.composer.is_modified() || self.mesh_ws.dirty();
        if ctx.input(|i| i.viewport().close_requested()) && any_dirty && !self.close_ok {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_prompt = true;
        }
        if let Some(action) = self.pending.clone() {
            egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.label("The graph has changes that are not saved.");
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.save(false);
                        if self.graph == self.saved {
                            self.pending = None;
                            self.run_pending(action.clone());
                        }
                    }
                    if ui.button("Discard").clicked() {
                        self.pending = None;
                        self.run_pending(action.clone());
                    }
                    if ui.button("Cancel").clicked() {
                        self.pending = None;
                    }
                });
            });
        }
        if self.close_prompt {
            let mut quit = false;
            egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.label("These tabs have changes that are not saved:");
                if self.graph != self.saved {
                    ui.horizontal(|ui| {
                        ui.label("Noise graph");
                        if icons::button(ui, "save", "Save the noise graph", None, true).on_hover_text("Save the noise graph").clicked() {
                            self.save(false);
                        }
                    });
                }
                if self.composer.is_modified() {
                    ui.horizontal(|ui| {
                        ui.label("SDF Composer");
                        if icons::button(ui, "save", "Save the composition", None, true).on_hover_text("Save the composition").clicked() {
                            self.composer.save_current();
                        }
                    });
                }
                if self.mesh_ws.dirty() {
                    ui.horizontal(|ui| {
                        ui.label("Mesh graph");
                        if icons::button(ui, "save", "Save the mesh graph", None, true).on_hover_text("Save the mesh graph").clicked() {
                            self.save_mesh_graph(false);
                        }
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Quit without saving").clicked() {
                        quit = true;
                    }
                    if ui.button("Cancel").clicked() {
                        self.close_prompt = false;
                    }
                });
            });
            if !(self.graph != self.saved || self.composer.is_modified() || self.mesh_ws.dirty()) {
                quit = true;
            }
            if quit {
                self.close_ok = true;
                self.close_prompt = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        egui::Window::new("Exposed parameters").open(&mut self.exposed_open).collapsible(false).show(&ctx, |ui| {
            ui.weak("Name a node setting so a script can set it: noiser-batch.cmd graph.noisegraph --param name=value. The field is the setting's name in the .noisegraph file (scale, octaves, jitter, weight, value for a Value node...).");
            let mut remove = None;
            let ids: Vec<graph::NodeId> = self.graph.nodes.keys().copied().collect();
            for i in 0..self.graph.exposed.len() {
                let shown = self.graph.exposed_value(&self.graph.exposed[i]);
                let e = &mut self.graph.exposed[i];
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut e.name).hint_text("name").desired_width(90.0));
                    egui::ComboBox::from_id_salt(("exposed node", i)).selected_text(format!("node {}", e.node)).show_ui(ui, |ui| {
                        for id in &ids {
                            ui.selectable_value(&mut e.node, *id, format!("node {id}"));
                        }
                    });
                    ui.add(egui::TextEdit::singleline(&mut e.field).hint_text("field").desired_width(90.0));
                    match &shown {
                        Ok(v) => _ = ui.weak(v.to_string()),
                        Err(why) => _ = ui.colored_label(editor::ORANGE, why),
                    }
                    if ui.small_button("×").on_hover_text("Remove").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                self.graph.exposed.remove(i);
            }
            if ui.button("Add parameter").clicked() {
                let node = self.editor.selected.iter().next().copied().unwrap_or(0);
                self.graph.exposed.push(graph::Exposed { name: format!("param{}", self.graph.exposed.len() + 1), node, field: "scale".into() });
            }
        });
        egui::Window::new("About Noiser").open(&mut self.about_open).collapsible(false).resizable(false).show(&ctx, |ui| {
            ui.heading(format!("Noiser {}", env!("CARGO_PKG_VERSION")));
            ui.label("A node editor for tiling 3D noise: build a graph of noises, patterns and math, watch it in 3D and as a slice, export it as textures, a volume, a math function, or use it to displace meshes.");
            ui.separator();
            ui.label("Getting started");
            ui.label("• Right-click the graph to add nodes (type to search); drag from a pin to connect, or drop a wire on empty space.");
            ui.label("• The eye on a node previews it; the cache button bakes everything upstream of it for speed.");
            ui.label("• Orange outline: the node breaks tiling (the details panel says why).");
            ui.label("• File > Import mesh shapes a low-poly OBJ, PLY or STL with the graph; Export displaced mesh writes the high-resolution result.");
            ui.label("• Help > Keyboard shortcuts lists the keys. Source and releases: github.com/simenstroek/Noiser");
        });
        egui::Window::new("Preferences").open(&mut self.prefs_open).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
            let effects = prefs::show(&mut self.prefs, &mut self.prefs_tab, ui, &SHORTCUTS);
            if effects.scale_committed {
                ctx.set_zoom_factor(self.prefs.ui_scale);
            }
        });
        self.apply_prefs(&ctx);
        self.export_window(&ui.ctx().clone());
        if let Some(s) = self.session.as_mut().filter(|_| self.graph != self.saved && self.prefs.backups)
            && s.backup(&self.graph, self.path.as_deref()) {
                ui.ctx().request_repaint_after(backup::INTERVAL); // flush the held-back change
            }
        // File shortcuts work even with a text field focused (Ctrl+S would otherwise do nothing there).
        let (save_as, save, open, new) = ui.input(|i| (pressed(i, Action::SaveAs), pressed(i, Action::Save), pressed(i, Action::Open), pressed(i, Action::New)));
        if self.workspace != meshworkspace::Workspace::Noise {
            // The mesh graph and composer tabs handle their own keys.
        } else if save_as {
            self.save(true);
        } else if save {
            self.save(false);
        } else if open {
            self.open();
        } else if new {
            self.request(Pending::New);
        }
        // Undo and redo, unless a text field has the keyboard.
        if !ui.ctx().egui_wants_keyboard_input() && self.workspace == meshworkspace::Workspace::Noise {
            let (undo, redo) = ui.input(|i| (pressed(i, Action::Undo), pressed(i, Action::Redo)));
            if redo {
                self.undo(true);
            } else if undo {
                self.undo(false);
            }
        }
        let tiling = self.graph.tiling();

        // The workspace tabs first, then the menu bar under them: its File and Edit menus belong to the tab shown.
        self.workspace_tabs(ui);
        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                if self.workspace == meshworkspace::Workspace::Composer {
                    self.composer.menus(ui, &mut self.prefs_open);
                } else if self.workspace == meshworkspace::Workspace::Mesh {
                    self.mesh_menus(ui);
                } else if self.workspace == meshworkspace::Workspace::Landscape {
                    self.landscape_menus(ui);
                } else {
                ui.menu_button("File", |ui| {
                    if icons::menu_item(ui, "new", "New", "Ctrl+N").clicked() {
                        self.request(Pending::New);
                    }
                    icons::submenu(ui, "template", "New from template", |ui| {
                        for name in Graph::TEMPLATES {
                            if ui.button(name).clicked() {
                                self.request(Pending::Template(name));
                                ui.close();
                            }
                        }
                    });
                    if icons::menu_item(ui, "open", "Open…", "Ctrl+O").clicked() {
                        self.open();
                    }
                    ui.add_enabled_ui(!self.recent.is_empty(), |ui| {
                        icons::submenu(ui, "recent", "Open recent", |ui| {
                            let mut pick = None;
                            for p in &self.recent {
                                let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                                if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                                    pick = Some(p.clone());
                                    ui.close();
                                }
                            }
                            if let Some(p) = pick {
                                self.request(Pending::Path(p));
                            }
                        });
                    });
                    if icons::menu_item(ui, "save", "Save", "Ctrl+S").clicked() {
                        self.save(false);
                    }
                    if icons::menu_item(ui, "save_as", "Save As…", "Ctrl+Shift+S").clicked() {
                        self.save(true);
                    }
                    ui.separator();
                    if icons::menu_item(ui, "import_mesh", "Import mesh (OBJ)…", "").clicked()
                        && let Some(p) = rfd::FileDialog::new().add_filter("Mesh (OBJ, PLY, STL, glTF, FBX, SDF)", &["obj", "ply", "stl", "gltf", "glb", "fbx", "sdf"]).pick_file() {
                            self.start_mesh_load(p);
                        }
                    if icons::menu_item_enabled(ui, "export", "Export displaced mesh…", "", self.mesh.is_some()).clicked() {
                        self.mesh_export_open = true;
                    }
                    if icons::menu_item(ui, "batch", "Batch displace meshes…", "").clicked() {
                        self.batch_open = true;
                    }
                    ui.separator();
                    if icons::menu_item(ui, "export", "Export…", "").clicked() {
                        self.export_open = true;
                    }
                    if icons::menu_item_enabled(ui, "function", "Export Math Function (.noiser)…", "", self.export_job.is_none()).clicked() {
                        self.export_function();
                    }
                    if icons::menu_item(ui, "code", "Export shader code (HLSL, GLSL, WGSL)…", "").clicked() {
                        let name = self.path.as_ref().and_then(|p| p.file_stem()).map_or("noise".into(), |n| n.to_string_lossy().into_owned());
                        if let Some(path) = rfd::FileDialog::new().add_filter("HLSL", &["hlsl"]).add_filter("GLSL", &["glsl"]).add_filter("WGSL", &["wgsl"]).set_file_name(format!("{name}.hlsl")).save_file() {
                            self.status = match path.extension().and_then(|e| e.to_str()).and_then(shaderexport::Lang::from_extension) {
                                Some(lang) => match shaderexport::export(&self.graph, self.pv.seed, lang).and_then(|code| std::fs::write(&path, code).map_err(|e| e.to_string())) {
                                    Ok(()) => format!("Exported {}", path.display()),
                                    Err(e) => format!("Shader export failed: {e}"),
                                },
                                None => "Pick a .hlsl, .glsl or .wgsl file name".into(),
                            };
                        }
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if icons::menu_item_enabled(ui, "undo", "Undo", "Ctrl+Z", self.history.can_undo()).clicked() {
                        self.undo(false);
                    }
                    if icons::menu_item_enabled(ui, "redo", "Redo", "Ctrl+Y", self.history.can_redo()).clicked() {
                        self.undo(true);
                    }
                    ui.separator();
                    if icons::menu_item(ui, "arrange", "Arrange nodes", "").on_hover_text("Lay the graph out in columns, inputs on the left").clicked() {
                        editor::arrange(&mut self.graph);
                        self.editor.frame_next = true;
                    }
                    if icons::menu_item(ui, "tune", "Exposed parameters…", "").clicked() {
                        self.exposed_open = true;
                        ui.close();
                    }
                    if icons::menu_item(ui, "settings", "Preferences…", "").clicked() {
                        self.prefs_open = true;
                        ui.close();
                    }
                });
                }
                ui.menu_button("Help", |ui| {
                    if icons::menu_item(ui, "keyboard", "Keyboard shortcuts", "").clicked() {
                        self.shortcuts_open = true;
                        ui.close();
                    }
                    if icons::menu_item(ui, "info", "About Noiser", "").clicked() {
                        self.about_open = true;
                        ui.close();
                    }
                });
                ui.separator();
                ui.weak(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icons::button(ui, "settings", "Preferences", None, true).on_hover_text("Preferences").clicked() {
                        self.prefs_open = true;
                    }
                    let key = keymap::text(keymap::Action::Library);
                    if icons::button(ui, "library", "Library", Some(self.drawer.open), true).on_hover_text(format!("Library: saved SDFs, noise graphs and mesh graphs ({key})")).clicked() {
                        self.drawer.open = !self.drawer.open;
                    }
                });
            });
        });
        self.library_drawer(ui);
        match self.workspace {
            meshworkspace::Workspace::Mesh => {
                self.mesh_workspace_ui(ui);
                return;
            }
            meshworkspace::Workspace::Composer => {
                self.composer.show(ui, self.rs.target_format);
                return;
            }
            meshworkspace::Workspace::Landscape => {
                self.landscape_ui(ui);
                return;
            }
            meshworkspace::Workspace::Noise => {}
        }
        // The side panel: viewport settings in tabs on top, the selected node's settings in the bottom half.
        egui::Panel::right("props").default_size(320.0).show(ui, |ui| {
            let half = ui.available_height() * 0.5;
            egui::Panel::bottom("node props").default_size(half).min_size(120.0).resizable(true).show(ui, |ui| {
                // Not shrinking to the content keeps the panel's top edge where the user put it.
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let sel: Vec<_> = self.editor.selected.iter().copied().collect();
                match sel.as_slice() {
                    [id] if self.graph.nodes.contains_key(id) => {
                        let names = self.graph.declared_names();
                        let node = self.graph.nodes.get_mut(id).unwrap();
                        let has_output_kind = editor::has_output(&node.kind);
                        ui.heading(editor::title(&node.kind));
                        let mut label = node.label.clone().unwrap_or_default();
                        if ui.add(egui::TextEdit::singleline(&mut label).hint_text("Label (shown on the node)").desired_width(f32::INFINITY)).changed() {
                            node.label = Some(label).filter(|l| !l.is_empty());
                        }
                        node_props(ui, &mut node.kind, &mut node.pins, &names);
                        // Below the controls: appearing above them mid-drag shifted the slider out from
                        // under the cursor and ended the drag.
                        if let Some(Tiling::Broken(why)) = tiling.get(id) {
                            ui.colored_label(editor::ORANGE, format!("Does not tile: {why}"));
                        }
                        let (lo, hi) = self.graph.node_range(*id);
                        if has_output_kind {
                            let show = |x: f32| if x.is_finite() { format!("{x:.3}") } else if x < 0.0 { "-∞".into() } else { "∞".into() };
                            ui.weak(format!("Output stays within {} … {}", show(lo), show(hi)));
                        }
                        cache_props(ui, &mut self.graph, *id, &tiling, self.caches.bytes());
                    }
                    [] => _ = ui.weak("Select a node to edit it."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
                });
            });
            ui.horizontal_wrapped(|ui| {
                for (tab, name) in SideTab::ALL {
                    if icons::text_tab(ui, tab.icon(), name, self.side_tab == tab).clicked() {
                        self.side_tab = tab;
                    }
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match self.side_tab {
                SideTab::Mesh => self.mesh_props(ui),
                tab => preview_props(ui, &mut self.pv, self.mesh.is_some(), tab),
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
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.failed_build = Some(wgsl.clone());
                    self.status = "The shader could not be built (details in crash.log in Noiser's data folder); change the graph to try again".into();
                    self.compiling = None;
                }
            }
        }
        // The mesh displacement mask is only compiled when it is in use; the pipeline key says so.
        let mask_on = self.pv.prim == meshapp::MESH_PRIM && self.mesh.is_some() && !self.mesh_mask.terms.is_empty();
        let extras = view::Extras { mask_view: mask_on && self.pv.show_mask && !self.pv.cloud };
        if let Ok(c) = &compiled {
            let key = format!("{}{}{}", if mask_on { "//mask\n" } else { "" }, if extras.mask_view { "//maskview\n" } else { "" }, c.wgsl);
            let built = self.built.as_ref().is_some_and(|b| b.0 == key);
            let queued = self.compiling.as_ref().is_some_and(|p| p.0 == key);
            if !built && !queued && self.failed_build.as_ref() != Some(&key) {
                let (tx, rx) = std::sync::mpsc::channel();
                let (device, queue, format, wgsl) = (self.rs.device.clone(), self.rs.queue.clone(), self.rs.target_format, c.wgsl.clone());
                std::thread::spawn(move || _ = tx.send(view::build(&device, &queue, format, &wgsl, mask_on, extras)));
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
            library::set_view_rect(r3d);
            let resp = ui.interact(r3d, ui.scope_id().with("view3d"), egui::Sense::click_and_drag());
            if resp.double_clicked() {
                (self.yaw, self.pitch, self.dist) = (0.6, 0.35, 3.5); // back to the starting camera
            }
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
                    // Value under the pointer: the slice values are copied back only while the pointer is over the view.
                    let hover = ui.input(|i| i.pointer.hover_pos()).filter(|p| rslice.contains(*p));
                    if let Some(gpu) = self.rs.renderer.read().callback_resources.get::<view::Gpu>() {
                        gpu.want_slice_values(hover.is_some());
                        if let Some(p) = hover {
                            // The same mapping as fs_slice in slice.wgsl: a square fitted into the view, tiled 1 or 2 times.
                            let aspect = rslice.width() / rslice.height().max(1.0);
                            let (ux, uy) = ((p.x - rslice.left()) / rslice.width() * 2.0 - 1.0, 1.0 - (p.y - rslice.top()) / rslice.height() * 2.0);
                            let (qx, qy) = (ux * aspect / aspect.min(1.0), uy / aspect.min(1.0));
                            if qx.abs() <= 1.0 && qy.abs() <= 1.0 {
                                let tiles = if self.pv.tiles_2x2 { 2.0 } else { 1.0 };
                                let (tx, ty) = ((qx * 0.5 * tiles + 0.5 * tiles).rem_euclid(1.0), (-qy * 0.5 * tiles + 0.5 * tiles).rem_euclid(1.0));
                                let n = view::SLICE_RES;
                                let (ix, iy) = (((tx * n as f32) as u32).min(n - 1), ((ty * n as f32) as u32).min(n - 1));
                                if let Some(v) = gpu.slice_value(ix, iy) {
                                    label = format!("{label}
value {v:.4} at ({tx:.3}, {ty:.3})");
                                }
                            }
                            ui.ctx().request_repaint_after(std::time::Duration::from_millis(50)); // enough for a readout
                        }
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
                    if !library::capturing() {
                        painter.text(r3d.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, timing, font, grey);
                    }
                }
                Err(e) => {
                    ui.painter().rect_filled(full, 0.0, egui::Color32::from_gray(30));
                    ui.painter().text(full.center(), egui::Align2::CENTER_CENTER, e, egui::FontId::proportional(16.0), egui::Color32::RED);
                }
            }
        });
    }
}


/// Writes panics to `crash.log` in Noiser's data folder (the release build has no console, so they would
/// otherwise vanish), then lets the normal handling run.
/// The window icon (the same image is the exe's icon, see build.rs).
fn app_icon() -> egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/icon.png")).expect("assets/icon.png").to_rgba8();
    egui::IconData { width: img.width(), height: img.height(), rgba: img.into_raw() }
}

fn install_crash_log() {
    let Some(dir) = eframe::storage_dir("Noiser") else { return };
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("crash.log")) {
            let _ = writeln!(f, "[{when} unix seconds, Noiser {}] {info}
{}
", env!("CARGO_PKG_VERSION"), std::backtrace::Backtrace::force_capture());
        }
        previous(info);
    }));
}

fn main() -> eframe::Result {
    install_crash_log();
    // `--window 1600x1000` sets the window size (for screenshots).
    let args: Vec<String> = std::env::args().collect();
    let window = args.iter().position(|a| a == "--window").and_then(|i| args.get(i + 1)).and_then(|v| v.split_once('x')).and_then(|(w, h)| Some([w.parse::<f32>().ok()?, h.parse::<f32>().ok()?])).unwrap_or([1280.0, 800.0]);
    if let Some(code) = agent_cli::try_run() {
        std::process::exit(code);
    }
    if let Some(code) = headless::try_run() {
        std::process::exit(code);
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: view::request_timestamps(Default::default()),
        viewport: egui::ViewportBuilder::default().with_inner_size(window).with_title("Noiser").with_icon(app_icon()),
        ..Default::default()
    };
    eframe::run_native("Noiser", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
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
        assert_eq!(Prefs { ui_scale: f32::NAN, ..Prefs::default() }.sane().ui_scale, 1.0);
    }

    #[test]
    fn the_library_key_older_versions_saved_becomes_the_first_library_binding() {
        use crate::keymap::{Action, Binding};
        let p: Prefs = serde_json::from_str(r#"{"library_key": "F2"}"#).unwrap();
        let p = p.sane();
        assert_eq!(p.keymap.get(Action::Library)[0], Some(Binding::plain(eframe::egui::Key::F2)));
        assert!(!serde_json::to_string(&p).unwrap().contains("library_key"), "it is saved as a keybind from now on");
    }
}

#[cfg(test)]
mod lighting_tests {
    use super::Lighting;

    #[test]
    fn default_lighting_is_the_old_fixed_sun() {
        let p = Lighting::default().pack();
        for (got, want) in p[0][..3].iter().zip([0.5345, 0.7127, 0.4543]) {
            assert!((got - want).abs() < 2e-3, "{got} vs {want}");
        }
        assert_eq!(p[0][3], 1.0);
        for (_, preset) in Lighting::PRESETS {
            let d = preset().pack()[0];
            assert!(((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - 1.0).abs() < 1e-4, "the sun direction is a unit vector");
        }
    }
}

#[cfg(test)]
mod curve_ui_tests {
    use super::*;
    use crate::graph::NodeKind;
    use egui_kittest::{Harness, kittest::Queryable};

    #[test]
    fn the_curve_panel_draws_and_lists_every_point() {
        let kind = NodeKind::Curve { points: vec![[0.0, 0.0], [0.5, 1.0], [1.0, 0.2]], mode: graph::CurveMode::Cubic };
        let mut h = Harness::builder().with_size(egui::Vec2::new(400.0, 600.0)).build_ui_state(|ui, kind: &mut NodeKind| node_props(ui, kind, &mut Vec::new(), &[]), kind);
        h.run();
        assert!(h.get_all_by_label("x").count() >= 3, "a remove button per point");
        h.get_by_label("Cubic");
    }

    #[test]
    fn dragging_a_point_in_the_curve_plot_moves_it() {
        let kind = NodeKind::Curve { points: vec![[0.0, 0.0], [0.5, 1.0], [1.0, 0.2]], mode: graph::CurveMode::Linear };
        let mut h = Harness::builder().with_size(egui::Vec2::new(400.0, 600.0)).with_step_dt(1.0 / 60.0).build_ui_state(|ui, kind: &mut NodeKind| node_props(ui, kind, &mut Vec::new(), &[]), kind);
        h.run();
        let label = h.get_by_label("Interpolation").rect();
        let (left, top) = (label.left(), label.bottom() + ui_spacing());
        // The plot is 300 x 150 and shows x in -0.08..1.08, y in -0.08..1.08 for these points.
        let at = |x: f32, y: f32| egui::Pos2::new(left + (x + 0.08) / 1.16 * 300.0, top + 150.0 - (y + 0.08) / 1.16 * 150.0);
        let (from, to) = (at(0.5, 1.0), at(0.4, 0.5));
        h.hover_at(from);
        h.run();
        h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() });
        h.run();
        for i in 1..=8 {
            h.hover_at(from + (to - from) * i as f32 / 8.0);
            h.run();
        }
        h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() });
        h.run();
        let NodeKind::Curve { points, .. } = h.state() else { panic!() };
        assert!((points[1][0] - 0.4).abs() < 0.08 && (points[1][1] - 0.5).abs() < 0.12, "the middle point is now {:?}", points[1]);
        assert_eq!(points[0], [0.0, 0.0], "the others stay");
    }

    /// The vertical gap egui puts between widgets by default.
    fn ui_spacing() -> f32 {
        egui::Style::default().spacing.item_spacing.y
    }
}

#[cfg(test)]
mod icon_tests {
    /// Regenerates assets/icon.png and assets/icon.ico (the exe's icon) from assets/icons/landslide.svg: a white glyph on a dark rounded square.
    #[test]
    #[ignore = "writes assets/icon.png and assets/icon.ico"]
    fn write_app_icon() {
        let svg = String::from_utf8_lossy(include_bytes!("../assets/icons/landslide.svg")).replace("currentColor", "#ffffff");
        let side = 512usize;
        let glyph_side = 360usize;
        let glyph = egui_extras::image::load_svg_bytes_with_size(svg.as_bytes(), eframe::egui::load::SizeHint::Size { width: glyph_side as u32, height: glyph_side as u32, maintain_aspect_ratio: true }, &Default::default()).unwrap();
        let mut img = image::RgbaImage::new(side as u32, side as u32);
        let (radius, bg) = (96.0f32, [36u8, 40, 46, 255]);
        for (x, y, px) in img.enumerate_pixels_mut() {
            // A rounded square covering the whole image.
            let (dx, dy) = ((x as f32 + 0.5 - 256.0).abs() - (256.0 - radius), (y as f32 + 0.5 - 256.0).abs() - (256.0 - radius));
            let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() - radius;
            if outside < 0.5 {
                *px = image::Rgba(bg);
            }
        }
        let (ox, oy) = ((side - glyph.size[0]) / 2, (side - glyph.size[1]) / 2);
        for y in 0..glyph.size[1] {
            for x in 0..glyph.size[0] {
                let g = glyph.pixels[y * glyph.size[0] + x];
                let a = g.a() as f32 / 255.0;
                let px = img.get_pixel_mut((ox + x) as u32, (oy + y) as u32);
                for c in 0..3 {
                    px.0[c] = (px.0[c] as f32 * (1.0 - a) + 255.0 * a) as u8;
                }
            }
        }
        img.save("assets/icon.png").unwrap();
        // An .ico holding PNG images at several sizes.
        let sizes = [256u32, 64, 48, 32, 16];
        let pngs: Vec<Vec<u8>> = sizes
            .iter()
            .map(|&s| {
                let mut out = Vec::new();
                image::DynamicImage::ImageRgba8(image::imageops::resize(&img, s, s, image::imageops::FilterType::Lanczos3)).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).unwrap();
                out
            })
            .collect();
        let mut ico = vec![0u8, 0, 1, 0];
        ico.extend((sizes.len() as u16).to_le_bytes());
        let mut offset = 6 + 16 * sizes.len();
        for (s, png) in sizes.iter().zip(&pngs) {
            ico.extend([if *s >= 256 { 0 } else { *s as u8 }, if *s >= 256 { 0 } else { *s as u8 }, 0, 0, 1, 0, 32, 0]);
            ico.extend((png.len() as u32).to_le_bytes());
            ico.extend((offset as u32).to_le_bytes());
            offset += png.len();
        }
        for png in pngs {
            ico.extend(png);
        }
        std::fs::write("assets/icon.ico", ico).unwrap();
    }
}
