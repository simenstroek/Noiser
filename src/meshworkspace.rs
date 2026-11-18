//! The Mesh graph tab: its own graph and node editor, a toolbar (new, open, save, run) and a panel with the selected
//! node's settings and the log of the last run. Running happens on a worker thread; see `meshgraph::Runner`.

use crate::graph::{Graph, NodeKind};
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
}

struct RunJob {
    progress: crate::meshexport::Progress,
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
}

impl Default for MeshWorkspace {
    fn default() -> Self {
        let graph = Self::example();
        let mut editor = crate::editor::Editor::default();
        editor.mesh_mode = true;
        editor.frame_next = true;
        MeshWorkspace { saved: graph.clone(), graph, editor, path: None, cache: Arc::new(Mutex::new(MeshCache::default())), job: None, log: Vec::new(), written: Vec::new(), seed: 0 }
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
        let remesh = g.add(NodeKind::Remesh { triangles: 20_000 }, [400.0, 60.0]);
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
        self.editor.frame_next = true;
    }

    pub fn dirty(&self) -> bool {
        self.graph != self.saved
    }

    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    fn set(&mut self, graph: Graph, path: Option<PathBuf>) {
        self.saved = graph.clone();
        self.graph = graph;
        self.path = path;
        self.editor.selected.clear();
        self.editor.frame_next = true;
    }

    /// Writes the graph, first refreshing every noise graph node's embedded copy from its file, so the mesh graph
    /// carries what it was saved with (the file stays the source of truth).
    pub fn save(&mut self, ask: bool) -> Result<(), String> {
        let path = match (&self.path, ask) {
            (Some(p), false) => p.clone(),
            _ => match rfd::FileDialog::new().add_filter("Mesh graph", &["meshgraph"]).set_file_name("mesh.meshgraph").save_file() {
                Some(p) => p,
                None => return Ok(()),
            },
        };
        for n in self.graph.nodes.values_mut() {
            if let NodeKind::NoiseGraph(p) = &mut n.kind {
                if let Ok(text) = std::fs::read_to_string(&p.file) {
                    p.embedded = text;
                }
            }
        }
        std::fs::write(&path, self.graph.to_json()).map_err(|e| format!("{}: {e}", path.display()))?;
        self.saved = self.graph.clone();
        self.path = Some(path);
        Ok(())
    }

    pub fn open(&mut self) -> Result<(), String> {
        let Some(path) = rfd::FileDialog::new().add_filter("Mesh graph", &["meshgraph"]).pick_file() else { return Ok(()) };
        let g = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display())).and_then(|s| Graph::from_json(&s))?;
        self.set(g, Some(path));
        Ok(())
    }

    fn start(&mut self, device: eframe::wgpu::Device, queue: eframe::wgpu::Queue) {
        let progress = crate::meshexport::Progress::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let (graph, cache, seed, p, l) = (self.graph.clone(), self.cache.clone(), self.seed, progress.clone(), log.clone());
        let base = self.path.as_ref().and_then(|p| p.parent()).map_or(std::env::current_dir().unwrap_or_default(), |p| p.to_path_buf());
        std::thread::spawn(move || {
            let mut note = |s: &str| l.lock().unwrap().push(s.to_string());
            let mut cache = cache.lock().unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Runner { device: &device, queue: &queue, cache: &mut cache, base, seed, progress: p, log: &mut note }.run(&graph)));
            let _ = tx.send(result.unwrap_or_else(|_| Err("the run crashed".into())));
        });
        self.log.clear();
        self.written.clear();
        self.job = Some(RunJob { progress, log, done: rx });
    }
}

impl crate::App {
    /// The whole Mesh graph tab (the menu bar above it is the app's).
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
                Ok(files) => {
                    self.status = format!("Mesh graph: {} file(s) written", files.len());
                    self.mesh_ws.written = files;
                }
                Err(e) => {
                    self.mesh_ws.log.push(format!("FAILED: {e}"));
                    self.status = format!("Mesh graph failed: {e}");
                }
            }
        }
        // Ctrl+S and Ctrl+O act on this graph here.
        let (ctrl_s, ctrl_shift) = ui.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::S), i.modifiers.shift));
        if ctrl_s {
            if let Err(e) = self.mesh_ws.save(ctrl_shift) {
                self.status = e;
            }
        }
        egui::Panel::right("mesh props").default_size(340.0).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("New").clicked() {
                    self.mesh_ws.set(MeshWorkspace::example(), None);
                }
                if ui.button("Open…").clicked() {
                    if let Err(e) = self.mesh_ws.open() {
                        self.status = e;
                    }
                }
                if ui.button("Save").on_hover_text("Ctrl+S. Also refreshes the embedded copy of every noise graph.").clicked() {
                    if let Err(e) = self.mesh_ws.save(false) {
                        self.status = e;
                    }
                }
                if ui.button("Save as…").clicked() {
                    if let Err(e) = self.mesh_ws.save(true) {
                        self.status = e;
                    }
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
                if ui.add_enabled(!running, egui::Button::new("▶ Run")).on_hover_text("Runs every Remesh, Auto UVs and Bake that feeds an output node and writes the files.").clicked() {
                    let (d, q) = (self.rs.device.clone(), self.rs.queue.clone());
                    self.mesh_ws.start(d, q);
                }
                ui.add(egui::DragValue::new(&mut self.mesh_ws.seed).prefix("seed "));
            });
            if let Some(job) = &self.mesh_ws.job {
                let p = job.progress.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                ui.add(egui::ProgressBar::new(p).show_percentage());
            }
            egui::CollapsingHeader::new("Log").default_open(true).show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(140.0).stick_to_bottom(true).show(ui, |ui| {
                    for l in &self.mesh_ws.log {
                        ui.weak(l);
                    }
                    for f in &self.mesh_ws.written {
                        ui.label(format!("wrote {}", f.display()));
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
                        crate::node_props(ui, &mut node.kind, &names);
                    }
                    [] => _ = ui.weak("Select a node to edit it. Double-click a Noise Graph node to open that graph in the Noise graph tab."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
            });
        });
        self.mesh_ws.graph.prune_edges();
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            self.mesh_ws.editor.show(ui, &mut self.mesh_ws.graph, &Default::default());
        });
        // Double-clicking a Noise Graph node goes to that graph.
        if let Some(id) = self.mesh_ws.editor.opened.take() {
            if let Some(NodeKind::NoiseGraph(p)) = self.mesh_ws.graph.nodes.get(&id).map(|n| n.kind.clone()) {
                if p.file.is_empty() {
                    self.status = "This Noise Graph node has no file chosen yet".into();
                } else {
                    self.workspace = Workspace::Noise;
                    self.request(crate::Pending::Path(PathBuf::from(p.file)));
                }
            }
        }
    }
}
