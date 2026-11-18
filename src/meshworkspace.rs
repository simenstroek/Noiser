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
    /// Start a run on the first frame (for `--run-mesh-graph`).
    pub autorun: bool,
    /// The first mesh the last run wrote, for the view above the graph.
    preview: Option<Preview>,
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
}

impl Preview {
    fn load(path: &std::path::Path) -> Result<Preview, String> {
        let mesh = crate::meshformats::load(path)?;
        let (lo, hi) = mesh.bounds();
        let centre = [0, 1, 2].map(|a| 0.5 * (lo[a] + hi[a]));
        let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max).max(1e-9) * 0.5;
        let positions = mesh.positions.iter().map(|p| [0, 1, 2].map(|a| (p[a] - centre[a]) / radius)).collect();
        Ok(Preview { positions, triangles: mesh.triangles, name: path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()) })
    }

    /// Flat-shaded triangles, far ones first, as one egui mesh in `rect`.
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
            let facing = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / len;
            // Faces turned away show dark red, which also exposes inside-out or folded triangles.
            let colour = if n[2] < 0.0 {
                egui::Color32::from_rgb(110, 25, 25)
            } else {
                let k = 0.2 + 0.8 * facing.max(0.0);
                egui::Color32::from_rgb((205.0 * k) as u8, (195.0 * k) as u8, (175.0 * k) as u8)
            };
            let base = mesh.vertices.len() as u32;
            for p in [a, b, c] {
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
        MeshWorkspace { saved: graph.clone(), graph, editor, path: None, cache: Arc::new(Mutex::new(MeshCache::default())), job: None, log: Vec::new(), written: Vec::new(), seed: 0, autorun: false, preview: None, history: crate::history::History::new(&Graph::default()), yaw: 0.6, pitch: 0.35, zoom: 1.0 }
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
            _ => match rfd::FileDialog::new().add_filter("Mesh graph", &["meshgraph"]).set_file_name("mesh.meshgraph").save_file() {
                Some(p) => p,
                None => return Ok(()),
            },
        };
        let base = path.parent().map_or(PathBuf::from("."), |p| p.to_path_buf());
        for n in self.graph.nodes.values_mut() {
            if let NodeKind::NoiseGraph(p) = &mut n.kind {
                if let Ok(text) = std::fs::read_to_string(crate::meshgraph::resolve_in(&base, &p.file)) {
                    p.embedded = text;
                }
            }
        }
        let text = self.graph.to_json();
        crate::keep_previous_version(&path, &text);
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
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

    fn start(&mut self, device: eframe::wgpu::Device, queue: eframe::wgpu::Queue, files: Vec<Option<PathBuf>>) {
        let progress = crate::meshexport::Progress::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let (graph, cache, seed, p, l) = (self.graph.clone(), self.cache.clone(), self.seed, progress.clone(), log.clone());
        let base = self.base();
        std::thread::spawn(move || {
            let mut cache = cache.lock().unwrap();
            let (mut written, mut failures) = (Vec::new(), Vec::new());
            let total = files.len();
            for (i, file) in files.into_iter().enumerate() {
                let label = file.as_ref().map_or(String::new(), |f| format!("[{}/{total}] {}: ", i + 1, f.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())));
                let mut note = |s: &str| l.lock().unwrap().push(format!("{label}{s}"));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Runner { device: &device, queue: &queue, cache: &mut cache, base: base.clone(), seed, progress: p.clone(), log: &mut note, batch_file: file }.run(&graph)));
                match result.unwrap_or_else(|_| Err("the run crashed".into())) {
                    Ok(f) => written.extend(f),
                    Err(e) => {
                        l.lock().unwrap().push(format!("{label}FAILED: {e}"));
                        failures.push(e);
                    }
                }
            }
            let _ = tx.send(if failures.is_empty() { Ok(written) } else { Err(format!("{} of {total} run(s) failed: {}", failures.len(), failures[0])) });
        });
        self.log.clear();
        self.written.clear();
        self.job = Some(RunJob { progress, log, done: rx });
    }
}

impl crate::App {
    /// Saves the mesh graph. A noise graph that is open and changed in the Noise graph tab, and that this mesh graph
    /// uses, is saved first, so the copy embedded in the mesh graph is the one just edited.
    fn save_mesh_graph(&mut self, ask: bool) {
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
                    self.mesh_ws.preview = files.iter().find(|f| f.extension().is_some_and(|e| e == "obj")).and_then(|f| Preview::load(f).ok());
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
            let (d, q) = (self.rs.device.clone(), self.rs.queue.clone());
            self.mesh_ws.start(d, q, vec![None]);
        }
        crate::meshgraph_ui::set_base(self.mesh_ws.base());
        // Ctrl+Z and Ctrl+Y (or Ctrl+Shift+Z) act on this graph here, unless a text field has the keyboard.
        if !ctx.egui_wants_keyboard_input() {
            let (z, y, shift) = ui.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::Z), i.modifiers.command && i.key_pressed(egui::Key::Y), i.modifiers.shift));
            if y || (z && shift) {
                self.mesh_ws.undo(true);
            } else if z {
                self.mesh_ws.undo(false);
            }
        }
        // Ctrl+S and Ctrl+O act on this graph here.
        let (ctrl_s, ctrl_shift) = ui.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::S), i.modifiers.shift));
        if ctrl_s {
            self.save_mesh_graph(ctrl_shift);
        }
        egui::Panel::right("mesh props").default_size(340.0).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("New").clicked() && self.mesh_ws.may_discard() {
                    self.mesh_ws.set(MeshWorkspace::example(), None);
                }
                if ui.button("Open…").clicked() && self.mesh_ws.may_discard() {
                    if let Err(e) = self.mesh_ws.open() {
                        self.status = e;
                    }
                }
                if ui.button("Save").on_hover_text("Ctrl+S. Saves the mesh graph, refreshes the embedded copy of every noise graph it uses, and saves the open noise graph too if it is one of them.").clicked() {
                    self.save_mesh_graph(false);
                }
                if ui.button("Save as…").clicked() {
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
                if ui.add_enabled(!running, egui::Button::new("Run")).on_hover_text("Runs every Remesh, Auto UVs and Bake that feeds an output node and writes the files.").clicked() {
                    let (d, q) = (self.rs.device.clone(), self.rs.queue.clone());
                    self.mesh_ws.start(d, q, vec![None]);
                }
                if ui.add_enabled(!running, egui::Button::new("Run on files…")).on_hover_text("Runs the graph once for each mesh file you pick: a Load Mesh node whose path is {batch} reads that file, and {name} in an output name becomes its name.").clicked() {
                    if let Some(files) = rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply", "stl", "gltf", "glb"]).pick_files() {
                        let (d, q) = (self.rs.device.clone(), self.rs.queue.clone());
                        self.mesh_ws.start(d, q, files.into_iter().map(Some).collect());
                    }
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
        let settled = !ui.input(|i| i.pointer.any_down());
        self.mesh_ws.history.commit(&self.mesh_ws.graph, settled);
        egui::Panel::bottom("mesh graph editor").resizable(true).default_size(330.0).show(ui, |ui| {
            self.mesh_ws.editor.show(ui, &mut self.mesh_ws.graph, &Default::default());
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            let resp = ui.interact(rect, ui.scope_id().with("mesh preview"), egui::Sense::click_and_drag());
            let d = resp.drag_delta();
            self.mesh_ws.yaw -= d.x * 0.01;
            self.mesh_ws.pitch = (self.mesh_ws.pitch + d.y * 0.01).clamp(-1.5, 1.5);
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.mesh_ws.zoom = (self.mesh_ws.zoom * (1.0 + scroll * 0.002)).clamp(0.2, 8.0);
            }
            if resp.double_clicked() {
                (self.mesh_ws.yaw, self.mesh_ws.pitch, self.mesh_ws.zoom) = (0.6, 0.35, 1.0);
            }
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_gray(28));
            match &self.mesh_ws.preview {
                Some(p) => {
                    p.paint(&painter, rect, self.mesh_ws.yaw, self.mesh_ws.pitch, self.mesh_ws.zoom);
                    painter.text(rect.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, format!("{}: {} triangles. Drag to turn, scroll to zoom, double-click to reset.", p.name, p.triangles.len()), egui::FontId::proportional(12.0), egui::Color32::from_gray(160));
                }
                None => _ = painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Run the graph to see its mesh here", egui::FontId::proportional(14.0), egui::Color32::from_gray(120)),
            }
        });
        // Double-clicking a Noise Graph node goes to that graph.
        if let Some(id) = self.mesh_ws.editor.opened.take() {
            if let Some(NodeKind::NoiseGraph(p)) = self.mesh_ws.graph.nodes.get(&id).map(|n| n.kind.clone()) {
                if p.file.is_empty() {
                    self.status = "This Noise Graph node has no file chosen yet".into();
                } else {
                    self.workspace = Workspace::Noise;
                    self.request(crate::Pending::Path(crate::meshgraph::resolve_in(&self.mesh_ws.base(), &p.file)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
