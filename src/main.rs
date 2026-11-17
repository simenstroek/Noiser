mod backup;
mod bluenoise;
mod codegen;
mod editor;
mod graph;
#[cfg(test)]
mod gpu_tests;
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
    /// WGSL the current pipeline was built from; a different compile result triggers a rebuild.
    built_wgsl: Option<String>,
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
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        // `noiser path.noisegraph` opens that graph.
        let (graph, path, status) = match std::env::args().nth(1).map(std::path::PathBuf::from) {
            Some(p) => match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
                Ok(g) => (g, Some(p.clone()), format!("Opened {}", p.display())),
                Err(e) => (Graph::default_graph(), None, format!("Could not open {}: {e}", p.display())),
            },
            None => (Graph::default_graph(), None, String::new()),
        };
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
        Self {
            rs: cc.wgpu_render_state.clone().expect("Noiser needs the wgpu renderer"),
            saved: graph.clone(),
            graph,
            editor,
            path,
            status,
            built_wgsl: None,
            yaw: 0.6,
            pitch: 0.35,
            dist: 3.5,
            pv: Preview::default(),
            session,
            recovery,
        }
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
        let push = view::push_range(range, pv.amount);
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
            pixel: 2.0 / view3d.height().max(1.0),
            cloud_density: pv.cloud_density,
            mode: pv.cloud as u32,
            prim: pv.prim,
            slice_axis: pv.slice_axis,
            slice_mode: pv.heatmap as u32,
            slice_pos: pv.slice_pos,
            slice_aspect: aspect(slice),
            slice_tiles: if pv.tiles_2x2 { 2.0 } else { 1.0 },
            slice_res: view::SLICE_RES,
            push_min: push.0,
            push_max: push.1,
            _pad: [0.0; 2],
        }
    }

    fn file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new().add_filter("Noiser graph", &["noisegraph"])
    }

    fn open(&mut self) {
        let Some(path) = Self::file_dialog().pick_file() else { return };
        match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| Graph::from_json(&s)) {
            Ok(g) => {
                self.saved = g.clone();
                self.graph = g;
                self.editor.selected.clear();
                self.editor.preview = None;
                self.status = format!("Opened {}", path.display());
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

fn preview_props(ui: &mut egui::Ui, pv: &mut Preview) {
    let d = Preview::default();
    ui.heading("3D view");
    row(ui, &mut pv.cloud, d.cloud, |ui, v| {
        ui.selectable_value(v, false, "Solid");
        ui.selectable_value(v, true, "Cloud");
    });
    row(ui, &mut pv.prim, d.prim, |ui, v| {
        egui::ComboBox::from_label("shape").selected_text(PRIMS[*v as usize]).show_ui(ui, |ui| {
            for (i, name) in PRIMS.iter().enumerate() {
                ui.selectable_value(v, i as u32, *name);
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
        NodeKind::Math { op, a, b } => {
            row(ui, op, MathOp::Add, |ui, op| {
                egui::ComboBox::from_label("operation").selected_text(format!("{op:?}")).show_ui(ui, |ui| {
                    for o in [MathOp::Add, MathOp::Subtract, MathOp::Multiply, MathOp::Divide, MathOp::Min, MathOp::Max] {
                        ui.selectable_value(op, o, format!("{o:?}"));
                    }
                });
            });
            ui.label("Used when the input is not connected:");
            row(ui, a, 0.0, |ui, a| _ = ui.add(egui::DragValue::new(a).speed(0.01).prefix("A ")));
            row(ui, b, 0.0, |ui, b| _ = ui.add(egui::DragValue::new(b).speed(0.01).prefix("B ")));
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
                    for (f, name) in [(F1, "F1"), (F2, "F2"), (F2MinusF1, "F2 − F1")] {
                        ui.selectable_value(v, f, name);
                    }
                });
            }
            row(ui, &mut n.scale, d.scale, |ui, v| _ = ui.add(egui::Slider::new(v, 0.5..=32.0).text("scale (cells/tile)")));
            row(ui, &mut n.normalize, d.normalize, |ui, v| _ = ui.checkbox(v, "normalize 0-1"));
            if n.kind.has_octaves() {
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

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(s) = serde_json::to_string(&self.editor.picks) {
            storage.set_string(PICKS_KEY, s);
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
        if let Some(s) = self.session.as_mut().filter(|_| self.graph != self.saved) {
            if s.backup(&self.graph, self.path.as_deref()) {
                ui.ctx().request_repaint_after(backup::INTERVAL); // flush the held-back change
            }
        }
        let tiling = self.graph.tiling();

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
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
                    if ui.button("Save").clicked() {
                        self.save(false);
                    }
                    if ui.button("Save As…").clicked() {
                        self.save(true);
                    }
                });
                ui.weak(&self.status);
            });
        });
        egui::Panel::right("props").default_size(320.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                preview_props(ui, &mut self.pv);
                ui.separator();
                let sel: Vec<_> = self.editor.selected.iter().copied().collect();
                match sel.as_slice() {
                    [id] if self.graph.nodes.contains_key(id) => {
                        let node = self.graph.nodes.get_mut(id).unwrap();
                        ui.heading(editor::title(&node.kind));
                        if let Some(Tiling::Broken(why)) = tiling.get(id) {
                            ui.colored_label(editor::ORANGE, format!("Does not tile: {why}"));
                        }
                        node_props(ui, &mut node.kind);
                    }
                    [] => _ = ui.weak("Select a node to edit it."),
                    _ => _ = ui.weak(format!("{} nodes selected", sel.len())),
                }
            });
        });
        egui::Panel::bottom("graph").resizable(true).default_size(330.0).show(ui, |ui| {
            self.editor.show(ui, &mut self.graph, &tiling);
        });
        // The views show the Density output, or the previewed node wired into it.
        let shown = match self.editor.preview {
            Some(id) => std::borrow::Cow::Owned(self.graph.previewing(id)),
            None => std::borrow::Cow::Borrowed(&self.graph),
        };
        let range = shown.density_range();
        let compiled = match codegen::compile(&shown) {
            Ok(c) if c.params.len() > MAX_PARAMS => Err(format!("graph uses {} parameters (max {MAX_PARAMS})", c.params.len())),
            r => r,
        };
        if let Ok(c) = &compiled {
            if self.built_wgsl.as_deref() != Some(&c.wgsl) {
                let gpu = view::build(&self.rs.device, &self.rs.queue, self.rs.target_format, &c.wgsl);
                self.rs.renderer.write().callback_resources.insert(gpu);
                self.built_wgsl = Some(c.wgsl.clone());
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
                Ok(c) => {
                    let uniforms = self.uniforms(r3d, rslice, range);
                    let painter = ui.painter();
                    let px = (r3d.size() * ui.ctx().pixels_per_point()).round();
                    let cb3d = view::View3d { uniforms, params: c.params.clone(), size: [px.x as u32, px.y as u32] };
                    painter.add(egui_wgpu::Callback::new_paint_callback(r3d, cb3d));
                    let cbs = view::SliceView { uniforms, params: c.params };
                    painter.add(egui_wgpu::Callback::new_paint_callback(rslice, cbs));
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
