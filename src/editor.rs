//! Node graph canvas: pan/zoom, drag nodes, wire ports, box select, add/delete/duplicate, preview.
//!
//! Controls: middle-drag pans, scroll zooms, left-drag on empty space box-selects, right-click
//! adds a node, Delete removes the selection, Ctrl+D duplicates it at the cursor.
//! Drag from an output to an input (or the other way) to connect; drag a connected input away to
//! disconnect it; drop a wire on empty space to pick a new node for it. The eye in a node's
//! header previews that node in the 3D and slice views.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, NoiseParams, PortType, Tiling};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, epaint::CubicBezierShape};
use std::collections::{BTreeMap, BTreeSet};

pub const ORANGE: Color32 = Color32::from_rgb(255, 150, 40);
const ACCENT: Color32 = Color32::from_rgb(140, 190, 255);
/// Pin and wire colors: floats blue, vectors yellow.
fn type_color(t: PortType) -> Color32 {
    match t {
        PortType::Float => Color32::from_rgb(100, 165, 255),
        PortType::Vector => Color32::from_rgb(255, 205, 70),
    }
}

/// Cache on, and nodes a cache spares the preview from evaluating.
const CACHED: Color32 = Color32::from_rgb(120, 220, 160);

const NODE_W: f32 = 140.0;
const HEADER_H: f32 = 22.0;
const ROW_H: f32 = 20.0;
const PORT_R: f32 = 5.0;

/// A wire being dragged: from a node's output, or backwards from an unconnected input.
#[derive(Clone, Copy)]
enum Wire {
    FromOutput(NodeId),
    FromInput(NodeId, usize),
}

/// The node menu shown after a wire is dropped on empty space.
struct PinMenu {
    screen: Pos2,
    graph: Pos2,
    wire: Wire,
}

pub struct Editor {
    pan: Vec2,
    zoom: f32,
    pub selected: BTreeSet<NodeId>,
    /// Snap nodes to a 20-unit grid when a drag ends (a preference).
    pub snap: bool,
    /// Frame all nodes on the next draw (set when a graph is opened, so it is in view).
    pub frame_next: bool,
    /// A node was dragged since the pointer went down (so releasing may splice it into a wire).
    node_dragged: bool,
    /// Node shown in the 3D and slice views instead of the Density output.
    pub preview: Option<NodeId>,
    /// How often each node was picked from the wire-drop menu; most used are listed first.
    pub picks: BTreeMap<String, u32>,
    wire: Option<Wire>,
    pin_menu: Option<PinMenu>,
    box_start: Option<Pos2>,
    /// Graph-space point where the add menu was opened.
    menu_at: Pos2,
    /// Text typed into the add menu's search box, and whether it still needs the keyboard focus.
    menu_search: String,
    menu_fresh: bool,
    /// A scroll gesture over an operator dropdown already stepped it (egui smooths one wheel notch
    /// over several frames, so a step happens once per gesture, not per frame).
    op_scroll_stepped: bool,
    #[cfg(test)]
    debug_origin: Pos2,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            pan: Vec2::new(40.0, 40.0),
            zoom: 1.0,
            selected: BTreeSet::new(),
            snap: false,
            frame_next: false,
            node_dragged: false,
            preview: None,
            picks: BTreeMap::new(),
            wire: None,
            pin_menu: None,
            box_start: None,
            menu_at: Pos2::ZERO,
            menu_search: String::new(),
            menu_fresh: false,
            op_scroll_stepped: false,
            #[cfg(test)]
            debug_origin: Pos2::ZERO,
        }
    }
}

pub fn title(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => "Value".into(),
        NodeKind::Math { .. } => "Math".into(), // the operator is the dropdown on the node
        NodeKind::Remap { .. } => "Remap".into(),
        NodeKind::Mix { .. } => "Mix".into(),
        NodeKind::Curve { .. } => "Curve".into(),
        NodeKind::Grid { .. } => "Grid".into(),
        NodeKind::Pattern(p) => p.kind.name().into(),
        NodeKind::Position => "Position".into(),
        NodeKind::Reroute => "Reroute".into(),
        NodeKind::Normal { .. } => "Normal".into(),
        NodeKind::Noise(n) => n.kind.name().into(),
        NodeKind::OutDensity => "Density Output".into(),
        NodeKind::OutNormal => "Normal Output".into(),
    }
}

fn input_label(kind: &NodeKind, port: usize) -> &'static str {
    match (kind, port) {
        (NodeKind::Math { .. }, 0) => "A",
        (NodeKind::Math { .. }, 1) => "B",
        (NodeKind::Math { .. }, _) => "Blend",
        (NodeKind::Reroute, _) => "In",
        (NodeKind::Curve { .. }, _) => "Value",
        (NodeKind::Mix { .. }, 0) => "Factor",
        (NodeKind::Mix { .. }, 1) => "A",
        (NodeKind::Mix { .. }, _) => "B",
        (NodeKind::Remap { .. }, 0) => "Value",
        (NodeKind::Remap { .. }, 1) => "From min",
        (NodeKind::Remap { .. }, 2) => "From max",
        (NodeKind::Remap { .. }, 3) => "To min",
        (NodeKind::Remap { .. }, _) => "To max",
        (NodeKind::Grid { .. } | NodeKind::Pattern(_), _) => "Position",
        (NodeKind::Noise(_), _) => "Position",
        (NodeKind::Normal { .. }, _) => "Field",
        (NodeKind::OutDensity, _) => "Density",
        (NodeKind::OutNormal, _) => "Normal",
        _ => "",
    }
}

pub fn has_output(kind: &NodeKind) -> bool {
    !matches!(kind, NodeKind::OutDensity | NodeKind::OutNormal)
}

/// One-line summary shown in the node body for nodes without inputs.
fn summary(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(v) => format!("{v:.3}"),
        NodeKind::Vec3(v) => format!("{:.2}, {:.2}, {:.2}", v[0], v[1], v[2]),
        NodeKind::Position => "world position".into(),
        NodeKind::Reroute => "passes its input on".into(),
        _ => String::new(),
    }
}

/// Nodes offered by the add menus.
fn palette() -> Vec<(&'static str, NodeKind)> {
    let noises = NoiseKind::ALL.map(|kind| (kind.name(), NodeKind::Noise(NoiseParams { kind, ..Default::default() })));
    noises
        .into_iter()
        .chain([
            ("Value", NodeKind::Float(1.0)),
            ("Math", NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }),
            ("Remap", NodeKind::Remap { vals: [0.0, 0.0, 1.0, 0.0, 1.0], clamp: false }),
            ("Mix", NodeKind::Mix { vals: [0.5, 0.0, 1.0], clamp: true }),
            ("Reroute", NodeKind::Reroute),
            ("Curve", NodeKind::Curve { points: vec![[0.0, 0.0], [1.0, 1.0]], mode: crate::graph::CurveMode::Linear }),
            ("Grid", NodeKind::Grid { counts: [8.0; 3], corners: false }),
            ("Bricks", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Bricks))),
            ("Dimples", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Dimples))),
            ("Checker", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Checker))),
            ("Stripes", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Stripes))),
            ("Gyroid", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Gyroid))),
            ("Fractal", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Fractal))),
            ("Truchet", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Truchet))),
            ("Honeycomb", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Honeycomb))),
            ("Position", NodeKind::Position),
            ("Normal", NodeKind::Normal { flip: false }),
            ("Density Output", NodeKind::OutDensity),
            ("Normal Output", NodeKind::OutNormal),
        ])
        .collect()
}

/// Nodes that can take the dragged wire, most picked first (palette order breaks ties).
fn wire_choices(wire: Wire, picks: &BTreeMap<String, u32>) -> Vec<(&'static str, NodeKind)> {
    let mut v: Vec<_> = palette()
        .into_iter()
        .filter(|(_, k)| match wire {
            Wire::FromOutput(_) => k.input_count() > 0,
            Wire::FromInput(..) => has_output(k),
        })
        .collect();
    v.sort_by_key(|(label, _)| std::cmp::Reverse(picks.get(*label).copied().unwrap_or(0))); // stable sort
    v
}

/// The operator dropdown inside a Math node whose rectangle is `rect`.
fn op_field(rect: Rect, z: f32) -> Rect {
    Rect::from_min_max(
        Pos2::new(rect.left() + 8.0 * z, rect.top() + (HEADER_H + 2.0) * z),
        Pos2::new(rect.right() - 8.0 * z, rect.top() + (HEADER_H + ROW_H - 2.0) * z),
    )
}

/// Where a node's input rows start, measured from its top: below the header, and below the
/// operator dropdown on Math nodes.
fn body_top(kind: &NodeKind) -> f32 {
    HEADER_H + if matches!(kind, NodeKind::Math { .. }) { ROW_H } else { 0.0 }
}

struct Layout {
    rect: Rect,
    inputs: Vec<Pos2>,
    output: Option<Pos2>,
}

impl Editor {
    /// Draws the comment frames under the nodes. Dragging a frame's title bar moves it and the nodes
    /// inside it, the corner handle resizes it, and its context menu renames or deletes it.
    fn frames(&mut self, ui: &mut egui::Ui, painter: &egui::Painter, origin: Pos2, graph: &mut Graph) {
        let z = self.zoom;
        let mut delete = None;
        for i in 0..graph.frames.len() {
            let f = &graph.frames[i];
            let rect = Rect::from_min_size(self.to_screen(origin, f.pos), Vec2::new(f.size[0], f.size[1]) * z);
            painter.rect(rect, CornerRadius::same((6.0 * z) as u8), Color32::from_rgba_unmultiplied(120, 140, 170, 28), Stroke::new(1.0, Color32::from_rgb(90, 105, 130)), egui::StrokeKind::Inside);
            let header = Rect::from_min_size(rect.min, Vec2::new(rect.width(), 22.0 * z));
            painter.rect_filled(header, CornerRadius::same((6.0 * z) as u8), Color32::from_rgba_unmultiplied(120, 140, 170, 70));
            painter.with_clip_rect(header).text(header.left_center() + Vec2::new(8.0 * z, 0.0), egui::Align2::LEFT_CENTER, &f.title, egui::FontId::proportional(13.0 * z), Color32::from_gray(225));
            let bar = ui.interact(header, ui.scope_id().with(("frame bar", i)), Sense::click_and_drag());
            if bar.dragged_by(egui::PointerButton::Primary) {
                let d = bar.drag_delta() / z;
                // The nodes whose top-left corner is inside the frame go with it.
                let (lo, hi) = (graph.frames[i].pos, [graph.frames[i].pos[0] + graph.frames[i].size[0], graph.frames[i].pos[1] + graph.frames[i].size[1]]);
                for n in graph.nodes.values_mut() {
                    if n.pos[0] >= lo[0] && n.pos[0] <= hi[0] && n.pos[1] >= lo[1] && n.pos[1] <= hi[1] {
                        n.pos = [n.pos[0] + d.x, n.pos[1] + d.y];
                    }
                }
                graph.frames[i].pos = [lo[0] + d.x, lo[1] + d.y];
            }
            let corner = Rect::from_min_size(rect.right_bottom() - Vec2::splat(16.0 * z), Vec2::splat(16.0 * z));
            painter.line_segment([corner.right_bottom() - Vec2::new(12.0 * z, 2.0 * z), corner.right_bottom() - Vec2::new(2.0 * z, 12.0 * z)], Stroke::new(1.5, Color32::from_gray(150)));
            let grip = ui.interact(corner, ui.scope_id().with(("frame grip", i)), Sense::drag());
            if grip.dragged_by(egui::PointerButton::Primary) {
                let d = grip.drag_delta() / z;
                let f = &mut graph.frames[i];
                f.size = [(f.size[0] + d.x).max(80.0), (f.size[1] + d.y).max(50.0)];
            }
            bar.context_menu(|ui| {
                ui.add(egui::TextEdit::singleline(&mut graph.frames[i].title).hint_text("Title"));
                if ui.button("Delete frame").clicked() {
                    delete = Some(i);
                    ui.close();
                }
            });
        }
        if let Some(i) = delete {
            graph.frames.remove(i);
        }
    }

    /// Pans and zooms so the given nodes (all of them when `only` is empty) fill the canvas.
    fn frame(&mut self, canvas_size: Vec2, graph: &Graph, only: &BTreeSet<NodeId>) {
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        for (id, n) in &graph.nodes {
            if only.is_empty() || only.contains(id) {
                let h = HEADER_H + ROW_H * n.kind.input_count().max(2) as f32 + 60.0; // a generous guess at the body
                lo = [lo[0].min(n.pos[0]), lo[1].min(n.pos[1])];
                hi = [hi[0].max(n.pos[0] + NODE_W), hi[1].max(n.pos[1] + h)];
            }
        }
        if lo[0] > hi[0] {
            return;
        }
        let (w, h) = (hi[0] - lo[0] + 80.0, hi[1] - lo[1] + 80.0);
        self.zoom = (canvas_size.x / w).min(canvas_size.y / h).clamp(0.2, 1.25);
        let centre = Vec2::new((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5);
        self.pan = canvas_size / (2.0 * self.zoom) - centre;
    }

    fn to_screen(&self, origin: Pos2, p: [f32; 2]) -> Pos2 {
        origin + (Vec2::new(p[0], p[1]) + self.pan) * self.zoom
    }

    fn to_graph(&self, origin: Pos2, s: Pos2) -> Pos2 {
        ((s - origin) / self.zoom - self.pan).to_pos2()
    }

    fn layout(&self, origin: Pos2, kind: &NodeKind, pos: [f32; 2]) -> Layout {
        let z = self.zoom;
        let n_in = kind.input_count();
        let top_left = self.to_screen(origin, pos);
        let top = body_top(kind);
        let h = (top + ROW_H * n_in.max(1) as f32 + 6.0) * z;
        let rect = Rect::from_min_size(top_left, Vec2::new(NODE_W * z, h));
        let inputs = (0..n_in).map(|i| Pos2::new(rect.left(), rect.top() + (top + ROW_H * (i as f32 + 0.5)) * z)).collect();
        let output = has_output(kind).then(|| Pos2::new(rect.right(), rect.top() + HEADER_H * 0.5 * z));
        Layout { rect, inputs, output }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, graph: &mut Graph, tiling: &BTreeMap<NodeId, Tiling>) {
        // Most edits land after the nodes were drawn this frame, and egui only redraws on input,
        // so redraw once more whenever the graph, selection, preview or menu changed.
        let before = (graph.clone(), self.selected.clone(), self.preview, self.pin_menu.is_some());
        self.show_inner(ui, graph, tiling);
        if before != (graph.clone(), self.selected.clone(), self.preview, self.pin_menu.is_some()) {
            ui.ctx().request_repaint();
        }
    }

    fn show_inner(&mut self, ui: &mut egui::Ui, graph: &mut Graph, tiling: &BTreeMap<NodeId, Tiling>) {
        let (canvas, bg) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if std::mem::take(&mut self.frame_next) {
            self.frame(canvas.size(), graph, &BTreeSet::new());
        }
        let painter = ui.painter_at(canvas);
        painter.rect_filled(canvas, 0.0, Color32::from_gray(24));
        let origin = canvas.min;
        let pointer = ui.input(|i| i.pointer.hover_pos());
        if self.preview.is_some_and(|p| !graph.nodes.contains_key(&p)) {
            self.preview = None;
        }

        // Pan and zoom.
        if bg.dragged_by(egui::PointerButton::Middle) {
            self.pan += bg.drag_delta() / self.zoom;
        }
        if let Some(p) = pointer.filter(|p| canvas.contains(*p)) {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            // Over a Math node's operator dropdown the wheel steps the operator instead, and while
            // any dropdown list or menu is open it belongs to that.
            let over_op = graph.nodes.values().any(|n| {
                matches!(n.kind, NodeKind::Math { .. }) && op_field(self.layout(origin, &n.kind, n.pos).rect, self.zoom).contains(p)
            });
            if scroll != 0.0 && self.pin_menu.is_none() && !over_op && !egui::Popup::is_any_open(ui.ctx()) {
                let before = self.to_graph(origin, p);
                self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.3, 2.5);
                let after = self.to_graph(origin, p);
                self.pan += after - before; // keep the point under the cursor fixed
            }
        }
        self.grid(&painter, canvas);

        // Box select / clear selection / add menu on empty space.
        if bg.drag_started_by(egui::PointerButton::Primary) {
            self.box_start = bg.interact_pointer_pos();
        }
        if bg.clicked() {
            self.selected.clear();
        }
        if bg.secondary_clicked() {
            if let Some(p) = bg.interact_pointer_pos() {
                self.menu_at = self.to_graph(origin, p);
                self.menu_search.clear();
                self.menu_fresh = true;
            }
        }
        bg.context_menu(|ui| {
            ui.label("Add node");
            let search = ui.add(egui::TextEdit::singleline(&mut self.menu_search).hint_text("Search…"));
            if std::mem::take(&mut self.menu_fresh) {
                search.request_focus();
            }
            let enter = search.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            for (i, (label, kind)) in palette_matching(&self.menu_search).into_iter().enumerate() {
                if ui.button(label).clicked() || (enter && i == 0) {
                    let id = graph.add(kind, [self.menu_at.x, self.menu_at.y]);
                    self.selected = BTreeSet::from([id]);
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Comment frame").clicked() {
                graph.frames.push(crate::graph::Frame { title: "Frame".into(), pos: [self.menu_at.x, self.menu_at.y], size: [300.0, 200.0] });
                ui.close();
            }
        });

        self.frames(ui, &painter, origin, graph);

        let layouts: BTreeMap<NodeId, Layout> =
            graph.nodes.iter().map(|(&id, n)| (id, self.layout(origin, &n.kind, n.pos))).collect();
        // Wires under nodes.
        for e in &graph.edges {
            if let (Some(a), Some(b)) = (layouts[&e.from].output, layouts[&e.to].inputs.get(e.port)) {
                painter.add(bezier(a, *b, Stroke::new(2.0 * self.zoom, type_color(graph.output_type(e.from)))));
            }
        }

        // Nodes.
        let z = self.zoom;
        let font = egui::FontId::proportional(13.0 * z);
        let small = egui::FontId::proportional(11.5 * z);
        let mut moved = Vec2::ZERO;
        let mut detach = None;
        let mut toggle_preview = None;
        let mut toggle_cache = None;
        let mut set_value = None;
        let mut set_op = None;
        let roots: Vec<_> = graph.nodes.iter().filter(|(_, n)| !has_output(&n.kind)).map(|(&id, _)| id).chain(self.preview).collect();
        let covered = graph.cache_covered(tiling, &roots);
        for (&id, node) in &graph.nodes {
            let l = &layouts[&id];
            let resp = ui.interact(l.rect, ui.scope_id().with(("node", id)), Sense::click_and_drag());
            if resp.clicked() || resp.drag_started() {
                let additive = ui.input(|i| i.modifiers.shift || i.modifiers.command);
                if additive && resp.clicked() {
                    if !self.selected.remove(&id) {
                        self.selected.insert(id);
                    }
                } else if !self.selected.contains(&id) {
                    self.selected = BTreeSet::from([id]);
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                moved = resp.drag_delta() / z;
            }

            let broken = match tiling.get(&id) {
                Some(Tiling::Broken(why)) => Some(why.clone()),
                _ => None,
            };
            let previewing = self.preview == Some(id);
            let r = CornerRadius::same((5.0 * z) as u8);
            painter.rect_filled(l.rect, r, Color32::from_gray(48));
            let header = Rect::from_min_size(l.rect.min, Vec2::new(l.rect.width(), HEADER_H * z));
            let header_color = if node.muted { Color32::from_gray(60) } else if has_output(&node.kind) { Color32::from_rgb(55, 70, 95) } else { Color32::from_rgb(80, 60, 95) };
            painter.rect_filled(header, CornerRadius { nw: r.nw, ne: r.ne, sw: 0, se: 0 }, header_color);
            let cacheable = matches!(node.kind, NodeKind::Noise(_) | NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Mix { .. } | NodeKind::Curve { .. } | NodeKind::Grid { .. } | NodeKind::Pattern(_));
            // Keep the title clear of the header buttons.
            let title_clip = header.with_max_x(header.right() - (if cacheable { 48.0 } else { 28.0 }) * z);
            painter.with_clip_rect(title_clip).text(header.left_center() + Vec2::new(8.0 * z, 0.0), egui::Align2::LEFT_CENTER, &if node.muted { format!("{} (muted)", title(&node.kind)) } else { title(&node.kind) }, font.clone(), if node.muted { Color32::from_gray(150) } else { Color32::WHITE });
            let is_covered = covered.contains(&id);
            let outline = if let Some(why) = &broken {
                resp.on_hover_text(format!("Does not tile: {why}"));
                Stroke::new(2.0 * z, ORANGE)
            } else if is_covered {
                resp.on_hover_text("Baked into a downstream cache: the preview doesn't evaluate this node");
                Stroke::new(2.0 * z, CACHED)
            } else if self.selected.contains(&id) {
                Stroke::new(1.5 * z, Color32::WHITE)
            } else {
                Stroke::new(1.0, Color32::from_gray(70))
            };
            painter.rect_stroke(l.rect, r, outline, StrokeKind::Outside);
            if (broken.is_some() || is_covered) && self.selected.contains(&id) {
                painter.rect_stroke(l.rect.expand(3.0 * z), r, Stroke::new(1.0, Color32::WHITE), StrokeKind::Outside);
            }
            if previewing {
                painter.rect_stroke(l.rect.expand(5.0 * z), r, Stroke::new(1.5 * z, ACCENT), StrokeKind::Outside);
            }

            let s = summary(&node.kind);
            if !s.is_empty() {
                let at = Pos2::new(l.rect.left() + 8.0 * z, l.rect.top() + (HEADER_H + ROW_H * 0.5) * z);
                painter.text(at, egui::Align2::LEFT_CENTER, s, small.clone(), Color32::from_gray(190));
            }

            // Cache toggle, beside the eye.
            if cacheable {
                let at = Rect::from_center_size(Pos2::new(header.right() - 38.0 * z, header.center().y), Vec2::splat(14.0 * z));
                let cr = ui.interact(at.expand(2.0 * z), ui.scope_id().with(("cache", id)), Sense::click_and_drag());
                let eligible = graph.cache_eligibility(id, tiling);
                let on = node.cache.is_some();
                let tint = match (&eligible, on) {
                    (Err(_), _) => Color32::from_gray(80),
                    (Ok(()), true) => CACHED,
                    (Ok(()), false) if cr.hovered() => Color32::WHITE,
                    (Ok(()), false) => Color32::from_gray(140),
                };
                let icon = if on { egui::include_image!("../assets/icons/sync-saved-locally.svg") } else { egui::include_image!("../assets/icons/sync-saved-locally-outline.svg") };
                egui::Image::new(icon).tint(tint).paint_at(ui, at);
                let tip = match (&eligible, on) {
                    (Err(why), _) => format!("Can't cache: {why}"),
                    (Ok(()), true) => "Cached: the preview reads a baked 3D texture. Click to evaluate live again.".into(),
                    (Ok(()), false) => "Cache in preview: bake everything upstream into a 3D texture.\nExport always uses the full graph.".into(),
                };
                // A stale cache on a node that became ineligible can still be switched off.
                if cr.on_hover_text(tip).clicked() && (eligible.is_ok() || on) {
                    toggle_cache = Some(id);
                }
            }

            // Preview eye (registered after the body so it wins the hit test). Outputs are the normal view.
            if has_output(&node.kind) {
                let eye = Rect::from_center_size(Pos2::new(header.right() - 18.0 * z, header.center().y), Vec2::splat(14.0 * z));
                // Also senses drags so a drag that starts on the eye doesn't move the node.
                let er = ui.interact(eye.expand(2.0 * z), ui.scope_id().with(("eye", id)), Sense::click_and_drag());
                let tint = if previewing { ACCENT } else if er.hovered() { Color32::WHITE } else { Color32::from_gray(140) };
                egui::Image::new(egui::include_image!("../assets/icons/visibility.svg")).tint(tint).paint_at(ui, eye);
                let tip = if previewing { "Back to the Density output" } else { "Preview this node in the 3D and slice views" };
                if er.on_hover_text(tip).clicked() {
                    toggle_preview = Some(id);
                }
            }

            // The Math operator as a dropdown on the node, like Blender, so switching it doesn't
            // mean selecting the node and going to the details panel.
            if let NodeKind::Math { op, .. } = node.kind {
                let field = op_field(l.rect, z);
                let mut cui = ui.new_child(egui::UiBuilder::new().max_rect(field).id_salt(("op", id)));
                cui.style_mut().override_font_id = Some(small.clone());
                cui.spacing_mut().interact_size.y = field.height();
                let mut chosen = op;
                let combo = egui::ComboBox::from_id_salt(("op combo", id)).selected_text(op.label()).width(field.width()).show_ui(&mut cui, |ui| {
                    for o in MathOp::ALL {
                        ui.selectable_value(&mut chosen, o, o.label());
                    }
                });
                combo.response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::ComboBox, true, "Math operator"));
                // The wheel over the closed dropdown steps through the operators (up = previous).
                if ui.rect_contains_pointer(field) && !egui::Popup::is_any_open(ui.ctx()) {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 && !self.op_scroll_stepped {
                        let at = MathOp::ALL.iter().position(|o| *o == op).unwrap_or(0);
                        let to = if scroll > 0.0 { at.saturating_sub(1) } else { (at + 1).min(MathOp::ALL.len() - 1) };
                        chosen = MathOp::ALL[to];
                        self.op_scroll_stepped = true;
                    } else if scroll == 0.0 {
                        self.op_scroll_stepped = false;
                    }
                }
                if chosen != op {
                    set_op = Some((id, chosen));
                }
            }

            // Ports.
            for (port, &p) in l.inputs.iter().enumerate() {
                let connected = graph.input(id, port);
                let color = type_color(graph.input_type(id, port));
                painter.circle(p, PORT_R * z, if connected.is_some() { color } else { color.gamma_multiply(0.3) }, Stroke::new(1.0, color));
                painter.text(p + Vec2::new(10.0 * z, 0.0), egui::Align2::LEFT_CENTER, input_label(&node.kind, port), small.clone(), Color32::from_gray(210));
                // Fallback values (Math, Remap) sit on the node, greyed out while the pin is connected.
                if let Some(mut v) = node.kind.const_input(port) {
                    let field = Rect::from_min_max(Pos2::new(l.rect.left() + 58.0 * z, p.y - ROW_H * 0.4 * z), Pos2::new(l.rect.right() - 8.0 * z, p.y + ROW_H * 0.4 * z));
                    let mut fui = ui.new_child(egui::UiBuilder::new().max_rect(field).id_salt(("math value", id, port)));
                    fui.style_mut().override_font_id = Some(small.clone());
                    let r = fui.add_enabled(connected.is_none(), |ui: &mut egui::Ui| ui.add_sized(field.size(), egui::DragValue::new(&mut v).speed(0.01)));
                    let label = input_label(&node.kind, port);
                    r.widget_info(|| egui::WidgetInfo::labeled(egui::Role::SpinButton, connected.is_none(), label));
                    if r.changed() {
                        set_value = Some((id, port, v));
                    }
                }
                let pr =ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("in", id, port)), Sense::drag());
                if pr.drag_started() {
                    self.wire = Some(match connected {
                        // Pull the existing link off its input.
                        Some(src) => {
                            detach = Some((id, port));
                            Wire::FromOutput(src)
                        }
                        None => Wire::FromInput(id, port),
                    });
                }
            }
            if let Some(p) = l.output {
                painter.circle(p, PORT_R * z, type_color(graph.output_type(id)), Stroke::new(1.0, Color32::from_gray(220)));
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("out", id)), Sense::drag());
                if pr.drag_started() {
                    self.wire = Some(Wire::FromOutput(id));
                }
            }
        }
        if let Some(id) = toggle_preview {
            self.preview = if self.preview == Some(id) { None } else { Some(id) };
        }
        if let Some(n) = toggle_cache.and_then(|id| graph.nodes.get_mut(&id)) {
            n.cache = if n.cache.is_some() { None } else { Some(0) };
        }
        if let Some((id, op)) = set_op {
            if let Some(NodeKind::Math { op: current, .. }) = graph.nodes.get_mut(&id).map(|n| &mut n.kind) {
                *current = op;
            }
            graph.prune_edges(); // smooth min/max have a third input the others don't
        }
        if let Some((id, port, v)) = set_value {
            if let Some(n) = graph.nodes.get_mut(&id) {
                n.kind.set_const_input(port, v);
            }
        }
        if let Some((to, port)) = detach {
            graph.disconnect(to, port);
        }
        // Optional snapping: when a drag ends, selected nodes settle on a 20-unit grid.
        if self.snap && ui.input(|i| i.pointer.primary_released()) {
            for id in &self.selected {
                if let Some(n) = graph.nodes.get_mut(id) {
                    n.pos = [(n.pos[0] / 20.0).round() * 20.0, (n.pos[1] / 20.0).round() * 20.0];
                }
            }
        }
        if moved != Vec2::ZERO {
            self.node_dragged = true;
            for id in &self.selected {
                if let Some(n) = graph.nodes.get_mut(id) {
                    n.pos[0] += moved.x;
                    n.pos[1] += moved.y;
                }
            }
        }

        // Wire being dragged: connect on release over a matching port, or offer new nodes on empty space.
        let released = ui.input(|i| i.pointer.primary_released());
        // A lone, unconnected node dropped on a wire is spliced into it.
        if released && self.node_dragged {
            if let [id] = self.selected.iter().copied().collect::<Vec<_>>()[..] {
                if let Some(rect) = layouts.get(&id).map(|l| l.rect) {
                    let n_in = graph.nodes.get(&id).map_or(0, |n| n.kind.input_count());
                    let free = !graph.edges.iter().any(|e| e.from == id || e.to == id);
                    let target = graph.edges.iter().find(|e| {
                        e.from != id && e.to != id && matches!((layouts.get(&e.from).and_then(|l| l.output), layouts.get(&e.to).and_then(|l| l.inputs.get(e.port))), (Some(a), Some(&b)) if wire_passes_through(a, b, rect.shrink(4.0 * z)))
                    }).copied();
                    if let (true, true, Some(edge)) = (free, n_in > 0 && graph.nodes.get(&id).is_some_and(|n| has_output(&n.kind)), target) {
                        graph.disconnect(edge.to, edge.port);
                        graph.connect(edge.from, id, 0);
                        graph.connect(id, edge.to, edge.port);
                    }
                }
            }
        }
        if released {
            self.node_dragged = false;
        }
        if let (Some(wire), Some(p)) = (self.wire, ui.input(|i| i.pointer.interact_pos())) {
            let drag_stroke = Stroke::new(
                2.0 * z,
                type_color(match wire {
                    Wire::FromOutput(from) => graph.output_type(from),
                    Wire::FromInput(to, port) => graph.input_type(to, port),
                }),
            );
            match wire {
                Wire::FromOutput(from) => {
                    if let Some(a) = layouts.get(&from).and_then(|l| l.output) {
                        painter.add(bezier(a, p, drag_stroke));
                    }
                }
                Wire::FromInput(to, port) => {
                    if let Some(b) = layouts.get(&to).and_then(|l| l.inputs.get(port)) {
                        painter.add(bezier(p, *b, drag_stroke));
                    }
                }
            }
            if released {
                let near = |q: &Pos2| q.distance(p) < PORT_R * 3.0 * z;
                match wire {
                    Wire::FromOutput(from) => {
                        let hit = layouts.iter().find_map(|(&id, l)| l.inputs.iter().position(near).map(|port| (id, port)));
                        match hit {
                            // Refuse links that would make a cycle.
                            Some((to, port)) if !graph.depends_on(from, to) => graph.connect(from, to, port),
                            Some(_) => {}
                            None => self.open_pin_menu(origin, canvas, &layouts, p, wire),
                        }
                    }
                    Wire::FromInput(to, port) => {
                        let hit = layouts.iter().find(|(_, l)| l.output.as_ref().is_some_and(near)).map(|(&id, _)| id);
                        match hit {
                            Some(from) if !graph.depends_on(from, to) => graph.connect(from, to, port),
                            Some(_) => {}
                            None => self.open_pin_menu(origin, canvas, &layouts, p, wire),
                        }
                    }
                }
                self.wire = None;
            }
        }
        self.pin_menu_ui(ui, graph);

        // Box selection.
        if let Some(start) = self.box_start {
            if let Some(p) = ui.input(|i| i.pointer.interact_pos()) {
                let r = Rect::from_two_pos(start, p);
                painter.rect(r, 0.0, Color32::from_rgba_unmultiplied(120, 160, 255, 25), Stroke::new(1.0, Color32::from_rgb(120, 160, 255)), StrokeKind::Inside);
                if released {
                    let additive = ui.input(|i| i.modifiers.shift || i.modifiers.command);
                    if !additive {
                        self.selected.clear();
                    }
                    self.selected.extend(layouts.iter().filter(|(_, l)| r.intersects(l.rect)).map(|(&id, _)| id));
                }
            }
            if released {
                self.box_start = None;
            }
        }

        // Keyboard: only while the canvas is hovered and no text field has focus.
        let over_canvas = pointer.is_some_and(|p| canvas.contains(p));
        let typing = ui.memory(|m| m.focused().is_some());
        if over_canvas && !typing {
            if ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
                for id in std::mem::take(&mut self.selected) {
                    graph.remove(id);
                }
            }
            if ui.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy))) {
                if let Some(t) = clipboard_text(graph, &self.selected) {
                    ui.ctx().copy_text(t);
                }
            }
            let pasted = ui.input(|i| i.events.iter().find_map(|e| if let egui::Event::Paste(t) = e { Some(t.clone()) } else { None }));
            if let (Some(t), Some(p)) = (pasted, pointer) {
                let at = self.to_graph(origin, p);
                let new = paste_text(graph, &t, [at.x, at.y]);
                if !new.is_empty() {
                    self.selected = new;
                }
            }
            if ui.input(|i| i.key_pressed(egui::Key::M) && !i.modifiers.command) {
                for id in &self.selected {
                    if let Some(n) = graph.nodes.get_mut(id).filter(|n| !matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal)) {
                        n.muted = !n.muted;
                    }
                }
            }
            if ui.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.command) {
                let only = self.selected.clone();
                self.frame(canvas.size(), graph, &only);
            }
            if ui.input(|i| i.key_pressed(egui::Key::Home)) {
                self.frame(canvas.size(), graph, &BTreeSet::new());
            }
            if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::D)) {
                if let Some(p) = pointer {
                    let at = self.to_graph(origin, p);
                    self.selected = duplicate(graph, &self.selected, [at.x, at.y]);
                }
            }
        }
    }

    fn open_pin_menu(&mut self, origin: Pos2, canvas: Rect, layouts: &BTreeMap<NodeId, Layout>, p: Pos2, wire: Wire) {
        // Only on empty canvas, not when the wire is dropped on a node body.
        if canvas.contains(p) && !layouts.values().any(|l| l.rect.contains(p)) {
            self.pin_menu = Some(PinMenu { screen: p, graph: self.to_graph(origin, p), wire });
        }
    }

    /// The wire-drop node menu: picking a node creates it at the drop point, wired up.
    fn pin_menu_ui(&mut self, ui: &mut egui::Ui, graph: &mut Graph) {
        let Some(menu) = &self.pin_menu else { return };
        let mut picked = None;
        let area = egui::Area::new(ui.scope_id().with("pin menu")).fixed_pos(menu.screen).order(egui::Order::Foreground).show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                ui.weak("New node");
                for (label, kind) in wire_choices(menu.wire, &self.picks) {
                    if ui.button(label).clicked() {
                        picked = Some((label, kind));
                    }
                }
            });
        });
        // Test the rect directly: the area's own contains_pointer() was false for presses on its
        // buttons in the real app, so the menu closed on mouse-down before the click landed.
        let clicked_outside = ui.input(|i| i.pointer.any_pressed() && !i.pointer.interact_pos().is_some_and(|p| area.response.rect.contains(p)));
        if let Some((label, kind)) = picked {
            let (g, wire) = (menu.graph, menu.wire);
            // Place the new node so its connecting port sits at the drop point.
            let pos = match wire {
                Wire::FromOutput(_) => [g.x, g.y - body_top(&kind) - ROW_H * 0.5],
                Wire::FromInput(..) => [g.x - NODE_W, g.y - HEADER_H * 0.5],
            };
            let id = graph.add(kind, pos);
            match wire {
                Wire::FromOutput(from) => graph.connect(from, id, 0),
                Wire::FromInput(to, port) => graph.connect(id, to, port),
            }
            *self.picks.entry(label.to_string()).or_default() += 1;
            self.selected = BTreeSet::from([id]);
            self.pin_menu = None;
        } else if clicked_outside || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.pin_menu = None;
        }
    }

    fn grid(&self, painter: &egui::Painter, canvas: Rect) {
        let step = 40.0 * self.zoom;
        if step < 8.0 {
            return;
        }
        let off = (self.pan * self.zoom).to_pos2();
        let x0 = canvas.left() + off.x.rem_euclid(step);
        let y0 = canvas.top() + off.y.rem_euclid(step);
        let mut x = x0;
        while x < canvas.right() {
            let mut y = y0;
            while y < canvas.bottom() {
                painter.circle_filled(Pos2::new(x, y), 1.0, Color32::from_gray(48));
                y += step;
            }
            x += step;
        }
    }
}

/// Copies `nodes` (and the links between them) so their centre lands on `at`. Returns the copies.
pub fn duplicate(graph: &mut Graph, nodes: &BTreeSet<NodeId>, at: [f32; 2]) -> BTreeSet<NodeId> {
    let src = graph.clone();
    merge(graph, &src, nodes, at)
}

/// Adds copies of `nodes` of `src` (and the links between them) to `graph`, centred on `at`.
fn merge(graph: &mut Graph, src: &Graph, nodes: &BTreeSet<NodeId>, at: [f32; 2]) -> BTreeSet<NodeId> {
    let picked: Vec<_> = nodes.iter().filter_map(|id| src.nodes.get(id).map(|n| (*id, n.clone()))).collect();
    if picked.is_empty() {
        return BTreeSet::new();
    }
    // Centre of the copied nodes, measured at their (approximate) middles.
    let mid = |n: &crate::graph::Node| [n.pos[0] + NODE_W * 0.5, n.pos[1] + HEADER_H];
    let c = picked.iter().fold([0.0, 0.0], |acc, (_, n)| [acc[0] + mid(n)[0], acc[1] + mid(n)[1]]);
    let c = [c[0] / picked.len() as f32, c[1] / picked.len() as f32];
    let mut map = BTreeMap::new();
    for (old, n) in &picked {
        let id = graph.add(n.kind.clone(), [n.pos[0] + at[0] - c[0], n.pos[1] + at[1] - c[1]]);
        map.insert(*old, id);
    }
    let internal: Vec<_> = src.edges.iter().filter_map(|e| Some((*map.get(&e.from)?, *map.get(&e.to)?, e.port))).collect();
    for (from, to, port) in internal {
        graph.connect(from, to, port);
    }
    map.into_values().collect()
}

/// The selection as text for the clipboard: a small graph of the selected nodes and the links between
/// them (the output nodes are left out). None when nothing copyable is selected.
pub fn clipboard_text(graph: &Graph, nodes: &BTreeSet<NodeId>) -> Option<String> {
    let mut sub = graph.clone();
    let keep: Vec<_> = nodes.iter().copied().filter(|id| graph.nodes.get(id).is_some_and(|n| !matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal))).collect();
    if keep.is_empty() {
        return None;
    }
    sub.nodes.retain(|id, _| keep.contains(id));
    sub.edges.retain(|e| keep.contains(&e.from) && keep.contains(&e.to));
    Some(sub.to_json())
}

/// Pastes clipboard text made by `clipboard_text` (from this or another Noiser) centred on `at`.
/// Returns the new nodes; empty when the text is not a graph.
pub fn paste_text(graph: &mut Graph, text: &str, at: [f32; 2]) -> BTreeSet<NodeId> {
    match Graph::from_json(text) {
        Ok(src) => {
            let all: BTreeSet<NodeId> = src.nodes.iter().filter(|(_, n)| !matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal)).map(|(id, _)| *id).collect();
            merge(graph, &src, &all, at)
        }
        Err(_) => BTreeSet::new(),
    }
}

/// The add-menu entries whose label contains `query` (case-insensitive); all of them when it is empty.
fn palette_matching(query: &str) -> Vec<(&'static str, NodeKind)> {
    let q = query.trim().to_lowercase();
    palette().into_iter().filter(|(label, _)| label.to_lowercase().contains(&q)).collect()
}

/// Whether the wire from `a` to `b` (drawn by `bezier`) passes through `rect`.
fn wire_passes_through(a: Pos2, b: Pos2, rect: Rect) -> bool {
    let dx = ((b.x - a.x).abs() * 0.5).max(30.0);
    let (c1, c2) = (a + Vec2::new(dx, 0.0), b - Vec2::new(dx, 0.0));
    (0..=32).any(|i| {
        let t = i as f32 / 32.0;
        let u = 1.0 - t;
        let p = a.to_vec2() * (u * u * u) + c1.to_vec2() * (3.0 * u * u * t) + c2.to_vec2() * (3.0 * u * t * t) + b.to_vec2() * (t * t * t);
        rect.contains(p.to_pos2())
    })
}

fn bezier(a: Pos2, b: Pos2, stroke: Stroke) -> CubicBezierShape {
    let dx = ((b.x - a.x).abs() * 0.5).max(30.0);
    CubicBezierShape::from_points_stroke([a, a + Vec2::new(dx, 0.0), b - Vec2::new(dx, 0.0), b], false, Color32::TRANSPARENT, stroke)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_menu_search_filters_by_label() {
        assert_eq!(palette_matching("").len(), palette().len());
        let hits = palette_matching("  PERL ");
        assert!(!hits.is_empty() && hits.iter().all(|(l, _)| l.to_lowercase().contains("perl")));
        assert!(palette_matching("zzzz-no-such-node").is_empty());
    }

    #[test]
    fn duplicate_keeps_internal_links_and_centres_on_cursor() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity, 3 OutNormal
        let copies = duplicate(&mut g, &BTreeSet::from([0, 1]), [1000.0, 500.0]);
        assert_eq!(copies.len(), 2);
        let ids: Vec<_> = copies.iter().copied().collect();
        // The Position → Noise link is copied; the link to the output (outside the selection) is not.
        assert!(g.edges.iter().any(|e| e.from == ids[0] && e.to == ids[1]));
        assert!(!g.edges.iter().any(|e| e.from == ids[1]));
        let c = ids.iter().fold(0.0, |acc, id| acc + g.nodes[id].pos[0] + NODE_W * 0.5) / 2.0;
        assert!((c - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn framing_fits_the_nodes_in_the_canvas() {
        let g = Graph::default_graph();
        let mut ed = Editor::default();
        let size = Vec2::new(800.0, 300.0);
        ed.frame(size, &g, &BTreeSet::new());
        for n in g.nodes.values() {
            let p = ed.to_screen(Pos2::ZERO, n.pos);
            let q = ed.to_screen(Pos2::ZERO, [n.pos[0] + NODE_W, n.pos[1] + HEADER_H]);
            assert!(p.x >= 0.0 && p.y >= 0.0 && q.x <= size.x && q.y <= size.y, "node at {:?} is outside the canvas", n.pos);
        }
    }

    #[test]
    fn wires_are_found_by_a_rect_on_them() {
        let (a, b) = (Pos2::new(0.0, 0.0), Pos2::new(300.0, 100.0));
        assert!(wire_passes_through(a, b, Rect::from_center_size(Pos2::new(150.0, 50.0), Vec2::splat(20.0))), "the middle of the wire");
        assert!(!wire_passes_through(a, b, Rect::from_center_size(Pos2::new(150.0, 400.0), Vec2::splat(20.0))));
    }

    #[test]
    fn clipboard_round_trip_pastes_nodes_and_links() {
        let g = Graph::default_graph(); // 0 Position -> 1 Noise -> 2 OutDensity, 3 OutNormal
        let text = clipboard_text(&g, &BTreeSet::from([0, 1, 2])).unwrap();
        let mut other = Graph::default();
        let new = paste_text(&mut other, &text, [100.0, 100.0]);
        assert_eq!(new.len(), 2, "the output node is not copied");
        let ids: Vec<_> = new.iter().copied().collect();
        assert!(other.edges.iter().any(|e| e.from == ids[0] && e.to == ids[1]));
        assert!(paste_text(&mut other, "not a graph", [0.0; 2]).is_empty());
        assert!(clipboard_text(&g, &BTreeSet::from([2])).is_none());
    }

    /// Drag a wire from Position's output onto empty space, then pick Math from the menu.
    #[test]
    fn wire_drop_menu_adds_node() {
        use egui_kittest::{Harness, kittest::{NodeT, Queryable}};
        // Laid out like the app: the editor in a 330 px bottom panel of a 1600×900 window.
        let mut h = Harness::builder().with_size(Vec2::new(1280.0, 800.0)).with_pixels_per_point(1.5).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx()); // no-op after the first frame
                egui::Panel::top("menu").show(ui, |ui| ui.label("menu"));
                egui::Panel::right("props").default_size(320.0).show(ui, |ui| ui.label("props"));
                egui::Panel::bottom("graph").resizable(true).default_size(330.0).show(ui, |ui| {
                    ed.debug_origin = ui.available_rect_before_wrap().min;
                    ed.show(ui, g, &BTreeMap::new())
                });
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
                    let r = ui.available_rect_before_wrap();
                    ui.interact(r, ui.scope_id().with("view3d"), Sense::drag());
                });
            },
            (Editor::default(), Graph::default_graph()),
        );
        h.run();
        let press = |h: &mut Harness<(Editor, Graph)>, pos: Pos2, pressed: bool| {
            h.event(egui::Event::PointerMoved(pos));
            h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() });
            h.run();
        };
        let ed = &h.state().0;
        let out = ed.to_screen(ed.debug_origin, [NODE_W, HEADER_H * 0.5]); // Position's output port
        h.hover_at(out);
        h.run();
        press(&mut h, out, true);
        for i in 1..=10 {
            h.hover_at(out + Vec2::new(10.0 * i as f32, 15.0 * i as f32));
            h.run();
        }
        let drop = out + Vec2::new(100.0, 150.0);
        press(&mut h, drop, false);
        assert!(h.state().0.pin_menu.is_some(), "menu should open");
        let at = h.get_by_label("Math").rect().center();
        for i in 1..=5 {
            h.hover_at(drop + (at - drop) * i as f32 / 5.0);
            h.run();
        }
        press(&mut h, at, true);
        press(&mut h, at, false);
        h.run();
        // A is wired from Position, so only B's on-node field is editable.
        assert!(h.get_by_label("A").accesskit_node().is_disabled());
        assert!(!h.get_by_label("B").accesskit_node().is_disabled());
        let g = &h.state().1;
        assert_eq!(g.nodes.len(), 5, "Math node should be added");
        assert!(g.edges.iter().any(|e| e.from == 0 && matches!(g.nodes[&e.to].kind, NodeKind::Math { .. })));
    }

    /// The operator dropdown on a Math node switches the mode without the details panel, and a
    /// link into the blend input is dropped when the new operator has no such input.
    #[test]
    fn math_dropdown_switches_operator_on_the_node() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut g = Graph::default();
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [20.0, 20.0]);
        let f = g.add(NodeKind::Float(1.0), [20.0, 200.0]);
        let mut h = Harness::builder().with_size(Vec2::new(800.0, 500.0)).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx());
                ed.show(ui, g, &BTreeMap::new())
            },
            (Editor::default(), g),
        );
        h.run();
        assert_eq!(h.state().1.nodes[&m].kind.input_count(), 2);
        h.get_by_label("Math operator").click();
        h.run();
        h.get_by_label("Smooth minimum").click();
        h.run();
        assert!(matches!(h.state().1.nodes[&m].kind, NodeKind::Math { op: MathOp::SmoothMin, .. }));
        assert_eq!(h.state().1.nodes[&m].kind.input_count(), 3);
        h.state_mut().1.connect(f, m, 2);
        h.run();
        h.get_by_label("Math operator").click();
        h.run();
        h.get_by_label("Add").click();
        h.run();
        let g = &h.state().1;
        assert!(matches!(g.nodes[&m].kind, NodeKind::Math { op: MathOp::Add, .. }));
        assert!(g.input(m, 2).is_none(), "the link into the blend input should be dropped");
    }

    /// The wheel over a Math node's dropdown steps the operator and leaves the graph's zoom alone;
    /// over empty canvas it still zooms.
    #[test]
    fn wheel_over_the_operator_dropdown_steps_it_instead_of_zooming() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut g = Graph::default();
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [20.0, 20.0]);
        let mut h = Harness::builder().with_size(Vec2::new(800.0, 500.0)).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx());
                ed.show(ui, g, &BTreeMap::new())
            },
            (Editor::default(), g),
        );
        h.run();
        let wheel = |h: &mut Harness<(Editor, Graph)>, at: Pos2, lines: f32| {
            h.hover_at(at);
            h.run();
            h.event(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: Vec2::new(0.0, lines),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: Default::default(),
            });
            h.run();
        };
        let on_dropdown = h.get_by_label("Math operator").rect().center();
        let op = |h: &Harness<(Editor, Graph)>| match h.state().1.nodes[&m].kind {
            NodeKind::Math { op, .. } => op,
            _ => unreachable!(),
        };
        wheel(&mut h, on_dropdown, -1.0); // scroll down: next operator
        assert_eq!(op(&h), MathOp::Subtract);
        assert_eq!(h.state().0.zoom, 1.0, "the wheel over the dropdown must not zoom the graph");
        wheel(&mut h, on_dropdown, 1.0); // scroll up: previous operator
        assert_eq!(op(&h), MathOp::Add);
        wheel(&mut h, on_dropdown, 1.0); // already first: stays
        assert_eq!(op(&h), MathOp::Add);
        wheel(&mut h, Pos2::new(600.0, 400.0), 1.0); // empty canvas
        assert!(h.state().0.zoom > 1.0, "the wheel over the canvas still zooms");
        assert_eq!(op(&h), MathOp::Add);
    }

    #[test]
    fn wire_choices_follow_direction_and_picks() {
        let picks = BTreeMap::from([("Worley".to_string(), 3), ("Math".to_string(), 1)]);
        let out = wire_choices(Wire::FromOutput(0), &picks);
        assert_eq!(out[0].0, "Worley");
        assert_eq!(out[1].0, "Math");
        assert!(out.iter().all(|(_, k)| k.input_count() > 0));
        let inp = wire_choices(Wire::FromInput(0, 0), &picks);
        assert!(inp.iter().all(|(_, k)| has_output(k)));
        assert!(inp.iter().any(|(l, _)| *l == "Position"));
    }
}
