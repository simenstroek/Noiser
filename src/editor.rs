//! Node graph canvas: pan/zoom, drag nodes, wire ports, box select, add/delete/duplicate, preview.
//!
//! Controls: middle-drag pans, scroll zooms, left-drag on empty space box-selects, right-click
//! adds a node, Delete removes the selection, Ctrl+D duplicates it at the cursor.
//! Drag from an output to an input (or the other way) to connect; drag a connected input away to
//! disconnect it; drop a wire on empty space to pick a new node for it. The eye in a node's
//! header previews that node in the 3D and slice views.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, NoiseParams, Tiling};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, epaint::CubicBezierShape};
use std::collections::{BTreeMap, BTreeSet};

pub const ORANGE: Color32 = Color32::from_rgb(255, 150, 40);
const ACCENT: Color32 = Color32::from_rgb(140, 190, 255);

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
    /// Node shown in the 3D and slice views instead of the Density output.
    pub preview: Option<NodeId>,
    /// How often each node was picked from the wire-drop menu; most used are listed first.
    pub picks: BTreeMap<String, u32>,
    wire: Option<Wire>,
    pin_menu: Option<PinMenu>,
    box_start: Option<Pos2>,
    /// Graph-space point where the add menu was opened.
    menu_at: Pos2,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            pan: Vec2::new(40.0, 40.0),
            zoom: 1.0,
            selected: BTreeSet::new(),
            preview: None,
            picks: BTreeMap::new(),
            wire: None,
            pin_menu: None,
            box_start: None,
            menu_at: Pos2::ZERO,
        }
    }
}

pub fn title(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => "Value".into(),
        NodeKind::Math { op, .. } => format!("Math · {op:?}"),
        NodeKind::Position => "Position".into(),
        NodeKind::Normal { .. } => "Normal".into(),
        NodeKind::Noise(n) => n.kind.name().into(),
        NodeKind::OutDensity => "Density Output".into(),
        NodeKind::OutNormal => "Normal Output".into(),
    }
}

fn input_label(kind: &NodeKind, port: usize) -> &'static str {
    match (kind, port) {
        (NodeKind::Math { .. }, 0) => "A",
        (NodeKind::Math { .. }, _) => "B",
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
            ("Math", NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }),
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

struct Layout {
    rect: Rect,
    inputs: Vec<Pos2>,
    output: Option<Pos2>,
}

impl Editor {
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
        let h = (HEADER_H + ROW_H * n_in.max(1) as f32 + 6.0) * z;
        let rect = Rect::from_min_size(top_left, Vec2::new(NODE_W * z, h));
        let inputs = (0..n_in).map(|i| Pos2::new(rect.left(), rect.top() + (HEADER_H + ROW_H * (i as f32 + 0.5)) * z)).collect();
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
            if scroll != 0.0 && self.pin_menu.is_none() {
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
            }
        }
        bg.context_menu(|ui| {
            ui.label("Add node");
            for (label, kind) in palette() {
                if ui.button(label).clicked() {
                    let id = graph.add(kind, [self.menu_at.x, self.menu_at.y]);
                    self.selected = BTreeSet::from([id]);
                    ui.close();
                }
            }
        });

        let layouts: BTreeMap<NodeId, Layout> =
            graph.nodes.iter().map(|(&id, n)| (id, self.layout(origin, &n.kind, n.pos))).collect();
        // Wires under nodes.
        let wire_stroke = Stroke::new(2.0 * self.zoom, Color32::from_gray(170));
        for e in &graph.edges {
            if let (Some(a), Some(b)) = (layouts[&e.from].output, layouts[&e.to].inputs.get(e.port)) {
                painter.add(bezier(a, *b, wire_stroke));
            }
        }

        // Nodes.
        let z = self.zoom;
        let font = egui::FontId::proportional(13.0 * z);
        let small = egui::FontId::proportional(11.5 * z);
        let mut moved = Vec2::ZERO;
        let mut detach = None;
        let mut toggle_preview = None;
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
            let header_color = if has_output(&node.kind) { Color32::from_rgb(55, 70, 95) } else { Color32::from_rgb(80, 60, 95) };
            painter.rect_filled(header, CornerRadius { nw: r.nw, ne: r.ne, sw: 0, se: 0 }, header_color);
            painter.text(header.left_center() + Vec2::new(8.0 * z, 0.0), egui::Align2::LEFT_CENTER, title(&node.kind), font.clone(), Color32::WHITE);
            let outline = if let Some(why) = &broken {
                resp.on_hover_text(format!("Does not tile: {why}"));
                Stroke::new(2.0 * z, ORANGE)
            } else if self.selected.contains(&id) {
                Stroke::new(1.5 * z, Color32::WHITE)
            } else {
                Stroke::new(1.0, Color32::from_gray(70))
            };
            painter.rect_stroke(l.rect, r, outline, StrokeKind::Outside);
            if broken.is_some() && self.selected.contains(&id) {
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

            // Ports.
            for (port, &p) in l.inputs.iter().enumerate() {
                let connected = graph.input(id, port);
                painter.circle(p, PORT_R * z, if connected.is_some() { Color32::from_gray(200) } else { Color32::from_gray(90) }, Stroke::new(1.0, Color32::from_gray(200)));
                painter.text(p + Vec2::new(10.0 * z, 0.0), egui::Align2::LEFT_CENTER, input_label(&node.kind, port), small.clone(), Color32::from_gray(210));
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("in", id, port)), Sense::drag());
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
                painter.circle(p, PORT_R * z, ACCENT, Stroke::new(1.0, Color32::from_gray(220)));
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("out", id)), Sense::drag());
                if pr.drag_started() {
                    self.wire = Some(Wire::FromOutput(id));
                }
            }
        }
        if let Some(id) = toggle_preview {
            self.preview = if self.preview == Some(id) { None } else { Some(id) };
        }
        if let Some((to, port)) = detach {
            graph.disconnect(to, port);
        }
        if moved != Vec2::ZERO {
            for id in &self.selected {
                if let Some(n) = graph.nodes.get_mut(id) {
                    n.pos[0] += moved.x;
                    n.pos[1] += moved.y;
                }
            }
        }

        // Wire being dragged: connect on release over a matching port, or offer new nodes on empty space.
        let released = ui.input(|i| i.pointer.primary_released());
        if let (Some(wire), Some(p)) = (self.wire, ui.input(|i| i.pointer.interact_pos())) {
            let drag_stroke = Stroke::new(2.0 * z, ACCENT);
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
        let clicked_outside = ui.input(|i| i.pointer.any_pressed()) && !area.response.contains_pointer();
        if let Some((label, kind)) = picked {
            let (g, wire) = (menu.graph, menu.wire);
            // Place the new node so its connecting port sits at the drop point.
            let pos = match wire {
                Wire::FromOutput(_) => [g.x, g.y - HEADER_H - ROW_H * 0.5],
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
    let src: Vec<_> = nodes.iter().filter_map(|id| graph.nodes.get(id).map(|n| (*id, n.clone()))).collect();
    if src.is_empty() {
        return BTreeSet::new();
    }
    // Centre of the copied nodes, measured at their (approximate) middles.
    let mid = |n: &crate::graph::Node| [n.pos[0] + NODE_W * 0.5, n.pos[1] + HEADER_H];
    let c = src.iter().fold([0.0, 0.0], |acc, (_, n)| [acc[0] + mid(n)[0], acc[1] + mid(n)[1]]);
    let c = [c[0] / src.len() as f32, c[1] / src.len() as f32];
    let mut map = BTreeMap::new();
    for (old, n) in &src {
        let id = graph.add(n.kind.clone(), [n.pos[0] + at[0] - c[0], n.pos[1] + at[1] - c[1]]);
        map.insert(*old, id);
    }
    let internal: Vec<_> = graph.edges.iter().filter_map(|e| Some((*map.get(&e.from)?, *map.get(&e.to)?, e.port))).collect();
    for (from, to, port) in internal {
        graph.connect(from, to, port);
    }
    map.into_values().collect()
}

fn bezier(a: Pos2, b: Pos2, stroke: Stroke) -> CubicBezierShape {
    let dx = ((b.x - a.x).abs() * 0.5).max(30.0);
    CubicBezierShape::from_points_stroke([a, a + Vec2::new(dx, 0.0), b - Vec2::new(dx, 0.0), b], false, Color32::TRANSPARENT, stroke)
}

#[cfg(test)]
mod tests {
    use super::*;

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
