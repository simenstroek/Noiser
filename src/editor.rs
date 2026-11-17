//! Node graph canvas: pan/zoom, drag nodes, wire ports, box select, add/delete.
//!
//! Controls: middle-drag pans, scroll zooms, left-drag on empty space box-selects,
//! right-click adds a node, Delete removes the selection. Drag from an output to an input
//! to connect; drag a connected input away to disconnect it.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseParams, Tiling};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, epaint::CubicBezierShape};
use std::collections::{BTreeMap, BTreeSet};

pub const ORANGE: Color32 = Color32::from_rgb(255, 150, 40);

const NODE_W: f32 = 140.0;
const HEADER_H: f32 = 22.0;
const ROW_H: f32 = 20.0;
const PORT_R: f32 = 5.0;

pub struct Editor {
    pan: Vec2,
    zoom: f32,
    pub selected: BTreeSet<NodeId>,
    /// Output a wire is being dragged from.
    wire_from: Option<NodeId>,
    box_start: Option<Pos2>,
    /// Graph-space point where the add menu was opened.
    menu_at: Pos2,
}

impl Default for Editor {
    fn default() -> Self {
        Self { pan: Vec2::new(40.0, 40.0), zoom: 1.0, selected: BTreeSet::new(), wire_from: None, box_start: None, menu_at: Pos2::ZERO }
    }
}

pub fn title(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => "Value".into(),
        NodeKind::Math { op, .. } => format!("Math · {op:?}"),
        NodeKind::Position => "Position".into(),
        NodeKind::Normal { .. } => "Normal".into(),
        NodeKind::Noise(n) => format!("{:?}", n.kind),
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

fn has_output(kind: &NodeKind) -> bool {
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

/// Nodes offered by the right-click menu.
fn palette() -> Vec<(&'static str, NodeKind)> {
    vec![
        ("Perlin", NodeKind::Noise(NoiseParams::default())),
        ("Value", NodeKind::Float(1.0)),
        ("Math", NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }),
        ("Position", NodeKind::Position),
        ("Normal", NodeKind::Normal { flip: false }),
        ("Density Output", NodeKind::OutDensity),
        ("Normal Output", NodeKind::OutNormal),
    ]
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
        let (canvas, bg) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(canvas);
        painter.rect_filled(canvas, 0.0, Color32::from_gray(24));
        let origin = canvas.min;
        let pointer = ui.input(|i| i.pointer.hover_pos());

        // Pan and zoom.
        if bg.dragged_by(egui::PointerButton::Middle) {
            self.pan += bg.drag_delta() / self.zoom;
        }
        if let Some(p) = pointer.filter(|p| canvas.contains(*p)) {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
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
        let wire = Stroke::new(2.0 * self.zoom, Color32::from_gray(170));
        for e in &graph.edges {
            if let (Some(a), Some(b)) = (layouts[&e.from].output, layouts[&e.to].inputs.get(e.port)) {
                painter.add(bezier(a, *b, wire));
            }
        }

        // Nodes.
        let z = self.zoom;
        let font = egui::FontId::proportional(13.0 * z);
        let small = egui::FontId::proportional(11.5 * z);
        let mut moved = Vec2::ZERO;
        let mut detach = None;
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

            let s = summary(&node.kind);
            if !s.is_empty() {
                let at = Pos2::new(l.rect.left() + 8.0 * z, l.rect.top() + (HEADER_H + ROW_H * 0.5) * z);
                painter.text(at, egui::Align2::LEFT_CENTER, s, small.clone(), Color32::from_gray(190));
            }

            // Ports are registered after the node body so they win the hit test.
            for (port, &p) in l.inputs.iter().enumerate() {
                let connected = graph.input(id, port);
                painter.circle(p, PORT_R * z, if connected.is_some() { Color32::from_gray(200) } else { Color32::from_gray(90) }, Stroke::new(1.0, Color32::from_gray(200)));
                painter.text(p + Vec2::new(10.0 * z, 0.0), egui::Align2::LEFT_CENTER, input_label(&node.kind, port), small.clone(), Color32::from_gray(210));
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("in", id, port)), Sense::drag());
                if pr.drag_started() {
                    if let Some(src) = connected {
                        detach = Some((id, port));
                        self.wire_from = Some(src);
                    }
                }
            }
            if let Some(p) = l.output {
                painter.circle(p, PORT_R * z, Color32::from_rgb(140, 190, 255), Stroke::new(1.0, Color32::from_gray(220)));
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("out", id)), Sense::drag());
                if pr.drag_started() {
                    self.wire_from = Some(id);
                }
            }
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

        // Wire being dragged: connect on release over an input port.
        let released = ui.input(|i| i.pointer.primary_released());
        if let (Some(from), Some(p)) = (self.wire_from, ui.input(|i| i.pointer.interact_pos())) {
            if let Some(a) = layouts.get(&from).and_then(|l| l.output) {
                painter.add(bezier(a, p, Stroke::new(2.0 * z, Color32::from_rgb(140, 190, 255))));
            }
            if released {
                let hit = layouts.iter().find_map(|(&id, l)| {
                    l.inputs.iter().position(|ip| ip.distance(p) < PORT_R * 3.0 * z).map(|port| (id, port))
                });
                if let Some((to, port)) = hit {
                    // Refuse links that would make a cycle.
                    if !graph.depends_on(from, to) {
                        graph.connect(from, to, port);
                    }
                }
                self.wire_from = None;
            }
        }

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

        // Delete selection (only while the canvas is hovered and no text field has focus).
        let over_canvas = pointer.is_some_and(|p| canvas.contains(p));
        let typing = ui.memory(|m| m.focused().is_some());
        if over_canvas && !typing && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
            for id in std::mem::take(&mut self.selected) {
                graph.remove(id);
            }
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

fn bezier(a: Pos2, b: Pos2, stroke: Stroke) -> CubicBezierShape {
    let dx = ((b.x - a.x).abs() * 0.5).max(30.0);
    CubicBezierShape::from_points_stroke([a, a + Vec2::new(dx, 0.0), b - Vec2::new(dx, 0.0), b], false, Color32::TRANSPARENT, stroke)
}
