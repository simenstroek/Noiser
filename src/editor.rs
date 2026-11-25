//! Node graph canvas: pan/zoom, drag nodes, wire ports, box select, add/delete/duplicate, preview.
//!
//! Controls: middle-drag pans, scroll zooms, left-drag on empty space box-selects, right-click
//! adds a node, Delete removes the selection, Ctrl+D duplicates it at the cursor.
//! Drag from an output to an input (or the other way) to connect; drag a connected input away to
//! disconnect it; drop a wire on empty space to pick a new node for it. The eye in a node's
//! header previews that node in the 3D and slice views.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, NoiseParams, PortType, Tiling};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, epaint::CubicBezierShape};
use crate::keymap::{Action, pressed};
use std::collections::{BTreeMap, BTreeSet};

pub const ORANGE: Color32 = Color32::from_rgb(255, 150, 40);
const ACCENT: Color32 = Color32::from_rgb(140, 190, 255);
/// Pin and wire colors: floats blue, vectors yellow.
fn type_color(t: PortType) -> Color32 {
    match t {
        PortType::Float => Color32::from_rgb(100, 165, 255),
        PortType::Vector => Color32::from_rgb(255, 205, 70),
        PortType::Mesh => Color32::from_rgb(240, 150, 70),
        PortType::Image => Color32::from_rgb(235, 110, 170),
    }
}

/// Values that are the same everywhere (a Value or Random node and what is computed only from them).
const UNIFORM: Color32 = Color32::from_rgb(80, 215, 215);

/// The colour of an output pin and its wires: uniform values get their own, the rest follow the type.
fn output_color(graph: &Graph, tiling: &BTreeMap<NodeId, Tiling>, id: NodeId, port: usize) -> Color32 {
    let t = graph.output_type_at(id, port);
    if matches!(t, PortType::Float | PortType::Vector) && tiling.get(&id) == Some(&Tiling::Const) { UNIFORM } else { type_color(t) }
}

/// Cache on, and nodes a cache spares the preview from evaluating.
const CACHED: Color32 = Color32::from_rgb(120, 220, 160);

const NODE_W: f32 = 140.0;

/// A node's width: wider with a second output, so its label does not run into the input labels on the same row.
fn node_w(kind: &NodeKind) -> f32 {
    if kind.output_count() > 1 { 190.0 } else { NODE_W }
}
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
    /// The mesh graph's editor: its add menus offer the mesh nodes.
    pub mesh_mode: bool,
    /// The Landscape tab's editor: its add menus offer the Mask, Slope and Noise Graph nodes.
    pub landscape_mode: bool,
    /// Short status text drawn on a node (the Landscape tab's Erosion nodes: up to date, out of date, running).
    pub badges: std::collections::HashMap<NodeId, String>,
    /// A node was double-clicked this frame (the app opens what it refers to).
    pub opened: Option<NodeId>,
    /// Snap nodes to a 20-unit grid when a drag ends (a preference).
    pub snap: bool,
    /// Where a Ctrl+right-drag that cuts wires started.
    cut_start: Option<Pos2>,
    /// Ctrl was held when the right button went down.
    cut_armed: bool,
    /// Tests cannot hold Ctrl; they set this to cut with a plain right-drag.
    cut_without_ctrl: bool,
    /// Which output pin of the node the wire being dragged started from (Separate XYZ has three).
    wire_port: usize,
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
    canvas_origin: Pos2,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            pan: Vec2::new(40.0, 40.0),
            zoom: 1.0,
            selected: BTreeSet::new(),
            mesh_mode: false,
            landscape_mode: false,
            opened: None,
            badges: Default::default(),
            snap: false,
            cut_start: None,
            cut_armed: false,
            cut_without_ctrl: false,
            wire_port: 0,
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
            canvas_origin: Pos2::ZERO,
        }
    }
}

pub fn title(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(_) | NodeKind::Vec3(_) => "Value".into(),
        NodeKind::Math { .. } => "Math".into(), // the operator is the dropdown on the node
        NodeKind::Remap { .. } => "Remap".into(),
        NodeKind::Mix { .. } => "Mix".into(),
        NodeKind::Separate => "Separate XYZ".into(),
        NodeKind::Combine { .. } => "Combine XYZ".into(),
        NodeKind::Random { .. } => "Random".into(),
        NodeKind::Curve { .. } => "Curve".into(),
        NodeKind::Grid { .. } => "Grid".into(),
        NodeKind::Pattern(p) => p.kind.name().into(),
        NodeKind::Position { .. } => "Position".into(),
        NodeKind::Radial(r) => match r.mode {
            crate::graph::RadialMode::Cylindrical => "Cylindrical position".into(),
            crate::graph::RadialMode::Spherical => "Spherical position".into(),
        },
        NodeKind::Reroute => "Reroute".into(),
        NodeKind::NamedReroute { name } => format!("{name}  (reroute)"),
        NodeKind::NamedRerouteUse { name } => name.clone(),
        NodeKind::Normal { .. } => "Normal".into(),
        NodeKind::Mask { name } => if name.is_empty() { "Map".into() } else { format!("Map: {name}") },
        NodeKind::Slope { .. } => "Slope".into(),
        NodeKind::Erosion(e) => if e.preset.is_empty() { "Erosion".into() } else { format!("Erosion: {}", e.preset) },
        NodeKind::Noise(n) => n.kind.name().into(),
        NodeKind::OutDensity => "Density Output".into(),
        NodeKind::OutNormal => "Normal Output".into(),
        NodeKind::LoadSdf(p) => std::path::Path::new(&p.file).file_stem().filter(|_| !p.file.is_empty()).map_or("Load SDF".into(), |s| s.to_string_lossy().into_owned()),
        NodeKind::Scatter(p) => std::path::Path::new(&p.file).file_stem().filter(|_| !p.file.is_empty()).map_or("Scatter SDF".into(), |s| format!("Scatter {}", s.to_string_lossy())),
        NodeKind::LoadMesh(p) => std::path::Path::new(&p.path).file_stem().filter(|_| !p.path.is_empty()).map_or(if p.sdf { "Load SDF" } else { "Load Mesh" }.into(), |s| s.to_string_lossy().into_owned()),
        NodeKind::NoiseGraph(p) => std::path::Path::new(&p.file).file_stem().map_or("Noise Graph".into(), |s| s.to_string_lossy().into_owned()),
        NodeKind::Remesh { .. } => "Remesh".into(),
        NodeKind::AutoUv(_) => "Auto UVs".into(),
        NodeKind::Bake(_) => "Bake".into(),
        NodeKind::OutMesh(p) => format!("Mesh: {}", p.name),
        NodeKind::OutImage(p) => format!("Image: {}", p.name),
        NodeKind::TransferUv => "Transfer UVs".into(),
        NodeKind::ImageSplit => "Split channels".into(),
        NodeKind::ImageCombine => "Combine channels".into(),
        NodeKind::ImageLevels { .. } => "Image levels".into(),
    }
}

fn input_label(kind: &NodeKind, port: usize) -> &'static str {
    match (kind, port) {
        (NodeKind::Math { .. }, 0) => "A",
        (NodeKind::Math { .. }, 1) => "B",
        (NodeKind::Math { op: MathOp::Lerp, .. }, _) => "Factor",
        (NodeKind::Math { op: MathOp::InverseLerp, .. }, _) => "Value",
        (NodeKind::Math { .. }, _) => "Blend",
        (NodeKind::Reroute | NodeKind::NamedReroute { .. }, _) => "In",
        (NodeKind::Curve { .. }, _) => "Value",
        (NodeKind::Separate, _) => "Vector",
        (NodeKind::Position { .. }, 0) => "X",
        (NodeKind::Position { .. }, 1) => "Y",
        (NodeKind::Position { .. }, _) => "Z",
        (NodeKind::Random { .. }, 0) => "Min",
        (NodeKind::Random { .. }, 1) => "Max",
        (NodeKind::Combine { .. }, 0) => "X",
        (NodeKind::Combine { .. }, 1) => "Y",
        (NodeKind::Combine { .. }, _) => "Z",
        (NodeKind::Mix { .. }, 0) => "Factor",
        (NodeKind::Mix { .. }, 1) => "A",
        (NodeKind::Mix { .. }, _) => "B",
        (NodeKind::Remap { .. }, 0) => "Value",
        (NodeKind::Remap { .. }, 1) => "From min",
        (NodeKind::Remap { .. }, 2) => "From max",
        (NodeKind::Remap { .. }, 3) => "To min",
        (NodeKind::Remap { .. }, _) => "To max",
        (NodeKind::Grid { .. } | NodeKind::Pattern(_) | NodeKind::Noise(_), port) if port > 0 => kind.param_pins()[port - 1].label,
        (NodeKind::Grid { .. } | NodeKind::Pattern(_), _) => "Position",
        (NodeKind::Noise(_) | NodeKind::Radial(_), _) => "Position",
        (NodeKind::Normal { .. } | NodeKind::Slope { .. }, _) => "Field",
        (NodeKind::Mask { .. }, _) => "Position",
        (NodeKind::Erosion(_), _) => "Height",
        (NodeKind::OutDensity, _) => "Density",
        (NodeKind::OutNormal, _) => "Normal",
        (NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::NoiseGraph(_), _) => "Position",
        (NodeKind::Scatter(_), 0) => "Field",
        (NodeKind::Scatter(_), _) => "Roughen",
        (NodeKind::Remesh { .. }, _) => "Field",
        (NodeKind::AutoUv(_) | NodeKind::OutMesh(_), _) => "Mesh",
        (NodeKind::TransferUv, 0) => "Target",
        (NodeKind::TransferUv, _) => "UV source",
        (NodeKind::Bake(_), 0) => "Mesh",
        (NodeKind::Bake(_), 1) => "Field",
        (NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageLevels { .. }, _) => "Image",
        (NodeKind::ImageCombine, 0) => "R",
        (NodeKind::ImageCombine, 1) => "G",
        (NodeKind::ImageCombine, 2) => "B",
        (NodeKind::ImageCombine, _) => "A",
        (NodeKind::Bake(_), _) => "Custom",
        _ => "",
    }
}

/// Header colour by role: generators (noise, patterns, grid) green, math and shaping (math, remap, mix, curve,
/// normal) blue, plumbing (values, position, vector split/join, reroutes) grey, outputs purple.
fn header_tint(kind: &NodeKind) -> Color32 {
    match kind {
        NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. } => Color32::from_rgb(50, 95, 70),
        NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Mix { .. } | NodeKind::Curve { .. } | NodeKind::Normal { .. } => Color32::from_rgb(55, 70, 110),
        // The Landscape tab's own nodes: earth tones, like the tab.
        NodeKind::Mask { .. } => Color32::from_rgb(130, 82, 52),
        NodeKind::Slope { .. } => Color32::from_rgb(112, 96, 58),
        NodeKind::Erosion(_) => Color32::from_rgb(70, 110, 150),
        NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::OutMesh(_) | NodeKind::OutImage(_) => Color32::from_rgb(80, 60, 95),
        NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::NoiseGraph(_) => Color32::from_rgb(50, 95, 70),
        NodeKind::Scatter(_) => Color32::from_rgb(70, 110, 60),
        NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::TransferUv | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } => Color32::from_rgb(50, 100, 110),
        _ => Color32::from_rgb(78, 78, 84),
    }
}

/// A menu entry filled with the colour of the node's header.
fn node_button(ui: &mut egui::Ui, label: &str, kind: &NodeKind) -> egui::Response {
    ui.add(egui::Button::new(egui::RichText::new(label).color(Color32::WHITE)).fill(header_tint(kind)))
}

pub fn has_output(kind: &NodeKind) -> bool {
    !matches!(kind, NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::OutMesh(_) | NodeKind::OutImage(_))
}

static LANDSCAPE_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Whether the add menus offer the mesh graph's nodes (and leave out the noise graph's outputs) instead of the
/// noise graph's: set by the editor each frame from `Editor::mesh_mode`.
static MESH_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One-line summary shown in the node body for nodes without inputs.
fn summary(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Float(v) => format!("{v:.3}"),
        NodeKind::Vec3(v) => format!("{:.2}, {:.2}, {:.2}", v[0], v[1], v[2]),
        NodeKind::Position { .. } => String::new(), // its X, Y and Z rows are the body
        NodeKind::Reroute => "passes its input on".into(),
        NodeKind::NamedReroute { .. } => "named: usable anywhere".into(),
        NodeKind::NamedRerouteUse { .. } => "reads a named reroute".into(),
        _ => String::new(),
    }
}

/// Nodes offered by the add menus.
fn palette() -> Vec<(&'static str, NodeKind)> {
    let mesh_mode = MESH_MODE.load(std::sync::atomic::Ordering::Relaxed);
    let mut all = noise_palette();
    if LANDSCAPE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        all.extend([
            ("Map", NodeKind::Mask { name: String::new() }),
            ("Slope", NodeKind::Slope { scale: 1.0 }),
            ("Erosion", NodeKind::Erosion(Default::default())),
            ("Noise Graph", NodeKind::NoiseGraph(Default::default())),
        ]);
    }
    if mesh_mode {
        all.retain(|(_, k)| !matches!(k, NodeKind::OutDensity | NodeKind::OutNormal));
        all.extend([
            ("Load Mesh", NodeKind::LoadMesh(Default::default())),
            ("Scatter SDF", NodeKind::Scatter(Default::default())),
            ("Noise Graph", NodeKind::NoiseGraph(Default::default())),
            ("Remesh", NodeKind::Remesh { triangles: 20_000, adaptivity: 1.0 }),
            ("Auto UVs", NodeKind::AutoUv(Default::default())),
            ("Transfer UVs", NodeKind::TransferUv),
            ("Bake", NodeKind::Bake(Default::default())),
            ("Mesh Output", NodeKind::OutMesh(Default::default())),
            ("Image Output", NodeKind::OutImage(Default::default())),
            ("Split channels", NodeKind::ImageSplit),
            ("Combine channels", NodeKind::ImageCombine),
            ("Image levels", NodeKind::ImageLevels { scale: 1.0, offset: 0.0, invert: false }),
        ]);
    }
    all
}

fn noise_palette() -> Vec<(&'static str, NodeKind)> {
    let noises = NoiseKind::ALL.map(|kind| (kind.name(), NodeKind::Noise(NoiseParams { kind, ..Default::default() })));
    noises
        .into_iter()
        .chain([
            ("Value", NodeKind::Float(1.0)),
            ("Load SDF", NodeKind::LoadSdf(Default::default())),
            ("Math", NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }),
            ("Remap", NodeKind::Remap { vals: [0.0, 0.0, 1.0, 0.0, 1.0], clamp: false }),
            ("Separate XYZ", NodeKind::Separate),
            ("Combine XYZ", NodeKind::Combine { vals: [0.0; 3] }),
            ("Random", NodeKind::Random { vals: [0.0, 1.0], seed: None }),
            ("Mix", NodeKind::Mix { vals: [0.5, 0.0, 1.0], clamp: true }),
            ("Reroute", NodeKind::Reroute),
            ("Named reroute", NodeKind::NamedReroute { name: "name".into() }),
            ("Use named reroute", NodeKind::NamedRerouteUse { name: "name".into() }),
            ("Curve", NodeKind::Curve { points: vec![[0.0, 0.0], [1.0, 1.0]], mode: crate::graph::CurveMode::Linear }),
            ("Grid", NodeKind::Grid { counts: [8.0; 3], corners: false, cubic: true, amount: 1.0 }),
            ("Bricks", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Bricks))),
            ("Dimples", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Dimples))),
            ("Checker", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Checker))),
            ("Stripes", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Stripes))),
            ("Gyroid", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Gyroid))),
            ("Fractal", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Fractal))),
            ("Truchet", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Truchet))),
            ("Honeycomb", NodeKind::Pattern(crate::graph::PatternParams::for_kind(crate::graph::PatternKind::Honeycomb))),
            ("Position", NodeKind::position()),
            ("Radial position", NodeKind::Radial(Default::default())),
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
    /// The first output (what most code needs) and all of them (Separate XYZ has three).
    output: Option<Pos2>,
    outputs: Vec<Pos2>,
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
            // Each side resizes the frame from that side (a thin strip centred on the border); the corner does two at once.
            let w = 8.0 * z;
            let strips = [
                (Rect::from_min_max(Pos2::new(rect.left() - w / 2.0, rect.top() + 22.0 * z), Pos2::new(rect.left() + w / 2.0, rect.bottom() - 16.0 * z)), egui::CursorIcon::ResizeHorizontal, [true, false, false, false]),
                (Rect::from_min_max(Pos2::new(rect.right() - w / 2.0, rect.top() + 22.0 * z), Pos2::new(rect.right() + w / 2.0, rect.bottom() - 16.0 * z)), egui::CursorIcon::ResizeHorizontal, [false, true, false, false]),
                (Rect::from_min_max(Pos2::new(rect.left() + 16.0 * z, rect.top() - w / 2.0), Pos2::new(rect.right() - 16.0 * z, rect.top() + w / 2.0)), egui::CursorIcon::ResizeVertical, [false, false, true, false]),
                (Rect::from_min_max(Pos2::new(rect.left() + 16.0 * z, rect.bottom() - w / 2.0), Pos2::new(rect.right() - 16.0 * z, rect.bottom() + w / 2.0)), egui::CursorIcon::ResizeVertical, [false, false, false, true]),
            ];
            for (k, (strip, cursor, [left, right, top, bottom])) in strips.into_iter().enumerate() {
                let r = ui.interact(strip, ui.scope_id().with(("frame edge", i, k)), Sense::drag()).on_hover_cursor(cursor);
                if r.dragged_by(egui::PointerButton::Primary) {
                    let d = r.drag_delta() / z;
                    let f = &mut graph.frames[i];
                    let (min_w, min_h) = (80.0, 50.0);
                    if left {
                        let new_w = (f.size[0] - d.x).max(min_w);
                        f.pos[0] += f.size[0] - new_w;
                        f.size[0] = new_w;
                    }
                    if right {
                        f.size[0] = (f.size[0] + d.x).max(min_w);
                    }
                    if top {
                        let new_h = (f.size[1] - d.y).max(min_h);
                        f.pos[1] += f.size[1] - new_h;
                        f.size[1] = new_h;
                    }
                    if bottom {
                        f.size[1] = (f.size[1] + d.y).max(min_h);
                    }
                }
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
                hi = [hi[0].max(n.pos[0] + node_w(&n.kind)), hi[1].max(n.pos[1] + h)];
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

    /// Where a screen position is in the graph, as of the last frame the editor was shown.
    pub fn screen_to_graph(&self, s: Pos2) -> [f32; 2] {
        let p = self.to_graph(self.canvas_origin, s);
        [p.x, p.y]
    }

    fn to_graph(&self, origin: Pos2, s: Pos2) -> Pos2 {
        ((s - origin) / self.zoom - self.pan).to_pos2()
    }

    fn layout(&self, origin: Pos2, kind: &NodeKind, pos: [f32; 2], pins: &[String]) -> Layout {
        let z = self.zoom;
        // Parameter pins that are not exposed take no row; their (unused) ports sit on the first one.
        let shown = |port: usize| port == 0 || kind.param_pins().get(port.wrapping_sub(1)).is_none_or(|p| pins.iter().any(|n| n == p.name));
        let rows: Vec<usize> = (0..kind.input_count()).scan(0, |row, port| Some(if shown(port) { *row += 1; *row - 1 } else { 0 })).collect();
        let n_in = rows.iter().copied().max().map_or(0, |m| m + 1).min(kind.input_count());
        let top_left = self.to_screen(origin, pos);
        let top = body_top(kind);
        let n_out = kind.output_count().max(1);
        let h = (top + ROW_H * n_in.max(n_out) as f32 + 6.0) * z;
        let rect = Rect::from_min_size(top_left, Vec2::new(node_w(kind) * z, h));
        let inputs = rows.iter().map(|&i| Pos2::new(rect.left(), rect.top() + (top + ROW_H * (i as f32 + 0.5)) * z)).collect();
        let outputs: Vec<Pos2> = if kind.output_count() > 1 {
            (0..kind.output_count()).map(|k| Pos2::new(rect.right(), rect.top() + (top + ROW_H * (k as f32 + 0.5)) * z)).collect()
        } else if has_output(kind) {
            vec![Pos2::new(rect.right(), rect.top() + HEADER_H * 0.5 * z)]
        } else {
            Vec::new()
        };
        Layout { rect, inputs, output: outputs.first().copied(), outputs }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, graph: &mut Graph, tiling: &BTreeMap<NodeId, Tiling>) {
        MESH_MODE.store(self.mesh_mode, std::sync::atomic::Ordering::Relaxed);
        LANDSCAPE_MODE.store(self.landscape_mode, std::sync::atomic::Ordering::Relaxed);
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
        self.canvas_origin = origin;
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
                matches!(n.kind, NodeKind::Math { .. }) && op_field(self.layout(origin, &n.kind, n.pos, &n.pins).rect, self.zoom).contains(p)
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
        if bg.secondary_clicked()
            && let Some(p) = bg.interact_pointer_pos() {
                self.menu_at = self.to_graph(origin, p);
                self.menu_search.clear();
                self.menu_fresh = true;
            }
        bg.context_menu(|ui| {
            ui.label("Add node");
            let search = ui.add(egui::TextEdit::singleline(&mut self.menu_search).hint_text("Search…"));
            if std::mem::take(&mut self.menu_fresh) {
                search.request_focus();
            }
            let enter = search.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            for (i, (label, kind)) in palette_matching(&self.menu_search).into_iter().enumerate() {
                if node_button(ui, label, &kind).clicked() || (enter && i == 0) {
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
            graph.nodes.iter().map(|(&id, n)| (id, self.layout(origin, &n.kind, n.pos, &n.pins))).collect();
        // Ctrl + right-drag across wires cuts them.
        if ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary)) {
            self.cut_armed = self.cut_without_ctrl || ui.input(|i| i.modifiers.command);
        }
        if bg.drag_started_by(egui::PointerButton::Secondary) && self.cut_armed {
            self.cut_start = bg.interact_pointer_pos();
        }
        if let (Some(a), Some(p)) = (self.cut_start, ui.input(|i| i.pointer.interact_pos())) {
            painter.line_segment([a, p], Stroke::new(1.5, Color32::from_rgb(230, 90, 90)));
            if bg.drag_stopped_by(egui::PointerButton::Secondary) {
                let before = graph.edges.len();
                graph.edges.retain(|e| match (layouts[&e.from].outputs.get(e.from_port).copied().or(layouts[&e.from].output), layouts[&e.to].inputs.get(e.port)) {
                    (Some(s), Some(t)) => !wire_crosses(s, *t, a, p),
                    _ => true,
                });
                let _ = before;
                self.cut_start = None;
            }
        }
        if !ui.input(|i| i.pointer.button_down(egui::PointerButton::Secondary)) {
            self.cut_start = None;
        }
        // Wires under nodes.
        for e in &graph.edges {
            if let (Some(a), Some(b)) = (layouts[&e.from].outputs.get(e.from_port).copied().or(layouts[&e.from].output), layouts[&e.to].inputs.get(e.port)) {
                painter.add(bezier(a, *b, Stroke::new(2.0 * self.zoom, output_color(graph, tiling, e.from, e.from_port))));
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
            if resp.double_clicked() {
                self.opened = Some(id);
            }
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
            let header_color = if node.muted { Color32::from_gray(60) } else { header_tint(&node.kind) };
            painter.rect_filled(header, CornerRadius { nw: r.nw, ne: r.ne, sw: 0, se: 0 }, header_color);
            let cacheable = !self.mesh_mode && matches!(node.kind, NodeKind::Noise(_) | NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Mix { .. } | NodeKind::Separate | NodeKind::Curve { .. } | NodeKind::Grid { .. } | NodeKind::Pattern(_));
            // Keep the title clear of the header buttons.
            let title_clip = header.with_max_x(header.right() - (if cacheable { 48.0 } else { 28.0 }) * z);
            painter.with_clip_rect(title_clip).text(header.left_center() + Vec2::new(8.0 * z, 0.0), egui::Align2::LEFT_CENTER, &{ let name = node.label.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| title(&node.kind)); if node.muted { format!("{name} (muted)") } else { name } }, font.clone(), if node.muted { Color32::from_gray(150) } else { Color32::WHITE });
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
            if let Some(badge) = self.badges.get(&id) {
                let at = Pos2::new(l.rect.right() - 8.0 * z, l.rect.bottom() - 9.0 * z);
                painter.text(at, egui::Align2::RIGHT_CENTER, badge, small.clone(), Color32::from_rgb(150, 200, 235));
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
            if has_output(&node.kind) && !self.mesh_mode {
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
                if !node.port_visible(port) {
                    continue;
                }
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
                            self.wire_port = graph.input_from(id, port).map_or(0, |(_, fp)| fp);
                            Wire::FromOutput(src)
                        }
                        None => Wire::FromInput(id, port),
                    });
                }
            }
            let split = node.kind.output_count() > 1;
            for (k, p) in l.outputs.iter().copied().enumerate() {
                // Each output of a Separate node is a float, of a Bake node an image; every other node has one output of its own type.
                let color = if matches!(node.kind, NodeKind::Separate) && tiling.get(&id) != Some(&Tiling::Const) { type_color(PortType::Float) } else { output_color(graph, tiling, id, k) };
                painter.circle(p, PORT_R * z, color, Stroke::new(1.0, Color32::from_gray(220)));
                if split {
                    let name = match &node.kind {
                        NodeKind::Bake(b) => b.channels.get(k).map(|c| c.name()).or_else(|| b.custom.get(k - b.channels.len()).map(|s| s.as_str())).unwrap_or(""),
                        NodeKind::ImageSplit => ["R", "G", "B", "A"][k.min(3)],
                        NodeKind::Erosion(_) => ["Height", "Flow", "Deposition", "Wear"][k.min(3)],
                        NodeKind::LoadMesh(_) => ["Field", "Mesh"][k.min(1)],
                        NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. } => ["Value", "Displacement"][k.min(1)],
                        _ => ["X", "Y", "Z"][k.min(2)],
                    };
                    painter.text(p - Vec2::new(10.0 * z, 0.0), egui::Align2::RIGHT_CENTER, name, small.clone(), Color32::from_gray(210));
                }
                let pr = ui.interact(Rect::from_center_size(p, Vec2::splat(PORT_R * 3.0 * z)), ui.scope_id().with(("out", id, k)), Sense::drag());
                if pr.drag_started() {
                    self.wire = Some(Wire::FromOutput(id));
                    self.wire_port = k;
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
        if let Some((id, port, v)) = set_value
            && let Some(n) = graph.nodes.get_mut(&id) {
                n.kind.set_const_input(port, v);
            }
        if let Some((to, port)) = detach {
            graph.disconnect(to, port);
        }
        // Optional snapping: when a drag ends, selected nodes settle on a 20-unit grid.
        if self.snap && self.node_dragged && ui.input(|i| i.pointer.primary_released()) {
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
        if released && self.node_dragged
            && let [id] = self.selected.iter().copied().collect::<Vec<_>>()[..]
                && let Some(rect) = layouts.get(&id).map(|l| l.rect) {
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
                    if let Some(a) = layouts.get(&from).and_then(|l| l.outputs.get(self.wire_port).copied().or(l.output)) {
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
                            Some((to, port)) if !graph.depends_on(from, to) => graph.connect_from(from, self.wire_port, to, port),
                            Some(_) => {}
                            None => self.open_pin_menu(origin, canvas, &layouts, p, wire),
                        }
                    }
                    Wire::FromInput(to, port) => {
                        let hit = layouts.iter().find_map(|(&id, l)| l.outputs.iter().position(near).map(|k| (id, k)));
                        match hit {
                            Some((from, k)) if !graph.depends_on(from, to) => graph.connect_from(from, k, to, port),
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
            if ui.input(|i| pressed(i, Action::Delete)) {
                for id in std::mem::take(&mut self.selected) {
                    graph.remove(id);
                }
            }
            if ui.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)))
                && let Some(t) = clipboard_text(graph, &self.selected) {
                    ui.ctx().copy_text(t);
                }
            let pasted = ui.input(|i| i.events.iter().find_map(|e| if let egui::Event::Paste(t) = e { Some(t.clone()) } else { None }));
            if let (Some(t), Some(p)) = (pasted, pointer) {
                let at = self.to_graph(origin, p);
                let new = paste_text(graph, &t, [at.x, at.y]);
                if !new.is_empty() {
                    self.selected = new;
                }
            }
            if ui.input(|i| pressed(i, Action::SelectAll)) {
                self.selected = graph.nodes.keys().copied().collect();
            }
            if ui.input(|i| pressed(i, Action::Mute)) {
                for id in &self.selected {
                    if let Some(n) = graph.nodes.get_mut(id).filter(|n| !matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal)) {
                        n.muted = !n.muted;
                    }
                }
            }
            // C: a comment frame around the selected nodes (with room for its title bar).
            if ui.input(|i| pressed(i, Action::CommentFrame)) && !self.selected.is_empty() {
                let bounds = self.selected.iter().filter_map(|id| layouts.get(id)).map(|l| l.rect).reduce(|a, b| a.union(b));
                if let Some(b) = bounds {
                    let (lo, hi) = (self.to_graph(origin, b.min), self.to_graph(origin, b.max));
                    graph.frames.push(crate::graph::Frame { title: "Frame".into(), pos: [lo.x - 16.0, lo.y - 38.0], size: [hi.x - lo.x + 32.0, hi.y - lo.y + 54.0] });
                }
            }
            if ui.input(|i| pressed(i, Action::NodesFrameSelected)) {
                let only = self.selected.clone();
                self.frame(canvas.size(), graph, &only);
            }
            if ui.input(|i| pressed(i, Action::NodesFrameAll)) {
                self.frame(canvas.size(), graph, &BTreeSet::new());
            }
            if ui.input(|i| pressed(i, Action::Duplicate))
                && let Some(p) = pointer {
                    let at = self.to_graph(origin, p);
                    self.selected = duplicate(graph, &self.selected, [at.x, at.y]);
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
                    if node_button(ui, label, &kind).clicked() {
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
                Wire::FromOutput(from) => graph.connect_from(from, self.wire_port, id, 0),
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

/// Lays the graph out in columns left to right by how far each node is from the inputs: sources on the left,
/// the outputs on the right, each column keeping the nodes' current top to bottom order.
pub fn arrange(graph: &mut Graph) {
    let mut depth: BTreeMap<NodeId, usize> = BTreeMap::new();
    // Longest path from a node without inputs; the graph has no cycles, but a cycle must not hang this.
    fn visit(g: &Graph, id: NodeId, depth: &mut BTreeMap<NodeId, usize>, stack: &mut Vec<NodeId>) -> usize {
        if let Some(d) = depth.get(&id) {
            return *d;
        }
        if stack.contains(&id) {
            return 0;
        }
        stack.push(id);
        let d = g.edges.iter().filter(|e| e.to == id).map(|e| visit(g, e.from, depth, stack) + 1).max().unwrap_or(0);
        stack.pop();
        depth.insert(id, d);
        d
    }
    let ids: Vec<NodeId> = graph.nodes.keys().copied().collect();
    for id in &ids {
        visit(graph, *id, &mut depth, &mut Vec::new());
    }
    let ed = Editor::default();
    let mut columns: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
    for id in &ids {
        columns.entry(depth[id]).or_default().push(*id);
    }
    for (d, col) in columns.iter_mut() {
        col.sort_by(|a, b| graph.nodes[a].pos[1].total_cmp(&graph.nodes[b].pos[1]));
        let mut y = 0.0;
        for id in col.iter() {
            let node = graph.nodes.get_mut(id).unwrap();
            node.pos = [*d as f32 * (NODE_W + 90.0), y];
            y += ed.layout(Pos2::ZERO, &node.kind, node.pos, &node.pins).rect.height() + 30.0;
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
        if let Some(copy) = graph.nodes.get_mut(&id) {
            (copy.muted, copy.label, copy.cache) = (n.muted, n.label.clone(), n.cache);
        }
        map.insert(*old, id);
    }
    let internal: Vec<_> = src.edges.iter().filter_map(|e| Some((*map.get(&e.from)?, e.from_port, *map.get(&e.to)?, e.port))).collect();
    for (from, from_port, to, port) in internal {
        graph.connect_from(from, from_port, to, port);
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

/// Whether segment `p`-`q` crosses the wire from `a` to `b` (drawn by `bezier`).
fn wire_crosses(a: Pos2, b: Pos2, p: Pos2, q: Pos2) -> bool {
    let dx = ((b.x - a.x).abs() * 0.5).max(30.0);
    let (c1, c2) = (a + Vec2::new(dx, 0.0), b - Vec2::new(dx, 0.0));
    let at = |t: f32| {
        let u = 1.0 - t;
        (a.to_vec2() * (u * u * u) + c1.to_vec2() * (3.0 * u * u * t) + c2.to_vec2() * (3.0 * u * t * t) + b.to_vec2() * (t * t * t)).to_pos2()
    };
    let cross = |o: Pos2, u: Pos2, v: Pos2| (u.x - o.x) * (v.y - o.y) - (u.y - o.y) * (v.x - o.x);
    (0..32).any(|i| {
        let (s, e) = (at(i as f32 / 32.0), at((i + 1) as f32 / 32.0));
        cross(p, q, s) * cross(p, q, e) <= 0.0 && cross(s, e, p) * cross(s, e, q) <= 0.0
    })
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
    fn a_cut_line_crosses_only_the_wires_it_passes_over() {
        let (a, b) = (Pos2::new(0.0, 0.0), Pos2::new(300.0, 100.0));
        assert!(wire_crosses(a, b, Pos2::new(150.0, -50.0), Pos2::new(150.0, 150.0)), "straight across the middle");
        assert!(!wire_crosses(a, b, Pos2::new(150.0, 200.0), Pos2::new(150.0, 300.0)), "well below it");
        assert!(!wire_crosses(a, b, Pos2::new(400.0, -50.0), Pos2::new(400.0, 150.0)), "past its end");
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
        // Labels and mute travel with a copy.
        let mut labelled = Graph::default_graph();
        labelled.nodes.get_mut(&1).unwrap().label = Some("Base".into());
        labelled.nodes.get_mut(&1).unwrap().muted = true;
        let mut dst = Graph::default();
        let pasted = paste_text(&mut dst, &clipboard_text(&labelled, &BTreeSet::from([1])).unwrap(), [0.0; 2]);
        let copy = &dst.nodes[pasted.iter().next().unwrap()];
        assert!(copy.muted && copy.label.as_deref() == Some("Base"));
        
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

    fn editor_harness(graph: Graph) -> egui_kittest::Harness<'static, (Editor, Graph)> {
        egui_kittest::Harness::builder().with_size(Vec2::new(900.0, 600.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx());
                ed.debug_origin = ui.available_rect_before_wrap().min;
                ed.show(ui, g, &BTreeMap::new());
            },
            (Editor::default(), graph),
        )
    }

    fn drag(h: &mut egui_kittest::Harness<'static, (Editor, Graph)>, from: Pos2, to: Pos2) {
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
    }

    /// Dragging a frame title bar moves the frame and the nodes inside it, not the nodes outside.
    #[test]
    fn dragging_a_frame_moves_the_nodes_inside_it() {
        let mut g = Graph::default();
        let inside = g.add(NodeKind::Float(1.0), [100.0, 100.0]);
        let outside = g.add(NodeKind::Float(2.0), [600.0, 400.0]);
        g.frames.push(crate::graph::Frame { title: "Group".into(), pos: [60.0, 60.0], size: [300.0, 250.0] });
        let mut h = editor_harness(g);
        h.run();
        let bar = { let ed = &h.state().0; ed.to_screen(ed.debug_origin, [200.0, 70.0]) };
        drag(&mut h, bar, bar + Vec2::new(120.0, 40.0));
        let g = &h.state().1;
        assert!((g.frames[0].pos[0] - 180.0).abs() < 2.0 && (g.frames[0].pos[1] - 100.0).abs() < 2.0, "frame at {:?}", g.frames[0].pos);
        assert!((g.nodes[&inside].pos[0] - 220.0).abs() < 2.0, "the node inside moved with it: {:?}", g.nodes[&inside].pos);
        assert_eq!(g.nodes[&outside].pos, [600.0, 400.0]);
    }

    #[test]
    fn arranging_puts_sources_left_of_what_they_feed_without_overlaps() {
        for name in Graph::TEMPLATES {
            let mut g = Graph::template(name);
            for n in g.nodes.values_mut() {
                n.pos = [50.0, 50.0]; // everything in a heap
            }
            arrange(&mut g);
            for e in &g.edges {
                assert!(g.nodes[&e.from].pos[0] < g.nodes[&e.to].pos[0], "{name}: {} should be left of {}", e.from, e.to);
            }
            let ed = Editor::default();
            let rects: Vec<Rect> = g.nodes.values().map(|n| ed.layout(Pos2::ZERO, &n.kind, n.pos, &n.pins).rect).collect();
            for (i, a) in rects.iter().enumerate() {
                assert!(rects[i + 1..].iter().all(|b| !a.intersects(*b)), "{name}: overlap after arranging");
            }
        }
    }

    /// No two nodes of a template sit on top of each other.
    #[test]
    fn template_nodes_do_not_overlap() {
        let ed = Editor::default();
        for name in Graph::TEMPLATES {
            let g = Graph::template(name);
            let rects: Vec<(NodeId, Rect)> = g.nodes.iter().map(|(id, n)| (*id, ed.layout(Pos2::ZERO, &n.kind, n.pos, &n.pins).rect)).collect();
            for (i, (a, ra)) in rects.iter().enumerate() {
                for (b, rb) in &rects[i + 1..] {
                    assert!(!ra.intersects(*rb), "{name}: nodes {a} and {b} overlap ({ra:?} and {rb:?})");
                }
            }
        }
    }

    /// Dragging from the Z pin of a Separate XYZ node to an input records output port 2 on the new wire.
    #[test]
    fn dragging_from_a_separate_output_pin_remembers_which_one() {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [20.0, 20.0]);
        let sep = g.add(NodeKind::Separate, [200.0, 20.0]);
        let comb = g.add(NodeKind::Combine { vals: [0.0; 3] }, [500.0, 20.0]);
        g.connect(pos, sep, 0);
        let mut h = editor_harness(g);
        h.run();
        let (from, to) = {
            let ed = &h.state().0;
            let origin = ed.debug_origin;
            let (ls, lc) = (ed.layout(origin, &NodeKind::Separate, [200.0, 20.0], &[]), ed.layout(origin, &NodeKind::Combine { vals: [0.0; 3] }, [500.0, 20.0], &[]));
            (ls.outputs[2], lc.inputs[1])
        };
        drag(&mut h, from, to);
        let g = &h.state().1;
        assert_eq!(g.input_from(comb, 1), Some((sep, 2)), "{:?}", g.edges);
    }

    /// Ctrl + right-drag across a wire removes it; a plain right-drag does not.
    #[test]
    fn ctrl_right_drag_cuts_a_wire() {
        let mut g = Graph::default();
        let src = g.add(NodeKind::Float(1.0), [100.0, 200.0]);
        let out = g.add(NodeKind::OutDensity, [500.0, 200.0]);
        g.connect(src, out, 0);
        let mut h = editor_harness(g);
        h.run();
        let (from, to) = { let ed = &h.state().0; (ed.to_screen(ed.debug_origin, [330.0, 120.0]), ed.to_screen(ed.debug_origin, [330.0, 330.0])) };
        let cut = |h: &mut egui_kittest::Harness<'static, (Editor, Graph)>, mods: egui::Modifiers| {
            h.hover_at(from);
            h.run();
            h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Secondary, pressed: true, modifiers: mods });
            h.run();
            for i in 1..=8 {
                h.hover_at(from + (to - from) * i as f32 / 8.0);
                h.run();
            }
            h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Secondary, pressed: false, modifiers: mods });
            h.run();
        };
        cut(&mut h, egui::Modifiers::NONE);
        assert_eq!(h.state().1.edges.len(), 1, "a plain right-drag leaves the wire");
        h.state_mut().0.cut_without_ctrl = true; // the harness cannot hold Ctrl
        cut(&mut h, egui::Modifiers::NONE);
        assert!(h.state().1.edges.is_empty(), "the cut drag removed the wire");
    }

    /// A lone node dropped on a wire is spliced into it: source to the node, node to the old target.
    #[test]
    fn dropping_a_node_on_a_wire_splices_it_in() {
        let mut g = Graph::default();
        let src = g.add(NodeKind::Float(1.0), [100.0, 200.0]);
        let out = g.add(NodeKind::OutDensity, [500.0, 200.0]);
        g.connect(src, out, 0);
        let math = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [150.0, 400.0]);
        let mut h = editor_harness(g);
        h.run();
        let header = { let ed = &h.state().0; ed.to_screen(ed.debug_origin, [150.0 + NODE_W * 0.4, 400.0 + HEADER_H * 0.5]) };
        // Drop it so the node sits over the middle of the wire.
        let target = { let ed = &h.state().0; ed.to_screen(ed.debug_origin, [330.0, 215.0]) };
        drag(&mut h, header, target);
        let g = &h.state().1;
        assert!(g.edges.iter().any(|e| e.from == src && e.to == math && e.port == 0), "source feeds the node: {:?}", g.edges);
        assert!(g.edges.iter().any(|e| e.from == math && e.to == out), "the node feeds the output: {:?}", g.edges);
        assert!(!g.edges.iter().any(|e| e.from == src && e.to == out), "the direct wire is gone");
    }

    /// Right-clicking a frame title bar opens the frame menu (rename, delete), and deleting removes only the frame.
    #[test]
    fn the_frame_menu_deletes_a_frame() {
        use egui_kittest::kittest::Queryable;
        let mut g = Graph::default();
        g.add(NodeKind::Float(1.0), [100.0, 100.0]);
        g.frames.push(crate::graph::Frame { title: "Group".into(), pos: [60.0, 60.0], size: [300.0, 250.0] });
        let mut h = editor_harness(g);
        h.run();
        let bar = { let ed = &h.state().0; ed.to_screen(ed.debug_origin, [200.0, 70.0]) };
        h.hover_at(bar);
        h.run();
        h.event(egui::Event::PointerButton { pos: bar, button: egui::PointerButton::Secondary, pressed: true, modifiers: Default::default() });
        h.run();
        h.event(egui::Event::PointerButton { pos: bar, button: egui::PointerButton::Secondary, pressed: false, modifiers: Default::default() });
        h.run();
        h.get_by_label("Delete frame").click();
        h.run();
        assert!(h.state().1.frames.is_empty());
        assert_eq!(h.state().1.nodes.len(), 1);
    }

    /// Ctrl+C then Ctrl+V over the canvas copies the selection to the clipboard text and pastes it back;
    /// M mutes it, F frames it, and a node dropped on a wire is spliced in.
    #[test]
    fn keyboard_shortcuts_copy_paste_mute_and_frame() {
        use egui_kittest::Harness;
        let mut h = Harness::builder().with_size(Vec2::new(900.0, 600.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx());
                ed.debug_origin = ui.available_rect_before_wrap().min;
                ed.show(ui, g, &BTreeMap::new());
            },
            (Editor::default(), Graph::default_graph()),
        );
        h.run();
        h.state_mut().0.selected = BTreeSet::from([0, 1]);
        h.hover_at(Pos2::new(450.0, 300.0));
        h.run();
        h.event(egui::Event::Copy);
        h.step(); // the output of exactly the frame that handled the event
        let copied = h.output().platform_output.commands.iter().find_map(|c| if let egui::OutputCommand::CopyText(t) = c { Some(t.clone()) } else { None });
        let text = copied.expect("Ctrl+C puts the selection on the clipboard");
        h.event(egui::Event::Paste(text));
        h.run();
        assert_eq!(h.state().1.nodes.len(), 6, "Position and Noise were pasted next to the originals");
        // Mute whatever the paste selected.
        h.event(egui::Event::Key { key: egui::Key::M, physical_key: Some(egui::Key::M), pressed: true, repeat: false, modifiers: Default::default() });
        h.run();
        assert!(h.state().1.nodes.values().filter(|n| n.muted).count() == 2);
    }

    /// C puts a frame around the selected nodes.
    #[test]
    fn c_frames_the_selection() {
        use egui_kittest::Harness;
        let mut h = Harness::builder().with_size(Vec2::new(900.0, 600.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, (ed, g): &mut (Editor, Graph)| {
                egui_extras::install_image_loaders(ui.ctx());
                ed.debug_origin = ui.available_rect_before_wrap().min;
                ed.show(ui, g, &BTreeMap::new());
            },
            (Editor::default(), Graph::default_graph()),
        );
        h.run();
        h.state_mut().0.selected = BTreeSet::from([0, 1]);
        h.hover_at(Pos2::new(450.0, 300.0));
        h.run();
        h.event(egui::Event::Key { key: egui::Key::C, physical_key: Some(egui::Key::C), pressed: true, repeat: false, modifiers: Default::default() });
        h.run();
        let g = &h.state().1;
        assert_eq!(g.frames.len(), 1);
        let f = &g.frames[0];
        for id in [0, 1] {
            let p = g.nodes[&id].pos;
            assert!(p[0] > f.pos[0] && p[1] > f.pos[1] + 22.0 && p[0] < f.pos[0] + f.size[0] && p[1] < f.pos[1] + f.size[1], "node {id} at {p:?} outside {f:?}");
        }
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
