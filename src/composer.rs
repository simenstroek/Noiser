//! The SDF Composer tab: build a shape from SDF primitives the way SDF Modeler does, and save it as a `.sdf` file that
//! the Noise graph and the Mesh graph can load. The document is an `SdfFile`, so what Noiser does not edit is kept.

use crate::composer_render::{Camera, Renderer, pick};
use crate::sdf_file::{Assembly, Layer, Node, SdfFile};
use crate::sdf_model::{LayerField, M3, V3, euler_matrix, into_frame};
use eframe::egui;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

const BRUSHES: [(&str, &str); 5] = [("sphere", "Sphere"), ("cube", "Cube"), ("cylinder", "Cylinder"), ("prism", "Prism"), ("polygon", "Polygon")];
const MODES: [(&str, &str); 9] = [
    ("add", "Add"),
    ("sub", "Subtract"),
    ("intersect", "Intersect"),
    ("paint", "Paint"),
    ("push", "Push"),
    ("avoid", "Avoid"),
    ("emboss", "Emboss"),
    ("deboss", "Deboss"),
    ("shell", "Shell"),
];
const RED: egui::Color32 = egui::Color32::from_rgb(230, 80, 80);
const GREEN: egui::Color32 = egui::Color32::from_rgb(90, 200, 100);
const BLUE: egui::Color32 = egui::Color32::from_rgb(90, 140, 245);
const MAX_UNDO: usize = 200;
const MAX_VIEW_SIDE: f32 = 900.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    Local,
    Global,
}

// ---- documents and tree paths -------------------------------------------------------------------------------------

fn blank_file() -> SdfFile {
    SdfFile::from_json(
        r#"{"version":[0,6,1],
            "environment":{"use_lightmap":false,"lightmap":"","lightmap_rotation":0,"luminance":1,"color":[0.327,0.333,0.34]},
            "assemblies":[{"resolution":6,"shapes":[]}],
            "layers":[{"name":"Layer","assembly":0,"image":"","hidden":false,"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1],
                       "roughness":0.3,"metallic":0,"emissive":0,"ior":1,"transparency":0,"image_opacity":1}]}"#,
    )
    .expect("the blank document is valid")
}

fn new_layer(name: &str, assembly: usize) -> Layer {
    let mut extra = Map::new();
    for (key, value) in [("image", Value::from("")), ("roughness", Value::from(0.3)), ("metallic", Value::from(0)), ("emissive", Value::from(0)), ("ior", Value::from(1)), ("transparency", Value::from(0)), ("image_opacity", Value::from(1))] {
        extra.insert(key.to_string(), value);
    }
    Layer { name: name.to_string(), assembly, hidden: false, position: [0.0; 3], rotation: [0.0; 3], scale: [1.0; 3], extra }
}

fn new_shape(brush: &str) -> Node {
    let mut params = Map::new();
    let mut put = |key: &str, value: Value| {
        params.insert(key.to_string(), value);
    };
    put("mode", "add".into());
    put("chamfer", false.into());
    put("blend", 0.02.into());
    put("blend_strength", 0.0.into());
    for key in ["mirror_x", "mirror_y", "mirror_z"] {
        put(key, false.into());
    }
    put("mirror_blend", 0.into());
    put("brush", brush.into());
    put("bevel", (if brush == "cube" { 1.0 } else { 0.0 }).into());
    put("round", 0.1.into());
    put("thickness", 1.into());
    put("cone", 0.into());
    put("star", 0.into());
    put("points", 6.into());
    put("repeat_mode", "linear".into());
    put("linear_repeats", Value::from(vec![1, 1, 1]));
    put("steps", Value::from(vec![1.5, 1.5, 1.5]));
    put("radial_repeats", 1.into());
    put("radius", 1.5.into());
    put("repeat_axis", "x".into());
    let label = BRUSHES.iter().find(|(id, _)| *id == brush).map_or("Shape", |(_, label)| label);
    Node { name: label.to_string(), hidden: false, kind: "shape".into(), position: [0.0; 3], rotation: [0.0; 3], scale: [1.0; 3], color: [0.8; 3], params, children: None }
}

fn new_group() -> Node {
    let mut params = Map::new();
    for (key, value) in [("mode", Value::from("add")), ("chamfer", false.into()), ("blend", 0.02.into()), ("blend_strength", 0.0.into()), ("tint", false.into())] {
        params.insert(key.to_string(), value);
    }
    Node { name: "Group".into(), hidden: false, kind: "group".into(), position: [0.0; 3], rotation: [0.0; 3], scale: [1.0; 3], color: [0.8; 3], params, children: Some(vec![]) }
}

fn node_at<'a>(nodes: &'a [Node], path: &[usize]) -> Option<&'a Node> {
    let (first, rest) = path.split_first()?;
    let node = nodes.get(*first)?;
    if rest.is_empty() { Some(node) } else { node_at(node.children(), rest) }
}

fn node_at_mut<'a>(nodes: &'a mut [Node], path: &[usize]) -> Option<&'a mut Node> {
    let (first, rest) = path.split_first()?;
    let node = nodes.get_mut(*first)?;
    if rest.is_empty() { Some(node) } else { node_at_mut(node.children.as_mut()?, rest) }
}

fn siblings_mut<'a>(nodes: &'a mut Vec<Node>, parent: &[usize]) -> Option<&'a mut Vec<Node>> {
    if parent.is_empty() {
        return Some(nodes);
    }
    Some(node_at_mut(nodes, parent)?.children.get_or_insert_with(Vec::new))
}

// ---- frames -------------------------------------------------------------------------------------------------------

/// A child frame inside its parent: `p_parent = t + r · (s · p_child)`.
#[derive(Clone, Copy, Debug)]
struct Xf {
    t: V3,
    r: M3,
    s: f64,
}

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn rotate(r: &M3, v: V3) -> V3 {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn apply(xf: &Xf, p: V3) -> V3 {
    let scaled = [p[0] * xf.s, p[1] * xf.s, p[2] * xf.s];
    let rotated = rotate(&xf.r, scaled);
    [xf.t[0] + rotated[0], xf.t[1] + rotated[1], xf.t[2] + rotated[2]]
}

fn compose(outer: &Xf, inner: &Xf) -> Xf {
    Xf { t: apply(outer, inner.t), r: mat_mul(&outer.r, &inner.r), s: outer.s * inner.s }
}

fn group_frame(node: &Node) -> Xf {
    Xf { t: node.position, r: euler_matrix(node.rotation), s: if node.scale[0] == 0.0 { 1.0 } else { node.scale[0] } }
}

fn column(m: &M3, i: usize) -> V3 {
    [m[0][i], m[1][i], m[2][i]]
}

fn axis_rotation(axis: usize, angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    match axis {
        0 => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        1 => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        _ => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// Euler angles `(x, y, z)` whose `Rz · Ry · Rx` is `m`.
fn euler_from_matrix(m: &M3) -> V3 {
    let y = (-m[2][0]).clamp(-1.0, 1.0).asin();
    if y.cos().abs() > 1e-6 {
        [m[2][1].atan2(m[2][2]), y, m[1][0].atan2(m[0][0])]
    } else {
        [(-m[1][2]).atan2(m[1][1]), y, 0.0]
    }
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn point_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let length_squared = ab.length_sq();
    if length_squared < 1e-6 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / length_squared).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

// ---- gizmo ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct GizmoGeometry {
    origin: V3,
    axes: [V3; 3],
    length: f64,
    parent: Xf,
}

#[derive(Clone, Debug)]
struct Drag {
    axis: usize,
    tool: Tool,
    start_pointer: egui::Pos2,
    start_position: V3,
    start_rotation: M3,
    start_scale: [f64; 3],
    geometry: GizmoGeometry,
    view: Camera,
}

fn axis_color(axis: usize) -> egui::Color32 {
    [RED, GREEN, BLUE][axis]
}

// ---- state --------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropZone {
    Before,
    Inside,
    After,
}

enum Action {
    Select(Vec<usize>),
    ToggleHidden(Vec<usize>),
    ToggleCollapsed(Vec<usize>),
    /// Move a node next to, into, or (onto a shape) together with another; `None` means the end of the top level.
    Drop { from: Vec<usize>, onto: Option<Vec<usize>>, zone: DropZone },
}

struct Row {
    path: Vec<usize>,
    depth: usize,
    label: String,
    mode: String,
    hidden: bool,
    is_group: bool,
    supported: bool,
}

pub struct Composer {
    file: SdfFile,
    path: Option<PathBuf>,
    layer: usize,
    selection: Option<Vec<usize>>,
    collapsed: HashSet<Vec<usize>>,
    last_committed: SdfFile,
    undo: Vec<SdfFile>,
    redo: Vec<SdfFile>,
    modified: bool,
    changed: bool,
    version: u64,
    camera: Camera,
    tool: Tool,
    space: Space,
    renderer: Option<Renderer>,
    field: Option<Arc<LayerField>>,
    texture: Option<egui::TextureHandle>,
    submitted: Option<(u64, Camera, [usize; 2], Option<Vec<usize>>)>,
    submitted_generation: u64,
    rendering: bool,
    frame_pending: bool,
    drag: Option<Drag>,
    pub status: String,
}

impl Default for Composer {
    fn default() -> Self {
        let file = blank_file();
        Composer {
            last_committed: file.clone(),
            file,
            path: None,
            layer: 0,
            selection: None,
            collapsed: HashSet::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            modified: false,
            changed: false,
            version: 0,
            camera: Camera::default(),
            tool: Tool::Move,
            space: Space::Local,
            renderer: None,
            field: None,
            texture: None,
            submitted: None,
            submitted_generation: 0,
            rendering: false,
            frame_pending: true,
            drag: None,
            status: String::new(),
        }
    }
}

impl Composer {
    fn assembly_index(&self) -> usize {
        self.file.layers.get(self.layer).map_or(0, |layer| layer.assembly)
    }

    fn shapes(&self) -> &[Node] {
        self.file.assemblies.get(self.assembly_index()).map_or(&[], |a| a.shapes.as_slice())
    }

    fn shapes_mut(&mut self) -> Option<&mut Vec<Node>> {
        let index = self.assembly_index();
        self.file.assemblies.get_mut(index).map(|a| &mut a.shapes)
    }

    /// Something in the document changed: redraw, and remember it for undo once the gesture ends.
    fn touch(&mut self) {
        self.changed = true;
        self.modified = true;
        self.version += 1;
        self.field = None;
    }

    fn replace_document(&mut self, file: SdfFile, path: Option<PathBuf>) {
        self.last_committed = file.clone();
        self.file = file;
        self.path = path;
        self.layer = 0;
        self.selection = None;
        self.collapsed.clear();
        self.undo.clear();
        self.redo.clear();
        self.modified = false;
        self.changed = false;
        self.version += 1;
        self.field = None;
        self.frame_pending = true;
    }

    fn new_document(&mut self) {
        self.replace_document(blank_file(), None);
        self.status = "New composition".into();
    }

    fn open(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]).pick_file() else { return };
        self.open_path(path);
    }

    pub fn open_path(&mut self, path: PathBuf) {
        match SdfFile::load(&path) {
            Ok(file) => {
                self.status = format!("Opened {}", path.display());
                self.replace_document(file, Some(path));
            }
            Err(e) => self.status = e,
        }
    }

    fn save(&mut self, ask: bool) {
        let path = match (&self.path, ask) {
            (Some(path), false) => path.clone(),
            _ => {
                let name = self.path.as_ref().and_then(|p| p.file_name()).map_or("composition.sdf".to_string(), |n| n.to_string_lossy().into_owned());
                match rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]).set_file_name(name).save_file() {
                    Some(path) => path,
                    None => return,
                }
            }
        };
        match self.file.save(&path) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                self.path = Some(path);
                self.modified = false;
            }
            Err(e) => self.status = e,
        }
    }

    fn commit_if_settled(&mut self, pointer_down: bool) {
        if !self.changed || pointer_down {
            return;
        }
        self.changed = false;
        if self.file == self.last_committed {
            return;
        }
        self.undo.push(std::mem::replace(&mut self.last_committed, self.file.clone()));
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn undo(&mut self) {
        if self.changed {
            self.commit_if_settled(false);
        }
        let Some(previous) = self.undo.pop() else { return };
        self.redo.push(std::mem::replace(&mut self.file, previous));
        self.after_history_move();
    }

    fn redo(&mut self) {
        let Some(next) = self.redo.pop() else { return };
        self.undo.push(std::mem::replace(&mut self.file, next));
        self.after_history_move();
    }

    fn after_history_move(&mut self) {
        self.last_committed = self.file.clone();
        self.layer = self.layer.min(self.file.layers.len().saturating_sub(1));
        self.modified = true;
        self.version += 1;
        self.field = None;
        if self.selection.as_ref().is_some_and(|path| node_at(self.shapes(), path).is_none()) {
            self.selection = None;
        }
    }

    // ---- tree edits ----

    fn add_node(&mut self, node: Node) {
        let selection = self.selection.clone();
        let target = match &selection {
            Some(path) => match node_at(self.shapes(), path) {
                Some(selected) if selected.kind == "group" => Some((path.clone(), None)),
                Some(_) => Some((path[..path.len() - 1].to_vec(), Some(path[path.len() - 1] + 1))),
                None => None,
            },
            None => None,
        };
        let (parent, at) = target.unwrap_or((Vec::new(), None));
        let Some(shapes) = self.shapes_mut() else { return };
        let Some(list) = siblings_mut(shapes, &parent) else { return };
        let index = at.unwrap_or(list.len()).min(list.len());
        list.insert(index, node);
        let mut path = parent;
        path.push(index);
        self.selection = Some(path);
        self.touch();
    }

    fn delete_selected(&mut self) {
        let Some(path) = self.selection.clone() else { return };
        let (index, parent) = (path[path.len() - 1], path[..path.len() - 1].to_vec());
        let Some(shapes) = self.shapes_mut() else { return };
        let Some(list) = siblings_mut(shapes, &parent) else { return };
        if index < list.len() {
            list.remove(index);
        }
        self.selection = None;
        self.touch();
    }

    fn duplicate_selected(&mut self) {
        let Some(path) = self.selection.clone() else { return };
        let Some(copy) = node_at(self.shapes(), &path).cloned() else { return };
        let (index, parent) = (path[path.len() - 1], path[..path.len() - 1].to_vec());
        let Some(shapes) = self.shapes_mut() else { return };
        let Some(list) = siblings_mut(shapes, &parent) else { return };
        list.insert(index + 1, copy);
        let mut new_path = parent;
        new_path.push(index + 1);
        self.selection = Some(new_path);
        self.touch();
    }

    fn move_selected(&mut self, down: bool) {
        let Some(path) = self.selection.clone() else { return };
        let (index, parent) = (path[path.len() - 1], path[..path.len() - 1].to_vec());
        let Some(shapes) = self.shapes_mut() else { return };
        let Some(list) = siblings_mut(shapes, &parent) else { return };
        let target = if down { index + 1 } else { index.wrapping_sub(1) };
        if target >= list.len() {
            return;
        }
        list.swap(index, target);
        let mut new_path = parent;
        new_path.push(target);
        self.selection = Some(new_path);
        self.touch();
    }

    fn group_selected(&mut self) {
        let Some(path) = self.selection.clone() else { return };
        let (index, parent) = (path[path.len() - 1], path[..path.len() - 1].to_vec());
        let Some(shapes) = self.shapes_mut() else { return };
        let Some(list) = siblings_mut(shapes, &parent) else { return };
        if index >= list.len() {
            return;
        }
        let inner = list.remove(index);
        let mut group = new_group();
        group.children = Some(vec![inner]);
        list.insert(index, group);
        let mut new_path = parent;
        new_path.push(index);
        new_path.push(0);
        self.selection = Some(new_path);
        self.touch();
    }

    /// Moves a node by drag and drop. `onto` is the row it was dropped on; dropping on a shape's middle wraps both in a
    /// new group.
    fn move_node(&mut self, from: &[usize], onto: Option<&[usize]>, zone: DropZone) {
        if from.is_empty() || onto.is_some_and(|o| o == from || o.starts_with(from)) {
            return;
        }
        let Some(node) = node_at(self.shapes(), from).cloned() else { return };
        let (from_index, from_parent) = (from[from.len() - 1], from[..from.len() - 1].to_vec());
        // Where the target sits once the dragged node is out of the tree.
        let onto: Option<Vec<usize>> = onto.map(|target| {
            let mut target = target.to_vec();
            let depth = from.len() - 1;
            if target.len() > depth && target[..depth] == from[..depth] && target[depth] > from_index {
                target[depth] -= 1;
            }
            target
        });
        let Some(shapes) = self.shapes_mut() else { return };
        {
            let Some(list) = siblings_mut(shapes, &from_parent) else { return };
            if from_index >= list.len() {
                return;
            }
            list.remove(from_index);
        }
        let new_path = match onto {
            None => {
                shapes.push(node);
                vec![shapes.len() - 1]
            }
            Some(target) => {
                let (index, parent) = (target[target.len() - 1], target[..target.len() - 1].to_vec());
                let target_is_group = node_at(shapes, &target).is_some_and(|n| n.kind == "group");
                match zone {
                    DropZone::Inside if target_is_group => {
                        let Some(group) = node_at_mut(shapes, &target) else { return };
                        let children = group.children.get_or_insert_with(Vec::new);
                        children.push(node);
                        let mut path = target;
                        path.push(children.len() - 1);
                        path
                    }
                    DropZone::Inside => {
                        let Some(list) = siblings_mut(shapes, &parent) else { return };
                        if index >= list.len() {
                            return;
                        }
                        let existing = list.remove(index);
                        let mut group = new_group();
                        group.children = Some(vec![existing, node]);
                        list.insert(index, group);
                        let mut path = parent;
                        path.push(index);
                        path.push(1);
                        path
                    }
                    DropZone::Before | DropZone::After => {
                        let Some(list) = siblings_mut(shapes, &parent) else { return };
                        let at = (if zone == DropZone::After { index + 1 } else { index }).min(list.len());
                        list.insert(at, node);
                        let mut path = parent;
                        path.push(at);
                        path
                    }
                }
            }
        };
        self.selection = Some(new_path);
        self.touch();
    }

    fn add_layer(&mut self) {
        self.file.assemblies.push(Assembly { resolution: 6, shapes: Vec::new(), extra: Map::new() });
        let assembly = self.file.assemblies.len() - 1;
        let name = format!("Layer {}", self.file.layers.len() + 1);
        self.file.layers.push(new_layer(&name, assembly));
        self.layer = self.file.layers.len() - 1;
        self.selection = None;
        self.touch();
    }

    fn delete_layer(&mut self) {
        if self.file.layers.len() <= 1 {
            return;
        }
        let removed = self.file.layers.remove(self.layer);
        if !self.file.layers.iter().any(|l| l.assembly == removed.assembly) {
            self.file.assemblies.remove(removed.assembly);
            for layer in &mut self.file.layers {
                if layer.assembly > removed.assembly {
                    layer.assembly -= 1;
                }
            }
        }
        self.layer = self.layer.min(self.file.layers.len() - 1);
        self.selection = None;
        self.touch();
    }

    // ---- derived data ----

    fn field(&mut self) -> Arc<LayerField> {
        if self.field.is_none() {
            let built = LayerField::for_assembly(&self.file, self.assembly_index()).unwrap_or_else(|| LayerField::for_assembly(&blank_file(), 0).unwrap());
            self.field = Some(Arc::new(built));
        }
        self.field.clone().unwrap()
    }

    fn parent_frame(&self, path: &[usize]) -> Xf {
        let mut frame = Xf { t: [0.0; 3], r: IDENTITY, s: 1.0 };
        let mut nodes = self.shapes();
        for &index in &path[..path.len().saturating_sub(1)] {
            let Some(node) = nodes.get(index) else { break };
            frame = compose(&frame, &group_frame(node));
            nodes = node.children();
        }
        frame
    }

    fn gizmo_geometry(&self, rect: egui::Rect) -> Option<GizmoGeometry> {
        let path = self.selection.as_ref()?;
        let node = node_at(self.shapes(), path)?;
        let parent = self.parent_frame(path);
        let origin = apply(&parent, node.position);
        let local = euler_matrix(node.rotation);
        let local_axes = [0, 1, 2].map(|i| {
            let axis = rotate(&parent.r, column(&local, i));
            axis
        });
        let axes = match (self.tool, self.space) {
            (Tool::Move, Space::Global) => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            _ => local_axes,
        };
        let length = self.camera.units_per_pixel(origin, rect.height()) * 100.0;
        Some(GizmoGeometry { origin, axes, length, parent })
    }
}

// ---- widgets --------------------------------------------------------------------------------------------------------

fn number(ui: &mut egui::Ui, label: &str, value: &mut f64, speed: f64, range: Option<std::ops::RangeInclusive<f64>>) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let mut widget = egui::DragValue::new(value).speed(speed);
        if let Some(range) = range {
            widget = widget.range(range);
        }
        changed = ui.add(widget).changed();
    });
    changed
}

fn vec3(ui: &mut egui::Ui, label: &str, value: &mut [f64; 3], speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for component in value.iter_mut() {
            changed |= ui.add(egui::DragValue::new(component).speed(speed)).changed();
        }
    });
    changed
}

fn param_number(ui: &mut egui::Ui, node: &mut Node, key: &str, label: &str, default: f64, speed: f64, range: Option<std::ops::RangeInclusive<f64>>) -> bool {
    let mut value = node.number(key).unwrap_or(default);
    if number(ui, label, &mut value, speed, range) {
        node.set(key, Value::from(value));
        return true;
    }
    false
}

fn param_flag(ui: &mut egui::Ui, node: &mut Node, key: &str, label: &str) -> bool {
    let mut value = node.flag(key).unwrap_or(false);
    if ui.checkbox(&mut value, label).changed() {
        node.set(key, Value::from(value));
        return true;
    }
    false
}

fn param_choice(ui: &mut egui::Ui, node: &mut Node, key: &str, label: &str, options: &[(&str, &str)]) -> bool {
    let current = node.text(key).unwrap_or(options[0].0).to_string();
    let shown = options.iter().find(|(id, _)| *id == current).map_or(current.as_str(), |(_, name)| name);
    let mut chosen = current.clone();
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(("composer choice", key)).selected_text(shown).show_ui(ui, |ui| {
            for (id, name) in options {
                ui.selectable_value(&mut chosen, id.to_string(), *name);
            }
        });
    });
    if chosen != current {
        node.set(key, Value::from(chosen));
        return true;
    }
    false
}

fn mode_symbol(mode: &str) -> &'static str {
    match mode {
        "sub" => "-",
        "intersect" => "&",
        "paint" => "~",
        "push" | "avoid" | "shell" => "o",
        "emboss" | "deboss" => "^",
        _ => "+",
    }
}

impl Composer {
    // ---- inspector ----

    fn inspector(&mut self, ui: &mut egui::Ui) {
        let Some(path) = self.selection.clone() else {
            ui.weak("Select a shape in the hierarchy or click one in the view.");
            return;
        };
        let assembly = self.assembly_index();
        let Some(node) = self.file.assemblies.get_mut(assembly).and_then(|a| node_at_mut(&mut a.shapes, &path)) else {
            self.selection = None;
            return;
        };
        let mut edited = false;
        let is_shape = node.kind == "shape";
        ui.heading(if is_shape { "Shape" } else if node.kind == "group" { "Group" } else { "Unsupported node" });
        if !is_shape && node.kind != "group" {
            ui.weak(format!("A {} is kept in the file but not shown or edited yet.", node.kind));
            return;
        }
        ui.horizontal(|ui| {
            ui.label("Name");
            edited |= ui.text_edit_singleline(&mut node.name).changed();
        });
        if is_shape {
            edited |= param_choice(ui, node, "brush", "Form", &BRUSHES);
        }
        ui.separator();
        edited |= vec3(ui, "Position", &mut node.position, 0.01);
        let mut degrees = node.rotation.map(f64::to_degrees);
        if vec3(ui, "Rotation", &mut degrees, 0.5) {
            node.rotation = degrees.map(f64::to_radians);
            edited = true;
        }
        if is_shape {
            edited |= vec3(ui, "Scale", &mut node.scale, 0.01);
        } else {
            let mut uniform = node.scale[0];
            if number(ui, "Scale", &mut uniform, 0.01, Some(0.001..=1000.0)) {
                node.scale = [uniform; 3];
                edited = true;
            }
        }
        if is_shape {
            let mut color = node.color.map(|c| c as f32);
            ui.horizontal(|ui| {
                ui.label("Color");
                if ui.color_edit_button_rgb(&mut color).changed() {
                    node.color = color.map(f64::from);
                    edited = true;
                }
            });
        }
        ui.separator();
        edited |= param_choice(ui, node, "mode", "Operation", &MODES);
        edited |= param_number(ui, node, "blend", "Blend", 0.02, 0.002, Some(0.0..=10.0));
        let mode = node.text("mode").unwrap_or("add").to_string();
        if matches!(mode.as_str(), "push" | "avoid" | "emboss" | "deboss" | "shell") {
            edited |= param_number(ui, node, "blend_strength", "Radius", 0.0, 0.002, Some(0.0..=10.0));
        }
        edited |= param_flag(ui, node, "chamfer", "Chamfer blend");
        if is_shape {
            let brush = node.text("brush").unwrap_or("sphere").to_string();
            ui.separator();
            ui.label("Modifiers");
            edited |= param_number(ui, node, "round", "Round", 0.0, 0.005, Some(0.0..=1.0));
            if brush != "sphere" {
                edited |= param_number(ui, node, "bevel", "Bevel", 0.0, 0.005, Some(0.0..=1.0));
            }
            if brush == "cube" {
                edited |= param_number(ui, node, "thickness", "Thickness", 1.0, 0.005, Some(0.0..=1.0));
                edited |= param_number(ui, node, "cone", "Cone", 0.0, 0.005, Some(0.0..=1.0));
            }
            if brush == "polygon" {
                edited |= param_number(ui, node, "points", "Points", 6.0, 0.1, Some(3.0..=32.0));
                edited |= param_number(ui, node, "star", "Star", 0.0, 0.005, Some(0.0..=1.0));
            }
            ui.separator();
            ui.label("Repetition");
            edited |= param_choice(ui, node, "repeat_mode", "Mode", &[("linear", "Linear"), ("radial", "Radial")]);
            if node.text("repeat_mode") == Some("radial") {
                edited |= param_number(ui, node, "radial_repeats", "Count", 1.0, 0.1, Some(1.0..=128.0));
                edited |= param_number(ui, node, "radius", "Radius", 1.5, 0.01, None);
                edited |= param_choice(ui, node, "repeat_axis", "Axis", &[("x", "X"), ("y", "Y"), ("z", "Z")]);
            } else {
                let mut counts = node.numbers::<3>("linear_repeats").unwrap_or([1.0; 3]);
                let mut steps = node.numbers::<3>("steps").unwrap_or([1.5; 3]);
                if vec3(ui, "Count", &mut counts, 0.1) {
                    node.set("linear_repeats", Value::from(counts.map(|c| c.round().max(1.0)).to_vec()));
                    edited = true;
                }
                if vec3(ui, "Steps", &mut steps, 0.01) {
                    node.set("steps", Value::from(steps.to_vec()));
                    edited = true;
                }
            }
            ui.separator();
            ui.label("Symmetry");
            ui.horizontal(|ui| {
                for (key, label) in [("mirror_x", "X"), ("mirror_y", "Y"), ("mirror_z", "Z")] {
                    edited |= param_flag(ui, node, key, label);
                }
            });
            if node.flag("mirror_x").unwrap_or(false) || node.flag("mirror_y").unwrap_or(false) || node.flag("mirror_z").unwrap_or(false) {
                edited |= param_number(ui, node, "mirror_blend", "Mirror blend", 0.0, 0.005, Some(0.0..=10.0));
            }
        }
        if edited {
            self.touch();
        }
    }

    // ---- hierarchy ----

    fn rows(&self) -> Vec<Row> {
        fn walk(nodes: &[Node], collapsed: &HashSet<Vec<usize>>, path: &mut Vec<usize>, out: &mut Vec<Row>) {
            for (index, node) in nodes.iter().enumerate() {
                path.push(index);
                let is_group = node.kind == "group";
                let supported = matches!(node.kind.as_str(), "shape" | "group");
                let brush = node.text("brush").unwrap_or("");
                let label = if !node.name.is_empty() {
                    node.name.clone()
                } else if is_group {
                    "Group".to_string()
                } else {
                    BRUSHES.iter().find(|(id, _)| *id == brush).map_or(node.kind.clone(), |(_, name)| name.to_string())
                };
                out.push(Row { path: path.clone(), depth: path.len() - 1, label, mode: node.text("mode").unwrap_or("add").to_string(), hidden: node.hidden, is_group, supported });
                if is_group && !collapsed.contains(path.as_slice()) {
                    walk(node.children(), collapsed, path, out);
                }
                path.pop();
            }
        }
        let mut out = Vec::new();
        walk(self.shapes(), &self.collapsed, &mut Vec::new(), &mut out);
        out
    }

    fn hierarchy(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let names: Vec<String> = self.file.layers.iter().map(|l| l.name.clone()).collect();
            egui::ComboBox::from_id_salt("composer layer").selected_text(names.get(self.layer).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    if ui.selectable_value(&mut self.layer, i, name).clicked() {
                        self.selection = None;
                        self.field = None;
                        self.version += 1;
                        self.frame_pending = true;
                    }
                }
            });
            if ui.button("+").on_hover_text("Add a layer").clicked() {
                self.add_layer();
            }
            if ui.add_enabled(self.file.layers.len() > 1, egui::Button::new("Delete")).on_hover_text("Delete this layer").clicked() {
                self.delete_layer();
            }
        });
        if let Some(layer) = self.file.layers.get_mut(self.layer)
            && ui.text_edit_singleline(&mut layer.name).changed()
        {
            self.touch();
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            for (brush, name) in BRUSHES {
                if ui.button(name).on_hover_text(format!("Add a {}", name.to_lowercase())).clicked() {
                    self.add_node(new_shape(brush));
                }
            }
            if ui.button("Group").on_hover_text("Add an empty group").clicked() {
                self.add_node(new_group());
            }
        });
        ui.horizontal_wrapped(|ui| {
            let has = self.selection.is_some();
            if ui.add_enabled(has, egui::Button::new("Duplicate")).clicked() {
                self.duplicate_selected();
            }
            if ui.add_enabled(has, egui::Button::new("Delete")).clicked() {
                self.delete_selected();
            }
            if ui.add_enabled(has, egui::Button::new("Up")).on_hover_text("Earlier: applied before its neighbour").clicked() {
                self.move_selected(false);
            }
            if ui.add_enabled(has, egui::Button::new("Down")).on_hover_text("Later: applied after its neighbour").clicked() {
                self.move_selected(true);
            }
            if ui.add_enabled(has, egui::Button::new("Wrap in group")).clicked() {
                self.group_selected();
            }
        });
        ui.separator();
        let rows = self.rows();
        let mut actions = Vec::new();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if rows.is_empty() {
                ui.weak("Empty. Add a shape above.");
            }
            for row in &rows {
                ui.horizontal(|ui| {
                    ui.add_space(row.depth as f32 * 14.0);
                    if row.is_group {
                        let open = !self.collapsed.contains(&row.path);
                        if ui.small_button(if open { "v" } else { ">" }).clicked() {
                            actions.push(Action::ToggleCollapsed(row.path.clone()));
                        }
                    } else {
                        ui.add_space(20.0);
                    }
                    let eye = if row.hidden { "  " } else { "o" };
                    if ui.small_button(eye).on_hover_text("Show or hide").clicked() {
                        actions.push(Action::ToggleHidden(row.path.clone()));
                    }
                    let selected = self.selection.as_ref() == Some(&row.path);
                    let text = format!("{} {}{}", mode_symbol(&row.mode), row.label, if row.supported { "" } else { " (not supported)" });
                    let mut label = egui::RichText::new(text);
                    if row.hidden || !row.supported {
                        label = label.weak();
                    }
                    let response = ui.add(egui::Button::selectable(selected, label).sense(egui::Sense::click_and_drag()));
                    if response.clicked() {
                        actions.push(Action::Select(row.path.clone()));
                    }
                    if response.drag_started() {
                        response.dnd_set_drag_payload(row.path.clone());
                    }
                    if response.dnd_hover_payload::<Vec<usize>>().is_some()
                        && let Some(pointer) = ui.input(|i| i.pointer.interact_pos())
                    {
                        let fraction = ((pointer.y - response.rect.top()) / response.rect.height().max(1.0)).clamp(0.0, 1.0);
                        let zone = if fraction < 0.25 { DropZone::Before } else if fraction > 0.75 { DropZone::After } else { DropZone::Inside };
                        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(235, 150, 70));
                        match zone {
                            DropZone::Before => _ = ui.painter().hline(response.rect.x_range(), response.rect.top(), stroke),
                            DropZone::After => _ = ui.painter().hline(response.rect.x_range(), response.rect.bottom(), stroke),
                            DropZone::Inside => _ = ui.painter().rect_stroke(response.rect, 3.0, stroke, egui::StrokeKind::Inside),
                        }
                        if let Some(payload) = response.dnd_release_payload::<Vec<usize>>() {
                            actions.push(Action::Drop { from: (*payload).clone(), onto: Some(row.path.clone()), zone });
                        }
                    }
                });
            }
            let rest = ui.allocate_response(egui::vec2(ui.available_width(), ui.available_height().max(24.0)), egui::Sense::hover());
            if rest.dnd_hover_payload::<Vec<usize>>().is_some() {
                ui.painter().rect_stroke(rest.rect, 3.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(235, 150, 70)), egui::StrokeKind::Inside);
                if let Some(payload) = rest.dnd_release_payload::<Vec<usize>>() {
                    actions.push(Action::Drop { from: (*payload).clone(), onto: None, zone: DropZone::After });
                }
            }
        });
        for action in actions {
            match action {
                Action::Drop { from, onto, zone } => self.move_node(&from, onto.as_deref(), zone),
                Action::Select(path) => self.selection = Some(path),
                Action::ToggleCollapsed(path) => {
                    if !self.collapsed.remove(&path) {
                        self.collapsed.insert(path);
                    }
                }
                Action::ToggleHidden(path) => {
                    let assembly = self.assembly_index();
                    if let Some(node) = self.file.assemblies.get_mut(assembly).and_then(|a| node_at_mut(&mut a.shapes, &path)) {
                        node.hidden = !node.hidden;
                        self.touch();
                    }
                }
            }
        }
    }

    // ---- viewport ----

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.interact(rect, ui.scope_id().with("composer view"), egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        let size = rect.size();
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new(ctx.clone()));
        }

        // Pictures from the worker.
        let frames = self.renderer.as_ref().map(|r| r.poll()).unwrap_or_default();
        for frame in frames {
            if frame.generation != self.submitted_generation {
                continue;
            }
            self.texture = Some(ctx.load_texture("composer view", frame.image, egui::TextureOptions::LINEAR));
            if frame.finest {
                self.rendering = false;
                if self.frame_pending
                    && let Some((lo, hi)) = frame.bounds
                {
                    self.camera = self.camera.framing(lo, hi);
                    self.frame_pending = false;
                }
            }
        }

        // The gizmo under the pointer, or a drag in progress.
        let geometry = self.gizmo_geometry(rect);
        let pointer = response.hover_pos().or_else(|| response.interact_pointer_pos());
        let hovered_axis = match (geometry, pointer, &self.drag) {
            (Some(g), Some(p), None) => self.hit_handle(&g, p - rect.min, size),
            _ => None,
        };
        if response.drag_started_by(egui::PointerButton::Primary)
            && let (Some(axis), Some(g), Some(start)) = (hovered_axis, geometry, response.interact_pointer_pos())
            && let Some(path) = &self.selection
            && let Some(node) = node_at(self.shapes(), path)
        {
            self.drag = Some(Drag {
                axis,
                tool: self.tool,
                start_pointer: start - rect.min.to_vec2(),
                start_position: node.position,
                start_rotation: euler_matrix(node.rotation),
                start_scale: node.scale,
                geometry: g,
                view: self.camera,
            });
        }
        if let Some(drag) = self.drag.clone() {
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(p) = response.interact_pointer_pos() {
                    self.apply_drag(&drag, p - rect.min.to_vec2(), size);
                }
            } else {
                self.drag = None;
            }
        } else if response.dragged_by(egui::PointerButton::Primary) && !ui.input(|i| i.modifiers.shift) {
            let delta = response.drag_delta();
            self.camera.yaw -= delta.x as f64 * 0.01;
            self.camera.pitch = (self.camera.pitch + delta.y as f64 * 0.01).clamp(-1.5, 1.5);
        } else if response.dragged_by(egui::PointerButton::Middle) || response.dragged_by(egui::PointerButton::Secondary) || (response.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.shift)) {
            let delta = response.drag_delta();
            let per_pixel = self.camera.units_per_pixel(self.camera.target, size.y);
            let (_, right, up) = self.camera.basis();
            for axis in 0..3 {
                self.camera.target[axis] += (-right[axis] * delta.x as f64 + up[axis] * delta.y as f64) * per_pixel;
            }
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.distance = (self.camera.distance * (1.0 - scroll as f64 * 0.002)).clamp(0.2, 500.0);
            }
        }
        if response.double_clicked() {
            self.frame_pending = true;
            self.submitted = None;
        }
        if response.clicked_by(egui::PointerButton::Primary)
            && hovered_axis.is_none()
            && let Some(p) = response.interact_pointer_pos()
        {
            let local = p - rect.min;
            let ndc = [2.0 * local.x as f64 / size.x as f64 - 1.0, 1.0 - 2.0 * local.y as f64 / size.y as f64];
            let field = self.field();
            self.selection = pick(&field, &self.camera, ndc, size.x as f64 / size.y as f64);
        }

        // Ask for a picture when anything it depends on changed.
        let scale = (MAX_VIEW_SIDE / size.x.max(size.y)).min(1.0) * ctx.pixels_per_point().min(1.0);
        let target = [((size.x * scale) as usize).max(16), ((size.y * scale) as usize).max(16)];
        let key = (self.version, self.camera, target, self.selection.clone());
        if self.submitted.as_ref() != Some(&key) {
            let field = self.field();
            if let Some(renderer) = &mut self.renderer {
                self.submitted_generation = renderer.submit(field, self.camera, target, self.selection.clone());
                self.rendering = true;
            }
            self.submitted = Some(key);
        }

        // Draw.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(30));
        if let Some(texture) = &self.texture {
            painter.image(texture.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        if self.shapes().is_empty() {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Add a shape from the hierarchy panel", egui::FontId::proportional(15.0), egui::Color32::from_gray(150));
        }
        if self.rendering {
            painter.text(rect.right_top() + egui::vec2(-10.0, 10.0), egui::Align2::RIGHT_TOP, "rendering…", egui::FontId::proportional(12.0), egui::Color32::from_gray(160));
        }
        let active = self.drag.as_ref().map(|d| d.axis);
        if let Some(g) = geometry {
            self.draw_gizmo(&painter, rect, &g, hovered_axis.or(active));
        }
        if let Some(axis) = hovered_axis {
            let _ = axis;
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }
        if self.rendering || self.drag.is_some() {
            ctx.request_repaint();
        }
    }

    fn handle_segment(&self, g: &GizmoGeometry, axis: usize, size: egui::Vec2) -> Option<(egui::Pos2, egui::Pos2)> {
        let view = [size.x, size.y];
        let a = self.camera.project(g.origin, view)?;
        let tip = [0, 1, 2].map(|k| g.origin[k] + g.axes[axis][k] * g.length);
        let b = self.camera.project(tip, view)?;
        Some((egui::pos2(a[0], a[1]), egui::pos2(b[0], b[1])))
    }

    fn ring_points(&self, g: &GizmoGeometry, axis: usize, size: egui::Vec2) -> Vec<egui::Pos2> {
        let (u, v) = (g.axes[(axis + 1) % 3], g.axes[(axis + 2) % 3]);
        let radius = g.length * 0.85;
        (0..=48)
            .filter_map(|k| {
                let angle = k as f64 / 48.0 * std::f64::consts::TAU;
                let (s, c) = angle.sin_cos();
                let p = [0, 1, 2].map(|i| g.origin[i] + (u[i] * c + v[i] * s) * radius);
                self.camera.project(p, [size.x, size.y]).map(|q| egui::pos2(q[0], q[1]))
            })
            .collect()
    }

    fn hit_handle(&self, g: &GizmoGeometry, pointer: egui::Vec2, size: egui::Vec2) -> Option<usize> {
        let p = egui::pos2(pointer.x, pointer.y);
        let mut best: Option<(f32, usize)> = None;
        for axis in 0..3 {
            let distance = if self.tool == Tool::Rotate {
                let points = self.ring_points(g, axis, size);
                points.windows(2).map(|w| point_to_segment(p, w[0], w[1])).fold(f32::MAX, f32::min)
            } else {
                match self.handle_segment(g, axis, size) {
                    Some((a, b)) => point_to_segment(p, a, b),
                    None => f32::MAX,
                }
            };
            if distance < 9.0 && best.is_none_or(|(d, _)| distance < d) {
                best = Some((distance, axis));
            }
        }
        best.map(|(_, axis)| axis)
    }

    fn draw_gizmo(&self, painter: &egui::Painter, rect: egui::Rect, g: &GizmoGeometry, highlighted: Option<usize>) {
        let size = rect.size();
        for axis in 0..3 {
            let lit = highlighted == Some(axis);
            let color = if lit { egui::Color32::WHITE } else { axis_color(axis) };
            let stroke = egui::Stroke::new(if lit { 3.5 } else { 2.5 }, color);
            if self.tool == Tool::Rotate {
                let points: Vec<egui::Pos2> = self.ring_points(g, axis, size).into_iter().map(|p| p + rect.min.to_vec2()).collect();
                painter.add(egui::Shape::line(points, stroke));
            } else if let Some((a, b)) = self.handle_segment(g, axis, size) {
                let (a, b) = (a + rect.min.to_vec2(), b + rect.min.to_vec2());
                painter.line_segment([a, b], stroke);
                if self.tool == Tool::Scale {
                    painter.rect_filled(egui::Rect::from_center_size(b, egui::vec2(9.0, 9.0)), 1.0, color);
                } else {
                    // An arrow head, so a move handle cannot be mistaken for a scale handle.
                    let direction = (b - a).normalized();
                    let side = egui::vec2(-direction.y, direction.x);
                    let tip = b + direction * 10.0;
                    let base = b - direction * 4.0;
                    painter.add(egui::Shape::convex_polygon(vec![tip, base + side * 6.5, base - side * 6.5], color, egui::Stroke::NONE));
                }
            }
        }
        if let Some(c) = self.camera.project(g.origin, [size.x, size.y]) {
            painter.circle_filled(egui::pos2(c[0], c[1]) + rect.min.to_vec2(), 3.5, egui::Color32::WHITE);
        }
    }

    fn apply_drag(&mut self, drag: &Drag, pointer: egui::Pos2, size: egui::Vec2) {
        let g = &drag.geometry;
        let view = [size.x, size.y];
        let origin = self.camera.project(g.origin, view).map(|p| egui::pos2(p[0], p[1]));
        let Some(path) = self.selection.clone() else { return };
        let assembly = self.assembly_index();
        let Some(node) = self.file.assemblies.get_mut(assembly).and_then(|a| node_at_mut(&mut a.shapes, &path)) else { return };
        let is_group = node.kind == "group";
        match drag.tool {
            Tool::Move | Tool::Scale => {
                let tip = [0, 1, 2].map(|k| g.origin[k] + g.axes[drag.axis][k] * g.length);
                let (Some(a), Some(b)) = (origin, drag.view.project(tip, view).map(|p| egui::pos2(p[0], p[1]))) else { return };
                let screen_axis = b - a;
                let pixels_per_unit = (screen_axis.length() as f64 / g.length).max(1e-6);
                let along = ((pointer - drag.start_pointer).dot(screen_axis.normalized()) as f64) / pixels_per_unit;
                let parent_units = along / g.parent.s.max(1e-9);
                if drag.tool == Tool::Move {
                    let direction = into_frame(&g.parent.r, g.axes[drag.axis]);
                    node.position = [0, 1, 2].map(|k| drag.start_position[k] + direction[k] * parent_units);
                } else if is_group {
                    let factor = (along / g.length).exp();
                    node.scale = drag.start_scale.map(|s| (s * factor).max(0.001));
                } else {
                    node.scale[drag.axis] = (drag.start_scale[drag.axis] + 2.0 * parent_units).max(0.001);
                }
            }
            Tool::Rotate => {
                let Some(center) = origin else { return };
                let before = drag.start_pointer - center;
                let after = pointer - center;
                let mut delta = (after.y.atan2(after.x) - before.y.atan2(before.x)) as f64;
                while delta > std::f64::consts::PI {
                    delta -= std::f64::consts::TAU;
                }
                while delta < -std::f64::consts::PI {
                    delta += std::f64::consts::TAU;
                }
                let (forward, ..) = drag.view.basis();
                let angle = if dot(g.axes[drag.axis], forward) < 0.0 { -delta } else { delta };
                let rotated = mat_mul(&drag.start_rotation, &axis_rotation(drag.axis, angle));
                node.rotation = euler_from_matrix(&rotated);
            }
        }
        self.touch();
    }

    // ---- the whole tab ----

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if !ctx.egui_wants_keyboard_input() {
            ctx.input(|i| {
                for (key, tool) in [(egui::Key::W, Tool::Move), (egui::Key::E, Tool::Rotate), (egui::Key::R, Tool::Scale)] {
                    if i.key_pressed(key) && !i.modifiers.command {
                        self.tool = tool;
                    }
                }
            });
            let (undo, redo, delete, frame) = ctx.input(|i| {
                (
                    i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                    i.modifiers.command && (i.key_pressed(egui::Key::Y) || (i.key_pressed(egui::Key::Z) && i.modifiers.shift)),
                    i.key_pressed(egui::Key::Delete),
                    i.key_pressed(egui::Key::F),
                )
            });
            if undo {
                self.undo();
            }
            if redo {
                self.redo();
            }
            if delete {
                self.delete_selected();
            }
            if frame {
                self.frame_pending = true;
                self.submitted = None;
            }
        }
        let (save, open) = ctx.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::S), i.modifiers.command && i.key_pressed(egui::Key::O)));
        if save {
            self.save(ctx.input(|i| i.modifiers.shift));
        }
        if open {
            self.open();
        }
        for dropped in ctx.input(|i| i.raw.dropped_files.clone()) {
            let path = dropped.path().to_path_buf();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("sdf")) {
                self.open_path(path);
            }
        }

        egui::Panel::top("composer toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("New").clicked() {
                    self.new_document();
                }
                if ui.button("Open…").clicked() {
                    self.open();
                }
                if ui.button("Save").clicked() {
                    self.save(false);
                }
                if ui.button("Save as…").clicked() {
                    self.save(true);
                }
                ui.separator();
                if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo")).clicked() {
                    self.undo();
                }
                if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo")).clicked() {
                    self.redo();
                }
                ui.separator();
                for (tool, name, hint) in [(Tool::Move, "Move", "W"), (Tool::Rotate, "Rotate", "E"), (Tool::Scale, "Scale", "R")] {
                    ui.selectable_value(&mut self.tool, tool, name).on_hover_text(format!("Shortcut {hint}"));
                }
                ui.separator();
                ui.selectable_value(&mut self.space, Space::Local, "Local");
                ui.selectable_value(&mut self.space, Space::Global, "Global").on_hover_text("Move along the world's axes");
                ui.separator();
                if ui.button("Frame").on_hover_text("Fit the view to the shape (F)").clicked() {
                    self.frame_pending = true;
                    self.submitted = None;
                }
                let title = self.path.as_ref().and_then(|p| p.file_name()).map_or("untitled".to_string(), |n| n.to_string_lossy().into_owned());
                ui.weak(format!("{title}{}", if self.modified { " *" } else { "" }));
                if !self.status.is_empty() {
                    ui.weak(&self.status);
                }
            });
        });
        egui::Panel::left("composer hierarchy").default_size(280.0).show(ui, |ui| self.hierarchy(ui));
        egui::Panel::right("composer inspector").default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.inspector(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.viewport(ui));
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        self.commit_if_settled(pointer_down);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(c: &Composer) -> usize {
        c.shapes().len()
    }

    #[test]
    fn euler_angles_survive_a_round_trip_through_a_matrix() {
        for angles in [[0.3, -0.5, 1.1], [-1.2, 0.4, 2.9], [0.0, 0.0, 0.0], [3.0, 0.2, -2.0]] {
            let back = euler_from_matrix(&euler_matrix(angles));
            let (a, b) = (euler_matrix(angles), euler_matrix(back));
            for i in 0..3 {
                for j in 0..3 {
                    assert!((a[i][j] - b[i][j]).abs() < 1e-9, "{angles:?}");
                }
            }
        }
    }

    #[test]
    fn shapes_can_be_added_grouped_duplicated_and_deleted() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        c.add_node(new_shape("sphere"));
        assert_eq!(count(&c), 2);
        assert_eq!(c.selection, Some(vec![1]));
        c.group_selected();
        assert_eq!(c.selection, Some(vec![1, 0]));
        assert_eq!(c.shapes()[1].kind, "group");
        c.duplicate_selected();
        assert_eq!(c.shapes()[1].children().len(), 2);
        c.delete_selected();
        assert_eq!(c.shapes()[1].children().len(), 1);
        assert_eq!(c.selection, None);
        c.selection = Some(vec![1, 0]);
        c.duplicate_selected();
        c.selection = Some(vec![1, 1]);
        c.move_selected(false);
        assert_eq!(c.selection, Some(vec![1, 0]));
        c.move_selected(false); // already first
        assert_eq!(c.selection, Some(vec![1, 0]));
    }

    #[test]
    fn a_selected_group_receives_new_shapes() {
        let mut c = Composer::default();
        c.add_node(new_group());
        c.add_node(new_shape("cube"));
        assert_eq!(c.shapes()[0].children().len(), 1);
        assert_eq!(c.selection, Some(vec![0, 0]));
    }

    #[test]
    fn undo_and_redo_step_through_edits() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        c.commit_if_settled(false);
        c.add_node(new_shape("sphere"));
        c.commit_if_settled(false);
        assert_eq!(count(&c), 2);
        c.undo();
        assert_eq!(count(&c), 1);
        c.undo();
        assert_eq!(count(&c), 0);
        c.redo();
        c.redo();
        assert_eq!(count(&c), 2);
    }

    #[test]
    fn an_edit_during_a_drag_is_one_undo_step() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        c.commit_if_settled(false);
        for step in 1..=5 {
            c.shapes_mut().unwrap()[0].position[0] = step as f64;
            c.touch();
            c.commit_if_settled(true); // the pointer is still down
        }
        c.commit_if_settled(false);
        assert_eq!(c.undo.len(), 2);
        c.undo();
        assert_eq!(c.shapes()[0].position[0], 0.0);
    }

    #[test]
    fn dragging_the_move_handle_moves_the_shape_by_the_distance_dragged() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let g = c.gizmo_geometry(rect).unwrap();
        let size = rect.size();
        let start = c.camera.project(g.origin, [size.x, size.y]).unwrap();
        let tip_world = [0, 1, 2].map(|k| g.origin[k] + g.axes[0][k] * g.length);
        let tip = c.camera.project(tip_world, [size.x, size.y]).unwrap();
        let handle = egui::vec2(tip[0] - start[0], tip[1] - start[1]);
        let drag = Drag {
            axis: 0,
            tool: Tool::Move,
            start_pointer: egui::pos2(start[0], start[1]),
            start_position: [0.0; 3],
            start_rotation: IDENTITY,
            start_scale: [1.0; 3],
            geometry: g,
            view: c.camera,
        };
        // Dragging by half the handle's length on screen moves half of its world length.
        let pointer = egui::pos2(start[0], start[1]) + handle * 0.5;
        c.apply_drag(&drag, pointer, size);
        let moved = c.shapes()[0].position[0];
        assert!((moved - g.length * 0.5).abs() < 1e-6 * g.length.max(1.0), "moved {moved}, expected {}", g.length * 0.5);
    }

    #[test]
    fn rotating_about_an_axis_turns_the_shape_by_the_dragged_angle() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        c.tool = Tool::Rotate;
        let g = c.gizmo_geometry(rect).unwrap();
        let size = rect.size();
        let center = c.camera.project(g.origin, [size.x, size.y]).unwrap();
        let center = egui::pos2(center[0], center[1]);
        let drag = Drag {
            axis: 1,
            tool: Tool::Rotate,
            start_pointer: center + egui::vec2(100.0, 0.0),
            start_position: [0.0; 3],
            start_rotation: IDENTITY,
            start_scale: [1.0; 3],
            geometry: g,
            view: c.camera,
        };
        // A quarter turn around the centre of the gizmo on screen.
        c.apply_drag(&drag, center + egui::vec2(0.0, 100.0), size);
        let r = euler_matrix(c.shapes()[0].rotation);
        // About y by 90 degrees, in one direction or the other.
        assert!((r[0][2].abs() - 1.0).abs() < 1e-6 && (r[1][1] - 1.0).abs() < 1e-6, "{r:?}");
    }
}

#[cfg(test)]
mod drag_tests {
    use super::*;

    fn composer_with(names: &[&str]) -> Composer {
        let mut c = Composer::default();
        for name in names {
            c.add_node(new_shape("cube"));
            let last = c.selection.clone().unwrap();
            node_at_mut(c.shapes_mut().unwrap(), &last).unwrap().name = name.to_string();
        }
        c
    }

    fn names(c: &Composer) -> Vec<String> {
        c.shapes().iter().map(|n| n.name.clone()).collect()
    }

    #[test]
    fn dropping_on_the_edges_reorders() {
        let mut c = composer_with(&["a", "b", "c"]);
        c.move_node(&[0], Some(&[2]), DropZone::After);
        assert_eq!(names(&c), ["b", "c", "a"]);
        c.move_node(&[2], Some(&[0]), DropZone::Before);
        assert_eq!(names(&c), ["a", "b", "c"]);
    }

    #[test]
    fn dropping_a_shape_on_a_shape_makes_a_group_of_both() {
        let mut c = composer_with(&["a", "b", "c"]);
        c.move_node(&[0], Some(&[2]), DropZone::Inside);
        assert_eq!(c.shapes().len(), 2);
        assert_eq!(c.shapes()[1].kind, "group");
        let inner: Vec<_> = c.shapes()[1].children().iter().map(|n| n.name.as_str()).collect();
        assert_eq!(inner, ["c", "a"]);
        assert_eq!(c.selection, Some(vec![1, 1]));
    }

    #[test]
    fn shapes_move_into_and_out_of_groups() {
        let mut c = composer_with(&["a", "b"]);
        c.add_node(new_group());
        assert_eq!(c.shapes().len(), 3);
        c.move_node(&[0], Some(&[2]), DropZone::Inside);
        assert_eq!(c.shapes().len(), 2);
        assert_eq!(c.shapes()[1].children().len(), 1);
        c.move_node(&[1, 0], None, DropZone::After);
        assert_eq!(c.shapes().len(), 3);
        assert_eq!(c.shapes()[2].name, "a");
        assert!(c.shapes()[1].children().is_empty());
    }

    #[test]
    fn a_group_cannot_be_dropped_into_itself() {
        let mut c = composer_with(&["a"]);
        c.add_node(new_group());
        c.move_node(&[1], Some(&[1]), DropZone::Inside);
        assert_eq!(c.shapes().len(), 2);
    }
}
