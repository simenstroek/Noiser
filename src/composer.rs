//! The SDF Composer tab: build a shape from SDF primitives the way SDF Modeler does, and save it as a `.sdf` file that
//! the Noise graph and the Mesh graph can load. The document is an `SdfFile`, so what Noiser does not edit is kept.

use crate::composer_gpu::{DEFAULT_STEPS, ReferenceDraw, ReferenceUniforms, Uniforms, ViewCallback};
use crate::composer_reference::{Reference, ReferenceMesh, ReferenceSettings};
use crate::composer_render::{Camera, pick};
use crate::keymap::{Action as Key, pressed};
use crate::sdf_file::{Assembly, Layer, Node, SdfFile};
use crate::sdf_model::{GpuScene, LayerField, M3, V3, euler_matrix, file_scene, into_frame, layer_fields};
use eframe::{egui, egui_wgpu, wgpu};
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
    put("bevel", 0.0.into());
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

/// Where the two-axis move squares sit along their axes, as fractions of the gizmo length.
const PLANE_NEAR: f64 = 0.3;
const PLANE_FAR: f64 = 0.6;

fn inside_quad(corners: &[egui::Pos2; 4], p: egui::Pos2) -> bool {
    let mut sign = 0.0f32;
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross != 0.0 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
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

/// What waits for an answer to "unsaved changes".
#[derive(Clone, Debug)]
enum Pending {
    New,
    Open,
    Path(PathBuf),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Answer {
    Save,
    Discard,
    Cancel,
}

enum Action {
    Select(Vec<usize>),
    ToggleHidden(Vec<usize>),
    ToggleCollapsed(Vec<usize>),
    /// Move a node next to, into, or (onto a shape) together with another; `None` means the end of the top level.
    Drop { from: Vec<usize>, onto: Option<Vec<usize>>, zone: DropZone },
    SetText { path: Vec<usize>, key: &'static str, value: String },
}

struct Row {
    path: Vec<usize>,
    depth: usize,
    label: String,
    mode: String,
    brush: Option<String>,
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
    snap: bool,
    /// Draw the shapes in the colours stored in the file.
    show_colors: bool,
    fields: Option<Arc<Vec<LayerField>>>,
    scene_cache: Option<((u64, Option<Vec<usize>>, usize), Arc<GpuScene>)>,
    scene_serial: u64,
    frame_pending: bool,
    drag: Option<Drag>,
    pending: Option<Pending>,
    /// Samples along the longest side when exporting a mesh.
    export_samples: usize,
    export_job: Option<std::sync::mpsc::Receiver<String>>,
    /// Where the viewport was drawn last (tests aim the pointer with it).
    last_view: egui::Rect,
    /// The picture is rendered at this fraction of the viewport's pixels (1, 0.5 or 0.25).
    render_scale: f32,
    /// The most steps a ray of the viewport may take.
    max_steps: u32,
    /// The model drawn beside the composition to model on top of; not part of the document.
    reference: Option<Reference>,
    reference_load: Option<std::sync::mpsc::Receiver<Result<(PathBuf, ReferenceMesh), String>>>,
    /// The placement a restored session asked for, applied when its mesh has loaded.
    reference_settings_pending: Option<ReferenceSettings>,
    reference_serial: u64,
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
            snap: false,
            show_colors: true,
            fields: None,
            scene_cache: None,
            scene_serial: 0,
            frame_pending: true,
            drag: None,
            pending: None,
            export_samples: 224,
            export_job: None,
            last_view: egui::Rect::NOTHING,
            render_scale: 1.0,
            max_steps: DEFAULT_STEPS,
            reference: None,
            reference_load: None,
            reference_settings_pending: None,
            reference_serial: 0,
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
        self.fields = None;
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
        self.fields = None;
        self.frame_pending = true;
    }

    /// Runs `action`, after asking when the composition has unsaved changes.
    fn request(&mut self, action: Pending) {
        if self.modified {
            self.pending = Some(action);
        } else {
            self.run(action);
        }
    }

    fn run(&mut self, action: Pending) {
        match action {
            Pending::New => self.new_document(),
            Pending::Open => self.open(),
            Pending::Path(path) => self.open_path(path),
        }
    }

    /// The document for the app's storage: its JSON, its path and whether it has unsaved changes. None when empty.
    pub fn storage(&self) -> Option<(String, String, bool)> {
        let empty = self.file.assemblies.iter().all(|a| a.shapes.is_empty());
        (!empty || self.path.is_some()).then(|| (self.file.to_json(), self.path.as_ref().map_or(String::new(), |p| p.to_string_lossy().into_owned()), self.modified))
    }

    /// Puts back what `storage` returned.
    pub fn restore(&mut self, json: &str, path: Option<PathBuf>, modified: bool) {
        if let Ok(file) = SdfFile::from_json(json) {
            self.replace_document(file, path);
            self.modified = modified;
        }
    }

    /// Selects a node if it exists.
    pub fn select(&mut self, path: Vec<usize>) {
        if node_at(self.shapes(), &path).is_some() {
            self.selection = Some(path);
        }
    }

    /// The file name and an asterisk for unsaved changes, for the window title.
    pub fn title(&self) -> String {
        let name = self.path.as_ref().and_then(|p| p.file_name()).map_or("untitled".to_string(), |n| n.to_string_lossy().into_owned());
        format!("{name}{}", if self.modified { " *" } else { "" })
    }

    /// Frames the selected node: its place in the view and a distance that shows it whole.
    fn frame_selection(&mut self, rect: egui::Rect) {
        let Some(g) = self.gizmo_geometry(rect) else { return };
        let Some(path) = self.selection.clone() else { return };
        let Some(node) = node_at(self.shapes(), &path) else { return };
        let size = if node.kind == "group" { 1.0 } else { node.scale[0].max(node.scale[1]).max(node.scale[2]) };
        let reach = (size * 0.75 * g.parent.s).max(0.05);
        self.camera = self.camera.framing([g.origin[0] - reach, g.origin[1] - reach, g.origin[2] - reach], [g.origin[0] + reach, g.origin[1] + reach, g.origin[2] + reach]);
    }

    /// Writes the visible layers as an OBJ or PLY mesh (Z up, like SDF Modeler's exports), on a worker thread.
    fn export_mesh(&mut self) {
        if self.export_job.is_some() {
            return;
        }
        let name = self.path.as_ref().and_then(|p| p.file_stem()).map_or("composition".to_string(), |n| n.to_string_lossy().into_owned());
        let Some(path) = rfd::FileDialog::new().add_filter("OBJ", &["obj"]).add_filter("PLY (binary)", &["ply"]).set_file_name(format!("{name}.obj")).save_file() else { return };
        let (file, samples) = (self.file.clone(), self.export_samples);
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let message = match crate::sdf_mesh::raw_mesh_from_file(&file, samples) {
                Ok(mesh) => {
                    let written = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ply")) { mesh.write_ply(&path) } else { mesh.write_obj(&path) };
                    match written {
                        Ok(()) => format!("Exported {} triangles to {}", mesh.triangles.len(), path.display()),
                        Err(e) => format!("Could not write {}: {e}", path.display()),
                    }
                }
                Err(e) => e,
            };
            let _ = sender.send(message);
        });
        self.export_job = Some(receiver);
        self.status = "Exporting the mesh…".into();
    }

    /// The File and Edit menus of the app's menu bar, while this tab is showing.
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Saves to the file it was opened from or saved to last (a dialog when it has none).
    pub fn save_current(&mut self) {
        self.save(false);
    }

    pub fn set_render_quality(&mut self, scale: f32, steps: u32) {
        self.render_scale = if [1.0, 0.5, 0.25].contains(&scale) { scale } else { 1.0 };
        self.max_steps = steps.clamp(32, 600);
    }

    /// Reads a mesh file on a worker thread and shows it beside the composition once it is ready.
    pub fn load_reference(&mut self, path: PathBuf) {
        self.reference_serial += 1;
        let id = self.reference_serial;
        let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        self.status = format!("Loading {name}…");
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(ReferenceMesh::load(&path, id).map(|mesh| (path, mesh)));
        });
        self.reference_load = Some(receiver);
    }

    fn set_reference(&mut self, path: PathBuf, mesh: ReferenceMesh) {
        let settings = match self.reference_settings_pending.take() {
            Some(settings) => ReferenceSettings { path: path.clone(), ..settings },
            None => ReferenceSettings { path: path.clone(), ..ReferenceSettings::default() },
        };
        let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        self.status = format!("Reference {name}: {} triangles", mesh.triangle_count());
        self.reference = Some(Reference { settings, mesh: Arc::new(mesh) });
        self.frame_pending = true;
    }

    fn poll_reference(&mut self, ctx: &egui::Context) {
        let Some(receiver) = &self.reference_load else { return };
        match receiver.try_recv() {
            Ok(Ok((path, mesh))) => {
                self.reference_load = None;
                self.set_reference(path, mesh);
            }
            Ok(Err(message)) => {
                self.reference_load = None;
                self.reference_settings_pending = None;
                self.status = message;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.reference_load = None,
        }
    }

    /// The reference's placement as a string for the session, empty when there is none.
    pub fn reference_storage(&self) -> String {
        self.reference.as_ref().and_then(|r| serde_json::to_string(&r.settings).ok()).unwrap_or_default()
    }

    pub fn restore_reference(&mut self, json: &str) {
        let Ok(settings) = serde_json::from_str::<ReferenceSettings>(json) else { return };
        if settings.path.is_file() {
            let path = settings.path.clone();
            self.reference_settings_pending = Some(settings);
            self.load_reference(path);
        }
    }

    pub fn set_export_samples(&mut self, samples: usize) {
        self.export_samples = samples.clamp(48, 640);
    }

    pub fn menus(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            if crate::icons::menu_item(ui, "new", "New", "Ctrl+N").clicked() {
                self.request(Pending::New);
                ui.close();
            }
            if crate::icons::menu_item(ui, "open", "Open…", "Ctrl+O").clicked() {
                self.request(Pending::Open);
                ui.close();
            }
            if crate::icons::menu_item(ui, "save", "Save", "Ctrl+S").clicked() {
                self.save(false);
                ui.close();
            }
            if crate::icons::menu_item(ui, "save_as", "Save as…", "Ctrl+Shift+S").clicked() {
                self.save(true);
                ui.close();
            }
            ui.separator();
            if crate::icons::menu_item_enabled(ui, "export", "Export mesh…", "", self.export_job.is_none()).clicked() {
                self.export_mesh();
                ui.close();
            }
        });
        ui.menu_button("Edit", |ui| {
            if crate::icons::menu_item_enabled(ui, "undo", "Undo", "Ctrl+Z", !self.undo.is_empty()).clicked() {
                self.undo();
                ui.close();
            }
            if crate::icons::menu_item_enabled(ui, "redo", "Redo", "Ctrl+Y", !self.redo.is_empty()).clicked() {
                self.redo();
                ui.close();
            }
        });
    }

    fn new_document(&mut self) {
        self.replace_document(blank_file(), None);
        self.status = "New composition".into();
    }

    fn open(&mut self) {
        let Some(path) = crate::library::dialog(crate::library::Kind::Sdf, rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"])).pick_file() else { return };
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
                match crate::library::dialog(crate::library::Kind::Sdf, rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]).set_file_name(name)).save_file() {
                    Some(path) => path,
                    None => return,
                }
            }
        };
        match self.file.save(&path) {
            Ok(()) => {
                self.status = format!("Saved {}", path.display());
                crate::library::request_thumbnail(&path);
                self.path = Some(path);
                self.modified = false;
            }
            Err(e) => self.status = e,
        }
    }

    /// Opens `path` (asking about unsaved changes first).
    pub fn open_request(&mut self, path: PathBuf) {
        self.request(Pending::Path(path));
    }

    /// Adds the visible layers of another file to this composition, each as a group that keeps the layer's placement.
    pub fn insert_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let file = SdfFile::load(path)?;
        let mut groups = Vec::new();
        for layer in file.layers.iter().filter(|l| !l.hidden) {
            let Some(assembly) = file.assemblies.get(layer.assembly) else { continue };
            let mut group = new_group();
            group.name = layer.name.clone();
            group.position = layer.position;
            group.rotation = layer.rotation;
            group.scale = [layer.scale[0]; 3];
            group.children = Some(assembly.shapes.clone());
            groups.push(group);
        }
        if groups.is_empty() {
            return Err("that file has nothing visible to add".into());
        }
        let name = path.file_stem().map_or_else(|| "Inserted".to_string(), |n| n.to_string_lossy().into_owned());
        let node = if groups.len() == 1 {
            let mut only = groups.remove(0);
            only.name = name;
            only
        } else {
            let mut outer = new_group();
            outer.name = name;
            outer.children = Some(groups);
            outer
        };
        self.add_node(node);
        self.status = format!("Added {}", path.display());
        Ok(())
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
        self.fields = None;
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
        // The shape above the deleted one, so a run of deletes can walk up the stack; the first of a list selects the
        // one that took its place, and an emptied group selects the group.
        let remaining = list.len();
        let mut next = parent;
        if index > 0 {
            next.push(index - 1);
            self.selection = Some(next);
        } else if remaining > 0 {
            next.push(0);
            self.selection = Some(next);
        } else if next.is_empty() {
            self.selection = None;
        } else {
            self.selection = Some(next);
        }
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

    fn fields(&mut self) -> Arc<Vec<LayerField>> {
        if self.fields.is_none() {
            self.fields = Some(Arc::new(layer_fields(&self.file)));
        }
        self.fields.clone().unwrap()
    }

    /// The shape tree flattened for the GPU, rebuilt when the document or the selection changed.
    fn scene(&mut self) -> Arc<GpuScene> {
        let key = (self.version, self.selection.clone(), self.layer);
        if let Some((cached, scene)) = &self.scene_cache
            && *cached == key
        {
            return scene.clone();
        }
        let fields = self.fields();
        let scene = Arc::new(file_scene(&fields, self.selection.as_deref().map(|path| (self.layer, path))));
        self.scene_serial += 1;
        self.scene_cache = Some((key, scene.clone()));
        scene
    }

    fn parent_frame(&self, path: &[usize]) -> Xf {
        // The layer's own placement comes first: the scene is drawn in the file's coordinates.
        let mut frame = match self.file.layers.get(self.layer) {
            Some(layer) => Xf { t: layer.position, r: euler_matrix(layer.rotation), s: if layer.scale[0] == 0.0 { 1.0 } else { layer.scale[0] } },
            None => Xf { t: [0.0; 3], r: IDENTITY, s: 1.0 },
        };
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
        let name = ui.label(label);
        let mut widget = egui::DragValue::new(value).speed(speed);
        if let Some(range) = range {
            widget = widget.range(range);
        }
        let response = ui.add(widget).labelled_by(name.id);
        changed = response.changed();
    });
    changed
}

fn vec3(ui: &mut egui::Ui, label: &str, value: &mut [f64; 3], speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let name = ui.label(label);
        for component in value.iter_mut() {
            changed |= ui.add(egui::DragValue::new(component).speed(speed)).labelled_by(name.id).changed();
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
        let name = ui.label(label);
        egui::ComboBox::from_id_salt(("composer choice", key)).selected_text(shown).show_ui(ui, |ui| {
            for (id, option) in options {
                ui.selectable_value(&mut chosen, id.to_string(), *option);
            }
        }).response.labelled_by(name.id);
    });
    if chosen != current {
        node.set(key, Value::from(chosen));
        return true;
    }
    false
}

/// A row of icon toggles for a choice; returns the new id when one is picked.
fn icon_choice(ui: &mut egui::Ui, current: &str, options: &[(&str, &str)], icon_of: fn(&str) -> &'static str, color_of: Option<fn(&str) -> egui::Color32>) -> Option<String> {
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        for (id, label) in options {
            if crate::icons::tinted_button(ui, icon_of(id), label, Some(current == *id), true, color_of.map(|f| f(id))).on_hover_text(*label).clicked() {
                picked = Some(id.to_string());
            }
        }
    });
    picked
}

impl Composer {
    // ---- inspector ----

    fn inspector(&mut self, ui: &mut egui::Ui) {
        let Some(path) = self.selection.clone() else {
            let mut edited = false;
            if let Some(layer) = self.file.layers.get_mut(self.layer) {
                ui.heading("Layer");
                ui.weak("Where this layer sits in the scene. Select a shape in the hierarchy or click one in the view to edit it.");
                ui.horizontal(|ui| {
                    let name = ui.label("Name");
                    edited |= ui.text_edit_singleline(&mut layer.name).labelled_by(name.id).changed();
                });
                edited |= vec3(ui, "Position", &mut layer.position, 0.01);
                let mut degrees = layer.rotation.map(f64::to_degrees);
                if vec3(ui, "Rotation", &mut degrees, 0.5) {
                    layer.rotation = degrees.map(f64::to_radians);
                    edited = true;
                }
                let mut uniform = layer.scale[0];
                if number(ui, "Scale", &mut uniform, 0.01, Some(0.001..=1000.0)) {
                    layer.scale = [uniform; 3];
                    edited = true;
                }
            }
            if edited {
                self.touch();
            }
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
            let name = ui.label("Name");
            edited |= ui.text_edit_singleline(&mut node.name).labelled_by(name.id).changed();
        });
        if is_shape {
            ui.label("Form");
            if let Some(brush) = icon_choice(ui, node.text("brush").unwrap_or("sphere"), &BRUSHES, crate::icons::brush_icon, None) {
                node.set("brush", Value::from(brush));
                edited = true;
            }
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
                let name = ui.label("Color");
                if ui.color_edit_button_rgb(&mut color).labelled_by(name.id).changed() {
                    node.color = color.map(f64::from);
                    edited = true;
                }
            });
        }
        ui.separator();
        ui.label("Operation");
        if let Some(mode) = icon_choice(ui, node.text("mode").unwrap_or("add"), &MODES, crate::icons::mode_icon, Some(crate::icons::mode_color)) {
            node.set("mode", Value::from(mode));
            edited = true;
        }
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
                let brush_name = BRUSHES.iter().find(|(id, _)| *id == brush).map(|(_, name)| name.to_string());
                let label = if !supported {
                    let mut kind = node.kind.replace('_', " ");
                    if let Some(first) = kind.get_mut(0..1) {
                        first.make_ascii_uppercase();
                    }
                    kind
                } else if is_group {
                    if node.name.is_empty() { "Group".to_string() } else { node.name.clone() }
                } else {
                    // A shape is named by what it is; a name given to it follows, greyed in the row's tooltip.
                    match (&brush_name, node.name.as_str()) {
                        (Some(brush), "" | "Stroke") => brush.clone(),
                        (Some(brush), name) if name == brush => brush.clone(),
                        (Some(_), name) => name.to_string(),
                        (None, "") => node.kind.clone(),
                        (None, name) => name.to_string(),
                    }
                };
                out.push(Row { path: path.clone(), depth: path.len() - 1, label, mode: node.text("mode").unwrap_or("add").to_string(), brush: if is_group { None } else { node.text("brush").map(str::to_string) }, hidden: node.hidden, is_group, supported });
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
        let with_splines: Vec<bool> = self.fields().iter().map(|f| f.has_unsupported).collect();
        ui.horizontal(|ui| {
            let names: Vec<String> = self.file.layers.iter().enumerate().map(|(i, l)| if with_splines.get(i).copied().unwrap_or(false) { format!("{} (splines left out)", l.name) } else { l.name.clone() }).collect();
            let heading = ui.label("Layer");
            egui::ComboBox::from_id_salt("composer layer").selected_text(names.get(self.layer).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    if ui.selectable_value(&mut self.layer, i, name).clicked() {
                        self.selection = None;
                        self.fields = None;
                        self.version += 1;
                        self.frame_pending = true;
                    }
                }
            }).response.labelled_by(heading.id);
            let mut toggled = false;
            if let Some(layer) = self.file.layers.get_mut(self.layer) {
                let mut visible = !layer.hidden;
                if ui.checkbox(&mut visible, "Visible").on_hover_text("Show this layer").changed() {
                    layer.hidden = !visible;
                    toggled = true;
                }
            }
            if toggled {
                self.touch();
            }
            if crate::icons::button(ui, "add_layer", "Add layer", None, true).on_hover_text("Add a layer").clicked() {
                self.add_layer();
            }
            if crate::icons::button(ui, "delete_layer", "Delete layer", None, self.file.layers.len() > 1).on_hover_text("Delete this layer").clicked() {
                self.delete_layer();
            }
        });
        if with_splines.get(self.layer).copied().unwrap_or(false) {
            ui.colored_label(crate::editor::ORANGE, "This layer uses splines, which are not supported yet. They are left out, so anything built on them (emboss, deboss, push, avoid, shell) can look wrong.");
        }
        let mut renamed = false;
        if let Some(layer) = self.file.layers.get_mut(self.layer) {
            ui.horizontal(|ui| {
                let name = ui.label("Name");
                renamed = ui.text_edit_singleline(&mut layer.name).labelled_by(name.id).changed();
            });
        }
        if renamed {
            self.touch();
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            for (brush, name) in BRUSHES {
                if crate::icons::button(ui, crate::icons::brush_icon(brush), &format!("Add {}", name.to_lowercase()), None, true).on_hover_text(format!("Add a {}", name.to_lowercase())).clicked() {
                    self.add_node(new_shape(brush));
                }
            }
            if crate::icons::button(ui, "group_row", "Add group", None, true).on_hover_text("Add an empty group").clicked() {
                self.add_node(new_group());
            }
            ui.separator();
            let has = self.selection.is_some();
            if crate::icons::button(ui, "duplicate", "Duplicate", None, has).on_hover_text("Duplicate (Ctrl+D)").clicked() {
                self.duplicate_selected();
            }
            if crate::icons::button(ui, "delete", "Delete", None, has).on_hover_text("Delete (Del)").clicked() {
                self.delete_selected();
            }
            if crate::icons::button(ui, "up", "Up", None, has).on_hover_text("Earlier: applied before its neighbour").clicked() {
                self.move_selected(false);
            }
            if crate::icons::button(ui, "down", "Down", None, has).on_hover_text("Later: applied after its neighbour").clicked() {
                self.move_selected(true);
            }
            if crate::icons::button(ui, "group", "Wrap in group", None, has).on_hover_text("Wrap in a group (Ctrl+G)").clicked() {
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
                        if crate::icons::button(ui, if open { "expanded" } else { "collapsed" }, if open { "Collapse" } else { "Expand" }, None, true).on_hover_text(if open { "Collapse the group" } else { "Expand the group" }).clicked() {
                            actions.push(Action::ToggleCollapsed(row.path.clone()));
                        }
                    } else {
                        ui.add_space(28.0);
                    }
                    let mut visible = !row.hidden;
                    let toggle = ui.add(egui::Checkbox::without_text(&mut visible));
                    toggle.widget_info(|| egui::WidgetInfo::selected(egui::accesskit::Role::CheckBox, true, visible, format!("Visible: {}", row.label)));
                    if toggle.on_hover_text("Show or hide").changed() {
                        actions.push(Action::ToggleHidden(row.path.clone()));
                    }
                    let mode_label = MODES.iter().find(|(id, _)| *id == row.mode).map_or("Add", |(_, label)| label);
                    let mode_button = crate::icons::tinted_button(ui, crate::icons::mode_icon(&row.mode), &format!("Operation: {mode_label}"), None, row.supported, Some(crate::icons::mode_color(&row.mode))).on_hover_text(format!("Operation: {mode_label}. Click to change it."));
                    egui::Popup::menu(&mode_button).id(ui.id().with(("composer mode menu", &row.path))).show(|ui| {
                        for (id, label) in MODES {
                            let item = egui::Button::image_and_text(crate::icons::image(crate::icons::mode_icon(id), 16.0, crate::icons::mode_color(id)), label).selected(row.mode == id);
                            if ui.add(item).clicked() {
                                actions.push(Action::SetText { path: row.path.clone(), key: "mode", value: id.to_string() });
                            }
                        }
                    });
                    if let Some(brush) = &row.brush {
                        let shape_button = crate::icons::button(ui, crate::icons::brush_icon(brush), &format!("Shape: {brush}"), None, true).on_hover_text(format!("Shape: {brush}. Click to change it."));
                        egui::Popup::menu(&shape_button).id(ui.id().with(("composer shape menu", &row.path))).show(|ui| {
                            for (id, label) in BRUSHES {
                                let item = egui::Button::image_and_text(crate::icons::image(crate::icons::brush_icon(id), 16.0, ui.visuals().text_color()), label).selected(brush == id);
                                if ui.add(item).clicked() {
                                    actions.push(Action::SetText { path: row.path.clone(), key: "brush", value: id.to_string() });
                                }
                            }
                        });
                    }
                    let selected = self.selection.as_ref() == Some(&row.path);
                    let text = format!("{}{}", row.label, if row.supported { "" } else { " (not supported)" });
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
                Action::SetText { path, key, value } => {
                    let assembly = self.assembly_index();
                    if let Some(node) = self.file.assemblies.get_mut(assembly).and_then(|a| node_at_mut(&mut a.shapes, &path)) {
                        node.set(key, Value::from(value));
                        self.touch();
                    }
                }
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

    fn reference_menu(&mut self, button: &egui::Response) {
        let mut load = false;
        let mut remove = false;
        let mut frame = false;
        egui::Popup::menu(button).id(button.id.with("menu")).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            ui.set_min_width(280.0);
            if ui.button("Load reference model…").on_hover_text("OBJ, PLY, STL, glTF or FBX").clicked() {
                load = true;
            }
            let Some(reference) = &mut self.reference else {
                ui.weak("Nothing loaded.");
                return;
            };
            let name = reference.settings.path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
            ui.weak(format!("{name}: {} triangles", reference.mesh.triangle_count()));
            ui.checkbox(&mut reference.settings.visible, "Show");
            egui::Grid::new("reference placement").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                let label = ui.label("Opacity");
                ui.add(egui::Slider::new(&mut reference.settings.opacity, 0.05..=1.0)).labelled_by(label.id);
                ui.end_row();
                let label = ui.label("Height");
                ui.add(egui::DragValue::new(&mut reference.settings.height).range(0.05..=200.0).speed(0.01)).labelled_by(label.id);
                ui.end_row();
                ui.label("Feet at");
                ui.horizontal(|ui| {
                    for (axis, value) in reference.settings.offset.iter_mut().enumerate() {
                        ui.add(egui::DragValue::new(value).speed(0.01)).on_hover_text(["x", "y", "z"][axis]);
                    }
                });
                ui.end_row();
                let label = ui.label("Turn");
                ui.add(egui::DragValue::new(&mut reference.settings.yaw).range(-360.0..=360.0).speed(0.5).suffix("°")).labelled_by(label.id);
                ui.end_row();
            });
            ui.horizontal(|ui| {
                if ui.button("Frame all").clicked() {
                    frame = true;
                }
                if ui.button("Remove").clicked() {
                    remove = true;
                }
            });
        });
        if load
            && let Some(path) = rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply", "stl", "gltf", "glb", "fbx"]).pick_file()
        {
            self.load_reference(path);
        }
        if remove {
            self.reference = None;
            self.status.clear();
        }
        if frame {
            self.frame_pending = true;
        }
    }

    fn quality_menu(&mut self, button: &egui::Response) {
        egui::Popup::menu(button).id(button.id.with("menu")).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            ui.set_min_width(260.0);
            ui.horizontal(|ui| {
                ui.label("Render resolution");
                for (scale, name) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                    ui.selectable_value(&mut self.render_scale, scale, name);
                }
            });
            ui.horizontal(|ui| {
                let label = ui.label("Max ray steps");
                ui.add(egui::Slider::new(&mut self.max_steps, 32..=600).logarithmic(true)).labelled_by(label.id);
            });
            ui.weak("Only this viewport. The starting values are in Preferences > Performance.");
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui, format: wgpu::TextureFormat) {
        let rect = ui.available_rect_before_wrap();
        self.last_view = rect;
        crate::library::set_view_rect(rect);
        let response = ui.interact(rect, ui.scope_id().with("composer view"), egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        let size = rect.size();
        let scene = self.scene();
        let reference_bounds = self.reference.as_ref().filter(|r| r.settings.visible).map(|r| r.bounds());
        let frame_bounds = match (scene.bounds, reference_bounds) {
            (Some((lo, hi)), Some((rlo, rhi))) => Some(([0, 1, 2].map(|k| lo[k].min(rlo[k])), [0, 1, 2].map(|k| hi[k].max(rhi[k])))),
            (bounds, None) => bounds,
            (None, bounds) => bounds,
        };
        if self.frame_pending
            && let Some((lo, hi)) = frame_bounds
        {
            self.camera = self.camera.framing(lo, hi);
            self.frame_pending = false;
        }

        // The gizmo under the pointer, or a drag in progress.
        let geometry = self.gizmo_geometry(rect);
        let pointer = response.hover_pos().or_else(|| response.interact_pointer_pos());
        let hovered_axis = match (geometry, pointer, &self.drag) {
            (Some(g), Some(p), None) => self.hit_handle(&g, p - rect.min, size),
            _ => None,
        };
        // A drag starts a few pixels after the button went down: the handle is the one under the press, and the
        // drag is measured from there.
        let press = ui.input(|i| i.pointer.press_origin());
        let grabbed = match (geometry, press) {
            (Some(g), Some(at)) if response.drag_started_by(egui::PointerButton::Primary) => self.hit_handle(&g, at - rect.min, size).map(|axis| (axis, at)),
            _ => None,
        };
        if grabbed.is_some() && ui.input(|i| i.modifiers.alt) {
            // Alt-drag leaves the original where it was and drags a copy.
            self.duplicate_selected();
        }
        if let Some((axis, start)) = grabbed
            && let Some(g) = geometry
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
        }
        if response.clicked_by(egui::PointerButton::Primary)
            && hovered_axis.is_none()
            && let Some(p) = response.interact_pointer_pos()
        {
            let local = p - rect.min;
            let ndc = [2.0 * local.x as f64 / size.x as f64 - 1.0, 1.0 - 2.0 * local.y as f64 / size.y as f64];
            let fields = self.fields();
            match pick(&fields, &self.camera, ndc, size.x as f64 / size.y as f64) {
                Some((layer, path)) => {
                    self.layer = layer.min(self.file.layers.len().saturating_sub(1));
                    self.selection = Some(path);
                }
                None => self.selection = None,
            }
        }

        // Draw: the GPU interprets the flattened tree straight into this rectangle.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(30));
        let fits = scene.shape_count <= crate::composer_gpu::MAX_SHAPES && scene.ops.len() <= crate::composer_gpu::MAX_OPS;
        if fits {
            let aspect = size.x / size.y.max(1.0);
            let uniforms = Uniforms::new(&self.camera, aspect, &scene, self.selection.is_some() && !crate::library::capturing()).with_colors(self.show_colors).with_steps(self.max_steps);
            let reference = self.reference.as_ref().filter(|r| r.settings.visible).map(|r| ReferenceDraw { mesh: r.mesh.clone(), uniforms: ReferenceUniforms::new(&self.camera, aspect, r.model()), opacity: r.settings.opacity });
            let scale = ctx.pixels_per_point() * self.render_scale;
            let pixels = [(size.x * scale).round().max(1.0) as u32, (size.y * scale).round().max(1.0) as u32];
            painter.add(egui_wgpu::Callback::new_paint_callback(rect, ViewCallback { uniforms, scene: scene.clone(), version: self.scene_serial, format, pixels, reference }));
        } else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "This composition is larger than the viewport can draw", egui::FontId::proportional(15.0), egui::Color32::from_rgb(230, 150, 90));
        }
        if self.shapes().is_empty() && self.reference.is_none() {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Add a shape from the hierarchy panel", egui::FontId::proportional(15.0), egui::Color32::from_gray(150));
        }
        let capturing = crate::library::capturing();
        if !capturing {
            self.draw_orientation(&painter, rect);
        }
        painter.text(
            rect.right_bottom() + egui::vec2(-10.0, -8.0),
            egui::Align2::RIGHT_BOTTOM,
            "drag: orbit  ·  shift-drag / right-drag: pan  ·  wheel: zoom  ·  F: frame all  ·  C: frame selection  ·  1 3 7: front, right, top  ·  W E R: tools",
            egui::FontId::proportional(11.0),
            if capturing { egui::Color32::TRANSPARENT } else { egui::Color32::from_gray(120) },
        );
        let active = self.drag.as_ref().map(|d| d.axis);
        if let Some(g) = geometry.filter(|_| !capturing) {
            self.draw_gizmo(&painter, rect, &g, hovered_axis.or(active));
        }
        if hovered_axis.is_some() {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }
    }

    /// Three short axes in the corner that show which way x, y and z point.
    fn draw_orientation(&self, painter: &egui::Painter, rect: egui::Rect) {
        let (_, right, up) = self.camera.basis();
        let origin = rect.left_bottom() + egui::vec2(46.0, -46.0);
        for (axis, label) in ["X", "Y", "Z"].into_iter().enumerate() {
            let mut world = [0.0; 3];
            world[axis] = 1.0;
            let tip = origin + egui::vec2(dot(world, right) as f32, -dot(world, up) as f32) * 30.0;
            painter.line_segment([origin, tip], egui::Stroke::new(2.0, axis_color(axis)));
            painter.text(tip + (tip - origin).normalized() * 8.0, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(11.0), axis_color(axis));
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

    /// The corners on screen of the small square that moves in the plane across `axis` (the other two axes).
    fn plane_corners(&self, g: &GizmoGeometry, axis: usize, size: egui::Vec2) -> Option<[egui::Pos2; 4]> {
        let (u, v) = (g.axes[(axis + 1) % 3], g.axes[(axis + 2) % 3]);
        let (near, far) = (g.length * PLANE_NEAR, g.length * PLANE_FAR);
        let mut corners = [egui::Pos2::ZERO; 4];
        for (corner, (a, b)) in corners.iter_mut().zip([(near, near), (far, near), (far, far), (near, far)]) {
            let world = [0, 1, 2].map(|k| g.origin[k] + u[k] * a + v[k] * b);
            let at = self.camera.project(world, [size.x, size.y])?;
            *corner = egui::pos2(at[0], at[1]);
        }
        Some(corners)
    }

    fn hit_handle(&self, g: &GizmoGeometry, pointer: egui::Vec2, size: egui::Vec2) -> Option<usize> {
        let p = egui::pos2(pointer.x, pointer.y);
        if self.tool == Tool::Move {
            for axis in 0..3 {
                if let Some(corners) = self.plane_corners(g, axis, size)
                    && inside_quad(&corners, p)
                {
                    return Some(4 + axis);
                }
            }
        }
        let mut best: Option<(f32, usize)> = None;
        for axis in 0..4 {
            let distance = if axis == 3 {
                // The middle box of the scale gizmo scales all three axes together.
                if self.tool != Tool::Scale {
                    continue;
                }
                match self.camera.project(g.origin, [size.x, size.y]) {
                    Some(c) => (p - egui::pos2(c[0], c[1])).length(),
                    None => f32::MAX,
                }
            } else if self.tool == Tool::Rotate {
                let points = self.ring_points(g, axis, size);
                points.windows(2).map(|w| point_to_segment(p, w[0], w[1])).fold(f32::MAX, f32::min)
            } else {
                match self.handle_segment(g, axis, size) {
                    Some((a, b)) => point_to_segment(p, a, b),
                    None => f32::MAX,
                }
            };
            // The middle handle wins only when the pointer is really on it, so the axes stay easy to grab.
            let reach = if axis == 3 { 7.0 } else { 9.0 };
            if distance < reach && best.is_none_or(|(d, _)| distance < d || axis == 3) {
                best = Some((distance, axis));
            }
        }
        best.map(|(_, axis)| axis)
    }

    fn draw_gizmo(&self, painter: &egui::Painter, rect: egui::Rect, g: &GizmoGeometry, highlighted: Option<usize>) {
        let size = rect.size();
        if self.tool == Tool::Move {
            for axis in 0..3 {
                if let Some(corners) = self.plane_corners(g, axis, size) {
                    let lit = highlighted == Some(4 + axis);
                    let alpha = if lit { 200 } else { 110 };
                    let base = axis_color(axis);
                    let fill = egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha);
                    let points: Vec<egui::Pos2> = corners.iter().map(|c| *c + rect.min.to_vec2()).collect();
                    painter.add(egui::Shape::convex_polygon(points, fill, egui::Stroke::new(1.5, if lit { egui::Color32::WHITE } else { base })));
                }
            }
        }
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
            let center = egui::pos2(c[0], c[1]) + rect.min.to_vec2();
            if self.tool == Tool::Scale {
                let lit = highlighted == Some(3);
                painter.rect_filled(egui::Rect::from_center_size(center, egui::vec2(11.0, 11.0)), 2.0, if lit { egui::Color32::WHITE } else { egui::Color32::from_gray(210) });
            } else {
                painter.circle_filled(center, 3.5, egui::Color32::WHITE);
            }
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
        let snap = self.snap;
        let snapped = |value: f64, step: f64| if snap { (value / step).round() * step } else { value };
        if drag.tool == Tool::Scale && drag.axis == 3 {
            // Dragging right grows, left shrinks; 100 pixels is a factor of e.
            let factor = ((pointer.x - drag.start_pointer.x) as f64 / 100.0).exp();
            for axis in 0..3 {
                node.scale[axis] = snapped((drag.start_scale[axis] * factor).max(0.001), 0.05).max(0.001);
            }
            if is_group {
                let uniform = node.scale[0];
                node.scale = [uniform; 3];
            }
            self.touch();
            return;
        }
        if drag.tool == Tool::Move && drag.axis >= 4 {
            let normal = g.axes[drag.axis - 4];
            let plane_hit = |at: egui::Pos2| -> Option<V3> {
                let ndc = [2.0 * at.x as f64 / size.x as f64 - 1.0, 1.0 - 2.0 * at.y as f64 / size.y as f64];
                let (from, direction) = drag.view.ray(ndc, size.x as f64 / size.y as f64);
                let facing = dot(direction, normal);
                if facing.abs() < 1e-6 {
                    return None;
                }
                let t = dot([g.origin[0] - from[0], g.origin[1] - from[1], g.origin[2] - from[2]], normal) / facing;
                Some([0, 1, 2].map(|k| from[k] + direction[k] * t))
            };
            let (Some(start), Some(now)) = (plane_hit(drag.start_pointer), plane_hit(pointer)) else { return };
            let world = [0, 1, 2].map(|k| (now[k] - start[k]) / g.parent.s.max(1e-9));
            let local = into_frame(&g.parent.r, world);
            node.position = [0, 1, 2].map(|k| snapped(drag.start_position[k] + local[k], 0.05));
            self.touch();
            return;
        }
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
                    node.position = [0, 1, 2].map(|k| snapped(drag.start_position[k] + direction[k] * parent_units, 0.05));
                } else if is_group {
                    let factor = (along / g.length).exp();
                    node.scale = drag.start_scale.map(|s| (s * factor).max(0.001));
                } else {
                    node.scale[drag.axis] = snapped((drag.start_scale[drag.axis] + 2.0 * parent_units).max(0.001), 0.05).max(0.001);
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
                let angle = snapped(angle, 15f64.to_radians());
                let rotated = mat_mul(&drag.start_rotation, &axis_rotation(drag.axis, angle));
                node.rotation = euler_from_matrix(&rotated);
            }
        }
        self.touch();
    }

    // ---- the whole tab ----

    pub fn show(&mut self, ui: &mut egui::Ui, format: wgpu::TextureFormat) {
        let ctx = ui.ctx().clone();
        egui_extras::install_image_loaders(&ctx);
        if !ctx.egui_wants_keyboard_input() {
            ctx.input(|i| {
                for (action, tool) in [(Key::ToolMove, Tool::Move), (Key::ToolRotate, Tool::Rotate), (Key::ToolScale, Tool::Scale)] {
                    if pressed(i, action) {
                        self.tool = tool;
                    }
                }
            });
            let (undo, redo, delete, frame) = ctx.input(|i| (pressed(i, Key::Undo), pressed(i, Key::Redo), pressed(i, Key::Delete), pressed(i, Key::ComposerFrameAll)));
            if undo {
                self.undo();
            }
            if redo {
                self.redo();
            }
            if delete {
                self.delete_selected();
            }
            let (duplicate, group) = ctx.input(|i| (pressed(i, Key::Duplicate), pressed(i, Key::Group)));
            if duplicate {
                self.duplicate_selected();
            }
            if group {
                self.group_selected();
            }
            if frame {
                self.frame_pending = true;
            }
            // Looking straight down an axis: 1 front, 3 right, 7 top.
            for (action, yaw, pitch) in [(Key::ViewFront, 0.0, 0.0), (Key::ViewRight, std::f64::consts::FRAC_PI_2, 0.0), (Key::ViewTop, self.camera.yaw, 1.5)] {
                if ctx.input(|i| pressed(i, action)) {
                    self.camera.yaw = yaw;
                    self.camera.pitch = pitch;
                }
            }
            if ctx.input(|i| pressed(i, Key::ComposerFrameSelection)) {
                let rect = self.last_view;
                self.frame_selection(rect);
            }
        }
        let (save_as, save, open) = ctx.input(|i| (pressed(i, Key::SaveAs), pressed(i, Key::Save), pressed(i, Key::Open)));
        if save_as {
            self.save(true);
        } else if save {
            self.save(false);
        }
        if open {
            self.request(Pending::Open);
        }
        if ctx.input(|i| pressed(i, Key::New)) {
            self.request(Pending::New);
        }
        for dropped in ctx.input(|i| i.raw.dropped_files.clone()) {
            let path = dropped.path().to_path_buf();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("sdf")) {
                self.request(Pending::Path(path));
            }
        }
        if let Some(action) = self.pending.clone() {
            let mut answer = None;
            egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.label("The composition has changes that are not saved.");
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        answer = Some(Answer::Save);
                    }
                    if ui.button("Discard").clicked() {
                        answer = Some(Answer::Discard);
                    }
                    if ui.button("Cancel").clicked() {
                        answer = Some(Answer::Cancel);
                    }
                });
            });
            match answer {
                Some(Answer::Save) => {
                    self.save(false);
                    if !self.modified {
                        self.pending = None;
                        self.run(action);
                    }
                }
                Some(Answer::Discard) => {
                    self.pending = None;
                    self.modified = false;
                    self.run(action);
                }
                Some(Answer::Cancel) => self.pending = None,
                None => {}
            }
        }

        self.poll_reference(&ctx);
        if let Some(job) = &self.export_job
            && let Ok(message) = job.try_recv()
        {
            self.status = message;
            self.export_job = None;
        }
        if self.export_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        egui::Panel::top("composer toolbar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if crate::icons::button(ui, "new", "New", None, true).on_hover_text("New (Ctrl+N)").clicked() {
                    self.request(Pending::New);
                }
                if crate::icons::button(ui, "open", "Open…", None, true).on_hover_text("Open… (Ctrl+O)").clicked() {
                    self.request(Pending::Open);
                }
                if crate::icons::button(ui, "save", "Save", None, true).on_hover_text("Save (Ctrl+S)").clicked() {
                    self.save(false);
                }
                if crate::icons::button(ui, "save_as", "Save as…", None, true).on_hover_text("Save as… (Ctrl+Shift+S)").clicked() {
                    self.save(true);
                }
                ui.separator();
                if crate::icons::button(ui, "export", "Export mesh…", None, self.export_job.is_none()).on_hover_text("Write the visible layers as an OBJ or PLY mesh").clicked() {
                    self.export_mesh();
                }
                ui.add(egui::DragValue::new(&mut self.export_samples).range(48..=640).speed(2.0).suffix(" samples")).on_hover_text("Samples along the longest side of the mesh extraction: more is finer and slower");
                ui.separator();
                if crate::icons::button(ui, "undo", "Undo", None, !self.undo.is_empty()).on_hover_text("Undo (Ctrl+Z)").clicked() {
                    self.undo();
                }
                if crate::icons::button(ui, "redo", "Redo", None, !self.redo.is_empty()).on_hover_text("Redo (Ctrl+Y)").clicked() {
                    self.redo();
                }
                ui.separator();
                for (tool, name, icon, hint) in [(Tool::Move, "Move", "move", "W"), (Tool::Rotate, "Rotate", "rotate", "E"), (Tool::Scale, "Scale", "scale", "R")] {
                    if crate::icons::button(ui, icon, name, Some(self.tool == tool), true).on_hover_text(format!("{name} ({hint})")).clicked() {
                        self.tool = tool;
                    }
                }
                ui.separator();
                if crate::icons::button(ui, "local", "Local", Some(self.space == Space::Local), true).on_hover_text("Gizmo follows the shape's own axes").clicked() {
                    self.space = Space::Local;
                }
                if crate::icons::button(ui, "global", "Global", Some(self.space == Space::Global), true).on_hover_text("Move along the world's axes").clicked() {
                    self.space = Space::Global;
                }
                if crate::icons::button(ui, "snap", "Snap", Some(self.snap), true).on_hover_text("Move and scale in steps of 0.05, rotate in steps of 15 degrees").clicked() {
                    self.snap = !self.snap;
                }
                if crate::icons::button(ui, "paint", "Colours", Some(self.show_colors), true).on_hover_text("Draw the shapes in the colours stored in the file").clicked() {
                    self.show_colors = !self.show_colors;
                }
                ui.separator();
                if crate::icons::button(ui, "frame", "Frame", None, true).on_hover_text("Fit the view to the scene (F)").clicked() {
                    self.frame_pending = true;
                }
                let reference_button = crate::icons::button(ui, "reference", "Reference model", Some(self.reference.is_some()), true).on_hover_text("A mesh to model on top of, such as a character from ZBrush: shown in the viewport, never saved or exported");
                self.reference_menu(&reference_button);
                let quality_button = crate::icons::button(ui, "quality", "Viewport quality", None, true).on_hover_text("Render resolution and ray steps of this viewport: lower them when a big composition gets slow");
                self.quality_menu(&quality_button);
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
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.viewport(ui, format));
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
        assert_eq!(c.selection, Some(vec![1, 0]));
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
mod reference_tests {
    use super::*;

    const SIZE: u32 = 64;

    fn sphere_reference(radius: f32, centre: V3) -> Reference {
        let mesh = ReferenceMesh::from_mesh(&crate::mesh::shapes::icosphere(radius, 3), 1);
        let settings = ReferenceSettings { height: 2.0 * radius as f64, offset: [centre[0], centre[1] - radius as f64, centre[2]], ..ReferenceSettings::default() };
        Reference { settings, mesh: Arc::new(mesh) }
    }

    /// The centre pixel and a corner pixel of a frame drawn on the GPU, as RGB.
    fn draw(c: &mut Composer, reference: Option<&Reference>) -> Option<([u8; 3], [u8; 3])> {
        let gpu = crate::gpu_tests::gpu()?;
        let scene = c.scene();
        let aspect = 1.0;
        let uniforms = Uniforms::new(&c.camera, aspect, &scene, false).with_colors(false).with_steps(c.max_steps);
        let draw = reference.map(|r| ReferenceDraw { mesh: r.mesh.clone(), uniforms: ReferenceUniforms::new(&c.camera, aspect, r.model()), opacity: r.settings.opacity });
        let mut view = crate::composer_gpu::ComposerGpu::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        view.render(&gpu.device, &gpu.queue, &mut encoder, &scene, 1, uniforms, draw.as_ref(), [SIZE, SIZE]);
        gpu.queue.submit([encoder.finish()]);
        let (_, pixels) = view.read_picture(&gpu.device, &gpu.queue);
        let at = |x: u32, y: u32| {
            let i = ((y * SIZE + x) * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2]]
        };
        Some((at(SIZE / 2, SIZE / 2), at(2, 2)))
    }

    fn is_blue(p: [u8; 3]) -> bool {
        p[2] as i32 > p[0] as i32 + 25
    }

    fn composer_with_a_sphere(size: f64) -> Composer {
        let mut c = Composer::default();
        let mut node = new_shape("sphere");
        node.scale = [size, size, size];
        c.add_node(node);
        c
    }

    #[test]
    fn a_reference_alone_is_drawn_over_the_background() {
        let mut c = Composer::default();
        let reference = sphere_reference(1.0, [0.0; 3]);
        let Some((centre, corner)) = draw(&mut c, Some(&reference)) else { return };
        assert!(is_blue(centre), "centre {centre:?}");
        assert!(!is_blue(corner), "corner {corner:?}");
    }

    #[test]
    fn a_shape_in_front_of_the_reference_hides_it_and_one_behind_does_not() {
        let mut c = composer_with_a_sphere(1.0);
        // Inside the shape: the shape is nearer along every ray.
        let inside = sphere_reference(0.4, [0.0; 3]);
        let Some((centre, _)) = draw(&mut c, Some(&inside)) else { return };
        assert!(!is_blue(centre), "the shape should hide a reference inside it: {centre:?}");
        // Between the camera and the shape.
        let eye = c.camera.eye();
        let towards = |k: f64| [eye[0] * k, eye[1] * k, eye[2] * k];
        let front = sphere_reference(0.4, towards(0.0625));
        let Some((centre, _)) = draw(&mut c, Some(&front)) else { return };
        assert!(is_blue(centre), "the reference in front should show: {centre:?}");
    }

    #[test]
    fn opacity_lets_the_composition_show_through_the_reference() {
        let mut c = composer_with_a_sphere(1.0);
        let eye = c.camera.eye();
        let mut front = sphere_reference(0.4, [eye[0] * 0.0625, eye[1] * 0.0625, eye[2] * 0.0625]);
        let Some((solid, _)) = draw(&mut c, Some(&front)) else { return };
        front.settings.opacity = 0.5;
        let Some((ghost, _)) = draw(&mut c, Some(&front)) else { return };
        let red = |p: [u8; 3]| p[0] as i32;
        assert!(red(ghost) > red(solid) + 15, "half opaque shows more of the grey shape behind: {solid:?} {ghost:?}");
        assert!(is_blue(ghost) || ghost[2] >= ghost[0], "{ghost:?}");
    }

    #[test]
    fn a_session_remembers_the_placement() {
        let mut c = Composer::default();
        let mut reference = sphere_reference(1.0, [0.0; 3]);
        reference.settings.opacity = 0.4;
        reference.settings.yaw = 33.0;
        c.reference = Some(reference);
        let json = c.reference_storage();
        let back: ReferenceSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.opacity, 0.4);
        assert_eq!(back.yaw, 33.0);
        assert!(Composer::default().reference_storage().is_empty());
    }

    #[test]
    fn the_quality_setting_only_takes_known_values() {
        let mut c = Composer::default();
        c.set_render_quality(0.5, 10_000);
        assert_eq!((c.render_scale, c.max_steps), (0.5, 600));
        c.set_render_quality(0.3, 1);
        assert_eq!((c.render_scale, c.max_steps), (1.0, 32));
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
    fn deleting_selects_the_shape_above_so_a_run_of_deletes_walks_up_the_stack() {
        let mut c = composer_with(&["A", "B", "C", "D"]);
        c.selection = Some(vec![3]);
        c.delete_selected();
        assert_eq!(c.selection, Some(vec![2]));
        c.delete_selected();
        assert_eq!(c.selection, Some(vec![1]));
        c.selection = Some(vec![0]);
        c.delete_selected();
        assert_eq!(names(&c), ["B"]);
        assert_eq!(c.selection, Some(vec![0]), "the first one hands the selection to the one that took its place");
        c.delete_selected();
        assert_eq!(c.selection, None);
        assert!(c.shapes().is_empty());
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

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn new_asks_first_when_there_are_unsaved_changes() {
        let mut c = Composer::default();
        c.add_node(new_shape("cube"));
        assert!(c.modified);
        c.request(Pending::New);
        assert!(c.pending.is_some());
        assert_eq!(c.shapes().len(), 1, "nothing happens until the question is answered");
        c.pending = None;
        c.modified = false;
        c.request(Pending::New);
        assert!(c.pending.is_none());
        assert_eq!(c.shapes().len(), 0);
    }

    #[test]
    fn the_document_survives_a_restart() {
        let mut c = Composer::default();
        assert!(c.storage().is_none(), "an empty blank composition is not worth keeping");
        c.add_node(new_shape("sphere"));
        c.add_node(new_shape("cube"));
        let (json, path, modified) = c.storage().unwrap();
        let mut restored = Composer::default();
        restored.restore(&json, (!path.is_empty()).then(|| PathBuf::from(path)), modified);
        assert_eq!(restored.shapes().len(), 2);
        assert!(restored.modified);
    }
}

#[cfg(test)]
mod ui_tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    fn harness() -> Harness<'static, Composer> {
        Harness::builder().with_size(egui::Vec2::new(1200.0, 800.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, composer: &mut Composer| composer.show(ui, wgpu::TextureFormat::Rgba8Unorm),
            Composer::default(),
        )
    }

    /// The first widget with this label (a name can appear twice: a button and a combo's current value).
    fn click(h: &mut Harness<'static, Composer>, label: &str) {
        h.get_all_by_label(label).next().unwrap_or_else(|| panic!("no widget labelled {label}")).click();
        h.run();
    }

    fn press(h: &mut Harness<'static, Composer>, pos: egui::Pos2, pressed: bool) {
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() });
        h.run();
    }

    #[test]
    fn the_add_buttons_add_shapes_and_a_row_selects_one() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        click(&mut h, "Add sphere");
        assert_eq!(h.state().shapes().len(), 2);
        assert_eq!(h.state().shapes()[0].text("brush"), Some("cube"));
        assert_eq!(h.state().shapes()[1].text("brush"), Some("sphere"));
        click(&mut h, "Cube");
        assert_eq!(h.state().selection, Some(vec![0]));
        // The inspector now shows the selected shape.
        h.get_by_label("Shape");
    }

    #[test]
    fn dragging_a_row_onto_another_makes_a_group() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        click(&mut h, "Add sphere");
        let from = h.get_all_by_label("Cube").next().unwrap().rect().center();
        let to = h.get_all_by_label("Sphere").next().unwrap().rect().center();
        h.hover_at(from);
        h.run();
        press(&mut h, from, true);
        for step in 1..=10 {
            h.hover_at(from + (to - from) * step as f32 / 10.0);
            h.run();
        }
        press(&mut h, to, false);
        assert_eq!(h.state().shapes().len(), 1, "the two shapes are now one group: {:?}", h.state().shapes().iter().map(|n| &n.kind).collect::<Vec<_>>());
        assert_eq!(h.state().shapes()[0].kind, "group");
        assert_eq!(h.state().shapes()[0].children().len(), 2);
    }

    #[test]
    fn undo_in_the_toolbar_takes_the_last_edit_back() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        click(&mut h, "Add sphere");
        assert_eq!(h.state().shapes().len(), 2);
        click(&mut h, "Undo");
        assert_eq!(h.state().shapes().len(), 1);
        click(&mut h, "Redo");
        assert_eq!(h.state().shapes().len(), 2);
    }

    #[test]
    fn the_quality_button_lowers_the_viewport_resolution() {
        let mut h = harness();
        h.run();
        click(&mut h, "Viewport quality");
        click(&mut h, "Half");
        assert_eq!(h.state().render_scale, 0.5);
    }

    #[test]
    fn deleting_with_the_toolbar_button_selects_the_shape_above() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        click(&mut h, "Add sphere");
        click(&mut h, "Delete");
        assert_eq!(h.state().shapes().len(), 1);
        assert_eq!(h.state().selection, Some(vec![0]));
    }

    #[test]
    fn duplicate_and_delete_work_on_the_selected_row() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add polygon");
        assert_eq!(h.state().shapes().len(), 1);
        click(&mut h, "Duplicate");
        assert_eq!(h.state().shapes().len(), 2);
        click(&mut h, "Delete");
        assert_eq!(h.state().shapes().len(), 1);
    }

    #[test]
    fn dragging_a_position_value_in_the_inspector_moves_the_shape() {
        use egui::accesskit::Role;
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        let fields: Vec<_> = h.get_all_by_role_and_label(Role::SpinButton, "Position").collect();
        assert_eq!(fields.len(), 3, "x, y and z");
        let from = fields[0].rect().center();
        drop(fields);
        h.hover_at(from);
        h.run();
        press(&mut h, from, true);
        for step in 1..=8 {
            h.hover_at(from + egui::vec2(5.0 * step as f32, 0.0));
            h.run();
        }
        press(&mut h, from + egui::vec2(40.0, 0.0), false);
        let moved = h.state().shapes()[0].position[0];
        assert!(moved > 0.1, "the shape's x moved to {moved}");
        assert_eq!(h.state().shapes()[0].position[1], 0.0);
        assert!(h.state().undo.len() == 2 || h.state().changed, "the move is an undo step (or about to become one)");
    }

    #[test]
    fn the_mirror_checkboxes_set_the_shape_s_mirror_flags() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add sphere");
        assert_eq!(h.state().shapes()[0].flag("mirror_x"), Some(false));
        click(&mut h, "X");
        assert_eq!(h.state().shapes()[0].flag("mirror_x"), Some(true));
        // A blend width appears once a mirror is on.
        h.get_by_label("Mirror blend");
    }

    /// Drags from `from` to `to` with the primary button, in small steps.
    fn drag(h: &mut Harness<'static, Composer>, from: egui::Pos2, to: egui::Pos2) {
        h.hover_at(from);
        h.run();
        press(h, from, true);
        for step in 1..=12 {
            h.hover_at(from + (to - from) * step as f32 / 12.0);
            h.run();
        }
        press(h, to, false);
    }

    fn view_projection(c: &Composer, p: V3) -> egui::Pos2 {
        let rect = c.last_view;
        let q = c.camera.project(p, [rect.width(), rect.height()]).expect("in front of the camera");
        rect.min + egui::vec2(q[0], q[1])
    }

    #[test]
    fn dragging_the_move_handle_in_the_viewport_moves_the_shape() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.run();
        let c = h.state();
        let rect = c.last_view;
        let g = c.gizmo_geometry(rect).expect("the new shape is selected");
        // Grab the X handle halfway along, and drag it along its own screen direction.
        let origin = view_projection(c, g.origin);
        let tip = view_projection(c, [0, 1, 2].map(|k| g.origin[k] + g.axes[0][k] * g.length));
        let grab = origin + (tip - origin) * 0.5;
        let to = grab + (tip - origin) * 0.5;
        drag(&mut h, grab, to);
        let moved = h.state().shapes()[0].position;
        assert!(moved[0] > 0.1, "x moved to {}", moved[0]);
        assert!(moved[1].abs() < 1e-6 && moved[2].abs() < 1e-6, "only x moves: {moved:?}");
    }

    #[test]
    fn dragging_a_plane_square_moves_the_shape_along_two_axes() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.run();
        let c = h.state();
        let rect = c.last_view;
        let g = c.gizmo_geometry(rect).expect("the new shape is selected");
        // The square across the Z axis spans X and Y.
        let middle = (PLANE_NEAR + PLANE_FAR) * 0.5 * g.length;
        let at = |a: f64, b: f64| view_projection(c, [0, 1, 2].map(|k| g.origin[k] + g.axes[0][k] * a + g.axes[1][k] * b));
        let grab = at(middle, middle);
        let to = at(middle + 0.5, middle + 0.5);
        drag(&mut h, grab, to);
        let moved = h.state().shapes()[0].position;
        assert!(moved[0] > 0.1 && moved[1] > 0.1, "x and y both moved: {moved:?}");
        assert!(moved[2].abs() < 1e-6, "z stays: {moved:?}");
    }

    #[test]
    fn alt_dragging_a_handle_moves_a_copy() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.run();
        let c = h.state();
        let g = c.gizmo_geometry(c.last_view).expect("the new shape is selected");
        let origin = view_projection(c, g.origin);
        let tip = view_projection(c, [0, 1, 2].map(|k| g.origin[k] + g.axes[0][k] * g.length));
        let grab = origin + (tip - origin) * 0.5;
        let to = grab + (tip - origin) * 0.5;
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::ALT));
        h.run();
        drag(&mut h, grab, to);
        let shapes = h.state().shapes();
        assert_eq!(shapes.len(), 2, "a copy was made");
        assert!(shapes[0].position[0].abs() < 1e-6, "the original stays: {:?}", shapes[0].position);
        assert!(shapes[1].position[0] > 0.1, "the copy moved: {:?}", shapes[1].position);
    }

    #[test]
    fn dragging_empty_space_orbits_the_camera() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        let before = h.state().camera;
        let rect = h.state().last_view;
        let from = rect.min + egui::vec2(rect.width() * 0.2, rect.height() * 0.8);
        drag(&mut h, from, from + egui::vec2(80.0, 0.0));
        let after = h.state().camera;
        assert!((after.yaw - before.yaw).abs() > 0.3, "yaw {} -> {}", before.yaw, after.yaw);
        assert_eq!(after.distance, before.distance);
    }

    #[test]
    fn clicking_a_shape_in_the_viewport_selects_it() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add sphere");
        click(&mut h, "Add cube");
        h.run();
        // Frame the view on the shapes, then click the middle of it.
        h.state_mut().selection = None;
        h.run();
        h.run();
        let rect = h.state().last_view;
        let center = view_projection(h.state(), h.state().camera.target);
        assert!(rect.contains(center));
        h.hover_at(center);
        h.run();
        press(&mut h, center, true);
        press(&mut h, center, false);
        assert!(h.state().selection.is_some(), "a shape sits at the middle of the framed view");
    }

    #[test]
    fn snapping_rounds_a_dragged_position_to_the_grid() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.state_mut().snap = true;
        h.run();
        let c = h.state();
        let g = c.gizmo_geometry(c.last_view).unwrap();
        let origin = view_projection(c, g.origin);
        let tip = view_projection(c, [0, 1, 2].map(|k| g.origin[k] + g.axes[0][k] * g.length));
        let grab = origin + (tip - origin) * 0.5;
        drag(&mut h, grab, grab + (tip - origin) * 0.31);
        let x = h.state().shapes()[0].position[0];
        assert!(x > 0.05, "moved: {x}");
        assert!(((x / 0.05).round() * 0.05 - x).abs() < 1e-9, "{x} is a multiple of 0.05");
    }

    #[test]
    fn the_middle_box_of_the_scale_gizmo_scales_every_axis() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.state_mut().tool = Tool::Scale;
        h.run();
        let c = h.state();
        let g = c.gizmo_geometry(c.last_view).unwrap();
        let middle = view_projection(c, g.origin);
        drag(&mut h, middle, middle + egui::vec2(70.0, 0.0));
        let scale = h.state().shapes()[0].scale;
        assert!(scale[0] > 1.5 && (scale[0] - scale[1]).abs() < 1e-9 && (scale[1] - scale[2]).abs() < 1e-9, "{scale:?}");
    }

    #[test]
    fn with_nothing_selected_the_inspector_edits_the_layer() {
        use egui::accesskit::Role;
        let mut h = harness();
        h.run();
        let fields: Vec<_> = h.get_all_by_role_and_label(Role::SpinButton, "Position").collect();
        assert_eq!(fields.len(), 3);
        let from = fields[1].rect().center();
        drop(fields);
        drag(&mut h, from, from + egui::vec2(40.0, 0.0));
        assert!(h.state().file.layers[0].position[1] > 0.1, "{:?}", h.state().file.layers[0].position);
        assert_eq!(h.state().file.layers[0].position[0], 0.0);
    }

    #[test]
    fn control_d_duplicates_the_selected_shape() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add sphere");
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::D);
        h.run();
        assert_eq!(h.state().shapes().len(), 2);
    }

    #[test]
    fn x_and_delete_both_delete_the_selected_shape() {
        for key in [egui::Key::X, egui::Key::Delete] {
            let mut h = harness();
            h.run();
            click(&mut h, "Add cube");
            click(&mut h, "Add sphere");
            h.key_press(key);
            h.run();
            assert_eq!(h.state().shapes().len(), 1, "{key:?}");
        }
    }

    #[test]
    fn the_visibility_checkbox_hides_a_shape_and_c_frames_the_selection() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        click(&mut h, "Add sphere");
        assert!(!h.state().shapes()[0].hidden);
        click(&mut h, "Visible: Cube");
        assert!(h.state().shapes()[0].hidden, "the cube is hidden");
        assert!(!h.state().shapes()[1].hidden);
        // Move the sphere away, then C brings the camera to it.
        h.state_mut().shapes_mut().unwrap()[1].position = [30.0, 0.0, 0.0];
        h.state_mut().touch();
        h.run();
        h.key_press(egui::Key::C);
        h.run();
        let target = h.state().camera.target;
        assert!((target[0] - 30.0).abs() < 1e-6, "the camera looks at the sphere: {target:?}");
    }

    #[test]
    fn the_window_title_follows_the_composition() {
        let mut c = Composer::default();
        assert_eq!(c.title(), "untitled");
        c.add_node(new_shape("cube"));
        assert_eq!(c.title(), "untitled *");
    }

    #[test]
    fn saving_into_the_library_asks_for_a_picture() {
        let _turn = crate::library::TURN.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join("noiser_composer_library");
        crate::library::create_folders(&root).unwrap();
        crate::library::set_root(Some(root.clone()));
        crate::library::set_thumbnails(true);
        crate::library::take_requests();
        let mut c = Composer::default();
        c.add_node(new_shape("sphere"));
        c.path = Some(root.join("SDFs").join("ball.sdf"));
        c.save(false);
        let asked = crate::library::take_requests();
        crate::library::set_root(None);
        assert_eq!(asked, vec![root.join("SDFs").join("ball.sdf")]);
        assert!(root.join("SDFs").join("ball.sdf").is_file());
    }

    #[test]
    fn a_layer_with_splines_says_so() {
        let mut h = harness();
        h.run();
        let mut spline = new_shape("sphere");
        spline.kind = "spline".into();
        h.state_mut().add_node(spline);
        h.state_mut().add_node(new_shape("sphere"));
        h.run();
        h.get_by_label_contains("uses splines");
    }

    #[test]
    fn the_number_keys_look_down_an_axis() {
        let mut h = harness();
        h.run();
        h.key_press(egui::Key::Num3);
        h.run();
        assert!((h.state().camera.yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-9 && h.state().camera.pitch == 0.0);
        h.key_press(egui::Key::Num7);
        h.run();
        assert!(h.state().camera.pitch > 1.4);
    }

    #[test]
    fn the_outliner_dropdowns_change_the_operation_and_the_shape() {
        let mut h = harness();
        h.run();
        click(&mut h, "Add cube");
        h.state_mut().selection = None;
        h.run();
        click(&mut h, "Operation: Add");
        click(&mut h, "Subtract");
        assert_eq!(h.state().shapes()[0].text("mode"), Some("sub"));
        click(&mut h, "Shape: cube");
        click(&mut h, "Cylinder");
        assert_eq!(h.state().shapes()[0].text("brush"), Some("cylinder"));
    }

    #[test]
    fn inserting_a_file_adds_a_group_per_layer_with_its_placement() {
        let path = std::env::temp_dir().join("noiser_insert_test.sdf");
        let mut file = blank_file();
        file.assemblies[0].shapes.push(new_shape("sphere"));
        file.layers[0].name = "First".into();
        let mut second = new_layer("Second", 0);
        second.position = [4.0, 0.0, 0.0];
        second.scale = [2.0; 3];
        file.layers.push(second);
        file.save(&path).unwrap();
        let mut c = Composer::default();
        c.insert_file(&path).unwrap();
        let outer = &c.shapes()[0];
        assert_eq!(outer.kind, "group");
        assert_eq!(outer.name, "noiser_insert_test");
        let layers = outer.children();
        assert_eq!(layers.len(), 2);
        assert_eq!((layers[0].name.as_str(), layers[0].position), ("First", [0.0; 3]));
        assert_eq!((layers[1].name.as_str(), layers[1].position, layers[1].scale), ("Second", [4.0, 0.0, 0.0], [2.0; 3]));
        assert_eq!(layers[1].children().len(), 1);
    }
}
