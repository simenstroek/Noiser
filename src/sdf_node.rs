//! The Load SDF node: a `.sdf` composition (made in the SDF Composer tab) as a field in the Noise graph and the Mesh graph.
//! The node outputs the shape's signed distance (negative inside) for the position it reads.
//!
//! The graph's shader gets its own copy of the SDF interpreter (`composer_sdf_lib.wgsl`, every name prefixed so several
//! nodes coexist) and the composition's flattened shapes as private arrays, so nothing but the graph's own bindings is
//! needed: previews, exports and shader export all keep working. Size and offset are graph parameters, so editing them
//! does not recompile the shader; changing the file does.

use crate::composer_gpu::{MAX_OPS, MAX_SHAPES, SDF_LIBRARY};
use crate::sdf_file::SdfFile;
use crate::sdf_model::{GPU_SHAPE_STRIDE, GpuScene, file_scene, layer_fields};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadSdfParams {
    pub file: String,
    /// Scale the shape so its longest side is `size` and put its centre at `offset`, in the unit tile. Off: the shape
    /// keeps the units of its file and its own origin.
    pub fit: bool,
    pub size: f32,
    pub offset: [f32; 3],
    /// Output the height of the shape's top surface at each (x, y) instead of its distance: how a shape goes into a
    /// height field (the Landscape tab). Where the shape is not, the output is -1000, so a Maximum with the ground ignores it.
    pub stamp: bool,
    /// Beside a terrain (a Load Mesh with a heightmap): stand the shape on it. The offset's height is ignored: the shape's
    /// lowest point sits on the lowest ground under its footprint, sunk by `sink` times its size.
    pub snap: bool,
    pub sink: f32,
    /// The file's up (+Y, as the Composer shows it) stands along the field's +Z: for terrain graphs, where z is up.
    pub z_up: bool,
}

impl Default for LoadSdfParams {
    fn default() -> Self {
        LoadSdfParams { file: String::new(), fit: true, size: 0.8, offset: [0.5; 3], stamp: false, snap: false, sink: 0.05, z_up: false }
    }
}

/// A loaded composition: its flattened shapes and the box they fill.
pub struct SdfData {
    pub scene: GpuScene,
}

impl LoadSdfParams {
    /// The seven numbers the shader reads: offset (3), tile units per file unit, and the point of the shape's own
    /// coordinates that lands on the offset (3). Seven floats in all.
    pub fn shader_params(&self, data: Option<&SdfData>) -> [f32; 7] {
        let bounds = data.and_then(|d| d.scene.bounds);
        match (self.fit, bounds) {
            (true, Some((lo, hi))) => {
                let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-9);
                let center = [0, 1, 2].map(|a| ((lo[a] + hi[a]) * 0.5) as f32);
                [self.offset[0], self.offset[1], self.offset[2], (self.size as f64 / longest) as f32, center[0], center[1], center[2]]
            }
            _ => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl LoadSdfParams {
    /// The box the shape fills in the field's own coordinates (after fitting), when the file can be read.
    pub fn field_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let loaded = load(&self.file);
        let data = loaded.as_ref().as_ref().ok()?;
        let (lo, hi) = data.scene.bounds?;
        let [ox, oy, oz, scale, cx, cy, cz] = self.shader_params(Some(data));
        let (offset, center) = ([ox, oy, oz], [cx, cy, cz]);
        let (mut a, mut b) = ([f32::MAX; 3], [f32::MIN; 3]);
        for corner in 0..8 {
            let p = [0, 1, 2].map(|k| (if corner >> k & 1 == 0 { lo[k] } else { hi[k] } as f32 - center[k]) * scale);
            let rel = if self.z_up { [p[0], -p[2], p[1]] } else { p };
            for k in 0..3 {
                a[k] = a[k].min(offset[k] + rel[k]);
                b[k] = b[k].max(offset[k] + rel[k]);
            }
        }
        Some((a, b))
    }
}

// ---- loading ----------------------------------------------------------------------------------------------------

struct Entry {
    checked: Instant,
    modified: Option<SystemTime>,
    result: Arc<Result<SdfData, String>>,
}

static CACHE: Mutex<Option<HashMap<String, Entry>>> = Mutex::new(None);

/// How often a file is looked at again for changes.
const RECHECK: Duration = Duration::from_millis(750);

fn read(path: &str) -> Result<SdfData, String> {
    let file = SdfFile::load(std::path::Path::new(path))?;
    let fields = layer_fields(&file);
    let scene = file_scene(&fields, None);
    if scene.shape_count == 0 {
        return Err("this .sdf file has nothing to draw: no visible shapes (splines are not supported)".into());
    }
    if scene.shape_count > MAX_SHAPES || scene.ops.len() > MAX_OPS {
        return Err(format!("this composition is too large for a graph ({} shapes, {} operations; the limits are {MAX_SHAPES} and {MAX_OPS})", scene.shape_count, scene.ops.len()));
    }
    Ok(SdfData { scene })
}

/// The folders of the graph files that are open (one slot per tab): a relative path in a Load SDF node is looked for
/// in each of them, so a graph and its shapes can be moved or shared together.
static BASES: Mutex<[Option<std::path::PathBuf>; 2]> = Mutex::new([None, None]);

pub const NOISE_GRAPH_BASE: usize = 0;
pub const MESH_GRAPH_BASE: usize = 1;

pub fn set_base(slot: usize, folder: Option<&std::path::Path>) {
    if let Ok(mut bases) = BASES.lock() {
        bases[slot] = folder.map(|f| f.to_path_buf());
    }
}

/// `path` as it is on disk: a relative path is taken from the first graph folder that has the file.
pub fn resolve(path: &str) -> String {
    let given = std::path::Path::new(path);
    if given.is_absolute() || path.is_empty() {
        return path.to_string();
    }
    let bases = BASES.lock().map(|b| b.clone()).unwrap_or_default();
    bases.iter().flatten().map(|base| base.join(given)).find(|candidate| candidate.is_file()).map_or_else(|| path.to_string(), |found| found.to_string_lossy().into_owned())
}

/// The composition in `path`, read again when the file changes. Cheap to call every frame.
pub fn load(path: &str) -> Arc<Result<SdfData, String>> {
    let resolved = resolve(path);
    let path = resolved.as_str();
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(entry) = cache.get_mut(path)
        && entry.checked.elapsed() < RECHECK
    {
        return entry.result.clone();
    }
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    if let Some(entry) = cache.get_mut(path)
        && entry.modified == modified
    {
        entry.checked = Instant::now();
        return entry.result.clone();
    }
    let result = Arc::new(read(path));
    cache.insert(path.to_string(), Entry { checked: Instant::now(), modified, result: result.clone() });
    result
}

/// Forgets what is known about a file, so the next `load` reads it.
pub fn reload(path: &str) {
    let path = resolve(path);
    if let Some(cache) = CACHE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        cache.remove(&path);
    }
}

// ---- the shader text ---------------------------------------------------------------------------------------------

/// The names the library defines (functions and constants) plus the three it expects from its host.
fn library_names() -> BTreeSet<String> {
    let mut names: BTreeSet<String> = ["shapes", "ops", "sdf_op_count"].iter().map(|s| s.to_string()).collect();
    for line in SDF_LIBRARY.lines() {
        let line = line.trim_start();
        let rest = line.strip_prefix("fn ").or_else(|| line.strip_prefix("const "));
        if let Some(rest) = rest {
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                names.insert(name);
            }
        }
    }
    names
}

/// `text` with every identifier in `names` given `prefix`.
fn prefix_identifiers(text: &str, names: &BTreeSet<String>, prefix: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4096);
    let mut chars = text.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            let mut end = start + c.len_utf8();
            while let Some(&(i, next)) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    end = i + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let word = &text[start..end];
            if names.contains(word) {
                out.push_str(prefix);
            }
            out.push_str(word);
        } else if c.is_ascii_digit() {
            // A number, possibly with an exponent or a suffix: copy it whole so `1e-5` or `12u` is never split.
            let mut end = start + 1;
            while let Some(&(i, next)) = chars.peek() {
                if next.is_alphanumeric() || next == '.' || next == '_' {
                    end = i + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            out.push_str(&text[start..end]);
        } else {
            out.push(c);
        }
    }
    out
}

fn write_array(out: &mut String, name: &str, values: &[[f32; 4]]) {
    let count = values.len().max(1);
    let _ = write!(out, "var<private> {name}: array<vec4f, {count}> = array<vec4f, {count}>(");
    if values.is_empty() {
        out.push_str("vec4f(0.0)");
    }
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "vec4f({:?},{:?},{:?},{:?})", v[0], v[1], v[2], v[3]);
    }
    out.push_str(");\n");
}

/// The WGSL one Load SDF node needs: the interpreter under `prefix`, the scene's arrays, and
/// `{prefix}sdf_map(p)`, the distance in the composition's own coordinates.
pub fn embed(prefix: &str, scene: &GpuScene, z_up: bool) -> String {
    debug_assert_eq!(scene.shapes.len(), scene.shape_count * GPU_SHAPE_STRIDE);
    let names = library_names();
    let mut out = String::new();
    write_array(&mut out, &format!("{prefix}shapes"), &scene.shapes);
    write_array(&mut out, &format!("{prefix}ops"), &scene.ops);
    let _ = writeln!(out, "fn {prefix}sdf_op_count() -> u32 {{ return {}u; }}", scene.ops.len());
    out.push_str(&prefix_identifiers(SDF_LIBRARY, &names, prefix));
    // The field's point to the composition's: undo the placement (offset, scale), and the up axis when asked.
    let up = if z_up { "vec3f(r.x, r.z, -r.y)" } else { "r" };
    let _ = writeln!(out, "fn {prefix}at(f: vec3f, off: vec3f, s: f32, c: vec3f) -> f32 {{ let r = (f - off) / s; return {prefix}map(c + {up}) * s; }}");
    out
}

// ---- the node's settings ------------------------------------------------------------------------------------------

pub fn props(ui: &mut eframe::egui::Ui, p: &mut LoadSdfParams) {
    use eframe::egui;
    ui.weak("A composition from the SDF Composer as a field: its signed distance, negative inside. It is evaluated live in the graph, so its detail is not limited by a grid.");
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut p.file).hint_text("a .sdf file").desired_width(150.0).id_salt("load sdf path"));
        if ui.button("Browse…").clicked() {
            let mut dialog = rfd::FileDialog::new().add_filter("SDF Modeler file", &["sdf"]);
            if let Some(dir) = std::path::Path::new(&p.file).parent().filter(|d| d.is_dir()) {
                dialog = dialog.set_directory(dir);
            }
            if let Some(picked) = dialog.pick_file() {
                p.file = picked.to_string_lossy().into_owned();
            }
        }
        if ui.button("Reload").on_hover_text("Read the file again (it is also read again when it changes)").clicked() {
            reload(&p.file);
        }
    });
    if p.file.is_empty() {
        ui.weak("No file chosen: the node outputs a large positive distance.");
    } else {
        match &*load(&p.file) {
            Ok(data) => {
                ui.weak(format!("{} shapes", data.scene.shape_count));
            }
            Err(e) => {
                ui.colored_label(crate::editor::ORANGE, e);
            }
        }
    }
    ui.checkbox(&mut p.z_up, "Z is up (terrain graphs)").on_hover_text("The Composer's up is +Y. Terrain, and the Landscape tab, have +Z up: tick this to stand the shape upright there.");
    ui.checkbox(&mut p.fit, "Fit to the tile").on_hover_text("Scale the shape so its longest side is the size below and put its centre at the offset. Off: the shape keeps the units and the origin of its file.");
    if p.fit {
        ui.horizontal(|ui| {
            ui.label("size");
            // Terrain graphs work in metres or centimetres: no ceiling, and the step follows the size.
            let step = (p.size * 0.01).max(0.005);
            ui.add(egui::DragValue::new(&mut p.size).range(1e-4..=1e8).speed(step));
        });
        ui.horizontal(|ui| {
            ui.label("offset");
            let step = (p.size * 0.01).max(0.005);
            for v in &mut p.offset {
                ui.add(egui::DragValue::new(v).speed(step));
            }
        });
    }
    if p.fit {
        ui.checkbox(&mut p.snap, "Stand on the terrain").on_hover_text("With a terrain in the graph (a Load Mesh node with a heightmap): the shape's lowest point sits on the ground under it, so only x and y of the offset matter. Without one it does nothing.");
        if p.snap {
            ui.horizontal(|ui| {
                ui.label("sink");
                ui.add(egui::DragValue::new(&mut p.sink).range(-0.5..=0.5).speed(0.005).suffix(" x size")).on_hover_text("How far into the ground the shape is pushed, as a fraction of its size.");
            });
        }
    }
    ui.checkbox(&mut p.stamp, "Height from above (stamp)").on_hover_text("Output the height of the shape's top surface at each x, y instead of its distance. This is how a shape goes into the Landscape tab's height graph: combine it with the terrain by a Maximum node (where the shape is not, the output is -1000, so the terrain wins). Heights are in the graph's own units: with the default fit the shape fills 0.1 to 0.9 of the tile, up from 0.");
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const SPHERE: &str = r#"{"version":[0,6,1],"environment":{},
        "assemblies":[{"resolution":5,"shapes":[{"name":"","hidden":false,"type":"shape","position":[0,0,0],"rotation":[0,0,0],
          "scale":[2,2,2],"color":[1,1,1],"mode":"add","chamfer":false,"blend":0,"blend_strength":0,"mirror_x":false,"mirror_y":false,
          "mirror_z":false,"mirror_blend":0,"brush":"sphere","bevel":0,"round":0.1,"thickness":1,"cone":0,"star":0,"points":6,
          "repeat_mode":"linear","linear_repeats":[1,1,1],"steps":[1.5,1.5,1.5],"radial_repeats":1,"radius":1.5,"repeat_axis":"x"}]}],
        "layers":[{"name":"L","assembly":0,"hidden":false,"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}]}"#;

    pub(crate) fn temp_sphere(name: &str) -> String {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, SPHERE).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn identifiers_of_the_library_get_the_prefix_and_nothing_else() {
        let names = library_names();
        for expected in ["shapes", "ops", "sdf_op_count", "map", "uber", "combine", "PI", "STRIDE", "shape_distance"] {
            assert!(names.contains(expected), "{expected}");
        }
        let text = prefix_identifiers("let x = map(p) * 1e-5 + 12u + shapes[0u].x + sign(y);", &names, "a_");
        assert_eq!(text, "let x = a_map(p) * 1e-5 + 12u + a_shapes[0u].x + sign(y);");
    }

    #[test]
    fn the_file_cache_reads_a_file_and_reports_a_missing_one() {
        let path = temp_sphere("noiser_sdf_node_cache.sdf");
        let loaded = load(&path);
        let data = loaded.as_ref().as_ref().unwrap();
        assert_eq!(data.scene.shape_count, 1);
        let missing = load(&std::env::temp_dir().join("noiser_no_such_file.sdf").to_string_lossy());
        assert!(missing.as_ref().is_err());
    }

    #[test]
    fn a_relative_path_is_found_in_the_graph_folder() {
        let dir = std::env::temp_dir().join("noiser_sdf_relative");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ball.sdf"), SPHERE).unwrap();
        set_base(NOISE_GRAPH_BASE, Some(&dir));
        let found = resolve("ball.sdf");
        let loaded = load("ball.sdf");
        set_base(NOISE_GRAPH_BASE, None);
        assert_eq!(std::path::PathBuf::from(&found), dir.join("ball.sdf"));
        assert!(loaded.is_ok(), "the file loads through its relative name");
        assert_eq!(resolve("ball.sdf"), "ball.sdf", "without a graph folder the name stays as it is");
    }

    #[test]
    fn the_field_bounds_follow_the_fit() {
        let path = temp_sphere("noiser_sdf_node_bounds.sdf");
        let loose = LoadSdfParams { file: path.clone(), fit: false, ..Default::default() }.field_bounds().unwrap();
        for a in 0..3 {
            assert!((loose.0[a] + 1.0).abs() < 0.05 && (loose.1[a] - 1.0).abs() < 0.05, "a sphere of size 2: {loose:?}");
        }
        let fitted = LoadSdfParams { file: path, fit: true, size: 0.8, offset: [0.5; 3], ..Default::default() }.field_bounds().unwrap();
        for a in 0..3 {
            assert!((fitted.0[a] - 0.1).abs() < 0.02 && (fitted.1[a] - 0.9).abs() < 0.02, "inside the tile: {fitted:?}");
        }
    }

    #[test]
    fn fitting_puts_the_shape_in_the_tile() {
        let path = temp_sphere("noiser_sdf_node_fit.sdf");
        let loaded = load(&path);
        let data = loaded.as_ref().as_ref().unwrap();
        let params = LoadSdfParams::default().shader_params(Some(data));
        // The sphere's longest side (2, grown a little by the bounds' margin) becomes 0.8 of the tile.
        let (lo, hi) = data.scene.bounds.unwrap();
        let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
        assert!(((params[3] as f64) * longest - 0.8).abs() < 1e-4);
        assert_eq!(&params[..3], &[0.5, 0.5, 0.5]);
        let unfitted = LoadSdfParams { fit: false, ..Default::default() }.shader_params(Some(data));
        assert_eq!(unfitted, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    fn graph_with_sphere(path: &str, params: LoadSdfParams) -> crate::graph::Graph {
        use crate::graph::{Graph, NodeKind};
        let mut g = Graph::default();
        let position = g.add(NodeKind::Position { scale: [1.0; 3] }, [0.0; 2]);
        let node = g.add(NodeKind::LoadSdf(LoadSdfParams { file: path.to_string(), ..params }), [0.0; 2]);
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(position, node, 0);
        g.connect(node, out, 0);
        g
    }

    #[test]
    fn a_graph_with_a_load_sdf_node_compiles_to_valid_wgsl() {
        let path = temp_sphere("noiser_sdf_node_wgsl.sdf");
        let g = graph_with_sphere(&path, LoadSdfParams::default());
        let compiled = crate::codegen::compile(&g).unwrap();
        assert!(compiled.wgsl.contains("sdf1_map(") && compiled.wgsl.contains("var<private> sdf1_shapes"), "{}", &compiled.wgsl[..compiled.wgsl.len().min(400)]);
        let src = format!(
            "{}
{}
{}
@fragment fn fs() -> @location(0) vec4f {{ return vec4f(graph_normal(vec3f(0.5)), graph_density(vec3f(0.5))); }}",
            crate::codegen::NOISE_LIB,
            compiled.wgsl,
            crate::codegen::EVAL_BINDINGS
        );
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        // A missing file leaves a far-away distance and no interpreter in the shader.
        let missing = graph_with_sphere("no/such/file.sdf", LoadSdfParams::default());
        assert!(!crate::codegen::compile(&missing).unwrap().wgsl.contains("var<private>"));
    }

    #[test]
    fn editing_size_or_offset_does_not_change_the_shader() {
        let path = temp_sphere("noiser_sdf_node_params.sdf");
        let a = crate::codegen::compile(&graph_with_sphere(&path, LoadSdfParams::default())).unwrap();
        let b = crate::codegen::compile(&graph_with_sphere(&path, LoadSdfParams { size: 0.5, offset: [0.2, 0.3, 0.4], ..Default::default() })).unwrap();
        assert_eq!(a.wgsl, b.wgsl);
        assert_ne!(a.params, b.params);
    }

    /// With the stamp on, the node gives the height of the shape's top surface over each (x, y), and -1000 off its footprint.
    #[test]
    fn a_stamp_is_the_height_of_the_shapes_top_surface() {
        let Some(gpu) = crate::gpu_tests::gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let path = temp_sphere("noiser_sdf_node_stamp.sdf");
        let g = graph_with_sphere(&path, LoadSdfParams { stamp: true, ..Default::default() });
        // Fitted: a sphere 0.8 across centred in the tile, so its top is at 0.9 over the middle and lower away from it.
        let v = crate::gpu_tests::eval(&gpu, &g, &[[0.5, 0.5, 0.0], [0.5, 0.7, 0.0], [0.05, 0.05, 0.0]]);
        assert!((v[0] - 0.9).abs() < 0.03, "middle: {}", v[0]);
        assert!(v[1] < v[0] - 0.02 && v[1] > 0.6, "off the middle: {}", v[1]);
        assert!(v[2] < -900.0, "off the footprint: {}", v[2]);
    }

    /// The node's output on the GPU is the composition's distance, fitted into the tile.
    #[test]
    fn the_node_gives_the_distance_of_the_fitted_shape_on_the_gpu() {
        let Some(gpu) = crate::gpu_tests::gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let path = temp_sphere("noiser_sdf_node_gpu.sdf");
        let params = LoadSdfParams::default();
        let g = graph_with_sphere(&path, params.clone());
        let loaded = load(&path);
        let data = loaded.as_ref().as_ref().unwrap();
        let shader = params.shader_params(Some(data));
        let (k, center) = (shader[3] as f64, [shader[4] as f64, shader[5] as f64, shader[6] as f64]);
        let points: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5, 0.1, 0.5], [0.2, 0.7, 0.4], [0.0, 0.0, 0.0], [0.5, 0.5, 0.88]];
        let got = crate::gpu_tests::eval(&gpu, &g, &points);
        // A sphere of radius 1: its distance is |p| - 1 in its own units, times the tile units per unit.
        for (p, d) in points.iter().zip(got) {
            let q = [0, 1, 2].map(|a| center[a] + (p[a] as f64 - 0.5) / k);
            let expected = ((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - 1.0) * k;
            assert!((d as f64 - expected).abs() < 2e-3, "{p:?}: gpu {d} expected {expected}");
        }
    }

    /// Writes a graph with a Load SDF node, for looking at it in the app: `NOISER_WRITE_SDF_GRAPH=out.noisegraph NOISER_SDF=file.sdf`.
    #[test]
    #[ignore]
    fn write_a_demo_graph() {
        let (Ok(out), Ok(sdf)) = (std::env::var("NOISER_WRITE_SDF_GRAPH"), std::env::var("NOISER_SDF")) else { return };
        std::fs::write(out, graph_with_sphere(&sdf, LoadSdfParams::default()).to_json()).unwrap();
    }

    /// Writes a noise graph and a mesh graph that roughens a `.sdf` shape with it, for screenshots:
    /// `NOISER_DEMO_DIR=dir cargo test write_demo_graphs -- --ignored` (the folder is the library's Mesh Graphs folder, next to SDFs and Noises).
    #[test]
    #[ignore]
    fn write_demo_graphs() {
        use crate::graph::{Graph, MathOp, NodeKind};
        let Ok(dir) = std::env::var("NOISER_DEMO_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        let noise_json = Graph::template("Warped cells").to_json();
        // Paths are relative to the graph's own folder, so the files can be shared together.
        let mut g = Graph::default();
        let shape = g.add(NodeKind::LoadSdf(LoadSdfParams { file: "../SDFs/Vase.sdf".into(), fit: false, ..Default::default() }), [0.0, 0.0]);
        let noise = g.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { file: "../Noises/Warped cells.noisegraph".into(), embedded: noise_json, frequency: 0.8, amplitude: 0.1, detached: false }), [0.0, 130.0]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [220.0, 60.0]);
        let remesh = g.add(NodeKind::Remesh { triangles: 20_000, adaptivity: 1.0 }, [420.0, 60.0]);
        let uv = g.add(NodeKind::AutoUv(Default::default()), [620.0, 60.0]);
        let out = g.add(NodeKind::OutMesh(crate::meshgraph::OutParams { folder: "out".into(), name: "vase_rough".into(), ..Default::default() }), [820.0, 60.0]);
        g.connect(shape, add, 0);
        g.connect(noise, add, 1);
        g.connect(add, remesh, 0);
        g.connect(remesh, uv, 0);
        g.connect(uv, out, 0);
        std::fs::write(dir.join("Rough vase.meshgraph"), g.to_json()).unwrap();
    }

    /// A Mesh graph run: a ball of radius 1 (a mesh file) with a sphere of radius 0.5 (a .sdf) carved out of it.
    #[test]
    fn a_load_sdf_node_carves_a_mesh_in_a_mesh_graph_run() {
        use crate::graph::{Graph, MathOp, NodeKind};
        let Some(gpu) = crate::gpu_tests::gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let dir = std::env::temp_dir().join("noiser-loadsdf-meshgraph");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ball = crate::mesh::shapes::icosphere(1.0, 3);
        crate::surface::RawMesh { normals: ball.positions.clone(), positions: ball.positions, triangles: ball.triangles }.write_obj(&dir.join("ball.obj")).unwrap();
        let sdf = dir.join("hole.sdf");
        std::fs::write(&sdf, SPHERE.replace(r#""scale":[2,2,2]"#, r#""scale":[1,1,1]"#)).unwrap();

        let mut g = Graph::default();
        let mesh = g.add(NodeKind::LoadMesh(crate::meshgraph::LoadMeshParams { path: "ball.obj".into(), resolution: 64, ..Default::default() }), [0.0; 2]);
        let hole = g.add(NodeKind::LoadSdf(LoadSdfParams { file: sdf.to_string_lossy().into_owned(), fit: false, ..Default::default() }), [0.0; 2]);
        let negated = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: -1.0, c: 0.5 }, [0.0; 2]);
        let carved = g.add(NodeKind::Math { op: MathOp::Max, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let remesh = g.add(NodeKind::Remesh { triangles: 4000, adaptivity: 1.0 }, [0.0; 2]);
        let out = g.add(NodeKind::OutMesh(crate::meshgraph::OutParams { folder: "out".into(), name: "carved".into(), ..Default::default() }), [0.0; 2]);
        g.connect(hole, negated, 0);
        g.connect(mesh, carved, 0);
        g.connect(negated, carved, 1);
        g.connect(carved, remesh, 0);
        g.connect(remesh, out, 0);
        let mut cache = crate::meshgraph::MeshCache::default();
        let mut log = Vec::new();
        let mut note = |s: &str| log.push(s.to_string());
        let written = crate::meshgraph::Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.clone(), seed: 0, progress: Default::default(), log: &mut note, batch_file: None, max_triangles: None }
            .run(&g)
            .unwrap_or_else(|e| panic!("{e}
{log:?}"));
        assert_eq!(written.len(), 1);
        let obj = std::fs::read_to_string(&written[0]).unwrap();
        let radii: Vec<f32> = obj
            .lines()
            .filter(|l| l.starts_with("v "))
            .map(|l| {
                let p: Vec<f32> = l.split_whitespace().skip(1).map(|n| n.parse().unwrap()).collect();
                (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
            })
            .collect();
        let (lo, hi) = (radii.iter().cloned().fold(9.0, f32::min), radii.iter().cloned().fold(0.0, f32::max));
        assert!((0.42..0.58).contains(&lo) && (0.9..1.1).contains(&hi), "the shell runs from {lo} to {hi}");
    }
}
