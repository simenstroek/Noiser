//! Command-line tools for scripts and agents: look up what nodes exist (`--schema`), read a graph as a short name-based spec
//! (`--describe`), write one from such a spec (`--build`), check a file (`--validate`) and look at a result without opening the
//! app (`--preview`). They need no window; only `--preview` of a .noisegraph needs a GPU.
//!
//! The spec is plain JSON: `{"nodes": {"name": <node>, ...}, "edges": [["from", "to", port], ...]}`. A node is written like in a
//! .noisegraph file (`"Separate"`, `{"Float": 2.0}`, `{"Noise": {"scale": 4}}`), and any setting left out keeps its default.
//! An edge is `[from, to, input port]`, or `[from, to, input port, output port]` for nodes with several outputs.

use crate::graph::{Edge, Graph, NodeKind};
use crate::graph_params::{CurveMode, MathOp, NoiseKind, PatternKind};
use eframe::wgpu;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// Every node kind: its name, one example with all settings at their defaults, and what it does. Keep in step with `NodeKind`
/// (a test checks that every variant is listed).
fn catalogue() -> Vec<(&'static str, NodeKind, &'static str)> {
    use NodeKind as K;
    vec![
        ("Float", K::Float(0.0), "A number. Output: the number."),
        ("Vec3", K::Vec3([0.0; 3]), "A vector. Output: the vector."),
        ("Math", K::Math { op: MathOp::ALL[0], a: 0.0, b: 0.0, c: 0.0 }, "Maths on inputs a, b (and c: the blend width for smooth min/max, the third operand for ternary ops). Unconnected inputs use the stored a, b, c. `op` is one of the MathOp values."),
        ("Remap", K::Remap { vals: [0.0, 0.0, 1.0, 0.0, 1.0], clamp: false }, "Maps input 0 from [from min, from max] to [to min, to max]. Inputs: value, from min, from max, to min, to max (defaults in `vals`)."),
        ("Separate", K::Separate, "Splits a vector into X, Y, Z (output ports 0, 1, 2)."),
        ("Combine", K::Combine { vals: [0.0; 3] }, "A vector from X, Y, Z inputs (defaults in `vals`)."),
        ("Random", K::Random { vals: [0.0, 1.0], seed: None }, "A random number between min and max, the same everywhere (from the global seed and the node id, or `seed`)."),
        ("Mix", K::Mix { vals: [0.5, 0.0, 1.0], clamp: true }, "Blends A into B by factor. Inputs: factor, A, B."),
        ("Curve", K::Curve { points: vec![[0.0, 0.0], [1.0, 1.0]], mode: CurveMode::Linear }, "A function of input 0 drawn through `points` [x, y]; `mode` is a CurveMode value."),
        ("Grid", K::Grid { counts: [4.0; 3], corners: false, cubic: true, amount: 1.0 }, "Distance to the nearest cell centre of a regular grid, scaled to -1..1. Input 0: position (default: world position)."),
        ("Pattern", K::Pattern(Default::default()), "A regular pattern (bricks, honeycomb, gyroid, ...); `kind` is a PatternKind value. Input 0: position."),
        ("Position", K::position(), "The world position, each axis multiplied by inputs X, Y, Z (default 1). Feed it to other nodes' position input."),
        ("Radial", K::Radial(Default::default()), "Position in cylindrical or spherical coordinates. Input 0: position."),
        ("Mask", K::Mask { name: "mask".into() }, "Landscape tab: the terrain map `name` (a painted mask today; shown as Map in the app), 0..1, at the position. Input 0: position."),
        ("Slope", K::Slope { scale: 1.0 }, "The slope in degrees of the input field over x and y."),
        ("Heightmap", K::Heightmap(Default::default()), "Landscape tab: an image (PNG or EXR) as terrain, read at the position (x and y across the terrain, 0 to 1), in the graph's units. A PNG the Landscape tab exported (with its .json beside it) returns in metres; another PNG is 0..1 of the height setting; `scale` and `offset` adjust it; relative paths are looked for next to the project. Shrunk to 2049 a side at most. Input 0: position."),
        ("Stamp", K::Stamp(Default::default()), "Landscape tab: an .sdf shape (`file`, relative to the project) pressed into the terrain height at input 0, seen from above (the shape's -Y; .sdf is Y-up). `size_m` is the widest side of its box in metres, `height_factor` stretches its height, `center` is its middle across the terrain (0..1), `rotation_deg` turns it. `mode`: Add (stands on the ground), Max (rises out of it where it covers), Cut (carves it). Caves and overhangs stay in the Mesh graph."),
        ("TerrainLayer", K::TerrainLayer { name: "rock".into() }, "Mesh graph: the terrain's paint layer `name` (a weightmap the Landscape tab wrote; list it in the terrain Load Mesh node's `maps`) at the position (field units), 0..1. Wire it to a Bake node's extra inputs to write per-chunk layer weights, or to Math nodes. Scatter's `density_map` takes the same name. Input 0: position."),
        ("Terrace", K::Terrace { steps: 8.0, sharpness: 4.0 }, "Steps input 0 (a height) into `steps` terraces per unit: flat treads, risers as sharp as `sharpness` (1 gentle, 10 nearly a staircase). Add noise before it for wobbly strata, or Mix it with the original height."),
        ("Craters", K::Craters(Default::default()), "Craters over the position (x and y across the terrain, 0..1): a bowl `depth` deep with a raised rim (`rim` of the depth) in each of `cells` x `cells` cells that has one (`density`), sizes between `size` (fractions of half a cell). Add the output to a height. Input 0: position."),
        ("Select", K::Select { lo: 0.0, hi: 1.0, soft: 0.1 }, "Landscape tab: 1 where input 0 lies between `lo` and `hi`, falling smoothly to 0 over `soft` beyond either end. Wire a height, a slope (degrees) or a map in; feeds a Layer."),
        ("Layer", K::Layer(Default::default()), "Landscape tab output: a paint layer named `name` (preview `color` as 0..1 RGB) whose weight is input 0 (0..1). Exports normalise all layers to sum to 1 per cell and write one 8-bit weightmap each."),
        ("Erosion", K::Erosion(Default::default()), "Landscape tab only: hydraulic and thermal erosion of input 0 (the height) on the terrain's working grid. Output 0: eroded height (input's units); 1: flow, 2: deposition, 3: wear (0..1). `auto` reruns it when the input changes."),
        ("Reroute", K::Reroute, "A waypoint on a wire; passes its input on."),
        ("NamedReroute", K::NamedReroute { name: "name".into() }, "The named end of a wire; read it with NamedRerouteUse."),
        ("NamedRerouteUse", K::NamedRerouteUse { name: "name".into() }, "Reads the NamedReroute with the same name."),
        ("Normal", K::Normal { flip: false }, "Unit gradient of the input field (low to high unless flipped)."),
        ("Noise", K::Noise(crate::graph_params::NoiseParams::default()), "Noise (`kind` is a NoiseKind value). Input 0: position. Output 0: value, output 1: displacement vector. Extra inputs drive scale/octaves/etc."),
        ("OutDensity", K::OutDensity, "The end of a .noisegraph: input 0 is the density. Exactly one is needed."),
        ("OutNormal", K::OutNormal, "Optional normal output of a .noisegraph."),
        ("LoadMesh", K::LoadMesh(Default::default()), "Mesh graph: a mesh file or a Landscape heightmap PNG as a field. Output 0: the field, 1: the mesh. `unit_cm` = centimetres per field unit."),
        ("LoadSdf", K::LoadSdf(Default::default()), "A .sdf composition (from the SDF Composer) as a field, placed with `offset`/`size`/`snap`/`z_up`/`stamp`. Input 0: position."),
        ("Scatter", K::Scatter(Default::default()), "Copies of a .sdf shape planted across a terrain by density/size/slope/elevation, or in a ring. Inputs: 0 the terrain field, 1 optional displacement for the copies."),
        ("NoiseGraph", K::NoiseGraph(Default::default()), "A .noisegraph as a node (use it as displacement). Input 0: position."),
        ("Remesh", K::Remesh { triangles: 100_000, adaptivity: 1.0 }, "Turns a field into an adaptive triangle mesh of about `triangles` triangles. Input 0: the field."),
        ("AutoUv", K::AutoUv(Default::default()), "UVs for a mesh (input 0). `planar` for terrain."),
        ("Bake", K::Bake(Default::default()), "Maps baked onto a mesh (input 0) from a field (input 1); one image output per map."),
        ("OutMesh", K::OutMesh(Default::default()), "Writes the mesh at input 0 to `folder`/`name`."),
        ("OutImage", K::OutImage(Default::default()), "Writes the image at input 0 to `folder`/`name`."),
        ("TransferUv", K::TransferUv, "UVs of a mesh (input 1 from a LoadMesh's mesh output) carried onto another (input 0)."),
        ("ImageSplit", K::ImageSplit, "An image split into R, G, B, A grey images (outputs 0..3)."),
        ("ImageCombine", K::ImageCombine, "Four grey images (R, G, B, A inputs) combined into one."),
        ("ImageLevels", K::ImageLevels { scale: 1.0, offset: 0.0, invert: false }, "value * scale + offset, optionally inverted, clamped to 0..1."),
    ]
}

/// The name a node kind goes by in files and specs: its serde variant name.
fn kind_name(kind: &NodeKind) -> String {
    match serde_json::to_value(kind) {
        Ok(Value::String(s)) => s,
        Ok(Value::Object(m)) => m.keys().next().cloned().unwrap_or_default(),
        _ => String::new(),
    }
}

fn names<T: serde::Serialize>(all: &[T]) -> Value {
    all.iter().filter_map(|v| serde_json::to_value(v).ok()).collect()
}

/// `--schema`: every node kind with its default settings, plus the enum values and the conventions that trip people up.
pub fn schema() -> Value {
    let nodes: Vec<Value> = catalogue()
        .into_iter()
        .map(|(name, kind, doc)| json!({ "kind": name, "doc": doc, "inputs": kind.input_count(), "outputs": kind.output_count(), "default": serde_json::to_value(&kind).unwrap_or(Value::Null) }))
        .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "nodes": nodes,
        "enums": { "MathOp": names(&MathOp::ALL), "NoiseKind": names(&NoiseKind::ALL), "PatternKind": names(&PatternKind::ALL), "CurveMode": names(&CurveMode::ALL) },
        "files": {
            ".noisegraph": "A graph ending in one OutDensity: a 3D field (density) from noise and maths.",
            ".meshgraph": "A graph that loads fields (LoadMesh/LoadSdf/Scatter/NoiseGraph), combines them with Math nodes (Add, Min, Max, SmoothMin), then Remesh -> AutoUv -> OutMesh (+ Bake -> OutImage).",
            ".sdf": "An SDF Composer file (shapes, booleans, layers). Edited in the app; Y-up. Referenced by LoadSdf and Scatter.",
            ".landscape": "A Landscape project: a heightmap graph, settings and painted masks.",
        },
        "notes": [
            "Spec format for --build: {\"nodes\": {name: node}, \"edges\": [[from, to, input_port, (output_port)]]}; settings left out keep their defaults.",
            "A mesh graph's field is in the units of its first LoadMesh (`unit_cm`); lengths on Scatter/LoadSdf are in those units.",
            "Terrain heightmaps are Z-up in the field; .sdf files are Y-up (set `z_up` on LoadSdf/Scatter to stand them up).",
            "Relative file paths in nodes are relative to the graph file's folder.",
        ],
    })
}

// ---- describe / build ---------------------------------------------------------------------------------------------

fn name_of(id: u32, kind: &NodeKind) -> String {
    format!("{}{}", kind_name(kind), id)
}

/// `--describe`: a graph as a spec (node names are the kind plus the node id).
pub fn describe(graph: &Graph) -> Value {
    let name: std::collections::BTreeMap<u32, String> = graph.nodes.iter().map(|(id, n)| (*id, name_of(*id, &n.kind))).collect();
    let mut nodes = Map::new();
    for (id, n) in &graph.nodes {
        nodes.insert(name[id].clone(), serde_json::to_value(&n.kind).unwrap_or(Value::Null));
    }
    let edges: Vec<Value> = graph
        .edges
        .iter()
        .map(|e| if e.from_port == 0 { json!([name[&e.from], name[&e.to], e.port]) } else { json!([name[&e.from], name[&e.to], e.port, e.from_port]) })
        .collect();
    json!({ "nodes": nodes, "edges": edges })
}

/// Fills in the settings a spec's node leaves out from the kind's defaults.
fn complete(node: &Value) -> Result<NodeKind, String> {
    let (name, given) = match node {
        Value::String(s) => (s.clone(), None),
        Value::Object(m) if m.len() == 1 => (m.keys().next().unwrap().clone(), m.values().next().cloned()),
        _ => return Err("a node is a kind name (\"Separate\") or {\"Kind\": settings}".into()),
    };
    let known: Vec<String> = catalogue().iter().map(|c| c.0.to_string()).collect();
    let sample = catalogue().into_iter().find(|c| c.0 == name).map(|c| c.1).ok_or(format!("unknown node kind {name} (known: {})", known.join(", ")))?;
    let base = serde_json::to_value(&sample).map_err(|e| e.to_string())?;
    let merged = match (base, given) {
        (Value::Object(mut b), Some(Value::Object(g))) => {
            let inner = b.get_mut(&name).and_then(|v| v.as_object_mut());
            match (inner, g) {
                (Some(inner), g) => {
                    for (k, v) in g {
                        if !inner.contains_key(&k) {
                            return Err(format!("{name} has no setting {k} (it has: {})", inner.keys().cloned().collect::<Vec<_>>().join(", ")));
                        }
                        inner.insert(k, v);
                    }
                }
                _ => return Err(format!("{name} takes a plain value, not an object of settings")),
            }
            Value::Object(b)
        }
        (_, Some(v)) => json!({ name.clone(): v }),
        (b, None) => b,
    };
    serde_json::from_value(merged).map_err(|e| format!("{name}: {e}"))
}

/// `--build`: a spec as a graph, laid out left to right.
pub fn build(spec: &Value) -> Result<Graph, String> {
    let nodes = spec["nodes"].as_object().ok_or("the spec needs a \"nodes\" object")?;
    let mut g = Graph::default();
    let mut ids = std::collections::BTreeMap::new();
    for (name, node) in nodes {
        let kind = complete(node).map_err(|e| format!("node {name}: {e}"))?;
        ids.insert(name.clone(), g.add(kind, [0.0; 2]));
    }
    for (i, e) in spec["edges"].as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter().enumerate() {
        let a = e.as_array().filter(|a| a.len() >= 3).ok_or(format!("edge {i}: expected [from, to, port] or [from, to, port, output port]"))?;
        let find = |v: &Value| v.as_str().and_then(|s| ids.get(s)).copied().ok_or(format!("edge {i}: no node named {v}"));
        let (from, to) = (find(&a[0])?, find(&a[1])?);
        let port = a[2].as_u64().ok_or(format!("edge {i}: the port must be a number"))? as usize;
        let from_port = a.get(3).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if port >= g.nodes[&to].kind.input_count() {
            return Err(format!("edge {i}: {} has {} input(s), there is no input {port}", a[1], g.nodes[&to].kind.input_count()));
        }
        if from_port >= g.nodes[&from].kind.output_count().max(1) {
            return Err(format!("edge {i}: {} has {} output(s), there is no output {from_port}", a[0], g.nodes[&from].kind.output_count()));
        }
        let before = g.edges.len();
        g.connect_from(from, from_port, to, port);
        if g.edges.len() == before && !g.edges.iter().any(|x| x.from == from && x.to == to && x.port == port && x.from_port == from_port) {
            return Err(format!("edge {i}: {} cannot feed {} (the pin types do not match)", a[0], a[1]));
        }
    }
    layout(&mut g);
    Ok(g)
}

fn layout(g: &mut Graph) {
    let ids: Vec<u32> = g.nodes.keys().copied().collect();
    let mut depth: std::collections::BTreeMap<u32, usize> = ids.iter().map(|i| (*i, 0)).collect();
    for _ in 0..ids.len() {
        for Edge { from, to, .. } in &g.edges {
            let d = depth[from] + 1;
            if d > depth[to] {
                depth.insert(*to, d);
            }
        }
    }
    let mut rows = std::collections::BTreeMap::<usize, usize>::new();
    for id in ids {
        let d = depth[&id];
        let row = rows.entry(d).or_insert(0);
        g.nodes.get_mut(&id).unwrap().pos = [d as f32 * 230.0, *row as f32 * 120.0];
        *row += 1;
    }
}

// ---- validate -----------------------------------------------------------------------------------------------------

/// Problems with a file: `(errors, warnings)`.
pub fn validate(path: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let (mut errors, mut warnings) = (Vec::new(), Vec::new());
    match ext.as_str() {
        "sdf" => match crate::sdf_file::SdfFile::from_json(&text) {
            Ok(f) => {
                if crate::sdf_model::layer_fields(&f).iter().all(|l| l.distance([0.0; 3]).is_none()) {
                    warnings.push("no layer has anything to draw".into());
                }
            }
            Err(e) => errors.push(e),
        },
        "landscape" => match serde_json::from_str::<Value>(&text) {
            Ok(v) => {
                for key in ["noiser_landscape", "graph", "settings"] {
                    if v.get(key).is_none() {
                        errors.push(format!("missing \"{key}\""));
                    }
                }
                if let Some(g) = v.get("graph").and_then(|g| g.as_str().map(String::from).or_else(|| Some(g.to_string()))) {
                    match Graph::from_json(&g) {
                        Err(e) => errors.push(format!("graph: {e}")),
                        Ok(graph) => check_landscape(&graph, v.get("settings"), path.parent().unwrap_or(Path::new(".")), &mut errors, &mut warnings),
                    }
                }
            }
            Err(e) => errors.push(e.to_string()),
        },
        "noisegraph" | "meshgraph" => {
            let g = match Graph::from_json(&text) {
                Ok(g) => g,
                Err(e) => return Ok((vec![e], warnings)),
            };
            check_graph(&g, ext == "meshgraph", path.parent().unwrap_or(Path::new(".")), &mut errors, &mut warnings);
        }
        _ => return Err(format!("{}: expected a .noisegraph, .meshgraph, .sdf or .landscape file", path.display())),
    }
    Ok((errors, warnings))
}

/// Landscape project checks: missing files, unwired Layer and Stamp nodes, and the fit setting that misleads them.
fn check_landscape(g: &Graph, settings: Option<&Value>, base: &Path, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    let exists = |file: &str| !file.trim().is_empty() && (Path::new(file).is_absolute() && Path::new(file).exists() || base.join(file).exists());
    for (id, n) in &g.nodes {
        if n.muted {
            continue;
        }
        match &n.kind {
            NodeKind::Heightmap(p) if !exists(&p.file) => errors.push(format!("Heightmap node {id}: the file \"{}\" does not exist (relative paths are looked for next to the project)", p.file)),
            NodeKind::Stamp(p) if !exists(&p.file) => errors.push(format!("Stamp node {id}: the file \"{}\" does not exist (relative paths are looked for next to the project)", p.file)),
            NodeKind::Layer(l) if g.input_from(*id, 0).is_none() => warnings.push(format!("Layer node {id} ({}) has nothing wired in: its weight is 0 everywhere", l.name)),
            NodeKind::Select { .. } | NodeKind::Erosion(_) | NodeKind::Terrace { .. } if g.input_from(*id, 0).is_none() => warnings.push(format!("{} node {id} has no input", kind_name(&n.kind))),
            _ => {}
        }
    }
    let fit = settings.and_then(|s| s.get("fit")).and_then(|f| f.as_bool()).unwrap_or(true);
    if crate::landscape::fit_misleads(g, fit) {
        warnings.push("\"fit\" is on: it rescales the terrain after Erosion, Stamp, Select and Layer nodes, so their heights and thresholds (in height units) are not the final metres. Set settings.fit to false for predictable heights".into());
    }
}

fn check_graph(g: &Graph, mesh: bool, base: &Path, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    let label = |id: u32| name_of(id, &g.nodes[&id].kind);
    let outs = g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::OutDensity)).count();
    if !mesh && outs != 1 {
        errors.push(format!("a .noisegraph needs exactly one OutDensity (found {outs})"));
    }
    let sinks = g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::OutMesh(_) | NodeKind::OutImage(_))).count();
    if mesh && sinks == 0 {
        errors.push("a .meshgraph needs an OutMesh or OutImage node, or running it writes nothing".into());
    }
    // A cycle: a node that can reach itself.
    let mut depth = std::collections::BTreeMap::<u32, usize>::new();
    for _ in 0..=g.nodes.len() {
        for e in &g.edges {
            let d = depth.get(&e.from).copied().unwrap_or(0) + 1;
            if d > g.nodes.len() {
                errors.push("the graph has a cycle".into());
                return;
            }
            let slot = depth.entry(e.to).or_insert(0);
            *slot = (*slot).max(d);
        }
    }
    let connected = |id: u32, port: usize| g.edges.iter().any(|e| e.to == id && e.port == port);
    let needs: &[(&str, &[usize])] = &[("Remesh", &[0]), ("AutoUv", &[0]), ("OutMesh", &[0]), ("OutImage", &[0]), ("Bake", &[0, 1]), ("OutDensity", &[0]), ("Scatter", &[0]), ("TransferUv", &[0, 1])];
    for (id, n) in &g.nodes {
        let k = kind_name(&n.kind);
        if n.muted {
            continue;
        }
        if let Some((_, ports)) = needs.iter().find(|(name, _)| *name == k) {
            for p in *ports {
                if !connected(*id, *p) {
                    errors.push(format!("{}: input {p} is not connected", label(*id)));
                }
            }
        }
        // Paint layers: the maps a terrain loads must be in its sidecar, and what reads them must be among them.
        if mesh {
            let terrain_maps = crate::codegen::terrain_maps(g);
            let asked = match &n.kind {
                NodeKind::TerrainLayer { name } => Some(name.as_str()),
                NodeKind::Scatter(p) if !p.density_map.is_empty() => Some(p.density_map.as_str()),
                _ => None,
            };
            if let Some(name) = asked.filter(|n| !terrain_maps.iter().any(|m| m == n)) {
                errors.push(format!("{}: the paint layer \"{name}\" is not among the terrain Load Mesh node's maps ({})", label(*id), if terrain_maps.is_empty() { "none".to_string() } else { terrain_maps.join(", ") }));
            }
            if let NodeKind::LoadMesh(p) = &n.kind && crate::landscape_field::is_heightmap(&p.path) && !p.maps.is_empty() {
                let known = crate::landscape_field::sidecar_layers(&crate::meshgraph::resolve_in(base, &p.path));
                for m in p.maps.iter().filter(|m| !known.iter().any(|(n, _)| n == *m)) {
                    errors.push(format!("{}: the heightmap's sidecar lists no paint layer called \"{m}\" (it has: {})", label(*id), known.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")));
                }
            }
        }
        let resolve = |p: &str| if Path::new(p).is_absolute() { PathBuf::from(p) } else { base.join(p) };
        let files: Vec<(&str, &str)> = match &n.kind {
            NodeKind::LoadMesh(p) => vec![("path", &p.path)],
            NodeKind::LoadSdf(p) => vec![("file", &p.file)],
            NodeKind::Scatter(p) => vec![("file", &p.file)],
            NodeKind::NoiseGraph(p) if !p.detached || p.embedded.is_empty() => vec![("file", &p.file)],
            _ => vec![],
        };
        for (field, f) in files {
            if f.is_empty() {
                if !matches!(&n.kind, NodeKind::NoiseGraph(p) if !p.embedded.is_empty()) {
                    errors.push(format!("{}: {field} is empty", label(*id)));
                }
            } else if !resolve(f).exists() && !matches!(&n.kind, NodeKind::NoiseGraph(p) if !p.embedded.is_empty()) {
                errors.push(format!("{}: {field} {f} does not exist", label(*id)));
            }
        }
        if let NodeKind::OutMesh(p) | NodeKind::OutImage(p) = &n.kind
            && p.name.is_empty() {
                warnings.push(format!("{}: name is empty", label(*id)));
            }
        if !mesh && n.kind.input_count() == 0 && !matches!(n.kind, NodeKind::Float(_) | NodeKind::Vec3(_) | NodeKind::NamedRerouteUse { .. }) {
            warnings.push(format!("{} has no inputs", label(*id)));
        }
    }
    for id in g.nodes.keys() {
        let n = &g.nodes[id];
        let is_end = matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::OutMesh(_) | NodeKind::OutImage(_));
        if !is_end && !g.edges.iter().any(|e| e.from == *id) && !matches!(n.kind, NodeKind::Reroute) {
            warnings.push(format!("{}: its output goes nowhere", label(*id)));
        }
    }
    for e in &g.exposed {
        match g.exposed_value(e) {
            Ok(_) => {}
            Err(err) => errors.push(format!("exposed {}: {err}", e.name)),
        }
    }
}

// ---- preview ------------------------------------------------------------------------------------------------------

/// `--preview FILE OUT.png [--size N] [--up y]`: a noise graph as a slice, a .sdf from the front, side and above, a mesh as shaded
/// views (Z is up unless `--up y`, which OBJ files from the SDF side usually are).
pub fn preview(file: &Path, out: &Path, size: u32, y_up: bool) -> Result<String, String> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let img = match ext.as_str() {
        "noisegraph" => preview_noise(file, size)?,
        "landscape" => crate::landscape::preview_project(file, size)?,
        "sdf" => preview_sdf(file, size)?,
        _ if crate::meshformats::is_mesh_file(file) => preview_mesh(file, size, y_up)?,
        _ => return Err(format!("{}: --preview takes a .noisegraph, .sdf or a mesh file (.obj, .ply, .glb, ...)", file.display())),
    };
    img.save(out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(format!("wrote {}", out.display()))
}

fn preview_noise(file: &Path, size: u32) -> Result<image::RgbImage, String> {
    let graph = Graph::from_json(&std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?)?;
    crate::sdf_node::set_base(crate::sdf_node::NOISE_GRAPH_BASE, file.parent());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))?;
    let ev = crate::export::Evaluator::new(&device, &queue, &graph, 0)?;
    let n = size.clamp(16, 1024);
    let values = ev.eval(n, [0; 3], [n, n, 1], false, Some((2, 0.0)));
    let (lo, hi) = values.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let span = (hi - lo).max(1e-9);
    Ok(image::RgbImage::from_fn(n, n, |x, y| {
        let g = (((values[(y * n + x) as usize] - lo) / span).clamp(0.0, 1.0) * 255.0) as u8;
        image::Rgb([g, g, g])
    }))
}

/// Three views side by side, each a sphere-traced picture lit from the upper left.
fn preview_sdf(file: &Path, size: u32) -> Result<image::RgbImage, String> {
    use rayon::prelude::*;
    let sdf = crate::sdf_file::SdfFile::load(file)?;
    let fields = crate::sdf_model::layer_fields(&sdf);
    let scene = crate::sdf_model::file_scene(&fields, None);
    let (lo, hi) = scene.bounds.ok_or("the file has nothing to draw")?;
    let dist = |p: [f64; 3]| fields.iter().filter(|f| !f.hidden).filter_map(|f| f.distance(p)).fold(f64::MAX, f64::min);
    let c = [0, 1, 2].map(|a| (lo[a] + hi[a]) * 0.5);
    let r = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max) * 0.5 * 1.1;
    let n = size.clamp(64, 1024);
    // (right, up, toward the viewer) for front (looking down -Z), side (looking down -X) and top (looking down -Y).
    let views: [([f64; 3], [f64; 3], [f64; 3]); 3] = [([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]), ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])];
    let mut img = image::RgbImage::new(n * 3, n);
    let rows: Vec<Vec<[u8; 3]>> = (0..n)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(n as usize * 3);
            for (right, up, back) in views {
                for x in 0..n {
                    let (u, v) = ((x as f64 + 0.5) / n as f64 * 2.0 - 1.0, 1.0 - (y as f64 + 0.5) / n as f64 * 2.0);
                    let origin = [0, 1, 2].map(|a| c[a] + back[a] * r * 3.0 + right[a] * u * r + up[a] * v * r);
                    let dir = back.map(|b| -b);
                    let (mut t, mut hit) = (0.0, false);
                    for _ in 0..160 {
                        let d = dist([0, 1, 2].map(|a| origin[a] + dir[a] * t));
                        if d < r * 1e-3 {
                            hit = true;
                            break;
                        }
                        t += d.max(r * 1e-4);
                        if t > r * 6.0 {
                            break;
                        }
                    }
                    row.push(if hit {
                        let p = [0, 1, 2].map(|a| origin[a] + dir[a] * t);
                        let e = r * 2e-3;
                        let g = [0, 1, 2].map(|a| {
                            let (mut a_p, mut a_m) = (p, p);
                            a_p[a] += e;
                            a_m[a] -= e;
                            dist(a_p) - dist(a_m)
                        });
                        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
                        let light = [-0.5, 0.7, 0.5];
                        let ll = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2] as f64).sqrt();
                        let lit = ((g[0] * light[0] + g[1] * light[1] + g[2] * light[2]) / (len * ll)).clamp(0.0, 1.0);
                        let s = (40.0 + 200.0 * (0.25 + 0.75 * lit)) as u8;
                        [s, s, s]
                    } else {
                        [24, 28, 36]
                    });
                }
            }
            row
        })
        .collect();
    for (y, row) in rows.iter().enumerate() {
        for (x, px) in row.iter().enumerate() {
            img.put_pixel(x as u32, y as u32, image::Rgb(*px));
        }
    }
    Ok(img)
}

/// Front, side and top of a mesh: a z-buffer with flat shading, enough to see its shape. Z is up.
fn preview_mesh(file: &Path, size: u32, y_up: bool) -> Result<image::RgbImage, String> {
    let mut mesh = crate::meshformats::load(file)?;
    if y_up {
        for p in &mut mesh.positions {
            *p = [p[0], -p[2], p[1]];
        }
    }
    if mesh.positions.is_empty() {
        return Err("the mesh is empty".into());
    }
    let n = size.clamp(64, 2048);
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let c = [0, 1, 2].map(|a| (lo[a] + hi[a]) * 0.5);
    let half = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) * 0.5 * 1.05;
    // (screen x axis, screen y axis, depth axis towards the viewer, sign of x) for front, side, top.
    let views = [(0usize, 2usize, 1usize, 1.0f32), (1, 2, 0, -1.0), (0, 1, 2, 1.0)];
    let mut img = image::RgbImage::from_pixel(n * 3, n, image::Rgb([24, 28, 36]));
    for (vi, (ax, ay, az, sx)) in views.into_iter().enumerate() {
        let mut depth = vec![f32::MIN; (n * n) as usize];
        let project = |p: [f32; 3]| ((((p[ax] - c[ax]) * sx / half) * 0.5 + 0.5) * n as f32, (1.0 - ((p[ay] - c[ay]) / half * 0.5 + 0.5)) * n as f32, p[az]);
        for t in &mesh.triangles {
            let p = t.map(|i| mesh.positions[i as usize]);
            let (a, b, cc) = (project(p[0]), project(p[1]), project(p[2]));
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let nrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt().max(1e-20);
            let light = [-0.5f32, 0.7, 0.5];
            let lit = ((nrm[0] * light[0] + nrm[1] * light[1] + nrm[2] * light[2]) / len / 0.99).abs().clamp(0.0, 1.0);
            let shade = (50.0 + 190.0 * (0.2 + 0.8 * lit)) as u8;
            let (x0, x1) = (a.0.min(b.0).min(cc.0).floor().max(0.0) as i32, a.0.max(b.0).max(cc.0).ceil().min(n as f32 - 1.0) as i32);
            let (y0, y1) = (a.1.min(b.1).min(cc.1).floor().max(0.0) as i32, a.1.max(b.1).max(cc.1).ceil().min(n as f32 - 1.0) as i32);
            let area = (b.0 - a.0) * (cc.1 - a.1) - (cc.0 - a.0) * (b.1 - a.1);
            if area.abs() < 1e-9 {
                continue;
            }
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((b.0 - px) * (cc.1 - py) - (cc.0 - px) * (b.1 - py)) / area;
                    let w1 = ((cc.0 - px) * (a.1 - py) - (a.0 - px) * (cc.1 - py)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let z = w0 * a.2 + w1 * b.2 + w2 * cc.2;
                    let slot = &mut depth[(y as u32 * n + x as u32) as usize];
                    if z > *slot {
                        *slot = z;
                        img.put_pixel(vi as u32 * n + x as u32, y as u32, image::Rgb([shade, shade, shade]));
                    }
                }
            }
        }
    }
    Ok(img)
}

// ---- command line -------------------------------------------------------------------------------------------------

/// Runs one of the commands above when the arguments ask for it; returns the exit code.
pub fn try_run() -> Option<i32> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let at = args.iter().position(|a| matches!(a.as_str(), "--schema" | "--describe" | "--build" | "--validate" | "--preview" | "--terrain-graph"))?;
    crate::headless::attach_console();
    let arg = |i: usize, what: &str| args.get(at + i).cloned().ok_or(format!("{} needs {what}", args[at]));
    let result: Result<i32, String> = (|| match args[at].as_str() {
        "--schema" => {
            println!("{}", serde_json::to_string_pretty(&schema()).unwrap());
            Ok(0)
        }
        "--describe" => {
            let file = PathBuf::from(arg(1, "a .noisegraph or .meshgraph file")?);
            let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            let (g, settings) = if file.extension().is_some_and(|e| e.eq_ignore_ascii_case("landscape")) {
                let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                (Graph::from_json(v["graph"].as_str().ok_or("not a landscape project")?)?, Some(v["settings"].clone()))
            } else {
                (Graph::from_json(&text)?, None)
            };
            let mut d = describe(&g);
            if let (Some(settings), Some(obj)) = (settings, d.as_object_mut()) {
                obj.insert("settings".into(), settings);
            }
            println!("{}", serde_json::to_string_pretty(&d).unwrap());
            Ok(0)
        }
        "--build" => {
            let (spec, out) = (PathBuf::from(arg(1, "a spec file")?), PathBuf::from(arg(2, "an output .noisegraph or .meshgraph")?));
            let g = build(&serde_json::from_str(&std::fs::read_to_string(&spec).map_err(|e| format!("{}: {e}", spec.display()))?).map_err(|e| format!("{}: {e}", spec.display()))?)?;
            let text = if out.extension().is_some_and(|e| e.eq_ignore_ascii_case("landscape")) {
                // A Landscape project: the graph, the spec's optional "settings" over the defaults, no painted masks.
                let spec: Value = serde_json::from_str(&std::fs::read_to_string(&spec).map_err(|e| format!("{}: {e}", spec.display()))?).map_err(|e| e.to_string())?;
                let mut settings = serde_json::to_value(crate::landscape::Settings::default()).map_err(|e| e.to_string())?;
                if let (Some(mine), Some(given)) = (settings.as_object_mut(), spec.get("settings").and_then(|s| s.as_object())) {
                    for (k, v) in given {
                        if !mine.contains_key(k) {
                            return Err(format!("unknown setting \"{k}\" (the settings are {})", mine.keys().cloned().collect::<Vec<_>>().join(", ")));
                        }
                        mine.insert(k.clone(), v.clone());
                    }
                }
                serde_json::json!({ "noiser_landscape": 1, "graph": g.to_json(), "settings": settings, "masks": [] }).to_string()
            } else {
                g.to_json()
            };
            std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
            let (errors, warnings) = validate(&out)?;
            report(&errors, &warnings);
            println!("wrote {}", out.display());
            Ok(i32::from(!errors.is_empty()))
        }
        "--terrain-graph" => {
            // The Mesh graph that cuts an exported heightmap (with its sidecar, and paint layers beside it) into Nanite chunks.
            let (png, out) = (PathBuf::from(arg(1, "an exported heightmap .png")?), PathBuf::from(arg(2, "an output .meshgraph")?));
            let side = std::fs::read_to_string(format!("{}.json", png.display())).map_err(|e| format!("{}.json: {e} (the sidecar the Landscape tab writes next to the heightmap)", png.display()))?;
            let world_m = serde_json::from_str::<Value>(&side).ok().and_then(|v| v["size_m"].as_f64()).ok_or("the sidecar has no size_m")? as f32;
            let g = crate::landscape_field::terrain_graph(&png, world_m);
            std::fs::write(&out, g.to_json()).map_err(|e| format!("{}: {e}", out.display()))?;
            println!("wrote {} ({} nodes): edit it with --describe / --build, run it with --headless --mesh-graph", out.display(), g.nodes.len());
            Ok(0)
        }
        "--validate" => {
            let (errors, warnings) = validate(Path::new(&arg(1, "a file")?))?;
            report(&errors, &warnings);
            if errors.is_empty() {
                println!("ok");
            }
            Ok(i32::from(!errors.is_empty()))
        }
        _ => {
            let (file, out) = (PathBuf::from(arg(1, "a file")?), PathBuf::from(arg(2, "an output .png")?));
            let size = args.iter().position(|a| a == "--size").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(384);
            let y_up = args.windows(2).any(|w| w[0] == "--up" && w[1].eq_ignore_ascii_case("y"));
            println!("{}", preview(&file, &out, size, y_up)?);
            Ok(0)
        }
    })();
    Some(result.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        2
    }))
}

fn report(errors: &[String], warnings: &[String]) {
    for e in errors {
        println!("error: {e}");
    }
    for w in warnings {
        println!("warning: {w}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_lists_every_node_kind_once() {
        let mut seen = std::collections::BTreeSet::new();
        for (name, kind, _) in catalogue() {
            assert_eq!(kind_name(&kind), name);
            assert!(seen.insert(name), "{name} listed twice");
        }
        // Adding a NodeKind variant breaks this match until the catalogue lists it.
        let _ = |k: &NodeKind| match k {
            NodeKind::Float(_) | NodeKind::Vec3(_) | NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Separate | NodeKind::Combine { .. } | NodeKind::Random { .. } | NodeKind::Mix { .. } | NodeKind::Curve { .. } | NodeKind::Grid { .. } | NodeKind::Pattern(_) | NodeKind::Position { .. } | NodeKind::Radial(_) | NodeKind::Mask { .. } | NodeKind::Slope { .. } | NodeKind::Erosion(_) | NodeKind::Select { .. } | NodeKind::Layer(_) | NodeKind::Heightmap(_) | NodeKind::Stamp(_) | NodeKind::TerrainLayer { .. } | NodeKind::Terrace { .. } | NodeKind::Craters(_) | NodeKind::Reroute | NodeKind::NamedReroute { .. } | NodeKind::NamedRerouteUse { .. } | NodeKind::Normal { .. } | NodeKind::Noise(_) | NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::Scatter(_) | NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::TransferUv | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } => {}
        };
        assert_eq!(seen.len(), 43);
    }

    #[test]
    fn every_default_round_trips_through_a_spec() {
        for (name, kind, _) in catalogue() {
            let spec = json!({ "nodes": { "n": serde_json::to_value(&kind).unwrap() }, "edges": [] });
            let g = build(&spec).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(g.nodes.values().next().unwrap().kind, kind, "{name}");
        }
    }

    #[test]
    fn a_spec_builds_describes_and_validates() {
        let spec = json!({
            "nodes": { "p": "Separate", "noise": { "Noise": { "scale": 3.0 } }, "out": "OutDensity" },
            "edges": [["noise", "out", 0]]
        });
        let g = build(&spec).unwrap();
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 1);
        let again = build(&describe(&g)).unwrap();
        assert_eq!(again.edges, g.edges);
        assert_eq!(again.nodes.values().map(|n| n.kind.clone()).collect::<Vec<_>>(), g.nodes.values().map(|n| n.kind.clone()).collect::<Vec<_>>());
        let mut errors = vec![];
        check_graph(&g, false, Path::new("."), &mut errors, &mut vec![]);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn mistakes_are_named() {
        assert!(build(&json!({ "nodes": { "a": "Nope" } })).unwrap_err().contains("unknown node kind Nope"));
        assert!(build(&json!({ "nodes": { "a": { "Noise": { "scael": 1 } } } })).unwrap_err().contains("no setting scael"));
        assert!(build(&json!({ "nodes": { "a": "OutDensity" }, "edges": [["a", "b", 0]] })).unwrap_err().contains("no node named"));
        assert!(build(&json!({ "nodes": { "a": { "Float": 1.0 }, "o": "OutDensity" }, "edges": [["a", "o", 3]] })).unwrap_err().contains("no input 3"));
        let g = build(&json!({ "nodes": { "o": "OutDensity" } })).unwrap();
        let mut errors = vec![];
        check_graph(&g, false, Path::new("."), &mut errors, &mut vec![]);
        assert!(errors.iter().any(|e| e.contains("input 0 is not connected")));
    }
}

#[cfg(test)]
mod example_tests {
    use super::*;

    /// Every spec in examples/Specs builds into a graph that validates (a missing heightmap PNG is the one thing allowed to be
    /// absent: those are made by the user's own Landscape export). Every example file validates too.
    #[test]
    fn the_examples_build_and_validate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let mut specs = 0;
        for entry in std::fs::read_dir(root.join("Specs")).unwrap() {
            let path = entry.unwrap().path();
            if !path.to_string_lossy().ends_with(".spec.json") {
                continue;
            }
            let spec: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let g = build(&spec).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let ext = if g.nodes.values().any(|n| matches!(n.kind, NodeKind::OutDensity)) { "noisegraph" } else { "meshgraph" };
            let built = path.with_file_name(format!("_built_{specs}.{ext}"));
            std::fs::write(&built, g.to_json()).unwrap();
            let (errors, _) = validate(&built).unwrap();
            std::fs::remove_file(&built).unwrap();
            let real: Vec<_> = errors.iter().filter(|e| !(e.contains("does not exist") && e.contains(".png"))).collect();
            assert!(real.is_empty(), "{}: {real:?}", path.display());
            specs += 1;
        }
        assert!(specs >= 3);
        for dir in ["Noises", "SDFs", "Mesh Graphs"] {
            for entry in std::fs::read_dir(root.join(dir)).unwrap() {
                let path = entry.unwrap().path();
                if matches!(path.extension().and_then(|e| e.to_str()), Some("noisegraph" | "meshgraph" | "sdf")) {
                    let (errors, _) = validate(&path).unwrap();
                    assert!(errors.is_empty(), "{}: {errors:?}", path.display());
                }
            }
        }
    }
}
