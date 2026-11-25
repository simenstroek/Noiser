//! Compiles a graph into WGSL: `graph_density(p) -> f32` and `graph_normal(p) -> vec3f`.
//!
//! Every node upstream of an output becomes one SSA `let` in a per-cone function. Node parameters
//! are read from the storage array `P` (filled from `Compiled::params`), so editing a value never
//! changes the WGSL text. Only topology changes do, and the app rebuilds the pipeline when it differs.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, PatternKind};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

pub const NOISE_LIB: &str = include_str!("noise.wgsl");

const MESH_MASK_LIB: &str = include_str!("mesh_mask.wgsl");

const PATTERN_LIB: &str = include_str!("patterns.wgsl");

const WORLEY_GENERAL_LIB: &str = include_str!("worley_general.wgsl");

/// `mesh_mask` as the shaders call it: the real one when a mask is in use, otherwise one that
/// returns 1 and costs nothing to compile.
pub fn mesh_mask_lib(on: bool) -> &'static str {
    if on {
        MESH_MASK_LIB
    } else {
        "fn mesh_mask(t: texture_3d<f32>, p: vec3f, origin: vec3f, voxel: f32, blo: vec3f, bhi: vec3f, terms: array<vec4f, 12>, count: u32) -> f32 { return 1.0; }\n"
    }
}

/// Minimal bindings the noise lib needs, for standalone evaluators (tests, export).
/// Binding numbers match common.wgsl.
#[allow(dead_code)] // used by tests now, texture export in milestone 5
pub const EVAL_BINDINGS: &str = "struct U { seed: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read> BN: array<f32>;";

#[derive(Debug)]
pub struct Compiled {
    pub wgsl: String,
    pub params: Vec<f32>,
    /// The loaded-mesh nodes (mesh graph), in the order of the `mesh_dist_0`, `mesh_dist_1`... functions they call.
    pub mesh_slots: Vec<NodeId>,
    /// The terrain maps the graph's Map (Mask) nodes read, by name, sorted (see `terrain_maps`).
    pub masks: Vec<String>,
    /// Where each Map node's two parameters sit in `params`, and the map it reads: the evaluator writes the map's
    /// offset (in units of `terrain_maps::ALIGN` from `MASK_BASE`) and side there once the maps are packed.
    pub map_params: Vec<(usize, String)>,
}

/// Samples along a painted mask's side (other terrain maps have sides of their own).
pub const MASK_RES: usize = 1024;
/// Where the terrain maps start in the parameter buffer (the same number as `MASK_BASE` in noise.wgsl).
pub const MASK_BASE: usize = 1 << 20;

/// The distinct names of the graph's Mask nodes, sorted.
fn mask_names(g: &Graph) -> Vec<String> {
    g.nodes.values().filter_map(|n| if let NodeKind::Mask { name } = &n.kind { Some(name.clone()) } else { None }).collect::<BTreeSet<_>>().into_iter().collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    F,
    V,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::F => "f32",
            Ty::V => "vec3f",
        }
    }
}

fn convert(expr: &str, from: Ty, to: Ty) -> String {
    match (from, to) {
        (Ty::F, Ty::V) => format!("vec3f({expr})"),
        // Blender's implicit vector → float conversion: the average.
        (Ty::V, Ty::F) => format!("dot({expr}, vec3f(1.0 / 3.0))"),
        _ => expr.to_string(),
    }
}

struct Gen<'a> {
    g: &'a Graph,
    caches: &'a BTreeMap<NodeId, u32>,
    slots: BTreeMap<NodeId, usize>,
    types: BTreeMap<NodeId, Ty>,
    /// Cone functions already written (keyed by their root node).
    done: BTreeSet<NodeId>,
    building: BTreeSet<NodeId>,
    /// Noise kinds used, each getting its own fBm function.
    kinds: BTreeSet<NoiseKind>,
    mesh_slots: Vec<NodeId>,
    eps: f32,
    /// Load SDF nodes whose interpreter the shader needs.
    sdf_used: BTreeSet<NodeId>,
    out: String,
}

pub fn compile(g: &Graph) -> Result<Compiled, String> {
    compile_inner(g, &BTreeMap::new(), NORMAL_EPS, false)
}

/// The step the Normal node differentiates over, in position units, unless told otherwise.
pub const NORMAL_EPS: f32 = 1e-3;

/// Like `compile`, but nodes in `caches` read their baked texture (`cache_sample(slot, p)`)
/// instead of evaluating their upstream graph. Preview only: when the Density output reads a
/// vector, the result also defines `graph_color(p)`, the vector as a colour (see view.rs).
pub fn compile_with(g: &Graph, caches: &BTreeMap<NodeId, u32>) -> Result<Compiled, String> {
    compile_inner(g, caches, NORMAL_EPS, true)
}

/// `compile` with the Normal node's differentiation step: the 1e-3 of the noise graph is right for tiles of one
/// unit, but a mesh graph works in the units of its mesh files, where a step should follow the size of the mesh.
pub fn compile_eps(g: &Graph, caches: &BTreeMap<NodeId, u32>, eps: f32) -> Result<Compiled, String> {
    compile_inner(g, caches, eps, false)
}

fn compile_inner(g: &Graph, caches: &BTreeMap<NodeId, u32>, eps: f32, preview: bool) -> Result<Compiled, String> {
    let lowered = g.lowered();
    let g = &lowered;
    // Parameter slots, in node-id order.
    let mut slots = BTreeMap::new();
    let mut params = Vec::new();
    let mask_names = mask_names(g);
    let mut map_params = Vec::new();
    for (&id, n) in &g.nodes {
        slots.insert(id, params.len());
        match &n.kind {
            NodeKind::Float(v) => params.push(*v),
            NodeKind::Vec3(v) => params.extend(v),
            NodeKind::Math { a, b, c, .. } => params.extend([*a, *b, *c]),
            NodeKind::Remap { vals, clamp } => {
                params.extend(vals);
                params.push(if *clamp { 1.0 } else { 0.0 });
            }
            NodeKind::Curve { points, mode } => {
                let p = crate::graph::curve_points(points);
                params.push(*mode as u32 as f32);
                params.push(p.len() as f32);
                for i in 0..crate::graph::MAX_CURVE_POINTS {
                    params.extend(p.get(i).copied().unwrap_or(*p.last().unwrap()));
                }
            }
            NodeKind::Combine { vals } => params.extend(vals),
            NodeKind::Position { scale } => params.extend(scale),
            NodeKind::Random { vals, seed } => params.extend([vals[0], vals[1], seed.map_or(-1.0, |s| s as f32)]),
            NodeKind::Mask { name } => {
                // Filled in by the evaluator: the map's offset and side.
                map_params.push((params.len(), name.clone()));
                params.extend([0.0, 2.0]);
            }
            NodeKind::Slope { scale } => params.push(*scale),
            NodeKind::Select { lo, hi, soft } => params.extend([*lo, *hi, *soft]),
            NodeKind::Radial(r) => {
                params.push(if r.mode == crate::graph::RadialMode::Spherical { 1.0 } else { 0.0 });
                params.extend(r.center);
                params.extend(r.axis);
            }
            NodeKind::Mix { vals, clamp } => {
                params.extend(vals);
                params.push(if *clamp { 1.0 } else { 0.0 });
            }
            NodeKind::Grid { counts, corners, cubic, .. } => {
                params.extend(counts);
                params.push(if *corners { 1.0 } else { 0.0 });
                params.push(if *cubic { 1.0 } else { 0.0 });
            }
            // 19 slots: counts (3), steps, gap, bevel, output, 2D, axis, seed (-1 = global), variant,
            // radius, profile, direction (3), width, softness, iterations. Each kind reads the ones it uses.
            NodeKind::Pattern(p) => params.extend([
                p.counts[0],
                p.counts[1],
                p.counts[2],
                p.steps as f32,
                p.gap,
                p.bevel,
                p.output as u32 as f32,
                if p.mode2d { 1.0 } else { 0.0 },
                p.axis as f32,
                p.seed.map_or(-1.0, |s| s as f32),
                p.variant as f32,
                p.radius,
                p.profile as f32,
                p.dir[0],
                p.dir[1],
                p.dir[2],
                p.width,
                p.softness,
                p.iterations as f32,
            ]),
            NodeKind::LoadSdf(p) => {
                let loaded = crate::sdf_node::load(&p.file);
                params.extend(p.shader_params(loaded.as_ref().as_ref().ok()));
            }
            NodeKind::Scatter(p) => params.extend(p.shader_params()),
            NodeKind::Noise(n) => params.extend([
                n.scale,
                if n.normalize { 1.0 } else { 0.0 },
                n.octaves_used() as f32,
                n.lacunarity,
                n.roughness,
                n.distortion,
                n.seed.map_or(-1.0, |s| s as f32), // shortcut: f32 holds seeds exactly only below 2^24
                n.feature as u32 as f32,
                n.metric as u32 as f32,
                n.jitter,
            ]),
            _ => {}
        }
    }

    let mesh_slots: Vec<NodeId> = g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).collect();
    let mut cg = Gen { g, caches, slots, types: BTreeMap::new(), done: BTreeSet::new(), building: BTreeSet::new(), kinds: BTreeSet::new(), mesh_slots: mesh_slots.clone(), eps, sdf_used: BTreeSet::new(), out: String::new() };
    let density = g.find(|k| *k == NodeKind::OutDensity).and_then(|o| g.input_from(o, 0));
    let normal = g.find(|k| *k == NodeKind::OutNormal).and_then(|o| g.input_from(o, 0));
    // One output of a Separate node is a component of the vector its function returns.
    let component = |src: NodeId, port: usize| if matches!(g.nodes[&src].kind, NodeKind::Separate) { Some(["x", "y", "z"][port.min(2)]) } else { None };

    let mut color_source = None;
    let density_expr = match density {
        Some((src, port)) => {
            let ty = cg.cone(src)?;
            match component(src, port) {
                Some(c) => format!("c{src}(p).{c}"),
                None => {
                    if ty == Ty::V {
                        color_source = Some(src);
                    }
                    convert(&format!("c{src}(p)"), ty, Ty::F)
                }
            }
        }
        None => "0.0".into(),
    };
    writeln!(cg.out, "fn graph_density(p: vec3f) -> f32 {{ return {density_expr}; }}").unwrap();
    if let (true, Some(src)) = (preview, color_source) {
        writeln!(cg.out, "fn graph_color(p: vec3f) -> vec3f {{ return safe_normalize(c{src}(p)) * 0.5 + 0.5; }}").unwrap();
    }

    let normal_expr = match normal {
        Some((src, port)) => {
            let ty = cg.cone(src)?;
            match component(src, port) {
                Some(c) => format!("safe_normalize(vec3f(c{src}(p).{c}))"),
                None => format!("safe_normalize({})", convert(&format!("c{src}(p)"), ty, Ty::V)),
            }
        }
        None => "tetra_grad_density(p)".into(),
    };
    writeln!(cg.out, "fn tetra_grad_density(p: vec3f) -> vec3f {{ {} }}", tetra_body(eps, |q| format!("graph_density({q})"))).unwrap();
    writeln!(cg.out, "fn graph_normal(p: vec3f) -> vec3f {{ return {normal_expr}; }}").unwrap();

    if params.is_empty() {
        params.push(0.0); // storage buffers cannot be empty
    }
    // The pattern kernels are only worth compiling when there is a pattern node.
    let mut wgsl = if g.nodes.values().any(|n| matches!(n.kind, NodeKind::Pattern(_))) { PATTERN_LIB.to_string() } else { String::new() };
    // The general Worley (metrics, jitter, cell edges) only when a node uses it.
    let general_worley = g.nodes.values().any(|n| matches!(&n.kind, NodeKind::Noise(p) if p.kind == NoiseKind::Worley
        && (p.metric != crate::graph::WorleyMetric::Euclidean || p.jitter != 1.0 || p.feature == crate::graph::WorleyFeature::Edges)));
    if general_worley {
        wgsl += WORLEY_GENERAL_LIB;
    }
    wgsl += &cg.kinds.iter().map(|&k| fbm_function(k, general_worley)).collect::<String>();
    for id in &cg.sdf_used {
        if let NodeKind::Scatter(p) = &g.nodes[id].kind
            && let Ok(data) = crate::sdf_node::load(&p.file).as_ref()
            && let Some(land) = cg.land_slot()
        {
            wgsl += &crate::sdf_node::embed(&format!("sdf{id}_"), &data.scene, true);
            wgsl += &crate::scatter::wgsl(*id, cg.slots[id], land);
        }
        if let NodeKind::LoadSdf(p) = &g.nodes[id].kind
            && let Ok(data) = crate::sdf_node::load(&p.file).as_ref()
        {
            wgsl += &crate::sdf_node::embed(&format!("sdf{id}_"), &data.scene, p.z_up);
            if p.stamp {
                // Sphere tracing straight down from the top of the shape's box: the first hit is its top surface.
                wgsl += &format!("
fn sdf{id}_top(q: vec2f, off: vec3f, s: f32, c: vec3f, zhi: f32, zlo: f32) -> f32 {{
    let eps = 0.0005 * (zhi - zlo);
    var z = zhi;
    for (var i = 0; i < 96; i++) {{
        let d = sdf{id}_at(vec3f(q, z), off, s, c);
        if (d < eps) {{ return z; }}
        z -= max(d, eps);
        if (z < zlo) {{ break; }}
    }}
    return -1000.0;
}}
");
            }
        }
    }
    wgsl += &cg.out;
    Ok(Compiled { wgsl, params, mesh_slots, masks: mask_names, map_params })
}

/// Body of a function returning the normalized tetrahedral gradient (IQ) of the float field `sample(q)`.
/// A loop, so the field is inlined once rather than four times (driver compile time).
fn tetra_body(eps: f32, sample: impl Fn(&str) -> String) -> String {
    format!(
        "let e = {eps:e}; var g = vec3f(0.0); for (var k = 0u; k < 4u; k++) {{ let o = tetra(k); g += o * {}; }} return safe_normalize(g);",
        sample("p + o * e")
    )
}

const FBM_TEMPLATE: &str = include_str!("noise_fbm.wgsl");

/// The fBm wrapper for one noise kind (see noise_fbm.wgsl).
fn fbm_function(kind: NoiseKind, general_worley: bool) -> String {
    let base = match kind {
        NoiseKind::Perlin => "clamp(perlin(x, period, seed + o), -1.0, 1.0)",
        NoiseKind::Simplex => "clamp(simplex(x, period, seed + o), -1.0, 1.0)",
        NoiseKind::Worley if general_worley => "worley(x, period, seed + o, feature, metric, jitter)",
        NoiseKind::Worley => "worley_classic(x, period, seed + o, feature)",
        NoiseKind::White => "white(x, period, seed + o)",
        NoiseKind::Blue => "blue(x, period, seed + o)",
    };
    FBM_TEMPLATE
        .replace("KIND", &(kind as u32).to_string())
        .replace("HAS_OCTAVES", if kind.has_octaves() { "true" } else { "false" })
        .replace("BASE", base)
}

impl Gen<'_> {
    /// The mesh slot of the terrain: the first Load Mesh node that has a heightmap.
    fn land_slot(&self) -> Option<usize> {
        self.mesh_slots.iter().position(|id| matches!(&self.g.nodes[id].kind, NodeKind::LoadMesh(p) if crate::landscape_field::is_heightmap(&p.path)))
    }

    /// Writes `fn c{root}(p) -> T` evaluating everything upstream of `root`. Returns T.
    fn cone(&mut self, root: NodeId) -> Result<Ty, String> {
        if self.done.contains(&root) {
            return Ok(self.types[&root]);
        }
        // A cone that (through a Normal node) needs itself is a cycle.
        if !self.building.insert(root) {
            return Err("graph has a cycle".into());
        }
        let mut order = Vec::new();
        self.topo(root, &mut BTreeSet::new(), &mut BTreeSet::new(), &mut order)?;

        let mut body = String::new();
        for &id in &order {
            let line = self.node_expr(id)?;
            // Wired parameter pins override the node's parameter slots while it is evaluated (see `pp` in noise.wgsl).
            let wired = self.wired_params(id);
            for (k, (slot, e)) in wired.iter().enumerate() {
                writeln!(body, "    ov_i[{k}] = {}u; ov_v[{k}] = {e};", self.slots[&id] + slot).unwrap();
            }
            if !wired.is_empty() {
                writeln!(body, "    ov_n = {}u;", wired.len()).unwrap();
            }
            writeln!(body, "    let v{id} = {line};").unwrap();
            if !wired.is_empty() {
                writeln!(body, "    ov_n = 0u;").unwrap();
            }
        }
        let ty = self.types[&root];
        writeln!(self.out, "fn c{root}(p: vec3f) -> {} {{\n{body}    return v{root};\n}}", ty.name()).unwrap();
        self.done.insert(root);
        Ok(ty)
    }

    /// Post-order DFS. Stops at Normal nodes: their input gets its own cone function.
    fn topo(
        &mut self,
        id: NodeId,
        visiting: &mut BTreeSet<NodeId>,
        seen: &mut BTreeSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<(), String> {
        if seen.contains(&id) {
            return Ok(());
        }
        if !visiting.insert(id) {
            return Err("graph has a cycle".into());
        }
        let kind = &self.g.nodes[&id].kind;
        if self.caches.contains_key(&id) {
            // A cached node is a texture read; its upstream graph isn't evaluated here.
        } else if matches!(kind, NodeKind::Normal { .. } | NodeKind::Slope { .. }) {
            if let Some(src) = self.g.input(id, 0) {
                self.cone(src)?;
            }
        } else {
            for port in 0..kind.input_count() {
                if let Some(src) = self.g.input(id, port) {
                    self.topo(src, visiting, seen, order)?;
                }
            }
        }
        visiting.remove(&id);
        seen.insert(id);
        order.push(id);
        Ok(())
    }

    /// The (slot, expression) of each wired parameter pin of a Noise or Pattern node.
    fn wired_params(&self, id: NodeId) -> Vec<(usize, String)> {
        if self.caches.contains_key(&id) || !matches!(self.g.nodes[&id].kind, NodeKind::Noise(_) | NodeKind::Pattern(_)) {
            return Vec::new();
        }
        self.g.nodes[&id].exposed().filter_map(|(port, pin)| Some((pin.slot?, self.operand(id, port, Ty::F, "")))).filter(|(_, e)| !e.is_empty()).collect()
    }

    /// The input on `port` as an expression of type `want`, or `fallback` when unconnected.
    fn operand(&self, id: NodeId, port: usize, want: Ty, fallback: &str) -> String {
        match self.source(id, port) {
            Some((expr, ty)) => convert(&expr, ty, want),
            None => fallback.to_string(),
        }
    }

    /// The expression feeding input `port` of `id` and its type. A Separate node holds its input vector, and
    /// the output port picks a component.
    fn source(&self, id: NodeId, port: usize) -> Option<(String, Ty)> {
        let (src, from_port) = self.g.input_from(id, port)?;
        Some(if matches!(self.g.nodes[&src].kind, NodeKind::Separate) {
            (format!("v{src}.{}", ["x", "y", "z"][from_port.min(2)]), Ty::F)
        } else {
            (format!("v{src}"), self.types[&src])
        })
    }

    fn node_expr(&mut self, id: NodeId) -> Result<String, String> {
        if let Some(slot) = self.caches.get(&id) {
            self.types.insert(id, Ty::F);
            return Ok(format!("cache_sample({slot}u, p)"));
        }
        let s = self.slots[&id];
        let (ty, expr) = match &self.g.nodes[&id].kind {
            NodeKind::Float(_) => (Ty::F, format!("P[{s}]")),
            NodeKind::Vec3(_) => (Ty::V, format!("vec3f(P[{s}], P[{}], P[{}])", s + 1, s + 2)),
            NodeKind::Position { .. } => {
                let x: Vec<String> = (0..3).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::V, format!("p * vec3f({}, {}, {})", x[0], x[1], x[2]))
            }
            NodeKind::Radial(_) => {
                let q = self.operand(id, 0, Ty::V, "p");
                (Ty::V, format!("radial_coords({q}, vec3f(P[{}], P[{}], P[{}]), vec3f(P[{}], P[{}], P[{}]), P[{s}] > 0.5)", s + 1, s + 2, s + 3, s + 4, s + 5, s + 6))
            }
            // A loaded mesh: the shader prelude of the evaluator defines mesh_dist_N(p) for each slot.
            NodeKind::LoadMesh(_) => {
                let slot = self.mesh_slots.iter().position(|&n| n == id).unwrap_or(0);
                (Ty::F, format!("mesh_dist_{slot}({})", self.operand(id, 0, Ty::V, "p")))
            }
            // A composition: the shader gets its own copy of the interpreter and the shapes (see sdf_node.rs). An empty or
            // unreadable file gives a far-away positive distance.
            NodeKind::LoadSdf(p) => {
                let q = self.operand(id, 0, Ty::V, "p");
                match crate::sdf_node::load(&p.file).as_ref() {
                    Ok(_) if p.stamp => {
                        self.sdf_used.insert(id);
                        let (lo, hi) = p.field_bounds().unwrap_or(([0.0; 3], [1.0; 3]));
                        (Ty::F, format!("sdf{id}_top({q}.xy, vec3f(P[{s}], P[{}], P[{}]), P[{}], vec3f(P[{}], P[{}], P[{}]), {:e}, {:e})", s + 1, s + 2, s + 3, s + 4, s + 5, s + 6, hi[2] + 1e-4, lo[2]))
                    }
                    Ok(_) => {
                        self.sdf_used.insert(id);
                        (Ty::F, format!("sdf{id}_at({q}, vec3f(P[{s}], P[{}], P[{}]), P[{}], vec3f(P[{}], P[{}], P[{}]))", s + 1, s + 2, s + 3, s + 4, s + 5, s + 6))
                    }
                    Err(_) => (Ty::F, "1000.0".into()),
                }
            }
            // Copies of a shape across the terrain: the shader gets its own copy of the interpreter and a function (scatter.rs).
            NodeKind::Scatter(p) => {
                let base = self.operand(id, 0, Ty::F, "1.0e6");
                let displace = self.operand(id, 1, Ty::F, "0.0");
                if self.land_slot().is_none() {
                    return Err("a Scatter SDF node needs a terrain: add a Load Mesh node with a heightmap".into());
                }
                match crate::sdf_node::load(&p.file).as_ref() {
                    Ok(_) => {
                        self.sdf_used.insert(id);
                        (Ty::F, format!("scatter{id}(p, {base}, {displace})"))
                    }
                    Err(_) => (Ty::F, base),
                }
            }
            NodeKind::NoiseGraph(_) => return Err("a Noise Graph node must be lowered first (meshgraph::lowered_mesh)".into()),
            NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv => return Err("this node does not produce a field".into()),
            NodeKind::Heightmap(_) => return Err("a Heightmap node works only in the Landscape graph, on a terrain".into()),
            NodeKind::Layer(_) => return Err("a Layer node is an output of the Landscape graph, not part of a field".into()),
            NodeKind::Erosion(_) => return Err("an Erosion node works only in the Landscape graph, on a terrain".into()),
            // Never reached (readers look through them), but the match must be complete.
            NodeKind::NamedReroute { .. } | NodeKind::NamedRerouteUse { .. } => (Ty::F, "0.0".into()),
            NodeKind::Reroute => {
                let ty = self.source(id, 0).map_or(Ty::F, |(_, ty)| ty);
                (ty, self.operand(id, 0, ty, &convert("0.0", Ty::F, ty)))
            }
            NodeKind::Math { op, .. } => {
                let in_ty = |port| self.source(id, port).map_or(Ty::F, |(_, ty)| ty);
                // Dot takes vectors and gives a float; everything else is a vector if any input is.
                let opty = if *op == MathOp::Dot || in_ty(0) == Ty::V || in_ty(1) == Ty::V { Ty::V } else { Ty::F };
                let ty = if *op == MathOp::Dot { Ty::F } else { opty };
                let a = self.operand(id, 0, opty, &convert(&format!("P[{s}]"), Ty::F, opty));
                let b = self.operand(id, 1, opty, &convert(&format!("P[{}]", s + 1), Ty::F, opty));
                let k = self.operand(id, 2, Ty::F, &format!("P[{}]", s + 2));
                let zero = convert("0.0", Ty::F, opty);
                let v = opty == Ty::V;
                let e = match op {
                    MathOp::Add => format!("{a} + {b}"),
                    MathOp::Subtract => format!("{a} - {b}"),
                    MathOp::Multiply => format!("{a} * {b}"),
                    // Blender's safe divide: x / 0 = 0.
                    MathOp::Divide => format!("select({a} / {b}, {zero}, {b} == {zero})"),
                    MathOp::Min => format!("min({a}, {b})"),
                    MathOp::Max => format!("max({a}, {b})"),
                    MathOp::Power => format!("{}({a}, {b})", if v { "pow_safe3" } else { "pow_safe" }),
                    MathOp::SmoothMin => format!("{}({a}, {b}, {k})", if v { "smin3" } else { "smin" }),
                    MathOp::SmoothMax => format!("{}({a}, {b}, {k})", if v { "smax3" } else { "smax" }),
                    MathOp::Lerp => format!("{a} + ({b} - {a}) * {}", convert(&k, Ty::F, opty)),
                    MathOp::InverseLerp => format!("select(({} - {a}) / ({b} - {a}), {zero}, {b} == {a})", convert(&k, Ty::F, opty)),
                    MathOp::Dot => format!("dot({a}, {b})"),
                    MathOp::Absolute => format!("abs({a})"),
                    MathOp::Ridge => format!("{} - abs({a})", convert("1.0", Ty::F, opty)),
                    MathOp::Floor => format!("floor({a})"),
                    MathOp::Fract => format!("fract({a})"),
                    MathOp::Sine => format!("sin({a})"),
                    MathOp::Sqrt => format!("sqrt(max({a}, {zero}))"),
                    MathOp::Modulo => format!("select({a} - {b} * floor({a} / {b}), {zero}, {b} == {zero})"),
                    MathOp::Greater => format!("select({zero}, {}, {a} > {b})", convert("1.0", Ty::F, opty)),
                };
                (ty, e)
            }
            NodeKind::Remap { .. } => {
                let x: Vec<String> = (0..5).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::F, format!("remap({}, {}, {}, {}, {}, P[{}] > 0.5)", x[0], x[1], x[2], x[3], x[4], s + 5))
            }
            // The vector itself; readers take the component their pin names (see `source`).
            NodeKind::Separate => (Ty::V, self.operand(id, 0, Ty::V, "vec3f(0.0)")),
            NodeKind::Combine { .. } => {
                let x: Vec<String> = (0..3).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::V, format!("vec3f({}, {}, {})", x[0], x[1], x[2]))
            }
            NodeKind::Random { .. } => {
                let x: Vec<String> = (0..2).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::F, format!("random_range({}, {}, P[{}], {id}u)", x[0], x[1], s + 2))
            }
            NodeKind::Mix { .. } => {
                let x: Vec<String>  = (0..3).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::F, format!("mix_ab({}, {}, {}, P[{}] > 0.5)", x[0], x[1], x[2], s + 3))
            }
            NodeKind::Curve { .. } => {
                let x = self.operand(id, 0, Ty::F, "0.0");
                (Ty::F, format!("curve_eval({x}, {}u, u32(P[{}]), u32(P[{}]))", s + 2, s + 1, s))
            }
            NodeKind::Grid { .. } => {
                let q = self.operand(id, 0, Ty::V, "p");
                let count = |k: usize| if self.g.nodes[&id].port_visible(1 + k) { self.operand(id, 1 + k, Ty::F, &format!("P[{}]", s + k)) } else { format!("P[{}]", s + k) };
                (Ty::F, format!("grid_dist({q}, vec3f({}, {}, {}), P[{}] > 0.5, P[{}] > 0.5)", count(0), count(1), count(2), s + 3, s + 4))
            }
            NodeKind::Pattern(p) => {
                let q = self.operand(id, 0, Ty::V, "p");
                let function = match p.kind {
                    PatternKind::Bricks => "bricks",
                    PatternKind::Dimples => "dimples",
                    PatternKind::Checker => "checker",
                    PatternKind::Stripes => "stripes",
                    PatternKind::Gyroid => "gyroid",
                    PatternKind::Fractal => "fractal",
                    PatternKind::Truchet => "truchet",
                    PatternKind::Honeycomb => "honeycomb",
                };
                (Ty::F, format!("{function}({q}, {s}u)"))
            }
            NodeKind::Select { .. } => {
                let x = self.operand(id, 0, Ty::F, "0.0");
                (Ty::F, format!("select_range({x}, P[{s}], P[{}], P[{}])", s + 1, s + 2))
            }
            NodeKind::Mask { .. } => {
                let q = self.operand(id, 0, Ty::V, "p");
                (Ty::F, format!("terrain_map(P[{s}], P[{}], {q})", s + 1))
            }
            NodeKind::Slope { .. } => match self.g.input_from(id, 0) {
                Some((src, from_port)) => {
                    // The input's cone function was written during topo(); add its raw gradient once. One output of a
                    // Separate node is a component of the vector its function returns.
                    let part = if matches!(self.g.nodes[&src].kind, NodeKind::Separate) { Some(["x", "y", "z"][from_port.min(2)]) } else { None };
                    let gname = format!("gr{src}{}", part.map_or(String::new(), |c| format!("_{c}")));
                    if !self.out.contains(&format!("fn {gname}(")) {
                        let ty = if part.is_some() { Ty::F } else { self.types[&src] };
                        let eps = self.eps;
                        let sample = convert(&format!("c{src}(p + o * e){}", part.map_or(String::new(), |c| format!(".{c}"))), ty, Ty::F);
                        writeln!(self.out, "fn {gname}(p: vec3f) -> vec3f {{ let e = {eps:e}; var g = vec3f(0.0); for (var k = 0u; k < 4u; k++) {{ let o = tetra(k); g += o * {sample}; }} return g / (4.0 * e); }}").unwrap();
                    }
                    (Ty::F, format!("slope_degrees({gname}(p).xy, P[{s}])"))
                }
                None => (Ty::F, "0.0".into()),
            },
            NodeKind::Normal { flip } => {
                let sign = if *flip { "-" } else { "" };
                match self.g.input_from(id, 0) {
                    Some((src, from_port)) => {
                        // The input's cone function was written during topo(); add its gradient once.
                        let part = if matches!(self.g.nodes[&src].kind, NodeKind::Separate) { Some(["x", "y", "z"][from_port.min(2)]) } else { None };
                        let gname = format!("g{src}{}", part.map_or(String::new(), |c| format!("_{c}")));
                        if !self.out.contains(&format!("fn {gname}(")) {
                            let ty = if part.is_some() { Ty::F } else { self.types[&src] };
                            let body = tetra_body(self.eps, |q| convert(&format!("c{src}({q}){}", part.map_or(String::new(), |c| format!(".{c}"))), ty, Ty::F));
                            writeln!(self.out, "fn {gname}(p: vec3f) -> vec3f {{ {body} }}").unwrap();
                        }
                        (Ty::V, format!("{sign}{gname}(p)"))
                    }
                    None => (Ty::V, "vec3f(0.0)".into()),
                }
            }
            NodeKind::Noise(n) => {
                let q = self.operand(id, 0, Ty::V, "p");
                self.kinds.insert(n.kind);
                (Ty::F, format!("noise_k{}({q}, {s}u)", n.kind as u32))
            }
            NodeKind::OutDensity | NodeKind::OutNormal => return Err("output nodes cannot feed other nodes".into()),
        };
        self.types.insert(id, ty);
        Ok(expr)
    }
}

/// A random graph using every node kind, with random links (no cycles) and muted nodes, for tests.
#[cfg(test)]
pub(crate) fn random_graph(rnd: &mut impl FnMut(usize) -> usize) -> Graph {
    use crate::graph::{CurveMode, NodeKind as K, NoiseParams, PatternKind, PatternParams, WorleyFeature, WorleyMetric};
    let mut g = Graph::default();
    let count = 3 + rnd(6);
    for i in 0..count {
        let kind = match rnd(16) {
            0 => K::Float(rnd(5) as f32 - 2.0),
            1 => K::Vec3([rnd(3) as f32, 0.5, -1.0]),
            2 => K::position(),
            3 => K::Math { op: MathOp::ALL[rnd(MathOp::ALL.len())], a: 0.5, b: 2.0, c: 0.3 },
            4 => K::Remap { vals: [0.0, 0.0, 1.0, -1.0, 2.0], clamp: rnd(2) == 0 },
            5 => K::Mix { vals: [0.5, 0.0, 1.0], clamp: rnd(2) == 0 },
            6 => K::Curve { points: vec![[1.0, 0.0], [0.0, 1.0], [0.5, 3.0]][..1 + rnd(3)].to_vec(), mode: CurveMode::ALL[rnd(4)] },
            7 => K::Grid { counts: [4.0; 3], corners: rnd(2) == 0, cubic: rnd(2) == 0, amount: 1.0 },
            8 => K::Pattern(PatternParams::for_kind(PatternKind::ALL[rnd(8)])),
            9 => K::Normal { flip: rnd(2) == 0 },
            10 => K::Reroute,
            11 => K::Separate,
            12 => K::Combine { vals: [0.5, 1.0, -1.0] },
            13 => K::NamedReroute { name: format!("n{i}") },
            14 => K::NamedRerouteUse { name: "none".into() },
            _ => K::Noise(NoiseParams {
                kind: crate::graph::NoiseKind::ALL[rnd(5)],
                feature: [WorleyFeature::F1, WorleyFeature::F2, WorleyFeature::F2MinusF1, WorleyFeature::CellValue, WorleyFeature::Edges][rnd(5)],
                metric: [WorleyMetric::Euclidean, WorleyMetric::Manhattan, WorleyMetric::Chebyshev][rnd(3)],
                jitter: [1.0, 0.5][rnd(2)],
                ..NoiseParams::default()
            }),
        };
        let id = g.add(kind, [i as f32 * 50.0, 0.0]);
        g.nodes.get_mut(&id).unwrap().muted = rnd(7) == 0;
    }
    // A use reads a declaration with a lower id (or nothing), so names cannot make cycles either.
    let decls: Vec<(u32, String)> = g.nodes.iter().filter_map(|(id, n)| if let K::NamedReroute { name } = &n.kind { Some((*id, name.clone())) } else { None }).collect();
    for (id, n) in g.nodes.iter_mut() {
        if let K::NamedRerouteUse { name } = &mut n.kind {
            if let Some((_, d)) = decls.iter().rev().find(|(did, _)| did < id) {
                *name = d.clone();
            }
        }
    }
    // Links only from earlier to later nodes, so there are no cycles.
    for to in 1..count as u32 {
        if matches!(g.nodes[&to].kind, K::Position { .. }) {
            continue; // a scaled Position has no .noiser form, and the export test reads these graphs
        }
        for port in 0..g.nodes[&to].kind.input_count() {
            if rnd(10) < 6 {
                let from = rnd(to as usize) as u32;
                let from_port = if matches!(g.nodes[&from].kind, K::Separate) { rnd(3) } else { 0 };
                g.connect_from(from, from_port, to, port);
            }
        }
    }
    let out = g.add(K::OutDensity, [0.0; 2]);
    let normal_out = g.add(K::OutNormal, [0.0; 2]);
    g.connect(count as u32 - 1, out, 0);
    let normal_src = rnd(count) as u32;
    let normal_port = if matches!(g.nodes[&normal_src].kind, K::Separate) { rnd(3) } else { 0 };
    g.connect_from(normal_src, normal_port, normal_out, 0);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MathOp, NoiseParams};

    #[test]
    fn every_template_is_a_valid_graph_that_compiles() {
        for name in Graph::TEMPLATES {
            let g = Graph::template(name);
            assert!(Graph::from_json(&g.to_json()).is_ok(), "{name}");
            validate(&compile(&g).unwrap());
            assert!(g.tiling().values().all(|t| !matches!(t, crate::graph::Tiling::Broken(_))), "{name} should tile");
        }
    }

    /// Parses and validates a full shader made of the noise lib, the generated graph, and a stub entry point.
    fn validate(c: &Compiled) {
        let src = format!(
            "{NOISE_LIB}\n{}\n{EVAL_BINDINGS}\n@fragment fn fs() -> @location(0) vec4f {{ return vec4f(graph_normal(vec3f(0.5)), graph_density(vec3f(0.5))); }}",
            c.wgsl
        );
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}\n---\n{src}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}\n---\n{src}"));
    }

    /// Random graphs from every node kind, with random links and muted nodes, must compile to valid WGSL.
    #[test]
    fn random_graphs_compile_to_valid_shaders() {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rnd = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let rounds: usize = std::env::var("NOISER_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        for round in 0..rounds {
            let g = random_graph(&mut rnd);
            let c = compile(&g).unwrap_or_else(|e| panic!("round {round}: {e}
{}", g.to_json()));
            validate(&c);
            if round % 20 == 0 {
                let src = crate::view::shader_source(&c.wgsl, round % 40 == 0, crate::view::Extras { mask_view: round % 40 == 0 });
                let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
            }
        }
    }

    #[test]
    fn default_graph_compiles() {
        let c = compile(&Graph::default_graph()).unwrap();
        assert!(c.wgsl.contains("noise_k0(v0, "));
        // Only the kinds used get an fBm function.
        assert!(c.wgsl.contains("fn noise_k0(") && !c.wgsl.contains("fn noise_k4("));
        validate(&c);
    }

    #[test]
    fn preview_shader_validates() {
        let c = compile(&Graph::default_graph()).unwrap();
        let src = crate::view::shader_source(&c.wgsl, true, crate::view::Extras { mask_view: true });
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
    }

    #[test]
    fn vector_preview_defines_a_colour_only_for_previews() {
        let mut g = Graph::default();
        let position = g.add(NodeKind::position(), [0.0; 2]);
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(position, out, 0);
        assert!(!compile(&g).unwrap().wgsl.contains("fn graph_color("));
        let preview = compile_with(&g, &BTreeMap::new()).unwrap();
        assert!(preview.wgsl.contains("fn graph_color("));
        let src = crate::view::shader_source(&preview.wgsl, false, Default::default());
        assert!(src.contains("const VECTOR_PREVIEW: bool = true;"));
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();

        let float_preview = compile_with(&Graph::default_graph(), &BTreeMap::new()).unwrap();
        assert!(!float_preview.wgsl.contains("fn graph_color("));
        assert!(crate::view::shader_source(&float_preview.wgsl, false, Default::default()).contains("const VECTOR_PREVIEW: bool = false;"));
    }

    #[test]
    fn cached_node_becomes_a_texture_read() {
        let g = Graph::default_graph(); // 0 Position → 1 Perlin → 2 OutDensity
        let c = compile_with(&g, &BTreeMap::from([(1, 3)])).unwrap();
        assert!(c.wgsl.contains("cache_sample(3u, p)"));
        assert!(!c.wgsl.contains("fn noise_k0(")); // the cached noise isn't evaluated
        let src = crate::view::shader_source(&c.wgsl, false, Default::default());
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
    }

    #[test]
    fn cycle_through_normal_is_an_error() {
        let mut g = Graph::default();
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let n = g.add(NodeKind::Normal { flip: false }, [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(m, n, 0);
        g.connect(n, m, 0);
        g.connect(m, o, 0);
        assert!(compile(&g).is_err());
    }

    #[test]
    fn every_node_kind_compiles() {
        let mut g = Graph::default();
        let p = g.add(NodeKind::position(), [0.0; 2]);
        let radial = g.add(NodeKind::Radial(Default::default()), [0.0; 2]);
        g.connect(p, radial, 0);
        let v = g.add(NodeKind::Vec3([0.5, 1.0, 2.0]), [0.0; 2]);
        let f = g.add(NodeKind::Float(3.0), [0.0; 2]);
        let mut ops = Vec::new();
        for op in MathOp::ALL {
            let m = g.add(NodeKind::Math { op, a: 1.0, b: 2.0, c: 0.5 }, [0.0; 2]);
            g.connect(p, m, 0);
            g.connect(v, m, 1);
            ops.push(m);
        }
        // Every noise kind, summed: noise_0 + noise_1 + ...
        let mut noise = None;
        for kind in crate::graph::NoiseKind::ALL {
            let n = g.add(NodeKind::Noise(NoiseParams { kind, distortion: 0.3, seed: Some(7), ..Default::default() }), [0.0; 2]);
            g.connect(ops[5], n, 0);
            noise = Some(match noise {
                None => n,
                Some(prev) => {
                    let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
                    g.connect(prev, sum, 0);
                    g.connect(n, sum, 1);
                    sum
                }
            });
        }
        let mut noise = noise.unwrap();
        // The newer Math operators and the other float nodes, each summed in so the shader has
        // to compile and validate all of them (Pattern in every output mode).
        let mut extra: Vec<NodeId> = ops[6..].to_vec();
        for kind in [
            NodeKind::Remap { vals: [0.0, 0.0, 1.0, -2.0, 2.0], clamp: true },
            NodeKind::Grid { counts: [4.0; 3], corners: false, cubic: false, amount: 1.0 },
            NodeKind::Grid { counts: [4.0; 3], corners: true, cubic: false, amount: 1.0 },
        ]
        .into_iter()
        .chain([crate::graph::PatternOutput::Distance, crate::graph::PatternOutput::Shape, crate::graph::PatternOutput::CellValue].map(|output| {
            NodeKind::Pattern(crate::graph::PatternParams { output, seed: Some(3), ..Default::default() })
        }))
        .chain([NodeKind::Pattern(crate::graph::PatternParams { mode2d: false, axis: 0, ..Default::default() })])
        {
            let n = g.add(kind, [0.0; 2]);
            g.connect(ops[5], n, 0);
            extra.push(n);
        }
        for n in extra {
            let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
            g.connect(noise, sum, 0);
            g.connect(n, sum, 1);
            noise = sum;
        }
        // float * noise, then the normal of that, flipped.
        let scaled = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(f, scaled, 0);
        g.connect(noise, scaled, 1);
        let nrm = g.add(NodeKind::Normal { flip: true }, [0.0; 2]);
        g.connect(scaled, nrm, 0);
        // Normal of a normal (vec3 → float conversion) to check nested cones.
        let nrm2 = g.add(NodeKind::Normal { flip: false }, [0.0; 2]);
        g.connect(nrm, nrm2, 0);
        let od = g.add(NodeKind::OutDensity, [0.0; 2]);
        let on = g.add(NodeKind::OutNormal, [0.0; 2]);
        g.connect(scaled, od, 0);
        g.connect(nrm2, on, 0);
        let c = compile(&g).unwrap();
        validate(&c);
    }

    #[test]
    fn empty_graph_compiles() {
        validate(&compile(&Graph::default()).unwrap());
    }

    #[test]
    fn param_edit_keeps_wgsl() {
        let mut g = Graph::default_graph();
        let before = compile(&g).unwrap();
        for n in g.nodes.values_mut() {
            if let NodeKind::Noise(p) = &mut n.kind {
                p.scale = 9.0;
            }
        }
        let after = compile(&g).unwrap();
        assert_eq!(before.wgsl, after.wgsl);
        assert_ne!(before.params, after.params);
    }

    #[test]
    fn cycle_is_an_error() {
        let mut g = Graph::default();
        let a = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let b = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let o = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(a, b, 0);
        g.connect(b, a, 0);
        g.connect(b, o, 0);
        assert!(compile(&g).is_err());
    }
}
