//! Compiles a graph into WGSL: `graph_density(p) -> f32` and `graph_normal(p) -> vec3f`.
//!
//! Every node upstream of an output becomes one SSA `let` in a per-cone function. Node parameters
//! are read from the storage array `P` (filled from `Compiled::params`), so editing a value never
//! changes the WGSL text. Only topology changes do, and the app rebuilds the pipeline when it differs.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, PatternKind};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

pub const NOISE_LIB: &str = include_str!("noise.wgsl");

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
    out: String,
}

pub fn compile(g: &Graph) -> Result<Compiled, String> {
    compile_with(g, &BTreeMap::new())
}

/// Like `compile`, but nodes in `caches` read their baked texture (`cache_sample(slot, p)`)
/// instead of evaluating their upstream graph. Preview only.
pub fn compile_with(g: &Graph, caches: &BTreeMap<NodeId, u32>) -> Result<Compiled, String> {
    // Parameter slots, in node-id order.
    let mut slots = BTreeMap::new();
    let mut params = Vec::new();
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
            NodeKind::Grid { counts, corners } => {
                params.extend(counts);
                params.push(if *corners { 1.0 } else { 0.0 });
            }
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
            ]),
            NodeKind::Noise(n) => params.extend([
                n.scale,
                if n.normalize { 1.0 } else { 0.0 },
                n.octaves_used() as f32,
                n.lacunarity,
                n.roughness,
                n.distortion,
                n.seed.map_or(-1.0, |s| s as f32), // shortcut: f32 holds seeds exactly only below 2^24
                n.feature as u32 as f32,
            ]),
            _ => {}
        }
    }

    let mut cg = Gen { g, caches, slots, types: BTreeMap::new(), done: BTreeSet::new(), building: BTreeSet::new(), kinds: BTreeSet::new(), out: String::new() };
    let density = g.find(|k| *k == NodeKind::OutDensity).and_then(|o| g.input(o, 0));
    let normal = g.find(|k| *k == NodeKind::OutNormal).and_then(|o| g.input(o, 0));

    let density_expr = match density {
        Some(src) => {
            let ty = cg.cone(src)?;
            convert(&format!("c{src}(p)"), ty, Ty::F)
        }
        None => "0.0".into(),
    };
    writeln!(cg.out, "fn graph_density(p: vec3f) -> f32 {{ return {density_expr}; }}").unwrap();

    let normal_expr = match normal {
        Some(src) => {
            let ty = cg.cone(src)?;
            format!("safe_normalize({})", convert(&format!("c{src}(p)"), ty, Ty::V))
        }
        None => "tetra_grad_density(p)".into(),
    };
    writeln!(cg.out, "fn tetra_grad_density(p: vec3f) -> vec3f {{ {} }}", tetra_body(|q| format!("graph_density({q})"))).unwrap();
    writeln!(cg.out, "fn graph_normal(p: vec3f) -> vec3f {{ return {normal_expr}; }}").unwrap();

    if params.is_empty() {
        params.push(0.0); // storage buffers cannot be empty
    }
    let mut wgsl: String = cg.kinds.iter().map(|&k| fbm_function(k)).collect();
    wgsl += &cg.out;
    Ok(Compiled { wgsl, params })
}

/// Body of a function returning the normalized tetrahedral gradient (IQ) of the float field `sample(q)`.
/// A loop, so the field is inlined once rather than four times (driver compile time).
fn tetra_body(sample: impl Fn(&str) -> String) -> String {
    format!(
        "let e = 1e-3; var g = vec3f(0.0); for (var k = 0u; k < 4u; k++) {{ let o = tetra(k); g += o * {}; }} return safe_normalize(g);",
        sample("p + o * e")
    )
}

const FBM_TEMPLATE: &str = include_str!("noise_fbm.wgsl");

/// The fBm wrapper for one noise kind (see noise_fbm.wgsl).
fn fbm_function(kind: NoiseKind) -> String {
    let base = match kind {
        NoiseKind::Perlin => "clamp(perlin(x, period, seed + o), -1.0, 1.0)",
        NoiseKind::Simplex => "clamp(simplex(x, period, seed + o), -1.0, 1.0)",
        NoiseKind::Worley => "worley(x, period, seed + o, feature)",
        NoiseKind::White => "white(x, period, seed + o)",
        NoiseKind::Blue => "blue(x, period, seed + o)",
    };
    FBM_TEMPLATE
        .replace("KIND", &(kind as u32).to_string())
        .replace("HAS_OCTAVES", if kind.has_octaves() { "true" } else { "false" })
        .replace("BASE", base)
}

impl Gen<'_> {
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
            writeln!(body, "    let v{id} = {line};").unwrap();
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
        } else if let NodeKind::Normal { .. } = kind {
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

    /// The input on `port` as an expression of type `want`, or `fallback` when unconnected.
    fn operand(&self, id: NodeId, port: usize, want: Ty, fallback: &str) -> String {
        match self.g.input(id, port) {
            Some(src) => convert(&format!("v{src}"), self.types[&src], want),
            None => fallback.to_string(),
        }
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
            NodeKind::Position => (Ty::V, "p".into()),
            NodeKind::Math { op, .. } => {
                let in_ty = |port| self.g.input(id, port).map_or(Ty::F, |src| self.types[&src]);
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
                    MathOp::Dot => format!("dot({a}, {b})"),
                };
                (ty, e)
            }
            NodeKind::Remap { .. } => {
                let x: Vec<String> = (0..5).map(|p| self.operand(id, p, Ty::F, &format!("P[{}]", s + p))).collect();
                (Ty::F, format!("remap({}, {}, {}, {}, {}, P[{}] > 0.5)", x[0], x[1], x[2], x[3], x[4], s + 5))
            }
            NodeKind::Grid { .. } => {
                let q = self.operand(id, 0, Ty::V, "p");
                (Ty::F, format!("grid_dist({q}, vec3f(P[{s}], P[{}], P[{}]), P[{}] > 0.5)", s + 1, s + 2, s + 3))
            }
            NodeKind::Pattern(p) => {
                let q = self.operand(id, 0, Ty::V, "p");
                match p.kind {
                    PatternKind::Bricks => (Ty::F, format!("bricks({q}, {s}u)")),
                }
            }
            NodeKind::Normal { flip } => {
                let sign = if *flip { "-" } else { "" };
                match self.g.input(id, 0) {
                    Some(src) => {
                        // The input's cone function was written during topo(); add its gradient once.
                        let gname = format!("g{src}");
                        if !self.out.contains(&format!("fn {gname}(")) {
                            let ty = self.types[&src];
                            let body = tetra_body(|q| convert(&format!("c{src}({q})"), ty, Ty::F));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MathOp, NoiseParams};

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
        let src = crate::view::shader_source(&c.wgsl);
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
    }

    #[test]
    fn cached_node_becomes_a_texture_read() {
        let g = Graph::default_graph(); // 0 Position → 1 Perlin → 2 OutDensity
        let c = compile_with(&g, &BTreeMap::from([(1, 3)])).unwrap();
        assert!(c.wgsl.contains("cache_sample(3u, p)"));
        assert!(!c.wgsl.contains("fn noise_k0(")); // the cached noise isn't evaluated
        let src = crate::view::shader_source(&c.wgsl);
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
        let p = g.add(NodeKind::Position, [0.0; 2]);
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
            NodeKind::Grid { counts: [4.0; 3], corners: false },
            NodeKind::Grid { counts: [4.0; 3], corners: true },
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
