//! The `.noiser` math-function export: the graph as a flat, ordered op list plus test vectors.
//! The format is specified in docs/noiser-format.md; samplers (blender/noiser) implement it.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, WorleyFeature};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub const VERSION: u32 = 1;
/// The first TEST_RES³ test points are one tile's grid centres.
pub const TEST_RES: u32 = 6;

fn kind_name(k: NoiseKind) -> &'static str {
    match k {
        NoiseKind::Perlin => "perlin",
        NoiseKind::Simplex => "simplex",
        NoiseKind::Worley => "worley",
        NoiseKind::White => "white",
        NoiseKind::Blue => "blue",
    }
}

/// The op list for everything upstream of the outputs, in evaluation order (inputs first).
/// `seed` is the global seed; each noise op carries its resolved seed.
pub fn ops(g: &Graph, seed: u32) -> Result<Value, String> {
    let lowered = g.lowered();
    let g = &lowered;
    // A reference to what feeds an input: the op id, or for one output of a Separate node the id of the
    // single-component `separate` op written for it (1000000 + 3 * node + port).
    let reference = |to: NodeId, port: usize| g.input_from(to, port).map(|(src, fp)| if matches!(g.nodes[&src].kind, NodeKind::Separate) { 1_000_000 + src as u64 * 3 + fp.min(2) as u64 } else { src as u64 });
    let out = |k: NodeKind| g.find(|n| *n == k).and_then(|o| reference(o, 0));
    let (density, normal) = (out(NodeKind::OutDensity), out(NodeKind::OutNormal));
    let mut order = Vec::new();
    let (mut seen, mut visiting) = (BTreeSet::new(), BTreeSet::new());
    for o in [NodeKind::OutDensity, NodeKind::OutNormal] {
        if let Some((root, _)) = g.find(|n| *n == o).and_then(|o| g.input_from(o, 0)) {
            topo(g, root, &mut seen, &mut visiting, &mut order)?;
        }
    }
    if order.iter().any(|id| matches!(g.nodes[id].kind, NodeKind::Radial(_))) {
        return Err("a Radial Position node has no .noiser form".into());
    }
    if order.iter().any(|id| matches!(g.nodes[id].kind, NodeKind::Mask { .. } | NodeKind::Slope { .. })) {
        return Err("a Mask or Slope node (Landscape tab) has no .noiser form".into());
    }
    if order.iter().any(|id| matches!(g.nodes[id].kind, NodeKind::Random { .. })) {
        return Err("a Random node has no .noiser form".into());
    }
    if order.iter().any(|id| matches!(g.nodes[id].kind, NodeKind::Position { scale } if scale != [1.0; 3]) || (matches!(g.nodes[id].kind, NodeKind::Position { .. }) && (0..3).any(|p| g.input(*id, p).is_some()))) {
        return Err("a scaled Position node has no .noiser form".into());
    }
    if order.iter().any(|id| g.pin_wired(*id, None)) {
        return Err("a wired parameter pin (frequency, count, ...) has no .noiser form".into());
    }
    let input = |id: NodeId, port: usize| reference(id, port).map_or(Value::Null, |s| json!(s));
    let ops: Vec<Value> = order
        .iter()
        .map(|&id| {
            if matches!(g.nodes[&id].kind, NodeKind::Separate) {
                // One single-component op per output in use.
                let mut used: Vec<usize> = g.edges.iter().filter(|e| e.from == id).map(|e| e.from_port.min(2)).collect();
                used.sort();
                used.dedup();
                return Value::Array(
                    used.into_iter()
                        .map(|fp| {
                            let name = ["x", "y", "z"][fp];
                            json!({"op": "separate", "vector": input(id, 0), "component": name, "id": 1_000_000 + id as u64 * 3 + fp as u64, "type": "float"})
                        })
                        .collect(),
                );
            }
            let ty = if g.yields_vector(id) { "vector" } else { "float" };
            let mut op = match &g.nodes[&id].kind {
                NodeKind::Position { .. } => json!({"op": "position"}), // a scaled one is refused above
                NodeKind::Radial(_) | NodeKind::Random { .. } | NodeKind::Mask { .. } | NodeKind::Slope { .. } => unreachable!("refused above"),
                NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::Scatter(_) | NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv => json!({"op": "value", "value": 0.0}), // mesh graph nodes have no .noiser form
                // A passthrough, written as adding zero.
                NodeKind::NamedReroute { .. } | NodeKind::NamedRerouteUse { .. } => unreachable!("readers look through named reroutes"),
                NodeKind::Reroute => json!({"op": "math", "fn": "add", "a": input(id, 0), "b": 0.0}),
                NodeKind::Float(v) => json!({"op": "value", "value": v}),
                NodeKind::Vec3(v) => json!({"op": "value", "value": v}),
                NodeKind::Math { op, a, b, c } => {
                    let f = match op {
                        MathOp::Add => "add",
                        MathOp::Subtract => "subtract",
                        MathOp::Multiply => "multiply",
                        MathOp::Divide => "divide",
                        MathOp::Min => "min",
                        MathOp::Max => "max",
                        MathOp::Power => "power",
                        MathOp::SmoothMin => "smooth_min",
                        MathOp::SmoothMax => "smooth_max",
                        MathOp::Dot => "dot",
                        MathOp::Absolute => "absolute",
                        MathOp::Ridge => "ridge",
                        MathOp::Floor => "floor",
                        MathOp::Fract => "fract",
                        MathOp::Sine => "sine",
                        MathOp::Sqrt => "sqrt",
                        MathOp::Modulo => "modulo",
                        MathOp::Greater => "greater",
                        MathOp::Lerp => "lerp",
                        MathOp::InverseLerp => "inverse_lerp",
                    };
                    // An unconnected operand is its constant.
                    let operand = |port, c: f32| reference(id, port).map_or(json!(c), |s| json!({"node": s}));
                    let mut o = json!({"op": "math", "fn": f, "a": operand(0, *a), "b": operand(1, *b)});
                    if op.is_ternary() {
                        o["k"] = operand(2, *c);
                    }
                    o
                }
                NodeKind::Remap { vals, clamp } => {
                    let operand = |port: usize| reference(id, port).map_or(json!(vals[port]), |s| json!({"node": s}));
                    json!({"op": "remap", "value": operand(0), "from_min": operand(1), "from_max": operand(2),
                           "to_min": operand(3), "to_max": operand(4), "clamp": clamp})
                }
                NodeKind::Separate => unreachable!("written above"),
                NodeKind::Combine { vals } => {
                    let operand = |port: usize| reference(id, port).map_or(json!(vals[port]), |s| json!({"node": s}));
                    json!({"op": "combine", "x": operand(0), "y": operand(1), "z": operand(2)})
                }
                NodeKind::Mix { vals, clamp } => {
                    let operand = |port: usize| reference(id, port).map_or(json!(vals[port]), |s| json!({"node": s}));
                    json!({"op": "mix", "factor": operand(0), "a": operand(1), "b": operand(2), "clamp": clamp})
                }
                NodeKind::Curve { points, mode } => {
                    let sorted = crate::graph::curve_points(points);
                    json!({"op": "curve", "value": reference(id, 0).map_or(json!(0.0), |s| json!({"node": s})), "points": sorted, "mode": mode.label().to_lowercase()})
                }
                NodeKind::Grid { counts, corners, cubic, .. } => {
                    let mut o = json!({"op": "grid", "position": input(id, 0), "counts": counts, "reference": if *corners { "corner" } else { "center" }});
                    // Only when set, so older samplers keep working on older graphs.
                    if *cubic {
                        o["shape"] = json!("cubic");
                    }
                    o
                }
                NodeKind::Pattern(p) => {
                    let output = match p.output {
                        crate::graph::PatternOutput::Distance => "distance",
                        crate::graph::PatternOutput::Shape => "shape",
                        crate::graph::PatternOutput::CellValue => "cell_value",
                    };
                    json!({"op": "pattern", "kind": p.kind.file_name(), "position": input(id, 0), "counts": p.counts, "mode2d": p.mode2d,
                           "axis": p.axis, "steps": p.steps, "gap": p.gap, "bevel": p.bevel, "output": output,
                           "variant": p.variant, "radius": p.radius, "profile": p.profile, "dir": p.dir,
                           "width": p.width, "softness": p.softness, "iterations": p.iterations, "seed": p.seed.unwrap_or(seed)})
                }
                NodeKind::Normal { flip } => json!({"op": "normal", "field": input(id, 0), "flip": flip}),
                NodeKind::Noise(n) => {
                    let mut o = json!({
                        "op": "noise", "kind": kind_name(n.kind), "position": input(id, 0),
                        "scale": n.scale, "normalize": n.normalize, "octaves": n.octaves_used(), "lacunarity": n.lacunarity,
                        "roughness": n.roughness, "distortion": n.distortion, "seed": n.seed.unwrap_or(seed),
                    });
                    if n.kind == NoiseKind::Worley {
                        o["feature"] = json!(match n.feature {
                            WorleyFeature::F1 => "f1",
                            WorleyFeature::F2 => "f2",
                            WorleyFeature::F2MinusF1 => "f2_minus_f1",
                            WorleyFeature::CellValue => "cell_value",
                            WorleyFeature::Edges => "edges",
                        });
                        // Only when they differ from the defaults, so older samplers keep working on older graphs.
                        if n.metric != crate::graph::WorleyMetric::Euclidean {
                            o["metric"] = json!(if n.metric == crate::graph::WorleyMetric::Manhattan { "manhattan" } else { "chebyshev" });
                        }
                        if n.jitter != 1.0 {
                            o["jitter"] = json!(n.jitter);
                        }
                    }
                    o
                }
                NodeKind::OutDensity | NodeKind::OutNormal => unreachable!("outputs are not upstream of outputs"),
            };
            op["id"] = json!(id);
            op["type"] = json!(ty);
            op
        })
        .flat_map(|v| match v {
            Value::Array(ops) => ops,
            op => vec![op],
        })
        .collect();
    Ok(json!({"format": "noiser", "version": VERSION, "ops": ops, "outputs": {"density": density, "normal": normal}}))
}

fn topo(g: &Graph, id: NodeId, seen: &mut BTreeSet<NodeId>, visiting: &mut BTreeSet<NodeId>, order: &mut Vec<NodeId>) -> Result<(), String> {
    if seen.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err("graph has a cycle".into());
    }
    for port in 0..g.nodes[&id].kind.input_count() {
        if let Some((src, _)) = g.input_from(id, port) {
            topo(g, src, seen, visiting, order)?;
        }
    }
    visiting.remove(&id);
    seen.insert(id);
    order.push(id);
    Ok(())
}

/// Test points: one tile's grid centres, then points outside the unit tile and on its seams and
/// lattice lines, where wrapping and sign bugs live. Samplers run in object space, so coordinates
/// below 0 and above 1 matter as much as the tile itself.
pub fn test_points() -> Vec<[f32; 3]> {
    let n = TEST_RES;
    let mut pts: Vec<[f32; 3]> = (0..n * n * n).map(|i| [i % n, i / n % n, i / (n * n)].map(|c| (c as f32 + 0.5) / n as f32)).collect();
    // Every combination of seam and lattice coordinates, including exact zero and one.
    let edge = [-1.0, 0.0, 0.5, 1.0];
    for &z in &edge {
        for &y in &edge {
            for &x in &edge {
                pts.push([x, y, z]);
            }
        }
    }
    // Scattered points over [-2, 3]³ from a fixed generator.
    let mut s = 0x2545F491u32;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / (1u32 << 24) as f32 * 5.0 - 2.0
    };
    pts.extend((0..64).map(|_| [next(), next(), next()]));
    pts
}

/// The full file: ops plus test vectors (`density` at `test_points()`, from the GPU evaluator).
pub fn file(g: &Graph, seed: u32, eval_points: impl FnOnce(&[[f32; 3]]) -> Vec<f32>) -> Result<String, String> {
    let mut v = ops(g, seed)?;
    let points = test_points();
    v["tests"] = json!({"points": points, "density": eval_points(&points)});
    serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_are_ordered_and_seeds_resolved() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let m = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 2.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, m, 0);
        g.connect(m, 2, 0);
        let v = ops(&g, 7).unwrap();
        let ids: Vec<_> = v["ops"].as_array().unwrap().iter().map(|o| o["id"].as_u64().unwrap()).collect();
        assert_eq!(ids, [0, 1, m as u64]);
        assert_eq!(v["ops"][1]["seed"], 7);
        assert_eq!(v["ops"][1]["position"], 0);
        assert_eq!(v["ops"][2]["a"], json!({"node": 1}));
        assert_eq!(v["ops"][2]["b"], 2.0);
        assert_eq!(v["ops"][0]["type"], "vector");
        assert_eq!(v["outputs"]["density"], m);
        assert!(v["outputs"]["normal"].is_null());
    }

    /// Every op the writer emits for random graphs is a documented op, with unique ids and inputs defined earlier.
    #[test]
    fn random_graphs_export_documented_ops_in_order() {
        let doc = include_str!("../docs/noiser-format.md");
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        for _ in 0..300 {
            let g = crate::codegen::random_graph(&mut rnd);
            let v = ops(&g, 1).unwrap();
            let mut defined = BTreeSet::new();
            for o in v["ops"].as_array().unwrap() {
                let name = o["op"].as_str().unwrap();
                assert!(doc.contains(&format!("### `{name}`")), "op {name} is not documented");
                if name == "math" {
                    let f = o["fn"].as_str().unwrap();
                    assert!(doc.contains(&format!("`{f}`")), "math fn {f} is not documented");
                }
                for key in ["a", "b", "k", "position", "value", "factor", "field", "from_min", "from_max", "to_min", "to_max"] {
                    // Operands are {"node": id}; position and field are plain ids.
                    let id = o.get(key).and_then(|x| x.get("node").and_then(|n| n.as_u64()).or_else(|| if key == "position" || key == "field" { x.as_u64() } else { None }));
                    if let Some(id) = id {
                        assert!(defined.contains(&id), "{name}.{key} reads node {id} before it is defined");
                    }
                }
                assert!(defined.insert(o["id"].as_u64().unwrap()), "duplicate id");
            }
        }
    }
}
