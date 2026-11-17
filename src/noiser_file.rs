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
    let out = |k: NodeKind| g.find(|n| *n == k).and_then(|o| g.input(o, 0));
    let (density, normal) = (out(NodeKind::OutDensity), out(NodeKind::OutNormal));
    let mut order = Vec::new();
    let (mut seen, mut visiting) = (BTreeSet::new(), BTreeSet::new());
    for root in density.into_iter().chain(normal) {
        topo(g, root, &mut seen, &mut visiting, &mut order)?;
    }
    let input = |id: NodeId, port: usize| g.input(id, port).map_or(Value::Null, |s| json!(s));
    let ops: Vec<Value> = order
        .iter()
        .map(|&id| {
            let ty = if g.yields_vector(id) { "vector" } else { "float" };
            let mut op = match &g.nodes[&id].kind {
                NodeKind::Position => json!({"op": "position"}),
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
                    };
                    // An unconnected operand is its constant.
                    let operand = |port, c: f32| g.input(id, port).map_or(json!(c), |s| json!({"node": s}));
                    let mut o = json!({"op": "math", "fn": f, "a": operand(0, *a), "b": operand(1, *b)});
                    if op.is_smooth() {
                        o["k"] = operand(2, *c);
                    }
                    o
                }
                NodeKind::Remap { vals, clamp } => {
                    let operand = |port: usize| g.input(id, port).map_or(json!(vals[port]), |s| json!({"node": s}));
                    json!({"op": "remap", "value": operand(0), "from_min": operand(1), "from_max": operand(2),
                           "to_min": operand(3), "to_max": operand(4), "clamp": clamp})
                }
                NodeKind::Grid { counts, corners } => {
                    json!({"op": "grid", "position": input(id, 0), "counts": counts, "reference": if *corners { "corner" } else { "center" }})
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
                           "width": p.width, "softness": p.softness, "seed": p.seed.unwrap_or(seed)})
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
                        });
                    }
                    o
                }
                NodeKind::OutDensity | NodeKind::OutNormal => unreachable!("outputs are not upstream of outputs"),
            };
            op["id"] = json!(id);
            op["type"] = json!(ty);
            op
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
        if let Some(src) = g.input(id, port) {
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
}
