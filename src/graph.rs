//! Node graph model, save/load, and tiling analysis.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
}

/// The discriminant is the `kind` constant noise.wgsl switches on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum NoiseKind {
    Perlin = 0,
    Simplex = 1,
    /// Flat random value per cell.
    Voronoi = 2,
    /// Voronoi summed over octaves (Blender 4.0+ fractal Voronoi).
    VoronoiFractal = 3,
    /// Distance to the nearest feature points (see `WorleyFeature`).
    Worley = 4,
    /// Random value per cell.
    White = 5,
    /// Tileable 3D blue noise table lookup.
    Blue = 6,
}

impl NoiseKind {
    pub const ALL: [NoiseKind; 7] = [
        NoiseKind::Perlin,
        NoiseKind::Simplex,
        NoiseKind::Voronoi,
        NoiseKind::VoronoiFractal,
        NoiseKind::Worley,
        NoiseKind::White,
        NoiseKind::Blue,
    ];

    pub fn name(self) -> &'static str {
        match self {
            NoiseKind::Perlin => "Perlin",
            NoiseKind::Simplex => "Simplex",
            NoiseKind::Voronoi => "Voronoi",
            NoiseKind::VoronoiFractal => "Voronoi Fractal",
            NoiseKind::Worley => "Worley",
            NoiseKind::White => "White Noise",
            NoiseKind::Blue => "Blue Noise",
        }
    }

    /// Whether octaves, lacunarity and roughness apply.
    pub fn has_octaves(self) -> bool {
        matches!(self, NoiseKind::Perlin | NoiseKind::Simplex | NoiseKind::VoronoiFractal | NoiseKind::Worley)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum WorleyFeature {
    #[default]
    F1 = 0,
    F2 = 1,
    F2MinusF1 = 2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseParams {
    pub kind: NoiseKind,
    /// Worley only.
    #[serde(default)]
    pub feature: WorleyFeature,
    /// Cells per tile. A whole number keeps the noise tiling.
    pub scale: f32,
    /// Map the signed [-1, 1] result to [0, 1].
    pub normalize: bool,
    pub octaves: u32,
    /// Frequency step between octaves. A whole number keeps every octave tiling.
    pub lacunarity: f32,
    /// Amplitude step between octaves (Blender's "Roughness").
    pub roughness: f32,
    /// Blender-style: the input is offset by a noise vector times this before the fBm loop.
    pub distortion: f32,
    /// Overrides the global seed when set.
    pub seed: Option<u32>,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            kind: NoiseKind::Perlin,
            feature: WorleyFeature::F1,
            scale: 4.0,
            normalize: false,
            octaves: 4,
            lacunarity: 2.0,
            roughness: 0.5,
            distortion: 0.0,
            seed: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Float(f32),
    Vec3([f32; 3]),
    /// Inputs: a, b. The stored values are used when an input is unconnected.
    Math { op: MathOp, a: f32, b: f32 },
    Position,
    /// Input: the field to differentiate. Output: unit gradient (points low → high unless flipped).
    Normal { flip: bool },
    /// Input: position (defaults to world position).
    Noise(NoiseParams),
    OutDensity,
    OutNormal,
}

impl NodeKind {
    pub fn input_count(&self) -> usize {
        match self {
            NodeKind::Float(_) | NodeKind::Vec3(_) | NodeKind::Position => 0,
            NodeKind::Math { .. } => 2,
            NodeKind::Normal { .. } | NodeKind::Noise(_) | NodeKind::OutDensity | NodeKind::OutNormal => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Position in the graph editor.
    pub pos: [f32; 2],
}

/// Every node has a single output, so an edge is "output of `from` → input `port` of `to`".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub port: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn add(&mut self, kind: NodeKind, pos: [f32; 2]) -> NodeId {
        let id = self.nodes.keys().next_back().map_or(0, |k| k + 1);
        self.nodes.insert(id, Node { kind, pos });
        id
    }

    /// Connects `from` into input `port` of `to`, replacing any existing link on that input.
    pub fn connect(&mut self, from: NodeId, to: NodeId, port: usize) {
        self.edges.retain(|e| !(e.to == to && e.port == port));
        self.edges.push(Edge { from, to, port });
    }

    pub fn remove(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    pub fn disconnect(&mut self, to: NodeId, port: usize) {
        self.edges.retain(|e| !(e.to == to && e.port == port));
    }

    /// True if `target` is `id` or anywhere upstream of it. Connecting `a → b` makes a cycle iff `a` depends on `b`.
    pub fn depends_on(&self, id: NodeId, target: NodeId) -> bool {
        let mut stack = vec![id];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(n) = stack.pop() {
            if n == target {
                return true;
            }
            if seen.insert(n) {
                stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
            }
        }
        false
    }

    pub fn input(&self, to: NodeId, port: usize) -> Option<NodeId> {
        self.edges.iter().find(|e| e.to == to && e.port == port).map(|e| e.from)
    }

    pub fn find(&self, pred: impl Fn(&NodeKind) -> bool) -> Option<NodeId> {
        self.nodes.iter().find(|(_, n)| pred(&n.kind)).map(|(id, _)| *id)
    }

    #[allow(dead_code)] // save/load UI arrives with the node editor (milestone 2)
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("graph serializes")
    }

    #[allow(dead_code)]
    pub fn from_json(s: &str) -> Result<Self, String> {
        let g: Graph = serde_json::from_str(s).map_err(|e| e.to_string())?;
        for e in &g.edges {
            let to = g.nodes.get(&e.to).ok_or(format!("edge to missing node {}", e.to))?;
            g.nodes.get(&e.from).ok_or(format!("edge from missing node {}", e.from))?;
            if e.port >= to.kind.input_count() {
                return Err(format!("edge into node {} uses port {} which does not exist", e.to, e.port));
            }
        }
        Ok(g)
    }

    /// A copy whose Density output reads `node` instead, for previewing an intermediate node.
    pub fn previewing(&self, node: NodeId) -> Graph {
        let mut g = self.clone();
        let out = match g.find(|k| *k == NodeKind::OutDensity) {
            Some(o) => o,
            None => g.add(NodeKind::OutDensity, [0.0, 0.0]),
        };
        if node != out && g.nodes.contains_key(&node) {
            g.connect(node, out, 0);
        }
        g
    }

    /// Position → Perlin → Density, plus an unconnected Normal output.
    pub fn default_graph() -> Self {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::Position, [0.0, 0.0]);
        let noise = g.add(NodeKind::Noise(NoiseParams::default()), [200.0, 0.0]);
        let out = g.add(NodeKind::OutDensity, [400.0, 0.0]);
        g.add(NodeKind::OutNormal, [400.0, 120.0]);
        g.connect(pos, noise, 0);
        g.connect(noise, out, 0);
        g
    }
}

// ---- tiling analysis ---------------------------------------------------

/// How a node's output behaves under p → p + 1 (one tile).
#[derive(Clone, Debug, PartialEq)]
pub enum Tiling {
    /// Same everywhere.
    Const,
    /// Repeats every tile.
    Periodic,
    /// Position-like (p, plus or times whole numbers, plus periodic offsets). Repeats once fed into a noise.
    Coord,
    /// Breaks tiling. The string says why.
    Broken(String),
}

fn is_whole(x: f32) -> bool {
    x.fract() == 0.0
}

impl Graph {
    /// Tiling class of every node. Nodes in a cycle are reported as Broken.
    pub fn tiling(&self) -> BTreeMap<NodeId, Tiling> {
        let mut memo = BTreeMap::new();
        for &id in self.nodes.keys() {
            self.tiling_of(id, &mut memo, &mut Vec::new());
        }
        memo
    }

    fn tiling_of(&self, id: NodeId, memo: &mut BTreeMap<NodeId, Tiling>, stack: &mut Vec<NodeId>) -> Tiling {
        if let Some(t) = memo.get(&id) {
            return t.clone();
        }
        if stack.contains(&id) {
            return Tiling::Broken("graph has a cycle".into());
        }
        stack.push(id);
        let input = |port: usize, memo: &mut BTreeMap<NodeId, Tiling>, stack: &mut Vec<NodeId>| {
            self.input(id, port).map(|src| self.tiling_of(src, memo, stack))
        };
        use Tiling::*;
        let t = match &self.nodes[&id].kind {
            NodeKind::Float(_) | NodeKind::Vec3(_) => Const,
            NodeKind::Position => Coord,
            NodeKind::Math { op, a, b } => {
                let ta = input(0, memo, stack).unwrap_or(Const);
                let tb = input(1, memo, stack).unwrap_or(Const);
                // Operand is a whole-number constant: the unconnected default, or a Float/Vec3 node.
                let whole_const = |port: usize, default: f32| match self.input(id, port).map(|s| &self.nodes[&s].kind) {
                    None => is_whole(default),
                    Some(NodeKind::Float(v)) => is_whole(*v),
                    Some(NodeKind::Vec3(v)) => v.iter().all(|x| is_whole(*x)),
                    Some(_) => false,
                };
                match (&ta, &tb) {
                    (Broken(r), _) | (_, Broken(r)) => Broken(r.clone()),
                    (Const, Const) => Const,
                    (Const | Periodic, Const | Periodic) => Periodic,
                    (Coord, _) | (_, Coord) => match op {
                        MathOp::Add => Coord,
                        MathOp::Subtract if tb != Coord => Coord,
                        // p - p cancels the position out.
                        MathOp::Subtract => Const,
                        MathOp::Multiply if ta == Coord && tb == Const && whole_const(1, *b) => Coord,
                        MathOp::Multiply if tb == Coord && ta == Const && whole_const(0, *a) => Coord,
                        MathOp::Multiply => Broken("position multiplied by a non-whole or non-constant value".into()),
                        _ => Broken(format!("{op:?} on a position does not tile")),
                    },
                }
            }
            NodeKind::Noise(n) => {
                if !is_whole(n.scale) {
                    Broken(format!("scale {} is not a whole number", n.scale))
                } else if n.kind == NoiseKind::Blue && n.scale % 4.0 != 0.0 {
                    // The blue noise tables are 4³, 8³, 16³ and 32³; the tile must hold whole tables.
                    Broken(format!("blue noise scale {} is not a multiple of 4", n.scale))
                } else if n.kind.has_octaves() && n.octaves > 1 && !is_whole(n.lacunarity) {
                    Broken(format!("lacunarity {} is not a whole number", n.lacunarity))
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Normal { .. } => match input(0, memo, stack).unwrap_or(Const) {
                Coord => Const, // gradient of a linear position map is constant
                t => t,
            },
            NodeKind::OutDensity | NodeKind::OutNormal => match input(0, memo, stack).unwrap_or(Const) {
                Coord => Broken("output depends directly on position".into()),
                t => t,
            },
        };
        stack.pop();
        memo.insert(id, t.clone());
        t
    }
}

// ---- value range -------------------------------------------------------

/// Closed interval [lo, hi]; infinite ends mean unbounded. Vectors use one interval for all components.
pub type Range = (f32, f32);

const UNBOUNDED: Range = (f32::NEG_INFINITY, f32::INFINITY);

impl Graph {
    /// A guaranteed range for the Density output, so shaders can skip the graph where the
    /// displaced surface cannot be. Unbounded when position flows through, or on cycles.
    pub fn density_range(&self) -> Range {
        match self.find(|k| *k == NodeKind::OutDensity).and_then(|o| self.input(o, 0)) {
            Some(src) => self.range_of(src, &mut Vec::new()),
            None => (0.0, 0.0),
        }
    }

    fn range_of(&self, id: NodeId, stack: &mut Vec<NodeId>) -> Range {
        if stack.contains(&id) {
            return UNBOUNDED;
        }
        stack.push(id);
        let r = match &self.nodes[&id].kind {
            NodeKind::Float(v) => (*v, *v),
            NodeKind::Vec3(v) => (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max)),
            NodeKind::Position => UNBOUNDED,
            NodeKind::Normal { .. } => (-1.0, 1.0),
            // noise.wgsl keeps every kind within [-1, 1] (Perlin and Simplex are clamped).
            NodeKind::Noise(n) => if n.normalize { (0.0, 1.0) } else { (-1.0, 1.0) },
            NodeKind::Math { op, a, b } => {
                let mut operand = |port: usize, default: f32| match self.input(id, port) {
                    Some(src) => self.range_of(src, stack),
                    None => (default, default),
                };
                let (ra, rb) = (operand(0, *a), operand(1, *b));
                math_range(*op, ra, rb)
            }
            NodeKind::OutDensity | NodeKind::OutNormal => UNBOUNDED,
        };
        stack.pop();
        r
    }
}

fn math_range(op: MathOp, (a0, a1): Range, (b0, b1): Range) -> Range {
    let span = |v: [f32; 4]| {
        // NaN (0 × ∞) only comes from unbounded inputs, so treat it as unbounded.
        if v.iter().any(|x| x.is_nan()) {
            return UNBOUNDED;
        }
        (v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    };
    match op {
        MathOp::Add => (a0 + b0, a1 + b1),
        MathOp::Subtract => (a0 - b1, a1 - b0),
        MathOp::Multiply => span([a0 * b0, a0 * b1, a1 * b0, a1 * b1]),
        // Safe divide returns 0 for b = 0, but b near 0 is unbounded.
        MathOp::Divide if b0 <= 0.0 && b1 >= 0.0 => UNBOUNDED,
        MathOp::Divide => span([a0 / b0, a0 / b1, a1 / b0, a1 / b1]),
        MathOp::Min => (a0.min(b0), a1.min(b1)),
        MathOp::Max => (a0.max(b0), a1.max(b1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_range_follows_math() {
        let mut g = Graph::default_graph();
        assert_eq!(g.density_range(), (-1.0, 1.0));
        // noise * 3 + 0.5
        let noise = g.find(|k| matches!(k, NodeKind::Noise(_))).unwrap();
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        let mul = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 3.0 }, [0.0; 2]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.5 }, [0.0; 2]);
        g.connect(noise, mul, 0);
        g.connect(mul, add, 0);
        g.connect(add, out, 0);
        assert_eq!(g.density_range(), (-2.5, 3.5));
        // Position reaching the output makes it unbounded.
        let pos = g.find(|k| *k == NodeKind::Position).unwrap();
        g.connect(pos, add, 1);
        assert_eq!(g.density_range(), UNBOUNDED);
        // Dividing by a range that spans zero is unbounded too.
        assert_eq!(math_range(MathOp::Divide, (1.0, 1.0), (-1.0, 1.0)), UNBOUNDED);
        assert_eq!(math_range(MathOp::Divide, (1.0, 2.0), (2.0, 4.0)), (0.25, 1.0));
    }

    fn tiling_at(g: &Graph, id: NodeId) -> Tiling {
        g.tiling()[&id].clone()
    }

    #[test]
    fn default_graph_tiles() {
        let g = Graph::default_graph();
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert_eq!(tiling_at(&g, out), Tiling::Periodic);
    }

    #[test]
    fn fractional_scale_breaks_tiling_downstream() {
        let mut g = Graph::default_graph();
        let noise = g.find(|k| matches!(k, NodeKind::Noise(_))).unwrap();
        if let NodeKind::Noise(n) = &mut g.nodes.get_mut(&noise).unwrap().kind {
            n.scale = 2.5;
        }
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert!(matches!(tiling_at(&g, out), Tiling::Broken(r) if r.contains("scale 2.5")));
    }

    #[test]
    fn blue_noise_scale_rule_and_octaveless_lacunarity() {
        let mut g = Graph::default();
        let mk = |kind, scale, lacunarity| NodeKind::Noise(NoiseParams { kind, scale, lacunarity, ..Default::default() });
        let blue6 = g.add(mk(NoiseKind::Blue, 6.0, 2.0), [0.0; 2]);
        let blue8 = g.add(mk(NoiseKind::Blue, 8.0, 2.0), [0.0; 2]);
        // Lacunarity is ignored by kinds without octaves.
        let white = g.add(mk(NoiseKind::White, 8.0, 2.5), [0.0; 2]);
        let perlin = g.add(mk(NoiseKind::Perlin, 8.0, 2.5), [0.0; 2]);
        assert!(matches!(tiling_at(&g, blue6), Tiling::Broken(_)));
        assert_eq!(tiling_at(&g, blue8), Tiling::Periodic);
        assert_eq!(tiling_at(&g, white), Tiling::Periodic);
        assert!(matches!(tiling_at(&g, perlin), Tiling::Broken(_)));
    }

    #[test]
    fn position_math_rules() {
        let mut g = Graph::default();
        let p = g.add(NodeKind::Position, [0.0; 2]);
        let times2 = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 2.0 }, [0.0; 2]);
        let times_half = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.5 }, [0.0; 2]);
        let n1 = g.add(NodeKind::Noise(NoiseParams::default()), [0.0; 2]);
        // Domain warp: noise(p + noise(p)) still tiles.
        let warp = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }, [0.0; 2]);
        let n2 = g.add(NodeKind::Noise(NoiseParams::default()), [0.0; 2]);
        g.connect(p, times2, 0);
        g.connect(p, times_half, 0);
        g.connect(p, n1, 0);
        g.connect(p, warp, 0);
        g.connect(n1, warp, 1);
        g.connect(warp, n2, 0);
        assert_eq!(tiling_at(&g, times2), Tiling::Coord);
        assert!(matches!(tiling_at(&g, times_half), Tiling::Broken(_)));
        assert_eq!(tiling_at(&g, warp), Tiling::Coord);
        assert_eq!(tiling_at(&g, n2), Tiling::Periodic);

        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(p, out, 0);
        assert!(matches!(tiling_at(&g, out), Tiling::Broken(_)));
    }

    #[test]
    fn cycle_is_reported() {
        let mut g = Graph::default();
        let a = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }, [0.0; 2]);
        let b = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }, [0.0; 2]);
        g.connect(a, b, 0);
        g.connect(b, a, 0);
        assert!(matches!(tiling_at(&g, a), Tiling::Broken(_)));
    }

    #[test]
    fn previewing_rewires_density_only_in_the_copy() {
        let g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let p = g.previewing(0);
        assert_eq!(p.input(2, 0), Some(0));
        assert_eq!(g.input(2, 0), Some(1));
        // Without a Density output one is added.
        let mut bare = Graph::default();
        let pos = bare.add(NodeKind::Position, [0.0; 2]);
        let pv = bare.previewing(pos);
        let out = pv.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert_eq!(pv.input(out, 0), Some(pos));
    }

    #[test]
    fn depends_on_and_remove() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity, 3 OutNormal
        assert!(g.depends_on(2, 0));
        assert!(g.depends_on(1, 1));
        assert!(!g.depends_on(0, 1));
        assert!(!g.depends_on(3, 0));
        g.remove(1);
        assert!(g.edges.is_empty());
        assert!(!g.depends_on(2, 0));
    }

    #[test]
    fn json_round_trip_and_validation() {
        let g = Graph::default_graph();
        assert_eq!(Graph::from_json(&g.to_json()).unwrap(), g);
        let mut bad = g.clone();
        bad.edges.push(Edge { from: 0, to: 2, port: 5 });
        assert!(Graph::from_json(&bad.to_json()).is_err());
    }
}
