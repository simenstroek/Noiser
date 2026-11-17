//! Node graph model, save/load, and tiling analysis.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type NodeId = u32;

pub const MAX_CURVE_POINTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveMode {
    Linear,
    /// Smoothstep between neighbouring points (flat at every point).
    Smooth,
    /// Each y holds until the next point.
    Step,
}

impl CurveMode {
    pub const ALL: [CurveMode; 3] = [CurveMode::Linear, CurveMode::Smooth, CurveMode::Step];
    pub fn label(self) -> &'static str {
        match self {
            CurveMode::Linear => "Linear",
            CurveMode::Smooth => "Smooth",
            CurveMode::Step => "Step",
        }
    }
}

/// A curve points sorted by x and limited to `MAX_CURVE_POINTS`; never empty.
pub fn curve_points(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut p: Vec<[f32; 2]> = points.iter().copied().filter(|q| q[0].is_finite() && q[1].is_finite()).collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.truncate(MAX_CURVE_POINTS);
    if p.is_empty() {
        p.push([0.0, 0.0]);
    }
    p
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
    /// Safe power: a negative base only takes whole exponents, and 0^x = 0.
    Power,
    /// Polynomial smooth minimum (Blender's), blending over the third input's width.
    SmoothMin,
    SmoothMax,
    /// Dot product of two vectors, as a float.
    Dot,
    /// |a| (unary): billow when applied to noise.
    Absolute,
    /// 1 - |a| (unary): ridged noise when applied to noise.
    Ridge,
}

impl MathOp {
    pub const ALL: [MathOp; 12] = [
        MathOp::Add,
        MathOp::Subtract,
        MathOp::Multiply,
        MathOp::Divide,
        MathOp::Min,
        MathOp::Max,
        MathOp::Power,
        MathOp::SmoothMin,
        MathOp::SmoothMax,
        MathOp::Dot,
        MathOp::Absolute,
        MathOp::Ridge,
    ];

    /// Name shown in the operator dropdown.
    pub fn label(self) -> &'static str {
        match self {
            MathOp::Add => "Add",
            MathOp::Subtract => "Subtract",
            MathOp::Multiply => "Multiply",
            MathOp::Divide => "Divide",
            MathOp::Min => "Minimum",
            MathOp::Max => "Maximum",
            MathOp::Power => "Power",
            MathOp::SmoothMin => "Smooth minimum",
            MathOp::SmoothMax => "Smooth maximum",
            MathOp::Dot => "Dot product",
            MathOp::Absolute => "Absolute",
            MathOp::Ridge => "Ridge (1 − |a|)",
        }
    }

    /// Smooth min and max take a third input, the blend width.
    pub fn is_smooth(self) -> bool {
        matches!(self, MathOp::SmoothMin | MathOp::SmoothMax)
    }

    /// Absolute and Ridge use only the first input.
    pub fn is_unary(self) -> bool {
        matches!(self, MathOp::Absolute | MathOp::Ridge)
    }
}

fn default_smooth() -> f32 {
    0.5
}

/// The discriminant names the generated fBm function (`noise_k{n}`, see codegen.rs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NoiseKind {
    Perlin = 0,
    Simplex = 1,
    /// Cellular noise: distance to the nearest feature points or a flat value per cell (see
    /// `WorleyFeature`), optionally fractal. Replaced the separate Voronoi and Voronoi Fractal nodes.
    /// (Discriminants are names, not indices: 2 and 3 were those two nodes.)
    Worley = 4,
    /// Random value per cell.
    White = 5,
    /// Tileable 3D blue noise table lookup.
    Blue = 6,
}

impl NoiseKind {
    pub const ALL: [NoiseKind; 5] = [
        NoiseKind::Perlin,
        NoiseKind::Simplex,
        NoiseKind::Worley,
        NoiseKind::White,
        NoiseKind::Blue,
    ];

    pub fn name(self) -> &'static str {
        match self {
            NoiseKind::Perlin => "Perlin",
            NoiseKind::Simplex => "Simplex",
            NoiseKind::Worley => "Worley",
            NoiseKind::White => "White Noise",
            NoiseKind::Blue => "Blue Noise",
        }
    }

    /// Whether octaves, lacunarity and roughness can apply (Worley only when fractal; see
    /// `NoiseParams::uses_octaves`).
    pub fn has_octaves(self) -> bool {
        matches!(self, NoiseKind::Perlin | NoiseKind::Simplex | NoiseKind::Worley)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum WorleyFeature {
    #[default]
    F1 = 0,
    F2 = 1,
    F2MinusF1 = 2,
    /// A flat random value per cell (what the Voronoi node used to output).
    CellValue = 3,
    /// The exact distance to the nearest cell wall: 0 on the cracks between cells.
    Edges = 4,
}

/// How Worley measures distance to its feature points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorleyMetric {
    #[default]
    Euclidean = 0,
    /// Sum of the axis distances: diamond-shaped cells.
    Manhattan = 1,
    /// Largest axis distance: square cells.
    Chebyshev = 2,
}

fn one() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseParams {
    pub kind: NoiseKind,
    /// Worley only.
    #[serde(default)]
    pub feature: WorleyFeature,
    /// Worley only: sum octaves (lacunarity, roughness). Off means a single octave.
    #[serde(default)]
    pub fractal: bool,
    /// Worley only: how the distance is measured (not used by Cell edges).
    #[serde(default)]
    pub metric: WorleyMetric,
    /// Worley only: how far each feature point strays from the middle of its cell, 0 (a perfect
    /// lattice) to 1 (anywhere in the cell).
    #[serde(default = "one")]
    pub jitter: f32,
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

/// Which regular pattern a `Pattern` node makes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternKind {
    /// A brick wall: rows of bricks, each row shifted along the row by a fraction of a brick.
    #[default]
    Bricks,
    /// Round dimples (or the distance to the nearest point) on a cubic, body-centred or face-centred lattice.
    Dimples,
    /// A checkerboard in 2D (extruded along an axis) or 3D.
    Checker,
    /// Parallel bands along a direction: sine, triangle, sawtooth or square waves.
    Stripes,
    /// A triply periodic minimal surface: gyroid, Schwarz P or diamond.
    Gyroid,
    /// A Menger sponge or Sierpinski tetrahedron, repeated across the tile.
    Fractal,
    /// Truchet tiles: random quarter-circle arcs or diagonals joining up across cells (2D, extruded).
    Truchet,
    /// The cells of a lattice: hexagons (2D, extruded) or the 3D cells of a body-centred or face-centred lattice.
    Honeycomb,
}

impl PatternKind {
    pub const ALL: [PatternKind; 8] = [PatternKind::Bricks, PatternKind::Dimples, PatternKind::Checker, PatternKind::Stripes, PatternKind::Gyroid, PatternKind::Fractal, PatternKind::Truchet, PatternKind::Honeycomb];

    pub fn name(self) -> &'static str {
        match self {
            PatternKind::Bricks => "Bricks",
            PatternKind::Dimples => "Dimples",
            PatternKind::Checker => "Checker",
            PatternKind::Stripes => "Stripes",
            PatternKind::Gyroid => "Gyroid",
            PatternKind::Fractal => "Fractal",
            PatternKind::Truchet => "Truchet",
            PatternKind::Honeycomb => "Honeycomb",
        }
    }

    /// The name the `.noiser` format uses.
    pub fn file_name(self) -> &'static str {
        match self {
            PatternKind::Bricks => "bricks",
            PatternKind::Dimples => "dimples",
            PatternKind::Checker => "checker",
            PatternKind::Stripes => "stripes",
            PatternKind::Gyroid => "gyroid",
            PatternKind::Fractal => "fractal",
            PatternKind::Truchet => "truchet",
            PatternKind::Honeycomb => "honeycomb",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternOutput {
    /// A measured value: the distance to the centre of the brick or dimple (-1 at the centre, +1 at
    /// the farthest point), or a gyroid's field.
    #[default]
    Distance = 0,
    /// The shape itself: -1 on the brick face and +1 in the mortar, a dimple's profile, or a gyroid's thin shell.
    Shape = 1,
    /// A flat random value in [-1, 1] per cell.
    CellValue = 2,
}

/// Settings of a `Pattern` node. Every field has a default so files and code stay valid when
/// more pattern kinds add fields; each kind reads the ones it needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PatternParams {
    pub kind: PatternKind,
    /// Cells per tile per axis (bricks along a row, rows, layers; see `axes`). Whole numbers tile.
    pub counts: [f32; 3],
    /// Extrude a 2D pattern along `axis` instead of using all three axes.
    pub mode2d: bool,
    /// The extrusion axis in 2D mode: 0 = X, 1 = Y, 2 = Z.
    pub axis: u32,
    /// Bricks: row k is shifted along the row by (k mod steps) / steps of a brick: 1 stacks bricks in
    /// a grid, 2 is a running bond, 3 shifts by thirds.
    pub steps: u32,
    /// Bricks: mortar width and edge softness, as fractions of half a brick.
    pub gap: f32,
    pub bevel: f32,
    pub output: PatternOutput,
    /// Overrides the global seed (Cell value output).
    pub seed: Option<u32>,
    /// Which lattice (Dimples: cubic, body-centred, face-centred), wave (Stripes: sine, triangle,
    /// sawtooth, square) or surface (Gyroid: gyroid, Schwarz P, diamond).
    pub variant: u32,
    /// Dimples: the dimple radius as a fraction of the room each one has (1 touches its neighbours).
    pub radius: f32,
    /// Dimples: the profile (sphere cap, cone, smooth bowl, flat disc).
    pub profile: u32,
    /// Stripes: the direction the phase runs along; whole numbers tile.
    pub dir: [f32; 3],
    /// Fractal: how many times the shape is subdivided (1 to 6).
    pub iterations: u32,
    /// Stripes: the share of each period that is +1 (square wave); Gyroid: the shell thickness;
    /// Truchet: the width of the line, as a share of a cell.
    pub width: f32,
    /// Checker, Stripes: how soft the edges are, 0 (hard) to 1.
    pub softness: f32,
}

impl Default for PatternParams {
    fn default() -> Self {
        PatternParams::for_kind(PatternKind::Bricks)
    }
}

impl PatternParams {
    /// A new node of this kind, with settings that make a sensible first look.
    pub fn for_kind(kind: PatternKind) -> PatternParams {
        // 8 bricks per row and 16 rows: bricks twice as wide as tall, as in a wall.
        let base = PatternParams {
            kind,
            counts: [8.0, 16.0, 8.0],
            mode2d: true,
            axis: 2,
            steps: 2,
            gap: 0.1,
            bevel: 0.1,
            output: PatternOutput::Distance,
            seed: None,
            variant: 0,
            radius: 0.9,
            profile: 0,
            dir: [1.0, 0.0, 0.0],
            iterations: 3,
            width: 0.5,
            softness: 0.1,
        };
        match kind {
            PatternKind::Bricks => base,
            PatternKind::Dimples => PatternParams { counts: [4.0; 3], mode2d: false, variant: 1, output: PatternOutput::Shape, ..base },
            PatternKind::Checker => PatternParams { counts: [8.0; 3], softness: 0.05, ..base },
            PatternKind::Stripes => PatternParams { counts: [8.0, 1.0, 1.0], mode2d: false, ..base },
            PatternKind::Gyroid => PatternParams { counts: [2.0; 3], mode2d: false, width: 0.25, ..base },
            PatternKind::Fractal => PatternParams { counts: [1.0; 3], mode2d: false, output: PatternOutput::Shape, ..base },
            PatternKind::Truchet => PatternParams { counts: [6.0, 6.0, 1.0], width: 0.3, softness: 0.1, output: PatternOutput::Shape, ..base },
            // 7 x 4 cells make nearly regular hexagons (the cell is 7:4, regular wants 1.732).
            PatternKind::Honeycomb => PatternParams { counts: [7.0, 4.0, 4.0], output: PatternOutput::Shape, ..base },
        }
    }

    /// (axis along a row, axis rows stack along, axis of layers in 3D). In 2D mode the pattern is
    /// extruded along `axis` and the other two axes, in order, are the row and the stacking axis.
    pub fn axes(&self) -> (usize, usize, Option<usize>) {
        if !self.mode2d || (self.kind == PatternKind::Truchet && self.variant == 2) {
            return (0, 1, Some(2));
        }
        match self.axis {
            0 => (1, 2, None),
            1 => (0, 2, None),
            _ => (0, 1, None),
        }
    }

    /// The axes whose counts matter.
    pub fn used_axes(&self) -> Vec<usize> {
        match self.kind {
            PatternKind::Stripes => vec![0],
            PatternKind::Dimples | PatternKind::Gyroid | PatternKind::Fractal => vec![0, 1, 2],
            PatternKind::Bricks | PatternKind::Checker | PatternKind::Truchet | PatternKind::Honeycomb => {
                let (r, s, l) = self.axes();
                [Some(r), Some(s), l].into_iter().flatten().collect()
            }
        }
    }

    /// Why this pattern would not repeat once per tile, in plain words, or None if it does.
    pub fn tiling_problem(&self) -> Option<String> {
        let whole = |a: usize| self.counts[a].fract() == 0.0;
        match self.kind {
            PatternKind::Stripes => {
                if !whole(0) || self.dir.iter().any(|d| d.fract() != 0.0) {
                    return Some("the stripe count or direction is not a whole number".into());
                }
            }
            _ => {
                if self.used_axes().iter().any(|a| !whole(*a)) {
                    return Some(format!("{} counts are not whole numbers", self.kind.name().to_lowercase()));
                }
            }
        }
        match self.kind {
            PatternKind::Bricks => {
                let rows = self.counts[self.axes().1];
                if self.steps > 1 && (rows as u32) % self.steps != 0 {
                    return Some(format!("the number of rows ({rows}) must be a multiple of the row-offset steps ({})", self.steps));
                }
            }
            PatternKind::Honeycomb if self.variant == 3 => {
                let rows = self.counts[self.axes().1];
                if (rows as u32) % 2 != 0 {
                    return Some(format!("triangles need an even number of rows ({rows} is odd)"));
                }
            }
            PatternKind::Checker => {
                if let Some(a) = self.used_axes().into_iter().find(|a| (self.counts[*a] as u32) % 2 != 0) {
                    return Some(format!("a checker needs an even count on every used axis ({} is odd)", self.counts[a]));
                }
            }
            _ => {}
        }
        None
    }
}

impl NoiseParams {
    /// Octaves actually evaluated: Perlin and Simplex always use `octaves`, Worley only when fractal.
    pub fn octaves_used(&self) -> u32 {
        if self.uses_octaves() { self.octaves.max(1) } else { 1 }
    }

    /// Whether the octave, lacunarity and roughness controls apply.
    pub fn uses_octaves(&self) -> bool {
        self.kind.has_octaves() && (self.kind != NoiseKind::Worley || self.fractal)
    }
}

/// Files saved before Voronoi, Voronoi Fractal and Worley became one node: Voronoi is Worley with
/// the cell-value feature, Voronoi Fractal adds fractal, and a Worley that had octaves keeps them.
fn migrate_noise_kinds(v: &mut serde_json::Value) {
    let Some(nodes) = v.get_mut("nodes").and_then(|n| n.as_object_mut()) else { return };
    for node in nodes.values_mut() {
        let Some(p) = node.pointer_mut("/kind/Noise").and_then(|p| p.as_object_mut()) else { continue };
        let kind = p.get("kind").and_then(|k| k.as_str()).unwrap_or_default().to_owned();
        let octaves = p.get("octaves").and_then(|o| o.as_u64()).unwrap_or(1);
        match kind.as_str() {
            "Voronoi" | "VoronoiFractal" => {
                p.insert("kind".into(), "Worley".into());
                p.insert("feature".into(), "CellValue".into());
                p.insert("fractal".into(), (kind == "VoronoiFractal").into());
            }
            "Worley" if !p.contains_key("fractal") => _ = p.insert("fractal".into(), (octaves > 1).into()),
            _ => {}
        }
    }
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            kind: NoiseKind::Perlin,
            feature: WorleyFeature::F1,
            fractal: false,
            metric: WorleyMetric::Euclidean,
            jitter: 1.0,
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
    /// Inputs: a, b (and the blend width for smooth min/max). The stored values are used when an
    /// input is unconnected.
    Math {
        op: MathOp,
        a: f32,
        b: f32,
        #[serde(default = "default_smooth")]
        c: f32,
    },
    /// Linear map of input 0 from [from min, from max] to [to min, to max]; the other four inputs
    /// are those bounds. `vals` holds the unconnected defaults of all five inputs.
    Remap { vals: [f32; 5], clamp: bool },
    /// A function of input 0 drawn through `points` (x, y), at most `MAX_CURVE_POINTS`; x need not be
    /// sorted (use `curve_points`). Outside the first and last x the output holds the end values.
    Curve { points: Vec<[f32; 2]>, mode: CurveMode },
    /// Distance to the nearest cell centre (or, with `corners`, the nearest cell corner) in a regular
    /// grid of `counts` cells per tile, scaled to [-1, 1]: -1 at the reference point and +1 as far from
    /// it as the grid gets. Input: position (defaults to world position). Saved as "Brick" before
    /// the rename.
    #[serde(alias = "Brick")]
    Grid {
        counts: [f32; 3],
        #[serde(default)]
        corners: bool,
    },
    /// A regular pattern (see `PatternParams`). Input: position (defaults to world position).
    Pattern(PatternParams),
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
            NodeKind::Math { op, .. } => if op.is_smooth() { 3 } else if op.is_unary() { 1 } else { 2 },
            NodeKind::Remap { .. } => 5,
            NodeKind::Curve { .. } => 1,
            NodeKind::Normal { .. } | NodeKind::Noise(_) | NodeKind::Grid { .. } | NodeKind::Pattern(_) | NodeKind::OutDensity | NodeKind::OutNormal => 1,
        }
    }

    /// The stored value an unconnected input falls back to, for nodes with such inputs.
    pub fn const_input(&self, port: usize) -> Option<f32> {
        match self {
            NodeKind::Math { a, b, c, .. } if port < self.input_count() => Some([*a, *b, *c][port]),
            NodeKind::Remap { vals, .. } => vals.get(port).copied(),
            _ => None,
        }
    }

    pub fn set_const_input(&mut self, port: usize, v: f32) {
        match self {
            NodeKind::Math { a, b, c, .. } => {
                if let Some(x) = [a, b, c].into_iter().nth(port) {
                    *x = v;
                }
            }
            NodeKind::Remap { vals, .. } => {
                if let Some(x) = vals.get_mut(port) {
                    *x = v;
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Position in the graph editor.
    pub pos: [f32; 2],
    /// Preview-only cache of this node's value: None = off, Some(0) = automatic resolution,
    /// Some(n) = n³ texels per tile. Export always ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<u32>,
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
        self.nodes.insert(id, Node { kind, pos, cache: None });
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

    /// Drops links into inputs a node no longer has (a Math node switched away from smooth min/max).
    pub fn prune_edges(&mut self) {
        let nodes = &self.nodes;
        self.edges.retain(|e| nodes.get(&e.to).is_some_and(|n| e.port < n.kind.input_count()));
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
        let mut v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        migrate_noise_kinds(&mut v);
        let g: Graph = serde_json::from_value(v).map_err(|e| e.to_string())?;
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

    /// Whether `id` can be cached, or why not. A cache bakes one tile of a float field, so the
    /// node's output must be a float that repeats every tile.
    pub fn cache_eligibility(&self, id: NodeId, tiling: &BTreeMap<NodeId, Tiling>) -> Result<(), String> {
        match &self.nodes.get(&id).ok_or("no such node")?.kind {
            NodeKind::Noise(_) | NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Curve { .. } | NodeKind::Grid { .. } | NodeKind::Pattern(_) => {}
            _ => return Err("only Noise, Math, Remap, Grid and pattern nodes can be cached".into()),
        }
        // Math on a vector input yields a vector (except dot); caches hold floats.
        if matches!(self.nodes[&id].kind, NodeKind::Math { .. }) && self.yields_vector(id) {
            return Err("this Math node outputs a vector; caches hold floats".into());
        }
        match tiling.get(&id) {
            Some(Tiling::Periodic) | Some(Tiling::Const) => Ok(()),
            Some(Tiling::Broken(why)) => Err(format!("it doesn't tile ({why})")),
            _ => Err("its value depends directly on position".into()),
        }
    }

    /// Nodes the preview no longer evaluates because every path from them to `roots` passes
    /// through an active cache (the caches themselves excluded).
    pub fn cache_covered(&self, tiling: &BTreeMap<NodeId, Tiling>, roots: &[NodeId]) -> BTreeSet<NodeId> {
        let cached = |id: NodeId| self.nodes[&id].cache.is_some() && self.cache_eligibility(id, tiling).is_ok();
        let upstream = |starts: Vec<NodeId>, stop_at_cache: bool| {
            let (mut stack, mut seen) = (starts, BTreeSet::new());
            while let Some(n) = stack.pop() {
                if self.nodes.contains_key(&n) && seen.insert(n) && !(stop_at_cache && cached(n)) {
                    stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
                }
            }
            seen
        };
        let evaluated = upstream(roots.to_vec(), true);
        let behind: Vec<_> = self.edges.iter().filter(|e| cached(e.to)).map(|e| e.from).collect();
        upstream(behind, false).difference(&evaluated).copied().collect()
    }

    pub fn yields_vector(&self, id: NodeId) -> bool {
        match &self.nodes[&id].kind {
            NodeKind::Vec3(_) | NodeKind::Position | NodeKind::Normal { .. } => true,
            NodeKind::Math { op, .. } => *op != MathOp::Dot && (0..2).any(|p| self.input(id, p).is_some_and(|s| self.yields_vector(s))),
            _ => false,
        }
    }

    /// Cache resolution: the stored one, or for automatic about 4 texels per finest noise cell
    /// upstream (a power of two, 16–512).
    pub fn cache_resolution(&self, id: NodeId) -> u32 {
        match self.nodes.get(&id).and_then(|n| n.cache) {
            Some(r) if r > 0 => r,
            _ => {
                let mut finest = 0.0f32;
                let mut stack = vec![id];
                let mut seen = std::collections::BTreeSet::new();
                while let Some(n) = stack.pop() {
                    if !seen.insert(n) {
                        continue;
                    }
                    match &self.nodes[&n].kind {
                        NodeKind::Noise(p) => {
                            let oct = p.octaves_used();
                            finest = finest.max(p.scale * p.lacunarity.max(1.0).powi(oct as i32 - 1));
                        }
                        NodeKind::Grid { counts, .. } => finest = finest.max(counts.iter().cloned().fold(0.0, f32::max)),
                        NodeKind::Pattern(p) => finest = finest.max(p.used_axes().iter().map(|a| p.counts[*a]).fold(0.0, f32::max)),
                        _ => {}
                    }
                    stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
                }
                ((finest * 4.0).ceil().max(16.0) as u32).next_power_of_two().min(512)
            }
        }
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
            NodeKind::Math { op, a, b, .. } => {
                let ta = input(0, memo, stack).unwrap_or(Const);
                let tb = input(1, memo, stack).unwrap_or(Const);
                let tc = if op.is_smooth() { input(2, memo, stack) } else { None };
                // Operand is a whole-number constant: the unconnected default, or a Float/Vec3 node.
                let whole_const = |port: usize, default: f32| match self.input(id, port).map(|s| &self.nodes[&s].kind) {
                    None => is_whole(default),
                    Some(NodeKind::Float(v)) => is_whole(*v),
                    Some(NodeKind::Vec3(v)) => v.iter().all(|x| is_whole(*x)),
                    Some(_) => false,
                };
                let t = match (&ta, &tb) {
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
                };
                // The blend width of smooth min/max joins in: it must not depend on position.
                match (t, tc) {
                    (Broken(r), _) | (_, Some(Broken(r))) => Broken(r),
                    (_, Some(Coord)) => Broken("blend width depends on position".into()),
                    (Const, Some(Periodic)) | (Periodic, Some(Const | Periodic)) => Periodic,
                    (t, _) => t,
                }
            }
            NodeKind::Curve { .. } => match input(0, memo, stack) {
                Some(Broken(r)) => Broken(r),
                Some(Coord) => Broken("A curve of a position does not tile".into()),
                Some(t) => t,
                None => Const,
            },
            NodeKind::Remap { .. } => {
                // A nonlinear-looking map of position breaks tiling; the rest follows the inputs.
                let ts: Vec<Tiling> = (0..5).filter_map(|p| input(p, memo, stack)).collect();
                match ts.iter().find(|t| matches!(t, Broken(_) | Coord)) {
                    Some(Broken(r)) => Broken(r.clone()),
                    Some(_) => Broken("Remap on a position does not tile".into()),
                    None if ts.iter().all(|t| *t == Const) => Const,
                    None => Periodic,
                }
            }
            NodeKind::Pattern(p) => {
                if let Some(why) = p.tiling_problem() {
                    Broken(why)
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Grid { counts, .. } => {
                if counts.iter().any(|c| !is_whole(*c)) {
                    Broken("grid counts are not whole numbers".into())
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Noise(n) => {
                if !is_whole(n.scale) {
                    Broken(format!("scale {} is not a whole number", n.scale))
                } else if n.kind == NoiseKind::Blue && n.scale % 4.0 != 0.0 {
                    // The blue noise tables are 4³, 8³, 16³ and 32³; the tile must hold whole tables.
                    Broken(format!("blue noise scale {} is not a multiple of 4", n.scale))
                } else if n.octaves_used() > 1 && !is_whole(n.lacunarity) {
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

/// What a pin or wire carries. Links convert implicitly (float to vector repeats, vector to float
/// averages), so this is what the pin shows, not a restriction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortType {
    Float,
    Vector,
}

impl Graph {
    /// The type of a node's output. A Math node favors vectors: it outputs one when either input is
    /// a vector (the other is repeated), and only a float when it has to (dot product).
    pub fn output_type(&self, id: NodeId) -> PortType {
        if self.yields_vector(id) { PortType::Vector } else { PortType::Float }
    }

    /// The type an input expects; for Math it follows what is connected.
    pub fn input_type(&self, id: NodeId, port: usize) -> PortType {
        match &self.nodes[&id].kind {
            NodeKind::Noise(_) | NodeKind::Grid { .. } | NodeKind::Pattern(_) | NodeKind::OutNormal => PortType::Vector,
            NodeKind::Math { op, .. } if port < 2 => {
                let vector = *op == MathOp::Dot || (0..2).any(|p| self.input(id, p).is_some_and(|s| self.yields_vector(s)));
                if vector { PortType::Vector } else { PortType::Float }
            }
            _ => PortType::Float,
        }
    }
}

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
            NodeKind::Math { op, a, b, c } => {
                let mut operand = |port: usize, default: f32| match self.input(id, port) {
                    Some(src) => self.range_of(src, stack),
                    None => (default, default),
                };
                let (ra, rb, rc) = (operand(0, *a), operand(1, *b), operand(2, *c));
                math_range(*op, ra, rb, rc)
            }
            NodeKind::Remap { vals, clamp } => {
                let [v, i0, i1, o0, o1] = *vals;
                let linked = |p: usize| self.input(id, p).is_some();
                let (lo, hi) = (o0.min(o1), o0.max(o1));
                if *clamp && !linked(3) && !linked(4) {
                    (lo, hi)
                } else if (1..5).any(linked) {
                    UNBOUNDED
                } else if i0 == i1 || o0 == o1 {
                    (o0, o0)
                } else {
                    let (x0, x1) = match self.input(id, 0) {
                        Some(src) => self.range_of(src, stack),
                        None => (v, v),
                    };
                    let f = |x: f32| o0 + (x - i0) / (i1 - i0) * (o1 - o0);
                    if x0.is_finite() && x1.is_finite() { (f(x0).min(f(x1)), f(x0).max(f(x1))) } else { UNBOUNDED }
                }
            }
            NodeKind::Curve { points, .. } => {
                let p = curve_points(points);
                (p.iter().map(|q| q[1]).fold(f32::MAX, f32::min), p.iter().map(|q| q[1]).fold(f32::MIN, f32::max))
            }
            NodeKind::Grid { .. } | NodeKind::Pattern(_) => (-1.0, 1.0),
            NodeKind::OutDensity | NodeKind::OutNormal => UNBOUNDED,
        };
        stack.pop();
        r
    }
}

fn math_range(op: MathOp, (a0, a1): Range, (b0, b1): Range, (_, k1): Range) -> Range {
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
        MathOp::Power => UNBOUNDED,
        // Within k/6 of the plain min or max, on one side only.
        MathOp::SmoothMin => (a0.min(b0) - k1.max(0.0) / 6.0, a1.min(b1)),
        MathOp::SmoothMax => (a0.max(b0), a1.max(b1) + k1.max(0.0) / 6.0),
        MathOp::Dot => {
            // Every component lies in the range, so |dot| <= 3 * max|a| * max|b|.
            let m = 3.0 * a0.abs().max(a1.abs()) * b0.abs().max(b1.abs());
            if m.is_finite() { (-m, m) } else { UNBOUNDED }
        }
        MathOp::Absolute | MathOp::Ridge => {
            let abs = (if a0 <= 0.0 && a1 >= 0.0 { 0.0 } else { a0.abs().min(a1.abs()) }, a0.abs().max(a1.abs()));
            if op == MathOp::Absolute { abs } else { (1.0 - abs.1, 1.0 - abs.0) }
        }
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
        let mul = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 3.0, c: 0.5 }, [0.0; 2]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.5, c: 0.5 }, [0.0; 2]);
        g.connect(noise, mul, 0);
        g.connect(mul, add, 0);
        g.connect(add, out, 0);
        assert_eq!(g.density_range(), (-2.5, 3.5));
        // Position reaching the output makes it unbounded.
        let pos = g.find(|k| *k == NodeKind::Position).unwrap();
        g.connect(pos, add, 1);
        assert_eq!(g.density_range(), UNBOUNDED);
        // Dividing by a range that spans zero is unbounded too.
        assert_eq!(math_range(MathOp::Divide, (1.0, 1.0), (-1.0, 1.0), (0.5, 0.5)), UNBOUNDED);
        assert_eq!(math_range(MathOp::Divide, (1.0, 2.0), (2.0, 4.0), (0.5, 0.5)), (0.25, 1.0));
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
        let times2 = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 2.0, c: 0.5 }, [0.0; 2]);
        let times_half = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.5, c: 0.5 }, [0.0; 2]);
        let n1 = g.add(NodeKind::Noise(NoiseParams::default()), [0.0; 2]);
        // Domain warp: noise(p + noise(p)) still tiles.
        let warp = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
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
        let a = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let b = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(a, b, 0);
        g.connect(b, a, 0);
        assert!(matches!(tiling_at(&g, a), Tiling::Broken(_)));
    }

    #[test]
    fn old_voronoi_files_migrate_to_worley() {
        // Rewrites the default graph's noise node as an old-format file would have stored it.
        let old = |kind: &str, octaves: u64, drop_fractal: bool| {
            let mut v: serde_json::Value = serde_json::from_str(&Graph::default_graph().to_json()).unwrap();
            let p = v.pointer_mut("/nodes/1/kind/Noise").unwrap().as_object_mut().unwrap();
            p.insert("kind".into(), kind.into());
            p.insert("octaves".into(), octaves.into());
            if drop_fractal {
                p.remove("fractal");
            }
            let g = Graph::from_json(&v.to_string()).unwrap();
            match &g.nodes[&1].kind {
                NodeKind::Noise(n) => n.clone(),
                _ => unreachable!(),
            }
        };
        let v = old("Voronoi", 4, true);
        assert_eq!((v.kind, v.feature, v.fractal), (NoiseKind::Worley, WorleyFeature::CellValue, false));
        let vf = old("VoronoiFractal", 3, true);
        assert_eq!((vf.kind, vf.feature, vf.fractal, vf.octaves_used()), (NoiseKind::Worley, WorleyFeature::CellValue, true, 3));
        // A Worley that used octaves stays fractal; one with a single octave doesn't.
        assert!(old("Worley", 4, true).fractal);
        assert!(!old("Worley", 1, true).fractal);
        // The test graph saved before the merge still opens, with its Worley kept fractal.
        let g = Graph::from_json(include_str!("../SimpleTestGraph.noisegraph")).unwrap();
        assert!(g.nodes.values().any(|n| matches!(&n.kind, NodeKind::Noise(p) if p.kind == NoiseKind::Worley && p.fractal)));
        // New files keep what they say.
        assert!(!old("Worley", 4, false).fractal);
        assert_eq!(old("Worley", 4, false).octaves_used(), 1);
    }

    #[test]
    fn math_favors_vectors_and_dot_gives_a_float() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 1.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.output_type(add), PortType::Float);
        g.connect(0, add, 0); // a vector and an unconnected float
        assert_eq!((g.output_type(add), g.input_type(add, 0), g.input_type(add, 1)), (PortType::Vector, PortType::Vector, PortType::Vector));
        g.connect(1, add, 1); // a vector and a connected float
        assert_eq!(g.output_type(add), PortType::Vector);
        let dot = g.add(NodeKind::Math { op: MathOp::Dot, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!((g.output_type(dot), g.input_type(dot, 0)), (PortType::Float, PortType::Vector));
        assert_eq!((g.output_type(0), g.output_type(1)), (PortType::Vector, PortType::Float));
        assert_eq!((g.input_type(1, 0), g.input_type(2, 0), g.input_type(3, 0)), (PortType::Vector, PortType::Float, PortType::Vector));
        // Caches hold floats: a vector-valued Math node can't be cached, a dot can (when it tiles).
        assert!(g.cache_eligibility(add, &g.tiling()).is_err());
    }

    #[test]
    fn remap_grid_and_smooth_ops_in_the_graph() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise(-1..1) → 2 OutDensity
        let t = |g: &Graph, id| g.tiling()[&id].clone();
        // Smooth min/max grow a third input; switching away prunes the link into it.
        let sm = g.add(NodeKind::Math { op: MathOp::SmoothMin, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.nodes[&sm].kind.input_count(), 3);
        g.connect(1, sm, 2);
        assert_eq!(t(&g, sm), Tiling::Periodic);
        g.connect(0, sm, 2); // a position as the blend width doesn't tile
        assert!(matches!(t(&g, sm), Tiling::Broken(_)));
        g.nodes.get_mut(&sm).unwrap().kind = NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 };
        g.prune_edges();
        assert!(g.input(sm, 2).is_none());
        // Ranges.
        let grid = g.add(NodeKind::Grid { counts: [8.0; 3], corners: false }, [0.0; 2]);
        g.connect(grid, 2, 0);
        assert_eq!(g.density_range(), (-1.0, 1.0));
        let remap = g.add(NodeKind::Remap { vals: [0.0, 0.0, 1.0, -5.0, 5.0], clamp: false }, [0.0; 2]);
        g.connect(1, remap, 0);
        g.connect(remap, 2, 0);
        assert_eq!(g.density_range(), (-15.0, 5.0)); // noise in [-1, 1] through a ×10 map from [0, 1]
        if let NodeKind::Remap { clamp, .. } = &mut g.nodes.get_mut(&remap).unwrap().kind {
            *clamp = true;
        }
        assert_eq!(g.density_range(), (-5.0, 5.0));
        // Tiling.
        assert_eq!(t(&g, grid), Tiling::Periodic);
        assert_eq!(t(&g, remap), Tiling::Periodic);
        g.connect(0, remap, 0);
        assert!(matches!(t(&g, remap), Tiling::Broken(_))); // remapping a position
        g.nodes.get_mut(&grid).unwrap().kind = NodeKind::Grid { counts: [8.5, 8.0, 8.0], corners: false };
        assert!(matches!(t(&g, grid), Tiling::Broken(_)));
        // The node was called Brick (without the corner option) until it was renamed.
        let old = serde_json::from_str::<NodeKind>(r#"{"Brick":{"counts":[4.0,4.0,4.0]}}"#).unwrap();
        assert_eq!(old, NodeKind::Grid { counts: [4.0; 3], corners: false });
        // A Math node saved before the blend width existed still loads.
        let old = r#"{"op":"Add","a":1.0,"b":2.0}"#;
        assert!(matches!(serde_json::from_str::<NodeKind>(&format!("{{\"Math\":{old}}}")), Ok(NodeKind::Math { c, .. }) if c == 0.5));
    }

    #[test]
    fn bricks_tiling_rules_and_axes() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let p = |f: &dyn Fn(&mut PatternParams)| {
            let mut p = PatternParams::default();
            f(&mut p);
            p
        };
        let tiling = |g: &mut Graph, p: PatternParams| {
            let id = g.add(NodeKind::Pattern(p), [0.0; 2]);
            let t = g.tiling()[&id].clone();
            g.remove(id);
            t
        };
        assert_eq!(tiling(&mut g, PatternParams::default()), Tiling::Periodic);
        // Rows must divide into the offset steps, or the pattern jumps at the tile edge.
        assert!(matches!(tiling(&mut g, p(&|p| p.counts[1] = 15.0)), Tiling::Broken(r) if r.contains("multiple of the row-offset")));
        assert_eq!(tiling(&mut g, p(&|p| { p.counts[1] = 15.0; p.steps = 1; })), Tiling::Periodic);
        assert_eq!(tiling(&mut g, p(&|p| { p.counts[1] = 15.0; p.steps = 3; })), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, p(&|p| p.counts[0] = 7.5)), Tiling::Broken(_)));
        // The layer axis only counts in 3D.
        assert_eq!(tiling(&mut g, p(&|p| p.counts[2] = 7.5)), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, p(&|p| { p.counts[2] = 7.5; p.mode2d = false; })), Tiling::Broken(_)));
        // Which axes are the row, the stack and the extrusion.
        assert_eq!(p(&|p| p.axis = 2).axes(), (0, 1, None));
        assert_eq!(p(&|p| p.axis = 1).axes(), (0, 2, None));
        assert_eq!(p(&|p| p.axis = 0).axes(), (1, 2, None));
        assert_eq!(p(&|p| p.mode2d = false).axes(), (0, 1, Some(2)));
        // Range, input pin, and a file written without the newer fields still loads.
        let id = g.add(NodeKind::Pattern(PatternParams::default()), [0.0; 2]);
        assert_eq!((g.nodes[&id].kind.input_count(), g.input_type(id, 0)), (1, PortType::Vector));
        g.connect(id, 2, 0);
        assert_eq!(g.density_range(), (-1.0, 1.0));
        let min: NodeKind = serde_json::from_str(r#"{"Pattern":{"steps":3}}"#).unwrap();
        assert!(matches!(min, NodeKind::Pattern(q) if q.steps == 3 && q.counts == [8.0, 16.0, 8.0] && q.mode2d));
    }

    #[test]
    fn unary_math_ops_have_one_input_and_the_right_ranges() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise(-1..1) → 2 OutDensity
        let m = g.add(NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.nodes[&m].kind.input_count(), 1);
        g.connect(1, m, 0);
        g.connect(m, 2, 0);
        assert_eq!(g.density_range(), (0.0, 1.0)); // |[-1, 1]|
        g.nodes.get_mut(&m).unwrap().kind = NodeKind::Math { op: MathOp::Ridge, a: 0.0, b: 0.0, c: 0.5 };
        assert_eq!(g.density_range(), (0.0, 1.0)); // 1 - |[-1, 1]|
        assert_eq!(math_range(MathOp::Absolute, (2.0, 5.0), (0.0, 0.0), (0.0, 0.0)), (2.0, 5.0));
        assert_eq!(math_range(MathOp::Absolute, (-5.0, -2.0), (0.0, 0.0), (0.0, 0.0)), (2.0, 5.0));
        assert_eq!(math_range(MathOp::Ridge, (2.0, 5.0), (0.0, 0.0), (0.0, 0.0)), (-4.0, -1.0));
        // Switching from Add to Absolute drops a link into B.
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, add, 0);
        g.connect(0, add, 1);
        g.nodes.get_mut(&add).unwrap().kind = NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 };
        g.prune_edges();
        assert!(g.input(add, 1).is_none());
        // |noise| still tiles; |position| does not.
        assert_eq!(g.tiling()[&m], Tiling::Periodic);
        let on_position = g.add(NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(0, on_position, 0);
        assert!(matches!(g.tiling()[&on_position], Tiling::Broken(_)));
    }

    #[test]
    fn pattern_tiling_rules_per_kind() {
        let mut g = Graph::default_graph();
        let tiling = |g: &mut Graph, p: PatternParams| {
            let id = g.add(NodeKind::Pattern(p), [0.0; 2]);
            let t = g.tiling()[&id].clone();
            g.remove(id);
            t
        };
        let with = |kind: PatternKind, f: &dyn Fn(&mut PatternParams)| {
            let mut p = PatternParams::for_kind(kind);
            f(&mut p);
            p
        };
        for kind in PatternKind::ALL {
            assert_eq!(tiling(&mut g, PatternParams::for_kind(kind)), Tiling::Periodic, "{kind:?} should tile by default");
        }
        // A checker needs even counts on the axes it uses; the extrusion axis does not count.
        assert!(matches!(tiling(&mut g, with(PatternKind::Checker, &|p| p.counts[0] = 7.0)), Tiling::Broken(r) if r.contains("even")));
        assert_eq!(tiling(&mut g, with(PatternKind::Checker, &|p| p.counts[2] = 7.0)), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, with(PatternKind::Checker, &|p| { p.counts[2] = 7.0; p.mode2d = false; })), Tiling::Broken(_)));
        // Stripes need a whole count and a whole direction.
        assert!(matches!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.counts[0] = 4.5)), Tiling::Broken(r) if r.contains("stripe")));
        assert!(matches!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.dir = [1.0, 0.5, 0.0])), Tiling::Broken(_)));
        assert_eq!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.dir = [2.0, -3.0, 1.0])), Tiling::Periodic);
        // Dimples and gyroids need whole counts on all three axes.
        for kind in [PatternKind::Dimples, PatternKind::Gyroid] {
            assert!(matches!(tiling(&mut g, with(kind, &|p| p.counts[1] = 2.5)), Tiling::Broken(_)), "{kind:?}");
        }
        // Every kind survives a save and load.
        for kind in PatternKind::ALL {
            let n = NodeKind::Pattern(PatternParams::for_kind(kind));
            assert_eq!(serde_json::from_str::<NodeKind>(&serde_json::to_string(&n).unwrap()).unwrap(), n);
        }
    }

    #[test]
    fn cache_rules() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Perlin(scale 4, 4 octaves, lac 2) → 2 OutDensity
        let t = g.tiling();
        assert!(g.cache_eligibility(1, &t).is_ok());
        assert!(g.cache_eligibility(0, &t).is_err()); // Position
        assert!(g.cache_eligibility(2, &t).is_err()); // output
        // Finest cells 4 × 2³ = 32 → 4 texels each → 128³.
        g.nodes.get_mut(&1).unwrap().cache = Some(0);
        assert_eq!(g.cache_resolution(1), 128);
        g.nodes.get_mut(&1).unwrap().cache = Some(64);
        assert_eq!(g.cache_resolution(1), 64);
        // A non-tiling noise can't be cached.
        if let NodeKind::Noise(n) = &mut g.nodes.get_mut(&1).unwrap().kind {
            n.scale = 2.5;
        }
        assert!(g.cache_eligibility(1, &g.tiling()).is_err());
        // Math on a vector yields a vector.
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(0, m, 0);
        assert!(g.cache_eligibility(m, &g.tiling()).is_err());
    }

    #[test]
    fn cache_covers_only_nodes_no_uncached_path_needs() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let roots = [2, 3];
        assert!(g.cache_covered(&g.tiling(), &roots).is_empty());
        g.nodes.get_mut(&1).unwrap().cache = Some(0);
        assert_eq!(g.cache_covered(&g.tiling(), &roots), BTreeSet::from([0]));
        // Position also feeds an uncached branch to the output, so it's still evaluated.
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, m, 0);
        g.connect(0, m, 1);
        g.connect(m, 2, 0);
        assert!(g.cache_covered(&g.tiling(), &roots).is_empty());
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
