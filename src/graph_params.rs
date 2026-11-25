//! Parameter types of the graph nodes: curves, math ops, noise, pattern and radial settings.

use serde::{Deserialize, Serialize};

pub const MAX_CURVE_POINTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveMode {
    Linear,
    /// Smoothstep between neighbouring points (flat at every point).
    Smooth,
    /// Each y holds until the next point.
    Step,
    /// A smooth cubic through the points that never overshoots them (monotone cubic Hermite).
    Cubic,
}

impl CurveMode {
    pub const ALL: [CurveMode; 4] = [CurveMode::Linear, CurveMode::Smooth, CurveMode::Step, CurveMode::Cubic];
    pub fn label(self) -> &'static str {
        match self {
            CurveMode::Linear => "Linear",
            CurveMode::Smooth => "Smooth",
            CurveMode::Step => "Step",
            CurveMode::Cubic => "Cubic",
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

/// The curve value at `x`, as the shader computes it (`curve_eval` in noise.wgsl): `points` sorted by x.
pub fn curve_value(points: &[[f32; 2]], mode: CurveMode, x: f32) -> f32 {
    let n = points.len();
    if n == 0 {
        return 0.0;
    }
    if x <= points[0][0] {
        return points[0][1];
    }
    let secant = |i: usize| (points[i + 1][1] - points[i][1]) / (points[i + 1][0] - points[i][0]).max(1e-12);
    let slope = |k: usize| {
        if n < 2 {
            return 0.0;
        }
        if k == 0 {
            return secant(0);
        }
        if k == n - 1 {
            return secant(n - 2);
        }
        let (d0, d1) = (secant(k - 1), secant(k));
        if d0 * d1 <= 0.0 {
            return 0.0;
        }
        let (h0, h1) = (points[k][0] - points[k - 1][0], points[k + 1][0] - points[k][0]);
        let (w0, w1) = (2.0 * h1 + h0, h1 + 2.0 * h0);
        (w0 + w1) / (w0 / d0 + w1 / d1)
    };
    for i in 1..n {
        if x < points[i][0] {
            let (x0, y0, x1, y1) = (points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]);
            let mut t = (x - x0) / (x1 - x0).max(1e-12);
            match mode {
                CurveMode::Step => return y0,
                CurveMode::Smooth => t = t * t * (3.0 - 2.0 * t),
                CurveMode::Cubic => {
                    let (h, t2) = (x1 - x0, t * t);
                    let t3 = t2 * t;
                    return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * slope(i - 1) + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * h * slope(i);
                }
                CurveMode::Linear => {}
            }
            return y0 + (y1 - y0) * t;
        }
    }
    points[n - 1][1]
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
    /// floor(a) (unary): steps and terraces when applied to a scaled field.
    Floor,
    /// a - floor(a) (unary): the fractional part, a sawtooth.
    Fract,
    /// sin(a) (unary, radians).
    Sine,
    /// sqrt(max(a, 0)) (unary).
    Sqrt,
    /// a - b * floor(a / b), 0 when b is 0 (the result has the sign of b).
    Modulo,
    /// 1 where a > b, else 0.
    Greater,
    /// a + (b - a) * t, the third input being t (not limited to 0..1).
    Lerp,
    /// (v - a) / (b - a), 0 when a = b; the third input is v.
    InverseLerp,
}

impl MathOp {
    pub const ALL: [MathOp; 20] = [
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
        MathOp::Floor,
        MathOp::Fract,
        MathOp::Sine,
        MathOp::Sqrt,
        MathOp::Modulo,
        MathOp::Greater,
        MathOp::Lerp,
        MathOp::InverseLerp,
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
            MathOp::Floor => "Floor",
            MathOp::Fract => "Fraction",
            MathOp::Sine => "Sine",
            MathOp::Sqrt => "Square root",
            MathOp::Modulo => "Modulo",
            MathOp::Greater => "Greater than",
            MathOp::Lerp => "Lerp (a to b by t)",
            MathOp::InverseLerp => "Inverse lerp",
        }
    }

    /// The ops with a third input: the blend width of smooth min/max, the factor of Lerp, the value of Inverse lerp.
    pub fn is_ternary(self) -> bool {
        matches!(self, MathOp::SmoothMin | MathOp::SmoothMax | MathOp::Lerp | MathOp::InverseLerp)
    }

    /// Absolute and Ridge use only the first input.
    pub fn is_unary(self) -> bool {
        matches!(self, MathOp::Absolute | MathOp::Ridge | MathOp::Floor | MathOp::Fract | MathOp::Sine | MathOp::Sqrt)
    }
}

pub(crate) fn default_smooth() -> f32 {
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

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn one() -> f32 {
    1.0
}

pub(crate) fn tenth() -> f32 {
    0.1
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
    /// The output is multiplied by this (see `Graph::lowered`).
    #[serde(default = "one")]
    pub amount: f32,
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
    /// The output is multiplied by this (see `Graph::lowered`).
    pub amount: f32,
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
            amount: 1.0,
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
                if self.steps > 1 && !(rows as u32).is_multiple_of(self.steps) {
                    return Some(format!("the number of rows ({rows}) must be a multiple of the row-offset steps ({})", self.steps));
                }
            }
            PatternKind::Honeycomb if self.variant == 3 => {
                let rows = self.counts[self.axes().1];
                if !(rows as u32).is_multiple_of(2) {
                    return Some(format!("triangles need an even number of rows ({rows} is odd)"));
                }
            }
            PatternKind::Checker => {
                if let Some(a) = self.used_axes().into_iter().find(|a| !(self.counts[*a] as u32).is_multiple_of(2)) {
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
pub(crate) fn migrate_noise_kinds(v: &mut serde_json::Value) {
    // Separate XYZ used to be one node per component ({"Separate": {"component": c}}); it now has three outputs.
    let separates: Vec<(u64, u64)> = v
        .get("nodes")
        .and_then(|n| n.as_object())
        .map(|nodes| nodes.iter().filter_map(|(id, n)| Some((id.parse().ok()?, n.pointer("/kind/Separate/component")?.as_u64().unwrap_or(0)))).collect())
        .unwrap_or_default();
    for (id, component) in separates {
        v["nodes"][id.to_string()]["kind"] = "Separate".into();
        if let Some(edges) = v.get_mut("edges").and_then(|e| e.as_array_mut()) {
            for e in edges.iter_mut().filter(|e| e.get("from").and_then(|f| f.as_u64()) == Some(id)) {
                e["from_port"] = component.into();
            }
        }
    }
    let Some(nodes) = v.get_mut("nodes").and_then(|n| n.as_object_mut()) else { return };
    for node in nodes.values_mut() {
        // Position used to be a bare name; it now holds its X, Y and Z scale.
        if node.get("kind").and_then(|k| k.as_str()) == Some("Position") {
            node["kind"] = serde_json::json!({"Position": {"scale": [1.0, 1.0, 1.0]}});
            continue;
        }
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
            amount: 1.0,
        }
    }
}

/// How a Radial position describes a point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadialMode {
    /// (distance from the axis, angle around it, height along it).
    #[default]
    Cylindrical,
    /// (distance from the centre, angle around the axis, angle from the axis).
    Spherical,
}

/// A position in radial coordinates: the angles are in radians, the angle around the axis jumps from pi to -pi on one half plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadialParams {
    pub mode: RadialMode,
    pub center: [f32; 3],
    /// The direction of the centre line (any length; zero means Z).
    pub axis: [f32; 3],
}

impl Default for RadialParams {
    fn default() -> Self {
        RadialParams { mode: RadialMode::Cylindrical, center: [0.0; 3], axis: [0.0, 0.0, 1.0] }
    }
}
