//! The distance field of an SDF Modeler assembly: what a `.sdf` file (see `sdf_file`) describes, as a function from a
//! point to a signed distance. The rules were measured against meshes exported from SDF Modeler; they are written up in
//! `docs/sdf-modeler-notes.md`, and `tools/sdf_calibration` holds the scripts that check them.
//!
//! Splines and control points are not supported: a layer that has them reports `has_unsupported` and leaves them out.

#![allow(dead_code)] // until the SDF Composer and the Load SDF node use it

use crate::sdf_file::{Layer, Node, SdfFile};
use std::f64::consts::{PI, TAU};

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

// ---- small maths ------------------------------------------------------------------------------------------------

/// Rotation matrix for Euler angles in radians: `Rz · Ry · Rx`.
pub fn euler_matrix(angles: V3) -> M3 {
    let (sx, cx) = angles[0].sin_cos();
    let (sy, cy) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    multiply(&rz, &multiply(&ry, &rx))
}

fn multiply(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// `Mᵀ · v`: a world vector into the frame the rotation `M` describes.
pub fn into_frame(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// GLSL `sign`: 0 for 0 (Rust's `signum` gives 1).
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn glsl_mod(x: f64, y: f64) -> f64 {
    x - y * (x / y).floor()
}

fn smooth_abs(value: f64, width: f64) -> f64 {
    let a = value.abs();
    if width <= 0.0 || a > width {
        return a;
    }
    let n = width * 0.25;
    let k0 = 2.0 * n - width;
    let k1 = 2.0 * width - 3.0 * n;
    let t = a / width;
    (k0 * t + k1) * t * t + n
}

// ---- combining shapes -------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Add,
    Sub,
    Intersect,
    Paint,
    Push,
    Avoid,
    Emboss,
    Deboss,
    Shell,
    None,
}

impl Mode {
    /// The number the GPU interpreter switches on.
    fn code(self) -> f32 {
        match self {
            Mode::Add => 0.0,
            Mode::Sub => 1.0,
            Mode::Intersect => 2.0,
            Mode::Paint => 3.0,
            Mode::Push => 4.0,
            Mode::Avoid => 5.0,
            Mode::Emboss => 6.0,
            Mode::Deboss => 7.0,
            Mode::Shell => 8.0,
            Mode::None => 9.0,
        }
    }

    fn parse(text: Option<&str>) -> Mode {
        match text {
            Some("sub") => Mode::Sub,
            Some("intersect") => Mode::Intersect,
            Some("paint") => Mode::Paint,
            Some("push") => Mode::Push,
            Some("avoid") => Mode::Avoid,
            Some("emboss") => Mode::Emboss,
            Some("deboss") => Mode::Deboss,
            Some("shell") => Mode::Shell,
            Some("none") => Mode::None,
            _ => Mode::Add,
        }
    }
}

fn union_smooth(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// `b` minus `a`.
fn subtract_smooth(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return b.max(-a);
    }
    let h = (0.5 - 0.5 * (b + a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) - a * h + k * h * (1.0 - h)
}

fn intersect_smooth(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (0.5 - 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h + k * h * (1.0 - h)
}

fn union_chamfer(a: f64, b: f64, k: f64) -> f64 {
    let half = 0.5 * k;
    a.min(b).min((a - half + b) * 0.5)
}

fn intersect_chamfer(a: f64, b: f64, k: f64) -> f64 {
    let half = 0.5 * k;
    a.max(b).max((b + half + a) * 0.5)
}

fn subtract_chamfer(a: f64, b: f64, k: f64) -> f64 {
    intersect_chamfer(-a, b, k)
}

/// One blend: `a` is the shape being added, `b` what is accumulated so far, `k` the blend width and `r` the radius of
/// the modes that have one.
fn combine(mode: Mode, a: f64, b: f64, k: f64, r: f64, chamfer: bool) -> f64 {
    let union = |x, y| if chamfer { union_chamfer(x, y, k) } else { union_smooth(x, y, k) };
    let subtract = |x, y| if chamfer { subtract_chamfer(x, y, k) } else { subtract_smooth(x, y, k) };
    let intersect = |x, y| if chamfer { intersect_chamfer(x, y, k) } else { intersect_smooth(x, y, k) };
    match mode {
        Mode::Add => union(a, b),
        Mode::Sub => subtract(a, b),
        Mode::Intersect => intersect(a, b),
        Mode::Paint | Mode::None => b,
        Mode::Push => subtract(a - r, b).min(a),
        Mode::Avoid => subtract(b - r, a).min(b),
        Mode::Emboss => union(b, intersect(a, b - r)),
        Mode::Deboss => subtract(subtract(b + r, a), b),
        Mode::Shell => {
            let radius = r + 0.01;
            intersect(a, (b + radius).abs() - radius)
        }
    }
}

// ---- primitives -------------------------------------------------------------------------------------------------

/// The shader's "uber primitive": a box with rounded vertical edges (`r.0`), rounded horizontal edges (`r.1`), a
/// taper (`r.2`) and an optional wall (`s.3`). `s` is the half size plus the wall, `r` the radii.
fn uber_primitive(p: V3, size: [f64; 4], radii: [f64; 3]) -> f64 {
    let (mut s, mut r) = (size, radii);
    s[0] -= r[0];
    s[2] -= r[0];
    r[0] -= s[3];
    s[3] -= r[1];
    s[1] -= r[1];
    let b = [r[2], -2.0 * s[1]];
    let length_squared = (b[0] * b[0] + b[1] * b[1]).max(1e-30);
    let ba = [b[0] / length_squared, b[1] / length_squared];
    let d = [p[0].abs() - s[0], p[1].abs() - s[1], p[2].abs() - s[2]];
    let mut q = d[0].max(0.0).hypot(d[2].max(0.0)) + d[0].max(d[2]).min(0.0) - r[0];
    q = q.abs() - s[3];
    let pa = [q, p[1] - s[1]];
    let t = (pa[0] * ba[0] + pa[1] * ba[1]).clamp(0.0, 1.0);
    let diagonal = [pa[0] - r[2] * t, pa[1] - b[1] * t];
    let h0 = [(q - r[2]).max(0.0), p[1] + s[1]];
    let h1 = [q.max(0.0), p[1] - s[1]];
    let squared = |v: [f64; 2]| v[0] * v[0] + v[1] * v[1];
    let nearest = squared(diagonal).min(squared(h0)).min(squared(h1));
    let side = sign((pa[0] * -ba[1] + pa[1] * ba[0]).max(d[1]));
    nearest.sqrt() * side - r[1]
}

fn ellipse_distance(p: [f64; 2], radii: [f64; 2]) -> f64 {
    let p = [p[0].abs(), p[1].abs()];
    let q = [radii[0] * (p[0] - radii[0]), radii[1] * (p[1] - radii[1])];
    let seed: [f64; 2] = if q[0] < q[1] { [0.01, 1.0] } else { [1.0, 0.01] };
    let length = seed[0].hypot(seed[1]);
    let mut cs = [seed[0] / length, seed[1] / length];
    for _ in 0..5 {
        let u = [radii[0] * cs[0], radii[1] * cs[1]];
        let v = [radii[0] * -cs[1], radii[1] * cs[0]];
        let a = (p[0] - u[0]) * v[0] + (p[1] - u[1]) * v[1];
        let c = (p[0] - u[0]) * u[0] + (p[1] - u[1]) * u[1] + v[0] * v[0] + v[1] * v[1];
        let b = (c * c - a * a).max(0.0).sqrt();
        cs = [(cs[0] * b - cs[1] * a) / c, (cs[1] * b + cs[0] * a) / c];
    }
    let d = (p[0] - radii[0] * cs[0]).hypot(p[1] - radii[1] * cs[1]);
    let scaled = (p[0] / radii[0]).powi(2) + (p[1] / radii[1]).powi(2);
    if scaled > 1.0 { d } else { -d }
}

fn cylinder_distance(p: V3, radii: V3) -> f64 {
    let d = ellipse_distance([p[0], p[2]], [radii[0], radii[2]]);
    let w = [d, p[1].abs() - radii[1]];
    w[0].max(w[1]).min(0.0) + w[0].max(0.0).hypot(w[1].max(0.0))
}

fn polygon_distance(p: V3, radii: [f64; 2], points: f64, star: f64, bevel: f64) -> f64 {
    let m = points + star * (2.0 - points);
    let an = PI / points;
    let en = PI / m;
    let racs = [radii[0] * an.cos(), radii[0] * an.sin()];
    let ecs = [en.cos(), en.sin()];
    let (x, y, z) = (p[0].abs(), p[1].abs(), p[2]);
    let bn = glsl_mod(x.atan2(z), 2.0 * an) - an;
    let radius = x.hypot(z);
    let mut px = radius * bn.cos() - racs[0];
    let mut pz = radius * bn.sin().abs() - racs[1];
    let t = (-(px * ecs[0] + pz * ecs[1])).clamp(0.0, racs[1] / ecs[1]);
    px += ecs[0] * t;
    pz += ecs[1] * t;
    let d1 = px.hypot(pz) * sign(px) - bevel;
    let d2 = y - radii[1];
    d1.max(d2).min(0.0) + d1.max(0.0).hypot(d2.max(0.0))
}

fn prism_distance(p: V3, half: V3, bevel: f64) -> f64 {
    let x = p[0].abs() - half[0];
    let y = p[1] + half[1];
    let z = p[2];
    let end = [-half[0], half[1] * 2.0];
    let t = ((x * end[0] + y * end[1]) / (end[0] * end[0] + end[1] * end[1])).clamp(0.0, 1.0);
    let segment = [x - end[0] * t, y - end[1] * t];
    let mut d1 = segment[0].hypot(segment[1]);
    if segment[0].max(segment[1]) < 0.0 {
        d1 = -d1.min(y);
    }
    d1 -= bevel;
    let d2 = z.abs() - half[2] - bevel;
    d1.max(0.0).hypot(d2.max(0.0)) + d1.max(d2).min(0.0)
}

/// The brush of a shape with the inputs its distance function needs, worked out once.
#[derive(Clone, Debug)]
enum Brush {
    Sphere { radii: V3 },
    Cube { half: V3, wall: f64, vertical_radius: f64, edge_radius: f64, taper: f64 },
    Cylinder { half: V3, rounding: f64 },
    Prism { half: V3, rounding: f64, inset: f64 },
    Polygon { half: V3, rounding: f64, inset: f64, points: f64, star: f64 },
}

impl Brush {
    fn from_node(node: &Node) -> Option<Brush> {
        let half = [node.scale[0] * 0.5, node.scale[1] * 0.5, node.scale[2] * 0.5];
        let smallest = half[0].min(half[1]).min(half[2]);
        let round = node.number("round").unwrap_or(0.0);
        let bevel = node.number("bevel").unwrap_or(0.0);
        let rounding = smallest * (0.01 + 0.99 * round);
        let inset = rounding.max(bevel * smallest);
        Some(match node.text("brush")? {
            "sphere" => Brush::Sphere { radii: half },
            "cube" => {
                let thickness = node.number("thickness").unwrap_or(1.0);
                let cone = node.number("cone").unwrap_or(0.0);
                let hxz = half[0].min(half[2]);
                let hollow = thickness < 0.999;
                let wall = 0.5 * hxz * (0.05 + 0.95 * thickness);
                let mut edge_radius = round * smallest;
                if hollow {
                    edge_radius = edge_radius.min(wall);
                }
                let vertical_radius = edge_radius.max(bevel * hxz);
                let taper = -cone * (hxz - edge_radius).max(0.0);
                let wall = if hollow { wall + taper.abs() } else { 1e3 };
                Brush::Cube { half, wall, vertical_radius, edge_radius, taper }
            }
            "cylinder" => Brush::Cylinder { half, rounding },
            "prism" => Brush::Prism { half, rounding, inset },
            "polygon" => Brush::Polygon {
                half,
                rounding,
                inset,
                points: node.number("points").unwrap_or(6.0).max(3.0),
                star: 1.0 - node.number("star").unwrap_or(0.0),
            },
            _ => return None,
        })
    }

    /// `(kind, half size, parameters)` as the GPU interpreter reads them. A solid cube's wall is a smaller number than
    /// the CPU's, which keeps single precision accurate.
    fn gpu(&self) -> (f32, V3, [f32; 4]) {
        match self {
            Brush::Sphere { radii } => (0.0, *radii, [0.0; 4]),
            Brush::Cube { half, wall, vertical_radius, edge_radius, taper } => {
                let wall = if *wall >= 1e2 { 4.0 * half[0].max(half[1]).max(half[2]).max(1.0) } else { *wall };
                (1.0, *half, [wall as f32, *vertical_radius as f32, *edge_radius as f32, *taper as f32])
            }
            Brush::Cylinder { half, rounding } => (2.0, *half, [*rounding as f32, 0.0, 0.0, 0.0]),
            Brush::Prism { half, rounding, inset } => (3.0, *half, [*rounding as f32, *inset as f32, 0.0, 0.0]),
            Brush::Polygon { half, rounding, inset, points, star } => (4.0, *half, [*rounding as f32, *inset as f32, *points as f32, *star as f32]),
        }
    }

    /// How far the shape reaches from its centre.
    fn extent(&self) -> f64 {
        let half = match self {
            Brush::Sphere { radii } => *radii,
            Brush::Cube { half, .. } | Brush::Cylinder { half, .. } | Brush::Prism { half, .. } | Brush::Polygon { half, .. } => *half,
        };
        (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()
    }

    fn distance(&self, p: V3) -> f64 {
        match self {
            Brush::Sphere { radii } => {
                let length = ((p[0] / radii[0]).powi(2) + (p[1] / radii[1]).powi(2) + (p[2] / radii[2]).powi(2)).sqrt();
                (length - 1.0) * radii[0].min(radii[1]).min(radii[2])
            }
            Brush::Cube { half, wall, vertical_radius, edge_radius, taper } => {
                uber_primitive([p[0], -p[1], p[2]], [half[0], half[1], half[2], *wall], [*vertical_radius, *edge_radius, *taper])
            }
            Brush::Cylinder { half, rounding } => {
                cylinder_distance(p, [half[0] - rounding, half[1] - rounding, half[2] - rounding]) - rounding
            }
            Brush::Prism { half, rounding, inset } => {
                prism_distance(p, [half[0] - inset, half[1] - inset, half[2] - inset], (inset - rounding).max(0.0)) - rounding
            }
            Brush::Polygon { half, rounding, inset, points, star } => {
                polygon_distance(p, [half[0] - inset, half[1] - rounding], *points, *star, (inset - rounding).max(0.0)) - rounding
            }
        }
    }
}

// ---- repeats and shapes -----------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Repeat {
    None,
    Linear { steps: V3, counts: V3 },
    Radial { radius: f64, count: f64, axis: usize },
}

impl Repeat {
    fn from_node(node: &Node) -> Repeat {
        match node.text("repeat_mode") {
            Some("radial") => {
                let count = node.number("radial_repeats").unwrap_or(1.0);
                if count <= 1.0 {
                    return Repeat::None;
                }
                let axis = match node.text("repeat_axis") {
                    Some("y") => 1,
                    Some("z") => 2,
                    _ => 0,
                };
                Repeat::Radial { radius: node.number("radius").unwrap_or(1.5), count, axis }
            }
            _ => {
                let counts = node.numbers::<3>("linear_repeats").unwrap_or([1.0; 3]);
                if counts.iter().all(|c| *c <= 1.0) {
                    return Repeat::None;
                }
                Repeat::Linear { steps: node.numbers::<3>("steps").unwrap_or([1.5; 3]), counts }
            }
        }
    }

    /// `(kind, a, b)` for the GPU interpreter: linear steps and counts, or radius, count and axis.
    fn gpu(&self) -> (f32, [f32; 3], [f32; 3]) {
        match self {
            Repeat::None => (0.0, [0.0; 3], [0.0; 3]),
            Repeat::Linear { steps, counts } => (1.0, steps.map(|v| v as f32), counts.map(|v| v as f32)),
            Repeat::Radial { radius, count, axis } => (2.0, [*radius as f32, *count as f32, *axis as f32], [0.0; 3]),
        }
    }

    /// How far the copies spread from the shape's own place.
    fn spread(&self) -> f64 {
        match self {
            Repeat::None => 0.0,
            Repeat::Linear { steps, counts } => (0..3).map(|i| ((counts[i] - 1.0) * steps[i]).abs()).fold(0.0, f64::max),
            Repeat::Radial { radius, .. } => radius.abs(),
        }
    }

    fn distance(&self, brush: &Brush, p: V3) -> f64 {
        match self {
            Repeat::None => brush.distance(p),
            Repeat::Linear { steps, counts } => {
                let mut cell = [0.0; 3];
                for axis in 0..3 {
                    let last = counts[axis] - 1.0;
                    let shifted = p[axis] + 0.5 * steps[axis] * last;
                    let id = (shifted / steps[axis]).round().clamp(0.0, last);
                    let local = shifted - steps[axis] * id;
                    cell[axis] = if (id as i64) & 1 == 0 { local } else { -local };
                }
                brush.distance(cell)
            }
            Repeat::Radial { radius, count, axis } => {
                let r = match axis {
                    0 => [p[2], p[1]],
                    1 => [p[2], p[0]],
                    _ => [p[1], p[0]],
                };
                let step = TAU / count;
                let id = (r[1].atan2(r[0]) / step).floor();
                [step * id, step * (id + 1.0)]
                    .iter()
                    .map(|angle| {
                        let (s, c) = angle.sin_cos();
                        let turned = [c * r[0] + s * r[1], -s * r[0] + c * r[1]];
                        let q = match axis {
                            0 => [p[0], turned[1], turned[0] - radius],
                            1 => [turned[1], p[1], turned[0] - radius],
                            _ => [turned[1], turned[0] - radius, p[2]],
                        };
                        brush.distance(q)
                    })
                    .fold(f64::INFINITY, f64::min)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Placement {
    mode: Mode,
    blend: f64,
    strength: f64,
    chamfer: bool,
}

impl Placement {
    fn from_node(node: &Node) -> Placement {
        Placement {
            mode: Mode::parse(node.text("mode")),
            blend: node.number("blend").unwrap_or(0.0),
            strength: node.number("blend_strength").unwrap_or(0.0),
            chamfer: node.flag("chamfer").unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug)]
struct Shape {
    placement: Placement,
    position: V3,
    rotation: M3,
    brush: Brush,
    repeat: Repeat,
    mirror: [bool; 3],
    mirror_blend: f64,
}

impl Shape {
    fn distance(&self, p: V3) -> f64 {
        let mut p = p;
        for axis in 0..3 {
            if self.mirror[axis] {
                p[axis] = smooth_abs(p[axis], self.mirror_blend);
            }
        }
        let local = into_frame(&self.rotation, sub(p, self.position));
        self.repeat.distance(&self.brush, local)
    }
}

#[derive(Clone, Debug)]
struct Group {
    placement: Placement,
    position: V3,
    rotation: M3,
    scale: f64,
    children: Vec<Item>,
}

#[derive(Clone, Debug)]
enum Item {
    Shape(Shape),
    Group(Group),
    /// A hidden or unsupported node: it keeps its place so indices match the file's tree.
    Skipped,
}

static SKIPPED_PLACEMENT: Placement = Placement { mode: Mode::None, blend: 0.0, strength: 0.0, chamfer: false };

impl Item {
    fn placement(&self) -> &Placement {
        match self {
            Item::Shape(shape) => &shape.placement,
            Item::Group(group) => &group.placement,
            Item::Skipped => &SKIPPED_PLACEMENT,
        }
    }

    fn distance(&self, p: V3) -> Option<f64> {
        match self {
            Item::Skipped => None,
            Item::Shape(shape) => Some(shape.distance(p)),
            Item::Group(group) => {
                let local = into_frame(&group.rotation, sub(p, group.position));
                let local = [local[0] / group.scale, local[1] / group.scale, local[2] / group.scale];
                accumulate(&group.children, local).map(|d| d * group.scale.abs())
            }
        }
    }
}

/// The siblings combined front to back; `None` when there is nothing to show.
fn accumulate(items: &[Item], p: V3) -> Option<f64> {
    let mut total: Option<f64> = None;
    for item in items {
        let Some(d) = item.distance(p) else { continue };
        let placement = item.placement();
        total = Some(match total {
            None => d,
            Some(so_far) => combine(placement.mode, d, so_far, placement.blend, placement.strength, placement.chamfer),
        });
    }
    total
}

fn compile_nodes(nodes: &[Node], has_unsupported: &mut bool) -> Vec<Item> {
    let mut items = Vec::new();
    for node in nodes {
        if node.hidden {
            items.push(Item::Skipped);
            continue;
        }
        match node.kind.as_str() {
            "shape" => {
                let Some(brush) = Brush::from_node(node) else {
                    *has_unsupported = true;
                    items.push(Item::Skipped);
                    continue;
                };
                items.push(Item::Shape(Shape {
                    placement: Placement::from_node(node),
                    position: node.position,
                    rotation: euler_matrix(node.rotation),
                    brush,
                    repeat: Repeat::from_node(node),
                    mirror: [
                        node.flag("mirror_x").unwrap_or(false),
                        node.flag("mirror_y").unwrap_or(false),
                        node.flag("mirror_z").unwrap_or(false),
                    ],
                    mirror_blend: node.number("mirror_blend").unwrap_or(0.0),
                }));
            }
            "group" => items.push(Item::Group(Group {
                placement: Placement::from_node(node),
                position: node.position,
                rotation: euler_matrix(node.rotation),
                scale: if node.scale[0] == 0.0 { 1.0 } else { node.scale[0] },
                children: compile_nodes(node.children(), has_unsupported),
            })),
            _ => {
                *has_unsupported = true;
                items.push(Item::Skipped);
            }
        }
    }
    items
}

// ---- layers -----------------------------------------------------------------------------------------------------

/// One layer of a file: its assembly's shape tree and the layer's placement of it.
#[derive(Clone, Debug)]
pub struct LayerField {
    pub name: String,
    pub hidden: bool,
    pub has_unsupported: bool,
    position: V3,
    rotation: M3,
    scale: V3,
    items: Vec<Item>,
}

impl LayerField {
    /// The signed distance at `p` in the file's own coordinates (the layer's position, rotation and scale applied);
    /// `None` for a layer with nothing to draw.
    pub fn distance(&self, p: V3) -> Option<f64> {
        let local = into_frame(&self.rotation, sub(p, self.position));
        let local = [local[0] / self.scale[0], local[1] / self.scale[1], local[2] / self.scale[2]];
        accumulate(&self.items, local).map(|d| d * self.scale[0].abs())
    }

    /// The distance of one node alone (its subtree, for a group) at `p` in the assembly's frame, group frames along
    /// `path` applied. `path` indexes the file's tree; `None` for a hidden or unsupported node.
    pub fn node_distance(&self, path: &[usize], p: V3) -> Option<f64> {
        let mut items = &self.items;
        let mut p = p;
        let mut scale = 1.0;
        for (depth, &index) in path.iter().enumerate() {
            let item = items.get(index)?;
            if depth + 1 == path.len() {
                return item.distance(p).map(|d| d * scale);
            }
            let Item::Group(group) = item else { return None };
            let local = into_frame(&group.rotation, sub(p, group.position));
            p = [local[0] / group.scale, local[1] / group.scale, local[2] / group.scale];
            scale *= group.scale.abs();
            items = &group.children;
        }
        None
    }

    /// The tree paths of every drawable shape.
    pub fn leaf_paths(&self) -> Vec<Vec<usize>> {
        fn walk(items: &[Item], prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            for (index, item) in items.iter().enumerate() {
                prefix.push(index);
                match item {
                    Item::Shape(_) => out.push(prefix.clone()),
                    Item::Group(group) => walk(&group.children, prefix, out),
                    Item::Skipped => {}
                }
                prefix.pop();
            }
        }
        let mut out = Vec::new();
        walk(&self.items, &mut Vec::new(), &mut out);
        out
    }

    /// The box around the layer's added shapes in the file's coordinates (the layer's placement applied).
    pub fn world_bounds(&self) -> Option<(V3, V3)> {
        let (lo, hi) = self.gpu_scene(None).bounds?;
        let mut out: Option<(V3, V3)> = None;
        for corner in 0..8 {
            let local = [0, 1, 2].map(|a| if corner >> a & 1 == 0 { lo[a] * self.scale[a] } else { hi[a] * self.scale[a] });
            let r = &self.rotation;
            let p = [
                self.position[0] + r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2],
                self.position[1] + r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2],
                self.position[2] + r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2],
            ];
            let (min, max) = out.get_or_insert((p, p));
            for a in 0..3 {
                min[a] = min[a].min(p[a]);
                max[a] = max[a].max(p[a]);
            }
        }
        out
    }

    /// The distance in the assembly's own frame, without the layer's placement.
    pub fn assembly_distance(&self, p: V3) -> Option<f64> {
        accumulate(&self.items, p)
    }

    /// An assembly on its own, without any layer's placement.
    pub fn for_assembly(file: &SdfFile, assembly: usize) -> Option<LayerField> {
        let assembly = file.assemblies.get(assembly)?;
        let mut has_unsupported = false;
        let items = compile_nodes(&assembly.shapes, &mut has_unsupported);
        Some(LayerField {
            name: String::new(),
            hidden: false,
            has_unsupported,
            position: [0.0; 3],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            scale: [1.0; 3],
            items,
        })
    }

    fn new(file: &SdfFile, layer: &Layer) -> Option<LayerField> {
        let assembly = file.assemblies.get(layer.assembly)?;
        let mut has_unsupported = false;
        let items = compile_nodes(&assembly.shapes, &mut has_unsupported);
        let scale = [layer.scale[0], layer.scale[1], layer.scale[2]].map(|s| if s == 0.0 { 1.0 } else { s });
        Some(LayerField {
            name: layer.name.clone(),
            hidden: layer.hidden,
            has_unsupported,
            position: layer.position,
            rotation: euler_matrix(layer.rotation),
            scale,
            items,
        })
    }
}

/// Every layer of a file as a distance field.
pub fn layer_fields(file: &SdfFile) -> Vec<LayerField> {
    file.layers.iter().filter_map(|layer| LayerField::new(file, layer)).collect()
}


// ---- the GPU interpreter's input -------------------------------------------------------------------------------

/// Vec4s per shape in `GpuScene::shapes`.
pub const GPU_SHAPE_STRIDE: usize = 12;

/// A shape tree flattened for the GPU: one record per shape and a postfix list of operations. The interpreter in
/// `composer_view.wgsl` reads exactly this layout.
#[derive(Clone, Debug, Default)]
pub struct GpuScene {
    pub shapes: Vec<[f32; 4]>,
    /// `[type, a, b, c]`: type 0 pushes shape `a`, 1 blends (mode a, width b, radius c), 2 blends with a chamfer,
    /// 3 multiplies the top by `a`.
    pub ops: Vec<[f32; 4]>,
    /// The box around the added shapes (spheres around each shape), when there are any.
    pub bounds: Option<(V3, V3)>,
    pub shape_count: usize,
}

/// Group frames from the world down: `p_local = rows · p + offset`.
#[derive(Clone, Copy)]
struct Down {
    m: M3,
    c: V3,
}

/// Group frames from the inside out: `p_world = t + r · (s · p_local)`.
#[derive(Clone, Copy)]
struct Up {
    t: V3,
    r: M3,
    s: f64,
}

fn apply_up(up: &Up, p: V3) -> V3 {
    let q = [p[0] * up.s, p[1] * up.s, p[2] * up.s];
    let r = [
        up.r[0][0] * q[0] + up.r[0][1] * q[1] + up.r[0][2] * q[2],
        up.r[1][0] * q[0] + up.r[1][1] * q[1] + up.r[1][2] * q[2],
        up.r[2][0] * q[0] + up.r[2][1] * q[1] + up.r[2][2] * q[2],
    ];
    [up.t[0] + r[0], up.t[1] + r[1], up.t[2] + r[2]]
}

fn transpose(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

fn row_vec4(row: V3, offset: f64) -> [f32; 4] {
    [row[0] as f32, row[1] as f32, row[2] as f32, offset as f32]
}

struct Flattener<'a> {
    scene: GpuScene,
    selected: Option<&'a [usize]>,
    added: Vec<(V3, f64)>,
    everything: Vec<(V3, f64)>,
}

impl Flattener<'_> {
    fn emit(&mut self, items: &[Item], down: Down, up: Up, path: &mut Vec<usize>, inside_selected: bool) -> bool {
        let mut emitted_any = false;
        for (index, item) in items.iter().enumerate() {
            path.push(index);
            let selected_here = inside_selected || self.selected.is_some_and(|s| s == path.as_slice());
            let before = self.scene.ops.len();
            let placement = match item {
                Item::Skipped => None,
                Item::Shape(shape) => {
                    self.emit_shape(shape, down, up, selected_here);
                    Some(&shape.placement)
                }
                Item::Group(group) => {
                    let inverse = transpose(&group.rotation);
                    let mut m = [[0.0; 3]; 3];
                    for (i, row) in m.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            *cell = inverse[i][j] / group.scale;
                        }
                    }
                    // The child's point is `M_g · (parent point − group position)`, and the parent's point is itself
                    // `down.m · world + down.c`.
                    let mut combined = [[0.0; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            combined[i][j] = (0..3).map(|k| m[i][k] * down.m[k][j]).sum();
                        }
                    }
                    let offset_in_parent = [down.c[0] - group.position[0], down.c[1] - group.position[1], down.c[2] - group.position[2]];
                    let c = [
                        m[0][0] * offset_in_parent[0] + m[0][1] * offset_in_parent[1] + m[0][2] * offset_in_parent[2],
                        m[1][0] * offset_in_parent[0] + m[1][1] * offset_in_parent[1] + m[1][2] * offset_in_parent[2],
                        m[2][0] * offset_in_parent[0] + m[2][1] * offset_in_parent[1] + m[2][2] * offset_in_parent[2],
                    ];
                    let inner_down = Down { m: combined, c };
                    let inner_up = Up { t: apply_up(&up, group.position), r: multiply(&up.r, &group.rotation), s: up.s * group.scale };
                    let any = self.emit(&group.children, inner_down, inner_up, path, selected_here);
                    if any && (group.scale.abs() - 1.0).abs() > 1e-9 {
                        self.scene.ops.push([3.0, group.scale.abs() as f32, 0.0, 0.0]);
                    }
                    any.then_some(&group.placement)
                }
            };
            if let Some(placement) = placement
                && self.scene.ops.len() > before
            {
                if emitted_any {
                    self.scene.ops.push([if placement.chamfer { 2.0 } else { 1.0 }, placement.mode.code(), placement.blend as f32, placement.strength as f32]);
                }
                emitted_any = true;
            }
            path.pop();
        }
        emitted_any
    }

    fn emit_shape(&mut self, shape: &Shape, down: Down, up: Up, selected: bool) {
        let index = self.scene.shape_count;
        self.scene.shape_count += 1;
        let (kind, half, b) = shape.brush.gpu();
        let (repeat, a_rep, b_rep) = shape.repeat.gpu();
        let rt = transpose(&shape.rotation);
        let offset = [
            -(rt[0][0] * shape.position[0] + rt[0][1] * shape.position[1] + rt[0][2] * shape.position[2]),
            -(rt[1][0] * shape.position[0] + rt[1][1] * shape.position[1] + rt[1][2] * shape.position[2]),
            -(rt[2][0] * shape.position[0] + rt[2][1] * shape.position[1] + rt[2][2] * shape.position[2]),
        ];
        let mirror_bits = (shape.mirror[0] as u32) | ((shape.mirror[1] as u32) << 1) | ((shape.mirror[2] as u32) << 2);
        let records = [
            row_vec4(down.m[0], down.c[0]),
            row_vec4(down.m[1], down.c[1]),
            row_vec4(down.m[2], down.c[2]),
            row_vec4(rt[0], offset[0]),
            row_vec4(rt[1], offset[1]),
            row_vec4(rt[2], offset[2]),
            [kind, mirror_bits as f32, shape.mirror_blend as f32, repeat],
            [half[0] as f32, half[1] as f32, half[2] as f32, 0.0],
            b,
            [a_rep[0], a_rep[1], a_rep[2], 0.0],
            [b_rep[0], b_rep[1], b_rep[2], 0.0],
            [if selected { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        ];
        self.scene.shapes.extend(records);
        self.scene.ops.push([0.0, index as f32, 0.0, 0.0]);

        // Bounds: a sphere around the shape (and its mirror image, and its copies).
        let reach = (shape.brush.extent() + shape.repeat.spread()) * up.s.abs() * 1.05;
        // Mirroring happens in the group's own frame.
        let mut locals = vec![shape.position];
        for axis in 0..3 {
            if shape.mirror[axis] {
                let flipped: Vec<V3> = locals
                    .iter()
                    .map(|l| {
                        let mut m = *l;
                        m[axis] = -m[axis];
                        m
                    })
                    .collect();
                locals.extend(flipped);
            }
        }
        for local in locals {
            let center = apply_up(&up, local);
            self.everything.push((center, reach));
            if shape.placement.mode == Mode::Add {
                self.added.push((center, reach));
            }
        }
    }
}

fn down_apply(down: &Down, p: V3) -> V3 {
    [
        down.m[0][0] * p[0] + down.m[0][1] * p[1] + down.m[0][2] * p[2] + down.c[0],
        down.m[1][0] * p[0] + down.m[1][1] * p[1] + down.m[1][2] * p[2] + down.c[1],
        down.m[2][0] * p[0] + down.m[2][1] * p[1] + down.m[2][2] * p[2] + down.c[2],
    ]
}

impl LayerField {
    /// The tree flattened for the GPU, with the shapes under `selected` marked. The layer's own placement is not
    /// applied: the scene is in the assembly's frame.
    pub fn gpu_scene(&self, selected: Option<&[usize]>) -> GpuScene {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut flattener = Flattener { scene: GpuScene::default(), selected, added: Vec::new(), everything: Vec::new() };
        flattener.emit(&self.items, Down { m: identity, c: [0.0; 3] }, Up { t: [0.0; 3], r: identity, s: 1.0 }, &mut Vec::new(), false);
        let spheres = if flattener.added.is_empty() { &flattener.everything } else { &flattener.added };
        let mut bounds: Option<(V3, V3)> = None;
        for (center, radius) in spheres {
            let (lo, hi) = bounds.get_or_insert(([f64::MAX; 3], [f64::MIN; 3]));
            for axis in 0..3 {
                lo[axis] = lo[axis].min(center[axis] - radius);
                hi[axis] = hi[axis].max(center[axis] + radius);
            }
        }
        flattener.scene.bounds = bounds;
        flattener.scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};

    fn node(brush: &str, extra: &str) -> String {
        format!(
            r#"{{"name":"","hidden":false,"type":"shape","position":[0,0,0],"rotation":[0,0,0],"scale":[2,2,2],"color":[1,1,1],
            "mode":"add","chamfer":false,"blend":0.0,"blend_strength":0.0,"mirror_x":false,"mirror_y":false,"mirror_z":false,
            "mirror_blend":0,"brush":"{brush}","bevel":0,"round":0,"thickness":1,"cone":0,"star":0,"points":6,
            "repeat_mode":"linear","linear_repeats":[1,1,1],"steps":[1.5,1.5,1.5],"radial_repeats":1,"radius":1.5,
            "repeat_axis":"x"{extra}}}"#
        )
    }

    fn field(shapes: &[String]) -> LayerField {
        let json = format!(
            r#"{{"version":[0,6,1],"environment":{{}},"assemblies":[{{"resolution":5,"shapes":[{}]}}],
            "layers":[{{"name":"L","assembly":0,"hidden":false,"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}}]}}"#,
            shapes.join(",")
        );
        layer_fields(&SdfFile::from_json(&json).unwrap()).remove(0)
    }

    #[test]
    fn a_sphere_is_an_ellipsoid_of_half_its_scale() {
        let sphere = field(&[node("sphere", "")]);
        assert!((sphere.distance([2.0, 0.0, 0.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!(sphere.distance([0.0, 0.0, 0.0]).unwrap() < 0.0);
        assert!(sphere.distance([1.0, 0.0, 0.0]).unwrap().abs() < 1e-12);
    }

    #[test]
    fn a_sharp_cube_reaches_half_its_scale() {
        let cube = field(&[node("cube", "")]);
        assert!(cube.distance([1.0, 0.0, 0.0]).unwrap().abs() < 1e-9);
        assert!(cube.distance([0.0, 1.0, 0.0]).unwrap().abs() < 1e-9);
        assert!((cube.distance([3.0, 0.0, 0.0]).unwrap() - 2.0).abs() < 1e-9);
        assert!(cube.distance([0.5, 0.5, 0.5]).unwrap() < 0.0);
    }

    #[test]
    fn subtracting_a_sphere_hollows_a_cube() {
        let cut = node("sphere", "").replace(r#""mode":"add""#, r#""mode":"sub""#).replace(r#""scale":[2,2,2]"#, r#""scale":[1,1,1]"#);
        let carved = field(&[node("cube", ""), cut]);
        assert!(carved.distance([0.0, 0.0, 0.0]).unwrap() > 0.0);
        assert!(carved.distance([0.9, 0.0, 0.0]).unwrap() < 0.0);
    }

    #[test]
    fn a_mirrored_shape_appears_on_both_sides() {
        let moved = node("sphere", "").replace(r#""position":[0,0,0]"#, r#""position":[3,0,0]"#).replace(r#""mirror_x":false"#, r#""mirror_x":true"#);
        let mirrored = field(&[moved]);
        assert!(mirrored.distance([3.0, 0.0, 0.0]).unwrap() < 0.0);
        assert!(mirrored.distance([-3.0, 0.0, 0.0]).unwrap() < 0.0);
    }

    #[test]
    fn linear_repeats_make_copies() {
        let repeated = node("sphere", "").replace(r#""scale":[2,2,2]"#, r#""scale":[1,1,1]"#).replace(r#""linear_repeats":[1,1,1]"#, r#""linear_repeats":[3,1,1]"#);
        let row = field(&[repeated]);
        for x in [-1.5, 0.0, 1.5] {
            assert!(row.distance([x, 0.0, 0.0]).unwrap() < 0.0, "copy at {x}");
        }
        assert!(row.distance([4.0, 0.0, 0.0]).unwrap() > 0.0);
    }

    #[test]
    fn splines_are_reported_and_left_out() {
        let spline = node("sphere", "").replace(r#""type":"shape""#, r#""type":"spline""#);
        let layer = field(&[spline, node("sphere", "")]);
        assert!(layer.has_unsupported);
        assert!(layer.distance([0.0, 0.0, 0.0]).unwrap() < 0.0);
    }

    // ---- comparison with meshes exported from SDF Modeler -----------------------------------------------------

    fn sample_folders() -> Vec<PathBuf> {
        let mut folders = vec![];
        if let Ok(dir) = std::env::var("NOISER_SDF_DIR") {
            folders.push(PathBuf::from(dir));
        }
        if let Ok(home) = std::env::var("USERPROFILE") {
            folders.push(Path::new(&home).join("Documents").join("SDF Modeler").join("sdfs"));
        }
        folders
    }

    /// About `count` vertices of an ASCII PLY, in the file's coordinates (SDF Modeler exports `(x, -z, y)`).
    fn read_vertices(path: &Path, count: usize) -> Vec<V3> {
        let mut lines = BufReader::new(std::fs::File::open(path).unwrap()).lines().map(|l| l.unwrap());
        let mut total = 0;
        for line in lines.by_ref() {
            if let Some(rest) = line.strip_prefix("element vertex ") {
                total = rest.trim().parse::<usize>().unwrap();
            }
            if line.starts_with("end_header") {
                break;
            }
        }
        let stride = (total / count).max(1);
        lines
            .take(total)
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, line)| {
                let v: Vec<f64> = line.split_whitespace().take(3).map(|n| n.parse().unwrap()).collect();
                [v[0], v[2], -v[1]]
            })
            .collect()
    }

    fn percentile(values: &mut [f64], fraction: f64) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[((values.len() - 1) as f64 * fraction) as usize]
    }

    /// Every layer of SDF Modeler's sample files (those without splines) agrees with its own export: the exported
    /// vertices lie on the zero set of the field. The exports are marching-cubes meshes, hence the tolerance.
    #[test]
    fn layers_match_the_meshes_sdf_modeler_exports() {
        let mut checked = 0;
        for folder in sample_folders() {
            for name in ["robot", "fried_egg", "icon", "bricks", "mech", "headphones", "mouse", "horse"] {
                let Ok(file) = SdfFile::load(&folder.join(format!("{name}.sdf"))) else { continue };
                for (index, layer) in layer_fields(&file).iter().enumerate() {
                    let ply = folder.join(if index == 0 { format!("{name}.ply") } else { format!("{name} - {index}.ply") });
                    if layer.has_unsupported || !ply.exists() {
                        continue;
                    }
                    let mut errors: Vec<f64> = read_vertices(&ply, 3000)
                        .into_iter()
                        .filter_map(|p| layer.distance(p))
                        .map(f64::abs)
                        .collect();
                    if errors.is_empty() {
                        continue;
                    }
                    let p95 = percentile(&mut errors, 0.95);
                    assert!(p95 < 0.012, "{name} layer {} {:?}: p95 {p95}", index + 1, layer.name);
                    checked += 1;
                }
            }
        }
        eprintln!("compared {checked} layer(s) with SDF Modeler's exports");
    }
}
