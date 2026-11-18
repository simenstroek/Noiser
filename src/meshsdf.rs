//! Signed distance to a triangle mesh. A BVH finds the closest point; the sign comes from the
//! generalized winding number (Jacobson et al. 2013, with the fast far-field approximation of Barill
//! et al. 2018), so it is right for meshes made of overlapping closed parts, inside-out parts and
//! meshes with small holes, which a plain closest-feature normal test gets wrong. The result is
//! baked into a grid that the preview and the export sample like a texture.

use crate::mesh::{Mesh, V3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
/// Which part of a triangle a closest point lies on, which decides the normal used for the sign.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Feature {
    Face,
    /// Edge i joins corners i and i + 1.
    Edge(usize),
    Vertex(usize),
}

/// The closest point to `p` on triangle `abc` (Ericson, Real-Time Collision Detection) and where it is.
fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> (V3, Feature) {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (add(a, scale(ab, d1 / (d1 - d3))), Feature::Edge(0));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (add(a, scale(ac, d2 / (d2 - d6))), Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(sub(c, b), w)), Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    (add(a, add(scale(ab, vb * denom), scale(ac, vc * denom))), Feature::Face)
}

struct Node {
    lo: V3,
    hi: V3,
    /// Leaf: the first slot in `order`. Inner node: the index of the right child (the left child
    /// is the next node).
    a: u32,
    /// Triangles in a leaf; 0 for an inner node.
    count: u32,
    /// The triangles under this node are `order[first..first + total]`.
    first: u32,
    total: u32,
    /// Sum of the triangles' area vectors, their area-weighted centre, and the distance from that
    /// centre to the farthest corner: the far-field approximation of the winding number.
    area: V3,
    centre: V3,
    radius: f32,
}

const LEAF_SIZE: usize = 4;

fn build_bvh(tri: &[[V3; 3]], order: &mut [u32], first: u32, nodes: &mut Vec<Node>) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let (mut clo, mut chi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for &t in order.iter() {
        let t = &tri[t as usize];
        let centroid = scale(add(t[0], add(t[1], t[2])), 1.0 / 3.0);
        for a in 0..3 {
            clo[a] = clo[a].min(centroid[a]);
            chi[a] = chi[a].max(centroid[a]);
            for v in t {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
    }
    let me = nodes.len();
    nodes.push(Node { lo, hi, a: 0, count: 0, first, total: order.len() as u32, area: [0.0; 3], centre: [0.0; 3], radius: 0.0 });
    let axis = (0..3).max_by(|&x, &y| (chi[x] - clo[x]).total_cmp(&(chi[y] - clo[y]))).unwrap();
    if order.len() <= LEAF_SIZE || chi[axis] <= clo[axis] {
        nodes[me].a = first;
        nodes[me].count = order.len() as u32;
        return;
    }
    let mid = order.len() / 2;
    let centre = |t: &u32| {
        let t = &tri[*t as usize];
        (t[0][axis] + t[1][axis] + t[2][axis]) / 3.0
    };
    order.select_nth_unstable_by(mid, |x, y| centre(x).total_cmp(&centre(y)));
    let (left, right) = order.split_at_mut(mid);
    build_bvh(tri, left, first, nodes);
    nodes[me].a = nodes.len() as u32;
    build_bvh(tri, right, first + mid as u32, nodes);
}

/// Fills in each node's winding-number data from the triangles under it.
fn fill_far_field(nodes: &mut [Node], tri: &[[V3; 3]], order: &[u32]) {
    for node in nodes.iter_mut() {
        let slots = &order[node.first as usize..(node.first + node.total) as usize];
        let (mut area, mut weighted, mut total) = ([0.0f32; 3], [0.0f32; 3], 0.0f32);
        for &t in slots {
            let [a, b, c] = tri[t as usize];
            let v = scale(cross(sub(b, a), sub(c, a)), 0.5);
            let m = length(v);
            area = add(area, v);
            weighted = add(weighted, scale(scale(add(a, add(b, c)), 1.0 / 3.0), m));
            total += m;
        }
        node.area = area;
        node.centre = if total > 0.0 { scale(weighted, 1.0 / total) } else { scale(add(node.lo, node.hi), 0.5) };
        node.radius = slots.iter().flat_map(|&t| tri[t as usize]).map(|v| length(sub(v, node.centre))).fold(0.0, f32::max);
    }
}

/// Squared distance from `p` to a box.
fn box_distance2(lo: V3, hi: V3, p: V3) -> f32 {
    let mut d2 = 0.0;
    for a in 0..3 {
        let d = (lo[a] - p[a]).max(0.0).max(p[a] - hi[a]);
        d2 += d * d;
    }
    d2
}

/// The signed solid angle a triangle subtends at the origin (Van Oosterom and Strackee): positive
/// when the origin is inside a surface wound counter-clockwise from outside.
fn solid_angle(a: V3, b: V3, c: V3) -> f32 {
    let (la, lb, lc) = (length(a), length(b), length(c));
    let numerator = dot(a, cross(b, c));
    let denominator = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    2.0 * numerator.atan2(denominator)
}

/// How far from a cluster of triangles (in cluster radii) the winding number uses its dipole.
const FAR: f32 = 3.0;

/// Closest-point and inside/outside queries against a mesh.
pub struct MeshQuery {
    tri: Vec<[V3; 3]>,
    nodes: Vec<Node>,
    order: Vec<u32>,
    /// A surface that does not enclose anything (a terrain, a leaf, a sheet): "inside" is then the
    /// side opposite the surface normal, since the winding number has no clean inside.
    open: bool,
    /// Smooth (angle-weighted) vertex normals of each triangle's corners, for the open sign.
    corner_normal: Vec<[V3; 3]>,
    /// The mesh's vertex indices per triangle and its vertex colours (empty without colours), for `color_at`.
    ids: Vec<[u32; 3]>,
    colors: Vec<V3>,
}

/// Angle-weighted vertex normals, per triangle corner.
fn corner_normals(mesh: &Mesh) -> Vec<[V3; 3]> {
    let mut sum = vec![[0.0f32; 3]; mesh.positions.len()];
    for t in &mesh.triangles {
        let p = [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]];
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let nl = length(n);
        if nl < 1e-20 {
            continue;
        }
        let n = scale(n, 1.0 / nl);
        for c in 0..3 {
            let (e1, e2) = (sub(p[(c + 1) % 3], p[c]), sub(p[(c + 2) % 3], p[c]));
            let (l1, l2) = (length(e1), length(e2));
            if l1 < 1e-20 || l2 < 1e-20 {
                continue;
            }
            let angle = (dot(e1, e2) / (l1 * l2)).clamp(-1.0, 1.0).acos();
            sum[t[c] as usize] = add(sum[t[c] as usize], scale(n, angle));
        }
    }
    mesh.triangles.iter().map(|t| std::array::from_fn(|c| sum[t[c] as usize])).collect()
}

/// Barycentric coordinates of `p` (on the triangle's plane) in triangle `abc`.
fn barycentric(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (v0, v1, v2) = (sub(b, a), sub(c, a), sub(p, a));
    let (d00, d01, d11, d20, d21) = (dot(v0, v0), dot(v0, v1), dot(v1, v1), dot(v2, v0), dot(v2, v1));
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-30 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    [1.0 - v - w, v, w]
}

impl MeshQuery {
    pub fn new(mesh: &Mesh) -> MeshQuery {
        let tri: Vec<[V3; 3]> = mesh.triangles.iter().map(|t| [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]]).collect();
        let mut order: Vec<u32> = (0..tri.len() as u32).collect();
        let mut nodes = Vec::with_capacity(tri.len() / 2 + 1);
        if !tri.is_empty() {
            build_bvh(&tri, &mut order, 0, &mut nodes);
            fill_far_field(&mut nodes, &tri, &order);
        }
        let mut q = MeshQuery { tri, nodes, order, open: false, corner_normal: Vec::new(), ids: mesh.triangles.clone(), colors: mesh.colors.clone() };
        q.open = q.looks_open(mesh);
        if q.open {
            q.corner_normal = corner_normals(mesh);
        }
        q
    }

    /// Whether the surface is an open sheet (a terrain, a leaf) rather than an enclosed solid.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Whether the surface is open: far from the mesh the winding number of a closed surface is a
    /// whole number (0, 1, 2...), while a sheet gives about one half on either side.
    fn looks_open(&self, mesh: &Mesh) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let (lo, hi) = mesh.bounds();
        let mut fractional = 0;
        let n = 7;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let t = |a: usize, c: usize| lo[a] + (hi[a] - lo[a]) * (0.5 + (c as f32 + 0.5 - n as f32 * 0.5) / n as f32 * 1.2);
                    let w = self.winding_number([t(0, i), t(1, j), t(2, k)]).abs();
                    if (w - w.round()).abs() > 0.25 {
                        fractional += 1;
                    }
                }
            }
        }
        fractional * 10 > n * n * n * 4
    }

    /// Squared distance to the nearest triangle, that triangle (an index into `tri`) and the closest point on it.
    fn closest(&self, p: V3) -> (f32, usize, V3) {
        let mut best = f32::INFINITY;
        let (mut best_tri, mut best_point) = (0usize, p);
        if self.nodes.is_empty() {
            return (best, best_tri, best_point);
        }
        let mut stack = [0u32; 96];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if box_distance2(node.lo, node.hi, p) >= best {
                continue;
            }
            if node.count > 0 {
                for slot in node.a..node.a + node.count {
                    let ti = self.order[slot as usize] as usize;
                    let [a, b, c] = self.tri[ti];
                    let cp = closest_on_triangle(p, a, b, c).0;
                    let d = sub(p, cp);
                    let d2 = dot(d, d);
                    if d2 < best {
                        (best, best_tri, best_point) = (d2, ti, cp);
                    }
                }
            } else {
                let (left, right) = (stack[top] + 1, node.a);
                let dl = box_distance2(self.nodes[left as usize].lo, self.nodes[left as usize].hi, p);
                let dr = box_distance2(self.nodes[right as usize].lo, self.nodes[right as usize].hi, p);
                // Visit the nearer child first: push it last.
                let (far, near) = if dl <= dr { (right, left) } else { (left, right) };
                stack[top] = far;
                stack[top + 1] = near;
                top += 2;
            }
        }
        (best, best_tri, best_point)
    }

    /// The generalized winding number at `p`: about 1 inside a closed surface wound outward, 0
    /// outside, 2 inside two overlapping ones, and fractional near holes. Far clusters of triangles
    /// are summed as one dipole.
    pub fn winding_number(&self, p: V3) -> f32 {
        let mut total = 0.0;
        if self.nodes.is_empty() {
            return total;
        }
        let mut stack = [0u32; 96];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let index = stack[top];
            let node = &self.nodes[index as usize];
            if node.count > 0 {
                for slot in node.a..node.a + node.count {
                    let [a, b, c] = self.tri[self.order[slot as usize] as usize];
                    total += solid_angle(sub(a, p), sub(b, p), sub(c, p));
                }
                continue;
            }
            let r = sub(node.centre, p);
            let r2 = dot(r, r);
            if r2 > FAR * FAR * node.radius * node.radius {
                total += dot(node.area, r) / (r2 * r2.sqrt());
            } else {
                // The left child is the next node, the right child is `a`.
                stack[top] = index + 1;
                stack[top + 1] = node.a;
                top += 2;
            }
        }
        total / (4.0 * std::f32::consts::PI)
    }

    /// Distance to the surface, negative inside.
    #[cfg(test)]
    pub fn signed_distance(&self, p: V3) -> f32 {
        self.signed_distance_and_color(p).0
    }

    /// The vertex colour interpolated at the surface point nearest to `p`; black for a mesh without colours.
    #[cfg(test)]
    pub fn color_at(&self, p: V3) -> V3 {
        self.signed_distance_and_color(p).1
    }

    /// Both at once, from a single closest-point search.
    pub fn signed_distance_and_color(&self, p: V3) -> (f32, V3) {
        let (d2, ti, cp) = self.closest(p);
        if !d2.is_finite() {
            return (f32::INFINITY, [0.0; 3]);
        }
        let color = if self.colors.is_empty() {
            [0.0; 3]
        } else {
            let [a, b, c] = self.tri[ti];
            let w = barycentric(cp, a, b, c);
            let ids = self.ids[ti];
            std::array::from_fn(|k| w[0] * self.colors[ids[0] as usize][k] + w[1] * self.colors[ids[1] as usize][k] + w[2] * self.colors[ids[2] as usize][k])
        };
        (self.sign_of(p, d2, ti, cp), color)
    }

    fn sign_of(&self, p: V3, d2: f32, ti: usize, cp: V3) -> f32 {
        if self.open {
            // Which side of the surface: the smooth normal at the closest point against the offset to it.
            let [a, b, c] = self.tri[ti];
            let w = barycentric(cp, a, b, c);
            let n = self.corner_normal[ti];
            let normal = add(add(scale(n[0], w[0]), scale(n[1], w[1])), scale(n[2], w[2]));
            return if dot(sub(p, cp), normal) < 0.0 { -d2.sqrt() } else { d2.sqrt() };
        }
        // Either orientation of a closed part counts as inside; overlapping parts add up.
        if self.winding_number(p).abs() > 0.5 { -d2.sqrt() } else { d2.sqrt() }
    }
}

/// How the separate parts of a mesh (pieces that share no vertices) are put together.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CombineMode {
    /// Solid where any part is (a single winding-number query over the whole mesh).
    #[default]
    Union,
    /// Solid only where every part is: overlaps become the shape, hollows stay hollow.
    Intersect,
    SmoothUnion,
    SmoothIntersect,
}

impl CombineMode {
    pub const ALL: [CombineMode; 4] = [CombineMode::Union, CombineMode::Intersect, CombineMode::SmoothUnion, CombineMode::SmoothIntersect];

    pub fn label(self) -> &'static str {
        match self {
            CombineMode::Union => "Union (min)",
            CombineMode::Intersect => "Intersect (max)",
            CombineMode::SmoothUnion => "Smooth union (smooth min)",
            CombineMode::SmoothIntersect => "Smooth intersect (smooth max)",
        }
    }

    pub fn is_smooth(self) -> bool {
        matches!(self, CombineMode::SmoothUnion | CombineMode::SmoothIntersect)
    }

    /// The name used on the command line.
    pub fn file_name(self) -> &'static str {
        match self {
            CombineMode::Union => "union",
            CombineMode::Intersect => "intersect",
            CombineMode::SmoothUnion => "smooth-union",
            CombineMode::SmoothIntersect => "smooth-intersect",
        }
    }
}

/// The part-combining settings: the mode and, for the smooth ones, the blend width in normalized mesh units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Combine {
    pub mode: CombineMode,
    pub smooth: f32,
}

impl Default for Combine {
    fn default() -> Self {
        Combine { mode: CombineMode::Union, smooth: 0.1 }
    }
}

impl Combine {
    fn pair(self, a: f32, b: f32) -> f32 {
        let smin = |a: f32, b: f32| {
            let k = self.smooth;
            if k <= 0.0 {
                return a.min(b);
            }
            let h = (k - (a - b).abs()).max(0.0) / k;
            a.min(b) - h * h * h * k / 6.0
        };
        match self.mode {
            CombineMode::Union => a.min(b),
            CombineMode::Intersect => a.max(b),
            CombineMode::SmoothUnion => smin(a, b),
            CombineMode::SmoothIntersect => -smin(-a, -b),
        }
    }
}

/// Splits a mesh into its connected parts (triangles linked by shared vertices).
pub fn split_parts(mesh: &Mesh) -> Vec<Mesh> {
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut parent: Vec<u32> = (0..mesh.positions.len() as u32).collect();
    for t in &mesh.triangles {
        let a = find(&mut parent, t[0]);
        let b = find(&mut parent, t[1]);
        parent[b as usize] = a;
        let c = find(&mut parent, t[2]);
        parent[c as usize] = a;
    }
    let mut index_of_root = std::collections::HashMap::new();
    // Each part with its vertex remap (u32::MAX = not used yet).
    let mut parts: Vec<(Mesh, Vec<u32>)> = Vec::new();
    for t in &mesh.triangles {
        let root = find(&mut parent, t[0]);
        let next = parts.len();
        let pi = *index_of_root.entry(root).or_insert(next);
        if pi == next {
            parts.push((Mesh::default(), vec![u32::MAX; mesh.positions.len()]));
        }
        let (part, remap) = &mut parts[pi];
        let tri = t.map(|v| {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = part.positions.len() as u32;
                part.positions.push(mesh.positions[v as usize]);
                if !mesh.colors.is_empty() {
                    part.colors.push(mesh.colors[v as usize]);
                }
            }
            remap[v as usize]
        });
        part.triangles.push(tri);
    }
    parts.into_iter().map(|(m, _)| m).collect()
}

/// One query for the whole mesh, or one per part when parts are combined with something other than a union.
enum Sampler {
    Whole(MeshQuery),
    Parts(Vec<MeshQuery>, Combine),
}

impl Sampler {
    fn new(mesh: &Mesh, combine: Combine) -> Sampler {
        if combine.mode != CombineMode::Union {
            let parts = split_parts(mesh);
            if parts.len() > 1 {
                return Sampler::Parts(parts.iter().map(MeshQuery::new).collect(), combine);
            }
        }
        Sampler::Whole(MeshQuery::new(mesh))
    }

    fn open(&self) -> bool {
        match self {
            Sampler::Whole(q) => q.open,
            Sampler::Parts(qs, _) => qs.iter().any(|q| q.open),
        }
    }

    /// The combined distance, and the vertex colour of the part whose surface is nearest.
    fn at(&self, p: V3) -> (f32, V3) {
        match self {
            Sampler::Whole(q) => q.signed_distance_and_color(p),
            Sampler::Parts(qs, combine) => {
                let mut acc: Option<(f32, V3, f32)> = None;
                for q in qs {
                    let (d, c) = q.signed_distance_and_color(p);
                    acc = Some(match acc {
                        None => (d, c, d.abs()),
                        Some((ad, ac, near)) => {
                            let (c, near) = if d.abs() < near { (c, d.abs()) } else { (ac, near) };
                            (combine.pair(ad, d), c, near)
                        }
                    });
                }
                acc.map_or((f32::INFINITY, [0.0; 3]), |(d, c, _)| (d, c))
            }
        }
    }
}

/// A signed distance grid. Texel (i, j, k) has its centre at `origin + (i + 0.5, j + 0.5, k + 0.5) × voxel`
/// (the layout of a 3D texture), x varying fastest.
pub struct MeshSdf {
    pub dims: [u32; 3],
    pub origin: V3,
    pub voxel: f32,
    pub data: Vec<f32>,
    /// Vertex colours on the same grid (None when the mesh has none).
    pub colors: Option<Vec<[f32; 3]>>,
    /// The mesh is an open surface (terrain, sheet): inside is the side opposite its normals.
    pub open: bool,
}

impl MeshSdf {
    /// Bakes the distance to `mesh` (already normalized) on a grid with `res` voxels along the
    /// longest side of its bounds, padded by `pad` on every side. `progress` counts finished slabs.
    #[cfg(test)]
    pub fn build(mesh: &Mesh, res: u32, pad: f32, progress: &AtomicU32) -> MeshSdf {
        Self::build_with(mesh, res, pad, progress, Combine::default())
    }

    /// Like `build`, with the parts of the mesh combined as `combine` says.
    pub fn build_with(mesh: &Mesh, res: u32, pad: f32, progress: &AtomicU32, combine: Combine) -> MeshSdf {
        let (lo, hi) = mesh.bounds();
        let origin = [lo[0] - pad, lo[1] - pad, lo[2] - pad];
        let extent = [hi[0] - lo[0] + 2.0 * pad, hi[1] - lo[1] + 2.0 * pad, hi[2] - lo[2] + 2.0 * pad];
        let voxel = extent[0].max(extent[1]).max(extent[2]) / res.max(2) as f32;
        let dims: [u32; 3] = std::array::from_fn(|a| ((extent[a] / voxel).ceil() as u32).max(1));
        let query = Sampler::new(mesh, combine);
        let (nx, ny) = (dims[0] as usize, dims[1] as usize);
        let mut data = vec![0.0f32; nx * ny * dims[2] as usize];
        let has_colors = !mesh.colors.is_empty();
        let mut colors = if has_colors { vec![[0.0f32; 3]; data.len()] } else { Vec::new() };
        let bake = |k: usize, slab: &mut [f32], color_slab: Option<&mut [[f32; 3]]>| {
            let mut color_slab = color_slab;
            for j in 0..ny {
                for i in 0..nx {
                    let p = [origin[0] + (i as f32 + 0.5) * voxel, origin[1] + (j as f32 + 0.5) * voxel, origin[2] + (k as f32 + 0.5) * voxel];
                    let (d, c) = query.at(p);
                    slab[j * nx + i] = d;
                    if let Some(cs) = color_slab.as_deref_mut() {
                        cs[j * nx + i] = c;
                    }
                }
            }
            progress.fetch_add(1, Ordering::Relaxed);
        };
        if has_colors {
            data.par_chunks_mut(nx * ny).zip(colors.par_chunks_mut(nx * ny)).enumerate().for_each(|(k, (slab, cs))| bake(k, slab, Some(cs)));
        } else {
            data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| bake(k, slab, None));
        }
        MeshSdf { dims, origin, voxel, data, colors: has_colors.then_some(colors), open: query.open() }
    }

    /// Puts the grid on the GPU as an R32Float 3D texture.
    pub fn upload(&self, device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue) -> (eframe::wgpu::Texture, eframe::wgpu::TextureView) {
        use eframe::wgpu;
        let [nx, ny, nz] = self.dims;
        let size = wgpu::Extent3d { width: nx, height: ny, depth_or_array_layers: nz };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh distance"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            // With vertex colours the texture is RGBA: distance, then red, green and blue.
            format: if self.colors.is_some() { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::R32Float },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texels: Vec<f32> = match &self.colors {
            Some(c) => self.data.iter().zip(c).flat_map(|(d, c)| [*d, c[0], c[1], c[2]]).collect(),
            None => Vec::new(),
        };
        let (bytes, texel_size): (&[f32], u32) = if self.colors.is_some() { (&texels, 16) } else { (&self.data, 4) };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(bytes),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(nx * texel_size), rows_per_image: Some(ny) },
            size,
        );
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    #[cfg(test)]
    /// Distance at `p`: trilinear between texel centres, and beyond the grid the nearest texel plus
    /// the distance to it. The shaders do the same, so a CPU check matches the GPU.
    pub fn sample(&self, p: V3) -> f32 {
        let dims = self.dims.map(|d| d as f32);
        let g: V3 = std::array::from_fn(|a| (p[a] - self.origin[a]) / self.voxel - 0.5);
        let gc: V3 = std::array::from_fn(|a| g[a].clamp(0.0, dims[a] - 1.0));
        let i0: [usize; 3] = std::array::from_fn(|a| gc[a].floor() as usize);
        let f: V3 = std::array::from_fn(|a| gc[a] - i0[a] as f32);
        let i1: [usize; 3] = std::array::from_fn(|a| (i0[a] + 1).min(self.dims[a] as usize - 1));
        let at = |x: usize, y: usize, z: usize| self.data[(z * self.dims[1] as usize + y) * self.dims[0] as usize + x];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x = |y: usize, z: usize| lerp(at(i0[0], y, z), at(i1[0], y, z), f[0]);
        let y = |z: usize| lerp(x(i0[1], z), x(i1[1], z), f[1]);
        let v = lerp(y(i0[2]), y(i1[2]), f[2]);
        let beyond = length(std::array::from_fn(|a| g[a] - gc[a])) * self.voxel;
        v + beyond
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::shapes;

    fn cheap_points(n: usize, lo: f32, hi: f32) -> Vec<V3> {
        let mut s = 12345u32;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            lo + (hi - lo) * ((s >> 8) as f32 / (1u32 << 24) as f32)
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn closest_point_regions_of_a_triangle() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(closest_on_triangle([-1.0, -1.0, 0.0], a, b, c), (a, Feature::Vertex(0)));
        assert_eq!(closest_on_triangle([2.0, -1.0, 0.0], a, b, c), (b, Feature::Vertex(1)));
        assert_eq!(closest_on_triangle([-1.0, 2.0, 0.0], a, b, c), (c, Feature::Vertex(2)));
        assert_eq!(closest_on_triangle([0.5, -1.0, 3.0], a, b, c), ([0.5, 0.0, 0.0], Feature::Edge(0)));
        assert_eq!(closest_on_triangle([1.0, 1.0, 0.0], a, b, c), ([0.5, 0.5, 0.0], Feature::Edge(1)));
        assert_eq!(closest_on_triangle([-1.0, 0.5, 0.0], a, b, c), ([0.0, 0.5, 0.0], Feature::Edge(2)));
        assert_eq!(closest_on_triangle([0.25, 0.25, 2.0], a, b, c), ([0.25, 0.25, 0.0], Feature::Face));
    }

    #[test]
    fn sphere_mesh_distance_matches_the_analytic_one() {
        let q = MeshQuery::new(&shapes::icosphere(1.0, 3));
        let worst = cheap_points(2000, -2.5, 2.5)
            .iter()
            .map(|p| {
                let exact = length(*p) - 1.0;
                (q.signed_distance(*p) - exact).abs()
            })
            .fold(0.0, f32::max);
        // A 1280-triangle sphere is a few thousandths inside the true one.
        assert!(worst < 0.02, "worst error {worst}");
    }

    #[test]
    fn sign_is_right_at_edges_and_corners_of_a_cube() {
        let cube = shapes::split_cube().welded();
        let q = MeshQuery::new(&cube);
        // Closest feature is a vertex, an edge or a face, outside and inside.
        for (p, want) in [
            ([1.2, 1.2, 1.2], 0.2 * 3.0f32.sqrt()),
            ([1.3, 1.3, 0.0], 0.3 * 2.0f32.sqrt()),
            ([1.5, 0.2, -0.3], 0.5),
            ([0.9, 0.9, 0.9], -0.1),
            ([0.0, 0.0, 0.0], -1.0),
            ([0.0, 0.95, 0.95], -0.05),
        ] {
            let d = q.signed_distance(p);
            assert!((d - want).abs() < 1e-4, "{p:?}: got {d}, want {want}");
        }
        // Points around the corner never get the wrong sign.
        for p in cheap_points(2000, -1.6, 1.6) {
            let inside = p.iter().all(|c| c.abs() < 1.0);
            let d = q.signed_distance(p);
            assert_eq!(d < 0.0, inside, "{p:?}: distance {d}");
        }
    }

    fn moved(mut m: Mesh, by: V3) -> Mesh {
        for p in &mut m.positions {
            *p = add(*p, by);
        }
        m
    }

    fn merged(a: Mesh, b: Mesh) -> Mesh {
        let base = a.positions.len() as u32;
        let mut m = a;
        m.positions.extend(b.positions);
        m.triangles.extend(b.triangles.iter().map(|t| t.map(|i| i + base)));
        m
    }

    #[test]
    fn winding_number_is_one_inside_zero_outside_and_the_far_field_matches_the_exact_sum() {
        let mesh = shapes::icosphere(1.0, 3);
        let q = MeshQuery::new(&mesh);
        assert!((q.winding_number([0.1, 0.2, -0.1]) - 1.0).abs() < 0.02);
        assert!(q.winding_number([1.6, 0.4, 0.2]).abs() < 0.02);
        for p in cheap_points(400, -2.5, 2.5) {
            let exact: f32 = q.tri.iter().map(|t| solid_angle(sub(t[0], p), sub(t[1], p), sub(t[2], p))).sum::<f32>() / (4.0 * std::f32::consts::PI);
            let fast = q.winding_number(p);
            assert!((fast - exact).abs() < 0.03, "{p:?}: fast {fast}, exact {exact}");
        }
    }

    #[test]
    fn overlapping_and_nested_parts_are_inside_where_they_should_be() {
        // A small ball entirely inside a big one: its surface is internal, so a closest-feature sign
        // test calls the points next to it "outside". The winding number says they are inside.
        let nested = merged(shapes::icosphere(1.0, 3), moved(shapes::icosphere(0.3, 2), [0.5, 0.0, 0.0]));
        let q = MeshQuery::new(&nested);
        assert!(q.signed_distance([0.85, 0.0, 0.0]) < 0.0, "next to the inner ball, inside the big one");
        assert!(q.signed_distance([0.0, 0.0, 0.0]) < 0.0);
        assert!(q.signed_distance([1.4, 0.0, 0.0]) > 0.0);
        // Two overlapping balls: every point not close to a surface is classified like the exact union.
        let (ca, cb) = ([-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]);
        let two = merged(moved(shapes::icosphere(1.0, 3), ca), moved(shapes::icosphere(1.0, 3), cb));
        let q = MeshQuery::new(&two);
        let mut checked = 0;
        for p in cheap_points(3000, -2.0, 2.0) {
            let (da, db) = (length(sub(p, ca)) - 1.0, length(sub(p, cb)) - 1.0);
            if da.abs() < 0.05 || db.abs() < 0.05 {
                continue;
            }
            checked += 1;
            assert_eq!(q.signed_distance(p) < 0.0, da < 0.0 || db < 0.0, "{p:?}: balls say {da} and {db}");
        }
        assert!(checked > 2000);
    }

    #[test]
    fn parts_combine_as_union_intersection_and_smooth_blends() {
        let (ca, cb) = ([-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]);
        let two = merged(moved(shapes::icosphere(1.0, 3), ca), moved(shapes::icosphere(1.0, 3), cb));
        assert_eq!(split_parts(&two).len(), 2);
        let sdf = |mode, smooth| MeshSdf::build_with(&two, 48, 0.5, &AtomicU32::new(0), Combine { mode, smooth });
        let (union, inter) = (sdf(CombineMode::Union, 0.1), sdf(CombineMode::Intersect, 0.1));
        let (smin, smax) = (sdf(CombineMode::SmoothUnion, 0.5), sdf(CombineMode::SmoothIntersect, 0.5));
        let only_a = [-1.3, 0.0, 0.0]; // inside one ball only
        let both = [0.0, 0.0, 0.0];
        assert!(union.sample(only_a) < 0.0 && inter.sample(only_a) > 0.0, "a point in one ball: in the union, not the intersection");
        assert!(union.sample(both) < 0.0 && inter.sample(both) < 0.0);
        // The smooth blends sit within k / 6 of the hard ones (below for min, above for max).
        for p in cheap_points(200, -1.5, 1.5) {
            let (da, db) = (length(sub(p, ca)) - 1.0, length(sub(p, cb)) - 1.0);
            let (u, i) = (da.min(db), da.max(db));
            assert!((inter.sample(p) - i).abs() < 0.05, "intersection at {p:?}");
            assert!(smin.sample(p) <= u + 0.05 && smin.sample(p) >= u - 0.5 / 6.0 - 0.05, "smooth min at {p:?}");
            assert!(smax.sample(p) >= i - 0.05 && smax.sample(p) <= i + 0.5 / 6.0 + 0.05, "smooth max at {p:?}");
        }
    }

    #[test]
    fn inside_out_and_holey_meshes_still_have_an_inside() {
        let mut flipped = shapes::split_cube().welded();
        for t in &mut flipped.triangles {
            t.swap(1, 2);
        }
        let q = MeshQuery::new(&flipped);
        assert!(q.signed_distance([0.0; 3]) < 0.0, "an inside-out cube is still a solid");
        assert!(q.signed_distance([1.5, 0.0, 0.0]) > 0.0);
        // A cube missing one face (two triangles): still mostly enclosed, so the middle is inside.
        let mut open = shapes::split_cube().welded();
        open.triangles.truncate(open.triangles.len() - 2);
        let q = MeshQuery::new(&open);
        assert!(q.signed_distance([0.0; 3]) < 0.0, "a cube with a missing face");
        assert!(q.signed_distance([0.0, 0.0, 3.0]) > 0.0);
    }

    /// A bumpy 16 x 16 sheet seen from above, like a terrain.
    fn sheet() -> Mesh {
        let n = 16;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, z) = (i as f32 / n as f32 * 2.0 - 1.0, j as f32 / n as f32 * 2.0 - 1.0);
                positions.push([x, 0.1 * (3.0 * x).sin() * (2.0 * z).cos(), z]);
            }
        }
        let mut triangles = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = (j * (n + 1) + i) as u32;
                let (b, c, d) = (a + 1, a + n as u32 + 1, a + n as u32 + 2);
                triangles.push([a, c, b]); // wound so the normal points up (+y)
                triangles.push([b, c, d]);
            }
        }
        Mesh { positions, triangles, colors: Vec::new() }
    }

    #[test]
    fn an_open_sheet_is_solid_below_and_empty_above() {
        let q = MeshQuery::new(&sheet());
        assert!(q.open, "a sheet is detected as open");
        for (x, z) in [(0.0, 0.0), (0.5, -0.4), (-0.7, 0.6), (0.9, 0.9)] {
            assert!(q.signed_distance([x, 0.5, z]) > 0.3, "above the sheet at {x},{z}");
            assert!(q.signed_distance([x, -0.5, z]) < -0.3, "below the sheet at {x},{z}");
        }
        // Closed solids are not mistaken for sheets.
        assert!(!MeshQuery::new(&shapes::icosphere(1.0, 2)).open);
        assert!(!MeshQuery::new(&shapes::split_cube().welded()).open);
        let mut hole = shapes::split_cube().welded();
        hole.triangles.truncate(hole.triangles.len() - 2);
        assert!(!MeshQuery::new(&hole).open, "a cube missing a face is still a solid");
    }

    #[test]
    fn vertex_colours_are_interpolated_at_the_nearest_surface_point() {
        let mut sphere = shapes::icosphere(1.0, 3);
        sphere.colors = sphere.positions.iter().map(|p| [p[0] * 0.5 + 0.5, 0.0, 1.0]).collect();
        let q = MeshQuery::new(&sphere);
        for p in [[0.9, 0.0, 0.0], [1.4, 0.2, 0.1], [-0.7, 0.3, 0.0]] {
            let n = length(p);
            let on_surface = scale(p, 1.0 / n);
            let want = on_surface[0] * 0.5 + 0.5;
            let c = q.color_at(p);
            assert!((c[0] - want).abs() < 0.06 && c[1] == 0.0 && (c[2] - 1.0).abs() < 1e-4, "{p:?}: {c:?} vs red {want}");
        }
        // The baked grid carries them too, and meshes without colours bake none.
        let sdf = MeshSdf::build(&sphere, 24, 0.3, &AtomicU32::new(0));
        assert_eq!(sdf.colors.as_ref().map(|c| c.len()), Some(sdf.data.len()));
        assert!(MeshSdf::build(&shapes::icosphere(1.0, 1), 16, 0.3, &AtomicU32::new(0)).colors.is_none());
    }

    #[test]
    fn baked_grid_matches_the_query_and_extends_past_its_edge() {
        let mesh = shapes::icosphere(1.0, 2);
        let sdf = MeshSdf::build(&mesh, 48, 0.3, &AtomicU32::new(0));
        assert!(sdf.dims.iter().all(|&d| d >= 40), "{:?}", sdf.dims);
        let q = MeshQuery::new(&mesh);
        let worst = cheap_points(500, -1.1, 1.1).iter().map(|p| (sdf.sample(*p) - q.signed_distance(*p)).abs()).fold(0.0, f32::max);
        assert!(worst < 0.03, "grid vs exact: {worst}");
        // Far outside the grid, the distance keeps growing roughly like the real one.
        let far = sdf.sample([4.0, 0.0, 0.0]);
        assert!((far - 3.0).abs() < 0.15, "got {far}");
        assert!(sdf.sample([0.0; 3]) < -0.9);
    }
}
