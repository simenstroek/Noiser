//! Objective numbers for a triangle mesh: triangle shape, smoothness between neighbours, topology, and (given a
//! way to evaluate the true field) how far it lies from the surface it is meant to describe. Every remeshing
//! method is scored with the same report, so they can be compared without looking at them.

use crate::mesh::V3;
use crate::surface::RawMesh;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub vertices: usize,
    pub triangles: usize,
    /// The smallest angle of any triangle, and the median over triangles of each one's smallest angle, in degrees.
    pub min_angle: f32,
    pub median_min_angle: f32,
    /// Fraction of triangles whose smallest angle is below 20 degrees (thin ones).
    pub thin_fraction: f32,
    /// Radius-ratio quality 4 sqrt(3) A / (a² + b² + c²): 1 for an equilateral triangle, towards 0 for slivers.
    pub mean_quality: f32,
    pub worst_quality: f32,
    /// Angle between the normals of neighbouring triangles, in degrees: 95th percentile and maximum.
    pub p95_normal_deviation: f32,
    pub max_normal_deviation: f32,
    /// Edges whose two triangles are nearly folded flat onto each other (normals more than 150 degrees apart).
    pub fold_edges: usize,
    /// Edges used by one triangle, and by more than two.
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub components: usize,
    /// Distance from the true surface (set by `with_surface_error`), in the mesh's units.
    pub mean_error: Option<f32>,
    pub max_error: Option<f32>,
    /// The 99th percentile of the distance (the maximum is often one stray vertex).
    pub p99_error: Option<f32>,
    /// Triangles whose normal points against the field's gradient at their centre (inside out), if measured.
    pub inverted: Option<usize>,
    /// Triangles that cross another triangle they do not share a vertex with (set by `with_intersections`).
    pub self_intersecting: Option<usize>,
}

impl Report {
    pub fn watertight(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} triangles, {} vertices, {} part(s){}", self.triangles, self.vertices, self.components, if self.watertight() { ", watertight" } else { "" })?;
        writeln!(f, "triangle angle: smallest {:.1}°, median of smallest {:.1}°, {:.1}% under 20°", self.min_angle, self.median_min_angle, 100.0 * self.thin_fraction)?;
        writeln!(f, "triangle quality (1 = equilateral): mean {:.2}, worst {:.2}", self.mean_quality, self.worst_quality)?;
        write!(f, "neighbour normal deviation: 95th percentile {:.1}°, max {:.1}°, {} folded edge(s)", self.p95_normal_deviation, self.max_normal_deviation, self.fold_edges)?;
        if self.boundary_edges + self.non_manifold_edges > 0 {
            write!(f, "\nopen edges {}, non-manifold edges {}", self.boundary_edges, self.non_manifold_edges)?;
        }
        if let (Some(mean), Some(max)) = (self.mean_error, self.max_error) {
            write!(f, "\ndistance from the true surface: mean {mean:.5}, 99th percentile {:.5}, max {max:.5}", self.p99_error.unwrap_or(max))?;
        }
        if let Some(n) = self.self_intersecting {
            write!(f, "\n{n} triangle(s) cross another triangle (self-intersections)")?;
        }
        if let Some(n) = self.inverted {
            write!(f, "\n{n} triangle(s) face against the field")?;
        }
        Ok(())
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f32 {
    dot(a, a).sqrt()
}

/// The `q` quantile (0 to 1) of `values`, which it sorts.
fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[((values.len() - 1) as f32 * q).round() as usize]
}

/// Triangle shape, neighbour smoothness and topology of `mesh`.
pub fn report(mesh: &RawMesh) -> Report {
    let mut r = Report { vertices: mesh.positions.len(), triangles: mesh.triangles.len(), ..Default::default() };
    if mesh.triangles.is_empty() {
        return r;
    }
    let (mut min_angles, mut qualities) = (Vec::with_capacity(r.triangles), Vec::with_capacity(r.triangles));
    let mut normals = Vec::with_capacity(r.triangles);
    // Each edge (low vertex, high vertex) with the triangles on it.
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let p = tri.map(|i| mesh.positions[i as usize]);
        let (e0, e1, e2) = (sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2]));
        let n = cross(e0, [-e2[0], -e2[1], -e2[2]]);
        let area2 = len(n);
        let (l0, l1, l2) = (len(e0), len(e1), len(e2));
        let angle = |a: V3, b: V3, la: f32, lb: f32| (dot(a, b) / (la * lb).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees();
        // The angle at each corner, between the two edges that leave it.
        let neg = |a: V3| [-a[0], -a[1], -a[2]];
        let corner = [angle(e0, neg(e2), l0, l2), angle(neg(e0), e1, l0, l1), angle(neg(e1), e2, l1, l2)];
        min_angles.push(corner.iter().cloned().fold(f32::MAX, f32::min));
        // 4 sqrt(3) A with A = area2 / 2.
        qualities.push(if area2 > 0.0 { 2.0 * 3f32.sqrt() * area2 / (l0 * l0 + l1 * l1 + l2 * l2) } else { 0.0 });
        normals.push(if area2 > 0.0 { [n[0] / area2, n[1] / area2, n[2] / area2] } else { [0.0; 3] });
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(t as u32);
        }
    }
    r.min_angle = min_angles.iter().cloned().fold(f32::MAX, f32::min);
    r.thin_fraction = min_angles.iter().filter(|a| **a < 20.0).count() as f32 / r.triangles as f32;
    r.median_min_angle = quantile(&mut min_angles, 0.5);
    r.mean_quality = qualities.iter().sum::<f32>() / r.triangles as f32;
    r.worst_quality = qualities.iter().cloned().fold(f32::MAX, f32::min);

    let mut deviations = Vec::new();
    // Triangles joined across manifold edges form a part; a union-find keeps this linear.
    let mut parent: Vec<u32> = (0..r.triangles as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for tris in edges.values() {
        match tris.len() {
            1 => r.boundary_edges += 1,
            2 => {
                let angle = dot(normals[tris[0] as usize], normals[tris[1] as usize]).clamp(-1.0, 1.0).acos().to_degrees();
                deviations.push(angle);
                let (a, b) = (root(&mut parent, tris[0]), root(&mut parent, tris[1]));
                parent[a as usize] = b;
            }
            _ => {
                r.non_manifold_edges += 1;
                for w in tris.windows(2) {
                    let (a, b) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
                    parent[a as usize] = b;
                }
            }
        }
    }
    r.components = (0..r.triangles as u32).filter(|&i| root(&mut parent, i) == i).count();
    r.max_normal_deviation = deviations.iter().cloned().fold(0.0, f32::max);
    r.fold_edges = deviations.iter().filter(|a| **a > 150.0).count();
    r.p95_normal_deviation = quantile(&mut deviations, 0.95);
    r
}

/// Whether segment (a, b) passes through the interior of triangle t (Moller-Trumbore; a margin keeps touching out).
fn pierces(a: V3, b: V3, t: [V3; 3]) -> bool {
    let d = sub(b, a);
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let h = cross(d, e2);
    let det = dot(e1, h);
    if det.abs() < 1e-20 {
        return false;
    }
    let f = 1.0 / det;
    let s = sub(a, t[0]);
    let u = f * dot(s, h);
    let q = cross(s, e1);
    let v = f * dot(d, q);
    let w = f * dot(e2, q);
    const M: f32 = 1e-4;
    u > M && v > M && u + v < 1.0 - M && w > M && w < 1.0 - M
}

/// Whether two triangles (corner positions and vertex ids) pass through each other: an edge of one pierces the other,
/// not counting edges that touch a vertex the two share. Triangles that share an edge never count.
pub fn triangles_cross(pa: [V3; 3], ia: [u32; 3], pb: [V3; 3], ib: [u32; 3]) -> bool {
    let shared: Vec<u32> = ia.iter().copied().filter(|v| ib.contains(v)).collect();
    if shared.len() >= 2 {
        return false;
    }
    for k in 0..3 {
        let (u, v) = (ia[k], ia[(k + 1) % 3]);
        if !shared.contains(&u) && !shared.contains(&v) && pierces(pa[k], pa[(k + 1) % 3], pb) {
            return true;
        }
        let (u, v) = (ib[k], ib[(k + 1) % 3]);
        if !shared.contains(&u) && !shared.contains(&v) && pierces(pb[k], pb[(k + 1) % 3], pa) {
            return true;
        }
    }
    false
}

/// The triangles that pass through another triangle: some edge of one pierces the other, ignoring edges that touch a
/// vertex the two share (those meet at the shared vertex). Edge-sharing neighbours folded over each other are not
/// found here; the folded-edge count reports those.
pub fn self_intersecting_triangles(mesh: &RawMesh) -> Vec<bool> {
    let n = mesh.triangles.len();
    let mut bad = vec![false; n];
    if n < 2 {
        return bad;
    }
    let tri = |t: usize| mesh.triangles[t].map(|i| mesh.positions[i as usize]);
    // A uniform grid over the triangles' boxes, cells about the average triangle size.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let mean_edge = {
        let mut total = 0.0;
        for t in 0..n {
            let p = tri(t);
            total += len(sub(p[1], p[0])) + len(sub(p[2], p[1])) + len(sub(p[0], p[2]));
        }
        (total / (3.0 * n as f32)).max(1e-9)
    };
    let cell = mean_edge * 2.0;
    let dims: [usize; 3] = [0, 1, 2].map(|a| (((hi[a] - lo[a]) / cell).ceil() as usize).clamp(1, 256));
    let cell_of = |p: V3| -> [usize; 3] { [0, 1, 2].map(|a| (((p[a] - lo[a]) / cell) as usize).min(dims[a] - 1)) };
    let mut grid: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
    for t in 0..n {
        let p = tri(t);
        let (mut a, mut b) = (cell_of(p[0]), cell_of(p[0]));
        for q in &p[1..] {
            let c = cell_of(*q);
            for k in 0..3 {
                a[k] = a[k].min(c[k]);
                b[k] = b[k].max(c[k]);
            }
        }
        for x in a[0]..=b[0] {
            for y in a[1]..=b[1] {
                for z in a[2]..=b[2] {
                    grid.entry((x * dims[1] + y) * dims[2] + z).or_default().push(t as u32);
                }
            }
        }
    }
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for list in grid.values() {
        for i in 0..list.len() {
            for j in i + 1..list.len() {
                let (ta, tb) = (list[i], list[j]);
                if bad[ta as usize] && bad[tb as usize] || !seen.insert((ta, tb)) {
                    continue;
                }
                let (ia, ib) = (mesh.triangles[ta as usize], mesh.triangles[tb as usize]);
                if triangles_cross(tri(ta as usize), ia, tri(tb as usize), ib) {
                    bad[ta as usize] = true;
                    bad[tb as usize] = true;
                }
            }
        }
    }
    bad
}

/// The triangles that are slivers (quality under 0.1) or folded back over a neighbour (normals more than 125 degrees
/// apart): the ones that cause trouble for UV charting and baking.
pub fn bad_faces(mesh: &RawMesh) -> Vec<bool> {
    let n = mesh.triangles.len();
    let mut normal = vec![[0.0f32; 3]; n];
    let mut bad = vec![false; n];
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let p = tri.map(|i| mesh.positions[i as usize]);
        let (e0, e1, e2) = (sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2]));
        let c = cross(e0, [-e2[0], -e2[1], -e2[2]]);
        let a2 = len(c);
        let q = if a2 > 0.0 { 2.0 * 3f32.sqrt() * a2 / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) } else { 0.0 };
        bad[t] = q < 0.1;
        if a2 > 0.0 {
            normal[t] = [c[0] / a2, c[1] / a2, c[2] / a2];
        }
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(t as u32);
        }
    }
    for tris in edges.values() {
        if let [a, b] = tris[..]
            && dot(normal[a as usize], normal[b as usize]) < -0.57 {
                bad[a as usize] = true;
                bad[b as usize] = true;
            }
    }
    bad
}

impl Report {
    /// Adds how far the mesh lies from the zero level of a field. `sample` takes points and returns
    /// (distance, gradient) for each; the distance to the surface is estimated as |d| / |gradient|, at every
    /// vertex and at the centre of every triangle (which is where a coarse mesh deviates most).
    #[allow(dead_code)] // the remesher will call it; tests do for now
    pub fn with_intersections(mut self, mesh: &RawMesh) -> Report {
        self.self_intersecting = Some(self_intersecting_triangles(mesh).iter().filter(|b| **b).count());
        self
    }

    pub fn with_surface_error(mut self, mesh: &RawMesh, sample: &mut dyn FnMut(&[V3]) -> Vec<[f32; 4]>) -> Report {
        let mut points: Vec<V3> = mesh.positions.clone();
        points.extend(mesh.triangles.iter().map(|t| {
            let p = t.map(|i| mesh.positions[i as usize]);
            [0, 1, 2].map(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0)
        }));
        let samples = sample(&points);
        let errors: Vec<f32> = samples.iter().map(|s| s[0].abs() / len([s[1], s[2], s[3]]).max(1e-6)).collect();
        self.inverted = Some(
            mesh.triangles
                .iter()
                .zip(&samples[mesh.positions.len()..])
                .filter(|(t, s)| {
                    let p = t.map(|i| mesh.positions[i as usize]);
                    dot(cross(sub(p[1], p[0]), sub(p[2], p[0])), [s[1], s[2], s[3]]) < 0.0
                })
                .count(),
        );
        if !errors.is_empty() {
            self.mean_error = Some(errors.iter().sum::<f32>() / errors.len() as f32);
            self.max_error = Some(errors.iter().cloned().fold(0.0, f32::max));
            self.p99_error = Some(quantile(&mut errors.clone(), 0.99));
        }
        self
    }
}

/// How good a UV layout is: what the packing wastes, how many islands there are, whether any overlap, how much the
/// map stretches the surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UvReport {
    pub triangles: usize,
    pub vertices: usize,
    /// Fraction of the 0 to 1 square the islands cover (the target is 70% or more).
    pub coverage: f32,
    /// Texels covered by two different islands, as a fraction of the covered texels (must be 0).
    pub overlap_fraction: f32,
    pub islands: usize,
    /// Islands smaller than 0.02% of the square (about 200 texels at 1024), and islands of a single triangle.
    pub tiny_islands: usize,
    pub single_triangle_islands: usize,
    /// How anisotropic the map is, as the ratio of the two stretch factors of each triangle (1 is conformal).
    pub mean_stretch: f32,
    pub p95_stretch: f32,
    pub max_stretch: f32,
    /// Triangles whose stretch is over 100 or whose UV area is zero (the maximum alone is capped at 1000 and says little).
    pub degenerate: usize,
    /// Texel density (UV size per surface size) of the 95th over the 5th percentile triangle: 1 means even.
    pub density_spread: f32,
    /// Triangles wound the other way in UV space than most of their island (mirrored or folded over).
    pub flipped: usize,
}

impl fmt::Display for UvReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "UVs: {} triangles, {} vertices, {} island(s), {} tiny, {} single-triangle", self.triangles, self.vertices, self.islands, self.tiny_islands, self.single_triangle_islands)?;
        writeln!(f, "coverage {:.1}% of the 0 to 1 square, overlap {:.3}%", 100.0 * self.coverage, 100.0 * self.overlap_fraction)?;
        write!(f, "stretch (1 = none): mean {:.2}, 95th percentile {:.2}, {} triangle(s) over 100; texel density spread {:.1}x; {} flipped", self.mean_stretch, self.p95_stretch, self.degenerate, self.density_spread, self.flipped)
    }
}

/// Scores the UVs of `mesh` by rasterizing them on a `res` by `res` grid.
pub fn uv_report(mesh: &crate::uvgen::UvMesh, res: usize) -> UvReport {
    let n = mesh.triangles.len();
    let mut r = UvReport { triangles: n, vertices: mesh.positions.len(), ..Default::default() };
    if n == 0 {
        return r;
    }
    // Islands: triangles that share a vertex (after the vertices were split along island borders).
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut first = vec![u32::MAX; mesh.positions.len()];
    for (t, tri) in mesh.triangles.iter().enumerate() {
        for &v in tri {
            if first[v as usize] == u32::MAX {
                first[v as usize] = t as u32;
            } else {
                let (a, b) = (root(&mut parent, t as u32), root(&mut parent, first[v as usize]));
                parent[a as usize] = b;
            }
        }
    }
    let island: Vec<u32> = (0..n as u32).map(|t| root(&mut parent, t)).collect();
    let mut area: HashMap<u32, (f32, usize)> = HashMap::new();
    let uv = |t: &[u32; 3]| t.map(|i| mesh.uvs[i as usize]);
    let (mut stretch, mut density) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let q = uv(tri);
        let a2 = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
        let e = area.entry(island[t]).or_insert((0.0, 0));
        e.0 += 0.5 * a2.abs();
        e.1 += 1;
        if a2.abs() < 1e-14 {
            r.degenerate += 1;
            continue;
        }
        // The surface vector per unit of s and of t (Sander et al.): its two singular values are the stretch factors.
        let p = tri.map(|i| mesh.positions[i as usize]);
        let ds = |i: usize, j: usize, k: usize| -> V3 { [0, 1, 2].map(|c| (p[i][c] * (q[j][1] - q[k][1]) + p[j][c] * (q[k][1] - q[i][1]) + p[k][c] * (q[i][1] - q[j][1])) / a2) };
        let dt = |i: usize, j: usize, k: usize| -> V3 { [0, 1, 2].map(|c| (p[i][c] * (q[k][0] - q[j][0]) + p[j][c] * (q[i][0] - q[k][0]) + p[k][c] * (q[j][0] - q[i][0])) / a2) };
        let (ss, st) = (ds(0, 1, 2), dt(0, 1, 2));
        let (a, b, c) = (dot(ss, ss), dot(ss, st), dot(st, st));
        let root_term = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        let (big, small) = (0.5 * (a + c + root_term), (0.5 * (a + c - root_term)).max(1e-20));
        stretch.push((big / small).sqrt().min(1000.0));
        r.degenerate += ((big / small).sqrt() >= 100.0) as usize;
        density.push((big * small).sqrt().sqrt());
    }
    // Winding per island: the sign most of its triangles have; the others are flipped.
    let mut signs: HashMap<u32, (i64, i64)> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let a2 = {
            let q = uv(tri);
            (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1])
        };
        let e = signs.entry(island[t]).or_default();
        if a2 > 0.0 { e.0 += 1 } else if a2 < 0.0 { e.1 += 1 }
    }
    r.flipped = signs.values().map(|(p, n)| *p.min(n) as usize).sum();
    r.islands = area.len();
    r.tiny_islands = area.values().filter(|(a, _)| *a < 2e-4).count();
    r.single_triangle_islands = area.values().filter(|(_, c)| *c == 1).count();
    r.mean_stretch = stretch.iter().sum::<f32>() / stretch.len().max(1) as f32;
    r.p95_stretch = quantile(&mut stretch.clone(), 0.95);
    r.max_stretch = stretch.iter().cloned().fold(0.0, f32::max);
    let (lo, hi) = (quantile(&mut density.clone(), 0.05), quantile(&mut density, 0.95));
    r.density_spread = if lo > 0.0 { hi / lo } else { f32::INFINITY };
    // Coverage and overlap: which island's triangle covers each texel centre.
    const OVER: u32 = u32::MAX - 1;
    let mut grid = vec![u32::MAX; res * res];
    let (mut covered, mut overlap) = (0usize, 0usize);
    let size = res as f32;
    for (t, tri) in mesh.triangles.iter().enumerate() {
        let q = uv(tri).map(|c| [c[0] * size, c[1] * size]);
        let (min_x, max_x) = (q.iter().map(|c| c[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize, (q.iter().map(|c| c[0]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(res));
        let (min_y, max_y) = (q.iter().map(|c| c[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize, (q.iter().map(|c| c[1]).fold(f32::MIN, f32::max).ceil().max(0.0) as usize).min(res));
        let edge = |a: [f32; 2], b: [f32; 2], x: f32, y: f32| (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        for y in min_y..max_y {
            for x in min_x..max_x {
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                let w = [edge(q[0], q[1], cx, cy), edge(q[1], q[2], cx, cy), edge(q[2], q[0], cx, cy)];
                if w.iter().all(|v| *v >= 0.0) || w.iter().all(|v| *v <= 0.0) {
                    let cell = &mut grid[y * res + x];
                    if *cell == u32::MAX {
                        *cell = island[t];
                        covered += 1;
                    } else if *cell != island[t] && *cell != OVER {
                        *cell = OVER;
                        overlap += 1;
                    }
                }
            }
        }
    }
    r.coverage = covered as f32 / (res * res) as f32;
    r.overlap_fraction = overlap as f32 / covered.max(1) as f32;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(positions: Vec<V3>, triangles: Vec<[u32; 3]>) -> RawMesh {
        RawMesh { normals: vec![[0.0; 3]; positions.len()], positions, triangles }
    }

    #[test]
    fn an_equilateral_triangle_is_perfect_and_a_sliver_is_not() {
        let h = 3f32.sqrt() / 2.0;
        let r = report(&mesh(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], vec![[0, 1, 2]]));
        assert!((r.min_angle - 60.0).abs() < 0.01 && (r.mean_quality - 1.0).abs() < 0.001, "{r}");
        assert_eq!((r.boundary_edges, r.components, r.thin_fraction), (3, 1, 0.0));
        let r = report(&mesh(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.01, 0.0]], vec![[0, 1, 2]]));
        assert!(r.min_angle < 2.0 && r.worst_quality < 0.05 && r.thin_fraction == 1.0, "{r}");
    }

    #[test]
    fn a_tetrahedron_is_watertight_and_two_apart_are_two_parts() {
        let tet = |o: f32| (vec![[o, 0.0, 0.0], [o + 1.0, 0.0, 0.0], [o, 1.0, 0.0], [o, 0.0, 1.0]], vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]);
        let (p, t) = tet(0.0);
        let r = report(&mesh(p.clone(), t.clone()));
        assert!(r.watertight() && r.components == 1, "{r}");
        // Dihedral angles of this corner tetrahedron are 90 degrees at the axes and 54.7 on the slanted face.
        assert!((r.max_normal_deviation - 125.26).abs() < 0.5, "{}", r.max_normal_deviation);
        let (p2, t2) = tet(5.0);
        let mut positions = p;
        positions.extend(p2);
        let mut triangles = t;
        triangles.extend(t2.iter().map(|t| t.map(|i| i + 4)));
        assert_eq!(report(&mesh(positions, triangles)).components, 2);
    }

    #[test]
    fn surface_error_reads_the_field() {
        // A triangle in the plane z = 0.1 against the field z (the plane z = 0): 0.1 away everywhere.
        let m = mesh(vec![[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]], vec![[0, 1, 2]]);
        let r = report(&m).with_surface_error(&m, &mut |pts| pts.iter().map(|p| [p[2], 0.0, 0.0, 1.0]).collect());
        assert!((r.mean_error.unwrap() - 0.1).abs() < 1e-6 && (r.max_error.unwrap() - 0.1).abs() < 1e-6);
    }
}
