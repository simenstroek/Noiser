//! Transferring the UVs of one mesh onto another that lies on (nearly) the same surface, for a remeshed or displaced
//! mesh whose source already has a good layout.
//!
//! Every target corner takes the UV at the closest point of the source surface. Near a seam the corners of one
//! target triangle could land on different islands and stretch the triangle across the gap, so each triangle picks
//! the island under its centre and its corners look for the closest point *on that island*; vertices are then
//! duplicated per (vertex, island), which is also where the new mesh's seams run.

use crate::mesh::{Mesh, V3};
use crate::surface::RawMesh;
use crate::uvgen::UvMesh;
use std::collections::HashMap;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The point of triangle `abc` closest to `p` as barycentric coordinates (Ericson, Real-Time Collision Detection).
fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> [f32; 3] {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    [1.0 - v - w, v, w]
}

/// The triangles of a mesh in a uniform grid, for nearest-triangle queries.
struct Grid {
    cell: f32,
    origin: V3,
    cells: HashMap<[i32; 3], Vec<u32>>,
    extent: f32,
}

impl Grid {
    fn new(mesh: &Mesh) -> Grid {
        let (lo, hi) = mesh.bounds();
        let extent = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max).max(1e-9);
        // About one triangle per cell on a surface: the cell is the average edge length.
        let mut edge = 0.0f32;
        for t in &mesh.triangles {
            let p = t.map(|i| mesh.positions[i as usize]);
            edge += (0..3).map(|k| (0..3).map(|a| (p[k][a] - p[(k + 1) % 3][a]).powi(2)).sum::<f32>().sqrt()).sum::<f32>() / 3.0;
        }
        let cell = (edge / mesh.triangles.len().max(1) as f32).max(extent / 512.0);
        let mut g = Grid { cell, origin: lo, cells: HashMap::new(), extent };
        for (f, t) in mesh.triangles.iter().enumerate() {
            let p = t.map(|i| mesh.positions[i as usize]);
            let (tlo, thi) = ([0, 1, 2].map(|a| p.iter().map(|q| q[a]).fold(f32::MAX, f32::min)), [0, 1, 2].map(|a| p.iter().map(|q| q[a]).fold(f32::MIN, f32::max)));
            let (clo, chi) = (g.index(tlo), g.index(thi));
            for x in clo[0]..=chi[0] {
                for y in clo[1]..=chi[1] {
                    for z in clo[2]..=chi[2] {
                        g.cells.entry([x, y, z]).or_default().push(f as u32);
                    }
                }
            }
        }
        g
    }

    fn index(&self, p: V3) -> [i32; 3] {
        [0, 1, 2].map(|a| ((p[a] - self.origin[a]) / self.cell).floor() as i32)
    }

    /// The triangle accepted by `ok` that is closest to `p`, with the barycentric coordinates of the closest point.
    fn nearest(&self, mesh: &Mesh, p: V3, ok: &dyn Fn(u32) -> bool) -> Option<(u32, [f32; 3], f32)> {
        let mut radius = self.cell;
        loop {
            let (lo, hi) = (self.index([0, 1, 2].map(|a| p[a] - radius)), self.index([0, 1, 2].map(|a| p[a] + radius)));
            let mut best: Option<(u32, [f32; 3], f32)> = None;
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        let Some(list) = self.cells.get(&[x, y, z]) else { continue };
                        for &f in list {
                            if !ok(f) {
                                continue;
                            }
                            let t = mesh.triangles[f as usize].map(|i| mesh.positions[i as usize]);
                            let b = closest_on_triangle(p, t[0], t[1], t[2]);
                            let q: V3 = std::array::from_fn(|a| b[0] * t[0][a] + b[1] * t[1][a] + b[2] * t[2][a]);
                            let d = dot(sub(p, q), sub(p, q)).sqrt();
                            if best.is_none_or(|(_, _, bd)| d < bd) {
                                best = Some((f, b, d));
                            }
                        }
                    }
                }
            }
            // Anything inside the searched cube is found; the best is final once it is within the cube's inner radius.
            if let Some(b) = best.filter(|b| b.2 <= radius) {
                return Some(b);
            }
            if radius > 4.0 * self.extent {
                return best;
            }
            radius *= 2.0;
        }
    }
}

/// How well the transfer went.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Corners of the target farther from the source surface than 2% of its size (the meshes disagree there).
    pub far_corners: usize,
    pub islands: usize,
}

/// The source mesh's islands: triangles joined across edges whose UVs agree at both ends.
fn islands(mesh: &Mesh) -> Vec<u32> {
    let n = mesh.triangles.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    // Per undirected edge: the triangles on it with the UV at the lower and the higher vertex.
    let mut edges: HashMap<(u32, u32), Vec<(u32, [f32; 2], [f32; 2])>> = HashMap::new();
    for (f, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (ua, ub) = (mesh.uvs[f][k], mesh.uvs[f][(k + 1) % 3]);
            if a < b { edges.entry((a, b)).or_default().push((f as u32, ua, ub)) } else { edges.entry((b, a)).or_default().push((f as u32, ub, ua)) }
        }
    }
    let same = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4;
    for list in edges.values() {
        for i in 0..list.len() {
            for j in i + 1..list.len() {
                if same(list[i].1, list[j].1) && same(list[i].2, list[j].2) {
                    let (a, b) = (root(&mut parent, list[i].0), root(&mut parent, list[j].0));
                    parent[a as usize] = b;
                }
            }
        }
    }
    (0..n as u32).map(|f| root(&mut parent, f)).collect()
}

/// The UVs of `source` (per triangle corner, V up) carried onto `target`; both in the same space.
pub fn transfer(target: &RawMesh, source: &Mesh) -> Result<(UvMesh, Stats), String> {
    transfer_with(target, source, 30)
}

pub fn transfer_with(target: &RawMesh, source: &Mesh, relax_iterations: usize) -> Result<(UvMesh, Stats), String> {
    if source.uvs.len() != source.triangles.len() || source.uvs.is_empty() {
        return Err("the source mesh has no UVs (an OBJ with vt lines, or a glTF with TEXCOORD_0, is needed)".into());
    }
    let island = islands(source);
    let grid = Grid::new(source);
    let mut out = UvMesh::default();
    let mut vertex: HashMap<(u32, u32), u32> = HashMap::new();
    let mut stats = Stats::default();
    let mut seen = std::collections::HashSet::new();
    let far = 0.02 * grid.extent;
    for t in &target.triangles {
        let p = t.map(|i| target.positions[i as usize]);
        let centre: V3 = std::array::from_fn(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0);
        let Some((t0, _, _)) = grid.nearest(source, centre, &|_| true) else { return Err("the source mesh is empty".into()) };
        let home = island[t0 as usize];
        seen.insert(home);
        let mut corner = [0u32; 3];
        for k in 0..3 {
            let key = (t[k], home);
            corner[k] = match vertex.get(&key) {
                Some(&v) => v,
                None => {
                    let Some((f, b, d)) = grid.nearest(source, p[k], &|f| island[f as usize] == home) else { return Err("the source mesh is empty".into()) };
                    if d > far {
                        stats.far_corners += 1;
                    }
                    let uv = source.uvs[f as usize];
                    let u = b[0] * uv[0][0] + b[1] * uv[1][0] + b[2] * uv[2][0];
                    let v = b[0] * uv[0][1] + b[1] * uv[1][1] + b[2] * uv[2][1];
                    out.positions.push(p[k]);
                    out.normals.push(target.normals.get(t[k] as usize).copied().unwrap_or([0.0; 3]));
                    out.uvs.push([u, 1.0 - v]); // UvMesh keeps V down
                    out.source.push(t[k]);
                    let id = out.positions.len() as u32 - 1;
                    vertex.insert(key, id);
                    id
                }
            };
        }
        out.triangles.push(corner);
    }
    stats.islands = seen.len();
    relax(&mut out, relax_iterations);
    Ok((out, stats))
}

/// Smooths the interior UVs of every island towards the cotangent-weighted average of their neighbours, keeping island
/// borders (and so the seams) where they are. A move that would turn a triangle over is skipped.
pub fn relax(uv: &mut UvMesh, iterations: usize) {
    let n = uv.positions.len();
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &uv.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *edge_count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut pinned = vec![false; n];
    for (&(a, b), &c) in &edge_count {
        if c != 2 {
            pinned[a as usize] = true;
            pinned[b as usize] = true;
        }
    }
    // Per vertex: (neighbour, cotangent weight) and the incident triangles.
    let mut nbr: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (f, t) in uv.triangles.iter().enumerate() {
        for k in 0..3 {
            let (i, a, b) = (t[k] as usize, t[(k + 1) % 3] as usize, t[(k + 2) % 3] as usize);
            inc[i].push(f as u32);
            // The angle at i is opposite edge a-b; its cotangent weighs that edge's two ends towards each other.
            let (u, v) = (sub(uv.positions[a], uv.positions[i]), sub(uv.positions[b], uv.positions[i]));
            let cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let w = (dot(u, v) / dot(cr, cr).sqrt().max(1e-20)).clamp(0.0, 5.0) * 0.5;
            nbr[a].push((b as u32, w));
            nbr[b].push((a as u32, w));
        }
    }
    let area = |uv: &UvMesh, f: u32| {
        let t = uv.triangles[f as usize].map(|i| uv.uvs[i as usize]);
        (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])
    };
    let sign: Vec<f32> = (0..uv.triangles.len() as u32).map(|f| area(uv, f).signum()).collect();
    for _ in 0..iterations {
        for i in 0..n {
            if pinned[i] || nbr[i].is_empty() {
                continue;
            }
            let (mut sum, mut wsum) = ([0.0f32; 2], 0.0f32);
            for &(j, w) in &nbr[i] {
                sum[0] += w * uv.uvs[j as usize][0];
                sum[1] += w * uv.uvs[j as usize][1];
                wsum += w;
            }
            if wsum <= 1e-9 {
                continue;
            }
            let old = uv.uvs[i];
            uv.uvs[i] = [old[0] + 0.5 * (sum[0] / wsum - old[0]), old[1] + 0.5 * (sum[1] / wsum - old[1])];
            if inc[i].iter().any(|&f| area(uv, f) * sign[f as usize] <= 0.0) {
                uv.uvs[i] = old;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshquality::uv_report;

    fn sphere(subdivisions: u32) -> RawMesh {
        let m = crate::mesh::shapes::icosphere(1.0, subdivisions);
        RawMesh { normals: m.positions.clone(), positions: m.positions, triangles: m.triangles }
    }

    /// A mesh with UVs made by the generator, as a loader would give it.
    fn unwrapped_source(raw: &RawMesh) -> Mesh {
        let uv = crate::uvgen::generate(raw, &crate::uvgen::UvParams { resolution: 512, ..Default::default() }).unwrap();
        let uvs = uv.triangles.iter().map(|t| t.map(|i| [uv.uvs[i as usize][0], 1.0 - uv.uvs[i as usize][1]])).collect();
        // The input's vertices, with the triangles as they were.
        Mesh { positions: raw.positions.clone(), triangles: uv.triangles.iter().map(|t| t.map(|i| uv.source[i as usize])).collect(), colors: Vec::new(), uvs }
    }

    #[test]
    fn a_coarser_sphere_takes_the_uvs_of_a_finer_one_without_overlaps() {
        let source = unwrapped_source(&sphere(3));
        let (uv, stats) = transfer(&sphere(2), &source).unwrap();
        assert_eq!(uv.triangles.len(), sphere(2).triangles.len());
        // Corners on a seam may have to reach across to their own island; most are on the surface.
        assert!(stats.far_corners * 5 < uv.positions.len(), "{} of {} corners are far from the source", stats.far_corners, uv.positions.len());
        assert!(stats.islands >= 2, "the layout has several islands");
        let r = uv_report(&uv, 512);
        eprintln!("{r}");
        // Triangles are never stretched across islands: overlaps are what that would cause, and a few at seams are
        // the price of a coarser mesh following a finer one's borders.
        assert!(r.overlap_fraction < 0.02 && r.coverage > 0.3, "{r}");
        assert!(uv.positions.len() > sphere(2).positions.len(), "seam vertices were duplicated");
        assert!(uv.uvs.iter().all(|t| (-0.01..=1.01).contains(&t[0]) && (-0.01..=1.01).contains(&t[1])));
        // Relaxing the interior must not hurt: no new flips and no more stretch.
        let (plain, _) = transfer_with(&sphere(2), &source, 0).unwrap();
        let p = uv_report(&plain, 512);
        eprintln!("unrelaxed:
{p}");
        assert!(r.flipped <= p.flipped && r.mean_stretch <= p.mean_stretch + 1e-4, "relaxed {r}
plain {p}");
        // A source without UVs is refused.
        let bare = Mesh { uvs: Vec::new(), ..source };
        assert!(transfer(&sphere(2), &bare).is_err());
    }
}
