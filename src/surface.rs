//! Turning a sampled field into a triangle mesh (surface nets), dropping small floating parts, and
//! writing OBJ and PLY files. Pure CPU code, so it is tested without a GPU.

use crate::mesh::V3;
use std::io::Write;

/// A triangle mesh with per-vertex normals.
#[derive(Clone, Debug, Default)]
pub struct RawMesh {
    pub positions: Vec<V3>,
    pub normals: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
}

fn unit(a: V3) -> V3 {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 1e-20 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0; 3] }
}

/// The surface of a field (negative inside) sampled on an `nx` × `ny` × `nz` grid, in index space:
/// vertex coordinates are in sample units, so a vertex at (1.5, 2, 3) is halfway between samples 1
/// and 2 along x. `layer(k)` returns sample layer k (`nx * ny` values, x fastest) and is called once
/// per layer, in order, so a huge grid never has to exist in memory at once.
///
/// Surface nets: one vertex per cell the surface passes through (the average of where it crosses the
/// cell's edges), and a quad for every grid edge it crosses. Normals are the field's gradient.
/// Faces wind counter-clockwise seen from outside.
pub fn surface_nets(nx: usize, ny: usize, nz: usize, mut layer: impl FnMut(usize) -> Vec<f32>) -> RawMesh {
    let mut mesh = RawMesh::default();
    if nx < 2 || ny < 2 || nz < 2 {
        return mesh;
    }
    let (cx, cy) = (nx - 1, ny - 1);
    // The cell vertex of the layer below and of the current layer (-1: the surface misses the cell).
    let mut below = vec![-1i32; cx * cy];
    let mut here = vec![-1i32; cx * cy];
    let mut s0 = layer(0);
    // The 12 edges of a cell between its 8 corners (corner bits: x = 1, y = 2, z = 4).
    const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
    let corner = |c: usize| [(c & 1) as f32, ((c >> 1) & 1) as f32, ((c >> 2) & 1) as f32];
    for k in 0..nz - 1 {
        let s1 = layer(k + 1);
        for j in 0..cy {
            for i in 0..cx {
                let at = |c: usize| {
                    let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let layer = if dz == 0 { &s0 } else { &s1 };
                    layer[(j + dy) * nx + i + dx]
                };
                let v: [f32; 8] = std::array::from_fn(at);
                let inside = v.iter().filter(|x| **x < 0.0).count();
                if inside == 0 || inside == 8 {
                    here[j * cx + i] = -1;
                    continue;
                }
                let (mut sum, mut n) = ([0.0f32; 3], 0.0);
                for (a, b) in EDGES {
                    if (v[a] < 0.0) != (v[b] < 0.0) {
                        let t = v[a] / (v[a] - v[b]);
                        let (pa, pb) = (corner(a), corner(b));
                        for d in 0..3 {
                            sum[d] += pa[d] + (pb[d] - pa[d]) * t;
                        }
                        n += 1.0;
                    }
                }
                let gradient = [
                    (v[1] - v[0]) + (v[3] - v[2]) + (v[5] - v[4]) + (v[7] - v[6]),
                    (v[2] - v[0]) + (v[3] - v[1]) + (v[6] - v[4]) + (v[7] - v[5]),
                    (v[4] - v[0]) + (v[5] - v[1]) + (v[6] - v[2]) + (v[7] - v[3]),
                ];
                here[j * cx + i] = mesh.positions.len() as i32;
                mesh.positions.push([i as f32 + sum[0] / n, j as f32 + sum[1] / n, k as f32 + sum[2] / n]);
                mesh.normals.push(unit(gradient));
            }
        }
        // A quad for each grid edge that starts on sample layer k and crosses the surface; its four
        // cells are the ones around the edge.
        let cell = |i: usize, j: usize, layer: usize| if layer == k { here[j * cx + i] } else { below[j * cx + i] };
        for j in 0..ny {
            for i in 0..nx {
                let start = s0[j * nx + i];
                let mut emit = |quad: [i32; 4], end: f32| {
                    if (start < 0.0) == (end < 0.0) {
                        return;
                    }
                    debug_assert!(quad.iter().all(|c| *c >= 0), "an edge with a sign change always has four surface cells");
                    let q = quad.map(|c| c as u32);
                    // Inside at the start means the field rises along the edge: outward is +axis.
                    let q = if start < 0.0 { q } else { [q[3], q[2], q[1], q[0]] };
                    mesh.triangles.push([q[0], q[1], q[2]]);
                    mesh.triangles.push([q[0], q[2], q[3]]);
                };
                // x edge: cells (i, j-1..j, k-1..k), ordered around +x as (y, z) = (0,0), (-1,0), (-1,-1), (0,-1).
                if i + 1 < nx && j >= 1 && j < cy && k >= 1 {
                    emit([cell(i, j, k), cell(i, j - 1, k), cell(i, j - 1, k - 1), cell(i, j, k - 1)], s0[j * nx + i + 1]);
                }
                // y edge: cells (i-1..i, j, k-1..k), ordered around +y as (z, x).
                if j + 1 < ny && i >= 1 && i < cx && k >= 1 {
                    emit([cell(i, j, k), cell(i, j, k - 1), cell(i - 1, j, k - 1), cell(i - 1, j, k)], s0[(j + 1) * nx + i]);
                }
                // z edge: cells (i-1..i, j-1..j, k), ordered around +z as (x, y).
                if i >= 1 && i < cx && j >= 1 && j < cy {
                    emit([cell(i, j, k), cell(i - 1, j, k), cell(i - 1, j - 1, k), cell(i, j - 1, k)], s1[j * nx + i]);
                }
            }
        }
        std::mem::swap(&mut below, &mut here);
        s0 = s1;
    }
    mesh
}

fn area(a: V3, b: V3, c: V3) -> f32 {
    let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    let w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    0.5 * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

impl RawMesh {
    /// Removes connected parts whose surface area is below `fraction` of the largest part's, and
    /// the vertices only they used. Returns how many parts were removed. A mesh made of several
    /// large parts keeps them all; only small floating bits go.
    pub fn remove_small_parts(&mut self, fraction: f32) -> usize {
        if fraction <= 0.0 || self.triangles.is_empty() {
            return 0;
        }
        let n = self.positions.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        for t in &self.triangles {
            let (a, b, c) = (find(&mut parent, t[0]), find(&mut parent, t[1]), find(&mut parent, t[2]));
            parent[b as usize] = a;
            let c = find(&mut parent, c);
            let a = find(&mut parent, a);
            parent[c as usize] = a;
        }
        let mut part_area = vec![0.0f32; n];
        for t in &self.triangles {
            let root = find(&mut parent, t[0]) as usize;
            part_area[root] += area(self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
        }
        let largest = part_area.iter().cloned().fold(0.0, f32::max);
        let keep_root = |root: usize| part_area[root] >= fraction * largest;
        let removed = (0..n).filter(|&v| find(&mut parent, v as u32) as usize == v && part_area[v] > 0.0 && !keep_root(v)).count();
        if removed == 0 {
            return 0;
        }
        let keep_triangle: Vec<bool> = self.triangles.iter().map(|t| keep_root(find(&mut parent, t[0]) as usize)).collect();
        let mut remap = vec![u32::MAX; n];
        let (mut positions, mut normals) = (Vec::new(), Vec::new());
        let mut triangles = Vec::new();
        for (t, keep) in self.triangles.iter().zip(keep_triangle) {
            if !keep {
                continue;
            }
            triangles.push(t.map(|v| {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = positions.len() as u32;
                    positions.push(self.positions[v as usize]);
                    normals.push(self.normals[v as usize]);
                }
                remap[v as usize]
            }));
        }
        *self = RawMesh { positions, normals, triangles };
        removed
    }

    #[cfg(test)]
    /// Volume enclosed by the surface (positive when the faces wind counter-clockwise from outside).
    pub fn signed_volume(&self) -> f32 {
        self.triangles
            .iter()
            .map(|t| {
                let (a, b, c) = (self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
                (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
            })
            .sum()
    }

    /// Wavefront OBJ with vertex normals.
    pub fn write_obj(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        writeln!(w, "# Noiser")?;
        for p in &self.positions {
            writeln!(w, "v {} {} {}", p[0], p[1], p[2])?;
        }
        for n in &self.normals {
            writeln!(w, "vn {} {} {}", n[0], n[1], n[2])?;
        }
        for t in &self.triangles {
            let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
            writeln!(w, "f {a}//{a} {b}//{b} {c}//{c}")?;
        }
        w.flush()
    }

    /// Binary little-endian PLY with vertex normals: much smaller and faster than OBJ for big meshes.
    pub fn write_ply(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        write!(
            w,
            "ply\nformat binary_little_endian 1.0\ncomment Noiser\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
            self.positions.len(),
            self.triangles.len()
        )?;
        for (p, n) in self.positions.iter().zip(&self.normals) {
            for f in p.iter().chain(n) {
                w.write_all(&f.to_le_bytes())?;
            }
        }
        for t in &self.triangles {
            w.write_all(&[3])?;
            for v in t {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A sphere's signed distance sampled on a grid of `n`³ samples over [-1.25, 1.25]³, extracted.
    fn sphere(n: usize, radius: f32, centre: V3, extent: f32) -> (RawMesh, f32) {
        let h = 2.0 * extent / (n - 1) as f32;
        let at = |i: usize, j: usize, k: usize| {
            let p = [-extent + i as f32 * h - centre[0], -extent + j as f32 * h - centre[1], -extent + k as f32 * h - centre[2]];
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - radius
        };
        let mut m = surface_nets(n, n, n, |k| (0..n * n).map(|ij| at(ij % n, ij / n, k)).collect());
        for p in &mut m.positions {
            *p = [-extent + p[0] * h, -extent + p[1] * h, -extent + p[2] * h];
        }
        (m, h)
    }

    fn edge_counts(m: &RawMesh) -> HashMap<(u32, u32), (u32, u32)> {
        // (times traversed a -> b, times traversed b -> a) per undirected edge.
        let mut e: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for t in &m.triangles {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                let entry = e.entry((a.min(b), a.max(b))).or_default();
                if a < b { entry.0 += 1 } else { entry.1 += 1 }
            }
        }
        e
    }

    #[test]
    fn a_sphere_comes_out_closed_outward_and_the_right_size() {
        let (m, h) = sphere(48, 1.0, [0.0; 3], 1.25);
        assert!(m.triangles.len() > 2000, "{} triangles", m.triangles.len());
        // Closed and consistently oriented: every edge is used once in each direction.
        assert!(edge_counts(&m).values().all(|&c| c == (1, 1)), "the surface is not a closed, consistently wound manifold");
        // Counter-clockwise from outside means a positive volume close to 4/3 π r³.
        let v = m.signed_volume();
        let exact = 4.0 / 3.0 * std::f32::consts::PI;
        assert!((v - exact).abs() < 0.03 * exact, "volume {v}, want about {exact}");
        // Vertices sit on the sphere, and normals point outward.
        for (p, n) in m.positions.iter().zip(&m.normals) {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 0.6 * h, "vertex at radius {r}");
            let outward = (p[0] * n[0] + p[1] * n[1] + p[2] * n[2]) / r;
            assert!(outward > 0.9, "normal {n:?} at {p:?} points {outward}");
        }
    }

    #[test]
    fn layers_are_requested_once_and_in_order() {
        let mut seen = Vec::new();
        surface_nets(6, 5, 7, |k| {
            seen.push(k);
            vec![1.0; 30]
        });
        assert_eq!(seen, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn small_floating_parts_are_removed_and_big_ones_kept() {
        // A big sphere, a second big sphere, and a tiny one, all in one field via min().
        let n = 80;
        let (lo, hi) = (-3.0f32, 3.0f32);
        let h = (hi - lo) / (n - 1) as f32;
        let ball = |p: V3, c: V3, r: f32| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() - r;
        let field = |i: usize, j: usize, k: usize| {
            let p = [lo + i as f32 * h, lo + j as f32 * h, lo + k as f32 * h];
            ball(p, [-1.5, 0.0, 0.0], 1.0).min(ball(p, [1.5, 0.0, 0.0], 0.8)).min(ball(p, [0.0, 2.2, 2.0], 0.22))
        };
        let m = surface_nets(n, n, n, |k| (0..n * n).map(|ij| field(ij % n, ij / n, k)).collect());
        let components = |m: &RawMesh| {
            let mut parent: Vec<u32> = (0..m.positions.len() as u32).collect();
            for t in &m.triangles {
                for e in 0..2 {
                    let (a, b) = (find(&mut parent, t[e]), find(&mut parent, t[e + 1]));
                    parent[b as usize] = a;
                }
            }
            (0..m.positions.len() as u32).filter(|&v| find(&mut parent, v) == v).count()
        };
        assert_eq!(components(&m), 3);
        let mut keep_all = m.clone();
        assert_eq!(keep_all.remove_small_parts(0.0), 0);
        // The big spheres' areas are 12.6 and 8.0; the small one's is 0.6, under 10% of the largest.
        let mut culled = m.clone();
        assert_eq!(culled.remove_small_parts(0.1), 1);
        assert_eq!(components(&culled), 2);
        assert!(culled.triangles.len() < m.triangles.len());
        assert!(culled.triangles.iter().flatten().all(|&v| (v as usize) < culled.positions.len()));
        // With a threshold above the second sphere's share, only the largest remains.
        let mut one = m.clone();
        assert_eq!(one.remove_small_parts(0.8), 2);
        assert_eq!(components(&one), 1);
    }

    #[test]
    fn obj_and_ply_files_have_the_right_contents() {
        let (m, _) = sphere(16, 1.0, [0.0; 3], 1.25);
        let dir = std::env::temp_dir().join("noiser-surface-test");
        std::fs::create_dir_all(&dir).unwrap();
        m.write_obj(&dir.join("s.obj")).unwrap();
        let back = crate::mesh::Mesh::load_obj(&dir.join("s.obj")).unwrap();
        assert_eq!((back.positions.len(), back.triangles.len()), (m.positions.len(), m.triangles.len()));
        m.write_ply(&dir.join("s.ply")).unwrap();
        let bytes = std::fs::read(dir.join("s.ply")).unwrap();
        let header_end = bytes.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
        assert!(header.contains(&format!("element vertex {}", m.positions.len())) && header.contains(&format!("element face {}", m.triangles.len())));
        assert_eq!(bytes.len() - header_end, m.positions.len() * 24 + m.triangles.len() * 13);
    }
}
