//! Triangle meshes: OBJ import, vertex welding, and the normalization that puts a mesh into the
//! space the preview and the export work in (centred, bounding sphere of radius 1).

use std::collections::HashMap;
use std::path::Path;

pub type V3 = [f32; 3];

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex colours (0 to 1), parallel to `positions`; empty when the file has none.
    pub colors: Vec<V3>,
    /// Texture coordinates of each triangle's corners (V up, as files store them), parallel to `triangles`; empty
    /// when the file has none.
    pub uvs: Vec<[[f32; 2]; 3]>,
}

/// Maps object space to the normalized space: `normalized = (object - center) / scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    pub center: V3,
    pub scale: f32,
}

impl Normalization {
    #[cfg(test)]
    pub const IDENTITY: Normalization = Normalization { center: [0.0; 3], scale: 1.0 };

    pub fn to_object(&self, p: V3) -> V3 {
        [p[0] * self.scale + self.center[0], p[1] * self.scale + self.center[1], p[2] * self.scale + self.center[2]]
    }
}

impl Mesh {
    /// Reads every object in an OBJ file as one mesh (faces are triangulated) and welds vertices
    /// that share a position, so the surface is connected even where UV seams split vertices.
    pub fn load_obj(path: &Path) -> Result<Mesh, String> {
        let options = tobj::LoadOptions { triangulate: true, single_index: false, ..Default::default() };
        let (models, _materials) = tobj::load_obj(path, &options).map_err(|e| format!("could not read {}: {e}", path.display()))?;
        let mut mesh = Mesh::default();
        for model in models {
            let base = mesh.positions.len() as u32;
            let count = model.mesh.positions.len() / 3;
            // `v x y z r g b` lines: colours (zero for objects that have none, so the lists stay parallel).
            if model.mesh.vertex_color.len() == count * 3 {
                mesh.colors.resize(mesh.positions.len(), [0.0; 3]);
                mesh.colors.extend(model.mesh.vertex_color.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
            } else if !mesh.colors.is_empty() {
                mesh.colors.resize(mesh.positions.len() + count, [0.0; 3]);
            }
            mesh.positions.extend(model.mesh.positions.chunks_exact(3).map(|p| [p[0], p[1], p[2]]));
            let before = mesh.triangles.len();
            mesh.triangles.extend(model.mesh.indices.chunks_exact(3).map(|t| [base + t[0], base + t[1], base + t[2]]));
            // `vt` coordinates, per corner (a vertex on a seam has several). Kept only if every object has them.
            if model.mesh.texcoord_indices.len() == model.mesh.indices.len() && !model.mesh.texcoords.is_empty() && mesh.uvs.len() == before {
                let uv = |i: u32| [model.mesh.texcoords[2 * i as usize], model.mesh.texcoords[2 * i as usize + 1]];
                mesh.uvs.extend(model.mesh.texcoord_indices.chunks_exact(3).map(|t| [uv(t[0]), uv(t[1]), uv(t[2])]));
            }
        }
        if mesh.triangles.is_empty() {
            return Err(format!("{} has no triangles", path.display()));
        }
        Ok(mesh.welded())
    }

    /// Axis-aligned bounds of the vertices the triangles use.
    pub fn bounds(&self) -> (V3, V3) {
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for t in &self.triangles {
            for &i in t {
                for a in 0..3 {
                    lo[a] = lo[a].min(self.positions[i as usize][a]);
                    hi[a] = hi[a].max(self.positions[i as usize][a]);
                }
            }
        }
        (lo, hi)
    }

    /// Merges vertices at (nearly) the same position and drops triangles that collapse.
    pub fn welded(&self) -> Mesh {
        let (lo, hi) = self.bounds();
        let diagonal = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1e-12);
        let cell = 1e-5 * diagonal;
        let mut by_cell: HashMap<[i64; 3], u32> = HashMap::new();
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let remap: Vec<u32> = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let key = [(p[0] / cell).round() as i64, (p[1] / cell).round() as i64, (p[2] / cell).round() as i64];
                *by_cell.entry(key).or_insert_with(|| {
                    positions.push(*p);
                    if let Some(c) = self.colors.get(i) {
                        colors.push(*c); // the first of the merged vertices keeps its colour
                    }
                    positions.len() as u32 - 1
                })
            })
            .collect();
        let mut triangles = Vec::new();
        let mut uvs = Vec::new();
        for (k, t) in self.triangles.iter().enumerate() {
            let t = [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]];
            // A vertex with NaN or infinite coordinates would poison the bounds and the distance grid.
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] && t.iter().all(|&i| positions[i as usize].iter().all(|x| x.is_finite())) {
                triangles.push(t);
                if let Some(uv) = self.uvs.get(k) {
                    uvs.push(*uv);
                }
            }
        }
        Mesh { positions, triangles, colors, uvs }
    }

    /// The mesh moved and scaled so its bounding box is centred on the origin and its bounding
    /// sphere has radius 1 (the size of the preview's unit shapes), and the transform that did it.
    pub fn normalized(&self) -> (Mesh, Normalization) {
        let (lo, hi) = self.bounds();
        let center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
        let mut radius = 0.0f32;
        for t in &self.triangles {
            for &i in t {
                let p = self.positions[i as usize];
                radius = radius.max(((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2)).sqrt());
            }
        }
        let norm = Normalization { center, scale: if radius > 0.0 { radius } else { 1.0 } };
        let positions = self.positions.iter().map(|p| [(p[0] - center[0]) / norm.scale, (p[1] - center[1]) / norm.scale, (p[2] - center[2]) / norm.scale]).collect();
        (Mesh { positions, triangles: self.triangles.clone(), colors: self.colors.clone(), uvs: self.uvs.clone() }, norm)
    }
}

/// Meshes built in code, for tests.
#[cfg(test)]
pub mod shapes {
    use super::*;

    /// An axis-aligned cube from -1 to 1, with the corner vertices duplicated per face (as an OBJ
    /// with UV seams would have them), wound outward.
    pub fn split_cube() -> Mesh {
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let mut m = Mesh::default();
        for (n, u, v) in faces {
            let base = m.positions.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                m.positions.push([n[0] + su * u[0] + sv * v[0], n[1] + su * u[1] + sv * v[1], n[2] + su * u[2] + sv * v[2]]);
            }
            m.triangles.push([base, base + 1, base + 2]);
            m.triangles.push([base, base + 2, base + 3]);
        }
        m
    }

    /// A sphere of the given radius from a subdivided icosahedron, wound outward.
    pub fn icosphere(radius: f32, subdivisions: u32) -> Mesh {
        let t = (1.0 + 5.0f32.sqrt()) / 2.0;
        let mut positions: Vec<V3> = vec![
            [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
            [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
            [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
        ];
        let mut triangles: Vec<[u32; 3]> = vec![
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        let unit = |p: V3| {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / l, p[1] / l, p[2] / l]
        };
        for p in &mut positions {
            *p = unit(*p);
        }
        for _ in 0..subdivisions {
            let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next = Vec::new();
            for t in &triangles {
                let mut m = [0u32; 3];
                for e in 0..3 {
                    let (a, b) = (t[e], t[(e + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    m[e] = *mid.entry(key).or_insert_with(|| {
                        let (pa, pb) = (positions[a as usize], positions[b as usize]);
                        positions.push(unit([(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0]));
                        positions.len() as u32 - 1
                    });
                }
                next.extend([[t[0], m[0], m[2]], [t[1], m[1], m[0]], [t[2], m[2], m[1]], [m[0], m[1], m[2]]]);
            }
            triangles = next;
        }
        for p in &mut positions {
            *p = [p[0] * radius, p[1] * radius, p[2] * radius];
        }
        Mesh { positions, triangles, colors: Vec::new(), uvs: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_vertex_colours_are_kept_through_welding_and_normalizing() {
        let dir = std::env::temp_dir().join("noiser-obj-colour-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.obj");
        // A quad of two triangles; the shared corners are listed twice (a seam) and welded.
        std::fs::write(&path, "v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 1 1 0 0 0 1
v 0 0 0 1 0 0
v 1 1 0 0 0 1
v 0 1 0 1 1 1
f 1 2 3
f 4 5 6
").unwrap();
        let m = Mesh::load_obj(&path).unwrap();
        assert_eq!(m.positions.len(), 4);
        assert_eq!(m.colors.len(), 4, "one colour per welded vertex");
        let (n, _) = m.normalized();
        assert_eq!(n.colors, m.colors);
        // An OBJ without colours has none.
        let plain = dir.join("p.obj");
        std::fs::write(&plain, "v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
").unwrap();
        assert!(Mesh::load_obj(&plain).unwrap().colors.is_empty());
    }

    #[test]
    fn triangles_with_non_finite_vertices_are_dropped() {
        let mut m = shapes::icosphere(1.0, 1);
        let before = m.triangles.len();
        m.positions[0][2] = f32::NAN;
        let w = m.welded();
        assert!(w.triangles.len() < before && !w.triangles.is_empty());
        assert!(w.triangles.iter().all(|t| t.iter().all(|&i| w.positions[i as usize].iter().all(|x| x.is_finite()))));
    }

    #[test]
    fn welding_merges_split_vertices_and_keeps_the_surface() {
        let cube = shapes::split_cube();
        assert_eq!(cube.positions.len(), 24);
        let w = cube.welded();
        assert_eq!((w.positions.len(), w.triangles.len()), (8, 12));
        // Every edge of a closed surface is shared by exactly two triangles.
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &w.triangles {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(edges.values().all(|&n| n == 2), "the welded cube should be closed");
    }

    #[test]
    fn normalizing_centres_and_fits_the_unit_sphere() {
        let mut m = shapes::icosphere(3.0, 2);
        for p in &mut m.positions {
            *p = [p[0] + 10.0, p[1] * 2.0 - 5.0, p[2]]; // moved and stretched
        }
        let (n, norm) = m.normalized();
        let (lo, hi) = n.bounds();
        for a in 0..3 {
            assert!((lo[a] + hi[a]).abs() < 1e-5, "axis {a} is not centred");
        }
        let radius = n.positions.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).fold(0.0, f32::max);
        assert!((radius - 1.0).abs() < 1e-5);
        // And the transform maps back.
        let back = norm.to_object(n.positions[7]);
        for a in 0..3 {
            assert!((back[a] - m.positions[7][a]).abs() < 1e-4);
        }
    }

    #[test]
    fn obj_files_load_and_weld() {
        let dir = std::env::temp_dir().join("noiser-obj-test");
        std::fs::create_dir_all(&dir).unwrap();
        // A quad as two triangles, with a duplicated vertex (as a UV seam would leave).
        let obj = "o quad\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 0\nf 1 2 3\nf 5 3 4\n";
        let path = dir.join("quad.obj");
        std::fs::write(&path, obj).unwrap();
        let m = Mesh::load_obj(&path).unwrap();
        assert_eq!((m.positions.len(), m.triangles.len()), (4, 2));
        assert!(Mesh::load_obj(&dir.join("missing.obj")).is_err());
        std::fs::write(dir.join("empty.obj"), "o nothing\nv 0 0 0\n").unwrap();
        assert!(Mesh::load_obj(&dir.join("empty.obj")).is_err());
    }
}
