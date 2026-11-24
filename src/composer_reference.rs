//! A reference model for the SDF Composer: a triangle mesh (a character from ZBrush, say) drawn in the viewport next to
//! the composition, to model on top of. It is never part of the composition or of what is exported, and it is
//! rasterised by the GPU (`composer_gpu`) so a mesh of a few hundred thousand triangles stays cheap.

use crate::mesh::{Mesh, V3};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The mesh as the GPU wants it: shared vertices with smooth normals, and indices.
pub struct ReferenceMesh {
    /// Changes with every loaded mesh, so the GPU knows when to upload again.
    pub id: u64,
    pub vertices: Vec<[f32; 6]>,
    pub indices: Vec<u32>,
    pub lo: V3,
    pub hi: V3,
}

impl ReferenceMesh {
    pub fn from_mesh(mesh: &Mesh, id: u64) -> ReferenceMesh {
        let (lo, hi) = mesh.bounds();
        let mut normals = vec![[0.0f32; 3]; mesh.positions.len()];
        for t in &mesh.triangles {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            // Not normalised: the cross product's length is the triangle's area, which weights the average.
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            for i in t {
                for k in 0..3 {
                    normals[*i as usize][k] += n[k];
                }
            }
        }
        let vertices = mesh
            .positions
            .iter()
            .zip(&normals)
            .map(|(p, n)| {
                let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
                [p[0], p[1], p[2], n[0] / length, n[1] / length, n[2] / length]
            })
            .collect();
        ReferenceMesh { id, vertices, indices: mesh.triangles.iter().flatten().copied().collect(), lo, hi }
    }

    pub fn load(path: &Path, id: u64) -> Result<ReferenceMesh, String> {
        let mesh = crate::meshformats::load(path)?;
        Ok(ReferenceMesh::from_mesh(&mesh, id))
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// How the reference is placed and drawn. Saved with the app's session, not in the `.sdf` file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReferenceSettings {
    pub path: PathBuf,
    pub visible: bool,
    /// 1 hides what is behind the reference; less shows the composition through it.
    pub opacity: f32,
    /// The model's height (its y extent) in the composer's units.
    pub height: f64,
    /// Where the middle of the model's feet is.
    pub offset: [f64; 3],
    /// Turn about the vertical axis, in degrees.
    pub yaw: f64,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        ReferenceSettings { path: PathBuf::new(), visible: true, opacity: 1.0, height: 1.8, offset: [0.0; 3], yaw: 0.0 }
    }
}

pub struct Reference {
    pub settings: ReferenceSettings,
    pub mesh: Arc<ReferenceMesh>,
}

impl Reference {
    /// The mesh point `p` as the composer sees it: the model is scaled to its height, stood on its feet at the origin
    /// (the middle of its bounding box in x and z), turned, and moved by the offset.
    pub fn place(&self, p: V3) -> [f64; 3] {
        let m = self.model();
        [0, 1, 2].map(|r| m[r][0] * p[0] as f64 + m[r][1] * p[1] as f64 + m[r][2] * p[2] as f64 + m[r][3])
    }

    /// The model matrix, three rows of a 3x4 matrix.
    pub fn model(&self) -> [[f64; 4]; 3] {
        let (lo, hi) = (self.mesh.lo, self.mesh.hi);
        let extent = (hi[1] - lo[1]).max(1e-6) as f64;
        let scale = self.settings.height.max(1e-3) / extent;
        let base = [(lo[0] + hi[0]) as f64 * 0.5, lo[1] as f64, (lo[2] + hi[2]) as f64 * 0.5];
        let (s, c) = self.settings.yaw.to_radians().sin_cos();
        // Rotation about y: x' = c x + s z, z' = -s x + c z.
        let rows = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let mut out = [[0.0; 4]; 3];
        for r in 0..3 {
            for k in 0..3 {
                out[r][k] = rows[r][k] * scale;
            }
            out[r][3] = self.settings.offset[r] - (0..3).map(|k| out[r][k] * base[k]).sum::<f64>();
        }
        out
    }

    /// The box around the placed model.
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let (lo, hi) = (self.mesh.lo, self.mesh.hi);
        let mut out = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for corner in 0..8 {
            let p = self.place([0, 1, 2].map(|k| if corner >> k & 1 == 0 { lo[k] } else { hi[k] }));
            for k in 0..3 {
                out.0[k] = out.0[k].min(p[k]);
                out.1[k] = out.1[k].max(p[k]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tall_box() -> Reference {
        let mesh = ReferenceMesh { id: 1, vertices: Vec::new(), indices: Vec::new(), lo: [-10.0, -50.0, 0.0], hi: [30.0, 150.0, 20.0] };
        Reference { settings: ReferenceSettings::default(), mesh: Arc::new(mesh) }
    }

    #[test]
    fn the_model_is_scaled_to_its_height_and_stands_on_the_origin() {
        let reference = tall_box();
        let (lo, hi) = reference.bounds();
        assert!((hi[1] - lo[1] - 1.8).abs() < 1e-9, "{lo:?} {hi:?}");
        assert!(lo[1].abs() < 1e-9, "feet on the floor");
        assert!((lo[0] + hi[0]).abs() < 1e-9 && (lo[2] + hi[2]).abs() < 1e-9, "centred");
    }

    #[test]
    fn the_offset_and_the_turn_move_the_model() {
        let mut reference = tall_box();
        reference.settings.offset = [1.0, 2.0, 3.0];
        let (lo, hi) = reference.bounds();
        assert!((lo[1] - 2.0).abs() < 1e-9 && ((lo[0] + hi[0]) * 0.5 - 1.0).abs() < 1e-9 && ((lo[2] + hi[2]) * 0.5 - 3.0).abs() < 1e-9);
        reference.settings.yaw = 90.0;
        let (lo, hi) = reference.bounds();
        // The box was 40 wide and 20 deep: a quarter turn swaps them.
        let scale = 1.8 / 200.0;
        assert!((hi[0] - lo[0] - 20.0 * scale).abs() < 1e-9 && (hi[2] - lo[2] - 40.0 * scale).abs() < 1e-9);
    }

    #[test]
    fn normals_point_out_of_a_single_triangle_by_its_winding() {
        let mesh = Mesh { positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], triangles: vec![[0, 1, 2]], ..Default::default() };
        let reference = ReferenceMesh::from_mesh(&mesh, 7);
        for v in &reference.vertices {
            assert!((v[5] - 1.0).abs() < 1e-6, "{v:?}");
        }
        assert_eq!(reference.triangle_count(), 1);
    }

    #[test]
    fn the_zbrush_reference_loads_when_it_is_there() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("zbrush_reference").join("male_zbrush.obj");
        if !path.exists() {
            return;
        }
        let mesh = ReferenceMesh::load(&path, 1).unwrap();
        assert!(mesh.triangle_count() > 100_000);
        assert!(mesh.vertices.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }
}
