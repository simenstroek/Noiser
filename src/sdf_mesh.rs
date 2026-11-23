//! An SDF Modeler composition as a mesh, so the Mesh graph's Load Mesh node (and everything else that reads meshes)
//! can take a `.sdf` file. The visible layers are sampled on a grid and the surface is extracted with surface nets. The
//! mesh is Z up, like the PLY files SDF Modeler exports: file `(x, y, z)` becomes `(x, -z, y)`.

use crate::mesh::Mesh;
use crate::sdf_file::SdfFile;
use crate::sdf_model::layer_fields;
use crate::surface::{RawMesh, surface_nets};
use rayon::prelude::*;
use std::path::Path;

/// Samples along the longest side of the grid the composition is extracted from.
pub const RESOLUTION: usize = 160;

pub fn load(path: &Path) -> Result<Mesh, String> {
    mesh_from_file(&SdfFile::load(path)?, RESOLUTION)
}

pub fn mesh_from_file(file: &SdfFile, resolution: usize) -> Result<Mesh, String> {
    let raw = raw_mesh_from_file(file, resolution)?;
    Ok(Mesh { positions: raw.positions, triangles: raw.triangles, colors: Vec::new(), uvs: Vec::new() })
}

/// The same mesh with its normals, ready to be written as OBJ or PLY.
pub fn raw_mesh_from_file(file: &SdfFile, resolution: usize) -> Result<RawMesh, String> {
    let fields: Vec<_> = layer_fields(file).into_iter().filter(|field| !field.hidden).collect();
    let mut bounds: Option<([f64; 3], [f64; 3])> = None;
    for field in &fields {
        if let Some((lo, hi)) = field.world_bounds() {
            let (min, max) = bounds.get_or_insert((lo, hi));
            for axis in 0..3 {
                min[axis] = min[axis].min(lo[axis]);
                max[axis] = max[axis].max(hi[axis]);
            }
        }
    }
    let Some((mut lo, mut hi)) = bounds else {
        return Err("this .sdf file has nothing to draw: no visible shapes (splines are not supported)".into());
    };
    let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-6);
    let pad = longest * 0.03;
    for axis in 0..3 {
        lo[axis] -= pad;
        hi[axis] += pad;
    }
    let step = (longest + 2.0 * pad) / resolution.max(8) as f64;
    let dims: [usize; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / step).ceil() as usize + 1);
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let raw = surface_nets(nx, ny, nz, |k| {
        let mut values = vec![0.0f32; nx * ny];
        values.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, value) in row.iter_mut().enumerate() {
                let p = [lo[0] + i as f64 * step, lo[1] + j as f64 * step, lo[2] + k as f64 * step];
                let d = fields.iter().filter_map(|field| field.distance(p)).fold(f64::INFINITY, f64::min);
                *value = d as f32;
            }
        });
        values
    });
    let positions = raw
        .positions
        .iter()
        .map(|p| {
            let file = [lo[0] + p[0] as f64 * step, lo[1] + p[1] as f64 * step, lo[2] + p[2] as f64 * step];
            [file[0] as f32, -file[2] as f32, file[1] as f32]
        })
        .collect();
    let normals = raw.normals.iter().map(|n| [n[0], -n[2], n[1]]).collect();
    Ok(RawMesh { positions, normals, triangles: raw.triangles })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sphere_becomes_a_round_mesh_standing_z_up() {
        // A sphere of radius 1 centred at height 2 (file y), in a layer shifted by one unit along x.
        let json = r#"{"version":[0,6,1],"environment":{},
            "assemblies":[{"resolution":5,"shapes":[{"name":"","hidden":false,"type":"shape","position":[0,2,0],"rotation":[0,0,0],
              "scale":[2,2,2],"color":[1,1,1],"mode":"add","chamfer":false,"blend":0,"blend_strength":0,"mirror_x":false,"mirror_y":false,
              "mirror_z":false,"mirror_blend":0,"brush":"sphere","bevel":0,"round":0.1,"thickness":1,"cone":0,"star":0,"points":6,
              "repeat_mode":"linear","linear_repeats":[1,1,1],"steps":[1.5,1.5,1.5],"radial_repeats":1,"radius":1.5,"repeat_axis":"x"}]}],
            "layers":[{"name":"L","assembly":0,"hidden":false,"position":[1,0,0],"rotation":[0,0,0],"scale":[1,1,1]}]}"#;
        let mesh = mesh_from_file(&SdfFile::from_json(json).unwrap(), 64).unwrap();
        assert!(mesh.triangles.len() > 500);
        for p in &mesh.positions {
            // The centre is file (1, 2, 0), which is (1, 0, 2) Z up.
            let d = ((p[0] - 1.0).powi(2) + p[1].powi(2) + (p[2] - 2.0).powi(2)).sqrt();
            assert!((d - 1.0).abs() < 0.03, "vertex {p:?} is {d} from the centre");
        }
    }

    #[test]
    fn a_sample_file_loads_through_the_mesh_loader() {
        let Ok(home) = std::env::var("USERPROFILE") else { return };
        let path = std::path::Path::new(&home).join("Documents").join("SDF Modeler").join("sdfs").join("robot.sdf");
        if !path.exists() {
            return;
        }
        let mesh = crate::meshformats::load(&path).unwrap();
        assert!(mesh.triangles.len() > 5000 && mesh.positions.len() > 2000, "{} triangles", mesh.triangles.len());
    }

    #[test]
    fn a_file_with_nothing_to_draw_is_an_error() {
        let json = r#"{"version":[0,6,1],"environment":{},"assemblies":[{"resolution":5,"shapes":[]}],
            "layers":[{"name":"L","assembly":0,"hidden":false,"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}]}"#;
        assert!(mesh_from_file(&SdfFile::from_json(json).unwrap(), 32).is_err());
    }

    #[test]
    fn the_exported_mesh_has_outward_normals_and_can_be_written() {
        let json = r#"{"version":[0,6,1],"environment":{},
            "assemblies":[{"resolution":5,"shapes":[{"name":"","hidden":false,"type":"shape","position":[0,2,0],"rotation":[0,0,0],
              "scale":[2,2,2],"color":[1,1,1],"mode":"add","chamfer":false,"blend":0,"blend_strength":0,"mirror_x":false,"mirror_y":false,
              "mirror_z":false,"mirror_blend":0,"brush":"sphere","bevel":0,"round":0.1,"thickness":1,"cone":0,"star":0,"points":6,
              "repeat_mode":"linear","linear_repeats":[1,1,1],"steps":[1.5,1.5,1.5],"radial_repeats":1,"radius":1.5,"repeat_axis":"x"}]}],
            "layers":[{"name":"L","assembly":0,"hidden":false,"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}]}"#;
        let mesh = raw_mesh_from_file(&SdfFile::from_json(json).unwrap(), 48).unwrap();
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        // The centre is file (0, 2, 0), which is (0, 0, 2) Z up; every normal points away from it.
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let out = [p[0], p[1], p[2] - 2.0];
            let dot = out[0] * n[0] + out[1] * n[1] + out[2] * n[2];
            assert!(dot > 0.8, "normal {n:?} at {p:?} points inwards (dot {dot})");
        }
        let dir = std::env::temp_dir();
        for name in ["noiser_export_test.obj", "noiser_export_test.ply"] {
            let path = dir.join(name);
            if name.ends_with("obj") { mesh.write_obj(&path).unwrap() } else { mesh.write_ply(&path).unwrap() }
            assert!(std::fs::metadata(&path).unwrap().len() > 1000);
        }
    }
}
