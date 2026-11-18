//! Automatic UVs for a closed triangle mesh: charts (islands) and a packed atlas, by xatlas, followed by a pass that
//! folds tiny islands into their neighbours. The numbers that say whether the result is good (coverage of the 0 to 1
//! square, island count, tiny islands, overlap, stretch) come from `meshquality::uv_report`.

use crate::mesh::V3;
use crate::surface::RawMesh;
use xatlas_rs_v2 as xatlas;

/// A mesh with a UV per vertex. Vertices are duplicated along island borders, so the vertex count grows.
#[derive(Clone, Debug, Default)]
pub struct UvMesh {
    pub positions: Vec<V3>,
    pub normals: Vec<V3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
}

impl UvMesh {
    /// Wavefront OBJ with vertex normals and texture coordinates (one UV per vertex, so `f v/v/v`).
    pub fn write_obj(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        writeln!(w, "# Noiser")?;
        for p in &self.positions {
            writeln!(w, "v {} {} {}", p[0], p[1], p[2])?;
        }
        for t in &self.uvs {
            writeln!(w, "vt {} {}", t[0], 1.0 - t[1])?;
        }
        for n in &self.normals {
            writeln!(w, "vn {} {} {}", n[0], n[1], n[2])?;
        }
        for t in &self.triangles {
            let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
            writeln!(w, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")?;
        }
        w.flush()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UvParams {

    /// The texture the atlas is packed for, in texels (square).
    pub resolution: u32,
    /// Empty texels around each island, so filtering and mip maps do not bleed across.
    pub padding: u32,
    /// How readily charts grow: higher gives fewer, larger islands with more stretch.
    pub max_cost: f32,
    /// How strongly a chart resists growing across a bend (angle between a face and the chart's average normal).
    pub normal_deviation: f32,
    /// Try harder to pack (slower).
    pub brute_force: bool,
    /// Rounds of smoothing applied to the surface the charts are cut from (the UVs still map the true surface).
    /// A noisy surface is full of small bumps that charts cannot flatten, so it breaks into hundreds of islands.
    pub smoothing: u32,
    /// How many times tiny charts are folded into their neighbours and the atlas rebuilt.
    pub merge_rounds: u32,
}

impl Default for UvParams {
    fn default() -> Self {
        UvParams { resolution: 2048, padding: 4, max_cost: 4.0, normal_deviation: 2.0, brute_force: false, smoothing: 20, merge_rounds: 3 }
    }
}

/// The positions after `rounds` of Taubin smoothing (a shrinking step and a slightly larger growing one, so the shape
/// keeps its volume): the bumps go and the overall form stays.
fn smoothed(mesh: &RawMesh, rounds: u32) -> Vec<V3> {
    let n = mesh.positions.len();
    let mut ring: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in &mesh.triangles {
        for i in 0..3 {
            ring[t[i] as usize].push(t[(i + 1) % 3]);
            ring[t[i] as usize].push(t[(i + 2) % 3]);
        }
    }
    for r in &mut ring {
        r.sort_unstable();
        r.dedup();
    }
    let mut p = mesh.positions.clone();
    for _ in 0..rounds {
        for factor in [0.5f32, -0.53] {
            let next: Vec<V3> = (0..n)
                .map(|v| {
                    if ring[v].is_empty() {
                        return p[v];
                    }
                    let c = ring[v].iter().fold([0.0f32; 3], |s, &w| [s[0] + p[w as usize][0], s[1] + p[w as usize][1], s[2] + p[w as usize][2]]);
                    let k = 1.0 / ring[v].len() as f32;
                    [0, 1, 2].map(|a| p[v][a] + factor * (c[a] * k - p[v][a]))
                })
                .collect();
            p = next;
        }
    }
    p
}

/// One run of xatlas: the UV mesh and, for each chart, the input faces in it. `material` (a label per face) keeps faces
/// with different labels in different charts.
fn run(mesh: &RawMesh, base: &[V3], material: Option<&[u32]>, params: &UvParams, max_cost: f32) -> Result<(UvMesh, Vec<Vec<u32>>), String> {
    let positions: Vec<f32> = base.iter().flatten().copied().collect();
    let indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
    let decl = xatlas::MeshDecl {
        vertex_position_data: xatlas::MeshData::Contiguous(&positions),
        index_data: Some(xatlas::IndexData::U32(&indices)),
        face_material_data: material,
        face_count: mesh.triangles.len() as u32,
        ..Default::default()
    };
    let mut atlas = xatlas::Xatlas::new();
    atlas.add_mesh(&decl).map_err(|e| format!("UV generation: {e:?}"))?;
    let chart = xatlas::ChartOptions { max_cost, normal_deviation_weight: params.normal_deviation, ..Default::default() };
    let pack = xatlas::PackOptions { resolution: params.resolution, padding: params.padding, bilinear: true, brute_force: params.brute_force, rotate_charts: true, rotate_charts_to_axis: true, ..Default::default() };
    atlas.generate(&chart, &pack);
    let (w, h) = (atlas.width().max(1) as f32, atlas.height().max(1) as f32);
    let meshes = atlas.meshes();
    let m = meshes.first().ok_or("UV generation produced no mesh")?;
    let mut out = UvMesh::default();
    for v in &m.vertex_array {
        let src = v.xref as usize;
        out.positions.push(mesh.positions[src]);
        out.normals.push(mesh.normals.get(src).copied().unwrap_or([0.0; 3]));
        out.uvs.push([v.uv[0] / w, v.uv[1] / h]);
    }
    out.triangles = m.index_array.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
        let mut kinds = std::collections::BTreeMap::new();
        for c in &m.chart_array {
            let e = kinds.entry(format!("{:?}", c.type_)).or_insert((0usize, 0usize));
            e.0 += 1;
            e.1 += (c.face_array.len() < 10) as usize;
        }
        eprintln!("charts by type (count, under 10 faces): {kinds:?}");
    }
    let charts = m.chart_array.iter().map(|c| c.face_array.to_vec()).collect();
    Ok((out, charts))
}

/// A label per face with every chart of fewer than `min_faces` faces folded into the neighbouring chart it shares the
/// most border with (repeatedly, smallest first). None when no chart is that small or none has a neighbour.
fn merge_small_charts(mesh: &RawMesh, charts: &[Vec<u32>], min_faces: usize) -> Option<Vec<u32>> {
    use std::collections::HashMap;
    let mut label = vec![0u32; mesh.triangles.len()];
    let mut members: HashMap<u32, Vec<u32>> = HashMap::new();
    for (c, faces) in charts.iter().enumerate() {
        for &f in faces {
            label[f as usize] = c as u32;
        }
        members.insert(c as u32, faces.clone());
    }
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (f, t) in mesh.triangles.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(f as u32);
        }
    }
    let mut changed = false;
    loop {
        let mut small: Vec<(usize, u32)> = members.iter().filter(|(_, f)| f.len() < min_faces).map(|(&l, f)| (f.len(), l)).collect();
        small.sort_unstable();
        let mut progress = false;
        for (_, l) in small {
            let Some(faces) = members.get(&l) else { continue };
            if faces.len() >= min_faces {
                continue;
            }
            let mut border: HashMap<u32, usize> = HashMap::new();
            for &f in faces {
                let t = mesh.triangles[f as usize];
                for i in 0..3 {
                    let (a, b) = (t[i], t[(i + 1) % 3]);
                    for &g in &edges[&(a.min(b), a.max(b))] {
                        if label[g as usize] != l {
                            *border.entry(label[g as usize]).or_default() += 1;
                        }
                    }
                }
            }
            let Some((&target, _)) = border.iter().max_by_key(|(k, v)| (**v, std::cmp::Reverse(**k))) else { continue };
            let moved = members.remove(&l).unwrap();
            for &f in &moved {
                label[f as usize] = target;
            }
            members.get_mut(&target).unwrap().extend(moved);
            progress = true;
            changed = true;
        }
        if !progress {
            break;
        }
    }
    changed.then_some(label)
}

/// UVs for `mesh`; every triangle of the input is a triangle of the output, with the same winding.
pub fn generate(mesh: &RawMesh, params: &UvParams) -> Result<UvMesh, String> {
    if mesh.triangles.is_empty() {
        return Ok(UvMesh::default());
    }
    let base = if params.smoothing > 0 { smoothed(mesh, params.smoothing) } else { mesh.positions.clone() };
    let (mut uv, mut charts) = run(mesh, &base, None, params, params.max_cost)?;
    let min_faces = (mesh.triangles.len() / 400).max(10);
    for _ in 0..params.merge_rounds {
        let Some(labels) = merge_small_charts(mesh, &charts, min_faces) else { break };
        // Charts are re-cut inside each label, and allowed to grow large, so the merged ones stay whole.
        let (u, c) = run(mesh, &base, Some(&labels), params, params.max_cost * 8.0)?;
        (uv, charts) = (u, c);
    }
    Ok(uv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshquality::uv_report;

    fn sphere(subdivisions: u32) -> RawMesh {
        let m = crate::mesh::shapes::icosphere(1.0, subdivisions);
        let normals = m.positions.clone();
        RawMesh { normals, positions: m.positions, triangles: m.triangles }
    }

    #[test]
    fn a_sphere_gets_a_packed_overlap_free_atlas() {
        let mesh = sphere(4); // 5120 triangles
        let uv = generate(&mesh, &UvParams::default()).unwrap();
        assert_eq!(uv.triangles.len(), mesh.triangles.len());
        // Every output vertex is a copy of an input vertex, and the winding is kept.
        let r = uv_report(&uv, 1024);
        eprintln!("{r}");
        assert!(r.overlap_fraction < 1e-4, "{r}");
        assert!(r.coverage > 0.5, "{r}");
        assert!(uv.uvs.iter().all(|t| (0.0..=1.0).contains(&t[0]) && (0.0..=1.0).contains(&t[1])), "UVs leave the 0 to 1 square");
    }
}
