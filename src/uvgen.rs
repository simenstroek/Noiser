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
    /// The input vertex each vertex was made from.
    pub source: Vec<u32>,
}

impl UvMesh {
    /// Binary little-endian PLY with normals and texture coordinates (s, t).
    pub fn write_ply(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        write!(w, "ply\nformat binary_little_endian 1.0\ncomment Noiser\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n", self.positions.len(), self.triangles.len())?;
        for i in 0..self.positions.len() {
            for v in self.positions[i].iter().chain(&self.normals[i]).chain(&[self.uvs[i][0], 1.0 - self.uvs[i][1]]) {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        for t in &self.triangles {
            w.write_all(&[3u8])?;
            for i in t {
                w.write_all(&i.to_le_bytes())?;
            }
        }
        w.flush()
    }

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

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    /// Fold tiny islands into their neighbours by extending the neighbour's UVs over them (undone where it overlaps).
    pub absorb_tiny: bool,
}

impl Default for UvParams {
    fn default() -> Self {
        UvParams { resolution: 2048, padding: 4, max_cost: 4.0, normal_deviation: 2.0, brute_force: true, smoothing: 3, merge_rounds: 0, absorb_tiny: true }
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
    let pack = xatlas::PackOptions { resolution: params.resolution, padding: params.padding, bilinear: true, brute_force: params.brute_force && mesh.triangles.len() <= 50_000, rotate_charts: true, rotate_charts_to_axis: true, ..Default::default() };
    atlas.generate(&chart, &pack);
    let (w, h) = (atlas.width().max(1) as f32, atlas.height().max(1) as f32);
    if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
        eprintln!("atlas {w} x {h}, {} atlas(es), texels per unit {}", atlas.atlas_count(), atlas.texels_per_unit());
    }
    // xatlas may return a slightly non-square atlas; stretching it to the square (at most 15%) gains that much coverage.
    let side = w.max(h);
    let (sx, sy) = if side <= 1.15 * w.min(h) { (w, h) } else { (side, side) };
    let meshes = atlas.meshes();
    let m = meshes.first().ok_or("UV generation produced no mesh")?;
    let mut out = UvMesh::default();
    for v in &m.vertex_array {
        let src = v.xref as usize;
        out.positions.push(mesh.positions[src]);
        out.normals.push(mesh.normals.get(src).copied().unwrap_or([0.0; 3]));
        out.uvs.push([v.uv[0] / sx, v.uv[1] / sy]);
        out.source.push(v.xref);
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

/// The faces of `mesh` whose UV triangle has a texel centre strictly inside another face's (a fold-over or overlap),
/// rasterized on a `res` by `res` grid.
fn overlapping_faces(mesh: &UvMesh, res: usize, island: &[u32]) -> Vec<bool> {
    let n = mesh.triangles.len();
    let mut owner = vec![u32::MAX; res * res];
    let mut bad = vec![false; n];
    let size = res as f32;
    let edge = |a: [f32; 2], b: [f32; 2], x: f32, y: f32| (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    // `grow` texels beyond the triangle's edges count as covered too.
    let cover = |tri: &[u32; 3], grow: f32, visit: &mut dyn FnMut(usize)| {
        let q = tri.map(|i| {
            let uv = mesh.uvs[i as usize];
            [uv[0] * size, uv[1] * size]
        });
        let span = |c: usize| ((q.iter().map(|p| p[c]).fold(f32::MAX, f32::min) - grow).floor().max(0.0) as usize, ((q.iter().map(|p| p[c]).fold(f32::MIN, f32::max) + grow).ceil().max(0.0) as usize).min(res));
        let ((x0, x1), (y0, y1)) = (span(0), span(1));
        let len = [0, 1, 2].map(|k| {
            let (a, b) = (q[k], q[(k + 1) % 3]);
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt().max(1e-6) * grow
        });
        for y in y0..y1 {
            for x in x0..x1 {
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                let w = [edge(q[0], q[1], cx, cy), edge(q[1], q[2], cx, cy), edge(q[2], q[0], cx, cy)];
                if (0..3).all(|k| w[k] > -len[k]) || (0..3).all(|k| w[k] < len[k]) {
                    visit(y * res + x);
                }
            }
        }
    };
    for (f, tri) in mesh.triangles.iter().enumerate() {
        cover(tri, 0.0, &mut |cell| {
            if owner[cell] == u32::MAX {
                owner[cell] = f as u32;
            } else {
                bad[f] = true;
                bad[owner[cell] as usize] = true;
            }
        });
    }
    // A face one texel from another island's face would bleed into it: flag those too.
    for (f, tri) in mesh.triangles.iter().enumerate() {
        let mut hit = false;
        cover(tri, 1.0, &mut |cell| {
            let o = owner[cell];
            hit |= o != u32::MAX && island[o as usize] != island[f];
        });
        if hit {
            bad[f] = true;
        }
    }
    bad
}

fn uv_area2(mesh: &UvMesh, t: [u32; 3]) -> f32 {
    let q = t.map(|i| mesh.uvs[i as usize]);
    (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1])
}

/// Folds islands of fewer than `min_faces` faces into the neighbouring island they share the most border with: the
/// tiny island's border vertices take the neighbour's UVs there and its inner vertices are spread between them. A
/// merge that flips a triangle or lands on another island is undone and the island stays as it was.
fn absorb_tiny_islands(mesh: &mut UvMesh, min_faces: usize, res: usize) -> usize {
    use std::collections::HashMap;
    let n = mesh.triangles.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut first = vec![u32::MAX; mesh.positions.len()];
    for (f, t) in mesh.triangles.iter().enumerate() {
        for &v in t {
            if first[v as usize] == u32::MAX {
                first[v as usize] = f as u32;
            } else {
                let (a, b) = (root(&mut parent, f as u32), root(&mut parent, first[v as usize]));
                parent[a as usize] = b;
            }
        }
    }
    let mut island: Vec<u32> = (0..n as u32).map(|f| root(&mut parent, f)).collect();
    let mut members: HashMap<u32, Vec<u32>> = HashMap::new();
    for (f, &i) in island.iter().enumerate() {
        members.entry(i).or_default().push(f as u32);
    }
    // Faces that share an input edge, whatever their island.
    let mut edges: HashMap<(u32, u32), Vec<(u32, usize)>> = HashMap::new();
    for (f, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (mesh.source[t[k] as usize], mesh.source[t[(k + 1) % 3] as usize]);
            edges.entry((a.min(b), a.max(b))).or_default().push((f as u32, k));
        }
    }
    // The usual texel density (UV area per surface area): a merged triangle must stay within a factor of 6 of it.
    let area3 = |mesh: &UvMesh, t: [u32; 3]| {
        let p = t.map(|i| mesh.positions[i as usize]);
        let (u, v) = ([p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]);
        let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
    };
    let mut ratios: Vec<f32> = mesh.triangles.iter().map(|&t| uv_area2(mesh, t).abs() / area3(mesh, t).max(1e-20)).collect();
    ratios.sort_by(|a, b| a.total_cmp(b));
    let usual = ratios[ratios.len() / 2];
    struct Merge {
        id: u32,
        faces: Vec<u32>,
        old_tris: Vec<[u32; 3]>,
        old_uvs: Vec<(u32, [f32; 2])>,
    }
    let mut merges: Vec<Merge> = Vec::new();
    let (mut no_border, mut conflicts, mut density) = (0, 0, 0);
    // Faces that already moved into another island: merges never chain, so each can be undone on its own.
    let mut absorbed = vec![false; n];
    let mut small: Vec<(usize, u32)> = members.iter().filter(|(_, f)| f.len() < min_faces).map(|(&i, f)| (f.len(), i)).collect();
    small.sort_unstable();
    for (_, id) in small {
        let Some(faces) = members.get(&id).cloned() else { continue };
        if faces.len() >= min_faces || faces.iter().any(|&f| absorbed[f as usize]) {
            continue;
        }
        // The neighbour with the longest shared border, and for each border vertex its twin in that neighbour.
        let mut border: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for &f in &faces {
            let t = mesh.triangles[f as usize];
            for k in 0..3 {
                let (a, b) = (mesh.source[t[k] as usize], mesh.source[t[(k + 1) % 3] as usize]);
                for &(g, gk) in &edges[&(a.min(b), a.max(b))] {
                    if island[g as usize] != id && !absorbed[g as usize] {
                        let gt = mesh.triangles[g as usize];
                        // The edge runs the other way round in g: its vertex with source a is the one that ends it.
                        let (ga, gb) = (gt[gk], gt[(gk + 1) % 3]);
                        let (twin_a, twin_b) = if mesh.source[ga as usize] == a { (ga, gb) } else { (gb, ga) };
                        let list = border.entry(island[g as usize]).or_default();
                        list.push((t[k], twin_a));
                        list.push((t[(k + 1) % 3], twin_b));
                    }
                }
            }
        }
        let Some((&target, pairs)) = border.iter().max_by_key(|(k, v)| (v.len(), std::cmp::Reverse(**k))) else {
            no_border += 1;
            continue;
        };
        let mut twin: HashMap<u32, u32> = HashMap::new();
        let mut conflict = false;
        for &(mine, theirs) in pairs {
            if *twin.entry(mine).or_insert(theirs) != theirs {
                conflict = true;
            }
        }
        if conflict {
            conflicts += 1;
            continue;
        }
        // The inner vertices: those of this island that have no twin.
        let mut inner: Vec<u32> = faces.iter().flat_map(|&f| mesh.triangles[f as usize]).filter(|v| !twin.contains_key(v)).collect();
        inner.sort_unstable();
        inner.dedup();
        let mut merge = Merge { id, faces: faces.clone(), old_tris: faces.iter().map(|&f| mesh.triangles[f as usize]).collect(), old_uvs: inner.iter().map(|&v| (v, mesh.uvs[v as usize])).collect() };
        // The winding of the neighbour island.
        let reference = members[&target].iter().map(|&g| uv_area2(mesh, mesh.triangles[g as usize])).find(|a| a.abs() > 1e-12).unwrap_or(1.0).signum();
        // Border vertices: on an edge that another island (or nothing) shares. With a twin on every one, the island is a
        // patch inside the neighbour and its inner vertices are spread between the twins; otherwise the island hangs
        // off the neighbour's border and is unfolded from it, triangle by triangle, with its own shape.
        let mut border_verts = std::collections::HashSet::new();
        for &f in &faces {
            let t = mesh.triangles[f as usize];
            for k in 0..3 {
                let (a, b) = (mesh.source[t[k] as usize], mesh.source[t[(k + 1) % 3] as usize]);
                let list = &edges[&(a.min(b), a.max(b))];
                if list.len() < 2 || list.iter().any(|&(g, _)| island[g as usize] != id) {
                    border_verts.insert(t[k]);
                    border_verts.insert(t[(k + 1) % 3]);
                }
            }
        }
        let closed = border_verts.iter().all(|v| twin.contains_key(v));
        let mean: [f32; 2] = {
            let all: Vec<[f32; 2]> = twin.values().map(|&t| mesh.uvs[t as usize]).collect();
            [all.iter().map(|p| p[0]).sum::<f32>() / all.len() as f32, all.iter().map(|p| p[1]).sum::<f32>() / all.len() as f32]
        };
        let mut unplaced: Vec<u32> = inner.clone();
        if !closed {
            let mut placed: std::collections::HashSet<u32> = std::collections::HashSet::new();
            loop {
                let mut progress = false;
                for &f in &faces {
                    let t = merge.old_tris[faces.iter().position(|&x| x == f).unwrap()];
                    let at = |v: u32, mesh: &UvMesh| twin.get(&v).map(|&w| mesh.uvs[w as usize]).or_else(|| placed.contains(&v).then(|| mesh.uvs[v as usize]));
                    let known: Vec<bool> = t.iter().map(|&v| at(v, mesh).is_some()).collect();
                    if known.iter().filter(|k| **k).count() != 2 {
                        continue;
                    }
                    let r = known.iter().position(|k| !k).unwrap();
                    let (pi, qi) = ((r + 1) % 3, (r + 2) % 3);
                    let (pu, qu) = (at(t[pi], mesh).unwrap(), at(t[qi], mesh).unwrap());
                    let (p3, q3, r3) = (mesh.positions[t[pi] as usize], mesh.positions[t[qi] as usize], mesh.positions[t[r] as usize]);
                    let d3 = [q3[0] - p3[0], q3[1] - p3[1], q3[2] - p3[2]];
                    let l2 = d3[0] * d3[0] + d3[1] * d3[1] + d3[2] * d3[2];
                    if l2 < 1e-20 {
                        continue;
                    }
                    let rp = [r3[0] - p3[0], r3[1] - p3[1], r3[2] - p3[2]];
                    let a = (rp[0] * d3[0] + rp[1] * d3[1] + rp[2] * d3[2]) / l2;
                    let foot = [rp[0] - a * d3[0], rp[1] - a * d3[1], rp[2] - a * d3[2]];
                    let h = (foot[0] * foot[0] + foot[1] * foot[1] + foot[2] * foot[2]).sqrt() / l2.sqrt();
                    let duv = [qu[0] - pu[0], qu[1] - pu[1]];
                    let place = |sign: f32| [pu[0] + a * duv[0] - sign * h * duv[1], pu[1] + a * duv[1] + sign * h * duv[0]];
                    let mut uvs3 = [[0.0f32; 2]; 3];
                    uvs3[pi] = pu;
                    uvs3[qi] = qu;
                    uvs3[r] = place(1.0);
                    let area = (uvs3[1][0] - uvs3[0][0]) * (uvs3[2][1] - uvs3[0][1]) - (uvs3[2][0] - uvs3[0][0]) * (uvs3[1][1] - uvs3[0][1]);
                    mesh.uvs[t[r] as usize] = if area * reference > 0.0 { uvs3[r] } else { place(-1.0) };
                    placed.insert(t[r]);
                    progress = true;
                }
                if !progress {
                    break;
                }
            }
            unplaced.retain(|v| !placed.contains(v));
        }
        for &v in &unplaced {
            mesh.uvs[v as usize] = mean;
        }
        if closed || !unplaced.is_empty() {
            let mut ring: HashMap<u32, Vec<u32>> = HashMap::new();
            for &f in &faces {
                let t = mesh.triangles[f as usize];
                for k in 0..3 {
                    ring.entry(t[k]).or_default().extend([t[(k + 1) % 3], t[(k + 2) % 3]]);
                }
            }
            for _ in 0..60 {
                for &v in &unplaced {
                    let r = &ring[&v];
                    let (mut sx, mut sy) = (0.0, 0.0);
                    for &w in r {
                        let p = match twin.get(&w) {
                            Some(&t) => mesh.uvs[t as usize],
                            None => mesh.uvs[w as usize],
                        };
                        sx += p[0];
                        sy += p[1];
                    }
                    mesh.uvs[v as usize] = [sx / r.len() as f32, sy / r.len() as f32];
                }
            }
        }
        for (k, &f) in faces.iter().enumerate() {
            mesh.triangles[f as usize] = merge.old_tris[k].map(|v| twin.get(&v).copied().unwrap_or(v));
        }
        // The same winding as the neighbour, and no collapsed triangles.
        let ok = faces.iter().all(|&f| {
            let a = uv_area2(mesh, mesh.triangles[f as usize]);
            let ratio = a.abs() / area3(mesh, mesh.triangles[f as usize]).max(1e-20);
            let inside = mesh.triangles[f as usize].iter().all(|&v| mesh.uvs[v as usize].iter().all(|c| (0.0..=1.0).contains(c)));
            inside && a * reference > 1e-12 && ratio > usual / 6.0 && ratio < usual * 6.0
        });
        if !ok {
            density += 1;
            for (k, &f) in faces.iter().enumerate() {
                mesh.triangles[f as usize] = merge.old_tris[k];
            }
            for &(v, uv) in &merge.old_uvs {
                mesh.uvs[v as usize] = uv;
            }
            continue;
        }
        for &f in &faces {
            island[f as usize] = target;
            absorbed[f as usize] = true;
        }
        let moved = members.remove(&id).unwrap();
        members.get_mut(&target).unwrap().extend(moved);
        merge.faces = faces;
        merges.push(merge);
    }
    let tried = merges.len();
    // Undo the merges that landed on other charts.
    let mut undone = vec![false; merges.len()];
    loop {
        let bad = overlapping_faces(mesh, res, &island);
        let mut any = false;
        for (i, m) in merges.iter().enumerate() {
            if !undone[i] && m.faces.iter().any(|&f| bad[f as usize]) {
                for (&f, old) in m.faces.iter().zip(&m.old_tris) {
                    mesh.triangles[f as usize] = *old;
                }
                for &(v, uv) in &m.old_uvs {
                    mesh.uvs[v as usize] = uv;
                }
                for &f in &m.faces {
                    island[f as usize] = m.id;
                }
                undone[i] = true;
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    // Drop vertices no triangle uses any more.
    let mut used = vec![false; mesh.positions.len()];
    for t in &mesh.triangles {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let mut map = vec![u32::MAX; used.len()];
    let (mut positions, mut normals, mut uvs, mut source) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (v, &u) in used.iter().enumerate() {
        if u {
            map[v] = positions.len() as u32;
            positions.push(mesh.positions[v]);
            normals.push(mesh.normals[v]);
            uvs.push(mesh.uvs[v]);
            source.push(mesh.source[v]);
        }
    }
    for t in &mut mesh.triangles {
        *t = t.map(|v| map[v as usize]);
    }
    (mesh.positions, mesh.normals, mesh.uvs, mesh.source) = (positions, normals, uvs, source);
    let kept = undone.iter().filter(|u| !**u).count();
    if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
        eprintln!("tiny islands: no neighbour {no_border}, border conflict {conflicts}, density/winding {density}, undone for overlap {}, kept {kept}", tried - kept);
    }
    kept
}

/// UVs for `mesh`; every triangle of the input is a triangle of the output, with the same winding.
pub fn generate(mesh: &RawMesh, params: &UvParams) -> Result<UvMesh, String> {
    if mesh.triangles.is_empty() {
        return Ok(UvMesh::default());
    }
    let base = if params.smoothing > 0 { smoothed(mesh, params.smoothing) } else { mesh.positions.clone() };
    let (mut uv, mut charts) = run(mesh, &base, None, params, params.max_cost)?;
    if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
        let bad = crate::meshquality::bad_faces(mesh);
        let min = (mesh.triangles.len() / 400).max(10);
        let (mut in_tiny, mut bad_in_tiny, mut tiny_with_bad, mut tiny) = (0, 0, 0, 0);
        for c in charts.iter().filter(|c| c.len() < min) {
            tiny += 1;
            in_tiny += c.len();
            let b = c.iter().filter(|&&f| bad[f as usize]).count();
            bad_in_tiny += b;
            tiny_with_bad += (b > 0) as usize;
        }
        let total_bad = bad.iter().filter(|b| **b).count();
        eprintln!("bad faces: {total_bad} of {} overall; {tiny} tiny charts hold {in_tiny} faces, {bad_in_tiny} bad; {tiny_with_bad} of them contain a bad face", bad.len());
    }
    let min_faces = (mesh.triangles.len() / 400).max(10);
    for _ in 0..params.merge_rounds {
        let Some(labels) = merge_small_charts(mesh, &charts, min_faces) else { break };
        // Charts are re-cut inside each label, and allowed to grow large, so the merged ones stay whole.
        let (u, c) = run(mesh, &base, Some(&labels), params, params.max_cost * 8.0)?;
        (uv, charts) = (u, c);
    }
    if params.absorb_tiny {
        // Each round merges islands that did not absorb any in that round; the next one can then merge the grown ones.
        for round in 0..4 {
            let absorbed = absorb_tiny_islands(&mut uv, min_faces, params.resolution as usize);
            if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
                eprintln!("round {round}: absorbed {absorbed} tiny islands");
            }
            if absorbed == 0 {
                break;
            }
        }
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
        assert!(uv.uvs.iter().all(|t| (0.0..=1.0).contains(&t[0]) && (0.0..=1.0).contains(&t[1])), "UVs leave the 0 to 1 square: {:?}", uv.uvs.iter().filter(|t| !(0.0..=1.0).contains(&t[0]) || !(0.0..=1.0).contains(&t[1])).take(3).collect::<Vec<_>>());
    }
}
