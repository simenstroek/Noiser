//! Terrain chunks in the Mesh graph: a heightmap through Load Mesh and Remesh. `terrain_chunks_meet_and_follow_the_ground`
//! is the quick check; `terrain_chunk_scaling` is a timing run for finding bottlenecks (ignored; see its comment).

use crate::gpu_tests::gpu;
use crate::graph::{Graph, MathOp, NodeKind, NoiseKind, NoiseParams};
use crate::meshgraph::{LoadMeshParams, MeshCache, OutParams, Runner};
use std::path::{Path, PathBuf};

fn hash(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h & 0xFFFF) as f32 / 65535.0
}

/// Smooth value noise.
fn vnoise(x: f32, y: f32) -> f32 {
    let (i, j, fx, fy) = (x.floor() as i32, y.floor() as i32, x.fract(), y.fract());
    let (u, v) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = hash(i, j) + (hash(i + 1, j) - hash(i, j)) * u;
    let b = hash(i, j + 1) + (hash(i + 1, j + 1) - hash(i, j + 1)) * u;
    a + (b - a) * v
}

/// A made-up mountain range of `world_m` metres square and about `height` metres tall, written the way the Landscape tab
/// exports it.
fn heightmap(dir: &Path, n: usize, world_m: f32, height: f32) -> PathBuf {
    let metres: Vec<f32> = (0..n * n)
        .map(|k| {
            let (x, y) = ((k % n) as f32 / (n - 1) as f32, (k / n) as f32 / (n - 1) as f32);
            let mut v = 0.0;
            let mut a = 0.5;
            let mut f = 3.0;
            for _ in 0..6 {
                v += a * (1.0 - (2.0 * vnoise(x * f, y * f) - 1.0).abs());
                a *= 0.5;
                f *= 2.0;
            }
            v * height
        })
        .collect();
    let path = dir.join("terrain.png");
    crate::landscape::write_unreal(&path, n, world_m, &crate::landscape::encode_unreal(&metres)).unwrap();
    path
}

/// Load Mesh (the heightmap, in metres) -> optional rough noise -> Remesh -> planar UVs -> OBJ.
fn graph(png: &Path, chunks: u32, triangles: u32, detail: bool) -> Graph {
    graph_with(png, chunks, triangles, detail, None)
}

/// `graph`, with a Bake of these maps at this size when asked.
fn graph_with(png: &Path, chunks: u32, triangles: u32, detail: bool, bake: Option<(u32, Vec<crate::bake::Channel>)>) -> Graph {
    let mut g = Graph::default();
    let land = g.add(NodeKind::LoadMesh(LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 100.0, chunks, ..Default::default() }), [0.0; 2]);
    let mut field = land;
    if detail {
        let mut n = Graph::default();
        let noise = n.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Perlin, scale: 1.0, octaves: 3, seed: Some(3), ..Default::default() }), [0.0; 2]);
        let out = n.add(NodeKind::OutDensity, [0.0; 2]);
        n.connect(noise, out, 0);
        let rough = g.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { embedded: n.to_json(), frequency: 0.05, amplitude: 3.0, ..Default::default() }), [0.0; 2]);
        let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(land, sum, 0);
        g.connect(rough, sum, 1);
        field = sum;
    }
    let remesh = g.add(NodeKind::Remesh { triangles, adaptivity: 1.0 }, [0.0; 2]);
    let uv = g.add(NodeKind::AutoUv(crate::uvgen::UvParams { planar: true, ..Default::default() }), [0.0; 2]);
    let out = g.add(NodeKind::OutMesh(OutParams { folder: "out".into(), name: "terrain".into(), ..Default::default() }), [0.0; 2]);
    g.connect(field, remesh, 0);
    g.connect(remesh, uv, 0);
    g.connect(uv, out, 0);
    if let Some((size, channels)) = bake {
        let b = g.add(NodeKind::Bake(crate::bake::Settings { size, channels, samples: 1, ..Default::default() }), [0.0; 2]);
        let img = g.add(NodeKind::OutImage(OutParams { folder: "out".into(), name: "terrain".into(), ..Default::default() }), [0.0; 2]);
        g.connect(uv, b, 0);
        g.connect(field, b, 1);
        g.connect_from(b, 0, img, 0);
    }
    g
}

fn vertices(path: &Path) -> Vec<[String; 3]> {
    std::fs::read_to_string(path).unwrap().lines().filter(|l| l.starts_with("v ")).map(|l| {
        let p: Vec<&str> = l.split_whitespace().collect();
        [p[1].to_string(), p[2].to_string(), p[3].to_string()]
    }).collect()
}

fn run(dir: &Path, g: &Graph) -> (Vec<PathBuf>, Vec<String>) {
    let gpu = gpu().expect("a GPU");
    let mut cache = MeshCache::default();
    let mut log = Vec::new();
    let mut note = |s: &str| log.push(s.to_string());
    let written = Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.to_path_buf(), seed: 0, progress: Default::default(), log: &mut note, batch_file: None, max_triangles: None }.run(g).unwrap();
    (written, log)
}

#[test]
fn terrain_chunks_meet_and_follow_the_ground() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 60.0);
    let (written, log) = run(&dir, &graph(&png, 2, 8000, true));
    assert_eq!(written.len(), 4, "{written:?} {log:?}");
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("out").join("terrain_chunks.json")).unwrap()).unwrap();
    assert_eq!(manifest["chunks"].as_array().unwrap().len(), 4);
    assert_eq!(manifest["chunks"][1]["column"], 1);
    let name = |cy: usize, cx: usize| dir.join("out").join(format!("terrain_{cy:02}_{cx:02}.obj"));
    let (a, b) = (vertices(&name(0, 0)), vertices(&name(0, 1)));
    assert!(a.len() > 1500 && b.len() > 1500, "{} {}", a.len(), b.len());
    // q = 64 quads over 320 m: cells are 5 m, and the column of cells just before x = 320 m is in both files. The first
    // chunk also owns the quads of the x edges there, which the second does not, so the second's vertices are a subset.
    // Their coordinates come from different origins: equal to a few millionths of a metre, not bit for bit.
    let column = |v: &[[String; 3]]| -> Vec<[f32; 3]> { v.iter().map(|p| p.clone().map(|c| c.parse::<f32>().unwrap())).filter(|p| (315.0..319.999).contains(&p[0])).collect() };
    let (ca, cb) = (column(&a), column(&b));
    assert!(cb.len() > 40, "{}", cb.len());
    let unmatched = cb.iter().filter(|q| !ca.iter().any(|p| (0..3).all(|k| (p[k] - q[k]).abs() < 1e-3))).count();
    assert_eq!(unmatched, 0, "vertices of the second chunk that the first lacks along the shared border");
    // Planar UVs fill the square.
    let obj = std::fs::read_to_string(name(0, 0)).unwrap();
    let uvs: Vec<[f32; 2]> = obj.lines().filter(|l| l.starts_with("vt ")).map(|l| { let p: Vec<f32> = l.split_whitespace().skip(1).map(|v| v.parse().unwrap()).collect(); [p[0], p[1]] }).collect();
    assert!(uvs.len() > 1500 && uvs.iter().all(|u| (0.0..=1.0).contains(&u[0]) && (0.0..=1.0).contains(&u[1])));
}

/// The graph the Landscape tab's button opens runs as it is: chunk meshes, three maps each, and the manifest.
#[test]
fn the_graph_the_landscape_tab_sends_runs_as_it_is() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-send");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 1000.0, 80.0);
    let mut g = crate::landscape_field::terrain_graph(&png, 1000.0);
    // Keep the test quick: smaller budget and maps.
    for n in g.nodes.values_mut() {
        match &mut n.kind {
            NodeKind::Remesh { triangles, .. } => *triangles = 20_000,
            NodeKind::Bake(b) => (b.size, b.samples) = (128, 1),
            _ => {}
        }
    }
    let (written, log) = run(&dir, &g);
    // 2 x 2 chunks of 500 m, each a mesh and three maps, plus the manifest (written separately).
    assert_eq!(written.len(), 16, "{log:?}");
    let out = dir.join("chunks");
    for f in ["terrain_00_00.obj", "terrain_01_01_normal_base.png", "terrain_00_01_ambient_occlusion.png", "terrain_01_00_curvature.png", "terrain_chunks.json"] {
        assert!(out.join(f).exists(), "{f} missing: {log:?}");
    }
}

/// A rock (a mesh file) standing on the terrain joins the surface with a smooth union, and lifts the chunk's z range.
#[test]
fn an_object_on_the_terrain_is_part_of_its_chunk() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-object");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 40.0);
    // A 30 m ball 70 m up in the first chunk; the terrain never goes above 40 m, so anything above that is the ball.
    let ball = crate::mesh::shapes::icosphere(30.0, 3);
    let moved: Vec<[f32; 3]> = ball.positions.iter().map(|p| [p[0] + 160.0, p[1] + 160.0, p[2] + 70.0]).collect();
    crate::surface::RawMesh { normals: ball.positions.clone(), positions: moved, triangles: ball.triangles }.write_obj(&dir.join("ball.obj")).unwrap();
    let mut g = graph(&png, 2, 20000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    let rock = g.add(NodeKind::LoadMesh(LoadMeshParams { path: "ball.obj".into(), unit_cm: 100.0, resolution: 96, ..Default::default() }), [0.0; 2]);
    let union = g.add(NodeKind::Math { op: MathOp::SmoothMin, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    g.connect(land, union, 0);
    g.connect(rock, union, 1);
    g.connect(union, remesh, 0);
    let (written, log) = run(&dir, &g);
    assert_eq!(written.len(), 4, "{log:?}");
    let top = |cy: usize, cx: usize| vertices(&dir.join("out").join(format!("terrain_{cy:02}_{cx:02}.obj"))).iter().map(|v| v[2].parse::<f32>().unwrap()).fold(f32::MIN, f32::max);
    assert!(top(0, 0) > 85.0, "the ball's top is at 100 m: {}", top(0, 0));
    assert!(top(0, 1) < 45.0, "the far chunk stays ground: {}", top(0, 1));
}

/// Time and size for growing chunks: `cargo test --release terrain_chunk_scaling -- --ignored --nocapture`.
/// Set `NOISER_SCALE_QUADS` to a comma separated list of quads per side to try others.
#[test]
#[ignore]
fn terrain_chunk_scaling() {
    let Some(_) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::env::temp_dir().join("noiser-terrain-scale");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let t = std::time::Instant::now();
    let png = heightmap(&dir, 4033, 2000.0, 250.0);
    eprintln!("heightmap made in {:.1} s", t.elapsed().as_secs_f32());
    let sizes: Vec<u32> = std::env::var("NOISER_SCALE_QUADS").ok().map_or(vec![100, 200, 350, 500, 707], |s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect());
    for detail in [false, true] {
        for q in &sizes {
            let tris = 2 * q * q;
            let t = std::time::Instant::now();
            let (_, log) = run(&dir, &graph(&png, 4, tris, detail));
            eprintln!("detail {detail}: {q} x {q} quads ({tris} triangles) per chunk, 16 chunks: {:.1} s", t.elapsed().as_secs_f32());
            for l in log.iter().filter(|l| l.starts_with("terrain chunk")).take(2) {
                eprintln!("    {l}");
            }
        }
    }
}

/// One chunk with maps: `cargo test --release terrain_chunk_bake -- --ignored --nocapture`. `NOISER_SCALE_QUADS` and
/// `NOISER_SCALE_MAP` (texels) pick the size.
#[test]
#[ignore]
fn terrain_chunk_bake() {
    use crate::bake::Channel;
    let Some(_) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::env::temp_dir().join("noiser-terrain-bake");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 1025, 500.0, 120.0);
    let q: u32 = std::env::var("NOISER_SCALE_QUADS").ok().and_then(|v| v.parse().ok()).unwrap_or(350);
    let map: u32 = std::env::var("NOISER_SCALE_MAP").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    for channels in [vec![Channel::Normal], vec![Channel::Occlusion], vec![Channel::Curvature]] {
        let t = std::time::Instant::now();
        let (_, log) = run(&dir, &graph_with(&png, 1, 2 * q * q, true, Some((map, channels.clone()))));
        eprintln!("{q} x {q} quads, {map} px, {channels:?}: {:.1} s", t.elapsed().as_secs_f32());
        for l in &log {
            eprintln!("    {l}");
        }
    }
}
