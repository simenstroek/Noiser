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
    let remesh = g.add(NodeKind::Remesh { triangles, adaptivity: 1.0, mesher: Default::default(), auto_detail: false }, [0.0; 2]);
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
    let written = Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.to_path_buf(), seed: 0, progress: Default::default(), log: &mut note, batch_file: None, max_triangles: None, tags: Default::default() }.run(g).unwrap();
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
    // Noise is added to the whole field, and must not stand a wall of surface along the outer border down to the bottom of the lattice.
    let below = a.iter().filter(|v| v[2].parse::<f32>().unwrap() < -5.0).count();
    assert!(below == 0, "no wall stands below the ground along the border: {below} of {} vertices", a.len());
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

/// The bare terrain has a surface of its own (not only the noise and shapes added to it), ending at its square.
#[test]
fn the_bare_terrain_is_a_surface_that_ends_at_its_square() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-bare");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 30.0);
    let (_, log) = run(&dir, &graph(&png, 1, 20000, false));
    let obj = std::fs::read_to_string(dir.join("out").join("terrain_00_00.obj").with_file_name("terrain.obj")).unwrap();
    let faces = obj.lines().filter(|l| l.starts_with("f ")).count();
    assert!(faces > 15000, "{faces} triangles {log:?}");
    let v = vertices(&dir.join("out").join("terrain.obj"));
    let (xmax, zmin) = v.iter().fold((f32::MIN, f32::MAX), |(x, z), p| (x.max(p[0].parse().unwrap()), z.min(p[2].parse().unwrap())));
    assert!(xmax < 640.0 + 6.0, "the terrain ends at 640 m: {xmax}");
    // The surface ends at the border: no wall under it.
    assert!(zmin > -6.0, "no surface below the ground: {zmin}");
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

/// A Load SDF node (live, placed by size and offset in the terrain's units) joins the terrain like a mesh does.
#[test]
fn a_placed_sdf_shape_is_part_of_its_chunk() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-sdf");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 40.0);
    let sdf = crate::sdf_node::tests::temp_sphere("noiser_terrain_sdf.sdf");
    let mut g = graph(&png, 2, 20000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    // Metres: a shape 60 m across, its centre 70 m up over (160, 160).
    let shape = g.add(NodeKind::LoadSdf(crate::sdf_node::LoadSdfParams { file: sdf, size: 60.0, offset: [160.0, 160.0, 70.0], ..Default::default() }), [0.0; 2]);
    let union = g.add(NodeKind::Math { op: MathOp::SmoothMin, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    g.connect(land, union, 0);
    g.connect(shape, union, 1);
    g.connect(union, remesh, 0);
    let (written, log) = run(&dir, &g);
    assert_eq!(written.len(), 4, "{log:?}");
    let top = |cy: usize, cx: usize| vertices(&dir.join("out").join(format!("terrain_{cy:02}_{cx:02}.obj"))).iter().map(|v| v[2].parse::<f32>().unwrap()).fold(f32::MIN, f32::max);
    assert!(top(0, 0) > 85.0, "the shape's top is at 100 m: {}", top(0, 0));
    assert!(top(0, 1) < 45.0, "the far chunk stays ground: {}", top(0, 1));
}

/// "Stand on the terrain" ignores the offset's height and puts the shape's base on the ground.
#[test]
fn a_shape_can_stand_on_the_terrain() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-snap");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 40.0);
    let sdf = crate::sdf_node::tests::temp_sphere("noiser_terrain_snap.sdf");
    let mut g = graph(&png, 2, 20000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    let shape = g.add(NodeKind::LoadSdf(crate::sdf_node::LoadSdfParams { file: sdf, size: 40.0, offset: [160.0, 160.0, 5000.0], snap: true, ..Default::default() }), [0.0; 2]);
    let union = g.add(NodeKind::Math { op: MathOp::Min, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    g.connect(land, union, 0);
    g.connect(shape, union, 1);
    g.connect(union, remesh, 0);
    let (_, log) = run(&dir, &g);
    let top = vertices(&dir.join("out").join("terrain_00_00.obj")).iter().map(|v| v[2].parse::<f32>().unwrap()).fold(f32::MIN, f32::max);
    // The ground there is under 40 m and the ball is 40 m across: its top is above 40 m but nowhere near the 5000 m offset.
    assert!(top > 45.0 && top < 90.0, "{top} {log:?}");
}

/// A steep ramp (rising 0.8 per metre along x) with a cave scattered into it, facing uphill: the cave is a hollow inside the hill
/// with its mouth on the slope, so the surface has walls below the ramp's own surface.
#[test]
fn a_cave_scattered_onto_a_slope_hollows_it_out() {
    if gpu().is_none() || !Path::new("H:/My Drive/NoiserLibrary/SDFs/Cave.sdf").exists() {
        return eprintln!("no GPU adapter or no library cave; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-cave");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let n = 129;
    let metres: Vec<f32> = (0..n * n).map(|k| 0.8 * (k % n) as f32 * 5.0 + 20.0).collect();
    let png = dir.join("terrain.png");
    crate::landscape::write_unreal(&png, n, 640.0, &crate::landscape::encode_unreal(&metres)).unwrap();
    let mut g = graph(&png, 1, 60000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    // 60 m caves in cells of 320 m: two across, the slope between 0.7 and 0.9.
    let scatter = g.add(NodeKind::Scatter(crate::scatter::ScatterParams { file: "H:/My Drive/NoiserLibrary/SDFs/Cave.sdf".into(), cell: 320.0, density: 1.0, size: [60.0, 60.0], jitter: 0.0, slope: [0.5, 2.0], face_uphill: true, subtract: true, sink: 0.1, blend: 1.0, ..Default::default() }), [0.0; 2]);
    g.connect(land, scatter, 0);
    g.connect(scatter, remesh, 0);
    let (_, log) = run(&dir, &g);
    let v = vertices(&dir.join("out").join("terrain.obj"));
    // Vertices more than 3 m under the ramp's surface are cavity walls.
    let inside = v.iter().filter(|p| { let (x, z): (f32, f32) = (p[0].parse().unwrap(), p[2].parse().unwrap()); z < 0.8 * x + 20.0 - 3.0 }).count();
    assert!(inside > 150, "{inside} of {} vertices lie inside the hill: {log:?}", v.len());
}

/// A ring: four copies evenly round the terrain's middle, and none in the middle.
#[test]
fn a_scatter_ring_stands_round_the_middle_of_the_terrain() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-ring");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 20.0);
    let sdf = crate::sdf_node::tests::temp_sphere("noiser_terrain_ring.sdf");
    let mut g = graph(&png, 1, 60000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    // Four 40 m balls on a circle of 200 m round the middle (320, 320), the first at 45 degrees.
    let ring = g.add(NodeKind::Scatter(crate::scatter::ScatterParams { file: sdf, ring: true, ring_count: 4, ring_radius: 200.0, ring_start: 45.0, size: [40.0, 40.0], slope: [0.0, 10.0], blend: 1.0, face_centre: true, ..Default::default() }), [0.0; 2]);
    g.connect(land, ring, 0);
    g.connect(ring, remesh, 0);
    let (_, log) = run(&dir, &g);
    let v: Vec<[f32; 3]> = vertices(&dir.join("out").join("terrain.obj")).iter().map(|p| [p[0].parse().unwrap(), p[1].parse().unwrap(), p[2].parse().unwrap()]).collect();
    let tall: Vec<&[f32; 3]> = v.iter().filter(|p| p[2] > 38.0).collect();
    assert!(tall.len() > 200, "{} vertices above the ground: {log:?}", tall.len());
    for k in 0..4 {
        let a = std::f32::consts::FRAC_PI_4 + std::f32::consts::FRAC_PI_2 * k as f32;
        let at = [320.0 + 200.0 * a.cos(), 320.0 + 200.0 * a.sin()];
        assert!(tall.iter().any(|p| (p[0] - at[0]).hypot(p[1] - at[1]) < 20.0), "a copy stands at {at:?}");
    }
    assert!(tall.iter().all(|p| (p[0] - 320.0).hypot(p[1] - 320.0) > 120.0), "nothing stands in the middle");
}

/// A Scatter SDF node plants copies of a shape on the ground by a rule.
#[test]
fn a_scatter_node_plants_shapes_on_the_terrain() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-terrain-scatter");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = heightmap(&dir, 129, 640.0, 20.0);
    let sdf = crate::sdf_node::tests::temp_sphere("noiser_terrain_scatter.sdf");
    let mut g = graph(&png, 1, 40000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    // Metres: a cell of 160 m with a copy in every cell, 30 to 40 m across: the ground is under 20 m, so the copies stand above it.
    let scatter = g.add(NodeKind::Scatter(crate::scatter::ScatterParams { file: sdf, cell: 160.0, density: 1.0, size: [30.0, 40.0], slope: [0.0, 10.0], blend: 2.0, ..Default::default() }), [0.0; 2]);
    g.connect(land, scatter, 0);
    g.connect(scatter, remesh, 0);
    let (_, log) = run(&dir, &g);
    let v = vertices(&dir.join("out").join("terrain.obj"));
    let top = v.iter().map(|v| v[2].parse::<f32>().unwrap()).fold(f32::MIN, f32::max);
    assert!(v.len() > 1000, "{} vertices {log:?}", v.len());
    assert!(top > 35.0 && top < 80.0, "the copies reach above the 20 m ground but not far: {top} {log:?}");
    // Cut out instead, the same shapes leave pits: the surface is ground, nothing above it, and more of it than before.
    let NodeKind::Scatter(p) = &mut g.nodes.get_mut(&scatter).unwrap().kind else { panic!() };
    p.subtract = true;
    p.density = 0.0;
    p.sink = 0.3;
    p.size = [60.0, 70.0];
    let (_, log) = run(&dir, &g);
    let pits = vertices(&dir.join("out").join("terrain.obj"));
    let high = pits.iter().map(|v| v[2].parse::<f32>().unwrap()).fold(f32::MIN, f32::max);
    assert!(pits.len() > 1000 && high < 25.0, "{} vertices, top {high} {log:?}", pits.len());
}

/// The field view of a terrain with scattered copies, traced from straight above and from beside it.
#[test]
fn the_field_view_shows_a_scattered_terrain_from_any_side_and_no_copies_past_its_edge() {
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let gpu = gpu().unwrap();
    let dir = std::env::temp_dir().join("noiser-terrain-field-view");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // 640 m square, ground under 20 m, a copy 30 to 40 m across in every 80 m cell.
    let png = heightmap(&dir, 129, 640.0, 20.0);
    let sdf = crate::sdf_node::tests::temp_sphere("noiser_terrain_field_view.sdf");
    let mut g = graph(&png, 1, 40000, false);
    let land = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::LoadMesh(_))).map(|(id, _)| *id).unwrap();
    let remesh = g.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::Remesh { .. })).map(|(id, _)| *id).unwrap();
    let scatter = g.add(NodeKind::Scatter(crate::scatter::ScatterParams { file: sdf, cell: 80.0, density: 1.0, size: [30.0, 40.0], slope: [0.0, 10.0], blend: 2.0, ..Default::default() }), [0.0; 2]);
    g.connect(land, scatter, 0);
    g.connect(scatter, remesh, 0);
    let mut cache = MeshCache::default();
    let mut quiet = |_: &str| {};
    let view = Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.clone(), seed: 0, progress: Default::default(), log: &mut quiet, batch_file: None, max_triangles: None, tags: Default::default() }.field_view(&g).unwrap();
    let clip = view.trace_box();
    assert!(clip.is_some(), "a terrain's field view traces only inside its box");
    let n = 128u32;
    let camera = |origin: [f32; 3], step_x: [f32; 3], step_y: [f32; 3], direction: [f32; 3]| crate::meshexport::MarchCamera { origin, step_x, step_y, direction, light: [0.3, 0.4, 0.86], t_max: 4000.0, hit: 0.05, gradient_step: 0.5, clip };
    // From above, over a 1280 m square centred on the terrain: x and y from -320 to 960 m, 10 m a pixel.
    let top = view.march(&camera([-315.0, -315.0, 1000.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, -1.0]), n, n);
    let (mut inside, mut outside) = (0, 0);
    for (k, px) in top.iter().enumerate() {
        let (x, y) = (-315.0 + 10.0 * (k as u32 % n) as f32, -315.0 + 10.0 * (k as u32 / n) as f32);
        let on = (-5.0..=645.0).contains(&x) && (-5.0..=645.0).contains(&y);
        if px[3] > 0.5 {
            if on { inside += 1 } else { outside += 1 }
        }
    }
    assert!(inside > 3000, "the terrain fills its square from above: {inside} pixels");
    assert_eq!(outside, 0, "nothing stands past the terrain's edge");
    // From the side, level with the ground (looking along +y from beyond the square): the terrain and its copies show.
    let side = view.march(&camera([-315.0, -2000.0, -100.0], [10.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0]), n, n);
    let seen = side.iter().filter(|p| p[3] > 0.5).count();
    assert!(seen > 500, "the terrain is seen from the side too: {seen} pixels");
    // From straight below, the ground is a sheet, not a solid block with a floor: nothing shows.
    let below = view.march(&camera([-315.0, 960.0 - 5.0, -1000.0], [10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 1.0]), n, n);
    let under = below.iter().filter(|p| p[3] > 0.5).count();
    assert_eq!(under, 0, "no floor under the terrain");
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

/// Timing of the real demo graph (needs the library's DemoTerrain.meshgraph): `cargo test --release scratch_demo_compile -- --ignored --nocapture`.
#[test]
#[ignore]
fn scratch_demo_compile() {
    let text = std::fs::read_to_string("H:/My Drive/NoiserLibrary/Mesh Graphs/DemoTerrain.meshgraph").unwrap();
    let mut g = Graph::from_json(&text).unwrap();
    for n in g.nodes.values_mut() {
        match &mut n.kind {
            NodeKind::LoadMesh(p) => p.chunks = 1,
            NodeKind::Remesh { triangles, .. } => *triangles = 20_000,
            NodeKind::Bake(b) => (b.size, b.samples) = (64, 1),
            NodeKind::OutMesh(o) | NodeKind::OutImage(o) => o.folder = std::env::temp_dir().join("noiser-scratch-demo").to_string_lossy().into_owned(),
            _ => {}
        }
    }
    let t = std::time::Instant::now();
    let (_, log) = run(&std::env::temp_dir(), &g);
    eprintln!("total {:.1} s", t.elapsed().as_secs_f32());
    for l in log {
        eprintln!("  {l}");
    }
}

/// Paint layers written beside a heightmap the way the Landscape tab does: 8-bit weightmaps and the sidecar's `layers`.
fn add_layers(png: &Path, n: usize, layers: &[(&str, &dyn Fn(usize, usize) -> u8)]) {
    let mut listed = Vec::new();
    for (name, f) in layers {
        let file = format!("terrain_{name}.png");
        let bytes: Vec<u8> = (0..n * n).map(|k| f(k % n, k / n)).collect();
        image::GrayImage::from_raw(n as u32, n as u32, bytes).unwrap().save(png.with_file_name(&file)).unwrap();
        listed.push(serde_json::json!({ "name": name, "file": file }));
    }
    let side = format!("{}.json", png.display());
    let mut json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&side).unwrap()).unwrap();
    json["layers"] = listed.into();
    std::fs::write(side, json.to_string()).unwrap();
}

#[test]
fn a_paint_layer_bakes_per_chunk() {
    use crate::bake::{Channel, Settings};
    let Some(_) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::env::temp_dir().join("noiser-terrain-layers");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let n = 129;
    let png = heightmap(&dir, n, 640.0, 60.0);
    // "west" is 1 on the left half and 0 on the right; "none" is 0 everywhere, "all" is 1.
    add_layers(&png, n, &[("west", &|x, _| if x < n / 2 { 255 } else { 0 }), ("none", &|_, _| 0), ("all", &|_, _| 255)]);
    let land_params = |maps: &[&str]| LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 100.0, chunks: 1, maps: maps.iter().map(|s| s.to_string()).collect(), ..Default::default() };

    // The layer baked as a custom map: bright on the left of the picture, dark on the right.
    let mut g = Graph::default();
    let land = g.add(NodeKind::LoadMesh(land_params(&["west"])), [0.0; 2]);
    let remesh = g.add(NodeKind::Remesh { triangles: 6000, adaptivity: 1.0, mesher: Default::default(), auto_detail: false }, [0.0; 2]);
    let uv = g.add(NodeKind::AutoUv(crate::uvgen::UvParams { planar: true, ..Default::default() }), [0.0; 2]);
    let bake = g.add(NodeKind::Bake(Settings { size: 64, padding: 2, channels: vec![Channel::Normal], custom: vec!["west".into()], samples: 1, ..Default::default() }), [0.0; 2]);
    let layer = g.add(NodeKind::TerrainLayer { name: "west".into() }, [0.0; 2]);
    let img = g.add(NodeKind::OutImage(OutParams { folder: "out".into(), name: "weights".into(), image: crate::bake::ImageFormat::Png8, ..Default::default() }), [0.0; 2]);
    g.connect(land, remesh, 0);
    g.connect(remesh, uv, 0);
    g.connect(uv, bake, 0);
    g.connect(land, bake, 1);
    g.connect(layer, bake, 2);
    g.connect_from(bake, 1, img, 0);
    let (written, log) = run(&dir, &g);
    let file = written.iter().find(|p| p.to_string_lossy().contains("weights")).unwrap_or_else(|| panic!("{written:?} {log:?}"));
    let image = image::open(file).unwrap().to_luma8();
    let mean = |x0: u32, x1: u32| -> f32 { (0..image.height()).flat_map(|y| (x0..x1).map(move |x| (x, y))).map(|(x, y)| image.get_pixel(x, y)[0] as f32).sum::<f32>() / (image.height() * (x1 - x0)) as f32 };
    let (left, right) = (mean(4, 24), mean(40, 60));
    assert!(left > 200.0 && right < 50.0, "the west layer is bright on the left ({left}) and dark on the right ({right})");

    // An unknown layer is an error that says what to do.
    let mut bad = g.clone();
    if let Some(NodeKind::TerrainLayer { name }) = bad.nodes.get_mut(&layer).map(|n| &mut n.kind) {
        *name = "nope".into();
    }
    let gpu = gpu().unwrap();
    let mut cache = MeshCache::default();
    let mut note = |_: &str| {};
    let err = Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.clone(), seed: 0, progress: Default::default(), log: &mut note, batch_file: None, max_triangles: None, tags: Default::default() }.run(&bad).unwrap_err();
    assert!(err.contains("nope") && err.contains("maps"), "{err}");

}

#[test]
fn a_paint_layer_scales_scatter_density() {
    let Some(_) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::env::temp_dir().join("noiser-terrain-layers-scatter");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let n = 129;
    let png = heightmap(&dir, n, 640.0, 60.0);
    add_layers(&png, n, &[("none", &|_, _| 0), ("all", &|_, _| 255)]);
    let land_params = |maps: &[&str]| LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 100.0, chunks: 1, maps: maps.iter().map(|s| s.to_string()).collect(), ..Default::default() };
    // Scatter: a density map of 0 plants nothing, of 1 plants copies.
    let sdf = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("SDFs").join("Mushroom.sdf");
    let count = |map: &str, name: &str| -> usize {
        let mut g = Graph::default();
        let land = g.add(NodeKind::LoadMesh(land_params(&["none", "all"])), [0.0; 2]);
        let scatter = g.add(NodeKind::Scatter(crate::scatter::ScatterParams { file: sdf.to_string_lossy().into_owned(), cell: 160.0, density: 1.0, size: [50.0, 50.0], density_map: map.into(), blend: 0.0, ..Default::default() }), [0.0; 2]);
        let remesh = g.add(NodeKind::Remesh { triangles: 6000, adaptivity: 1.0, mesher: Default::default(), auto_detail: false }, [0.0; 2]);
        let out = g.add(NodeKind::OutMesh(OutParams { folder: "out".into(), name: name.into(), ..Default::default() }), [0.0; 2]);
        g.connect(land, scatter, 0);
        g.connect(scatter, remesh, 0);
        g.connect(remesh, out, 0);
        let (written, log) = run(&dir, &g);
        assert_eq!(written.len(), 1, "{written:?} {log:?}");
        vertices(&written[0]).iter().filter(|v| v[2].parse::<f32>().unwrap() > 70.0).count()
    };
    // The field is in metres here (unit 100 cm): a 640 m terrain, copies 50 m across in 160 m cells.
    let none = count("none", "scatter_none");
    let all = count("all", "scatter_all");
    assert_eq!(none, 0, "no copies where the map is 0");
    assert!(all > 0, "copies where the map is 1");
}

/// The mesh graph Noise Graph amplitude has the sign of the noise graph tab: a positive density pushes the surface outward.
#[test]
fn a_positive_noise_graph_amplitude_pushes_the_surface_outward() {
    use crate::graph::{MathOp, Mesher};
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-amplitude-sign");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ball = crate::mesh::shapes::icosphere(1.0, 3);
    crate::surface::RawMesh { normals: ball.positions.clone(), positions: ball.positions, triangles: ball.triangles }.write_obj(&dir.join("ball.obj")).unwrap();
    // A noise graph whose density is the constant 0.2.
    let mut n = Graph::default();
    let c = n.add(NodeKind::Float(0.2), [0.0; 2]);
    let out = n.add(NodeKind::OutDensity, [0.0; 2]);
    n.connect(c, out, 0);
    let radius = |amplitude: f32, mesher: Mesher, name: &str| -> f32 {
        let mut g = Graph::default();
        let mesh = g.add(NodeKind::LoadMesh(LoadMeshParams { path: "ball.obj".into(), resolution: 64, ..Default::default() }), [0.0; 2]);
        let noise = g.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { embedded: n.to_json(), frequency: 1.0, amplitude, ..Default::default() }), [0.0; 2]);
        let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let remesh = g.add(NodeKind::Remesh { triangles: 3000, adaptivity: 1.0, mesher, auto_detail: false }, [0.0; 2]);
        let o = g.add(NodeKind::OutMesh(OutParams { folder: "out".into(), name: name.into(), ..Default::default() }), [0.0; 2]);
        g.connect(mesh, sum, 0);
        g.connect(noise, sum, 1);
        g.connect(sum, remesh, 0);
        g.connect(remesh, o, 0);
        let (written, log) = run(&dir, &g);
        let v = vertices(&written[0]);
        assert!(!v.is_empty(), "{log:?}");
        let r: Vec<f32> = v.iter().map(|p| p.iter().map(|c| c.parse::<f32>().unwrap().powi(2)).sum::<f32>().sqrt()).collect();
        r.iter().sum::<f32>() / r.len() as f32
    };
    let (plain, up, down) = (radius(0.0, Mesher::Adaptive, "plain"), radius(1.0, Mesher::Adaptive, "up"), radius(-1.0, Mesher::Adaptive, "down"));
    assert!(up > plain + 0.1 && down < plain - 0.1, "plain {plain}, positive {up}, negative {down}");
    // The other meshers follow the same field.
    for mesher in [Mesher::SurfaceNets, Mesher::MarchingTets] {
        let r = radius(1.0, mesher, "grid");
        assert!((r - up).abs() < 0.08, "{mesher:?}: {r} against {up}");
    }
}

/// With no input meshes a batch is the same graph run again, the seed changing and the smart tags keeping the files apart.
#[test]
fn a_batch_without_meshes_makes_one_file_per_seed() {
    use crate::graph::MathOp;
    let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
    let dir = std::env::temp_dir().join("noiser-seed-batch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ball = crate::mesh::shapes::icosphere(1.0, 3);
    crate::surface::RawMesh { normals: ball.positions.clone(), positions: ball.positions, triangles: ball.triangles }.write_obj(&dir.join("ball.obj")).unwrap();
    let mut n = Graph::default();
    let pos = n.add(NodeKind::position(), [0.0; 2]);
    let noise = n.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Perlin, scale: 2.0, octaves: 2, amount: 0.1, ..Default::default() }), [0.0; 2]);
    let out = n.add(NodeKind::OutDensity, [0.0; 2]);
    n.connect(pos, noise, 0);
    n.connect(noise, out, 0);
    let mut g = Graph::default();
    let mesh = g.add(NodeKind::LoadMesh(LoadMeshParams { path: "ball.obj".into(), resolution: 64, ..Default::default() }), [0.0; 2]);
    let rough = g.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { embedded: n.to_json(), frequency: 1.0, amplitude: 1.0, ..Default::default() }), [0.0; 2]);
    let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
    let remesh = g.add(NodeKind::Remesh { triangles: 800, adaptivity: 1.0, mesher: Default::default(), auto_detail: false }, [0.0; 2]);
    let o = g.add(NodeKind::OutMesh(OutParams { folder: "out".into(), name: "{graph}_{seed}_{index}".into(), ..Default::default() }), [0.0; 2]);
    g.connect(mesh, sum, 0);
    g.connect(rough, sum, 1);
    g.connect(sum, remesh, 0);
    g.connect(remesh, o, 0);
    let tags = crate::meshgraph::RunTags { graph: "rock".into(), count: 3, ..Default::default() };
    let mut cache = MeshCache::default();
    let mut files = Vec::new();
    for i in 0..3 {
        let mut note = |_: &str| {};
        let w = Runner { device: &gpu.device, queue: &gpu.queue, cache: &mut cache, base: dir.clone(), seed: crate::meshexport::SeedMode::Increment.seed_for(10, i), progress: Default::default(), log: &mut note, batch_file: None, max_triangles: None, tags: tags.run(i) }.run(&g).unwrap();
        files.extend(w);
    }
    let names: Vec<String> = files.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["rock_10_01.obj", "rock_11_02.obj", "rock_12_03.obj"]);
    let contents: std::collections::BTreeSet<String> = files.iter().map(|f| std::fs::read_to_string(f).unwrap()).collect();
    assert_eq!(contents.len(), 3, "a different seed makes a different rock");
}

/// Every mesher aims at the triangle count asked for, and the complexity option gives a plain sphere about the count and a
/// rough one more.
#[test]
fn meshers_hit_the_triangle_count_and_complexity_scales_it() {
    use crate::graph::{MathOp, Mesher};
    if gpu().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    let dir = std::env::temp_dir().join("noiser-meshers");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ball = crate::mesh::shapes::icosphere(1.0, 3);
    crate::surface::RawMesh { normals: ball.positions.clone(), positions: ball.positions, triangles: ball.triangles }.write_obj(&dir.join("ball.obj")).unwrap();
    let count = |mesher: Mesher, auto: bool, name: &str, bumps: Option<f32>| -> usize {
        let mut g = Graph::default();
        let mesh = g.add(NodeKind::LoadMesh(LoadMeshParams { path: "ball.obj".into(), resolution: 64, ..Default::default() }), [0.0; 2]);
        let mut field = mesh;
        if let Some(a) = bumps {
            let mut n = Graph::default();
            let pos = n.add(NodeKind::position(), [0.0; 2]);
            let noise = n.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Perlin, scale: 1.0, octaves: 1, amount: 1.0, seed: Some(1), ..Default::default() }), [0.0; 2]);
            let out = n.add(NodeKind::OutDensity, [0.0; 2]);
            n.connect(pos, noise, 0);
            n.connect(noise, out, 0);
            let rough = g.add(NodeKind::NoiseGraph(crate::meshgraph::NoiseGraphParams { embedded: n.to_json(), frequency: 4.0, amplitude: a, ..Default::default() }), [0.0; 2]);
            let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
            g.connect(mesh, sum, 0);
            g.connect(rough, sum, 1);
            field = sum;
        }
        let remesh = g.add(NodeKind::Remesh { triangles: 2000, adaptivity: 1.0, mesher, auto_detail: auto }, [0.0; 2]);
        let o = g.add(NodeKind::OutMesh(OutParams { folder: "out".into(), name: name.into(), ..Default::default() }), [0.0; 2]);
        g.connect(field, remesh, 0);
        g.connect(remesh, o, 0);
        let (written, log) = run(&dir, &g);
        let n = std::fs::read_to_string(&written[0]).unwrap().lines().filter(|l| l.starts_with("f ")).count();
        assert!(n > 0, "{log:?}");
        n
    };
    for (mesher, name) in [(Mesher::Adaptive, "a"), (Mesher::SurfaceNets, "n"), (Mesher::MarchingTets, "t")] {
        let n = count(mesher, false, name, None);
        assert!((1500..2700).contains(&n), "{mesher:?} gave {n} triangles for 2000");
    }
    let plain = count(Mesher::Adaptive, true, "plain_auto", None);
    let rough = count(Mesher::Adaptive, true, "rough_auto", Some(0.12));
    assert!((1500..2800).contains(&plain), "a sphere keeps about the count: {plain}");
    assert!(rough > plain * 3 / 2, "a bumpy shape gets more: {rough} against {plain}");
}
