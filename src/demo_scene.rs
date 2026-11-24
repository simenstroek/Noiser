//! Writes the demo terrain: noise graphs, a layered landscape project and the Mesh graph that puts SDF shapes on it and cuts
//! it into Nanite chunks. Not part of the app; run by hand:
//!
//!   NOISER_DEMO_LIB="H:/My Drive/NoiserLibrary" NOISER_DEMO_OUT="E:/Unreal Projects/FAB_StagingZone/NoiserExport" \
//!   NOISER_DEMO_TRIS=250000 cargo test --release write_demo_terrain -- --ignored --nocapture
//!
//! It needs the shapes of `tools/demo/make_shapes.py` and the library's Mushroom. The landscape project is made first and its
//! heightmap exported (with erosion) through the same code as `noiser --headless --landscape-export`; the shapes are placed
//! on the result by looking at its slopes and heights.

use crate::graph::{Graph, MathOp, NodeId, NodeKind, NoiseKind, NoiseParams};
use crate::landscape_masks::{Brush, Mask, Tool};
use crate::meshgraph::{LoadMeshParams, NoiseGraphParams, OutParams};
use crate::scatter::ScatterParams;
use std::path::{Path, PathBuf};

fn math(g: &mut Graph, op: MathOp, a: f32, b: f32, c: f32) -> NodeId {
    g.add(NodeKind::Math { op, a, b, c }, [0.0; 2])
}

/// `inputs[i]` goes to input `i` of a new Math node.
fn op(g: &mut Graph, op: MathOp, inputs: &[NodeId], consts: (f32, f32, f32)) -> NodeId {
    let n = math(g, op, consts.0, consts.1, consts.2);
    for (i, from) in inputs.iter().enumerate() {
        g.connect(*from, n, i);
    }
    n
}

fn noise(g: &mut Graph, pos: NodeId, kind: NoiseKind, scale: f32, octaves: u32, seed: u32, normalize: bool) -> NodeId {
    let n = g.add(NodeKind::Noise(NoiseParams { kind, scale, octaves, seed: Some(seed), normalize, ..Default::default() }), [0.0; 2]);
    g.connect(pos, n, 0);
    n
}

/// Nodes in columns by how far they are from the sources, so a generated graph opens readable.
fn layout(g: &mut Graph) {
    let ids: Vec<NodeId> = g.nodes.keys().copied().collect();
    let mut depth: std::collections::BTreeMap<NodeId, usize> = ids.iter().map(|i| (*i, 0)).collect();
    for _ in 0..ids.len() {
        for e in &g.edges {
            let d = depth[&e.from] + 1;
            if d > depth[&e.to] {
                depth.insert(e.to, d);
            }
        }
    }
    let mut rows: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for id in ids {
        let d = depth[&id];
        let row = rows.entry(d).or_insert(0);
        g.nodes.get_mut(&id).unwrap().pos = [d as f32 * 230.0, *row as f32 * 120.0];
        *row += 1;
    }
}

fn finish(mut g: Graph, density: NodeId) -> Graph {
    let out = g.add(NodeKind::OutDensity, [0.0; 2]);
    g.connect(density, out, 0);
    layout(&mut g);
    g
}

/// The detail noise graphs, each a signed field (-1 to 1) that a shape's amplitude and frequency scale.
fn detail_graphs() -> Vec<(&'static str, Graph)> {
    let mut out = Vec::new();
    // Warty caps and stems: Worley cells plus a little Perlin.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let cells = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Worley, scale: 3.0, seed: Some(4), normalize: false, ..Default::default() }), [0.0; 2]);
        g.connect(pos, cells, 0);
        let fine = noise(&mut g, pos, NoiseKind::Perlin, 9.0, 2, 8, false);
        let cells = op(&mut g, MathOp::Multiply, &[cells], (0.0, 0.7, 0.0));
        let fine = op(&mut g, MathOp::Multiply, &[fine], (0.0, 0.3, 0.0));
        let sum = op(&mut g, MathOp::Add, &[cells, fine], (0.0, 0.0, 0.0));
        out.push(("MushroomBumps", finish(g, sum)));
    }
    // Boulders: rough, billowing rock.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let n = noise(&mut g, pos, NoiseKind::Perlin, 3.0, 6, 21, false);
        let ridged = op(&mut g, MathOp::Ridge, &[n], (0.0, 0.0, 0.0));
        let centred = op(&mut g, MathOp::Subtract, &[ridged], (0.0, 0.5, 0.0));
        let scaled = op(&mut g, MathOp::Multiply, &[centred], (0.0, 2.0, 0.0));
        out.push(("RockSurface", finish(g, scaled)));
    }
    // Strata: horizontal bands that wobble, for crags and arches.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let sep = g.add(NodeKind::Separate, [0.0; 2]);
        g.connect(pos, sep, 0);
        let wob = noise(&mut g, pos, NoiseKind::Perlin, 2.0, 3, 33, false);
        let wob = op(&mut g, MathOp::Multiply, &[wob], (0.0, 2.5, 0.0));
        let z = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 9.0, c: 0.0 }, [0.0; 2]);
        g.connect_from(sep, 2, z, 0);
        let phase = op(&mut g, MathOp::Add, &[z, wob], (0.0, 0.0, 0.0));
        let bands = op(&mut g, MathOp::Sine, &[phase], (0.0, 0.0, 0.0));
        let grit = noise(&mut g, pos, NoiseKind::Perlin, 14.0, 3, 35, false);
        let grit = op(&mut g, MathOp::Multiply, &[grit], (0.0, 0.35, 0.0));
        let bands = op(&mut g, MathOp::Multiply, &[bands], (0.0, 0.65, 0.0));
        let sum = op(&mut g, MathOp::Add, &[bands, grit], (0.0, 0.0, 0.0));
        out.push(("StoneStrata", finish(g, sum)));
    }
    // Cave walls: lumpy cellular rock.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let cells = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Worley, scale: 2.0, seed: Some(51), normalize: false, ..Default::default() }), [0.0; 2]);
        g.connect(pos, cells, 0);
        let n = noise(&mut g, pos, NoiseKind::Perlin, 6.0, 4, 52, false);
        let cells = op(&mut g, MathOp::Multiply, &[cells], (0.0, 0.6, 0.0));
        let n = op(&mut g, MathOp::Multiply, &[n], (0.0, 0.5, 0.0));
        let sum = op(&mut g, MathOp::Add, &[cells, n], (0.0, 0.0, 0.0));
        out.push(("CaveWalls", finish(g, sum)));
    }
    // Fine grit for the terrain itself.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let n = noise(&mut g, pos, NoiseKind::Perlin, 4.0, 6, 61, false);
        out.push(("TerrainGrit", finish(g, n)));
    }
    // Dunes for the landscape: soft ridges.
    {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let n = noise(&mut g, pos, NoiseKind::Perlin, 7.0, 2, 71, false);
        let ridged = op(&mut g, MathOp::Ridge, &[n], (0.0, 0.0, 0.0));
        out.push(("Dunes", finish(g, ridged)));
    }
    out
}

fn embedded(path: &Path, g: &Graph) -> NoiseGraphParams {
    NoiseGraphParams { file: path.to_string_lossy().into_owned(), embedded: g.to_json(), ..Default::default() }
}

/// The layered landscape: plains with hills and dunes (the dunes only where the "basin" mask is painted), and a mountain
/// range (the "range" mask) the plains blend into.
fn landscape_graph(dunes: &NoiseGraphParams) -> Graph {
    let mut g = Graph::default();
    let pos = g.add(NodeKind::position(), [0.0; 2]);
    let hills = noise(&mut g, pos, NoiseKind::Perlin, 2.0, 5, 11, true);
    let ridge_n = noise(&mut g, pos, NoiseKind::Perlin, 3.0, 6, 5, false);
    let ridge = op(&mut g, MathOp::Ridge, &[ridge_n], (0.0, 0.0, 0.0));
    let ridge = op(&mut g, MathOp::Power, &[ridge], (0.0, 2.0, 0.0));
    let dune = g.add(NodeKind::NoiseGraph(dunes.clone()), [0.0; 2]);
    g.connect(pos, dune, 0);
    let basin = g.add(NodeKind::Mask { name: "basin".into() }, [0.0; 2]);
    let range = g.add(NodeKind::Mask { name: "range".into() }, [0.0; 2]);
    let dune = op(&mut g, MathOp::Multiply, &[dune, basin], (0.0, 0.0, 0.0));
    let dune = op(&mut g, MathOp::Multiply, &[dune], (0.0, 0.2, 0.0));
    let flat = op(&mut g, MathOp::Multiply, &[hills], (0.0, 0.42, 0.0));
    let plains = op(&mut g, MathOp::Add, &[flat, dune], (0.0, 0.0, 0.0));
    let peaks = op(&mut g, MathOp::Multiply, &[ridge], (0.0, 0.95, 0.0));
    let rise = op(&mut g, MathOp::Multiply, &[hills], (0.0, 0.25, 0.0));
    let mountains = op(&mut g, MathOp::Add, &[peaks, rise], (0.0, 0.0, 0.0));
    let height = op(&mut g, MathOp::Lerp, &[plains, mountains, range], (0.0, 0.0, 0.0));
    finish(g, height)
}

fn masks() -> Vec<Mask> {
    let mut range = Mask::new("range");
    range.stroke((0.15, 0.8), (0.55, 0.55), &Brush { radius: 0.15, strength: 1.0, hardness: 0.2 }, Tool::Paint);
    range.stroke((0.55, 0.55), (0.85, 0.2), &Brush { radius: 0.17, strength: 1.0, hardness: 0.2 }, Tool::Paint);
    range.blur(10);
    let mut basin = Mask::new("basin");
    basin.stroke((0.2, 0.2), (0.35, 0.3), &Brush { radius: 0.18, strength: 1.0, hardness: 0.2 }, Tool::Paint);
    basin.stroke((0.8, 0.8), (0.65, 0.9), &Brush { radius: 0.15, strength: 1.0, hardness: 0.2 }, Tool::Paint);
    basin.blur(10);
    vec![range, basin]
}

/// The Mesh graph: the terrain with grit, then one Scatter SDF node per kind of object (each with its own noise graph
/// roughening its copies), then Remesh in chunks, planar UVs and the maps.
fn mesh_graph(png: &Path, lib: &Path, out_dir: &Path, triangles: u32) -> Graph {
    let mut g = Graph::default();
    let folder = out_dir.to_string_lossy().into_owned();
    let noises = lib.join("Noises");
    let graphs: std::collections::BTreeMap<&str, Graph> = detail_graphs().into_iter().collect();
    let noise_node = |g: &mut Graph, name: &str, amplitude: f32, frequency: f32| {
        let mut p = embedded(&noises.join(format!("{name}.noisegraph")), &graphs[name]);
        (p.amplitude, p.frequency) = (amplitude, frequency);
        g.add(NodeKind::NoiseGraph(p), [0.0; 2])
    };
    let terrain = g.add(NodeKind::LoadMesh(LoadMeshParams { path: png.to_string_lossy().into_owned(), unit_cm: 1.0, chunks: 2, ..Default::default() }), [0.0, 0.0]);
    let grit = noise_node(&mut g, "TerrainGrit", 0.8, 0.05);
    let mut field = op(&mut g, MathOp::Add, &[terrain, grit], (0.0, 0.0, 0.0));
    let sdf = |name: &str| lib.join("SDFs").join(format!("{name}.sdf")).to_string_lossy().into_owned();
    // (shape, noise graph, noise amplitude in metres and frequency per metre, the rule)
    let kinds = [
        ("Boulder", "RockSurface", 0.8, 0.07, ScatterParams { file: sdf("Boulder"), seed: 3, cell: 3500.0, density: 0.45, size: [700.0, 1800.0], slope: [0.05, 0.6], blend: 120.0, ..Default::default() }),
        ("Crag", "StoneStrata", 0.6, 0.08, ScatterParams { file: sdf("Crag"), seed: 5, cell: 8000.0, density: 0.45, size: [2000.0, 3200.0], slope: [0.25, 1.0], elevation: [0.3, 1.0], blend: 150.0, ..Default::default() }),
        ("Mushroom", "MushroomBumps", 0.3, 0.12, ScatterParams { file: sdf("Mushroom"), seed: 7, cell: 6000.0, density: 0.3, size: [1500.0, 2800.0], slope: [0.0, 0.15], elevation: [0.0, 0.5], blend: 100.0, ..Default::default() }),
        ("Arch", "StoneStrata", 0.8, 0.05, ScatterParams { file: sdf("Arch"), seed: 11, cell: 18000.0, density: 0.35, size: [4200.0, 5400.0], slope: [0.0, 0.2], elevation: [0.0, 0.6], blend: 200.0, ..Default::default() }),
        ("Cave", "CaveWalls", 0.8, 0.04, ScatterParams { file: sdf("Cave"), seed: 13, cell: 15000.0, density: 0.35, size: [3000.0, 4500.0], slope: [0.7, 1.8], elevation: [0.2, 0.97], blend: 150.0, subtract: true, face_uphill: true, sink: 0.12, ..Default::default() }),
    ];
    for (_, noise, amplitude, frequency, params) in kinds {
        let scatter = g.add(NodeKind::Scatter(params), [0.0; 2]);
        let rough = noise_node(&mut g, noise, amplitude, frequency);
        g.connect(field, scatter, 0);
        g.connect(rough, scatter, 1);
        field = scatter;
    }
    let remesh = g.add(NodeKind::Remesh { triangles, adaptivity: 1.0 }, [0.0; 2]);
    let uv = g.add(NodeKind::AutoUv(crate::uvgen::UvParams { planar: true, ..Default::default() }), [0.0; 2]);
    let mesh_out = g.add(NodeKind::OutMesh(OutParams { folder: folder.clone(), name: "Terrain".into(), ..Default::default() }), [0.0; 2]);
    let bake = g.add(NodeKind::Bake(crate::bake::Settings { channels: vec![crate::bake::Channel::Normal, crate::bake::Channel::Occlusion, crate::bake::Channel::Curvature], size: 2048, ..Default::default() }), [0.0; 2]);
    g.connect(field, remesh, 0);
    g.connect(remesh, uv, 0);
    g.connect(uv, mesh_out, 0);
    g.connect(uv, bake, 0);
    g.connect(field, bake, 1);
    for i in 0..3 {
        let img = g.add(NodeKind::OutImage(OutParams { folder: folder.clone(), name: "Terrain".into(), ..Default::default() }), [0.0; 2]);
        g.connect_from(bake, i, img, 0);
    }
    layout(&mut g);
    g
}

#[test]
#[ignore]
fn write_demo_terrain() {
    let lib = PathBuf::from(std::env::var("NOISER_DEMO_LIB").expect("NOISER_DEMO_LIB"));
    let out = PathBuf::from(std::env::var("NOISER_DEMO_OUT").expect("NOISER_DEMO_OUT"));
    let triangles: u32 = std::env::var("NOISER_DEMO_TRIS").ok().and_then(|v| v.parse().ok()).unwrap_or(250_000);
    for (name, g) in detail_graphs() {
        std::fs::write(lib.join("Noises").join(format!("{name}.noisegraph")), g.to_json()).unwrap();
    }
    let terrains = lib.join("Terrains");
    std::fs::create_dir_all(&terrains).unwrap();
    let dunes = detail_graphs().into_iter().find(|(n, _)| *n == "Dunes").unwrap().1;
    let graph = landscape_graph(&embedded(&lib.join("Noises").join("Dunes.noisegraph"), &dunes));
    let settings = crate::landscape::Settings { world_m: 1000.0, height_m: 140.0, ..Default::default() };
    let masks: Vec<_> = masks().iter().map(|m| serde_json::json!({ "name": m.name, "png": crate::landscape_masks::base64_encode(&m.to_png()) })).collect();
    let project = serde_json::json!({ "noiser_landscape": 1, "graph": graph.to_json(), "settings": settings, "masks": masks });
    let project_path = terrains.join("DemoLandscape.landscape");
    std::fs::write(&project_path, project.to_string()).unwrap();
    let png = terrains.join("DemoHeightmap.png");
    let msg = crate::landscape::export_headless(&project_path, &png, true, &mut |s| eprintln!("{s}")).unwrap();
    eprintln!("{msg}");
    let g = mesh_graph(&png, &lib, &out, triangles);
    let path = lib.join("Mesh Graphs").join("DemoTerrain.meshgraph");
    std::fs::write(&path, g.to_json()).unwrap();
    eprintln!("wrote {}", path.display());
}
