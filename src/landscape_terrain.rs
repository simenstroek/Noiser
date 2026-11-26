//! Grid boundaries in the Landscape graph. The graph is a function the GPU evaluates anywhere; an Erosion node is a
//! simulation over the whole terrain. So before a Landscape graph is compiled, each Erosion node's input is sampled onto
//! the working grid and eroded (or its cached result reused), its outputs become terrain maps
//! (`erosion#<id>.height`, `.flow`, `.deposition`, `.wear`), and the node is replaced by Map nodes that read them. What
//! is left is an ordinary graph over maps. Inner Erosion nodes go first, so erosion can follow erosion.
//! See docs/terrain-maps-plan.md.

use crate::export::Progress;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::landscape::{build_evaluator, sample, Settings};
use crate::landscape_erosion::{erode, Erosion, ErosionMaps};
use crate::terrain_maps::{Source, TerrainMaps};
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The names of an Erosion node's outputs, in port order (they end its maps' names).
pub const OUTPUTS: [&str; 4] = ["height", "flow", "deposition", "wear"];

/// The name of output `port` of Erosion node `id` as a terrain map.
pub fn map_name(id: NodeId, port: usize) -> String {
    format!("erosion#{id}.{}", OUTPUTS[port.min(3)])
}

/// What one Erosion node made: its four outputs on an `side` x `side` grid (the height in the graph's units).
pub struct Eroded {
    pub key: u64,
    pub side: usize,
    pub outputs: [Arc<Vec<f32>>; 4],
}

/// Results by node, kept between runs so editing what comes after an Erosion node never runs it again.
pub type ErosionCache = Arc<Mutex<HashMap<NodeId, Arc<Eroded>>>>;

/// How an Erosion node's result stood when the graph was last prepared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Run this time.
    Ran,
    /// Up to date: a cached result was used.
    Cached,
    /// Not running by itself, and what feeds it changed: its last result was used (or none, if it never ran).
    OutOfDate,
}

/// A Landscape graph ready to compile: Erosion nodes replaced by Map reads, the maps they read, and each node's state.
pub struct Prepared {
    pub graph: Graph,
    pub maps: TerrainMaps,
    pub states: HashMap<NodeId, State>,
}

/// True when the graph has an Erosion node that is not muted.
pub fn has_erosion(graph: &Graph) -> bool {
    graph.nodes.values().any(|n| !n.muted && matches!(n.kind, NodeKind::Erosion(_)))
}

/// The nodes `from` depends on (itself included), following inputs through muted nodes and reroutes.
fn upstream(graph: &Graph, from: NodeId) -> BTreeSet<NodeId> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let inputs = graph.nodes.get(&id).map_or(0, |n| n.kind.input_count());
        for port in 0..inputs {
            if let Some((src, _)) = graph.input_from(id, port) {
                stack.push(src);
            }
        }
    }
    seen
}

/// Prepares `graph` (see the module docs). `maps` are the maps there already are (painted masks...), and `maps_key`
/// changes whenever they do. Erosion nodes in `run` run even when they do not run by themselves.
pub fn prepare(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    graph: &Graph,
    s: &Settings,
    maps: &TerrainMaps,
    maps_key: u64,
    base: &Path,
    cache: &ErosionCache,
    run: &BTreeSet<NodeId>,
    progress: &Progress,
) -> Result<Prepared, String> {
    let mut g = graph.clone();
    let mut maps = maps.clone();
    let mut states = HashMap::new();
    let n = s.resolution.max(2);
    lower_heightmaps(&mut g, &mut maps, base, s.height_m)?;
    lower_stamps(&mut g, &mut maps, base, s)?;
    loop {
        let erosion: Vec<NodeId> = g.nodes.iter().filter(|(_, node)| !node.muted && matches!(node.kind, NodeKind::Erosion(_))).map(|(id, _)| *id).collect();
        // The first whose input has no Erosion node of its own before it.
        let Some(&id) = erosion.iter().find(|&&id| g.input_from(id, 0).is_none_or(|(src, _)| !upstream(&g, src).iter().any(|u| erosion.contains(u)))) else {
            break;
        };
        let NodeKind::Erosion(params) = g.nodes[&id].kind.clone() else { unreachable!() };
        let source = g.input_from(id, 0);
        // What the result depends on: everything upstream, the settings, the grid and the maps.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        if let Some((src, port)) = source {
            (src, port).hash(&mut h);
            for u in upstream(&g, src) {
                let node = &g.nodes[&u];
                (u, serde_json::to_string(&node.kind).unwrap_or_default()).hash(&mut h);
                for p in 0..node.kind.input_count() {
                    g.input_from(u, p).hash(&mut h);
                }
            }
        }
        (params.key(), n, s.seed, s.world_m.to_bits(), s.height_m.to_bits(), maps_key).hash(&mut h);
        let key = h.finish();
        let cached = cache.lock().map_err(|_| "the erosion cache is unusable")?.get(&id).cloned();
        let eroded = match cached {
            Some(c) if c.key == key => {
                states.insert(id, State::Cached);
                Some(c)
            }
            c if !params.auto && !run.contains(&id) => {
                states.insert(id, State::OutOfDate);
                c
            }
            _ => {
                let e = Arc::new(run_erosion(device, queue, &g, source, &params, s, &maps, base, n, key, progress)?);
                cache.lock().map_err(|_| "the erosion cache is unusable")?.insert(id, e.clone());
                states.insert(id, State::Ran);
                Some(e)
            }
        };
        // Its outputs as maps (all zero before a first run), and Map nodes in its place.
        for port in 0..4 {
            let (side, data) = match &eroded {
                Some(e) => (e.side, e.outputs[port].clone()),
                None => (2, Arc::new(vec![0.0; 4])),
            };
            maps.insert(map_name(id, port), side, data, Source::Erosion);
        }
        let pos = g.nodes[&id].pos;
        let readers: Vec<NodeId> = (0..4).map(|port| g.add(NodeKind::Mask { name: map_name(id, port) }, pos)).collect();
        for e in &mut g.edges {
            if e.from == id {
                e.from = readers[e.from_port.min(3)];
                e.from_port = 0;
            }
        }
        g.remove(id);
    }
    Ok(Prepared { graph: g, maps, states })
}

/// Heightmap nodes read their images and become Map nodes over them. The map's name carries a hash of what was read, so
/// everything keyed by the graph (the cached erosion results) sees a changed image.
fn lower_heightmaps(g: &mut Graph, maps: &mut TerrainMaps, base: &Path, height_m: f32) -> Result<(), String> {
    let ids: Vec<NodeId> = g.nodes.iter().filter(|(_, n)| !n.muted && matches!(n.kind, NodeKind::Heightmap(_))).map(|(id, _)| *id).collect();
    for id in ids {
        let NodeKind::Heightmap(p) = g.nodes[&id].kind.clone() else { continue };
        let loaded = crate::landscape_heightmap::load(&p, base, height_m).map_err(|e| format!("Heightmap node {id}: {e}"))?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        loaded.data.iter().for_each(|v| v.to_bits().hash(&mut h));
        let name = format!("heightmap#{id}#{:x}", h.finish());
        maps.insert(name.clone(), loaded.side, Arc::new(loaded.data), Source::Imported);
        g.nodes.get_mut(&id).expect("listed").kind = NodeKind::Mask { name };
    }
    Ok(())
}

/// Stamp nodes trace their shapes onto the working grid and become Map reads and arithmetic: Add is `input + height`,
/// Cut `input - height`, Max `mix(input, max(input, height), cover)`.
fn lower_stamps(g: &mut Graph, maps: &mut TerrainMaps, base: &Path, s: &Settings) -> Result<(), String> {
    use crate::graph::MathOp;
    use crate::landscape_stamp::StampMode;
    let ids: Vec<NodeId> = g.nodes.iter().filter(|(_, n)| !n.muted && matches!(n.kind, NodeKind::Stamp(_))).map(|(id, _)| *id).collect();
    for id in ids {
        let NodeKind::Stamp(p) = g.nodes[&id].kind.clone() else { continue };
        let side = s.resolution.max(2);
        let (key, baked) = crate::landscape_stamp::bake(&p, base, side, s.world_m).map_err(|e| format!("Stamp node {id}: {e}"))?;
        let k = 1.0 / s.height_m.max(1e-3);
        let units: Vec<f32> = baked.height_m.iter().map(|m| m * k).collect();
        let (hname, cname) = (format!("stamp#{id}#{key:x}.height"), format!("stamp#{id}#{key:x}.cover"));
        maps.insert(hname.clone(), side, Arc::new(units), Source::Stamp);
        maps.insert(cname.clone(), side, Arc::new(baked.cover.clone()), Source::Stamp);
        let pos = g.nodes[&id].pos;
        let terrain = g.edges.iter().find(|e| e.to == id && e.port == 0).map(|e| (e.from, e.from_port));
        g.edges.retain(|e| e.to != id);
        let height = g.add(NodeKind::Mask { name: hname }, pos);
        match p.mode {
            StampMode::Add | StampMode::Cut => {
                let op = if p.mode == StampMode::Add { MathOp::Add } else { MathOp::Subtract };
                g.nodes.get_mut(&id).expect("listed").kind = NodeKind::Math { op, a: 0.0, b: 0.0, c: 0.0 };
                if let Some((from, port)) = terrain {
                    g.connect_from(from, port, id, 0);
                }
                g.connect(height, id, 1);
            }
            StampMode::Max => {
                let cover = g.add(NodeKind::Mask { name: cname }, pos);
                let top = g.add(NodeKind::Math { op: MathOp::Max, a: 0.0, b: 0.0, c: 0.0 }, pos);
                if let Some((from, port)) = terrain {
                    g.connect_from(from, port, top, 0);
                }
                g.connect(height, top, 1);
                g.nodes.get_mut(&id).expect("listed").kind = NodeKind::Mix { vals: [0.0; 3], clamp: true };
                g.connect(cover, id, 0);
                if let Some((from, port)) = terrain {
                    g.connect_from(from, port, id, 1);
                }
                g.connect(top, id, 2);
            }
        }
    }
    Ok(())
}

/// Samples what feeds an Erosion node onto the grid and erodes it.
#[allow(clippy::too_many_arguments)]
fn run_erosion(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    g: &Graph,
    source: Option<(NodeId, usize)>,
    params: &Erosion,
    s: &Settings,
    maps: &TerrainMaps,
    base: &Path,
    n: usize,
    key: u64,
    progress: &Progress,
) -> Result<Eroded, String> {
    // The input alone as the graph's output (nodes that do not feed it are not compiled).
    let values = match source {
        Some((src, port)) => {
            let mut sub = g.clone();
            sub.nodes.retain(|_, node| !matches!(node.kind, NodeKind::OutDensity | NodeKind::OutNormal));
            sub.prune_edges();
            let out = sub.add(NodeKind::OutDensity, [0.0; 2]);
            sub.connect_from(src, port, out, 0);
            let ev = build_evaluator(device, queue, &sub, s, maps, base)?;
            sample(&ev, n, &Progress::default())
        }
        None => vec![0.0; n * n],
    };
    // Erosion works in metres: one unit of the graph is the height setting.
    let k = s.height_m.max(1e-3);
    let metres: Vec<f32> = values.iter().map(|v| v * k).collect();
    let cell_m = s.world_m / (n - 1) as f32;
    let result = if params.kind == crate::landscape_erosion::ErosionKind::Detail {
        crate::landscape_detail::detail(device, queue, &metres, n, cell_m, params, s.seed, progress)?
    } else {
        erode(device, queue, &metres, n, cell_m, params, progress)?
    };
    let extra = ErosionMaps::new(&metres, &result);
    let height: Vec<f32> = result.heights.iter().map(|m| m / k).collect();
    Ok(Eroded { key, side: n, outputs: [Arc::new(height), Arc::new(extra.flow), Arc::new(extra.deposition), Arc::new(extra.wear)] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_tests::gpu;
    use crate::graph::NodeKind;
    use crate::graph::NoiseParams;

    /// position -> noise -> Erosion -> density output, on a small grid.
    fn graph(auto: bool) -> (Graph, NodeId) {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0, 0.0]);
        let noise = g.add(NodeKind::Noise(NoiseParams { scale: 3.0, octaves: 4, ..Default::default() }), [200.0, 0.0]);
        let ero = g.add(NodeKind::Erosion(Erosion { iterations: 20, thermal_iterations: 2, auto, ..Default::default() }), [400.0, 0.0]);
        let out = g.add(NodeKind::OutDensity, [600.0, 0.0]);
        g.connect(pos, noise, 0);
        g.connect(noise, ero, 0);
        g.connect(ero, out, 0);
        (g, ero)
    }

    fn prep(gpu: &crate::gpu_tests::Gpu, g: &Graph, cache: &ErosionCache, run: &BTreeSet<NodeId>) -> Prepared {
        let s = Settings { resolution: 64, ..Default::default() };
        prepare(&gpu.device, &gpu.queue, g, &s, &TerrainMaps::default(), 0, Path::new("."), cache, run, &Default::default()).unwrap()
    }

    #[test]
    fn erosion_is_replaced_by_maps_and_cached() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let (g, ero) = graph(true);
        let cache = ErosionCache::default();
        let first = prep(&gpu, &g, &cache, &BTreeSet::new());
        assert_eq!(first.states[&ero], State::Ran);
        assert!(!has_erosion(&first.graph), "the Erosion node is gone");
        for port in 0..4 {
            assert!(first.maps.get(&map_name(ero, port)).is_some(), "output {port} is a map");
        }
        // Same graph again: the cached result is used.
        assert_eq!(prep(&gpu, &g, &cache, &BTreeSet::new()).states[&ero], State::Cached);
        // Something downstream changes: still cached.
        let mut g2 = g.clone();
        g2.add(NodeKind::Float(1.0), [0.0, 100.0]);
        assert_eq!(prep(&gpu, &g2, &cache, &BTreeSet::new()).states[&ero], State::Cached);
        // Something upstream changes: it runs again.
        let mut g3 = g.clone();
        for node in g3.nodes.values_mut() {
            if let NodeKind::Noise(p) = &mut node.kind {
                p.scale = 5.0;
            }
        }
        assert_eq!(prep(&gpu, &g3, &cache, &BTreeSet::new()).states[&ero], State::Ran);
    }

    #[test]
    fn without_auto_an_erosion_node_waits_for_run() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let (g, ero) = graph(false);
        let cache = ErosionCache::default();
        assert_eq!(prep(&gpu, &g, &cache, &BTreeSet::new()).states[&ero], State::OutOfDate, "never ran: out of date, maps are zero");
        assert_eq!(prep(&gpu, &g, &cache, &BTreeSet::from([ero])).states[&ero], State::Ran, "Run runs it");
        assert_eq!(prep(&gpu, &g, &cache, &BTreeSet::new()).states[&ero], State::Cached);
    }

    #[test]
    fn a_heightmap_node_brings_an_exported_terrain_back() {
        use crate::landscape_heightmap::HeightmapParams;
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let s = Settings { resolution: 33, height_m: 400.0, ..Default::default() };
        // A terrain whose height is x, from 0 to 0.5 of the height setting, written the way the export does it.
        let units: Vec<f32> = (0..33 * 33).map(|k| (k % 33) as f32 / 32.0 * 0.5).collect();
        let metres: Vec<f32> = units.iter().map(|u| u * s.height_m).collect();
        let dir = std::env::temp_dir().join(format!("noiser_hm_node_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        crate::landscape::write_unreal(&dir.join("t.png"), 33, s.world_m, &crate::landscape::encode_unreal(&metres)).unwrap();
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0; 2]);
        let hm = g.add(NodeKind::Heightmap(HeightmapParams { file: "t.png".into(), ..Default::default() }), [0.0; 2]);
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(pos, hm, 0);
        g.connect(hm, out, 0);
        let prepared = prepare(&gpu.device, &gpu.queue, &g, &s, &TerrainMaps::default(), 0, &dir, &ErosionCache::default(), &BTreeSet::new(), &Default::default()).unwrap();
        assert!(!prepared.graph.nodes.values().any(|n| matches!(n.kind, NodeKind::Heightmap(_))), "lowered to a map");
        let ev = build_evaluator(&gpu.device, &gpu.queue, &prepared.graph, &s, &prepared.maps, &dir).unwrap();
        let got = sample(&ev, 33, &Default::default());
        let worst = got.iter().zip(&units).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "worst difference {worst} units");
        // A missing file is an error naming the node.
        let mut bad = g.clone();
        if let Some(NodeKind::Heightmap(p)) = bad.nodes.get_mut(&hm).map(|n| &mut n.kind) {
            p.file = "gone.png".into();
        }
        let err = prepare(&gpu.device, &gpu.queue, &bad, &s, &TerrainMaps::default(), 0, &dir, &ErosionCache::default(), &BTreeSet::new(), &Default::default()).err().unwrap();
        assert!(err.contains("Heightmap node"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn terraces_make_flat_treads_and_craters_dig_bowls() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let s = Settings { resolution: 65, ..Default::default() };
        let sample_graph = |kind: NodeKind, from_x: bool| -> Vec<f32> {
            let mut g = Graph::default();
            let pos = g.add(NodeKind::position(), [0.0; 2]);
            let sep = g.add(NodeKind::Separate, [0.0; 2]);
            let node = g.add(kind, [0.0; 2]);
            let out = g.add(NodeKind::OutDensity, [0.0; 2]);
            g.connect(pos, sep, 0);
            if from_x {
                g.connect_from(sep, 0, node, 0);
            } else {
                g.connect(pos, node, 0);
            }
            g.connect(node, out, 0);
            let ev = build_evaluator(&gpu.device, &gpu.queue, &g, &s, &TerrainMaps::default(), Path::new(".")).unwrap();
            sample(&ev, 65, &Default::default())
        };
        // Four sharp terraces over x = 0..1: nearly every sample sits on a tread at k / 4.
        let t = sample_graph(NodeKind::Terrace { steps: 4.0, sharpness: 40.0 }, true);
        let on_tread = t.iter().filter(|v| ((**v * 4.0) - (**v * 4.0).round()).abs() < 0.05).count();
        assert!(on_tread * 10 > t.len() * 8, "{on_tread} of {} on a tread", t.len());
        // One crater per cell: the lowest point is about the depth below the highest rim, and there are rims.
        let c = sample_graph(NodeKind::Craters(crate::graph::CraterParams { cells: 2.0, depth: 0.5, rim: 0.5, density: 1.0, ..Default::default() }), false);
        let (lo, hi) = c.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(lo < -0.4 && hi > 0.1, "bowls down to {lo}, rims up to {hi}");
    }
}
