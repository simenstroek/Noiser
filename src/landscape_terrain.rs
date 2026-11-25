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
    let result = erode(device, queue, &metres, n, s.world_m / (n - 1) as f32, params, progress)?;
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
}
