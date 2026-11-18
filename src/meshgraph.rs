//! The mesh graph: a graph (the same `Graph` as the noise graph, in its own tab) whose nodes load meshes as fields,
//! shape them with noise graphs and math, and then run the operations that make a game mesh: remesh, UVs, bake, write.
//!
//! Wire types: Float and Vector wires carry *fields* (functions of position, compiled into one shader, so a loaded
//! mesh is a field too: its signed distance), Mesh wires carry triangle meshes and Image wires baked maps. The field
//! part is lowered into one plain noise graph (`lowered_mesh`): a Noise Graph node is replaced by a copy of the
//! graph it names, so the existing compiler needs no changes except for the loaded-mesh node.

use crate::graph::{Graph, NodeKind};
use serde::{Deserialize, Serialize};

/// Loads a mesh file as a field: its signed distance (negative inside), in the units of the file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadMeshParams {
    pub path: String,
    /// Samples along the longest side of the distance grid the mesh is baked into.
    pub resolution: u32,
    /// Margin around the mesh in normalized units (the mesh is scaled to a unit sphere for baking).
    pub padding: f32,
    pub combine: crate::meshsdf::Combine,
}

impl Default for LoadMeshParams {
    fn default() -> Self {
        LoadMeshParams { path: String::new(), resolution: 128, padding: 0.5, combine: Default::default() }
    }
}

/// A noise graph used as a node: the position it reads is scaled by `frequency`, its density output by `amplitude`.
/// `file` is the source of truth; `embedded` is a copy of its JSON kept for when the file is missing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseGraphParams {
    pub file: String,
    pub embedded: String,
    pub frequency: f32,
    pub amplitude: f32,
}

impl Default for NoiseGraphParams {
    fn default() -> Self {
        NoiseGraphParams { file: String::new(), embedded: String::new(), frequency: 1.0, amplitude: 1.0 }
    }
}

/// Where an output node writes: a folder and a file name (without extension).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutParams {
    pub folder: String,
    pub name: String,
}

impl Default for OutParams {
    fn default() -> Self {
        OutParams { folder: String::new(), name: "output".into() }
    }
}

/// The noise graph a Noise Graph node refers to: the file when it can be read, else the embedded copy.
pub fn noise_graph_source(p: &NoiseGraphParams) -> Result<Graph, String> {
    let from_file = if p.file.is_empty() { None } else { std::fs::read_to_string(&p.file).ok() };
    let text = from_file.as_deref().or(if p.embedded.is_empty() { None } else { Some(p.embedded.as_str()) }).ok_or_else(|| format!("the noise graph {} cannot be read and has no embedded copy", if p.file.is_empty() { "(none chosen)" } else { &p.file }))?;
    Graph::from_json(text)
}

/// The field part of a mesh graph as one plain graph: every Noise Graph node becomes a copy of the graph it names.
/// The copy's Position nodes read the Noise Graph node's position input times `frequency`; its density output times
/// `amplitude` takes the node's place (and id), so everything downstream and the node's preview keep working.
pub fn lowered_mesh(g: &Graph) -> Result<Graph, String> {
    use crate::graph::MathOp;
    let mut out = g.clone();
    let ids: Vec<_> = g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::NoiseGraph(_)) && !n.muted).map(|(id, _)| *id).collect();
    for id in ids {
        let NodeKind::NoiseGraph(p) = g.nodes[&id].kind.clone() else { continue };
        let sub = noise_graph_source(&p)?;
        let Some(out_id) = sub.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutDensity)).map(|(i, _)| *i) else { return Err("the noise graph has no Density Output".into()) };
        // The position the sub graph sees: the node's input (or the world position) times the frequency.
        let world = out.add(NodeKind::Position, [0.0; 2]);
        let pos_in = out.edges.iter().find(|e| e.to == id && e.port == 0).map(|e| (e.from, e.from_port));
        let scaled = out.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: p.frequency, c: 0.5 }, [0.0; 2]);
        match pos_in {
            Some((from, port)) => out.connect_from(from, port, scaled, 0),
            None => out.connect(world, scaled, 0),
        }
        // Copy the sub graph's nodes under new ids.
        let mut map = std::collections::BTreeMap::new();
        for (sid, n) in &sub.nodes {
            if matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal) {
                continue;
            }
            let new = out.nodes.keys().next_back().map_or(0, |k| k + 1);
            let mut copy = n.clone();
            copy.cache = None;
            copy.kind = match &copy.kind {
                NodeKind::NamedReroute { name } => NodeKind::NamedReroute { name: format!("{id}:{name}") },
                NodeKind::NamedRerouteUse { name } => NodeKind::NamedRerouteUse { name: format!("{id}:{name}") },
                k => k.clone(),
            };
            out.nodes.insert(new, copy);
            map.insert(*sid, new);
        }
        for e in &sub.edges {
            if let (Some(&from), Some(&to)) = (map.get(&e.from), map.get(&e.to)) {
                out.edges.push(crate::graph::Edge { from, from_port: e.from_port, to, port: e.port });
            }
        }
        // Position nodes become the scaled position.
        let positions: Vec<_> = map.iter().filter(|(sid, _)| matches!(sub.nodes[*sid].kind, NodeKind::Position)).map(|(_, n)| *n).collect();
        for n in positions {
            out.nodes.get_mut(&n).unwrap().kind = NodeKind::Reroute;
            out.connect(scaled, n, 0);
        }
        // Generators with no position wire read the world position: they read the scaled one instead.
        for &n in map.values() {
            if matches!(out.nodes[&n].kind, NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. }) && !out.edges.iter().any(|e| e.to == n && e.port == 0) {
                out.connect(scaled, n, 0);
            }
        }
        // This node becomes `density * amplitude`.
        let Some(src) = sub.edges.iter().find(|e| e.to == out_id && e.port == 0) else { return Err("the noise graph's Density Output is not connected".into()) };
        let Some(&src_new) = map.get(&src.from) else { return Err("the noise graph's Density Output reads a missing node".into()) };
        out.edges.retain(|e| !(e.to == id));
        out.nodes.get_mut(&id).unwrap().kind = NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: p.amplitude, c: 0.5 };
        out.connect_from(src_new, src.from_port, id, 0);
    }
    Ok(out)
}
