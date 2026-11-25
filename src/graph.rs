//! Node graph model, save/load, and tiling analysis.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type NodeId = u32;

pub use crate::graph_params::*;
use crate::graph_params::{default_smooth, default_true, migrate_noise_kinds, one, tenth};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Float(f32),
    Vec3([f32; 3]),
    /// Inputs: a, b (and the blend width for smooth min/max). The stored values are used when an
    /// input is unconnected.
    Math {
        op: MathOp,
        a: f32,
        b: f32,
        #[serde(default = "default_smooth")]
        c: f32,
    },
    /// Linear map of input 0 from [from min, from max] to [to min, to max]; the other four inputs
    /// are those bounds. `vals` holds the unconnected defaults of all five inputs.
    Remap { vals: [f32; 5], clamp: bool },
    /// Splits a vector input into three float outputs (output ports 0 X, 1 Y, 2 Z: see `Edge::from_port`).
    Separate,
    /// A vector from three floats; `vals` holds the unconnected defaults of X, Y and Z.
    Combine { vals: [f32; 3] },
    /// A random number between `vals[0]` (min) and `vals[1]` (max), the same everywhere: it comes from the global
    /// seed and the node's id, or from `seed` when that is set. Inputs are min and max; `vals` holds their defaults.
    Random { vals: [f32; 2], seed: Option<u32> },
    /// Blends A into B: `a + (b - a) * factor`, the factor limited to 0..1 when `clamp`. Inputs are
    /// factor, A and B; `vals` holds their unconnected defaults.
    Mix { vals: [f32; 3], clamp: bool },
    /// A function of input 0 drawn through `points` (x, y), at most `MAX_CURVE_POINTS`; x need not be
    /// sorted (use `curve_points`). Outside the first and last x the output holds the end values.
    Curve { points: Vec<[f32; 2]>, mode: CurveMode },
    /// Distance to the nearest cell centre (or, with `corners`, the nearest cell corner) in a regular
    /// grid of `counts` cells per tile, scaled to [-1, 1]: -1 at the reference point and +1 as far from
    /// it as the grid gets. Input: position (defaults to world position). Saved as "Brick" before
    /// the rename.
    #[serde(alias = "Brick")]
    Grid {
        counts: [f32; 3],
        #[serde(default)]
        corners: bool,
        /// Square cells (the largest offset along any axis, so the cells are sharp-cornered boxes) instead of round ones.
        #[serde(default = "default_true")]
        cubic: bool,
        /// Scales the output (see `lower`: it becomes a Multiply after the node).
        #[serde(default = "one")]
        amount: f32,
    },
    /// A regular pattern (see `PatternParams`). Input: position (defaults to world position).
    Pattern(PatternParams),
    /// The world position, each axis multiplied by the matching input (X, Y, Z; `scale` holds their defaults, 1).
    Position { scale: [f32; 3] },
    /// Position in cylindrical or spherical coordinates around a centre and an axis (see `RadialParams`). Input: position (defaults to world position).
    Radial(RadialParams),
    /// The painted mask `name` of the Landscape tab at the position (x and y across the terrain, 0 to 1): a value from 0 to 1.
    /// Input: position (defaults to world position).
    Mask { name: String },
    /// The slope of its input field over x and y across the terrain, in degrees: `atan(scale * |gradient|)`. The
    /// Landscape tab sets `scale` to the height divided by the size, so one unit of the field is one height.
    Slope {
        #[serde(default = "one")]
        scale: f32,
    },
    /// Landscape graph: hydraulic and thermal erosion of its input (a height field) on the terrain's working grid.
    /// Outputs: the eroded height (in the input's units), then flow, deposition and wear (0 to 1). Not a function: the
    /// Landscape tab runs it before compiling and puts terrain maps in its place (see `landscape_erosion_node`).
    Erosion(crate::landscape_erosion::Erosion),
    /// Landscape graph: keeps a range of its input with soft edges: 1 from `lo` to `hi`, falling to 0 over `soft` beyond
    /// either end (smoothly). The selector terrain artists know from other tools; wire a height, slope or map in.
    Select {
        #[serde(default)]
        lo: f32,
        #[serde(default = "one")]
        hi: f32,
        #[serde(default = "tenth")]
        soft: f32,
    },
    /// Landscape graph output: a paint layer. Its input is the layer's weight (0 to 1); exports normalise all layers to
    /// sum to 1 per cell and write one weightmap each.
    Layer(crate::landscape_layers::LayerParams),
    /// Landscape graph: an image (PNG or EXR) as a terrain map, read at the position (x and y across the terrain, 0 to 1).
    /// Input: position (defaults to world position). Heights come out in the graph's units (see `landscape_heightmap`).
    Heightmap(crate::landscape_heightmap::HeightmapParams),
    /// Landscape graph: an `.sdf` shape pressed into the height at input 0 (see `landscape_stamp`).
    Stamp(crate::landscape_stamp::StampParams),
    /// A waypoint on a wire: passes its input through (every reader skips it, like a muted node).
    Reroute,
    /// The named end of a wire (Unreal's named reroute): passes its input on, and any `NamedRerouteUse` with the
    /// same name reads it without a wire.
    NamedReroute { name: String },
    /// Reads the value of the `NamedReroute` with its name (the lowest node id wins when there are several).
    NamedRerouteUse { name: String },
    /// Input: the field to differentiate. Output: unit gradient (points low → high unless flipped).
    Normal { flip: bool },
    /// Input: position (defaults to world position).
    Noise(NoiseParams),
    OutDensity,
    OutNormal,
    /// Mesh graph: a mesh file as a field (its signed distance). Input: position (defaults to world position).
    LoadMesh(crate::meshgraph::LoadMeshParams),
    /// A composition from the SDF Composer as a field (its signed distance). Input: position (defaults to world position).
    LoadSdf(crate::sdf_node::LoadSdfParams),
    /// Mesh graph: copies of an SDF shape planted across a terrain by a rule (see scatter.rs). Inputs: the field to scatter
    /// onto, and an optional displacement for the copies.
    Scatter(crate::scatter::ScatterParams),
    /// Mesh graph: a noise graph as a node. Input: position (defaults to world position).
    NoiseGraph(crate::meshgraph::NoiseGraphParams),
    /// Mesh graph: the surface of a field as an adaptive triangle mesh of about `triangles` triangles.
    Remesh {
        triangles: u32,
        /// 0: even triangles everywhere; 1: strongly smaller ones where the surface bends (the default).
        #[serde(default = "one")]
        adaptivity: f32,
    },
    /// Mesh graph: UVs for a mesh.
    AutoUv(crate::uvgen::UvParams),
    /// Mesh graph: maps baked onto a mesh (input 0) from a field (input 1); one image output per map.
    Bake(crate::bake::Settings),
    /// Mesh graph: writes a mesh.
    OutMesh(crate::meshgraph::OutParams),
    /// Mesh graph: writes an image.
    OutImage(crate::meshgraph::OutParams),
    /// Mesh graph: the UVs of a mesh (input 2, a Load Mesh's mesh output) carried onto another (input 1), with the interior of each island relaxed (borders stay put) so the coarser mesh is less stretched.
    TransferUv,
    /// Mesh graph: an image split into its red, green, blue and alpha channels (four grey images).
    ImageSplit,
    /// Mesh graph: four grey images (red, green, blue, alpha) combined into one. Unconnected channels are 0, alpha 1.
    ImageCombine,
    /// Mesh graph: `value * scale + offset`, optionally inverted afterwards, clamped to 0..1.
    ImageLevels { scale: f32, offset: f32, invert: bool },
}

impl NodeKind {
    /// How many output pins the node has (a Bake node has one image per map, Separate XYZ three).
    pub fn output_count(&self) -> usize {
        match self {
            NodeKind::Separate => 3,
            NodeKind::ImageSplit | NodeKind::Erosion(_) => 4,
            // The second output of a generator is its displacement vector.
            NodeKind::LoadMesh(_) | NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. } => 2,
            NodeKind::Bake(s) => (s.channels.len() + s.custom.len()).max(1),
            NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::Layer(_) => 0,
            _ => 1,
        }
    }

    /// The numeric parameters the node can take from an input pin (ports 1 and up, in this order).
    /// The world position with every axis scale at 1.
    pub fn position() -> NodeKind {
        NodeKind::Position { scale: [1.0; 3] }
    }

    pub fn param_pins(&self) -> &'static [ParamPin] {
        match self {
            NodeKind::Noise(_) => &NOISE_PINS,
            NodeKind::Pattern(_) => &PATTERN_PINS,
            NodeKind::Grid { .. } => &GRID_PINS,
            _ => &[],
        }
    }

    pub fn input_count(&self) -> usize {
        match self {
            NodeKind::Float(_) | NodeKind::Vec3(_) => 0,
            NodeKind::Position { .. } => 3,
            NodeKind::Reroute | NodeKind::NamedReroute { .. } => 1,
            NodeKind::NamedRerouteUse { .. } => 0,
            NodeKind::Math { op, .. } => if op.is_ternary() { 3 } else if op.is_unary() { 1 } else { 2 },
            NodeKind::Remap { .. } => 5,
            NodeKind::Random { .. } => 2,
            NodeKind::Mix { .. } | NodeKind::Combine { .. } => 3,
            NodeKind::Separate => 1,
            NodeKind::Curve { .. } => 1,
            NodeKind::Noise(_) | NodeKind::Grid { .. } | NodeKind::Pattern(_) => 1 + self.param_pins().len(),
            NodeKind::Normal { .. } | NodeKind::Radial(_) | NodeKind::Mask { .. } | NodeKind::Slope { .. } | NodeKind::Erosion(_) | NodeKind::Select { .. } | NodeKind::Layer(_) | NodeKind::Heightmap(_) | NodeKind::Stamp(_) | NodeKind::OutDensity | NodeKind::OutNormal => 1,
            NodeKind::Scatter(_) => 2,
            NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageLevels { .. } => 1,
            NodeKind::Bake(s) => 2 + s.custom.len(),
            NodeKind::ImageCombine => 4,
            NodeKind::TransferUv => 2,
        }
    }

    /// The stored value an unconnected input falls back to, for nodes with such inputs.
    pub fn const_input(&self, port: usize) -> Option<f32> {
        match self {
            NodeKind::Math { a, b, c, .. } if port < self.input_count() => Some([*a, *b, *c][port]),
            NodeKind::Remap { vals, .. } => vals.get(port).copied(),
            NodeKind::Mix { vals, .. } => vals.get(port).copied(),
            NodeKind::Combine { vals } => vals.get(port).copied(),
            NodeKind::Position { scale } => scale.get(port).copied(),
            NodeKind::Random { vals, .. } => vals.get(port).copied(),
            _ => None,
        }
    }

    pub fn set_const_input(&mut self, port: usize, v: f32) {
        match self {
            NodeKind::Math { a, b, c, .. } => {
                if let Some(x) = [a, b, c].into_iter().nth(port) {
                    *x = v;
                }
            }
            NodeKind::Remap { vals, .. } => {
                if let Some(x) = vals.get_mut(port) {
                    *x = v;
                }
            }
            NodeKind::Mix { vals, .. } => {
                if let Some(x) = vals.get_mut(port) {
                    *x = v;
                }
            }
            NodeKind::Combine { vals } => {
                if let Some(x) = vals.get_mut(port) {
                    *x = v;
                }
            }
            NodeKind::Random { vals, .. } => {
                if let Some(x) = vals.get_mut(port) {
                    *x = v;
                }
            }
            NodeKind::Position { scale } => {
                if let Some(x) = scale.get_mut(port) {
                    *x = v;
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Position in the graph editor.
    pub pos: [f32; 2],
    /// Preview-only cache of this node's value: None = off, Some(0) = automatic resolution,
    /// Some(n) = n³ texels per tile. Export always ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<u32>,
    /// Bypassed: whatever reads this node gets its first input instead (or nothing if that is unconnected).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub muted: bool,
    /// A name shown in the node header instead of its type (for organizing a big graph).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Names (`ParamPin::name`) of the numeric parameters shown as input pins: the pin takes over from the field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pins: Vec<String>,
}

/// A numeric parameter of a Noise, Pattern or Grid node that can be exposed as an input pin.
pub struct ParamPin {
    pub name: &'static str,
    pub label: &'static str,
    /// Its place in the node's parameter slots; `None` is the amount, which `Graph::lowered` turns into a Multiply.
    pub slot: Option<usize>,
}

const fn pin(name: &'static str, label: &'static str, slot: Option<usize>) -> ParamPin {
    ParamPin { name, label, slot }
}

const NOISE_PINS: [ParamPin; 7] = [
    pin("scale", "Frequency", Some(0)),
    pin("octaves", "Octaves", Some(2)),
    pin("lacunarity", "Lacunarity", Some(3)),
    pin("roughness", "Roughness", Some(4)),
    pin("distortion", "Distortion", Some(5)),
    pin("jitter", "Jitter", Some(9)),
    pin("amount", "Amplitude", None),
];
const PATTERN_PINS: [ParamPin; 11] = [
    pin("count_x", "Count X", Some(0)),
    pin("count_y", "Count Y", Some(1)),
    pin("count_z", "Count Z", Some(2)),
    pin("steps", "Offset steps", Some(3)),
    pin("gap", "Gap", Some(4)),
    pin("bevel", "Softness", Some(5)),
    pin("radius", "Radius", Some(11)),
    pin("width", "Width", Some(16)),
    pin("softness", "Edge softness", Some(17)),
    pin("iterations", "Iterations", Some(18)),
    pin("amount", "Amplitude", None),
];
const GRID_PINS: [ParamPin; 4] = [pin("count_x", "Count X", Some(0)), pin("count_y", "Count Y", Some(1)), pin("count_z", "Count Z", Some(2)), pin("amount", "Amplitude", None)];

impl Node {
    /// The exposed parameter pins with their input ports (port 1 + the pin's place in `NodeKind::param_pins`).
    pub fn exposed(&self) -> impl Iterator<Item = (usize, &'static ParamPin)> + '_ {
        self.kind.param_pins().iter().enumerate().filter(|(_, p)| self.pins.iter().any(|n| n == p.name)).map(|(i, p)| (i + 1, p))
    }

    /// Whether an input port is shown: the parameter ports of a node only while exposed.
    pub fn port_visible(&self, port: usize) -> bool {
        port == 0 || self.kind.param_pins().is_empty() || self.exposed().any(|(p, _)| p == port)
    }
}

/// An edge is "output `from_port` of `from` → input `port` of `to`". Nearly every node has one output (port 0);
/// Separate XYZ has three.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub from_port: usize,
    pub to: NodeId,
    pub port: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, Node>,
    pub edges: Vec<Edge>,
    /// Comment boxes drawn behind the nodes; they have no effect on the result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frames: Vec<Frame>,
    /// Named node settings that scripts and batches may override (`--param name=value`), see `Exposed`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exposed: Vec<Exposed>,
}

/// A node setting with a name: `field` is the setting's name in the .noisegraph file (for example `scale` on a
/// Noise node, `value` on a Value node, `weight`...), so the name stands for "node N's field".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Exposed {
    pub name: String,
    pub node: NodeId,
    pub field: String,
}

impl Graph {
    /// The current value of an exposed setting as JSON, or why it cannot be read.
    pub fn exposed_value(&self, e: &Exposed) -> Result<serde_json::Value, String> {
        let node = self.nodes.get(&e.node).ok_or(format!("there is no node {}", e.node))?;
        let kind = serde_json::to_value(&node.kind).map_err(|x| x.to_string())?;
        if e.field == "value"
            && let Some(v) = kind.as_object().and_then(|m| m.values().next()).filter(|v| v.is_number()) {
                return Ok(v.clone());
            }
        kind.as_object().and_then(|m| m.values().next()).and_then(|v| v.get(e.field.as_str())).cloned().ok_or(format!("node {} has no setting {}", e.node, e.field))
    }
}

/// A labelled box in the graph editor for grouping nodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub title: String,
    /// Top-left corner and size in graph units.
    pub pos: [f32; 2],
    pub size: [f32; 2],
}

impl Graph {
    /// The graph with every Noise and Pattern `amount` written out as a Multiply node, so codegen, the
    /// `.noiser` writer and the other hosts never see it. The Multiply takes the node's id (and so its
    /// cache, eye preview and readers); the noise itself moves to a new id and takes over the position wire.
    /// A muted node is left alone: bypassing it ignores the amount too.
    pub fn lowered(&self) -> Graph {
        let mut g = self.clone();
        for (id, n) in &self.nodes {
            let amount = match &n.kind {
                NodeKind::Noise(p) => p.amount,
                NodeKind::Pattern(p) => p.amount,
                NodeKind::Grid { amount, .. } => *amount,
                _ => continue,
            };
            // A wired amount pin replaces the field: its wire goes to the Multiply's second input.
            let amount_port = n.exposed().find(|(_, p)| p.name == "amount").map(|(port, _)| port).filter(|port| self.edges.iter().any(|e| e.to == *id && e.port == *port));
            if (amount == 1.0 && amount_port.is_none()) || n.muted {
                continue;
            }
            let mut inner = n.clone();
            inner.cache = None;
            match &mut inner.kind {
                NodeKind::Noise(p) => p.amount = 1.0,
                NodeKind::Pattern(p) => p.amount = 1.0,
                NodeKind::Grid { amount, .. } => *amount = 1.0,
                _ => {}
            }
            let new = g.nodes.keys().next_back().map_or(0, |k| k + 1);
            g.nodes.insert(new, inner);
            for e in g.edges.iter_mut().filter(|e| e.to == *id) {
                if Some(e.port) == amount_port {
                    e.port = 1;
                } else {
                    e.to = new;
                }
            }
            let m = g.nodes.get_mut(id).unwrap();
            m.kind = NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: amount, c: default_smooth() };
            g.connect(new, *id, 0);
        }
        // The displacement output of a generator (output 1): its unit gradient times its value.
        let generators: BTreeSet<NodeId> = self
            .edges
            .iter()
            .filter(|e| e.from_port == 1 && self.nodes.get(&e.from).is_some_and(|n| !n.muted && matches!(n.kind, NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. })))
            .map(|e| e.from)
            .collect();
        for id in generators {
            let normal = g.add(NodeKind::Normal { flip: false }, [0.0; 2]);
            let times = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.0, c: default_smooth() }, [0.0; 2]);
            for e in g.edges.iter_mut().filter(|e| e.from == id && e.from_port == 1) {
                (e.from, e.from_port) = (times, 0);
            }
            g.connect(id, normal, 0);
            g.connect(normal, times, 0);
            g.connect(id, times, 1);
        }
        g
    }

    pub fn add(&mut self, kind: NodeKind, pos: [f32; 2]) -> NodeId {
        let id = self.nodes.keys().next_back().map_or(0, |k| k + 1);
        self.nodes.insert(id, Node { kind, pos, cache: None, muted: false, label: None, pins: Vec::new() });
        id
    }

    /// Connects `from` into input `port` of `to`, replacing any existing link on that input.
    pub fn connect(&mut self, from: NodeId, to: NodeId, port: usize) {
        self.connect_typed(from, 0, to, port);
    }

    fn connect_typed(&mut self, from: NodeId, from_port: usize, to: NodeId, port: usize) {
        // Meshes and images only go to inputs of their own kind.
        let special = |id: &NodeId| matches!(self.nodes.get(id).map(|n| &n.kind), Some(NodeKind::LoadMesh(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv));
        if (special(&from) || special(&to)) && self.nodes.contains_key(&from) && self.nodes.contains_key(&to) && !self.output_type_at(from, from_port).connects_to(self.input_type(to, port)) {
            return;
        }
        self.edges.retain(|e| !(e.to == to && e.port == port));
        self.edges.push(Edge { from, from_port, to, port });
    }

    /// Like `connect`, from output `from_port` of `from`.
    pub fn connect_from(&mut self, from: NodeId, from_port: usize, to: NodeId, port: usize) {
        self.connect_typed(from, from_port, to, port);
    }

    /// The node feeding `port` of `to` and which of its outputs, looking through muted nodes and reroutes.
    pub fn input_from(&self, to: NodeId, port: usize) -> Option<(NodeId, usize)> {
        let e = self.edges.iter().find(|e| e.to == to && e.port == port)?;
        self.follow(e.from, e.from_port)
    }

    /// Looks through muted nodes, reroutes and named reroutes (a use jumps to its declaration's source).
    fn follow(&self, mut src: NodeId, mut from_port: usize) -> Option<(NodeId, usize)> {
        for _ in 0..=2 * self.nodes.len() {
            let Some(n) = self.nodes.get(&src) else { return Some((src, from_port)) };
            let next_from = match &n.kind {
                NodeKind::NamedRerouteUse { name } => self.named_declaration(name)?,
                k if n.muted || matches!(k, NodeKind::Reroute | NodeKind::NamedReroute { .. }) => src,
                _ => return Some((src, from_port)),
            };
            let e = self.edges.iter().find(|e| e.to == next_from && e.port == 0)?;
            (src, from_port) = (e.from, e.from_port);
        }
        None // a loop
    }

    /// The declaration a name refers to.
    pub fn named_declaration(&self, name: &str) -> Option<NodeId> {
        self.nodes.iter().find(|(_, n)| matches!(&n.kind, NodeKind::NamedReroute { name: d } if d == name)).map(|(id, _)| *id)
    }

    /// The names declared by named reroutes, sorted.
    pub fn declared_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.nodes.values().filter_map(|n| if let NodeKind::NamedReroute { name } = &n.kind { Some(name.clone()) } else { None }).collect();
        v.sort();
        v.dedup();
        v
    }

    pub fn remove(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    /// Drops links into inputs a node no longer has (a Math node switched away from smooth min/max).
    pub fn prune_edges(&mut self) {
        let nodes = &self.nodes;
        // A wire also goes when its output pin is gone (a Bake node with a map unticked).
        self.edges.retain(|e| nodes.get(&e.to).is_some_and(|n| e.port < n.kind.input_count() && n.port_visible(e.port)) && nodes.get(&e.from).is_some_and(|n| e.from_port == 0 || e.from_port < n.kind.output_count()));
    }

    pub fn disconnect(&mut self, to: NodeId, port: usize) {
        self.edges.retain(|e| !(e.to == to && e.port == port));
    }

    /// True if `target` is `id` or anywhere upstream of it. Connecting `a → b` makes a cycle iff `a` depends on `b`.
    pub fn depends_on(&self, id: NodeId, target: NodeId) -> bool {
        let mut stack = vec![id];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(n) = stack.pop() {
            if n == target {
                return true;
            }
            if seen.insert(n) {
                stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
            }
        }
        false
    }

    /// The node feeding `port` of `to`, looking through muted nodes (which pass their first input on).
    pub fn input(&self, to: NodeId, port: usize) -> Option<NodeId> {
        self.input_from(to, port).map(|(src, _)| src)
    }

    pub fn find(&self, pred: impl Fn(&NodeKind) -> bool) -> Option<NodeId> {
        self.nodes.iter().find(|(_, n)| pred(&n.kind)).map(|(id, _)| *id)
    }

    #[allow(dead_code)] // save/load UI arrives with the node editor (milestone 2)
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("graph serializes")
    }

    #[allow(dead_code)]
    pub fn from_json(s: &str) -> Result<Self, String> {
        let mut v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        migrate_noise_kinds(&mut v);
        let g: Graph = serde_json::from_value(v).map_err(|e| e.to_string())?;
        for e in &g.edges {
            let to = g.nodes.get(&e.to).ok_or(format!("edge to missing node {}", e.to))?;
            g.nodes.get(&e.from).ok_or(format!("edge from missing node {}", e.from))?;
            if e.port >= to.kind.input_count() {
                return Err(format!("edge into node {} uses port {} which does not exist", e.to, e.port));
            }
        }
        Ok(g)
    }

    /// A copy whose Density output reads `node` instead, for previewing an intermediate node.
    pub fn previewing(&self, node: NodeId) -> Graph {
        let mut g = self.clone();
        let out = match g.find(|k| *k == NodeKind::OutDensity) {
            Some(o) => o,
            None => g.add(NodeKind::OutDensity, [0.0, 0.0]),
        };
        if node != out && g.nodes.contains_key(&node) {
            g.connect(node, out, 0);
        }
        g
    }

    /// Whether `id` can be cached, or why not. A cache bakes one tile of a float field, so the
    /// node's output must be a float that repeats every tile.
    pub fn cache_eligibility(&self, id: NodeId, tiling: &BTreeMap<NodeId, Tiling>) -> Result<(), String> {
        match &self.nodes.get(&id).ok_or("no such node")?.kind {
            NodeKind::Noise(_) | NodeKind::Math { .. } | NodeKind::Remap { .. } | NodeKind::Mix { .. } | NodeKind::Curve { .. } | NodeKind::Grid { .. } | NodeKind::Pattern(_) => {}
            _ => return Err("only Noise, Math, Remap, Mix, Curve, Grid and pattern nodes can be cached".into()),
        }
        // Math on a vector input yields a vector (except dot); caches hold floats.
        if matches!(self.nodes[&id].kind, NodeKind::Math { .. }) && self.yields_vector(id) {
            return Err("this Math node outputs a vector; caches hold floats".into());
        }
        match tiling.get(&id) {
            Some(Tiling::Periodic) | Some(Tiling::Const) => Ok(()),
            Some(Tiling::Broken(why)) => Err(format!("it doesn't tile ({why})")),
            _ => Err("its value depends directly on position".into()),
        }
    }

    /// Nodes the preview no longer evaluates because every path from them to `roots` passes
    /// through an active cache (the caches themselves excluded).
    pub fn cache_covered(&self, tiling: &BTreeMap<NodeId, Tiling>, roots: &[NodeId]) -> BTreeSet<NodeId> {
        let cached = |id: NodeId| self.nodes[&id].cache.is_some() && self.cache_eligibility(id, tiling).is_ok();
        let upstream = |starts: Vec<NodeId>, stop_at_cache: bool| {
            let (mut stack, mut seen) = (starts, BTreeSet::new());
            while let Some(n) = stack.pop() {
                if self.nodes.contains_key(&n) && seen.insert(n) && !(stop_at_cache && cached(n)) {
                    stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
                }
            }
            seen
        };
        let evaluated = upstream(roots.to_vec(), true);
        let behind: Vec<_> = self.edges.iter().filter(|e| cached(e.to)).map(|e| e.from).collect();
        upstream(behind, false).difference(&evaluated).copied().collect()
    }

    /// Whether what feeds `port` of `to` is a vector: the output pin matters (a generator's second output is one).
    pub fn input_is_vector(&self, to: NodeId, port: usize) -> bool {
        self.input_from(to, port).is_some_and(|(src, fp)| self.output_type_at(src, fp) == PortType::Vector)
    }

    pub fn yields_vector(&self, id: NodeId) -> bool {
        match &self.nodes[&id].kind {
            NodeKind::Vec3(_) | NodeKind::Position { .. } | NodeKind::Radial(_) | NodeKind::Normal { .. } | NodeKind::Combine { .. } => true,
            NodeKind::Reroute | NodeKind::NamedReroute { .. } => self.input_is_vector(id, 0),
            NodeKind::NamedRerouteUse { name } => self.named_declaration(name).is_some_and(|d| self.input_is_vector(d, 0)),
            NodeKind::Math { op, .. } => *op != MathOp::Dot && (0..2).any(|p| self.input_is_vector(id, p)),
            _ => false,
        }
    }

    /// Cache resolution: the stored one, or for automatic about 4 texels per finest noise cell
    /// upstream (a power of two, 16–512).
    pub fn cache_resolution(&self, id: NodeId) -> u32 {
        match self.nodes.get(&id).and_then(|n| n.cache) {
            Some(r) if r > 0 => r,
            _ => {
                let mut finest = 0.0f32;
                let mut stack = vec![id];
                let mut seen = std::collections::BTreeSet::new();
                while let Some(n) = stack.pop() {
                    if !seen.insert(n) {
                        continue;
                    }
                    match &self.nodes[&n].kind {
                        NodeKind::Noise(p) => {
                            let oct = p.octaves_used();
                            finest = finest.max(p.scale * p.lacunarity.max(1.0).powi(oct as i32 - 1));
                        }
                        NodeKind::Grid { counts, .. } => finest = finest.max(counts.iter().cloned().fold(0.0, f32::max)),
                        NodeKind::Pattern(p) => finest = finest.max(p.used_axes().iter().map(|a| p.counts[*a]).fold(0.0, f32::max)),
                        _ => {}
                    }
                    stack.extend(self.edges.iter().filter(|e| e.to == n).map(|e| e.from));
                }
                ((finest * 4.0).ceil().max(16.0) as u32).next_power_of_two().min(512)
            }
        }
    }

    /// Position → Perlin → Density, plus an unconnected Normal output.
    /// Names of the example graphs offered by File > New from template.
    pub const TEMPLATES: [&'static str; 6] = ["Clouds", "Ridged mountains", "Cracked rock", "Brick wall", "Gyroid foam", "Warped cells"];

    /// An example graph, wired to both outputs. Unknown names give the default graph.
    pub fn template(name: &str) -> Self {
        let mut g = Graph::default();
        let noise = |kind, scale: f32, octaves| NodeKind::Noise(NoiseParams { kind, scale, octaves, ..NoiseParams::default() });
        let last = match name {
            "Clouds" => g.add(noise(NoiseKind::Perlin, 4.0, 5), [200.0, 0.0]),
            "Ridged mountains" => {
                let n = g.add(noise(NoiseKind::Simplex, 4.0, 6), [200.0, 0.0]);
                let r = g.add(NodeKind::Math { op: MathOp::Ridge, a: 0.0, b: 0.0, c: 0.5 }, [400.0, 0.0]);
                g.connect(n, r, 0);
                r
            }
            "Cracked rock" => {
                let cells = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Worley, feature: WorleyFeature::F2MinusF1, scale: 6.0, ..NoiseParams::default() }), [200.0, 0.0]);
                let grain = g.add(noise(NoiseKind::Simplex, 12.0, 4), [200.0, 200.0]);
                let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [400.0, 80.0]);
                g.connect(cells, sum, 0);
                g.connect(grain, sum, 1);
                sum
            }
            "Brick wall" => g.add(NodeKind::Pattern(PatternParams::for_kind(PatternKind::Bricks)), [200.0, 0.0]),
            "Gyroid foam" => g.add(NodeKind::Pattern(PatternParams::for_kind(PatternKind::Gyroid)), [200.0, 0.0]),
            "Warped cells" => {
                // Domain warping: the position is pushed around by three noises before it feeds the cells.
                let pos = g.add(NodeKind::position(), [0.0, 0.0]);
                let warp: Vec<NodeId> = (0..3u32)
                    .map(|i| g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Perlin, scale: 2.0, octaves: 2, seed: Some(10 + i), ..NoiseParams::default() }), [200.0, i as f32 * 160.0]))
                    .collect();
                let combine = g.add(NodeKind::Combine { vals: [0.0; 3] }, [420.0, 160.0]);
                for (port, n) in warp.iter().enumerate() {
                    g.connect(*n, combine, port);
                }
                let scaled = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.15, c: 0.5 }, [600.0, 160.0]);
                g.connect(combine, scaled, 0);
                let moved = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [780.0, 80.0]);
                g.connect(pos, moved, 0);
                g.connect(scaled, moved, 1);
                let cells = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Worley, feature: WorleyFeature::Edges, scale: 5.0, ..NoiseParams::default() }), [960.0, 80.0]);
                g.connect(moved, cells, 0);
                cells
            }
            _ => return Graph::default_graph(),
        };
        // The outputs go to the right of everything else.
        let x = g.nodes.values().map(|n| n.pos[0]).fold(0.0, f32::max) + 220.0;
        let out = g.add(NodeKind::OutDensity, [x, 0.0]);
        g.add(NodeKind::OutNormal, [x, 120.0]);
        g.connect(last, out, 0);
        g
    }

    pub fn default_graph() -> Self {
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0, 0.0]);
        let noise = g.add(NodeKind::Noise(NoiseParams::default()), [200.0, 0.0]);
        let out = g.add(NodeKind::OutDensity, [400.0, 0.0]);
        g.add(NodeKind::OutNormal, [400.0, 120.0]);
        g.connect(pos, noise, 0);
        g.connect(noise, out, 0);
        g
    }
}

// ---- tiling analysis ---------------------------------------------------

/// How a node's output behaves under p → p + 1 (one tile).
#[derive(Clone, Debug, PartialEq)]
pub enum Tiling {
    /// Same everywhere.
    Const,
    /// Repeats every tile.
    Periodic,
    /// Position-like (p, plus or times whole numbers, plus periodic offsets). Repeats once fed into a noise.
    Coord,
    /// Breaks tiling. The string says why.
    Broken(String),
}

fn is_whole(x: f32) -> bool {
    x.fract() == 0.0
}

impl Graph {
    /// Tiling class of every node. Nodes in a cycle are reported as Broken.
    pub fn tiling(&self) -> BTreeMap<NodeId, Tiling> {
        let mut memo = BTreeMap::new();
        for &id in self.nodes.keys() {
            self.tiling_of(id, &mut memo, &mut Vec::new());
        }
        memo
    }

    fn tiling_of(&self, id: NodeId, memo: &mut BTreeMap<NodeId, Tiling>, stack: &mut Vec<NodeId>) -> Tiling {
        if let Some(t) = memo.get(&id) {
            return t.clone();
        }
        if stack.contains(&id) {
            return Tiling::Broken("graph has a cycle".into());
        }
        stack.push(id);
        let input = |port: usize, memo: &mut BTreeMap<NodeId, Tiling>, stack: &mut Vec<NodeId>| {
            self.input(id, port).map(|src| self.tiling_of(src, memo, stack))
        };
        use Tiling::*;
        let t = match &self.nodes[&id].kind {
            NodeKind::Float(_) | NodeKind::Vec3(_) => Const,
            NodeKind::LoadMesh(_) => Broken("a loaded mesh does not tile".into()),
            NodeKind::LoadSdf(_) => Broken("a loaded SDF does not tile".into()),
            NodeKind::Scatter(_) => Broken("scattered shapes do not tile".into()),
            NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv => Const,
            // Each axis times a whole number (a default, or a Value node's) is still a coordinate.
            NodeKind::Position { scale } => {
                let whole = (0..3).all(|p| match self.input(id, p).map(|s| &self.nodes[&s].kind) {
                    None => is_whole(scale[p]),
                    Some(NodeKind::Float(v)) => is_whole(*v),
                    Some(_) => false,
                });
                if whole { Coord } else { Broken("a position scaled by a value that is not a known whole number".into()) }
            }
            NodeKind::Radial(_) => Broken("a radial position does not tile".into()),
            NodeKind::Mask { .. } => Broken("a painted mask does not tile".into()),
            NodeKind::Erosion(_) => Broken("erosion does not tile".into()),
            NodeKind::Heightmap(_) => Broken("an imported heightmap does not tile".into()),
            NodeKind::Stamp(_) => Broken("a stamp does not tile".into()),
            NodeKind::Slope { .. } => match input(0, memo, stack).unwrap_or(Const) {
                Coord => Const, // the slope of a linear position map is constant
                t => t,
            },
            NodeKind::Reroute | NodeKind::NamedReroute { .. } => input(0, memo, stack).unwrap_or(Const),
            NodeKind::NamedRerouteUse { name } => match self.named_declaration(name).and_then(|d| self.input(d, 0)) {
                Some(src) => self.tiling_of(src, memo, stack),
                None => Const,
            },
            // One coordinate of a position is still position-like (it shifts by whole numbers with the tile).
            NodeKind::Separate => input(0, memo, stack).unwrap_or(Const),
            NodeKind::Combine { .. } => {
                // A whole position feeding one of the float inputs is averaged to a single number, which shifts by
                // a third per tile: not a coordinate any more (Separate XYZ takes one exact component).
                let averaged = (0..3).any(|p| self.input_is_vector(id, p) && input(p, memo, stack) == Some(Coord));
                let ts: Vec<Tiling> = (0..3).filter_map(|p| input(p, memo, stack)).collect();
                if averaged {
                    Broken("a position is averaged into one number here (use Separate XYZ to take a component)".into())
                } else {
                    match ts.iter().find(|t| matches!(t, Broken(_))) {
                        Some(Broken(r)) => Broken(r.clone()),
                        _ if ts.contains(&Coord) => Coord,
                        _ if ts.iter().all(|t| *t == Const) => Const,
                        _ => Periodic,
                    }
                }
            }
            NodeKind::Math { op, a, b, .. } => {
                let ta = input(0, memo, stack).unwrap_or(Const);
                let tb = input(1, memo, stack).unwrap_or(Const);
                let tc = if op.is_ternary() { input(2, memo, stack) } else { None };
                // Operand is a whole-number constant: the unconnected default, or a Float/Vec3 node.
                let whole_const = |port: usize, default: f32| match self.input(id, port).map(|s| &self.nodes[&s].kind) {
                    None => is_whole(default),
                    Some(NodeKind::Float(v)) => is_whole(*v),
                    Some(NodeKind::Vec3(v)) => v.iter().all(|x| is_whole(*x)),
                    Some(_) => false,
                };
                let t = match (&ta, &tb) {
                    (Broken(r), _) | (_, Broken(r)) => Broken(r.clone()),
                    (Const, Const) => Const,
                    (Const | Periodic, Const | Periodic) => Periodic,
                    (Coord, _) | (_, Coord) => match op {
                        MathOp::Add => Coord,
                        // A difference with a position is position-like again (-p, p - 2p, 0.5 - p): it can look constant
                        // only when the coefficients cancel, which is not tracked, so it is never called Const.
                        MathOp::Subtract => Coord,
                        MathOp::Multiply if ta == Coord && tb == Const && whole_const(1, *b) => Coord,
                        MathOp::Multiply if tb == Coord && ta == Const && whole_const(0, *a) => Coord,
                        MathOp::Multiply => Broken("position multiplied by a non-whole or non-constant value".into()),
                        // The fractional part of a position repeats every tile (as does a modulo 1).
                        MathOp::Fract => Periodic,
                        MathOp::Modulo if ta == Coord && tb == Const && self.input(id, 1).is_none() && *b == 1.0 => Periodic,
                        _ => Broken(format!("{op:?} on a position does not tile")),
                    },
                };
                // The blend width of smooth min/max joins in: it must not depend on position.
                match (t, tc) {
                    (Broken(r), _) | (_, Some(Broken(r))) => Broken(r),
                    (_, Some(Coord)) => Broken("blend width depends on position".into()),
                    (Const, Some(Periodic)) | (Periodic, Some(Const | Periodic)) => Periodic,
                    (t, _) => t,
                }
            }
            NodeKind::Curve { .. } => match input(0, memo, stack) {
                Some(Broken(r)) => Broken(r),
                Some(Coord) => Broken("A curve of a position does not tile".into()),
                Some(t) => t,
                None => Const,
            },
            NodeKind::Mix { .. } => {
                // The factor times the difference is nonlinear in a position; the rest follows the inputs.
                let ts: Vec<Tiling> = (0..3).filter_map(|p| input(p, memo, stack)).collect();
                match ts.iter().find(|t| matches!(t, Broken(_) | Coord)) {
                    Some(Broken(r)) => Broken(r.clone()),
                    Some(_) => Broken("Mix on a position does not tile".into()),
                    None if ts.iter().all(|t| *t == Const) => Const,
                    None => Periodic,
                }
            }
            NodeKind::Random { .. } => {
                let ts: Vec<Tiling> = (0..2).filter_map(|p| input(p, memo, stack)).collect();
                match ts.iter().find(|t| matches!(t, Broken(_) | Coord)) {
                    Some(Broken(r)) => Broken(r.clone()),
                    Some(_) => Broken("a random range that depends on position does not tile".into()),
                    None if ts.iter().all(|t| *t == Const) => Const,
                    None => Periodic,
                }
            }
            NodeKind::Remap { .. } => {
                // A nonlinear-looking map of position breaks tiling; the rest follows the inputs.
                let ts: Vec<Tiling> = (0..5).filter_map(|p| input(p, memo, stack)).collect();
                match ts.iter().find(|t| matches!(t, Broken(_) | Coord)) {
                    Some(Broken(r)) => Broken(r.clone()),
                    Some(_) => Broken("Remap on a position does not tile".into()),
                    None if ts.iter().all(|t| *t == Const) => Const,
                    None => Periodic,
                }
            }
            NodeKind::Pattern(p) => {
                if let Some(why) = p.tiling_problem() {
                    Broken(why)
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Grid { counts, .. } => {
                if counts.iter().any(|c| !is_whole(*c)) {
                    Broken("grid counts are not whole numbers".into())
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Noise(n) => {
                if !is_whole(n.scale) {
                    Broken(format!("scale {} is not a whole number", n.scale))
                } else if n.kind == NoiseKind::Blue && n.scale % 4.0 != 0.0 {
                    // The blue noise tables are 4³, 8³, 16³ and 32³; the tile must hold whole tables.
                    Broken(format!("blue noise scale {} is not a multiple of 4", n.scale))
                } else if n.octaves_used() > 1 && !is_whole(n.lacunarity) {
                    Broken(format!("lacunarity {} is not a whole number", n.lacunarity))
                } else {
                    match input(0, memo, stack) {
                        None | Some(Coord | Periodic) => Periodic,
                        Some(Const) => Const,
                        Some(Broken(r)) => Broken(r),
                    }
                }
            }
            NodeKind::Normal { .. } => match input(0, memo, stack).unwrap_or(Const) {
                Coord => Const, // gradient of a linear position map is constant
                t => t,
            },
            NodeKind::Select { .. } => input(0, memo, stack).unwrap_or(Const),
            NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::Layer(_) => match input(0, memo, stack).unwrap_or(Const) {
                Coord => Broken("output depends directly on position".into()),
                t => t,
            },
        };
        stack.pop();
        // A wired parameter keeps the tiling only as a uniform value, and where the field must be a whole number
        // (a frequency, a count) only as a Value node holding one.
        // shortcut: only a Value node's number is known here (not Math of values, not a Random node); upgrade by evaluating constants.
        let ports: Vec<(usize, &str)> = self.nodes[&id].exposed().map(|(port, p)| (port, p.name)).collect();
        let mut bad = None;
        for (port, name) in ports {
            let Some(src) = self.input(id, port) else { continue };
            if self.tiling_of(src, memo, stack) != Const {
                bad = Some("a wired parameter that varies with position".to_string());
            } else if ["scale", "lacunarity", "steps", "count_x", "count_y", "count_z"].contains(&name) && !matches!(&self.nodes[&src].kind, NodeKind::Float(v) if is_whole(*v)) {
                bad = Some(format!("the wired {name} is not a known whole number"));
            }
        }
        let t = match bad {
            Some(why) if !matches!(t, Broken(_)) => Broken(why),
            _ => t,
        };
        memo.insert(id, t.clone());
        t
    }
}

// ---- value range -------------------------------------------------------

/// What a pin or wire carries. Links convert implicitly (float to vector repeats, vector to float
/// averages), so this is what the pin shows, not a restriction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortType {
    Float,
    Vector,
    /// Mesh graph: a triangle mesh (not convertible to or from the others).
    Mesh,
    /// Mesh graph: a baked image.
    Image,
}

impl PortType {
    /// Whether a wire of this type may end in an input of type `to`: floats and vectors convert, meshes and images
    /// only connect to their own kind.
    pub fn connects_to(self, to: PortType) -> bool {
        let value = |t: PortType| matches!(t, PortType::Float | PortType::Vector);
        (value(self) && value(to)) || self == to
    }
}

impl Graph {
    /// The type of a node's output. A Math node favors vectors: it outputs one when either input is
    /// a vector (the other is repeated), and only a float when it has to (dot product).
    /// The type of one output pin: a Load Mesh node's second output is its mesh, the rest follow `output_type`.
    pub fn output_type_at(&self, id: NodeId, port: usize) -> PortType {
        if port == 1 && matches!(self.nodes.get(&id).map(|n| &n.kind), Some(NodeKind::LoadMesh(_))) {
            return PortType::Mesh;
        }
        if port == 1 && matches!(self.nodes.get(&id).map(|n| &n.kind), Some(NodeKind::Noise(_) | NodeKind::Pattern(_) | NodeKind::Grid { .. })) {
            return PortType::Vector;
        }
        self.output_type(id)
    }

    pub fn output_type(&self, id: NodeId) -> PortType {
        match &self.nodes[&id].kind {
            NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::TransferUv => return PortType::Mesh,
            NodeKind::Bake(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } => return PortType::Image,
            _ => {}
        }
        if self.yields_vector(id) { PortType::Vector } else { PortType::Float }
    }

    /// The type an input expects; for Math it follows what is connected.
    pub fn input_type(&self, id: NodeId, port: usize) -> PortType {
        if port > 0 && !self.nodes[&id].kind.param_pins().is_empty() {
            return PortType::Float;
        }
        match &self.nodes[&id].kind {
            NodeKind::AutoUv(_) | NodeKind::OutMesh(_) | NodeKind::TransferUv => PortType::Mesh,
            NodeKind::Bake(_) if port == 0 => PortType::Mesh,
            NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } => PortType::Image,
            NodeKind::Noise(_) | NodeKind::Grid { .. } | NodeKind::Pattern(_) | NodeKind::OutNormal | NodeKind::Separate => PortType::Vector,
            NodeKind::Math { op, .. } if port < 2 => {
                let vector = *op == MathOp::Dot || (0..2).any(|p| self.input_is_vector(id, p));
                if vector { PortType::Vector } else { PortType::Float }
            }
            _ => PortType::Float,
        }
    }
}

/// Closed interval [lo, hi]; infinite ends mean unbounded. Vectors use one interval for all components.
pub type Range = (f32, f32);

const UNBOUNDED: Range = (f32::NEG_INFINITY, f32::INFINITY);

impl Graph {
    /// A guaranteed range for the Density output, so shaders can skip the graph where the
    /// displaced surface cannot be. Unbounded when position flows through, or on cycles.
    pub fn density_range(&self) -> Range {
        match self.find(|k| *k == NodeKind::OutDensity).and_then(|o| self.input(o, 0)) {
            Some(src) => self.range_of(src, &mut Vec::new()),
            None => (0.0, 0.0),
        }
    }

    /// Why the result of the outputs does not tile, if some node feeding them breaks tiling.
    pub fn tiling_problem_upstream(&self) -> Option<String> {
        let tiling = self.tiling();
        let mut seen = BTreeSet::new();
        let mut stack: Vec<NodeId> = [NodeKind::OutDensity, NodeKind::OutNormal].iter().filter_map(|k| self.find(|n| n == k)).collect();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(Tiling::Broken(why)) = tiling.get(&id) {
                return Some(why.clone());
            }
            for port in 0..self.nodes.get(&id).map_or(0, |n| n.kind.input_count()) {
                stack.extend(self.input(id, port));
            }
        }
        None
    }

    /// The interval a node's output stays within (infinite ends when unbounded); vectors use one interval for all components.
    pub fn node_range(&self, id: NodeId) -> Range {
        if self.nodes.contains_key(&id) { self.range_of(id, &mut Vec::new()) } else { (0.0, 0.0) }
    }

    fn range_of(&self, id: NodeId, stack: &mut Vec<NodeId>) -> Range {
        if stack.contains(&id) {
            return UNBOUNDED;
        }
        stack.push(id);
        let r = match &self.nodes[&id].kind {
            NodeKind::Float(v) => (*v, *v),
            NodeKind::Vec3(v) => (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max)),
            NodeKind::Position { .. } | NodeKind::Radial(_) | NodeKind::LoadMesh(_) | NodeKind::LoadSdf(_) | NodeKind::Scatter(_) | NodeKind::NoiseGraph(_) | NodeKind::Remesh { .. } | NodeKind::AutoUv(_) | NodeKind::Bake(_) | NodeKind::OutMesh(_) | NodeKind::OutImage(_) | NodeKind::ImageSplit | NodeKind::ImageCombine | NodeKind::ImageLevels { .. } | NodeKind::TransferUv => UNBOUNDED,
            NodeKind::Separate => match self.input(id, 0) {
                Some(src) => self.range_of(src, stack),
                None => (0.0, 0.0),
            },
            NodeKind::Random { vals, .. } => {
                let mut end = |p: usize| match self.input(id, p) {
                    Some(src) => self.range_of(src, stack),
                    None => (vals[p], vals[p]),
                };
                let (a, b) = (end(0), end(1));
                (a.0.min(b.0), a.1.max(b.1))
            }
            NodeKind::Combine { vals } => {
                let mut r = (f32::MAX, f32::MIN);
                for p in 0..3 {
                    let (lo, hi) = match self.input(id, p) {
                        Some(src) => self.range_of(src, stack),
                        None => (vals[p], vals[p]),
                    };
                    r = (r.0.min(lo), r.1.max(hi));
                }
                r
            }
            NodeKind::Reroute | NodeKind::NamedReroute { .. } => match self.input(id, 0) {
                Some(src) => self.range_of(src, stack),
                None => (0.0, 0.0),
            },
            NodeKind::NamedRerouteUse { name } => match self.named_declaration(name).and_then(|d| self.input(d, 0)) {
                Some(src) => self.range_of(src, stack),
                None => (0.0, 0.0),
            },
            NodeKind::Normal { .. } => (-1.0, 1.0),
            NodeKind::Mask { .. } => (0.0, 1.0),
            // The eroded height stays within its input's range (erosion only moves material); the maps are 0 to 1.
            NodeKind::Erosion(_) => {
                let h = self.input(id, 0).map_or((0.0, 0.0), |src| self.range_of(src, stack));
                (h.0.min(0.0), h.1.max(1.0))
            }
            NodeKind::Slope { .. } => (0.0, 90.0),
            // noise.wgsl keeps every kind within [-1, 1] (Perlin and Simplex are clamped).
            NodeKind::Noise(n) => scaled(if n.normalize { (0.0, 1.0) } else { (-1.0, 1.0) }, n.amount),
            NodeKind::Math { op, a, b, c } => {
                let mut operand = |port: usize, default: f32| match self.input(id, port) {
                    Some(src) => self.range_of(src, stack),
                    None => (default, default),
                };
                let (ra, rb, rc) = (operand(0, *a), operand(1, *b), operand(2, *c));
                math_range(*op, ra, rb, rc)
            }
            NodeKind::Remap { vals, clamp } => {
                let [v, i0, i1, o0, o1] = *vals;
                let linked = |p: usize| self.input(id, p).is_some();
                let (lo, hi) = (o0.min(o1), o0.max(o1));
                if *clamp && !linked(3) && !linked(4) {
                    (lo, hi)
                } else if (1..5).any(linked) {
                    UNBOUNDED
                } else if i0 == i1 || o0 == o1 {
                    (o0, o0)
                } else {
                    let (x0, x1) = match self.input(id, 0) {
                        Some(src) => self.range_of(src, stack),
                        None => (v, v),
                    };
                    let f = |x: f32| o0 + (x - i0) / (i1 - i0) * (o1 - o0);
                    if x0.is_finite() && x1.is_finite() { (f(x0).min(f(x1)), f(x0).max(f(x1))) } else { UNBOUNDED }
                }
            }
            NodeKind::Mix { vals, clamp } => {
                let mut operand = |port: usize| match self.input(id, port) {
                    Some(src) => self.range_of(src, stack),
                    None => (vals[port], vals[port]),
                };
                let (t, a, b) = (operand(0), operand(1), operand(2));
                let hull = (a.0.min(b.0), a.1.max(b.1));
                // A factor kept inside 0..1 (by the clamp or by its range) never leaves the hull of A and B.
                if *clamp || (t.0 >= 0.0 && t.1 <= 1.0) {
                    hull
                } else {
                    UNBOUNDED
                }
            }
            NodeKind::Curve { points, .. } => {
                let p = curve_points(points);
                (p.iter().map(|q| q[1]).fold(f32::MAX, f32::min), p.iter().map(|q| q[1]).fold(f32::MIN, f32::max))
            }
            NodeKind::Grid { amount, .. } => scaled((-1.0, 1.0), *amount),
            NodeKind::Pattern(p) => scaled((-1.0, 1.0), p.amount),
            NodeKind::Select { .. } => (0.0, 1.0),
            NodeKind::Heightmap(_) | NodeKind::Stamp(_) => UNBOUNDED,
            NodeKind::OutDensity | NodeKind::OutNormal | NodeKind::Layer(_) => UNBOUNDED,
        };
        stack.pop();
        if self.pin_wired(id, Some("amount")) { UNBOUNDED } else { r }
    }

    /// Whether a wire feeds the exposed parameter pin `name` of a node, or any parameter pin (`None`).
    pub fn pin_wired(&self, id: NodeId, name: Option<&str>) -> bool {
        self.nodes[&id].exposed().any(|(port, p)| name.is_none_or(|n| n == p.name) && self.edges.iter().any(|e| e.to == id && e.port == port))
    }
}

fn scaled((lo, hi): Range, k: f32) -> Range {
    ((lo * k).min(hi * k), (lo * k).max(hi * k))
}

fn math_range(op: MathOp, (a0, a1): Range, (b0, b1): Range, (k0, k1): Range) -> Range {
    let span = |v: [f32; 4]| {
        // NaN (0 × ∞) only comes from unbounded inputs, so treat it as unbounded.
        if v.iter().any(|x| x.is_nan()) {
            return UNBOUNDED;
        }
        (v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    };
    match op {
        MathOp::Add => (a0 + b0, a1 + b1),
        MathOp::Subtract => (a0 - b1, a1 - b0),
        MathOp::Multiply => span([a0 * b0, a0 * b1, a1 * b0, a1 * b1]),
        // Safe divide returns 0 for b = 0, but b near 0 is unbounded.
        MathOp::Divide if b0 <= 0.0 && b1 >= 0.0 => UNBOUNDED,
        MathOp::Divide => span([a0 / b0, a0 / b1, a1 / b0, a1 / b1]),
        MathOp::Min => (a0.min(b0), a1.min(b1)),
        MathOp::Max => (a0.max(b0), a1.max(b1)),
        MathOp::Power => UNBOUNDED,
        // Within k/6 of the plain min or max, on one side only.
        MathOp::SmoothMin => (a0.min(b0) - k1.max(0.0) / 6.0, a1.min(b1)),
        MathOp::SmoothMax => (a0.max(b0), a1.max(b1) + k1.max(0.0) / 6.0),
        MathOp::Dot => {
            // Every component lies in the range, so |dot| <= 3 * max|a| * max|b|.
            let m = 3.0 * a0.abs().max(a1.abs()) * b0.abs().max(b1.abs());
            if m.is_finite() { (-m, m) } else { UNBOUNDED }
        }
        MathOp::Floor => (a0.floor(), a1.floor()),
        MathOp::Fract | MathOp::Greater => (0.0, 1.0),
        MathOp::Sine => (-1.0, 1.0),
        MathOp::Sqrt => (a0.max(0.0).sqrt(), a1.max(0.0).sqrt()),
        MathOp::Modulo if b0 > 0.0 => (0.0, b1),
        MathOp::Modulo => UNBOUNDED,
        MathOp::Lerp if k0 >= 0.0 && k1 <= 1.0 => (a0.min(b0), a1.max(b1)),
        MathOp::Lerp | MathOp::InverseLerp => UNBOUNDED,
        MathOp::Absolute | MathOp::Ridge => {
            let abs = (if a0 <= 0.0 && a1 >= 0.0 { 0.0 } else { a0.abs().min(a1.abs()) }, a0.abs().max(a1.abs()));
            if op == MathOp::Absolute { abs } else { (1.0 - abs.1, 1.0 - abs.0) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_displacement_output_makes_math_a_vector() {
        let mut g = Graph::default_graph(); // 0 Position -> 1 Noise -> 2 OutDensity
        let m = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 2.0, c: 0.5 }, [0.0; 2]);
        g.connect_from(1, 0, m, 0);
        assert_eq!(g.output_type(m), PortType::Float);
        g.connect_from(1, 1, m, 0);
        assert_eq!((g.output_type(m), g.input_type(m, 0)), (PortType::Vector, PortType::Vector));
    }

    #[test]
    fn density_range_follows_math() {
        let mut g = Graph::default_graph();
        assert_eq!(g.density_range(), (-1.0, 1.0));
        // noise * 3 + 0.5
        let noise = g.find(|k| matches!(k, NodeKind::Noise(_))).unwrap();
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        let mul = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 3.0, c: 0.5 }, [0.0; 2]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.5, c: 0.5 }, [0.0; 2]);
        g.connect(noise, mul, 0);
        g.connect(mul, add, 0);
        g.connect(add, out, 0);
        assert_eq!(g.density_range(), (-2.5, 3.5));
        // Position reaching the output makes it unbounded.
        let pos = g.find(|k| matches!(k, NodeKind::Position { .. })).unwrap();
        g.connect(pos, add, 1);
        assert_eq!(g.density_range(), UNBOUNDED);
        // Dividing by a range that spans zero is unbounded too.
        assert_eq!(math_range(MathOp::Divide, (1.0, 1.0), (-1.0, 1.0), (0.5, 0.5)), UNBOUNDED);
        assert_eq!(math_range(MathOp::Divide, (1.0, 2.0), (2.0, 4.0), (0.5, 0.5)), (0.25, 1.0));
    }

    fn tiling_at(g: &Graph, id: NodeId) -> Tiling {
        g.tiling()[&id].clone()
    }

    #[test]
    fn default_graph_tiles() {
        let g = Graph::default_graph();
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert_eq!(tiling_at(&g, out), Tiling::Periodic);
    }

    #[test]
    fn fractional_scale_breaks_tiling_downstream() {
        let mut g = Graph::default_graph();
        let noise = g.find(|k| matches!(k, NodeKind::Noise(_))).unwrap();
        if let NodeKind::Noise(n) = &mut g.nodes.get_mut(&noise).unwrap().kind {
            n.scale = 2.5;
        }
        let out = g.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert!(matches!(tiling_at(&g, out), Tiling::Broken(r) if r.contains("scale 2.5")));
    }

    #[test]
    fn blue_noise_scale_rule_and_octaveless_lacunarity() {
        let mut g = Graph::default();
        let mk = |kind, scale, lacunarity| NodeKind::Noise(NoiseParams { kind, scale, lacunarity, ..Default::default() });
        let blue6 = g.add(mk(NoiseKind::Blue, 6.0, 2.0), [0.0; 2]);
        let blue8 = g.add(mk(NoiseKind::Blue, 8.0, 2.0), [0.0; 2]);
        // Lacunarity is ignored by kinds without octaves.
        let white = g.add(mk(NoiseKind::White, 8.0, 2.5), [0.0; 2]);
        let perlin = g.add(mk(NoiseKind::Perlin, 8.0, 2.5), [0.0; 2]);
        assert!(matches!(tiling_at(&g, blue6), Tiling::Broken(_)));
        assert_eq!(tiling_at(&g, blue8), Tiling::Periodic);
        assert_eq!(tiling_at(&g, white), Tiling::Periodic);
        assert!(matches!(tiling_at(&g, perlin), Tiling::Broken(_)));
    }

    #[test]
    fn position_math_rules() {
        let mut g = Graph::default();
        let p = g.add(NodeKind::position(), [0.0; 2]);
        let times2 = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 2.0, c: 0.5 }, [0.0; 2]);
        let times_half = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.5, c: 0.5 }, [0.0; 2]);
        let n1 = g.add(NodeKind::Noise(NoiseParams::default()), [0.0; 2]);
        // Domain warp: noise(p + noise(p)) still tiles.
        let warp = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let n2 = g.add(NodeKind::Noise(NoiseParams::default()), [0.0; 2]);
        g.connect(p, times2, 0);
        g.connect(p, times_half, 0);
        g.connect(p, n1, 0);
        g.connect(p, warp, 0);
        g.connect(n1, warp, 1);
        g.connect(warp, n2, 0);
        assert_eq!(tiling_at(&g, times2), Tiling::Coord);
        assert!(matches!(tiling_at(&g, times_half), Tiling::Broken(_)));
        assert_eq!(tiling_at(&g, warp), Tiling::Coord);
        assert_eq!(tiling_at(&g, n2), Tiling::Periodic);

        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect(p, out, 0);
        assert!(matches!(tiling_at(&g, out), Tiling::Broken(_)));
    }

    #[test]
    fn cycle_is_reported() {
        let mut g = Graph::default();
        let a = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        let b = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(a, b, 0);
        g.connect(b, a, 0);
        assert!(matches!(tiling_at(&g, a), Tiling::Broken(_)));
    }

    #[test]
    fn old_voronoi_files_migrate_to_worley() {
        // Rewrites the default graph's noise node as an old-format file would have stored it.
        let old = |kind: &str, octaves: u64, drop_fractal: bool| {
            let mut v: serde_json::Value = serde_json::from_str(&Graph::default_graph().to_json()).unwrap();
            let p = v.pointer_mut("/nodes/1/kind/Noise").unwrap().as_object_mut().unwrap();
            p.insert("kind".into(), kind.into());
            p.insert("octaves".into(), octaves.into());
            if drop_fractal {
                p.remove("fractal");
            }
            let g = Graph::from_json(&v.to_string()).unwrap();
            match &g.nodes[&1].kind {
                NodeKind::Noise(n) => n.clone(),
                _ => unreachable!(),
            }
        };
        let v = old("Voronoi", 4, true);
        assert_eq!((v.kind, v.feature, v.fractal), (NoiseKind::Worley, WorleyFeature::CellValue, false));
        let vf = old("VoronoiFractal", 3, true);
        assert_eq!((vf.kind, vf.feature, vf.fractal, vf.octaves_used()), (NoiseKind::Worley, WorleyFeature::CellValue, true, 3));
        // A Worley that used octaves stays fractal; one with a single octave doesn't.
        assert!(old("Worley", 4, true).fractal);
        assert!(!old("Worley", 1, true).fractal);
        // The test graph saved before the merge still opens, with its Worley kept fractal.
        let g = Graph::from_json(include_str!("../SimpleTestGraph.noisegraph")).unwrap();
        assert!(g.nodes.values().any(|n| matches!(&n.kind, NodeKind::Noise(p) if p.kind == NoiseKind::Worley && p.fractal)));
        // New files keep what they say.
        assert!(!old("Worley", 4, false).fractal);
        assert_eq!(old("Worley", 4, false).octaves_used(), 1);
    }

    #[test]
    fn math_favors_vectors_and_dot_gives_a_float() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 1.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.output_type(add), PortType::Float);
        g.connect(0, add, 0); // a vector and an unconnected float
        assert_eq!((g.output_type(add), g.input_type(add, 0), g.input_type(add, 1)), (PortType::Vector, PortType::Vector, PortType::Vector));
        g.connect(1, add, 1); // a vector and a connected float
        assert_eq!(g.output_type(add), PortType::Vector);
        let dot = g.add(NodeKind::Math { op: MathOp::Dot, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!((g.output_type(dot), g.input_type(dot, 0)), (PortType::Float, PortType::Vector));
        assert_eq!((g.output_type(0), g.output_type(1)), (PortType::Vector, PortType::Float));
        assert_eq!((g.input_type(1, 0), g.input_type(2, 0), g.input_type(3, 0)), (PortType::Vector, PortType::Float, PortType::Vector));
        // Caches hold floats: a vector-valued Math node can't be cached, a dot can (when it tiles).
        assert!(g.cache_eligibility(add, &g.tiling()).is_err());
    }

    #[test]
    fn remap_grid_and_smooth_ops_in_the_graph() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise(-1..1) → 2 OutDensity
        let t = |g: &Graph, id| g.tiling()[&id].clone();
        // Smooth min/max grow a third input; switching away prunes the link into it.
        let sm = g.add(NodeKind::Math { op: MathOp::SmoothMin, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.nodes[&sm].kind.input_count(), 3);
        g.connect(1, sm, 2);
        assert_eq!(t(&g, sm), Tiling::Periodic);
        g.connect(0, sm, 2); // a position as the blend width doesn't tile
        assert!(matches!(t(&g, sm), Tiling::Broken(_)));
        g.nodes.get_mut(&sm).unwrap().kind = NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 };
        g.prune_edges();
        assert!(g.input(sm, 2).is_none());
        // Ranges.
        let grid = g.add(NodeKind::Grid { counts: [8.0; 3], corners: false, cubic: false, amount: 1.0 }, [0.0; 2]);
        g.connect(grid, 2, 0);
        assert_eq!(g.density_range(), (-1.0, 1.0));
        let remap = g.add(NodeKind::Remap { vals: [0.0, 0.0, 1.0, -5.0, 5.0], clamp: false }, [0.0; 2]);
        g.connect(1, remap, 0);
        g.connect(remap, 2, 0);
        assert_eq!(g.density_range(), (-15.0, 5.0)); // noise in [-1, 1] through a ×10 map from [0, 1]
        if let NodeKind::Remap { clamp, .. } = &mut g.nodes.get_mut(&remap).unwrap().kind {
            *clamp = true;
        }
        assert_eq!(g.density_range(), (-5.0, 5.0));
        // Tiling.
        assert_eq!(t(&g, grid), Tiling::Periodic);
        assert_eq!(t(&g, remap), Tiling::Periodic);
        g.connect(0, remap, 0);
        assert!(matches!(t(&g, remap), Tiling::Broken(_))); // remapping a position
        g.nodes.get_mut(&grid).unwrap().kind = NodeKind::Grid { counts: [8.5, 8.0, 8.0], corners: false, cubic: false, amount: 1.0 };
        assert!(matches!(t(&g, grid), Tiling::Broken(_)));
        // The node was called Brick (without the corner option) until it was renamed.
        let old = serde_json::from_str::<NodeKind>(r#"{"Brick":{"counts":[4.0,4.0,4.0]}}"#).unwrap();
        assert_eq!(old, NodeKind::Grid { counts: [4.0; 3], corners: false, cubic: true, amount: 1.0 });
        // A Math node saved before the blend width existed still loads.
        let old = r#"{"op":"Add","a":1.0,"b":2.0}"#;
        assert!(matches!(serde_json::from_str::<NodeKind>(&format!("{{\"Math\":{old}}}")), Ok(NodeKind::Math { c, .. }) if c == 0.5));
    }

    #[test]
    fn bricks_tiling_rules_and_axes() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let p = |f: &dyn Fn(&mut PatternParams)| {
            let mut p = PatternParams::default();
            f(&mut p);
            p
        };
        let tiling = |g: &mut Graph, p: PatternParams| {
            let id = g.add(NodeKind::Pattern(p), [0.0; 2]);
            let t = g.tiling()[&id].clone();
            g.remove(id);
            t
        };
        assert_eq!(tiling(&mut g, PatternParams::default()), Tiling::Periodic);
        // Rows must divide into the offset steps, or the pattern jumps at the tile edge.
        assert!(matches!(tiling(&mut g, p(&|p| p.counts[1] = 15.0)), Tiling::Broken(r) if r.contains("multiple of the row-offset")));
        assert_eq!(tiling(&mut g, p(&|p| { p.counts[1] = 15.0; p.steps = 1; })), Tiling::Periodic);
        assert_eq!(tiling(&mut g, p(&|p| { p.counts[1] = 15.0; p.steps = 3; })), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, p(&|p| p.counts[0] = 7.5)), Tiling::Broken(_)));
        // The layer axis only counts in 3D.
        assert_eq!(tiling(&mut g, p(&|p| p.counts[2] = 7.5)), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, p(&|p| { p.counts[2] = 7.5; p.mode2d = false; })), Tiling::Broken(_)));
        // Which axes are the row, the stack and the extrusion.
        assert_eq!(p(&|p| p.axis = 2).axes(), (0, 1, None));
        assert_eq!(p(&|p| p.axis = 1).axes(), (0, 2, None));
        assert_eq!(p(&|p| p.axis = 0).axes(), (1, 2, None));
        assert_eq!(p(&|p| p.mode2d = false).axes(), (0, 1, Some(2)));
        // Range, input pin, and a file written without the newer fields still loads.
        let id = g.add(NodeKind::Pattern(PatternParams::default()), [0.0; 2]);
        assert_eq!((g.nodes[&id].kind.input_count(), g.input_type(id, 0), g.input_type(id, 1)), (12, PortType::Vector, PortType::Float));
        g.connect(id, 2, 0);
        assert_eq!(g.density_range(), (-1.0, 1.0));
        let min: NodeKind = serde_json::from_str(r#"{"Pattern":{"steps":3}}"#).unwrap();
        assert!(matches!(min, NodeKind::Pattern(q) if q.steps == 3 && q.counts == [8.0, 16.0, 8.0] && q.mode2d));
    }

    #[test]
    fn unary_math_ops_have_one_input_and_the_right_ranges() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise(-1..1) → 2 OutDensity
        let m = g.add(NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        assert_eq!(g.nodes[&m].kind.input_count(), 1);
        g.connect(1, m, 0);
        g.connect(m, 2, 0);
        assert_eq!(g.density_range(), (0.0, 1.0)); // |[-1, 1]|
        g.nodes.get_mut(&m).unwrap().kind = NodeKind::Math { op: MathOp::Ridge, a: 0.0, b: 0.0, c: 0.5 };
        assert_eq!(g.density_range(), (0.0, 1.0)); // 1 - |[-1, 1]|
        assert_eq!(math_range(MathOp::Absolute, (2.0, 5.0), (0.0, 0.0), (0.0, 0.0)), (2.0, 5.0));
        assert_eq!(math_range(MathOp::Absolute, (-5.0, -2.0), (0.0, 0.0), (0.0, 0.0)), (2.0, 5.0));
        assert_eq!(math_range(MathOp::Ridge, (2.0, 5.0), (0.0, 0.0), (0.0, 0.0)), (-4.0, -1.0));
        // Switching from Add to Absolute drops a link into B.
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, add, 0);
        g.connect(0, add, 1);
        g.nodes.get_mut(&add).unwrap().kind = NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 };
        g.prune_edges();
        assert!(g.input(add, 1).is_none());
        // |noise| still tiles; |position| does not.
        assert_eq!(g.tiling()[&m], Tiling::Periodic);
        let on_position = g.add(NodeKind::Math { op: MathOp::Absolute, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(0, on_position, 0);
        assert!(matches!(g.tiling()[&on_position], Tiling::Broken(_)));
    }

    #[test]
    fn pattern_tiling_rules_per_kind() {
        let mut g = Graph::default_graph();
        let tiling = |g: &mut Graph, p: PatternParams| {
            let id = g.add(NodeKind::Pattern(p), [0.0; 2]);
            let t = g.tiling()[&id].clone();
            g.remove(id);
            t
        };
        let with = |kind: PatternKind, f: &dyn Fn(&mut PatternParams)| {
            let mut p = PatternParams::for_kind(kind);
            f(&mut p);
            p
        };
        for kind in PatternKind::ALL {
            assert_eq!(tiling(&mut g, PatternParams::for_kind(kind)), Tiling::Periodic, "{kind:?} should tile by default");
        }
        // A checker needs even counts on the axes it uses; the extrusion axis does not count.
        assert!(matches!(tiling(&mut g, with(PatternKind::Checker, &|p| p.counts[0] = 7.0)), Tiling::Broken(r) if r.contains("even")));
        assert_eq!(tiling(&mut g, with(PatternKind::Checker, &|p| p.counts[2] = 7.0)), Tiling::Periodic);
        assert!(matches!(tiling(&mut g, with(PatternKind::Checker, &|p| { p.counts[2] = 7.0; p.mode2d = false; })), Tiling::Broken(_)));
        // Stripes need a whole count and a whole direction.
        assert!(matches!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.counts[0] = 4.5)), Tiling::Broken(r) if r.contains("stripe")));
        assert!(matches!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.dir = [1.0, 0.5, 0.0])), Tiling::Broken(_)));
        assert_eq!(tiling(&mut g, with(PatternKind::Stripes, &|p| p.dir = [2.0, -3.0, 1.0])), Tiling::Periodic);
        // Dimples and gyroids need whole counts on all three axes.
        for kind in [PatternKind::Dimples, PatternKind::Gyroid] {
            assert!(matches!(tiling(&mut g, with(kind, &|p| p.counts[1] = 2.5)), Tiling::Broken(_)), "{kind:?}");
        }
        // Every kind survives a save and load.
        for kind in PatternKind::ALL {
            let n = NodeKind::Pattern(PatternParams::for_kind(kind));
            assert_eq!(serde_json::from_str::<NodeKind>(&serde_json::to_string(&n).unwrap()).unwrap(), n);
        }
    }

    #[test]
    fn cache_rules() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Perlin(scale 4, 4 octaves, lac 2) → 2 OutDensity
        let t = g.tiling();
        assert!(g.cache_eligibility(1, &t).is_ok());
        assert!(g.cache_eligibility(0, &t).is_err()); // Position
        assert!(g.cache_eligibility(2, &t).is_err()); // output
        // Finest cells 4 × 2³ = 32 → 4 texels each → 128³.
        g.nodes.get_mut(&1).unwrap().cache = Some(0);
        assert_eq!(g.cache_resolution(1), 128);
        g.nodes.get_mut(&1).unwrap().cache = Some(64);
        assert_eq!(g.cache_resolution(1), 64);
        // A non-tiling noise can't be cached.
        if let NodeKind::Noise(n) = &mut g.nodes.get_mut(&1).unwrap().kind {
            n.scale = 2.5;
        }
        assert!(g.cache_eligibility(1, &g.tiling()).is_err());
        // Math on a vector yields a vector.
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(0, m, 0);
        assert!(g.cache_eligibility(m, &g.tiling()).is_err());
    }

    #[test]
    fn cache_covers_only_nodes_no_uncached_path_needs() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let roots = [2, 3];
        assert!(g.cache_covered(&g.tiling(), &roots).is_empty());
        g.nodes.get_mut(&1).unwrap().cache = Some(0);
        assert_eq!(g.cache_covered(&g.tiling(), &roots), BTreeSet::from([0]));
        // Position also feeds an uncached branch to the output, so it's still evaluated.
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, m, 0);
        g.connect(0, m, 1);
        g.connect(m, 2, 0);
        assert!(g.cache_covered(&g.tiling(), &roots).is_empty());
    }

    #[test]
    fn previewing_rewires_density_only_in_the_copy() {
        let g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity
        let p = g.previewing(0);
        assert_eq!(p.input(2, 0), Some(0));
        assert_eq!(g.input(2, 0), Some(1));
        // Without a Density output one is added.
        let mut bare = Graph::default();
        let pos = bare.add(NodeKind::position(), [0.0; 2]);
        let pv = bare.previewing(pos);
        let out = pv.find(|k| *k == NodeKind::OutDensity).unwrap();
        assert_eq!(pv.input(out, 0), Some(pos));
    }

    #[test]
    fn depends_on_and_remove() {
        let mut g = Graph::default_graph(); // 0 Position → 1 Noise → 2 OutDensity, 3 OutNormal
        assert!(g.depends_on(2, 0));
        assert!(g.depends_on(1, 1));
        assert!(!g.depends_on(0, 1));
        assert!(!g.depends_on(3, 0));
        g.remove(1);
        assert!(g.edges.is_empty());
        assert!(!g.depends_on(2, 0));
    }

    #[test]
    fn json_round_trip_and_validation() {
        let g = Graph::default_graph();
        assert_eq!(Graph::from_json(&g.to_json()).unwrap(), g);
        let mut bad = g.clone();
        bad.edges.push(Edge { from: 0, from_port: 0, to: 2, port: 5 });
        assert!(Graph::from_json(&bad.to_json()).is_err());
    }
}

#[cfg(test)]
mod mute_tests {
    use super::*;

    #[test]
    fn a_reroute_is_transparent() {
        let mut g = Graph::default_graph(); // 0 Position -> 1 Noise -> 2 OutDensity
        let r = g.add(NodeKind::Reroute, [0.0; 2]);
        g.connect(1, r, 0);
        g.connect(r, 2, 0);
        assert_eq!(g.input(2, 0), Some(1), "the output reads the noise through the reroute");
        assert!(!g.yields_vector(r));
        assert_eq!(g.density_range(), (-1.0, 1.0));
    }

    #[test]
    fn a_muted_node_passes_its_first_input_on() {
        let mut g = Graph::default_graph(); // 0 Position -> 1 Noise -> 2 OutDensity
        assert_eq!(g.input(2, 0), Some(1));
        g.nodes.get_mut(&1).unwrap().muted = true;
        assert_eq!(g.input(2, 0), Some(0), "the output reads the Position through the muted noise");
        g.disconnect(1, 0);
        assert_eq!(g.input(2, 0), None, "nothing to pass on");
        assert!(Graph::from_json(&g.to_json()).unwrap().nodes[&1].muted, "survives a save");
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    #[test]
    fn frames_round_trip_and_old_files_have_none() {
        let mut g = Graph::default_graph();
        assert!(!g.to_json().contains("frames"), "an empty list is not written");
        g.frames.push(Frame { title: "A".into(), pos: [1.0, 2.0], size: [3.0, 4.0] });
        assert_eq!(Graph::from_json(&g.to_json()).unwrap().frames, g.frames);
        assert!(Graph::from_json(&Graph::default_graph().to_json()).unwrap().frames.is_empty());
    }

    #[test]
    fn labels_survive_a_save_and_empty_ones_are_not_written() {
        let mut g = Graph::default_graph();
        assert!(!g.to_json().contains("label"));
        g.nodes.get_mut(&1).unwrap().label = Some("Base noise".into());
        assert_eq!(Graph::from_json(&g.to_json()).unwrap().nodes[&1].label.as_deref(), Some("Base noise"));
    }

    #[test]
    fn the_export_warning_finds_a_broken_node_upstream_of_the_outputs() {
        let mut g = Graph::default_graph(); // Position -> Noise (scale 8?) -> Density
        assert_eq!(g.tiling_problem_upstream(), None);
        if let NodeKind::Noise(p) = &mut g.nodes.get_mut(&1).unwrap().kind {
            p.scale = 2.5; // a fractional scale breaks tiling
        }
        assert!(g.tiling_problem_upstream().is_some());
        // A broken node that nothing reads is not a problem.
        let mut g = Graph::default_graph();
        g.add(NodeKind::Noise(NoiseParams { scale: 2.5, ..NoiseParams::default() }), [0.0; 2]);
        assert_eq!(g.tiling_problem_upstream(), None);
    }

    #[test]
    fn old_single_component_separate_nodes_become_output_pins() {
        let old = r#"{"nodes": {"0": {"kind": "Position", "pos": [0,0]}, "1": {"kind": {"Separate": {"component": 2}}, "pos": [0,0]}, "2": {"kind": "OutDensity", "pos": [0,0]}},
                       "edges": [{"from": 0, "to": 1, "port": 0}, {"from": 1, "to": 2, "port": 0}]}"#;
        let g = Graph::from_json(old).unwrap();
        assert_eq!(g.nodes[&1].kind, NodeKind::Separate);
        assert_eq!(g.input_from(2, 0), Some((1, 2)), "the old Z component is output pin 2");
        let again = Graph::from_json(&g.to_json()).unwrap();
        assert_eq!(again.input_from(2, 0), Some((1, 2)));
        assert!(!Graph::default_graph().to_json().contains("from_port"));
    }

    #[test]
    fn a_named_reroute_use_reads_its_declaration_without_a_wire() {
        let mut g = Graph::default_graph(); // 0 Position -> 1 Noise -> 2 OutDensity
        let decl = g.add(NodeKind::NamedReroute { name: "base".into() }, [0.0; 2]);
        let usage = g.add(NodeKind::NamedRerouteUse { name: "base".into() }, [0.0; 2]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 1.0, c: 0.5 }, [0.0; 2]);
        g.connect(1, decl, 0);
        g.connect(usage, add, 0);
        g.connect(add, 2, 0);
        assert_eq!(g.input(add, 0), Some(1), "the use resolves through the declaration to the noise");
        assert_eq!(g.declared_names(), vec!["base".to_string()]);
        assert_eq!(g.density_range(), (0.0, 2.0), "ranges follow it too");
        // An unknown name reads nothing.
        if let NodeKind::NamedRerouteUse { name } = &mut g.nodes.get_mut(&usage).unwrap().kind {
            *name = "missing".into();
        }
        assert_eq!(g.input(add, 0), None);
        assert!(!Graph::from_json(&g.to_json()).unwrap().nodes.is_empty());
    }

    #[test]
    fn meshes_and_images_only_connect_to_their_own_kind() {
        let mut g = Graph::default();
        let mesh = g.add(NodeKind::LoadMesh(Default::default()), [0.0; 2]);
        let remesh = g.add(NodeKind::Remesh { triangles: 1000, adaptivity: 1.0 }, [0.0; 2]);
        let uv = g.add(NodeKind::AutoUv(Default::default()), [0.0; 2]);
        let bake = g.add(NodeKind::Bake(Default::default()), [0.0; 2]);
        let out_mesh = g.add(NodeKind::OutMesh(Default::default()), [0.0; 2]);
        let out_image = g.add(NodeKind::OutImage(Default::default()), [0.0; 2]);
        let add = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [0.0; 2]);
        g.connect(mesh, remesh, 0); // a field into Remesh: fine
        g.connect(remesh, uv, 0);
        g.connect(uv, bake, 0);
        g.connect(mesh, bake, 1);
        g.connect(uv, out_mesh, 0);
        g.connect_from(bake, 0, out_image, 0);
        assert_eq!(g.edges.len(), 6);
        // A mesh cannot feed a field input, a field cannot feed a mesh input, an image cannot feed a mesh output.
        g.connect(remesh, add, 0);
        g.connect(add, uv, 0);
        g.connect(bake, out_mesh, 0);
        assert_eq!(g.edges.len(), 6, "{:?}", g.edges);
        assert_eq!((g.output_type(remesh), g.output_type(bake), g.input_type(bake, 1)), (PortType::Mesh, PortType::Image, PortType::Float));
    }

    #[test]
    fn unticking_a_bake_map_drops_its_wire() {
        let mut g = Graph::default();
        let bake = g.add(NodeKind::Bake(crate::bake::Settings { channels: vec![crate::bake::Channel::Normal, crate::bake::Channel::Occlusion], ..Default::default() }), [0.0; 2]);
        let out = g.add(NodeKind::OutImage(Default::default()), [0.0; 2]);
        g.connect_from(bake, 1, out, 0);
        g.prune_edges();
        assert_eq!(g.edges.len(), 1);
        if let NodeKind::Bake(b) = &mut g.nodes.get_mut(&bake).unwrap().kind {
            b.channels.retain(|c| *c == crate::bake::Channel::Normal);
        }
        g.prune_edges();
        assert!(g.edges.is_empty());
    }
}

#[cfg(test)]
mod example_writer {
    use super::*;

    /// Writes the example noise graphs for the library: `NOISER_DEMO_DIR=dir cargo test write_example_noises -- --ignored`.
    #[test]
    #[ignore]
    fn write_example_noises() {
        let Ok(dir) = std::env::var("NOISER_DEMO_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in Graph::TEMPLATES {
            std::fs::write(dir.join(format!("{name}.noisegraph")), Graph::template(name).to_json()).unwrap();
        }
        let finish = |mut g: Graph, last: NodeId, name: &str| {
            let x = g.nodes.values().map(|n| n.pos[0]).fold(0.0, f32::max) + 220.0;
            let out = g.add(NodeKind::OutDensity, [x, 0.0]);
            g.add(NodeKind::OutNormal, [x, 120.0]);
            g.connect(last, out, 0);
            std::fs::write(dir.join(format!("{name}.noisegraph")), g.to_json()).unwrap();
        };
        let perlin = |scale: f32, octaves: u32, seed: u32| NodeKind::Noise(NoiseParams { kind: NoiseKind::Perlin, scale, octaves, seed: Some(seed), ..NoiseParams::default() });

        // Marble: sine stripes through a position that three noises push around.
        let mut g = Graph::default();
        let pos = g.add(NodeKind::position(), [0.0, 0.0]);
        let warp: Vec<NodeId> = (0..3u32).map(|i| g.add(perlin(2.0, 3, 30 + i), [200.0, i as f32 * 160.0])).collect();
        let combine = g.add(NodeKind::Combine { vals: [0.0; 3] }, [420.0, 160.0]);
        for (port, n) in warp.iter().enumerate() {
            g.connect(*n, combine, port);
        }
        let scaled = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.35, c: 0.5 }, [600.0, 160.0]);
        g.connect(combine, scaled, 0);
        let moved = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [780.0, 80.0]);
        g.connect(pos, moved, 0);
        g.connect(scaled, moved, 1);
        let mut stripes = PatternParams::for_kind(PatternKind::Stripes);
        stripes.dir = [3.0, 1.0, 0.0];
        let veins = g.add(NodeKind::Pattern(stripes), [960.0, 80.0]);
        g.connect(moved, veins, 0);
        finish(g, veins, "Marble");

        // Hammered metal: round dimples with a little fine grain on top.
        let mut g = Graph::default();
        let dimples = g.add(NodeKind::Pattern(PatternParams::for_kind(PatternKind::Dimples)), [200.0, 0.0]);
        let grain = g.add(NodeKind::Noise(NoiseParams { kind: NoiseKind::Simplex, scale: 24.0, octaves: 3, ..NoiseParams::default() }), [200.0, 200.0]);
        let faint = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.08, c: 0.5 }, [400.0, 200.0]);
        let sum = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0, c: 0.5 }, [600.0, 80.0]);
        g.connect(grain, faint, 0);
        g.connect(dimples, sum, 0);
        g.connect(faint, sum, 1);
        finish(g, sum, "Hammered metal");

        // Honeycomb and Truchet tiles, as they come.
        for (kind, name) in [(PatternKind::Honeycomb, "Honeycomb"), (PatternKind::Truchet, "Truchet maze")] {
            let mut g = Graph::default();
            let pattern = g.add(NodeKind::Pattern(PatternParams::for_kind(kind)), [200.0, 0.0]);
            finish(g, pattern, name);
        }
    }
}
