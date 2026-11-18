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

// ---- running a mesh graph ----------------------------------------------------------------------------------

use crate::graph::NodeId;
use crate::mesh::Normalization;
use crate::meshexport::{Displace, GraphMesh, Job, MeshEvaluator, Settings};
use crate::surface::RawMesh;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A GPU device for running graphs without a window.
pub fn headless_gpu() -> Result<(eframe::wgpu::Device, eframe::wgpu::Queue), String> {
    use eframe::wgpu;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))
}

/// What a node of the operation part of the graph produced.
#[derive(Clone)]
enum Value {
    Mesh(RawMesh),
    Uv(crate::uvgen::UvMesh),
    Images(Vec<crate::bake::Image>),
}

/// A loaded mesh baked into a distance grid, kept between runs (the grid takes seconds to build).
pub struct LoadedMesh {
    texture: eframe::wgpu::Texture,
    view: eframe::wgpu::TextureView,
    origin: [f32; 3],
    voxel: f32,
    norm: Normalization,
    /// Bounds in the mesh file's units.
    lo: [f32; 3],
    hi: [f32; 3],
}

#[derive(Default)]
pub struct MeshCache {
    meshes: BTreeMap<String, std::sync::Arc<LoadedMesh>>,
}

pub struct Runner<'a> {
    pub device: &'a eframe::wgpu::Device,
    pub queue: &'a eframe::wgpu::Queue,
    pub cache: &'a mut MeshCache,
    /// Relative paths in the graph are relative to this folder (the graph file's).
    pub base: PathBuf,
    pub seed: u32,
    /// Progress in thousandths and log lines.
    pub progress: crate::meshexport::Progress,
    pub log: &'a mut dyn FnMut(&str),
}

impl Runner<'_> {
    fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        if path.is_absolute() { path.to_path_buf() } else { self.base.join(path) }
    }

    fn mesh(&mut self, p: &LoadMeshParams) -> Result<std::sync::Arc<LoadedMesh>, String> {
        if p.path.is_empty() {
            return Err("a Load Mesh node has no file chosen".into());
        }
        let path = self.resolve(&p.path);
        let key = format!("{}|{}|{}|{:?}|{}", path.display(), p.resolution, p.padding, p.combine.mode, p.combine.smooth);
        if let Some(m) = self.cache.meshes.get(&key) {
            return Ok(m.clone());
        }
        (self.log)(&format!("loading {}", path.display()));
        let (mesh, norm) = crate::meshformats::load(&path)?.normalized();
        let sdf = crate::meshsdf::MeshSdf::build_with(&mesh, p.resolution, p.padding, &Default::default(), p.combine);
        let (texture, view) = sdf.upload(self.device, self.queue);
        let (nlo, nhi) = mesh.bounds();
        let loaded = std::sync::Arc::new(LoadedMesh { texture, view, origin: sdf.origin, voxel: sdf.voxel, norm, lo: norm.to_object(nlo), hi: norm.to_object(nhi) });
        self.cache.meshes.insert(key, loaded.clone());
        Ok(loaded)
    }

    /// The evaluator of the field that feeds `source`, and the bounds and reach extraction needs.
    fn field(&mut self, original: &Graph, lowered: &Graph, source: (NodeId, usize), require_mesh: bool) -> Result<FieldSource, String> {
        // The cone of the source with an output node on it: that is all the compiler looks at.
        let mut g = lowered.clone();
        g.nodes.retain(|_, n| !matches!(n.kind, NodeKind::OutDensity | NodeKind::OutNormal));
        let nodes: std::collections::BTreeSet<NodeId> = g.nodes.keys().copied().collect();
        g.edges.retain(|e| nodes.contains(&e.from) && nodes.contains(&e.to));
        let out = g.add(NodeKind::OutDensity, [0.0; 2]);
        g.connect_from(source.0, source.1, out, 0);
        let compiled = crate::codegen::compile(&g)?;
        let mut loaded = Vec::new();
        let mut graph_meshes = Vec::new();
        for id in &compiled.mesh_slots {
            let NodeKind::LoadMesh(p) = &g.nodes[id].kind else { continue };
            let m = self.mesh(p)?;
            graph_meshes.push(GraphMesh { view: m.view.clone(), origin: m.origin, voxel: m.voxel, centre: m.norm.center, scale: m.norm.scale });
            loaded.push(m);
        }
        if loaded.is_empty() && require_mesh {
            return Err("the field has no Load Mesh node, so there is no surface to extract".into());
        }
        let ev = MeshEvaluator::new_graph(self.device, self.queue, &g, self.seed, graph_meshes)?;
        let (mut lo, mut hi) = if loaded.is_empty() { ([0.0; 3], [0.0; 3]) } else { ([f32::MAX; 3], [f32::MIN; 3]) };
        for m in &loaded {
            for a in 0..3 {
                lo[a] = lo[a].min(m.lo[a]);
                hi[a] = hi[a].max(m.hi[a]);
            }
        }
        let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) * 0.5;
        // How far noise graphs can move the surface: their amplitudes times the largest value of their density, plus a margin.
        let mut reach = 0.05 * radius;
        for n in original.nodes.values() {
            if let NodeKind::NoiseGraph(p) = &n.kind {
                if let Ok(sub) = noise_graph_source(p) {
                    let (a, b) = sub.density_range();
                    let size = if a.is_finite() && b.is_finite() { a.abs().max(b.abs()) } else { 1.0 };
                    reach += p.amplitude.abs() * size;
                }
            }
        }
        Ok(FieldSource { ev, lo, hi, reach: reach.min(2.0 * radius), _loaded: loaded })
    }

    fn output_path(&self, o: &OutParams, suffix: &str, ext: &str) -> PathBuf {
        let dir = if o.folder.is_empty() { self.base.clone() } else { self.resolve(&o.folder) };
        let _ = std::fs::create_dir_all(&dir);
        crate::meshapp::unique_path(&dir, &format!("{}{suffix}", o.name), ext)
    }

    /// Runs the operation nodes upstream of every output node and writes the files; returns the files written.
    pub fn run(&mut self, graph: &Graph) -> Result<Vec<PathBuf>, String> {
        let lowered = lowered_mesh(graph)?;
        let outputs: Vec<NodeId> = graph.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::OutMesh(_) | NodeKind::OutImage(_)) && !n.muted).map(|(id, _)| *id).collect();
        if outputs.is_empty() {
            return Err("the graph has no Mesh or Image output node".into());
        }
        let mut memo: BTreeMap<NodeId, Value> = BTreeMap::new();
        let mut fields: Fields = BTreeMap::new();
        let mut written = Vec::new();
        for out in outputs {
            let Some((src, port)) = graph.input_from(out, 0) else { return Err(format!("output node {out} has nothing connected")) };
            let value = self.value(graph, &lowered, src, &mut memo, &mut fields)?;
            match (&graph.nodes[&out].kind, value) {
                (NodeKind::OutMesh(o), Value::Uv(uv)) => {
                    let path = self.output_path(o, "", "obj");
                    uv.write_obj(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                    (self.log)(&format!("wrote {}", path.display()));
                    written.push(path);
                }
                (NodeKind::OutMesh(o), Value::Mesh(mesh)) => {
                    let path = self.output_path(o, "", "obj");
                    mesh.write_obj(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                    (self.log)(&format!("wrote {}", path.display()));
                    written.push(path);
                }
                (NodeKind::OutImage(o), Value::Images(images)) => {
                    let Some(image) = images.get(port) else { return Err(format!("output node {out} reads image {port}, which the Bake node does not produce")) };
                    let path = self.output_path(o, &format!("_{}", image.label), "png");
                    image.write_png(&path)?;
                    (self.log)(&format!("wrote {}", path.display()));
                    written.push(path);
                }
                (NodeKind::OutMesh(_), _) => return Err(format!("output node {out} needs a mesh")),
                _ => return Err(format!("output node {out} needs an image from a Bake node")),
            }
        }
        Ok(written)
    }

    fn field_for(&mut self, original: &Graph, lowered: &Graph, src: (NodeId, usize), require_mesh: bool, fields: &mut Fields) -> Result<std::rc::Rc<FieldSource>, String> {
        if let Some(f) = fields.get(&(src.0, src.1, require_mesh)) {
            return Ok(f.clone());
        }
        let f = std::rc::Rc::new(self.field(original, lowered, src, require_mesh)?);
        fields.insert((src.0, src.1, require_mesh), f.clone());
        Ok(f)
    }

    /// The image on input `port` of node `id`: the output of a Bake or image node it is wired to.
    fn image_in(&mut self, original: &Graph, lowered: &Graph, id: NodeId, port: usize, memo: &mut BTreeMap<NodeId, Value>, fields: &mut Fields) -> Result<Option<crate::bake::Image>, String> {
        let Some((src, sport)) = original.input_from(id, port) else { return Ok(None) };
        match self.value(original, lowered, src, memo, fields)? {
            Value::Images(images) => images.get(sport).cloned().map(Some).ok_or_else(|| format!("node {id} reads image {sport} of node {src}, which has no such output")),
            _ => Err(format!("node {id} input {} needs an image", port + 1)),
        }
    }

    fn value(&mut self, original: &Graph, lowered: &Graph, id: NodeId, memo: &mut BTreeMap<NodeId, Value>, fields: &mut Fields) -> Result<Value, String> {
        if let Some(v) = memo.get(&id) {
            return Ok(v.clone());
        }
        let node = original.nodes.get(&id).ok_or("a wire reads a missing node")?;
        let input = |port: usize| original.input_from(id, port).ok_or_else(|| format!("node {id} has input {} unconnected", port + 1));
        let v = match &node.kind {
            NodeKind::Remesh { triangles } => {
                let field = self.field_for(original, lowered, input(0)?, true, fields)?;
                (self.log)(&format!("remeshing to {triangles} triangles"));
                let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
                let norm = Normalization { center: [0.0; 3], scale: 1.0 };
                let settings = Settings { remesh: Some(*triangles as usize), ..Default::default() };
                let job = Job { ev: &field.ev, d, lo: field.lo, hi: field.hi, reach: field.reach, norm, settings };
                let mut mesh = job.remeshed(*triangles as usize, &self.progress);
                mesh.recompute_normals();
                Value::Mesh(mesh)
            }
            NodeKind::AutoUv(params) => {
                let (src, _) = input(0)?;
                let Value::Mesh(mesh) = self.value(original, lowered, src, memo, fields)? else { return Err(format!("node {id} (Auto UVs) needs a mesh")) };
                (self.log)("unwrapping");
                Value::Uv(crate::uvgen::generate(&mesh, params)?)
            }
            NodeKind::Bake(settings) => {
                let (mesh_src, _) = input(0)?;
                let field_src = input(1)?;
                let Value::Uv(uv) = self.value(original, lowered, mesh_src, memo, fields)? else { return Err(format!("node {id} (Bake) needs a mesh with UVs: put Auto UVs before it")) };
                let field = self.field_for(original, lowered, field_src, true, fields)?;
                // Custom maps: the fields wired into the extra inputs.
                let mut custom_fields = Vec::new();
                for j in 0..settings.custom.len() {
                    let src = input(2 + j)?;
                    custom_fields.push(self.field_for(original, lowered, src, false, fields)?);
                }
                (self.log)("baking");
                let (lo, hi) = (field.lo, field.hi);
                let diagonal = (0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f32>().sqrt();
                let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
                let mut source = crate::meshexport::EvalField::new(&field.ev, d, 0.0005 * diagonal);
                let progress = self.progress.clone();
                let mut custom_sources: Vec<crate::meshexport::EvalField> = custom_fields.iter().map(|f| crate::meshexport::EvalField::new(&f.ev, d, 0.0005 * diagonal)).collect();
                let mut custom_refs: Vec<&mut dyn crate::bake::Field> = custom_sources.iter_mut().map(|f| f as &mut dyn crate::bake::Field).collect();
                let images = crate::bake::bake(&uv, &mut source, &mut custom_refs, settings, &|done: usize, total: usize| progress.store((done * 1000 / total.max(1)) as u32, std::sync::atomic::Ordering::Relaxed))?;
                Value::Images(images)
            }
            NodeKind::ImageSplit => {
                let Some(image) = self.image_in(original, lowered, id, 0, memo, fields)? else { return Err(format!("node {id} (Split channels) has no image connected")) };
                let n = (image.size * image.size) as usize;
                let c = image.components;
                let names = ["R", "G", "B", "A"];
                let split = (0..4)
                    .map(|k| {
                        let data: Vec<f32> = (0..n).map(|i| if k < c { image.data[i * c + k] } else if k == 3 { 1.0 } else { image.data[i * c] }).collect();
                        crate::bake::Image { channel: None, label: format!("{}_{}", image.label, names[k]), size: image.size, components: 1, data }
                    })
                    .collect();
                Value::Images(split)
            }
            NodeKind::ImageCombine => {
                let mut parts = Vec::new();
                for port in 0..4 {
                    parts.push(self.image_in(original, lowered, id, port, memo, fields)?);
                }
                let Some(size) = parts.iter().flatten().map(|i| i.size).next() else { return Err(format!("node {id} (Combine channels) has no image connected")) };
                if parts.iter().flatten().any(|i| i.size != size) {
                    return Err(format!("node {id} (Combine channels): the images differ in size"));
                }
                let n = (size * size) as usize;
                let mut data = vec![0.0f32; n * 4];
                for (k, part) in parts.iter().enumerate() {
                    for i in 0..n {
                        data[i * 4 + k] = match part {
                            Some(img) => img.data[i * img.components],
                            None if k == 3 => 1.0,
                            None => 0.0,
                        };
                    }
                }
                Value::Images(vec![crate::bake::Image { channel: None, label: "Combined".into(), size, components: 4, data }])
            }
            NodeKind::ImageLevels { scale, offset, invert } => {
                let Some(mut image) = self.image_in(original, lowered, id, 0, memo, fields)? else { return Err(format!("node {id} (Image levels) has no image connected")) };
                for v in &mut image.data {
                    let x = (*v * scale + offset).clamp(0.0, 1.0);
                    *v = if *invert { 1.0 - x } else { x };
                }
                Value::Images(vec![image])
            }
            _ => return Err(format!("node {id} cannot be run: it is not an operation (Remesh, Auto UVs or Bake)")),
        };
        memo.insert(id, v.clone());
        Ok(v)
    }
}

/// The field sources built so far in a run, by source node and output, and whether they need a mesh.
type Fields = BTreeMap<(NodeId, usize, bool), std::rc::Rc<FieldSource>>;

struct FieldSource {
    ev: MeshEvaluator,
    lo: [f32; 3],
    hi: [f32; 3],
    reach: f32,
    /// Keeps the distance grids' textures alive as long as the evaluator.
    _loaded: Vec<std::sync::Arc<LoadedMesh>>,
}
