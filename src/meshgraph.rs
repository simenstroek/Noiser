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
    /// How many centimetres one unit of the file is (100: the file is in metres). Noise Graph nodes work in metres:
    /// their frequency and amplitude are scaled by this, so the same noise fits a mesh in metres or in centimetres.
    pub unit_cm: f32,
    /// A Load SDF node: the same node, offering .sdf files.
    pub sdf: bool,
    /// A heightmap (a PNG from the Landscape tab) as the file: the terrain is cut into this many chunks per side, and a
    /// run makes every chunk. 1 is the whole terrain.
    pub chunks: u32,
    /// A heightmap: the paint layers (names from the sidecar's `layers`) loaded beside the heights, for Scatter's density
    /// map and Terrain Layer nodes (the Bake node writes them per chunk).
    pub maps: Vec<String>,
}

impl Default for LoadMeshParams {
    fn default() -> Self {
        LoadMeshParams { path: String::new(), resolution: 128, padding: 0.5, combine: Default::default(), unit_cm: 100.0, sdf: false, chunks: 1, maps: Vec::new() }
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
    /// Detached: only the embedded copy is used, the file is ignored (it lives in this mesh graph alone).
    pub detached: bool,
}

impl Default for NoiseGraphParams {
    fn default() -> Self {
        NoiseGraphParams { file: String::new(), embedded: String::new(), frequency: 1.0, amplitude: 1.0, detached: false }
    }
}

/// Where an output node writes: a folder and a file name (without extension).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutParams {
    pub folder: String,
    pub name: String,
    /// Image outputs: the file format.
    pub image: crate::bake::ImageFormat,
    /// Mesh outputs: PLY instead of OBJ.
    pub ply: bool,
    /// Mesh outputs: a binary glTF (.glb) instead (takes precedence over `ply`).
    pub glb: bool,
}

impl Default for OutParams {
    fn default() -> Self {
        OutParams { folder: String::new(), name: "output".into(), image: Default::default(), ply: false, glb: false }
    }
}

/// A path from a graph: relative ones are relative to the folder of the graph file.
pub fn resolve_in(base: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() { path.to_path_buf() } else { base.join(path) }
}

/// The noise graph a Noise Graph node refers to: the file when it can be read, else the embedded copy.
pub fn noise_graph_source(p: &NoiseGraphParams, base: &Path) -> Result<Graph, String> {
    let from_file = if p.file.is_empty() || p.detached { None } else { std::fs::read_to_string(resolve_in(base, &p.file)).ok() };
    let text = from_file.as_deref().or(if p.embedded.is_empty() { None } else { Some(p.embedded.as_str()) }).ok_or_else(|| format!("the noise graph {} cannot be read and has no embedded copy", if p.file.is_empty() { "(none chosen)" } else { &p.file }))?;
    Graph::from_json(text)
}

/// The field part of a mesh graph as one plain graph: every Noise Graph node becomes a copy of the graph it names.
/// The copy's Position nodes read the Noise Graph node's position input times `frequency`; its density output times
/// minus `amplitude` takes the node's place (and id): a mesh graph's field is a distance, negative inside, so subtracting
/// the density moves the surface outward where the noise graph's density is positive, as in the noise graph, so everything downstream and the node's preview keep working.
pub fn lowered_mesh(g: &Graph, base: &Path) -> Result<Graph, String> {
    use crate::graph::MathOp;
    let mut out = g.clone();
    // The unit of the first Load Mesh node (a file in centimetres: 1) turns the noise's metres into the file's units.
    let unit_cm = g.nodes.values().filter(|n| !n.muted).find_map(|n| if let NodeKind::LoadMesh(p) = &n.kind { Some(p.unit_cm) } else { None }).filter(|u| u.is_finite() && *u > 0.0).unwrap_or(100.0);
    let (freq_k, amp_k) = (unit_cm / 100.0, 100.0 / unit_cm);
    let ids: Vec<_> = g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::NoiseGraph(_)) && !n.muted).map(|(id, _)| *id).collect();
    for id in ids {
        let NodeKind::NoiseGraph(p) = g.nodes[&id].kind.clone() else { continue };
        let sub = noise_graph_source(&p, base).map_err(|e| format!("Noise Graph node {id}: {e}"))?;
        let Some(out_id) = sub.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutDensity)).map(|(i, _)| *i) else { return Err("the noise graph has no Density Output".into()) };
        // The position the sub graph sees: the node's input (or the world position) times the frequency.
        let world = out.add(NodeKind::position(), [0.0; 2]);
        let pos_in = out.edges.iter().find(|e| e.to == id && e.port == 0).map(|e| (e.from, e.from_port));
        let scaled = out.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: p.frequency * freq_k, c: 0.5 }, [0.0; 2]);
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
        let positions: Vec<_> = map.iter().filter(|(sid, _)| matches!(sub.nodes[*sid].kind, NodeKind::Position { .. })).map(|(_, n)| *n).collect();
        for n in positions {
            // A Position with its own scale becomes `scaled position * Combine(scale)`; a plain one just the scaled position.
            let NodeKind::Position { scale } = out.nodes[&n].kind else { continue };
            if scale == [1.0; 3] && !out.edges.iter().any(|e| e.to == n) {
                out.nodes.get_mut(&n).unwrap().kind = NodeKind::Reroute;
                out.connect(scaled, n, 0);
            } else {
                let by = out.add(NodeKind::Combine { vals: scale }, [0.0; 2]);
                for e in out.edges.iter_mut().filter(|e| e.to == n) {
                    e.to = by;
                }
                out.nodes.get_mut(&n).unwrap().kind = NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 0.0, c: 0.5 };
                out.connect(scaled, n, 0);
                out.connect(by, n, 1);
            }
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
        out.nodes.get_mut(&id).unwrap().kind = NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: -p.amplitude * amp_k, c: 0.5 };
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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: crate::view::preferred_backends(), ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))
}

/// What a node of the operation part of the graph produced.
#[derive(Clone)]
enum Value {
    Mesh(RawMesh),
    Uv(crate::uvgen::UvMesh),
    Images(Vec<crate::bake::Image>),
    /// A loaded mesh file as it is (a Load Mesh's mesh output).
    Source(crate::mesh::Mesh),
}

/// A loaded mesh baked into a distance grid, kept between runs (the grid takes seconds to build).
pub struct LoadedMesh {
    /// Kept alive with the view.
    _texture: eframe::wgpu::Texture,
    view: eframe::wgpu::TextureView,
    origin: [f32; 3],
    voxel: f32,
    norm: Normalization,
    /// Bounds in the mesh file's units.
    lo: [f32; 3],
    hi: [f32; 3],
    /// The mesh as the file has it, in its units, with its UVs (for Transfer UVs).
    raw: crate::mesh::Mesh,
    /// A terrain: its heights (`norm.scale` is -1 for the shader, see `land_dist`).
    land: Option<std::sync::Arc<crate::landscape_field::Heights>>,
}

#[derive(Default)]
pub struct MeshCache {
    /// The terrain chunk a run is making, (column, row).
    chunk: Option<(u32, u32)>,
    /// Where that chunk is: (x0, y0, size) in the field's units, for planar UVs.
    chunk_box: Option<(f32, f32, f32)>,
    meshes: BTreeMap<String, std::sync::Arc<LoadedMesh>>,
    /// Keys of the meshes a batch run read from `{batch}`: dropped when the next file starts, so a long batch does not keep every mesh.
    batch_keys: Vec<String>,
}

/// Set by the Mesh tab's Cancel button; a run stops before its next node. One run at a time, so one flag will do.
pub static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
    /// Batch runs: the mesh file a Load Mesh node whose path is `{batch}` reads, and the `{name}` output names use.
    pub batch_file: Option<PathBuf>,
    /// Quick previews: Remesh nodes make at most this many triangles.
    pub max_triangles: Option<u32>,
    /// What the `{tags}` in output names become: the graph, the run's place in its batch, the time.
    pub tags: RunTags,
}

/// What the smart tags of an output name stand for in one run.
#[derive(Clone, Debug, Default)]
pub struct RunTags {
    /// The mesh graph file's name without its extension ("" for a graph never saved).
    pub graph: String,
    /// Which run of the batch this is (from 0) and how many there are.
    pub index: usize,
    pub count: usize,
    /// When the batch started, local time as `YYYYMMDD-HHMMSS`: the same for every run of one batch.
    pub timestamp: String,
}

impl RunTags {
    /// Tags for a batch of `count` runs starting now, of the graph in `file`.
    pub fn new(file: Option<&Path>, count: usize) -> RunTags {
        let graph = file.and_then(|f| f.file_stem()).map_or(String::new(), |s| s.to_string_lossy().into_owned());
        RunTags { graph, index: 0, count: count.max(1), timestamp: timestamp_now() }
    }

    /// The tags for run `index` of the same batch.
    pub fn run(&self, index: usize) -> RunTags {
        RunTags { index, ..self.clone() }
    }
}

/// Local time as `YYYYMMDD-HHMMSS`.
pub fn timestamp_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    // The local offset from the clock the OS reports for this moment; UTC where it cannot be had.
    let local = secs + local_offset_secs();
    let (days, rem) = (local.div_euclid(86_400), local.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let (y, doy) = (yoe + era * 400, doe - (365 * yoe + yoe / 4 - yoe / 100));
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(windows)]
fn local_offset_secs() -> i64 {
    #[repr(C)]
    struct TimeZoneInformation {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: [u16; 8],
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: [u16; 8],
        daylight_bias: i32,
    }
    unsafe extern "system" {
        fn GetTimeZoneInformation(info: *mut TimeZoneInformation) -> u32;
    }
    // SAFETY: a plain Win32 call that fills the struct it is given.
    unsafe {
        let mut info: TimeZoneInformation = std::mem::zeroed();
        let id = GetTimeZoneInformation(&mut info);
        let extra = match id {
            1 => info.standard_bias,
            2 => info.daylight_bias,
            _ => 0,
        };
        -((info.bias + extra) as i64) * 60
    }
}

#[cfg(not(windows))]
fn local_offset_secs() -> i64 {
    0
}

/// `name` with its smart tags replaced. `{name}`: the batch mesh's name, else the graph's; `{graph}`: the graph file's name;
/// `{seed}`; `{index}`: the run's number from 1, as wide as the count needs (01 to 10); `{count}`; `{timestamp}`
/// (`20261127-141502`), `{date}` and `{time}` (its parts). Unknown tags stay as written.
pub fn expand_tags(name: &str, batch_stem: Option<&str>, seed: u32, t: &RunTags) -> String {
    let graph = if t.graph.is_empty() { "mesh" } else { t.graph.as_str() };
    let (date, time) = t.timestamp.split_once('-').unwrap_or((t.timestamp.as_str(), ""));
    let width = t.count.to_string().len().max(2);
    name.replace("{name}", batch_stem.filter(|s| !s.is_empty()).unwrap_or(graph))
        .replace("{graph}", graph)
        .replace("{seed}", &seed.to_string())
        .replace("{index}", &format!("{:0width$}", t.index + 1))
        .replace("{count}", &t.count.to_string())
        .replace("{timestamp}", &t.timestamp)
        .replace("{date}", date)
        .replace("{time}", time)
}

/// The smart tags output names understand, for hints.
pub const TAG_HELP: &str = "{name} (the batch mesh's name, else the graph's), {graph}, {seed}, {index} (run number), {count}, {timestamp}, {date}, {time}";

impl Runner<'_> {
    fn resolve(&self, p: &str) -> PathBuf {
        resolve_in(&self.base, p)
    }

    fn mesh(&mut self, p: &LoadMeshParams) -> Result<std::sync::Arc<LoadedMesh>, String> {
        if p.path.is_empty() {
            return Err("a Load Mesh node has no file chosen".into());
        }
        let path = match (&self.batch_file, p.path.as_str()) {
            (Some(f), "{batch}") => f.clone(),
            (None, "{batch}") => return Err("a Load Mesh node reads {batch}, which needs a batch run (--batch-dir or --batch-file)".into()),
            _ => self.resolve(&p.path),
        };
        let key = format!("{}|{}|{}|{:?}|{}|{}", path.display(), p.resolution, p.padding, p.combine.mode, p.combine.smooth, p.maps.join(","));
        if let Some(m) = self.cache.meshes.get(&key) {
            return Ok(m.clone());
        }
        (self.log)(&format!("loading {}", path.display()));
        if crate::landscape_field::is_heightmap(&p.path) {
            let heights = std::sync::Arc::new(crate::landscape_field::load_with_maps(&path, p.unit_cm, &p.maps)?);
            (self.log)(&format!("  terrain {0} x {0} samples, {1:.1} units apart, heights {2:.1} to {3:.1}", heights.n, heights.cell, heights.z.0, heights.z.1));
            let (texture, view) = heights.upload(self.device, self.queue);
            let w = heights.world();
            let loaded = std::sync::Arc::new(LoadedMesh {
                _texture: texture,
                view,
                origin: [0.0, 0.0, heights.n as f32],
                voxel: heights.cell,
                norm: Normalization { center: [heights.tiles() as f32, heights.z.0, heights.z.1], scale: -1.0 },
                lo: [0.0, 0.0, heights.z.0],
                hi: [w, w, heights.z.1],
                raw: heights.coarse_mesh(),
                land: Some(heights),
            });
            self.cache.meshes.insert(key, loaded.clone());
            return Ok(loaded);
        }
        let raw = crate::meshformats::load(&path)?;
        let (mesh, norm) = raw.normalized();
        let topology = crate::meshquality::report(&crate::surface::RawMesh { positions: mesh.positions.clone(), normals: Vec::new(), triangles: mesh.triangles.clone() });
        if !topology.watertight() {
            (self.log)(&format!("  warning: this mesh is not closed ({} open edge(s), {} non-manifold): the inside of the field is unreliable near them, and thin open sheets give a thin shell", topology.boundary_edges, topology.non_manifold_edges));
        }
        let sdf = crate::meshsdf::MeshSdf::build_with(&mesh, p.resolution, p.padding, &Default::default(), p.combine);
        let (texture, view) = sdf.upload(self.device, self.queue);
        let (nlo, nhi) = mesh.bounds();
        let loaded = std::sync::Arc::new(LoadedMesh { _texture: texture, view, origin: sdf.origin, voxel: sdf.voxel, norm, lo: norm.to_object(nlo), hi: norm.to_object(nhi), raw, land: None });
        if p.path == "{batch}" {
            self.cache.batch_keys.push(key.clone());
        }
        self.cache.meshes.insert(key, loaded.clone());
        Ok(loaded)
    }

    /// One chunk of the terrain with at most `triangles` triangles (about that many: `2 q^2`), or the whole terrain at a coarse
    /// budget for a preview.
    fn terrain_chunk(&mut self, field: &FieldSource, land: &Land, triangles: usize) -> crate::surface::RawMesh {
        use crate::meshexport::{ChunkWindow, Displace};
        let (n, cx, cy) = if self.max_triangles.is_some() { (1, 0, 0) } else { (land.chunks.max(1), self.cache.chunk.map_or(0, |c| c.0), self.cache.chunk.map_or(0, |c| c.1)) };
        let size = land.heights.world() / n as f32;
        // Steep ground crosses more lattice edges than flat ground, so the budget is spent on the steepest chunk: every chunk
        // shares one lattice spacing, which is what makes their borders meet.
        let steepest = (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).map(|(i, j)| land.heights.quads_per_column(i as f32 * size, (i + 1) as f32 * size, j as f32 * size, (j + 1) as f32 * size)).fold(1.0f32, f32::max);
        let q = ((triangles as f32 / (2.0 * steepest)).sqrt().ceil() as usize).max(4);
        let h = size / q as f32;
        let (x0, y0) = (cx as f32 * size, cy as f32 * size);
        // The surface can leave the ground by the reach, and objects stand wherever they stand.
        let (lo, hi) = land.heights.z_range(x0 - 2.0 * h, x0 + size + 2.0 * h, y0 - 2.0 * h, y0 + size + 2.0 * h);
        let mut z = (lo - field.reach - field.lift.1 - 3.0 * h, hi + field.reach + field.lift.0 + 3.0 * h);
        for (blo, bhi) in &field.objects {
            if blo[0] < x0 + size + h && bhi[0] > x0 - h && blo[1] < y0 + size + h && bhi[1] > y0 - h {
                z = (z.0.min(blo[2] - field.reach - 3.0 * h), z.1.max(bhi[2] + field.reach + 3.0 * h));
            }
        }
        let z_origin = field.lo[2] - field.reach - field.lift.1 - 3.0 * h;
        self.cache.chunk_box = Some((x0, y0, size));
        let t0 = std::time::Instant::now();
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
        let mesh = field.ev.chunk_surface(d, &ChunkWindow { x0, y0, size, q, z_origin, z }, &self.progress);
        (self.log)(&format!("terrain chunk {cy},{cx}: {q} x {q} quads of {h:.2} units, z {:.1} to {:.1}, {} triangles in {:.1} s", z.0, z.1, mesh.triangles.len(), t0.elapsed().as_secs_f32()));
        mesh
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
        let (mut land, mut objects) = (None, Vec::new());
        for id in &compiled.mesh_slots {
            let NodeKind::LoadMesh(p) = &g.nodes[id].kind else { continue };
            let m = self.mesh(p)?;
            match &m.land {
                Some(heights) => land = Some(Land { heights: heights.clone(), chunks: p.chunks }),
                None => objects.push((m.lo, m.hi)),
            }
            graph_meshes.push(GraphMesh { view: m.view.clone(), origin: m.origin, voxel: m.voxel, centre: m.norm.center, scale: m.norm.scale });
            loaded.push(m);
        }
        // Shapes that stand on the terrain: their height comes from the ground under them.
        if let Some(l) = &land {
            for node in g.nodes.values_mut() {
                if let NodeKind::LoadSdf(p) = &mut node.kind
                    && p.snap && p.fit {
                    let mut flat = p.clone();
                    flat.offset[2] = 0.0;
                    if let Some((lo, _)) = flat.field_bounds() {
                        let ground = if p.snap_high { l.heights.highest_ground(p.offset[0], p.offset[1], 0.4 * p.size) } else { l.heights.lowest_ground(p.offset[0], p.offset[1], 0.4 * p.size) };
                        p.offset[2] = ground - lo[2] - p.sink * p.size;
                    }
                }
            }
        }
// A Load SDF node knows where its shape is, so it can stand in for a mesh as the place to look for the surface.
        let sdf_boxes: Vec<([f32; 3], [f32; 3])> = g.nodes.values().filter_map(|n| if let NodeKind::LoadSdf(p) = &n.kind { p.field_bounds() } else { None }).collect();
        objects.extend(sdf_boxes.iter().copied());
        if loaded.is_empty() && sdf_boxes.is_empty() && require_mesh {
            return Err("the field has no Load Mesh or Load SDF node, so there is no surface to extract".into());
        }
        let compile_started = std::time::Instant::now();
        let ev = MeshEvaluator::new_graph(self.device, self.queue, &g, self.seed, graph_meshes)?;
        (self.log)(&format!("compiled the field in {:.1} s", compile_started.elapsed().as_secs_f32()));
let (mut lo, mut hi) = if loaded.is_empty() && sdf_boxes.is_empty() { ([0.0; 3], [0.0; 3]) } else { ([f32::MAX; 3], [f32::MIN; 3]) };
        for m in &loaded {
            for a in 0..3 {
                lo[a] = lo[a].min(m.lo[a]);
                hi[a] = hi[a].max(m.hi[a]);
            }
        }
        for (blo, bhi) in &sdf_boxes {
            for a in 0..3 {
                lo[a] = lo[a].min(blo[a]);
                hi[a] = hi[a].max(bhi[a]);
            }
        }
        if loaded.is_empty() && !sdf_boxes.is_empty() {
            // The shape's own box is tight: leave room for noise and the surface sitting on the edge.
            let margin = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) * 0.08;
            for a in 0..3 {
                lo[a] -= margin;
                hi[a] += margin;
            }
        }
        let radius = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) * 0.5;
        // How far noise graphs can move the surface: their amplitudes times the largest value of their density, plus a margin.
        let mut reach = if land.is_some() { 0.0 } else { 0.05 * radius };
        for n in original.nodes.values() {
            if let NodeKind::NoiseGraph(p) = &n.kind
                && let Ok(sub) = noise_graph_source(p, &self.base) {
                    let (a, b) = sub.density_range();
                    let size = if a.is_finite() && b.is_finite() { a.abs().max(b.abs()) } else { 1.0 };
                    reach += p.amplitude.abs() * size;
                }
        }
        let scatters: Vec<&crate::scatter::ScatterParams> = original.nodes.values().filter_map(|n| if let NodeKind::Scatter(p) = &n.kind { Some(p) } else { None }).collect();
        let lift = (scatters.iter().map(|p| p.reach()).fold(0.0, f32::max), scatters.iter().filter(|p| p.subtract).map(|p| p.reach()).fold(0.0, f32::max));
        Ok(FieldSource { ev, lo, hi, reach: reach.min(2.0 * radius), land, lift, objects, _loaded: loaded })
    }

    fn output_path(&self, o: &OutParams, suffix: &str, ext: &str) -> PathBuf {
        let dir = if !o.folder.is_empty() {
            self.resolve(&o.folder)
        } else {
            // Nothing chosen: the library's Bakes folder when there is one, else next to the graph.
            crate::library::dir(crate::library::Kind::Bake).filter(|d| d.is_dir()).unwrap_or_else(|| self.base.clone())
        };
        let _ = std::fs::create_dir_all(&dir);
        let stem = self.batch_file.as_ref().and_then(|f| f.file_stem()).map(|s| s.to_string_lossy().into_owned());
        let chunk = self.cache.chunk.map_or(String::new(), |(cx, cy)| format!("_{cy:02}_{cx:02}"));
        let name = format!("{}{chunk}{suffix}", expand_tags(&o.name, stem.as_deref(), self.seed, &self.tags));
        // An existing file is overwritten: a rerun must keep the same names.
        dir.join(format!("{name}.{ext}"))
    }

    /// The field that feeds the first Remesh upstream of the first mesh output, ready to be drawn by ray marching.
    pub fn field_view(&mut self, graph: &Graph) -> Result<FieldView, String> {
        let lowered = lowered_mesh(graph, &self.base)?;
        let Some(out) = graph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutMesh(_))).map(|(id, _)| *id) else { return Err("the graph has no Mesh output node".into()) };
        let mut at = graph.input_from(out, 0).ok_or("the Mesh output has nothing connected")?.0;
        for _ in 0..graph.nodes.len() {
            if matches!(graph.nodes[&at].kind, NodeKind::Remesh { .. }) {
                break;
            }
            at = graph.input_from(at, 0).ok_or("the chain before the Mesh output has no Remesh node")?.0;
        }
        let src = graph.input_from(at, 0).ok_or("the Remesh node has nothing connected")?;
        let mut fields: Fields = BTreeMap::new();
        Ok(FieldView(self.field_for(graph, &lowered, src, true, &mut fields)?))
    }

    /// The first Remesh result upstream of the first mesh output, at the triangle limit: a quick look at the shape
    /// while the noise graphs are being tuned.
    pub fn preview(&mut self, graph: &Graph) -> Result<RawMesh, String> {
        let lowered = lowered_mesh(graph, &self.base)?;
        let Some(out) = graph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::OutMesh(_))).map(|(id, _)| *id) else { return Err("the graph has no Mesh output node".into()) };
        // Walk back along mesh wires to the Remesh node.
        let mut at = graph.input_from(out, 0).ok_or("the Mesh output has nothing connected")?.0;
        for _ in 0..graph.nodes.len() {
            if matches!(graph.nodes[&at].kind, NodeKind::Remesh { .. }) {
                break;
            }
            at = graph.input_from(at, 0).ok_or("the chain before the Mesh output has no Remesh node")?.0;
        }
        let mut memo = BTreeMap::new();
        let mut fields: Fields = BTreeMap::new();
        match self.value(graph, &lowered, at, &mut memo, &mut fields)? {
            Value::Mesh(m) => Ok(m),
            _ => Err("no Remesh node found upstream of the Mesh output".into()),
        }
    }

    /// Runs the operation nodes upstream of every output node and writes the files; returns the files written.
    /// Runs the graph; with a terrain cut into chunks it runs once for every chunk, file names ending `_row_column`.
    pub fn run(&mut self, graph: &Graph) -> Result<Vec<PathBuf>, String> {
        let n = graph.nodes.values().filter(|n| !n.muted).filter_map(|n| if let NodeKind::LoadMesh(p) = &n.kind { Some(p) } else { None }).filter(|p| crate::landscape_field::is_heightmap(&p.path)).map(|p| p.chunks).max().unwrap_or(1);
        if n <= 1 {
            self.cache.chunk = None;
            return self.run_once(graph);
        }
        let mut written = Vec::new();
        let mut manifest = Vec::new();
        let result: Result<(), String> = (|| {
            for cy in 0..n {
                for cx in 0..n {
                    self.cache.chunk = Some((cx, cy));
                    (self.log)(&format!("chunk {} of {}: row {cy}, column {cx}", cy * n + cx + 1, n * n));
                    let files = self.run_once(graph)?;
                    let (x0, y0, size) = self.cache.chunk_box.unwrap_or_default();
                    manifest.push(serde_json::json!({"row": cy, "column": cx, "origin": [x0, y0], "size": size, "files": files.iter().map(|f| f.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())).collect::<Vec<_>>()}));
                    written.extend(files);
                }
            }
            Ok(())
        })();
        self.cache.chunk = None;
        result?;
        // Next to the files: where each chunk is (the meshes already hold their world positions, in the field's units).
        if let Some(dir) = written.first().and_then(|f| f.parent()) {
            let path = dir.join("terrain_chunks.json");
            let terrain = graph.nodes.values().filter_map(|n| if let NodeKind::LoadMesh(p) = &n.kind { Some(p) } else { None }).find(|p| crate::landscape_field::is_heightmap(&p.path));
            let water_cm = terrain.and_then(|p| crate::landscape_field::water_level_cm(&resolve_in(&self.base, &p.path)));
            // The paint layers baked per chunk: the first four of the terrain's maps are the channels of the "splat" image.
            let colours = terrain.map(|p| crate::landscape_field::sidecar_layer_colours(&resolve_in(&self.base, &p.path))).unwrap_or_default();
            let layers: Vec<_> = terrain.map(|p| p.maps.iter().take(4).enumerate().map(|(k, name)| serde_json::json!({"name": name, "channel": "rgba".chars().nth(k).map(String::from).unwrap_or_default(), "color": colours.iter().find(|(n, _)| n == name).map_or([0.5; 3], |(_, c)| *c)})).collect()).unwrap_or_default();
            let json = serde_json::json!({"chunks_per_side": n, "unit_cm": terrain.map_or(1.0, |p| p.unit_cm), "water_level_cm": water_cm, "layers": layers, "chunks": manifest});
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap_or_default()).map_err(|e| format!("{}: {e}", path.display()))?;
            (self.log)(&format!("wrote {}", path.display()));
        }
        Ok(written)
    }

    fn run_once(&mut self, graph: &Graph) -> Result<Vec<PathBuf>, String> {
        for key in std::mem::take(&mut self.cache.batch_keys) {
            self.cache.meshes.remove(&key);
        }
        let lowered = lowered_mesh(graph, &self.base)?;
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
                    let path = self.output_path(o, "", if o.glb { "glb" } else if o.ply { "ply" } else { "obj" });
                    (if o.glb { crate::meshformats::write_glb(&path, &uv.positions, &uv.normals, Some(&uv.uvs), &uv.triangles) } else if o.ply { uv.write_ply(&path) } else { uv.write_obj(&path) }).map_err(|e| format!("{}: {e}", path.display()))?;
                    (self.log)(&format!("wrote {}", path.display()));
                    written.push(path);
                }
                (NodeKind::OutMesh(o), Value::Mesh(mesh)) => {
                    let path = self.output_path(o, "", if o.glb { "glb" } else if o.ply { "ply" } else { "obj" });
                    (if o.glb { crate::meshformats::write_glb(&path, &mesh.positions, &mesh.normals, None, &mesh.triangles) } else if o.ply { mesh.write_ply(&path) } else { mesh.write_obj(&path) }).map_err(|e| format!("{}: {e}", path.display()))?;
                    (self.log)(&format!("wrote {}", path.display()));
                    written.push(path);
                }
                (NodeKind::OutImage(o), Value::Images(images)) => {
                    let Some(image) = images.get(port) else { return Err(format!("output node {out} reads image {port}, which the Bake node does not produce")) };
                    let path = self.output_path(o, &format!("_{}", image.label), o.image.extension());
                    image.write(&path, o.image)?;
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
        if CANCEL.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let node = original.nodes.get(&id).ok_or("a wire reads a missing node")?;
        let input = |port: usize| original.input_from(id, port).ok_or_else(|| format!("node {id} has input {} unconnected", port + 1));
        let v = match &node.kind {
            NodeKind::Remesh { triangles, adaptivity, mesher, auto_detail } => {
                let field = self.field_for(original, lowered, input(0)?, true, fields)?;
                // A terrain keeps its own meshing (chunks cut from one lattice); the options below are for shapes.
                let mut triangles = *triangles;
                if let (true, None) = (*auto_detail, &field.land) {
                    let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
                    let norm = Normalization { center: [0.0; 3], scale: 1.0 };
                    let job = Job { ev: &field.ev, d, lo: field.lo, hi: field.hi, reach: field.reach, norm, settings: Settings { remesh: Some(triangles as usize), ..Default::default() } };
                    let (complexity, _) = job.complexity_probe(&self.progress);
                    // 1 is a sphere: plainer shapes get fewer triangles, complex ones more, within a factor of 4 either way.
                    let factor = complexity.clamp(0.4, 4.0);
                    let scaled = ((triangles as f32 * factor) as u32).max(200);
                    (self.log)(&format!("shape complexity {complexity:.2} (a sphere is 1): {triangles} x {factor:.2} = {scaled} triangles"));
                    triangles = scaled;
                }
                let triangles = &self.max_triangles.map_or(triangles, |m| triangles.min(m));
                if let Some(land) = &field.land {
                    Value::Mesh(self.terrain_chunk(&field, land, *triangles as usize))
                } else {
                (self.log)(&format!("meshing to about {triangles} triangles ({mesher:?})"));
                let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
                let norm = Normalization { center: [0.0; 3], scale: 1.0 };
                let settings = Settings { remesh: Some(*triangles as usize), ..Default::default() };
                let job = Job { ev: &field.ev, d, lo: field.lo, hi: field.hi, reach: field.reach, norm, settings };
                let mut mesh = job.meshed(*mesher, *triangles as usize, *adaptivity, &self.progress);
                mesh.recompute_normals();
                // The numbers that say whether the remesh is any good: triangle shape, folds, distance from the field's surface.
                let diagonal = (0..3).map(|a| (field.hi[a] - field.lo[a]).powi(2)).sum::<f32>().sqrt();
                let report = crate::meshquality::report(&mesh).with_surface_error(&mesh, &mut |pts| field.ev.points(d, pts, 0.0005 * diagonal)).with_intersections(&mesh);
                for line in report.to_string().lines() {
                    (self.log)(&format!("  {line}"));
                }
                Value::Mesh(mesh)
                }
            }
            NodeKind::LoadMesh(p) => Value::Source(self.mesh(p)?.raw.clone()),
            NodeKind::TransferUv => {
                let (target_src, _) = input(0)?;
                let (source_src, _) = input(1)?;
                let Value::Mesh(target) = self.value(original, lowered, target_src, memo, fields)? else { return Err(format!("node {id} (Transfer UVs): the target needs a mesh without UVs, such as Remesh")) };
                let source = match self.value(original, lowered, source_src, memo, fields)? {
                    Value::Source(m) => m,
                    // The UVs of an unwrapped mesh (another level of detail, say) carried onto this one.
                    Value::Uv(u) => crate::mesh::Mesh { positions: u.positions.clone(), triangles: u.triangles.clone(), colors: Vec::new(), uvs: u.triangles.iter().map(|t| t.map(|i| [u.uvs[i as usize][0], 1.0 - u.uvs[i as usize][1]])).collect() },
                    _ => return Err(format!("node {id} (Transfer UVs): the UV source must be a Load Mesh's mesh output or a mesh with UVs")),
                };
                (self.log)("transferring UVs");
                let (uv, stats) = crate::uvtransfer::transfer(&target, &source)?;
                (self.log)(&format!("  {} island(s) used; {} corner(s) farther than 2% of the mesh size from the source surface", stats.islands, stats.far_corners));
                for line in crate::meshquality::uv_report(&uv, 512).to_string().lines() {
                    (self.log)(&format!("  {line}"));
                }
                Value::Uv(uv)
            }
            NodeKind::AutoUv(params) => {
                let (src, _) = input(0)?;
                let Value::Mesh(mesh) = self.value(original, lowered, src, memo, fields)? else { return Err(format!("node {id} (Auto UVs) needs a mesh")) };
                let uv = match (params.planar, self.cache.chunk_box) {
                    (true, Some((x0, y0, size))) => {
                        (self.log)("planar UVs from above");
                        crate::uvgen::planar(&mesh, x0, y0, size)
                    }
                    _ => {
                        if params.planar {
                            (self.log)("note: planar UVs are for terrain chunks (a Load Mesh with a heightmap before the Remesh); unwrapping this mesh instead");
                        }
                        (self.log)("unwrapping");
                        crate::uvgen::generate(&mesh, params)?
                    }
                };
                for line in crate::meshquality::uv_report(&uv, params.resolution as usize).to_string().lines() {
                    (self.log)(&format!("  {line}"));
                }
                Value::Uv(uv)
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

/// A field that can be drawn (see `Runner::field_view`).
pub struct FieldView(std::rc::Rc<FieldSource>);

impl FieldView {
    /// The field's bounds in the units of the mesh files.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.0.lo, self.0.hi)
    }

    /// For a terrain, the box its surface can be in (the square and its heights, widened by how far the surface can
    /// leave the ground, how high scattered shapes stand, and the other objects' boxes): the field view traces only
    /// inside it. None for other fields, which trace the whole way.
    pub fn trace_box(&self) -> Option<([f32; 3], [f32; 3])> {
        let f = &self.0;
        let land = f.land.as_ref()?;
        let pad = f.reach;
        // Across, exactly the terrain's square (half a sample larger, as the field's own cut is): past it the field is
        // many times the true distance, and a ray let in there would overshoot into the ground.
        let (side, half) = (land.heights.world(), 0.5 * land.heights.cell);
        let mut lo = [-half, -half, f.lo[2] - pad - f.lift.1];
        let mut hi = [side + half, side + half, f.hi[2] + pad + f.lift.0];
        for (olo, ohi) in &f.objects {
            for a in 0..3 {
                lo[a] = lo[a].min(olo[a] - pad);
                hi[a] = hi[a].max(ohi[a] + pad);
            }
        }
        // A sliver more up and down, so a surface lying on the top or bottom is not cut by rounding. Not across: just past
        // the square the field is 1024 times the distance, and a ray let in there would take one huge step into the ground.
        let margin = 1e-3 * (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max);
        lo[2] -= margin;
        hi[2] += margin;
        Some((lo, hi))
    }

    pub fn march(&self, cam: &crate::meshexport::MarchCamera, w: u32, h: u32) -> Vec<[f32; 4]> {
        let d = Displace { amount: 0.0, tile_size: 1.0, prim_size: 1.0, mask: crate::meshmask::PackedMask::NONE };
        self.0.ev.march(d, cam, w, h)
    }
}

/// A terrain among the field's meshes.
struct Land {
    heights: std::sync::Arc<crate::landscape_field::Heights>,
    chunks: u32,
}

struct FieldSource {
    ev: MeshEvaluator,
    lo: [f32; 3],
    hi: [f32; 3],
    reach: f32,
    land: Option<Land>,
    /// How far scattered shapes can stand above the ground, and how far cut-outs can reach below it.
    lift: (f32, f32),
    /// The boxes of the other meshes and SDFs, which can stand above or inside the terrain.
    objects: Vec<([f32; 3], [f32; 3])>,
    /// Keeps the distance grids' textures alive as long as the evaluator.
    _loaded: Vec<std::sync::Arc<LoadedMesh>>,
}

#[cfg(test)]
mod tag_tests {
    use super::*;

    #[test]
    fn smart_tags_fill_in_names_for_a_batch_with_and_without_meshes() {
        let t = RunTags { graph: "ProcRock01".into(), index: 2, count: 10, timestamp: "20261127-141502".into() };
        // No batch mesh: {name} is the graph's name, not nothing.
        assert_eq!(expand_tags("{name}", None, 7, &t), "ProcRock01");
        assert_eq!(expand_tags("{name}_{seed}_{index}", Some("rock.obj".trim_end_matches(".obj")), 9, &t), "rock_9_03");
        assert_eq!(expand_tags("{graph}-{timestamp}-{date}-{time}-{count}", None, 0, &t), "ProcRock01-20261127-141502-20261127-141502-10");
        // The index is as wide as the count needs, and never narrower than two digits.
        assert_eq!(expand_tags("{index}", None, 0, &RunTags { index: 99, count: 100, ..t.clone() }), "100");
        assert_eq!(expand_tags("{index}", None, 0, &RunTags { index: 0, count: 3, ..t.clone() }), "01");
        // An unsaved graph is called "mesh"; unknown tags stay as they are.
        assert_eq!(expand_tags("{name}{x}", None, 0, &RunTags::default()), "mesh{x}");
    }

    #[test]
    fn the_timestamp_has_the_documented_shape() {
        let t = timestamp_now();
        assert_eq!(t.len(), 15, "{t}");
        assert_eq!(t.as_bytes()[8], b'-');
        assert!(t.chars().filter(|c| *c != '-').all(|c| c.is_ascii_digit()), "{t}");
        assert!(t.starts_with("20"), "{t}");
    }
}
