//! Per-node preview caches. A cached node's value (its whole upstream graph) is baked into one
//! tile of an R32Float 3D texture on the GPU and the preview shader samples it (codegen
//! `compile_with`, common.wgsl `cache_sample`). Export never uses caches.
//!
//! Bake pipelines are keyed by the upstream graph's WGSL: the first bake of a new topology
//! compiles on a worker thread; later parameter changes only re-dispatch (milliseconds).

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::{Graph, NodeId, Tiling};
use crate::view::CACHE_SLOTS;
use eframe::wgpu;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

/// Linear filtering of float32 textures is an optional GPU feature; caches need it.
pub const FEATURE: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE;

pub fn texture(device: &wgpu::Device, res: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("node cache"),
        size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

const BAKE_WGSL: &str = "
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(dst);
    if (any(id >= n)) { return; }
    textureStore(dst, id, vec4f(graph_density((vec3f(id) + 0.5) / vec3f(n)), 0.0, 0.0, 1.0));
}
";

pub struct Baker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    blue: wgpu::Buffer,
}

impl Baker {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> Self {
        use wgpu::util::DeviceExt;
        let src = format!("{NOISE_LIB}\n{wgsl}\n{EVAL_BINDINGS}\n{BAKE_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("cache bake"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let buf = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                buf(0, wgpu::BufferBindingType::Uniform),
                buf(1, ro),
                buf(3, ro),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cache bake"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let blue = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&crate::bluenoise::tables()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { device: device.clone(), queue: queue.clone(), pipeline, bgl, blue }
    }

    /// Queues the bake; the GPU runs it before anything submitted later samples the texture.
    pub(crate) fn bake(&self, tex: &wgpu::Texture, res: u32, params: &[f32], seed: u32) {
        use wgpu::util::DeviceExt;
        let d = &self.device;
        let init = |contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage });
        let uniforms = init(bytemuck::bytes_of(&[seed, 0, 0, 0]), wgpu::BufferUsages::UNIFORM);
        let params = init(bytemuck::cast_slice(params), wgpu::BufferUsages::STORAGE);
        let view = tex.create_view(&Default::default());
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.blue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let g = res.div_ceil(4);
            pass.dispatch_workgroups(g, g, g);
        }
        self.queue.submit([enc.finish()]);
    }
}

struct Entry {
    res: u32,
    /// Hash of what the texture holds (upstream WGSL, params, seed, resolution).
    key: u64,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Default)]
pub struct Caches {
    entries: BTreeMap<NodeId, Entry>,
    bakers: HashMap<String, Arc<Baker>>,
    compiling: HashMap<String, Receiver<Baker>>,
    /// Changes whenever the set of textures changes, so the preview rebinds them.
    pub generation: u64,
    /// Cached node → texture slot, for codegen.
    pub slots: BTreeMap<NodeId, u32>,
    /// Some cache waits for its bake pipeline to compile.
    pub pending: bool,
}

impl Caches {
    /// Brings the caches in line with the graph: bakes what changed, drops what's no longer cached.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, graph: &Graph, tiling: &BTreeMap<NodeId, Tiling>, seed: u32) {
        for (wgsl, rx) in std::mem::take(&mut self.compiling) {
            match rx.try_recv() {
                Ok(b) => _ = self.bakers.insert(wgsl, Arc::new(b)),
                Err(std::sync::mpsc::TryRecvError::Empty) => _ = self.compiling.insert(wgsl, rx),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
        let wanted: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(id, n)| n.cache.is_some() && graph.cache_eligibility(**id, tiling).is_ok())
            .map(|(&id, _)| id)
            .take(CACHE_SLOTS)
            .collect();
        let before = self.entries.len();
        self.entries.retain(|id, _| wanted.contains(id));
        if self.entries.len() != before {
            self.generation += 1;
        }
        self.slots = wanted.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();
        self.pending = false;

        for &id in &wanted {
            let res = graph.cache_resolution(id);
            // The bake evaluates the node's full upstream graph (no nested caches).
            let Ok(c) = codegen::compile(&graph.previewing(id)) else { continue };
            let mut h = std::collections::hash_map::DefaultHasher::new();
            c.wgsl.hash(&mut h);
            c.params.iter().for_each(|p| p.to_bits().hash(&mut h));
            (seed, res).hash(&mut h);
            let key = h.finish();
            if self.entries.get(&id).is_some_and(|e| e.key == key) {
                continue;
            }
            let Some(baker) = self.bakers.get(&c.wgsl).cloned() else {
                if !self.compiling.contains_key(&c.wgsl) {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let (d, q, w) = (device.clone(), queue.clone(), c.wgsl.clone());
                    std::thread::spawn(move || _ = tx.send(Baker::new(&d, &q, &w)));
                    self.compiling.insert(c.wgsl.clone(), rx);
                }
                self.pending = true;
                continue;
            };
            if self.entries.get(&id).is_none_or(|e| e.res != res) {
                let tex = texture(device, res);
                let view = tex.create_view(&Default::default());
                self.entries.insert(id, Entry { res, key: 0, tex, view });
                self.generation += 1;
            }
            let e = self.entries.get_mut(&id).unwrap();
            baker.bake(&e.tex, res, &c.params, seed);
            e.key = key;
        }
    }

    /// Texture views in slot order; slots still waiting for their first bake get None.
    pub fn views(&self) -> Vec<Option<&wgpu::TextureView>> {
        self.slots.keys().map(|id| self.entries.get(id).map(|e| &e.view)).collect()
    }

    /// GPU memory held by cache textures.
    pub fn bytes(&self) -> u64 {
        self.entries.values().map(|e| (e.res as u64).pow(3) * 4).sum()
    }
}
