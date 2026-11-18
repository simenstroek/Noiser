//! GPU side of the 3D preview and 2D slice view: one shader module built from the graph,
//! a raymarch pipeline (rendering into an offscreen texture), a slice compute pipeline and a
//! slice display pipeline, plus GPU timestamps for both views.

use crate::codegen;
use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// shortcut: fixed-size param buffer (16 KiB); grow on demand if graphs ever need more.
pub const MAX_PARAMS: usize = 4096;
/// Slice texels per side.
pub const SLICE_RES: u32 = 512;

/// Mirrors `Uniforms` in common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub cam_pos: [f32; 3],
    pub aspect: f32,
    pub cam_fwd: [f32; 3],
    pub amount: f32,
    pub cam_right: [f32; 3],
    pub step_scale: f32,
    pub cam_up: [f32; 3],
    pub seed: u32,
    pub tile_size: f32,
    pub prim_size: f32,
    /// 2 / viewport height in pixels (size of one pixel in uv units).
    pub pixel: f32,
    pub cloud_density: f32,
    pub mode: u32,
    pub prim: u32,
    pub slice_axis: u32,
    pub slice_mode: u32,
    pub slice_pos: f32,
    pub slice_aspect: f32,
    pub slice_tiles: f32,
    pub slice_res: u32,
    pub push_min: f32,
    pub push_max: f32,
    /// [march step limit (0 = 300), unused]
    pub _pad: [f32; 2],
    /// Min corner and texel size of the mesh distance grid.
    pub mesh_origin: [f32; 3],
    pub mesh_voxel: f32,
    /// The mesh's bounds, the number of mask terms and the terms (see meshmask.rs).
    pub mesh_lo: [f32; 3],
    pub mask_n: u32,
    pub mesh_hi: [f32; 3],
    pub _pad_mask: f32,
    pub mask: [[f32; 4]; 12],
    /// [colour rgb, specular strength], [roughness, shadows (0/1), unused, unused]: viewport only.
    pub look: [[f32; 4]; 2],
    /// Lighting (viewport only): [sun direction xyz, sun intensity], [sun colour rgb, ambient], [sky top rgb, 0], [sky bottom rgb, 0].
    pub light: [[f32; 4]; 4],
}

/// How far the surface can move in/out: amount × the displacement mask × the graph's density range,
/// kept finite for the shader (1e6 effectively disables the shell skip for unbounded graphs).
/// `mask` is the range of the mask (1, 1 when there is none).
pub fn push_range((lo, hi): (f32, f32), amount: f32, mask: (f32, f32)) -> (f32, f32) {
    let products = [mask.0 * lo, mask.0 * hi, mask.1 * lo, mask.1 * hi].map(|p| amount * p);
    if products.iter().any(|p| p.is_nan()) {
        // 0 times infinity: an unbounded graph with nothing to scale it.
        return (-1e6, 1e6);
    }
    let (a, b) = (products.iter().cloned().fold(f32::INFINITY, f32::min), products.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    (a.clamp(-1e6, 1e6), b.clamp(-1e6, 1e6))
}

// ---- GPU timing ------------------------------------------------------------

const IDLE: u8 = 0;
const WRITTEN: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;

/// Timestamp queries around the 3D pass (0, 1) and the slice pass (2, 3), read back without
/// stalling: written in one frame, mapped in a later one, read when the map completes.
pub struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    state: Arc<AtomicU8>,
    period_ns: f32,
    /// Last measured GPU time in milliseconds: [3D view, slice].
    pub ms: [f32; 2],
}

impl Timer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("timer"), ty: wgpu::QueryType::Timestamp, count: 4 });
        let buf = |usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some("timer"), size: 32, usage, mapped_at_creation: false });
        Some(Self {
            set,
            resolve: buf(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buf(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            state: Arc::new(AtomicU8::new(IDLE)),
            period_ns: queue.get_timestamp_period(),
            ms: [0.0; 2],
        })
    }

    fn pass_writes(&self, pass: u32) -> Option<(u32, u32)> {
        (self.state.load(Ordering::Acquire) == IDLE).then_some((pass * 2, pass * 2 + 1))
    }

    /// Copies this frame's timestamps out. Call after both passes are encoded.
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.state.load(Ordering::Acquire) == IDLE {
            encoder.resolve_query_set(&self.set, 0..4, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, 32);
            self.state.store(WRITTEN, Ordering::Release);
        }
    }

    /// Advances the readback; call once per frame (before encoding) and after submits.
    pub fn collect(&mut self, device: &wgpu::Device) {
        match self.state.load(Ordering::Acquire) {
            WRITTEN => {
                self.state.store(MAPPING, Ordering::Release);
                let state = self.state.clone();
                self.read.map_async(wgpu::MapMode::Read, .., move |r| {
                    state.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release)
                });
            }
            READY => {
                {
                    let view = self.read.get_mapped_range(..).unwrap();
                    let t: &[u64] = bytemuck::cast_slice(&view);
                    for i in 0..2 {
                        self.ms[i] = t[2 * i + 1].saturating_sub(t[2 * i]) as f32 * self.period_ns / 1e6;
                    }
                }
                self.read.unmap();
                self.state.store(IDLE, Ordering::Release);
            }
            _ => {}
        }
        let _ = device.poll(wgpu::PollType::Poll);
    }
}

// ---- pipelines and passes ----------------------------------------------------

struct Target {
    size: [u32; 2],
    view: wgpu::TextureView,
    blit: wgpu::BindGroup,
}

pub struct Gpu {
    device: wgpu::Device,
    format: wgpu::TextureFormat,
    raymarch: wgpu::RenderPipeline,
    raymarch_cloud: wgpu::RenderPipeline,
    slice_cs: wgpu::ComputePipeline,
    slice_fs: wgpu::RenderPipeline,
    /// CPU copy of the slice values, read back only while the pointer is over the slice view (for the value readout).
    slice_read: wgpu::Buffer,
    slice_state: Arc<AtomicU8>,
    slice_cpu: Arc<std::sync::Mutex<Vec<f32>>>,
    slice_wanted: Arc<std::sync::atomic::AtomicBool>,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    slice: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cache_layout: wgpu::BindGroupLayout,
    cache_sampler: wgpu::Sampler,
    cache_dummy: wgpu::TextureView,
    cache_bind: wgpu::BindGroup,
    /// Which set of cache textures cache_bind holds (see set_caches).
    pub cache_gen: u64,
    mesh_layout: wgpu::BindGroupLayout,
    mesh_dummy: wgpu::TextureView,
    mesh_bind: wgpu::BindGroup,
    /// Which mesh grid mesh_bind holds (see set_mesh).
    pub mesh_gen: u64,
    target: Option<Target>,
    pub timer: Option<Timer>,
}

impl Gpu {
    /// Uploads uniforms and params. Both views do it (same data), so draw order doesn't matter.
    pub fn upload(&self, queue: &wgpu::Queue, u: &Uniforms, params: &[f32]) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(u));
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(params));
    }

    /// Raymarches the 3D view into the offscreen target (resized to `size` pixels as needed).
    pub fn render_3d(&mut self, encoder: &mut wgpu::CommandEncoder, size: [u32; 2], cloud: bool) {
        let size = [size[0].max(1), size[1].max(1)];
        if self.target.as_ref().is_none_or(|t| t.size != size) {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("3d view"),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let blit = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.blit_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.target = Some(Target { size, view, blit });
        }
        let timestamps = self.timer.as_ref().and_then(|t| {
            t.pass_writes(0).map(|(b, e)| wgpu::RenderPassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(b), end_of_pass_write_index: Some(e) })
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3d view"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target.as_ref().unwrap().view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timestamps,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(if cloud { &self.raymarch_cloud } else { &self.raymarch });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &self.cache_bind, &[]);
        pass.set_bind_group(2, &self.mesh_bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Evaluates the graph over the slice and its min/max.
    /// Advances the slice readback (map, copy to `slice_cpu`, unmap); call once per frame.
    pub fn collect_slice(&self) {
        match self.slice_state.load(Ordering::Acquire) {
            WRITTEN => {
                self.slice_state.store(MAPPING, Ordering::Release);
                let state = self.slice_state.clone();
                self.slice_read.map_async(wgpu::MapMode::Read, .., move |r| state.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release));
            }
            READY => {
                {
                    let view = self.slice_read.get_mapped_range(..).unwrap();
                    if let Ok(mut cpu) = self.slice_cpu.lock() {
                        cpu.clear();
                        cpu.extend_from_slice(bytemuck::cast_slice(&view));
                    }
                }
                self.slice_read.unmap();
                self.slice_state.store(IDLE, Ordering::Release);
            }
            _ => {}
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Asks for (or stops asking for) the CPU copy of the slice values.
    pub fn want_slice_values(&self, wanted: bool) {
        self.slice_wanted.store(wanted, Ordering::Relaxed);
    }

    /// The slice value at texel (x, y) of the last copy that arrived, if any.
    pub fn slice_value(&self, x: u32, y: u32) -> Option<f32> {
        self.slice_cpu.lock().ok()?.get((y * SLICE_RES + x) as usize).copied()
    }

    pub fn compute_slice(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        // Reset min/max (queue writes land before this frame's commands run).
        queue.write_buffer(&self.slice, 0, bytemuck::cast_slice(&[u32::MAX, 0u32]));
        let timestamps = self.timer.as_ref().and_then(|t| {
            t.pass_writes(1).map(|(b, e)| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(b), end_of_pass_write_index: Some(e) })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("slice"), timestamp_writes: timestamps });
        pass.set_pipeline(&self.slice_cs);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &self.cache_bind, &[]);
        pass.set_bind_group(2, &self.mesh_bind, &[]);
        let groups = SLICE_RES.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
        drop(pass);
        if self.slice_wanted.load(Ordering::Relaxed) && self.slice_state.load(Ordering::Acquire) == IDLE {
            encoder.copy_buffer_to_buffer(&self.slice, 8, &self.slice_read, 0, (SLICE_RES * SLICE_RES * 4) as u64);
            self.slice_state.store(WRITTEN, Ordering::Release);
        }
        if let Some(t) = &self.timer {
            t.resolve(encoder);
        }
    }
}

// ---- egui callbacks ------------------------------------------------------------

pub struct View3d {
    pub uniforms: Uniforms,
    pub params: Vec<f32>,
    /// Viewport size in physical pixels.
    pub size: [u32; 2],
}

impl egui_wgpu::CallbackTrait for View3d {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu: &mut Gpu = res.get_mut().unwrap();
        if let Some(t) = &mut gpu.timer {
            t.collect(device);
        }
        gpu.upload(queue, &self.uniforms, &self.params);
        gpu.render_3d(encoder, self.size, self.uniforms.mode == 1);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let gpu: &Gpu = res.get().unwrap();
        if let Some(t) = &gpu.target {
            pass.set_pipeline(&gpu.blit);
            pass.set_bind_group(0, &t.blit, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

pub struct SliceView {
    pub uniforms: Uniforms,
    pub params: Vec<f32>,
}

impl egui_wgpu::CallbackTrait for SliceView {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu: &Gpu = res.get().unwrap();
        gpu.collect_slice();
        gpu.upload(queue, &self.uniforms, &self.params);
        gpu.compute_slice(queue, encoder);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let gpu: &Gpu = res.get().unwrap();
        pass.set_pipeline(&gpu.slice_fs);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.set_bind_group(1, &gpu.cache_bind, &[]);
        pass.set_bind_group(2, &gpu.mesh_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ---- construction --------------------------------------------------------------

/// Node caches the preview shader can sample (bindings in common.wgsl).
pub const CACHE_SLOTS: usize = 8;

fn cache_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, views: &[Option<&wgpu::TextureView>], dummy: &wgpu::TextureView) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(sampler) }];
    for slot in 0..CACHE_SLOTS {
        let view = views.get(slot).copied().flatten().unwrap_or(dummy);
        entries.push(wgpu::BindGroupEntry { binding: slot as u32 + 1, resource: wgpu::BindingResource::TextureView(view) });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("caches"), layout, entries: &entries })
}

fn mesh_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, view: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh"),
        layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }],
    })
}

impl Gpu {
    /// Binds the mesh distance grid (None goes back to the dummy). `generation` identifies it so the
    /// app can tell when a freshly built pipeline still needs it.
    pub fn set_mesh(&mut self, view: Option<&wgpu::TextureView>, generation: u64) {
        self.mesh_bind = mesh_bind_group(&self.device, &self.mesh_layout, view.unwrap_or(&self.mesh_dummy));
        self.mesh_gen = generation;
    }

    /// Binds the cache textures, in slot order. `generation` identifies this set so the app can
    /// tell when a freshly built pipeline still needs them.
    pub fn set_caches(&mut self, views: &[Option<&wgpu::TextureView>], generation: u64) {
        self.cache_bind = cache_bind_group(&self.device, &self.cache_layout, &self.cache_sampler, views, &self.cache_dummy);
        self.cache_gen = generation;
    }
}

/// Full shader source for a compiled graph. `mask` compiles the mesh displacement mask; without it the
/// mask is a stub, which keeps pipeline builds fast for everything that is not a masked mesh.
pub fn shader_source(graph_wgsl: &str, mask: bool, extras: Extras) -> String {
    let mut s = format!(
        "{}\n{}\n{graph_wgsl}\n{}\n{}\n{}",
        codegen::NOISE_LIB,
        codegen::mesh_mask_lib(mask),
        include_str!("common.wgsl"),
        include_str!("raymarch.wgsl"),
        include_str!("slice.wgsl")
    );
    if extras.mask_view {
        s = s.replace("const MASK_VIEW: bool = false;", "const MASK_VIEW: bool = true;");
        s = s.replace("fn view_mask(p: vec3f) -> f32 { return 1.0; } // STUB view_mask", "fn view_mask(p: vec3f) -> f32 { return mask_at(p); }");
    }
    s
}

/// Parts of the preview shader that are compiled only when used (see the stubs in raymarch.wgsl).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extras {
    /// Colour the mesh by its displacement mask (and draw it undisplaced).
    pub mask_view: bool,
}

const BLIT_WGSL: &str = "
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return VsOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}
@fragment fn fs(in: VsOut) -> @location(0) vec4f { return textureSample(img, smp, in.uv); }
";

/// Adds TIMESTAMP_QUERY (GPU timing) and FLOAT32_FILTERABLE (smooth caches) to eframe's device request when supported.
pub fn request_timestamps(mut cfg: egui_wgpu::WgpuConfiguration) -> egui_wgpu::WgpuConfiguration {
    if let egui_wgpu::WgpuSetup::CreateNew(new) = &mut cfg.wgpu_setup {
        // Never probe OpenGL: Noiser needs Vulkan or DX12, and some AMD GL drivers (atio6axx.dll)
        // segfault during adapter enumeration, killing the app before a window appears.
        // WGPU_BACKEND still overrides this.
        if wgpu::Backends::from_env().is_none() {
            new.instance_descriptor.backends = wgpu::Backends::PRIMARY;
        }
        let base = new.device_descriptor.clone();
        new.device_descriptor = Arc::new(move |adapter| {
            let mut d = base(adapter);
            d.required_features |= adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | crate::cache::FEATURE);
            d
        });
    }
    cfg
}

pub fn build(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, graph_wgsl: &str, mask: bool, extras: Extras) -> Gpu {
    // shortcut: no error scope here; the full shader is naga-validated in tests. Add one if users hit shader panics.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("preview"),
        source: wgpu::ShaderSource::Wgsl(shader_source(graph_wgsl, mask, extras).into()),
    });
    let stages = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
        ],
    });
    // Group 1: node caches (sampler + 8 float 3D textures). Without float32 filtering the
    // caches are sampled unfiltered (blocky), but everything still binds.
    let filterable = device.features().contains(crate::cache::FEATURE);
    let tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    };
    let mut cache_entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: stages,
        ty: wgpu::BindingType::Sampler(if filterable { wgpu::SamplerBindingType::Filtering } else { wgpu::SamplerBindingType::NonFiltering }),
        count: None,
    }];
    cache_entries.extend((1..=CACHE_SLOTS as u32).map(tex));
    let cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("caches"), entries: &cache_entries });
    let cache_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("caches"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: if filterable { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest },
        min_filter: if filterable { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest },
        ..Default::default()
    });
    let cache_dummy = crate::cache::texture(device, 1).create_view(&Default::default());
    let cache_bind = cache_bind_group(device, &cache_layout, &cache_sampler, &[], &cache_dummy);
    // Group 2: the mesh distance grid, read with textureLoad (no filtering needed).
    let mesh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stages,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }],
    });
    let mesh_dummy = crate::cache::texture(device, 1).create_view(&Default::default());
    let mesh_bind = mesh_bind_group(device, &mesh_layout, &mesh_dummy);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl), Some(&cache_layout), Some(&mesh_layout)],
        immediate_size: 0,
    });
    let render = |module: &wgpu::ShaderModule, layout: Option<&wgpu::PipelineLayout>, fs: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs),
            layout,
            vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    // The driver compiles each entry point separately and takes about a second for a big graph, so the three
    // pipelines are created on parallel threads.
    let (raymarch, raymarch_cloud, slice_fs, slice_cs) = std::thread::scope(|scope| {
        let cloud = scope.spawn(|| render(&module, Some(&layout), "fs_cloud"));
        let fs = scope.spawn(|| render(&module, Some(&layout), "fs_slice"));
        let cs = scope.spawn(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slice"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("cs_slice"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let raymarch = render(&module, Some(&layout), "fs");
        (raymarch, cloud.join().expect("cloud pipeline"), fs.join().expect("slice pipeline"), cs.join().expect("slice compute pipeline"))
    });
    let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()) });
    let blit = render(&blit_module, None, "fs");
    let blit_layout = blit.get_bind_group_layout(0);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    };
    let uniforms = buffer("uniforms", std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM);
    let params = buffer("params", (MAX_PARAMS * 4) as u64, wgpu::BufferUsages::STORAGE);
    let slice = buffer("slice", 8 + (SLICE_RES * SLICE_RES * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let slice_read = buffer("slice readback", (SLICE_RES * SLICE_RES * 4) as u64, wgpu::BufferUsages::MAP_READ);
    let blue: &[f32] = &crate::bluenoise::tables();
    let blue_noise = buffer("blue noise", std::mem::size_of_val(blue) as u64, wgpu::BufferUsages::STORAGE);
    queue.write_buffer(&blue_noise, 0, bytemuck::cast_slice(blue));
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: slice.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: blue_noise.as_entire_binding() },
        ],
    });
    Gpu {
        device: device.clone(),
        format,
        raymarch,
        raymarch_cloud,
        slice_cs,
        slice_fs,
        slice_read,
        slice_state: Arc::new(AtomicU8::new(IDLE)),
        slice_cpu: Arc::new(std::sync::Mutex::new(Vec::new())),
        slice_wanted: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        blit,
        blit_layout,
        sampler,
        uniforms,
        params,
        slice,
        bind_group,
        cache_bind,
        cache_layout,
        cache_sampler,
        cache_dummy,
        cache_gen: 0,
        mesh_layout,
        mesh_dummy,
        mesh_bind,
        mesh_gen: 0,
        target: None,
        timer: Timer::new(device, queue),
    }
}
