//! GPU side of the 3D preview and 2D slice view: one shader module built from the graph,
//! a raymarch pipeline, a slice compute pipeline and a slice display pipeline.

use crate::codegen;
use eframe::{egui, egui_wgpu, wgpu};

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
}

pub struct Gpu {
    raymarch: wgpu::RenderPipeline,
    slice_cs: wgpu::ComputePipeline,
    slice_fs: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    slice: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Uploads uniforms and params. Every callback does it (same data), so draw order doesn't matter.
fn upload(queue: &wgpu::Queue, gpu: &Gpu, u: &Uniforms, params: &[f32]) {
    queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(u));
    queue.write_buffer(&gpu.params, 0, bytemuck::cast_slice(params));
}

pub struct View3d {
    pub uniforms: Uniforms,
    pub params: Vec<f32>,
}

impl egui_wgpu::CallbackTrait for View3d {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        upload(queue, res.get().unwrap(), &self.uniforms, &self.params);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let gpu: &Gpu = res.get().unwrap();
        pass.set_pipeline(&gpu.raymarch);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.draw(0..3, 0..1);
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
        upload(queue, gpu, &self.uniforms, &self.params);
        // Reset min/max (queue writes land before this frame's commands run).
        queue.write_buffer(&gpu.slice, 0, bytemuck::cast_slice(&[u32::MAX, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("slice"), timestamp_writes: None });
        pass.set_pipeline(&gpu.slice_cs);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        let groups = SLICE_RES.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let gpu: &Gpu = res.get().unwrap();
        pass.set_pipeline(&gpu.slice_fs);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Full shader source for a compiled graph.
pub fn shader_source(graph_wgsl: &str) -> String {
    format!(
        "{}\n{graph_wgsl}\n{}\n{}\n{}",
        codegen::NOISE_LIB,
        include_str!("common.wgsl"),
        include_str!("raymarch.wgsl"),
        include_str!("slice.wgsl")
    )
}

pub fn build(rs: &egui_wgpu::RenderState, graph_wgsl: &str) -> Gpu {
    let device = &rs.device;
    // shortcut: no error scope here; the full shader is naga-validated in tests. Add one if users hit shader panics.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("preview"),
        source: wgpu::ShaderSource::Wgsl(shader_source(graph_wgsl).into()),
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
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let render = |fs: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(rs.target_format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let raymarch = render("fs");
    let slice_fs = render("fs_slice");
    let slice_cs = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("slice"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("cs_slice"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    };
    let uniforms = buffer("uniforms", std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM);
    let params = buffer("params", (MAX_PARAMS * 4) as u64, wgpu::BufferUsages::STORAGE);
    let slice = buffer("slice", 8 + (SLICE_RES * SLICE_RES * 4) as u64, wgpu::BufferUsages::STORAGE);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: slice.as_entire_binding() },
        ],
    });
    Gpu { raymarch, slice_cs, slice_fs, uniforms, params, slice, bind_group }
}
