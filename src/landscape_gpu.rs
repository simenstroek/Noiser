//! The Landscape tab's 3D view on the GPU: the height grid goes up once as a texture when the terrain changes, and
//! `landscape_view.wgsl` ray marches it into egui's pass for the view's rectangle, so orbiting costs only a few uniforms.
//! The CPU renderer in `landscape.rs` is the reference the tests compare it with.

use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;

/// The smallest march step, in cells (the CPU renderer's).
pub const MIN_STEP_CELLS: f32 = 0.9;

/// The heights the view draws, and what is worked out from them once: the lowest and highest ground and the steepest
/// slope (both with the water surface counted as ground).
pub struct Grid {
    pub key: u64,
    pub heights: Arc<Vec<f32>>,
    pub n: usize,
    pub lo: f32,
    pub hi: f32,
    /// Metres up per metre across, with a margin for diagonals and interpolation.
    pub steepest: f32,
}

impl Grid {
    pub fn new(key: u64, heights: Vec<f32>, n: usize, world_m: f32, water: Option<f32>) -> Grid {
        let level = water.unwrap_or(f32::MIN);
        let (lo, hi) = heights.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        let cell = world_m / (n - 1) as f32;
        let steepest = steepest_slope(&heights, n, level) / cell * 1.5;
        Grid { key, heights: Arc::new(heights), n, lo, hi, steepest }
    }
}

/// The biggest height step between neighbouring samples (water counted as ground at its level).
pub fn steepest_slope(heights: &[f32], n: usize, level: f32) -> f32 {
    (0..n * n)
        .map(|k| {
            let (i, j) = (k % n, k / n);
            let h = heights[k].max(level);
            let dx = if i + 1 < n { (heights[k + 1].max(level) - h).abs() } else { 0.0 };
            let dy = if j + 1 < n { (heights[k + n].max(level) - h).abs() } else { 0.0 };
            dx.max(dy)
        })
        .fold(0.0f32, f32::max)
}

/// The orbiting camera: it looks at the middle of the terrain from `dist` away. Z is up. Returns eye, forward, right, up.
pub fn camera(world_m: f32, lo: f32, hi: f32, (yaw, pitch, zoom): (f32, f32, f32)) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let target = [world_m * 0.5, world_m * 0.5, (lo + hi) * 0.5];
    let dist = world_m * 1.3 / zoom.max(0.1);
    let eye = [target[0] + dist * pitch.cos() * yaw.cos(), target[1] + dist * pitch.cos() * yaw.sin(), target[2] + dist * pitch.sin()];
    let fwd = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let right = norm([fwd[1], -fwd[0], 0.0]);
    let up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
    (eye, fwd, right, up)
}

/// tan of half the vertical field of view (40 degrees).
pub fn tan_half_fov() -> f32 {
    (40.0f32.to_radians() * 0.5).tan()
}

pub fn light() -> [f32; 3] {
    let l = [0.5f32, 0.35, 0.8];
    let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
    [l[0] / n, l[1] / n, l[2] / n]
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniforms {
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    light: [f32; 4],
    range: [f32; 4],
    extra: [f32; 4],
}

impl ViewUniforms {
    /// The uniforms for a picture `aspect` (width / height) wide, drawn into a target that is sRGB or not.
    pub fn new(grid: &Grid, world_m: f32, water: Option<f32>, view: (f32, f32, f32), aspect: f32, srgb: bool, min_step_cells: f32, slope_steps: bool) -> ViewUniforms {
        let (eye, fwd, right, up) = camera(world_m, grid.lo, grid.hi, view);
        let cell = world_m / (grid.n - 1) as f32;
        let level = water.unwrap_or(-1.0e30);
        let (floor, top) = (grid.lo.min(level.max(grid.lo)) - 1.0, grid.hi.max(level) + 1.0);
        // Without slope steps (for checking that they miss nothing) every step is the smallest.
        let steepest = if slope_steps { grid.steepest } else { 1.0e30 };
        let l = light();
        ViewUniforms {
            eye: [eye[0], eye[1], eye[2], tan_half_fov()],
            fwd: [fwd[0], fwd[1], fwd[2], aspect],
            right: [right[0], right[1], right[2], world_m],
            up: [up[0], up[1], up[2], cell],
            light: [l[0], l[1], l[2], steepest],
            range: [grid.lo, grid.hi, level, if srgb { 1.0 } else { 0.0 }],
            extra: [grid.n as f32, floor, top, min_step_cells],
        }
    }
}

/// The pipeline, the uniforms and the uploaded heights, kept in egui's callback resources.
pub struct LandscapeGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    format: wgpu::TextureFormat,
    /// The key of the heights uploaded, their texture and the bind group that reads them.
    heights: Option<(u64, wgpu::Texture, wgpu::BindGroup)>,
}

impl LandscapeGpu {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> LandscapeGpu {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("landscape view"), source: wgpu::ShaderSource::Wgsl(include_str!("landscape_view.wgsl").into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("landscape view"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("landscape view"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("landscape view"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("landscape view"), size: std::mem::size_of::<ViewUniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        LandscapeGpu { pipeline, layout, uniforms, format, heights: None }
    }

    /// Uploads the heights when they are not the ones already there.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: u64, heights: &[f32], n: usize) {
        if self.heights.as_ref().is_some_and(|(k, _, _)| *k == key) {
            return;
        }
        let size = wgpu::Extent3d { width: n as u32, height: n as u32, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("landscape heights"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(heights),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * n as u32), rows_per_image: Some(n as u32) },
            size,
        );
        let view = texture.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("landscape view"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) }],
        });
        self.heights = Some((key, texture, group));
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some((_, _, group)) = &self.heights else { return };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The view's paint callback: the camera, and the heights (uploaded only when their key changes).
pub struct ViewCallback {
    pub uniforms: ViewUniforms,
    pub key: u64,
    pub heights: Arc<Vec<f32>>,
    pub n: usize,
    pub format: wgpu::TextureFormat,
}

impl egui_wgpu::CallbackTrait for ViewCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, _screen: &egui_wgpu::ScreenDescriptor, _encoder: &mut wgpu::CommandEncoder, resources: &mut egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<LandscapeGpu>().is_none_or(|g| g.format != self.format) {
            resources.insert(LandscapeGpu::new(device, self.format));
        }
        let gpu: &mut LandscapeGpu = resources.get_mut().expect("just inserted");
        gpu.upload(device, queue, self.key, &self.heights, self.n);
        queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        if let Some(gpu) = resources.get::<LandscapeGpu>() {
            gpu.draw(pass);
        }
    }
}

/// Renders the view into a `w` x `h` RGBA picture (for tests).
#[cfg(test)]
pub fn render_image(device: &wgpu::Device, queue: &wgpu::Queue, grid: &Grid, uniforms: &ViewUniforms, (w, h): (u32, u32)) -> Vec<[u8; 4]> {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut gpu = LandscapeGpu::new(device, format);
    gpu.upload(device, queue, grid.key, &grid.heights, grid.n);
    queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(uniforms));
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let row = (4 * w).div_ceil(256) * 256;
    let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        gpu.draw(&mut pass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let slice = read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let bytes = slice.get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let k = (y * row + 4 * x) as usize;
            out.push([bytes[k], bytes[k + 1], bytes[k + 2], bytes[k + 3]]);
        }
    }
    out
}
