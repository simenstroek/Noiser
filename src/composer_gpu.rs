//! The SDF Composer's GPU viewport: the flattened shape tree (`sdf_model::GpuScene`) goes into two storage buffers and
//! `composer_view.wgsl` interprets it, so changing a value never rebuilds a pipeline. The picture is drawn straight into
//! egui's pass for the viewport's rectangle.

use crate::composer_reference::ReferenceMesh;
use crate::composer_render::{Camera, FIELD_OF_VIEW};
use crate::sdf_model::{GPU_SHAPE_STRIDE, GpuScene};
use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;

/// The interpreter's WGSL, shared with the Load SDF node.
pub const SDF_LIBRARY: &str = include_str!("composer_sdf_lib.wgsl");

pub const MAX_SHAPES: usize = 1024;
pub const MAX_OPS: usize = 4096;
/// Points the test entry point can evaluate in one go.
pub const MAX_TEST_POINTS: usize = 8192;

pub const DEFAULT_STEPS: u32 = 260;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    light: [f32; 4],
    params: [f32; 4],
    counts: [u32; 4],
    box_lo: [f32; 4],
    box_hi: [f32; 4],
    settings: [f32; 4],
}

fn light_direction(forward: [f64; 3], right: [f64; 3], up: [f64; 3]) -> [f32; 4] {
    let l = [
        -forward[0] * 0.6 - right[0] * 0.5 + up[0] * 0.7,
        -forward[1] * 0.6 - right[1] * 0.5 + up[1] * 0.7,
        -forward[2] * 0.6 - right[2] * 0.5 + up[2] * 0.7,
    ];
    let length = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt().max(1e-9);
    [(l[0] / length) as f32, (l[1] / length) as f32, (l[2] / length) as f32, 0.0]
}

impl Uniforms {
    /// Whether the shapes are drawn in their own colours (else all grey).
    pub fn with_colors(mut self, on: bool) -> Uniforms {
        self.counts[3] = on as u32;
        self
    }

    /// The most steps a ray may take before it counts as a miss.
    pub fn with_steps(mut self, steps: u32) -> Uniforms {
        self.settings[0] = steps as f32;
        self
    }

    pub fn new(camera: &Camera, aspect: f32, scene: &GpuScene, show_selection: bool) -> Uniforms {
        let (forward, right, up) = camera.basis();
        let eye = camera.eye();
        let f = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        // The scene's box, grown a little: smooth blends reach past the shapes' own extents. No box: nothing is drawn.
        let (box_lo, box_hi) = match scene.bounds {
            Some((lo, hi)) => {
                let grow = [0, 1, 2].map(|a| (hi[a] - lo[a]) * 0.08 + 0.05);
                ([0, 1, 2].map(|a| (lo[a] - grow[a]) as f32), [0, 1, 2].map(|a| (hi[a] + grow[a]) as f32))
            }
            None => ([1.0; 3], [-1.0; 3]),
        };
        Uniforms {
            box_lo: [box_lo[0], box_lo[1], box_lo[2], 0.0],
            box_hi: [box_hi[0], box_hi[1], box_hi[2], 0.0],
            eye: f(eye),
            forward: f(forward),
            right: f(right),
            up: f(up),
            light: light_direction(forward, right, up),
            params: [aspect, (FIELD_OF_VIEW * 0.5).tan() as f32, (camera.distance * 3.0 + 60.0) as f32, if show_selection { 1.0 } else { 0.0 }],
            counts: [scene.ops.len() as u32, scene.shape_count as u32, 0, 0],
            settings: [DEFAULT_STEPS as f32, 1.0, 0.0, 0.0],
        }
    }
}

/// What the reference model's vertex shader needs: the camera and the model matrix.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReferenceUniforms {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    light: [f32; 4],
    params: [f32; 4],
    rows: [[f32; 4]; 3],
}

impl ReferenceUniforms {
    pub fn new(camera: &Camera, aspect: f32, model: [[f64; 4]; 3]) -> ReferenceUniforms {
        let (forward, right, up) = camera.basis();
        let f = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        ReferenceUniforms {
            eye: f(camera.eye()),
            forward: f(forward),
            right: f(right),
            up: f(up),
            light: light_direction(forward, right, up),
            params: [aspect, (FIELD_OF_VIEW * 0.5).tan() as f32, 0.05, (camera.distance * 4.0 + 200.0) as f32],
            rows: model.map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]),
        }
    }
}

/// A reference model to draw this frame.
pub struct ReferenceDraw {
    pub mesh: Arc<ReferenceMesh>,
    pub uniforms: ReferenceUniforms,
    pub opacity: f32,
}

const REFERENCE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const REFERENCE_DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const REFERENCE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The off-screen pictures of one frame size: the composition, and the reference model's colour and distance.
struct Targets {
    size: [u32; 2],
    with_reference: bool,
    #[cfg_attr(not(test), allow(dead_code))]
    scene_texture: wgpu::Texture,
    scene: wgpu::TextureView,
    reference_color: wgpu::TextureView,
    reference_distance: wgpu::TextureView,
    reference_depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    blit_group: wgpu::BindGroup,
}

/// The buffers every frame's bind group points at.
struct Buffers {
    uniforms: wgpu::Buffer,
    shapes: wgpu::Buffer,
    ops: wgpu::Buffer,
    points: wgpu::Buffer,
}

pub struct ComposerGpu {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    #[cfg_attr(not(test), allow(dead_code))]
    evaluate: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: Buffers,
    /// The scene version the buffers hold.
    uploaded: Option<u64>,
    targets: Targets,
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    reference_pipeline: wgpu::RenderPipeline,
    reference_uniforms: wgpu::Buffer,
    reference_group: wgpu::BindGroup,
    reference_buffers: Option<(u64, wgpu::Buffer, wgpu::Buffer, u32)>,
}

impl ComposerGpu {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> ComposerGpu {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composer view"),
            source: wgpu::ShaderSource::Wgsl(format!("{}
{}", include_str!("composer_view.wgsl"), SDF_LIBRARY).into()),
        });
        let entry = |binding: u32, ty, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let graphics = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composer view"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, graphics),
                entry(1, read_only, graphics),
                entry(2, read_only, graphics),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }, wgpu::ShaderStages::COMPUTE),
                texture_entry(4),
                texture_entry(5),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composer view"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composer view"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let evaluate = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("composer evaluate"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_eval"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let buffers = Buffers {
            uniforms: buffer("composer uniforms", std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM),
            shapes: buffer("composer shapes", (MAX_SHAPES * GPU_SHAPE_STRIDE * 16) as u64, wgpu::BufferUsages::STORAGE),
            ops: buffer("composer ops", (MAX_OPS * 16) as u64, wgpu::BufferUsages::STORAGE),
            points: buffer("composer test points", (MAX_TEST_POINTS * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
        };

        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("composer blit"), source: wgpu::ShaderSource::Wgsl(include_str!("composer_blit.wgsl").into()) });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composer blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composer blit"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("composer blit"), bind_group_layouts: &[Some(&blit_layout)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &blit_module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &blit_module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });

        let reference_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("composer reference"), source: wgpu::ShaderSource::Wgsl(include_str!("composer_reference.wgsl").into()) });
        let reference_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composer reference"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let vertex_attributes = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
        ];
        let reference_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composer reference"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("composer reference"), bind_group_layouts: &[Some(&reference_layout)], immediate_size: 0 })),
            vertex: wgpu::VertexState {
                module: &reference_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 24, step_mode: wgpu::VertexStepMode::Vertex, attributes: &vertex_attributes })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &reference_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(REFERENCE_COLOR_FORMAT.into()), Some(REFERENCE_DISTANCE_FORMAT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: REFERENCE_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let reference_uniforms = buffer("composer reference uniforms", std::mem::size_of::<ReferenceUniforms>() as u64, wgpu::BufferUsages::UNIFORM);
        let reference_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composer reference"),
            layout: &reference_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: reference_uniforms.as_entire_binding() }],
        });

        let targets = Self::make_targets(device, format, &layout, &blit_layout, &sampler, &buffers, [1, 1], false);
        ComposerGpu {
            format,
            pipeline,
            evaluate,
            layout,
            buffers,
            uploaded: None,
            targets,
            blit_pipeline,
            blit_layout,
            sampler,
            reference_pipeline,
            reference_uniforms,
            reference_group,
            reference_buffers: None,
        }
    }

    fn make_targets(device: &wgpu::Device, format: wgpu::TextureFormat, layout: &wgpu::BindGroupLayout, blit_layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, buffers: &Buffers, size: [u32; 2], with_reference: bool) -> Targets {
        let texture = |label: &str, size: [u32; 2], format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let reference_size = if with_reference { size } else { [1, 1] };
        let scene_texture = texture("composer picture", size, format);
        let scene = view(&scene_texture);
        let reference_color = view(&texture("composer reference colour", reference_size, REFERENCE_COLOR_FORMAT));
        let reference_distance = view(&texture("composer reference distance", reference_size, REFERENCE_DISTANCE_FORMAT));
        let reference_depth = view(&texture("composer reference depth", reference_size, REFERENCE_DEPTH_FORMAT));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composer view"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffers.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers.shapes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffers.ops.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buffers.points.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&reference_color) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&reference_distance) },
            ],
        });
        let blit_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composer blit"),
            layout: blit_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) }],
        });
        Targets { size, with_reference, scene_texture, scene, reference_color, reference_distance, reference_depth, bind_group, blit_group }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, size: [u32; 2], with_reference: bool) {
        if self.targets.size == size && self.targets.with_reference == with_reference {
            return;
        }
        self.targets = Self::make_targets(device, self.format, &self.layout, &self.blit_layout, &self.sampler, &self.buffers, size, with_reference);
    }

    fn upload_reference(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &ReferenceMesh) {
        if self.reference_buffers.as_ref().is_some_and(|(id, ..)| *id == mesh.id) {
            return;
        }
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composer reference vertices"),
            size: (mesh.vertices.len() * 24).max(24) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composer reference indices"),
            size: (mesh.indices.len() * 4).max(4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !mesh.vertices.is_empty() {
            queue.write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
            queue.write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));
        }
        self.reference_buffers = Some((mesh.id, vertices, indices, mesh.indices.len() as u32));
    }

    /// Draws one frame into the off-screen picture: the reference model first (colour and distance), then the
    /// composition, which reads them. `size` is in pixels of the picture, which `paint` stretches over the viewport.
    #[allow(clippy::too_many_arguments)]
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, scene: &GpuScene, version: u64, uniforms: Uniforms, reference: Option<&ReferenceDraw>, size: [u32; 2]) {
        self.upload(queue, scene, version);
        let limit = device.limits().max_texture_dimension_2d;
        let size = size.map(|s| s.clamp(1, limit));
        self.ensure_targets(device, size, reference.is_some());
        let mut uniforms = uniforms;
        uniforms.settings[1] = reference.map_or(1.0, |r| r.opacity.clamp(0.0, 1.0));
        uniforms.settings[2] = if reference.is_some() { 1.0 } else { 0.0 };
        self.write_uniforms(queue, &uniforms);
        if let Some(reference) = reference {
            self.upload_reference(device, queue, &reference.mesh);
            queue.write_buffer(&self.reference_uniforms, 0, bytemuck::bytes_of(&reference.uniforms));
            if let Some((_, vertices, indices, count)) = &self.reference_buffers {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("composer reference"),
                    color_attachments: &[
                        Some(clear_attachment(&self.targets.reference_color, wgpu::Color::TRANSPARENT)),
                        Some(clear_attachment(&self.targets.reference_distance, wgpu::Color { r: 1.0e9, g: 0.0, b: 0.0, a: 0.0 })),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.targets.reference_depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.reference_pipeline);
                pass.set_bind_group(0, &self.reference_group, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composer picture"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.scene,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.targets.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Copies a scene into the buffers (once per version). Returns false when it does not fit.
    pub fn upload(&mut self, queue: &wgpu::Queue, scene: &GpuScene, version: u64) -> bool {
        if self.uploaded == Some(version) {
            return true;
        }
        if scene.shape_count > MAX_SHAPES || scene.ops.len() > MAX_OPS {
            return false;
        }
        if !scene.shapes.is_empty() {
            queue.write_buffer(&self.buffers.shapes, 0, bytemuck::cast_slice(&scene.shapes));
        }
        if !scene.ops.is_empty() {
            queue.write_buffer(&self.buffers.ops, 0, bytemuck::cast_slice(&scene.ops));
        }
        self.uploaded = Some(version);
        true
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.buffers.uniforms, 0, bytemuck::bytes_of(uniforms));
    }

    /// The last picture as RGBA bytes (the test format is Rgba8Unorm), row by row. For tests.
    #[cfg(test)]
    pub fn read_picture(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ([u32; 2], Vec<u8>) {
        let [width, height] = self.targets.size;
        let row = (width * 4).next_multiple_of(256);
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * height) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.targets.scene_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = read.get_mapped_range(..).unwrap();
        let mut out = Vec::new();
        for y in 0..height as usize {
            out.extend_from_slice(&mapped[y * row as usize..y * row as usize + width as usize * 4]);
        }
        ([width, height], out)
    }

    /// Distances of the uploaded scene at `points`, computed on the GPU. For tests.
    #[cfg(test)]
    pub fn evaluate_points(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &GpuScene, points: &[[f32; 3]]) -> Vec<f32> {
        assert!(points.len() <= MAX_TEST_POINTS);
        let mut uniforms = Uniforms::new(&Camera::default(), 1.0, scene, false);
        uniforms.counts[2] = points.len() as u32;
        self.write_uniforms(queue, &uniforms);
        let data: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        queue.write_buffer(&self.buffers.points, 0, bytemuck::cast_slice(&data));
        let size = (points.len() * 16) as u64;
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.evaluate);
            pass.set_bind_group(0, &self.targets.bind_group, &[]);
            pass.dispatch_workgroups(points.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.buffers.points, 0, &read, 0, size);
        queue.submit([encoder.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = read.get_mapped_range(..).unwrap();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view[..]).to_vec();
        out.iter().map(|v| v[3]).collect()
    }
}

fn clear_attachment(view: &wgpu::TextureView, clear: wgpu::Color) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store } }
}

/// What the viewport callback needs: the camera, and the scene when it changed.
pub struct ViewCallback {
    pub uniforms: Uniforms,
    pub scene: Arc<GpuScene>,
    pub version: u64,
    pub format: wgpu::TextureFormat,
    /// The size of the picture to render, in pixels (the viewport's, times the render resolution).
    pub pixels: [u32; 2],
    pub reference: Option<ReferenceDraw>,
}

impl egui_wgpu::CallbackTrait for ViewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<ComposerGpu>().is_none() {
            resources.insert(ComposerGpu::new(device, self.format));
        }
        let gpu: &mut ComposerGpu = resources.get_mut().unwrap();
        gpu.render(device, queue, encoder, &self.scene, self.version, self.uniforms, self.reference.as_ref(), self.pixels);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(gpu) = resources.get::<ComposerGpu>() else { return };
        pass.set_pipeline(&gpu.blit_pipeline);
        pass.set_bind_group(0, &gpu.targets.blit_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_file::SdfFile;
    use crate::sdf_model::LayerField;
    use std::path::{Path, PathBuf};

    fn sample_files() -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut folders = Vec::new();
        if let Ok(dir) = std::env::var("NOISER_SDF_DIR") {
            folders.push(PathBuf::from(dir));
        }
        if let Ok(home) = std::env::var("USERPROFILE") {
            folders.push(Path::new(&home).join("Documents").join("SDF Modeler").join("sdfs"));
        }
        for folder in folders {
            for name in ["robot", "fried_egg", "icon", "bricks", "mech", "headphones", "shapes"] {
                let path = folder.join(format!("{name}.sdf"));
                if path.exists() {
                    files.push(path);
                }
            }
        }
        files
    }

    /// The GPU interpreter agrees with the CPU evaluator on every sample assembly, at points around the surface.
    #[test]
    fn the_gpu_interpreter_matches_the_cpu_evaluator() {
        let Some(gpu) = crate::gpu_tests::gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let view = ComposerGpu::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
        let mut compared = 0;
        let mut worst: f64 = 0.0;
        for path in sample_files() {
            let Ok(file) = SdfFile::load(&path) else { continue };
            for assembly in 0..file.assemblies.len() {
                let Some(field) = LayerField::for_assembly(&file, assembly) else { continue };
                if field.has_unsupported {
                    continue;
                }
                let scene = field.gpu_scene(None);
                let Some((lo, hi)) = scene.bounds else { continue };
                if scene.shape_count == 0 || !view.upload_for_test(&gpu.queue, &scene, assembly as u64 + 1) {
                    continue;
                }
                // A deterministic spread of points through the shapes' bounding box.
                let mut points = Vec::new();
                let mut state = 0x9E3779B97F4A7C15u64 ^ (assembly as u64 + 1);
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64
                };
                for _ in 0..2000 {
                    points.push([0, 1, 2].map(|k| (lo[k] + (hi[k] - lo[k]) * next()) as f32));
                }
                let gpu_d = view.evaluate_points(&gpu.device, &gpu.queue, &scene, &points);
                for (p, g) in points.iter().zip(gpu_d) {
                    let Some(c) = field.assembly_distance([p[0] as f64, p[1] as f64, p[2] as f64]) else { continue };
                    // Single precision: compare only where the surface is near, where the distance is meaningful.
                    if c.abs() > 1.0 {
                        continue;
                    }
                    let error = (c - g as f64).abs();
                    worst = worst.max(error);
                    assert!(error < 2e-3 + 2e-3 * c.abs(), "{} assembly {assembly}: cpu {c} gpu {g} at {p:?}", path.display());
                    compared += 1;
                }
            }
        }
        eprintln!("compared {compared} points, worst difference {worst}");
        assert!(compared > 0 || sample_files().is_empty());
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use crate::sdf_file::SdfFile;
    use crate::sdf_model::{file_scene, layer_fields};

    /// A whole file (every layer, with its placement, united) on the GPU matches the CPU's union of the layers.
    #[test]
    fn a_whole_file_on_the_gpu_matches_the_union_of_its_layers() {
        let Some(gpu) = crate::gpu_tests::gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let Ok(home) = std::env::var("USERPROFILE") else { return };
        let folder = std::path::Path::new(&home).join("Documents").join("SDF Modeler").join("sdfs");
        let view = ComposerGpu::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
        let mut compared = 0;
        for name in ["robot", "mech", "headphones", "bricks", "fried_egg"] {
            let Ok(file) = SdfFile::load(&folder.join(format!("{name}.sdf"))) else { continue };
            let fields = layer_fields(&file);
            let scene = file_scene(&fields, None);
            let Some((lo, hi)) = scene.bounds else { continue };
            if !view.upload_for_test(&gpu.queue, &scene, 0) {
                continue;
            }
            let mut state = 0x2545F4914F6CDD1Du64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let points: Vec<[f32; 3]> = (0..3000).map(|_| [0, 1, 2].map(|k| (lo[k] + (hi[k] - lo[k]) * next()) as f32)).collect();
            let gpu_d = view.evaluate_points(&gpu.device, &gpu.queue, &scene, &points);
            for (p, g) in points.iter().zip(gpu_d) {
                let c = fields.iter().filter(|f| !f.hidden).filter_map(|f| f.distance([p[0] as f64, p[1] as f64, p[2] as f64])).fold(f64::INFINITY, f64::min);
                if !c.is_finite() || c.abs() > 1.0 {
                    continue;
                }
                assert!((c - g as f64).abs() < 3e-3 + 3e-3 * c.abs(), "{name}: cpu {c} gpu {g} at {p:?}");
                compared += 1;
            }
        }
        eprintln!("compared {compared} points across whole files");
    }
}

#[cfg(test)]
impl ComposerGpu {
    fn upload_for_test(&self, queue: &wgpu::Queue, scene: &GpuScene, _version: u64) -> bool {
        if scene.shape_count > MAX_SHAPES || scene.ops.len() > MAX_OPS {
            return false;
        }
        queue.write_buffer(&self.buffers.shapes, 0, bytemuck::cast_slice(&scene.shapes));
        queue.write_buffer(&self.buffers.ops, 0, bytemuck::cast_slice(&scene.ops));
        true
    }
}
