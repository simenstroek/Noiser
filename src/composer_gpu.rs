//! The SDF Composer's GPU viewport: the flattened shape tree (`sdf_model::GpuScene`) goes into two storage buffers and
//! `composer_view.wgsl` interprets it, so changing a value never rebuilds a pipeline. The picture is drawn straight into
//! egui's pass for the viewport's rectangle.

use crate::composer_render::{Camera, FIELD_OF_VIEW};
use crate::sdf_model::{GPU_SHAPE_STRIDE, GpuScene};
use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;

pub const MAX_SHAPES: usize = 1024;
pub const MAX_OPS: usize = 4096;
/// Points the test entry point can evaluate in one go.
pub const MAX_TEST_POINTS: usize = 8192;

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
}

impl Uniforms {
    pub fn new(camera: &Camera, aspect: f32, scene: &GpuScene, show_selection: bool) -> Uniforms {
        let (forward, right, up) = camera.basis();
        let eye = camera.eye();
        let f = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        let light = {
            let l = [
                -forward[0] * 0.6 - right[0] * 0.5 + up[0] * 0.7,
                -forward[1] * 0.6 - right[1] * 0.5 + up[1] * 0.7,
                -forward[2] * 0.6 - right[2] * 0.5 + up[2] * 0.7,
            ];
            let length = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt().max(1e-9);
            [l[0] / length, l[1] / length, l[2] / length]
        };
        Uniforms {
            eye: f(eye),
            forward: f(forward),
            right: f(right),
            up: f(up),
            light: f(light),
            params: [aspect, (FIELD_OF_VIEW * 0.5).tan() as f32, (camera.distance * 3.0 + 60.0) as f32, if show_selection { 1.0 } else { 0.0 }],
            counts: [scene.ops.len() as u32, scene.shape_count as u32, 0, 0],
        }
    }
}

pub struct ComposerGpu {
    pipeline: wgpu::RenderPipeline,
    #[cfg_attr(not(test), allow(dead_code))]
    evaluate: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
    shapes: wgpu::Buffer,
    ops: wgpu::Buffer,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) points: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// The scene version the buffers hold.
    uploaded: Option<u64>,
}

impl ComposerGpu {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> ComposerGpu {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composer view"),
            source: wgpu::ShaderSource::Wgsl(include_str!("composer_view.wgsl").into()),
        });
        let entry = |binding: u32, ty, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
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
        let uniforms = buffer("composer uniforms", std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM);
        let shapes = buffer("composer shapes", (MAX_SHAPES * GPU_SHAPE_STRIDE * 16) as u64, wgpu::BufferUsages::STORAGE);
        let ops = buffer("composer ops", (MAX_OPS * 16) as u64, wgpu::BufferUsages::STORAGE);
        let points = buffer("composer test points", (MAX_TEST_POINTS * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composer view"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shapes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ops.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: points.as_entire_binding() },
            ],
        });
        ComposerGpu { pipeline, evaluate, uniforms, shapes, ops, points, bind_group, uploaded: None }
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
            queue.write_buffer(&self.shapes, 0, bytemuck::cast_slice(&scene.shapes));
        }
        if !scene.ops.is_empty() {
            queue.write_buffer(&self.ops, 0, bytemuck::cast_slice(&scene.ops));
        }
        self.uploaded = Some(version);
        true
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
    }

    /// Distances of the uploaded scene at `points`, computed on the GPU. For tests.
    #[cfg(test)]
    pub fn evaluate_points(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &GpuScene, points: &[[f32; 3]]) -> Vec<f32> {
        assert!(points.len() <= MAX_TEST_POINTS);
        let mut uniforms = Uniforms::new(&Camera::default(), 1.0, scene, false);
        uniforms.counts[2] = points.len() as u32;
        self.write_uniforms(queue, &uniforms);
        let data: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        queue.write_buffer(&self.points, 0, bytemuck::cast_slice(&data));
        let size = (points.len() * 16) as u64;
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.evaluate);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(points.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.points, 0, &read, 0, size);
        queue.submit([encoder.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = read.get_mapped_range(..).unwrap();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view[..]).to_vec();
        out.iter().map(|v| v[3]).collect()
    }
}

/// What the viewport callback needs: the camera, and the scene when it changed.
pub struct ViewCallback {
    pub uniforms: Uniforms,
    pub scene: Arc<GpuScene>,
    pub version: u64,
    pub format: wgpu::TextureFormat,
}

impl egui_wgpu::CallbackTrait for ViewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<ComposerGpu>().is_none() {
            resources.insert(ComposerGpu::new(device, self.format));
        }
        let gpu: &mut ComposerGpu = resources.get_mut().unwrap();
        gpu.upload(queue, &self.scene, self.version);
        gpu.write_uniforms(queue, &self.uniforms);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(gpu) = resources.get::<ComposerGpu>() else { return };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
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
impl ComposerGpu {
    fn upload_for_test(&self, queue: &wgpu::Queue, scene: &GpuScene, _version: u64) -> bool {
        if scene.shape_count > MAX_SHAPES || scene.ops.len() > MAX_OPS {
            return false;
        }
        queue.write_buffer(&self.shapes, 0, bytemuck::cast_slice(&scene.shapes));
        queue.write_buffer(&self.ops, 0, bytemuck::cast_slice(&scene.ops));
        true
    }
}
