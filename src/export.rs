//! Texture export: evaluates the full graph (never caches or previews) on the GPU at the export
//! resolution, block by block, and writes 2D images, 3D slice grids or OpenVDB.
//!
//! Sample positions are texel centres, (i + 0.5) / N, so a texture repeats seamlessly.
//! Image row 0 is the top row and holds y = 0.5 / N (no flip).

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::Graph;
use eframe::wgpu;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// One slice through the tile, along the slice view's axis and position.
    Image2d,
    /// All N slices laid out in one image.
    SliceGrid,
    Vdb,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Output {
    Density,
    Normal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Png8,
    Png16,
    /// Raw signed values.
    Exr32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    Square,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Levels {
    /// Map the graph's guaranteed density range to 0–1 (consistent between exports).
    GraphRange,
    /// Stretch this export's own min/max to 0–1.
    DataMinMax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub kind: Kind,
    pub output: Output,
    pub res: u32,
    pub format: Format,
    pub layout: Layout,
    pub levels: Levels,
    pub seed: u32,
    /// 2D only: slice axis (0 = X, 1 = Y, 2 = Z) and position in the tile.
    pub axis: u32,
    pub pos: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            kind: Kind::Image2d,
            output: Output::Density,
            res: 1024,
            format: Format::Png16,
            layout: Layout::Square,
            levels: Levels::GraphRange,
            seed: 0,
            axis: 2,
            pos: 0.5,
        }
    }
}

impl Settings {
    fn channels(&self) -> u64 {
        match self.output {
            Output::Density => 1,
            Output::Normal => 3,
        }
    }

    /// (tiles per row, rows) of the slice grid.
    pub fn grid(&self) -> (u32, u32) {
        let n = self.res;
        match self.layout {
            Layout::Vertical => (1, n),
            Layout::Square => {
                let c = (n as f64).sqrt().ceil() as u32;
                (c, n.div_ceil(c))
            }
        }
    }

    /// Output image size in pixels (None for VDB).
    pub fn image_size(&self) -> Option<(u64, u64)> {
        let n = self.res as u64;
        match self.kind {
            Kind::Image2d => Some((n, n)),
            Kind::SliceGrid => {
                let (c, r) = self.grid();
                Some((c as u64 * n, r as u64 * n))
            }
            Kind::Vdb => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match (self.kind, self.format) {
            (Kind::Vdb, _) => "vdb",
            (_, Format::Exr32) => "exr",
            _ => "png",
        }
    }

    /// Rough output file size and peak memory, in bytes.
    pub fn estimate(&self) -> (u64, u64) {
        let n = self.res as u64;
        match self.image_size() {
            Some((w, h)) => {
                let bytes_per = match self.format {
                    Format::Png8 => 1,
                    Format::Png16 => 2,
                    Format::Exr32 => 4,
                };
                // Whole image as f32 while assembling, plus the encoded copy.
                (w * h * self.channels() * bytes_per, w * h * self.channels() * (4 + bytes_per))
            }
            // Dense float leaves plus node overhead; evaluated 128³ at a time.
            None => (n * n * n * 4 + (n * n * n / 512) * 65, 128 * 128 * 128 * 4 * 2),
        }
    }

    /// Reasons this export is unreasonable (shown as warnings, never blocking).
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        let (file, mem) = self.estimate();
        const GB: u64 = 1 << 30;
        if mem > 2 * GB {
            w.push(format!("needs about {:.1} GB of memory", mem as f64 / GB as f64));
        }
        if file > 2 * GB {
            w.push(format!("the file will be about {:.1} GB", file as f64 / GB as f64));
        }
        if let Some((iw, ih)) = self.image_size() {
            if iw.max(ih) > 16384 {
                w.push(format!("{iw}×{ih} px is larger than most viewers and engines accept (16384)"));
            }
        }
        if self.kind == Kind::Vdb && self.output == Output::Normal {
            w.push("VDB export writes density only".into());
        }
        w
    }
}

// ---- GPU evaluation ------------------------------------------------------------

/// Mirrors `X` in the export shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Block {
    n: u32,
    origin: [u32; 3],
    size: [u32; 3],
    normal: u32,
    two_d: u32,
    axis: u32,
    pos: f32,
    _pad: u32,
}

const EXPORT_WGSL: &str = "
struct X { n: u32, ox: u32, oy: u32, oz: u32, sx: u32, sy: u32, sz: u32, normal: u32, two_d: u32, axis: u32, pos: f32, _pad: u32 };
@group(0) @binding(4) var<uniform> x: X;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= x.sx || id.y >= x.sy || id.z >= x.sz) { return; }
    var p = (vec3f(vec3u(x.ox, x.oy, x.oz) + id) + 0.5) / f32(x.n);
    if (x.two_d == 1u) {
        switch x.axis {
            case 0u: { p = vec3f(x.pos, p.x, p.y); }
            case 1u: { p = vec3f(p.x, x.pos, p.y); }
            default: { p = vec3f(p.x, p.y, x.pos); }
        }
    }
    let i = (id.z * x.sy + id.y) * x.sx + id.x;
    if (x.normal == 1u) {
        let nrm = graph_normal(p);
        out[3u * i] = nrm.x; out[3u * i + 1u] = nrm.y; out[3u * i + 2u] = nrm.z;
    } else {
        out[i] = graph_density(p);
    }
}
";

/// Evaluates a graph's outputs on boxes of the N³ grid.
pub struct Evaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    blue: wgpu::Buffer,
}

impl Evaluator {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, graph: &Graph, seed: u32) -> Result<Self, String> {
        use wgpu::util::DeviceExt;
        let c = codegen::compile(graph)?;
        let src = format!("{NOISE_LIB}\n{}\n{EVAL_BINDINGS}\n{EXPORT_WGSL}", c.wgsl);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("export"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        // Explicit layout: an automatic one would drop bindings that this graph doesn't use.
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(3, ro),
                entry(4, wgpu::BufferBindingType::Uniform),
                entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("export"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let init = |contents: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bgl,
            uniforms: init(bytemuck::bytes_of(&[seed, 0, 0, 0]), wgpu::BufferUsages::UNIFORM),
            params: init(bytemuck::cast_slice(&c.params), wgpu::BufferUsages::STORAGE),
            blue: init(bytemuck::cast_slice(&crate::bluenoise::tables()), wgpu::BufferUsages::STORAGE),
        })
    }

    /// Values for the box `origin..origin+size` of the N³ grid, x fastest. 1 or 3 floats per voxel.
    /// With `two_d`, z is ignored and samples come from the slice `(axis, pos)`.
    pub fn eval(&self, n: u32, origin: [u32; 3], size: [u32; 3], normal: bool, two_d: Option<(u32, f32)>) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let d = &self.device;
        let (axis, pos) = two_d.unwrap_or((2, 0.0));
        let block = Block { n, origin, size, normal: normal as u32, two_d: two_d.is_some() as u32, axis, pos, _pad: 0 };
        let block = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&block), usage: wgpu::BufferUsages::UNIFORM });
        let count = size.iter().map(|&s| s as u64).product::<u64>() * if normal { 3 } else { 1 };
        let bytes = count * 4;
        let out = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
        }
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[entry(0, &self.uniforms), entry(1, &self.params), entry(3, &self.blue), entry(4, &block), entry(5, &out)],
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(4), size[1].div_ceil(4), size[2].div_ceil(4));
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        self.queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map export buffer"));
        d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = read.get_mapped_range(..).expect("mapped");
        bytemuck::cast_slice(&view).to_vec()
    }
}

// ---- running an export ------------------------------------------------------------

/// Progress in thousandths, shared with the UI.
pub type Progress = Arc<AtomicU32>;

/// Runs the export (call from a worker thread). Returns a short summary.
pub fn run(ev: &Evaluator, s: &Settings, range: (f32, f32), path: &Path, progress: &Progress) -> Result<String, String> {
    if s.res == 0 {
        return Err("resolution must be at least 1".into());
    }
    if s.kind == Kind::Vdb {
        let n = s.res;
        crate::vdb::write(path, n, |origin, size| ev.eval(n, origin, size, false, None), |f| progress.store((f * 1000.0) as u32, Ordering::Relaxed))
            .map_err(|e| e.to_string())?;
        return Ok(format!("Exported {n}³ VDB to {}", path.display()));
    }

    let n = s.res;
    let normal = s.output == Output::Normal;
    let ch = s.channels() as usize;
    let (w, h) = s.image_size().unwrap();
    let (w, h) = (w as usize, h as usize);
    let mut img = vec![0.0f32; w * h * ch];
    // Min/max of evaluated values only; unused grid tiles must not count.
    let mut data = (f32::MAX, f32::MIN);
    let mut track = |v: &[f32]| {
        for &x in v.iter().filter(|x| x.is_finite()) {
            data = (data.0.min(x), data.1.max(x));
        }
    };
    if s.kind == Kind::Image2d {
        let v = ev.eval(n, [0, 0, 0], [n, n, 1], normal, Some((s.axis, s.pos)));
        track(&v);
        img.copy_from_slice(&v);
    } else {
        let (cols, _) = s.grid();
        // As many slices per GPU round trip as fit in ~64 MB.
        let per = ((64usize << 20) / (n as usize * n as usize * ch * 4)).clamp(1, n as usize) as u32;
        let mut z0 = 0;
        while z0 < n {
            let k = per.min(n - z0);
            let v = ev.eval(n, [0, 0, z0], [n, n, k], normal, None);
            track(&v);
            for dz in 0..k {
                let z = z0 + dz;
                let (tx, ty) = ((z % cols) as usize, (z / cols) as usize);
                for y in 0..n as usize {
                    let src = ((dz as usize * n as usize + y) * n as usize) * ch;
                    let dst = ((ty * n as usize + y) * w + tx * n as usize) * ch;
                    img[dst..dst + n as usize * ch].copy_from_slice(&v[src..src + n as usize * ch]);
                }
            }
            z0 += k;
            progress.store(z0 * 900 / n, Ordering::Relaxed);
        }
    }

    // 8/16-bit hold 0–1: density by the chosen levels, normals as n × 0.5 + 0.5. EXR keeps raw values.
    if s.format != Format::Exr32 {
        let (lo, hi) = if normal {
            (-1.0, 1.0)
        } else if s.levels == Levels::GraphRange && range.0.is_finite() && range.1.is_finite() && range.1 > range.0 {
            range
        } else {
            data
        };
        let scale = 1.0 / (hi - lo).max(1e-12);
        for v in &mut img {
            *v = ((*v - lo) * scale).clamp(0.0, 1.0);
        }
    }
    progress.store(950, Ordering::Relaxed);
    write_image(path, s, &img, w as u32, h as u32).map_err(|e| e.to_string())?;
    progress.store(1000, Ordering::Relaxed);
    Ok(format!("Exported {w}×{h} to {}", path.display()))
}

fn write_image(path: &Path, s: &Settings, img: &[f32], w: u32, h: u32) -> Result<(), Box<dyn std::error::Error>> {
    use image::{ImageBuffer, Luma, Rgb};
    let rgb = s.output == Output::Normal;
    match s.format {
        Format::Png8 => {
            let px: Vec<u8> = img.iter().map(|v| (v * 255.0).round() as u8).collect();
            if rgb { ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, px).unwrap().save(path)? } else { ImageBuffer::<Luma<u8>, _>::from_raw(w, h, px).unwrap().save(path)? }
        }
        Format::Png16 => {
            let px: Vec<u16> = img.iter().map(|v| (v * 65535.0).round() as u16).collect();
            if rgb { ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, px).unwrap().save(path)? } else { ImageBuffer::<Luma<u16>, _>::from_raw(w, h, px).unwrap().save(path)? }
        }
        Format::Exr32 => {
            use exr::prelude::*;
            let channel = |name: &'static str, c: usize, stride: usize| AnyChannel::new(name, FlatSamples::F32(img.iter().skip(c).step_by(stride).copied().collect()));
            let channels: SmallVec<[AnyChannel<FlatSamples>; 4]> =
                if rgb { SmallVec::from_vec(vec![channel("R", 0, 3), channel("G", 1, 3), channel("B", 2, 3)]) } else { SmallVec::from_vec(vec![channel("Y", 0, 1)]) };
            let layer = Layer::new((w as usize, h as usize), LayerAttributes::named("noiser"), Encoding::FAST_LOSSLESS, AnyChannels::sort(channels));
            Image::from_layer(layer).write().to_file(path)?;
        }
    }
    Ok(())
}
