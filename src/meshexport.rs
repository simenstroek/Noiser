//! Exporting a displaced mesh. The graph's noise is subtracted from the distance field of a loaded
//! mesh on the GPU, one slab of layers at a time, and the zero level of that field is extracted as
//! a triangle mesh (surface.rs). It is the same field the 3D preview shows.

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::Graph;
use crate::mesh::{Normalization, V3};
use crate::surface::{RawMesh, surface_nets};
use eframe::wgpu;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Progress in thousandths, shared with the UI.
pub type Progress = crate::export::Progress;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Obj,
    Ply,
}

impl Format {
    pub fn extension(self) -> &'static str {
        match self {
            Format::Obj => "obj",
            Format::Ply => "ply",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Samples along the longest side of the exported region.
    pub res: u32,
    pub format: Format,
    /// Parts with less than this fraction of the largest part's surface area are dropped (0 keeps all).
    pub cull: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self { res: 256, format: Format::Obj, cull: 0.01 }
    }
}

/// How the preview displaces the shape, which the export repeats.
#[derive(Clone, Copy, Debug)]
pub struct Displace {
    pub amount: f32,
    pub tile_size: f32,
    /// The "shape size" scale of the mesh in the preview.
    pub prim_size: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Block {
    grid_min: [f32; 3],
    h: f32,
    mesh_origin: [f32; 3],
    mesh_voxel: f32,
    nx: u32,
    ny: u32,
    z0: u32,
    count: u32,
    amount: f32,
    tile_size: f32,
    prim_size: f32,
    _pad: f32,
}

const MESH_EVAL_WGSL: &str = "
struct X { grid_min: vec3f, h: f32, mesh_origin: vec3f, mesh_voxel: f32, nx: u32, ny: u32, z0: u32, count: u32, amount: f32, tile_size: f32, prim_size: f32, _pad: f32 };
@group(0) @binding(4) var<uniform> x: X;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var mesh_tex: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= x.nx || id.y >= x.ny || id.z >= x.count) { return; }
    let p = x.grid_min + vec3f(f32(id.x), f32(id.y), f32(x.z0 + id.z)) * x.h;
    let base = mesh_sdf_at(mesh_tex, p / x.prim_size, x.mesh_origin, x.mesh_voxel) * x.prim_size;
    out[(id.z * x.ny + id.y) * x.nx + id.x] = base - x.amount * graph_density(p / x.tile_size);
}
";

/// The sample grid of an export: sample (i, j, k) is at `min + (i, j, k) × h` in the preview's space.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub min: V3,
    pub h: f32,
    pub dims: [usize; 3],
}

impl Grid {
    /// A grid with `res` samples along the longest side of the mesh bounds `lo..hi` (normalized
    /// space), scaled by the shape size, and room for the surface to move `reach` either way.
    pub fn plan(lo: V3, hi: V3, d: Displace, reach: f32, res: u32) -> Grid {
        let (lo, hi) = (lo.map(|v| v * d.prim_size), hi.map(|v| v * d.prim_size));
        let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f32::max) + 2.0 * reach;
        let h = (longest / res.max(8) as f32).max(1e-6);
        let pad = reach + 3.0 * h;
        let min = lo.map(|v| v - pad);
        let dims = std::array::from_fn(|a| (((hi[a] - lo[a]) + 2.0 * pad) / h).ceil() as usize + 1);
        Grid { min, h, dims }
    }

    pub fn samples(&self) -> u64 {
        self.dims.iter().map(|&d| d as u64).product()
    }
}

/// Evaluates the displaced field on the GPU.
pub struct MeshEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    blue: wgpu::Buffer,
    mesh: wgpu::TextureView,
    mesh_origin: V3,
    mesh_voxel: f32,
}

impl MeshEvaluator {
    /// `mesh` is the baked distance grid (see `MeshSdf::upload`); `origin` and `voxel` describe it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, graph: &Graph, seed: u32, mesh: wgpu::TextureView, origin: V3, voxel: f32) -> Result<Self, String> {
        use wgpu::util::DeviceExt;
        let c = codegen::compile(graph)?;
        let src = format!("{NOISE_LIB}\n{}\n{EVAL_BINDINGS}\n{MESH_EVAL_WGSL}", c.wgsl);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mesh export"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                buffer(1, ro),
                buffer(3, ro),
                buffer(4, wgpu::BufferBindingType::Uniform),
                buffer(5, wgpu::BufferBindingType::Storage { read_only: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh export"),
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
            mesh,
            mesh_origin: origin,
            mesh_voxel: voxel,
        })
    }

    /// The field at layers `z0..z0 + count` of the grid: `nx * ny` values per layer, x fastest.
    pub fn layers(&self, grid: &Grid, d: Displace, z0: usize, count: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let dev = &self.device;
        let [nx, ny, _] = grid.dims;
        let block = Block {
            grid_min: grid.min,
            h: grid.h,
            mesh_origin: self.mesh_origin,
            mesh_voxel: self.mesh_voxel,
            nx: nx as u32,
            ny: ny as u32,
            z0: z0 as u32,
            count: count as u32,
            amount: d.amount,
            tile_size: d.tile_size,
            prim_size: d.prim_size,
            _pad: 0.0,
        };
        let block = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&block), usage: wgpu::BufferUsages::UNIFORM });
        let bytes = (nx * ny * count * 4) as u64;
        let out = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
        }
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                entry(0, &self.uniforms),
                entry(1, &self.params),
                entry(3, &self.blue),
                entry(4, &block),
                entry(5, &out),
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh) },
            ],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((nx as u32).div_ceil(4), (ny as u32).div_ceil(4), (count as u32).div_ceil(4));
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        self.queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map mesh export buffer"));
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = read.get_mapped_range(..).expect("mapped");
        bytemuck::cast_slice(&view).to_vec()
    }
}

/// One export: what to evaluate and how.
pub struct Job<'a> {
    pub ev: &'a MeshEvaluator,
    pub d: Displace,
    /// Bounds of the mesh in normalized space.
    pub lo: V3,
    pub hi: V3,
    /// How far the displacement can move the surface in or out.
    pub reach: f32,
    /// Turns normalized space back into the mesh's own units.
    pub norm: Normalization,
    pub settings: Settings,
}

/// Layers evaluated per GPU dispatch.
const SLAB: usize = 8;

impl Job<'_> {
    pub fn grid(&self) -> Grid {
        Grid::plan(self.lo, self.hi, self.d, self.reach, self.settings.res)
    }

    /// The displaced surface in the preview's space (before the shape size and normalization are undone).
    pub fn extract(&self, progress: &Progress) -> RawMesh {
        let grid = self.grid();
        let [nx, ny, nz] = grid.dims;
        let mut slab: (usize, Vec<f32>) = (usize::MAX, Vec::new());
        let mut mesh = surface_nets(nx, ny, nz, |k| {
            if slab.0 == usize::MAX || k < slab.0 || k >= slab.0 + SLAB {
                let count = SLAB.min(nz - k);
                slab = (k, self.ev.layers(&grid, self.d, k, count));
            }
            progress.store((k * 900 / nz) as u32, Ordering::Relaxed);
            let at = (k - slab.0) * nx * ny;
            slab.1[at..at + nx * ny].to_vec()
        });
        for p in &mut mesh.positions {
            *p = [grid.min[0] + p[0] * grid.h, grid.min[1] + p[1] * grid.h, grid.min[2] + p[2] * grid.h];
        }
        mesh
    }

    /// Extracts the surface, drops small parts, puts it back in the mesh's own units and writes it.
    pub fn run(&self, path: &Path, progress: &Progress) -> Result<String, String> {
        let mut mesh = self.extract(progress);
        if mesh.triangles.is_empty() {
            return Err("the displaced surface is empty (is the mesh outside the exported region?)".into());
        }
        let removed = mesh.remove_small_parts(self.settings.cull);
        for p in &mut mesh.positions {
            *p = self.norm.to_object(p.map(|v| v / self.d.prim_size));
        }
        let written = match self.settings.format {
            Format::Obj => mesh.write_obj(path),
            Format::Ply => mesh.write_ply(path),
        };
        written.map_err(|e| format!("could not write {}: {e}", path.display()))?;
        progress.store(1000, Ordering::Relaxed);
        let parts = if removed > 0 { format!(", {removed} small parts removed") } else { String::new() };
        Ok(format!("Exported {} vertices, {} triangles{parts} to {}", mesh.positions.len(), mesh.triangles.len(), path.display()))
    }
}

/// How a batch picks the seed for each mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SeedMode {
    /// Every mesh gets the global seed.
    #[default]
    Same,
    /// The global seed plus the mesh's position in the batch.
    Increment,
    /// A different pseudo-random seed per mesh, repeatable from the global seed.
    Random,
}

impl SeedMode {
    /// The seed for the `index`-th mesh of a batch started with global seed `base`.
    pub fn seed_for(self, base: u32, index: usize) -> u32 {
        match self {
            SeedMode::Same => base,
            SeedMode::Increment => base.wrapping_add(index as u32),
            SeedMode::Random => {
                // splitmix64 of (base, index); kept below 2^24 so it is exact as a float.
                let mut z = ((base as u64) << 32 | index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) & 0x00FF_FFFF) as u32
            }
        }
    }
}

/// How far the surface can move: the displacement's reach, limited to what a distance grid padded
/// by `grid_pad` covers (it keeps 0.05 of margin).
pub fn reach(graph: &Graph, amount: f32, grid_pad: f32) -> f32 {
    let push = crate::view::push_range(graph.density_range(), amount);
    push.0.abs().max(push.1.abs()).min((grid_pad - 0.05).max(0.0))
}

/// Everything but the mesh itself for taking an OBJ file through the whole pipeline.
pub struct FileJob<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub graph: &'a Graph,
    pub seed: u32,
    pub d: Displace,
    pub settings: Settings,
    /// The distance grid: samples along the longest side, and padding in normalized units.
    pub grid_res: u32,
    pub grid_pad: f32,
}

impl FileJob<'_> {
    /// Loads `input`, displaces it like the preview would and writes the result to `output`.
    pub fn run(&self, input: &Path, output: &Path, progress: &Progress) -> Result<String, String> {
        let (mesh, norm) = crate::mesh::Mesh::load_obj(input)?.normalized();
        let sdf = crate::meshsdf::MeshSdf::build(&mesh, self.grid_res, self.grid_pad, &Default::default());
        let (_texture, view) = sdf.upload(self.device, self.queue); // keep the texture alive for the whole job
        let ev = MeshEvaluator::new(self.device, self.queue, self.graph, self.seed, view, sdf.origin, sdf.voxel)?;
        let (lo, hi) = mesh.bounds();
        let job = Job { ev: &ev, d: self.d, lo, hi, reach: reach(self.graph, self.d.amount, self.grid_pad), norm, settings: self.settings.clone() };
        job.run(output, progress)
    }
}
