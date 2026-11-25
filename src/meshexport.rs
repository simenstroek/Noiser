//! Exporting a displaced mesh. The graph's noise is subtracted from the distance field of a loaded
//! mesh on the GPU, one slab of layers at a time, and the zero level of that field is extracted as
//! a triangle mesh (surface.rs). It is the same field the 3D preview shows.

use crate::codegen::{self, EVAL_BINDINGS, NOISE_LIB};
use crate::graph::Graph;
use crate::mesh::{Normalization, V3};
use crate::meshmask::PackedMask;
use crate::surface::{RawMesh, marching_tets, surface_nets};
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
    /// Remesh to about this many triangles (adaptive, see remesh.rs) instead of writing the uniform grid mesh.
    pub remesh: Option<usize>,
    /// Also generate UVs (OBJ only).
    pub uv: Option<crate::uvgen::UvParams>,
    /// Also bake maps from the field into PNG files next to the mesh (needs UVs, which are generated if not asked for).
    pub bake: Option<crate::bake::Settings>,
    /// Append the quality numbers of the result (see meshquality.rs) to the summary.
    pub report: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { res: 256, format: Format::Obj, cull: 0.01, remesh: None, uv: None, bake: None, report: false }
    }
}

/// How the preview displaces the shape, which the export repeats.
#[derive(Clone, Copy, Debug)]
pub struct Displace {
    pub amount: f32,
    pub tile_size: f32,
    /// The "shape size" scale of the mesh in the preview.
    pub prim_size: f32,
    /// Where the noise displaces more or less (see meshmask.rs).
    pub mask: PackedMask,
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
    mesh_lo: [f32; 3],
    mask_n: u32,
    mesh_hi: [f32; 3],
    /// Samples farther than this from the surface cannot cross zero, so the noise is skipped there.
    band: f32,
    mask: [[f32; 4]; 12],
}

// The displaced field at `p` (the preview's space); beyond x.band of the mesh the noise is skipped.
const CLASSIC_FIELD_WGSL: &str = "
fn field_at(p: vec3f) -> f32 {
    let base = mesh_sdf_at(mesh_tex, p / x.prim_size, x.mesh_origin, x.mesh_voxel) * x.prim_size;
    if (abs(base) > x.band) { return base; }
    let m = mesh_mask(mesh_tex, p / x.prim_size, x.mesh_origin, x.mesh_voxel, x.mesh_lo, x.mesh_hi, x.mask, x.mask_n);
    return base - x.amount * m * graph_density(p / x.tile_size);
}
";

const MESH_EVAL_WGSL: &str = "
struct X { grid_min: vec3f, h: f32, mesh_origin: vec3f, mesh_voxel: f32, nx: u32, ny: u32, z0: u32, count: u32, amount: f32, tile_size: f32, prim_size: f32, _pad: f32, mesh_lo: vec3f, mask_n: u32, mesh_hi: vec3f, band: f32, mask: array<vec4f, 12> };
@group(0) @binding(4) var<uniform> x: X;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var mesh_tex: texture_3d<f32>;
@group(0) @binding(7) var<storage, read> pts: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> samples: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= x.nx || id.y >= x.ny || id.z >= x.count) { return; }
    let p = x.grid_min + vec3f(f32(id.x), f32(id.y), f32(x.z0 + id.z)) * x.h;
    out[(id.z * x.ny + id.y) * x.nx + id.x] = field_at(p);
}

// Sphere tracing of the field for a picture (an orthographic camera): pts[0] is the first ray origin and pts[1] and
// pts[2] the world steps per pixel along the image's x and y; pts[3] the ray direction; pts[4] = (light dir, t_max);
// pts[5] = (step scale, hit distance, gradient step, 0); pts[6] = (box low corner, 1 to trace only inside the box) and
// pts[7] = (box high corner, 0). The picture is x.nx by x.ny in `samples`.
@compute @workgroup_size(8, 8)
fn march(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= x.nx || id.y >= x.ny) { return; }
    let o = pts[0].xyz + pts[1].xyz * f32(id.x) + pts[2].xyz * f32(id.y);
    let d = pts[3].xyz;
    let light = pts[4].xyz;
    var t_max = pts[4].w;
    let scale = pts[5].x;
    let hit_d = pts[5].y;
    let e = pts[5].z;
    var t = 0.0;
    var hit = false;
    // A terrain is a sheet, as its mesh is: a ray that starts inside the ground (it came in through the side or the
    // bottom of the box, under the surface) shows nothing. Drawing it would show the ground's body as walls along the
    // square and a floor underneath, or valleys seen through the near edge.
    var outside = true;
    if (pts[6].w > 0.5) {
        // Only where the ray crosses the box (slabs). WGSL leaves 1/0 undefined, so a zero component becomes a tiny one.
        let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
        let a = (pts[6].xyz - o) * inv;
        let b = (pts[7].xyz - o) * inv;
        let t_in = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
        let t_out = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
        t = max(t_in, 0.0);
        t_max = min(t_max, t_out);
        // Judged a hair inside the box, clear of the square's edge where the field jumps.
        outside = field_at(o + d * (t + hit_d)) >= hit_d;
    }
    for (var i = 0; i < 220; i = i + 1) {
        if (t > t_max) { break; }
        if (!outside) { break; }
        let v = field_at(o + d * t);
        if (v < hit_d) { hit = true; break; }
        t = t + max(v * scale, hit_d * 0.5);
        if (t > t_max) { break; }
    }
    var colour = vec4f(0.11, 0.11, 0.11, 0.0);
    if (hit) {
        let p = o + d * t;
        let g = vec3f(
            field_at(p + vec3f(e, 0.0, 0.0)) - field_at(p - vec3f(e, 0.0, 0.0)),
            field_at(p + vec3f(0.0, e, 0.0)) - field_at(p - vec3f(0.0, e, 0.0)),
            field_at(p + vec3f(0.0, 0.0, e)) - field_at(p - vec3f(0.0, 0.0, e)));
        let n = normalize(g + vec3f(1e-9));
        let k = 0.2 + 0.8 * max(dot(n, light), 0.0);
        colour = vec4f(0.804 * k, 0.765 * k, 0.686 * k, 1.0);
    }
    samples[id.y * x.nx + id.x] = colour;
}

// The distance and its gradient (central differences over x.h) at each of x.count points; with x.ny = 1 the
// distance only (six times cheaper).
@compute @workgroup_size(64)
fn points(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= x.count) { return; }
    let p = pts[id.x].xyz;
    if (x.ny == 1u) { samples[id.x] = vec4f(field_at(p), 0.0, 0.0, 0.0); return; }
    let e = x.h;
    let g = vec3f(
        field_at(p + vec3f(e, 0.0, 0.0)) - field_at(p - vec3f(e, 0.0, 0.0)),
        field_at(p + vec3f(0.0, e, 0.0)) - field_at(p - vec3f(0.0, e, 0.0)),
        field_at(p + vec3f(0.0, 0.0, e)) - field_at(p - vec3f(0.0, 0.0, e))) / (2.0 * e);
    samples[id.x] = vec4f(field_at(p), g);
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
    // Used by the remesher and the baker (see docs/game-mesh-plan.md); only tests call `points` so far.
    #[allow(dead_code)]
    points_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    points_bgl: wgpu::BindGroupLayout,
    march_pipeline: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
    params: wgpu::Buffer,
    blue: wgpu::Buffer,
    mesh: wgpu::TextureView,
    mesh_origin: V3,
    mesh_voxel: f32,
    /// The mesh's bounds in normalized space (the mask's position ramps run across them).
    mesh_bounds: (V3, V3),
    /// Mesh graph: the distance grids of the loaded meshes after the first (which is `mesh`), padded with dummies,
    /// and the (origin, voxel, centre, scale) of all four.
    extra: Vec<wgpu::TextureView>,
    info: wgpu::Buffer,
}

/// A mesh baked as a distance grid, as a mesh graph's Load Mesh node uses it.
pub struct GraphMesh {
    pub view: wgpu::TextureView,
    pub origin: V3,
    pub voxel: f32,
    /// Normalization of the mesh: positions of the field are in the mesh file's units, the grid in the normalized ones.
    pub centre: V3,
    pub scale: f32,
}

/// Most meshes a mesh graph can load.
pub const MAX_GRAPH_MESHES: usize = 4;

const GRAPH_FIELD_WGSL: &str = "
struct MeshInfo { origin: vec3f, voxel: f32, centre: vec3f, scale: f32 };
@group(0) @binding(9) var<uniform> meshes: array<MeshInfo, 4>;
@group(0) @binding(10) var mesh_tex1: texture_3d<f32>;
@group(0) @binding(11) var mesh_tex2: texture_3d<f32>;
@group(0) @binding(12) var mesh_tex3: texture_3d<f32>;
// A terrain (scale < 0): the texture holds heights in 2048 x 2048 tiles, one per layer, `centre.x` tiles per side, `origin.z`
// samples per side and `voxel` the distance between samples. The field is the height above the ground, made a distance
// to first order by the slope.
fn land_h(t: texture_3d<f32>, x: i32, y: i32, n: i32, tiles: i32) -> f32 {
    let cx = clamp(x, 0, n - 1);
    let cy = clamp(y, 0, n - 1);
    return textureLoad(t, vec3i(cx & 2047, cy & 2047, (cy >> 11) * tiles + (cx >> 11)), 0).x;
}
fn land_dist(t: texture_3d<f32>, p: vec3f, m: MeshInfo) -> f32 {
    let n = i32(m.origin.z);
    let tiles = i32(m.centre.x);
    let g = clamp((p.xy - m.origin.xy) / m.voxel, vec2f(0.0), vec2f(f32(n - 1)));
    let i = vec2i(floor(g));
    let f = g - vec2f(i);
    let h00 = land_h(t, i.x, i.y, n, tiles);
    let h10 = land_h(t, i.x + 1, i.y, n, tiles);
    let h01 = land_h(t, i.x, i.y + 1, n, tiles);
    let h11 = land_h(t, i.x + 1, i.y + 1, n, tiles);
    let h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    let dx = mix(h10 - h00, h11 - h01, f.y) / m.voxel;
    let dy = mix(h01 - h00, h11 - h10, f.x) / m.voxel;
    // The terrain ends at its square: the ground below the height, cut by the signed distance to the square (negative
    // inside), so the field is empty outside it. The square is a hair larger than the terrain so that lattice points on
    // its edge are inside: the surface then simply ends at the border (no wall, no ragged edge).
    let middle = 0.5 * f32(n - 1) * m.voxel;
    let half = middle + 0.5 * m.voxel;
    let q = abs(p.xy - (m.origin.xy + vec2f(middle))) - vec2f(half);
    let square = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
    // The square counts 1024 times over, and reaches half a sample past the terrain, so the lattice points on the border are
    // far enough inside that noise added to the field cannot flip them: without that, noise of half a metre stood a ragged
    // wall of surface along the whole border, as deep as the lattice goes.
    return max((p.z - h) * inverseSqrt(1.0 + dx * dx + dy * dy), 1024.0 * square);
}
// The ground under (x, y) for scattering: height and the slope along x and along y; and the terrain's lowest and highest
// height (centre.y and centre.z).
fn land_sample(t: texture_3d<f32>, xy: vec2f, m: MeshInfo) -> vec3f {
    let n = i32(m.origin.z);
    let tiles = i32(m.centre.x);
    let g = clamp((xy - m.origin.xy) / m.voxel, vec2f(0.0), vec2f(f32(n - 1)));
    let i = vec2i(floor(g));
    let f = g - vec2f(i);
    let h00 = land_h(t, i.x, i.y, n, tiles);
    let h10 = land_h(t, i.x + 1, i.y, n, tiles);
    let h01 = land_h(t, i.x, i.y + 1, n, tiles);
    let h11 = land_h(t, i.x + 1, i.y + 1, n, tiles);
    return vec3f(mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y), mix(h10 - h00, h11 - h01, f.y) / m.voxel, mix(h01 - h00, h11 - h10, f.x) / m.voxel);
}
fn land_ground_0(xy: vec2f) -> vec3f { return land_sample(mesh_tex, xy, meshes[0]); }
fn land_ground_1(xy: vec2f) -> vec3f { return land_sample(mesh_tex1, xy, meshes[1]); }
fn land_ground_2(xy: vec2f) -> vec3f { return land_sample(mesh_tex2, xy, meshes[2]); }
fn land_ground_3(xy: vec2f) -> vec3f { return land_sample(mesh_tex3, xy, meshes[3]); }
fn land_centre_0() -> vec2f { let m = meshes[0]; return m.origin.xy + vec2f(0.5 * (m.origin.z - 1.0) * m.voxel); }
fn land_centre_1() -> vec2f { let m = meshes[1]; return m.origin.xy + vec2f(0.5 * (m.origin.z - 1.0) * m.voxel); }
fn land_centre_2() -> vec2f { let m = meshes[2]; return m.origin.xy + vec2f(0.5 * (m.origin.z - 1.0) * m.voxel); }
fn land_centre_3() -> vec2f { let m = meshes[3]; return m.origin.xy + vec2f(0.5 * (m.origin.z - 1.0) * m.voxel); }
// True when (x, y) is at least `margin` inside the terrain's square (scattered copies stand only on the terrain).
fn land_inside(xy: vec2f, margin: f32, m: MeshInfo) -> bool {
    let side = (m.origin.z - 1.0) * m.voxel;
    let q = xy - m.origin.xy;
    return all(q >= vec2f(margin)) && all(q <= vec2f(side - margin));
}
fn land_inside_0(xy: vec2f, margin: f32) -> bool { return land_inside(xy, margin, meshes[0]); }
fn land_inside_1(xy: vec2f, margin: f32) -> bool { return land_inside(xy, margin, meshes[1]); }
fn land_inside_2(xy: vec2f, margin: f32) -> bool { return land_inside(xy, margin, meshes[2]); }
fn land_inside_3(xy: vec2f, margin: f32) -> bool { return land_inside(xy, margin, meshes[3]); }
fn land_range_0() -> vec2f { return meshes[0].centre.yz; }
fn land_range_1() -> vec2f { return meshes[1].centre.yz; }
fn land_range_2() -> vec2f { return meshes[2].centre.yz; }
fn land_range_3() -> vec2f { return meshes[3].centre.yz; }
fn mesh_dist_0(p: vec3f) -> f32 { let m = meshes[0]; if (m.scale < 0.0) { return land_dist(mesh_tex, p, m); } return mesh_sdf_at(mesh_tex, (p - m.centre) / m.scale, m.origin, m.voxel) * m.scale; }
fn mesh_dist_1(p: vec3f) -> f32 { let m = meshes[1]; if (m.scale < 0.0) { return land_dist(mesh_tex1, p, m); } return mesh_sdf_at(mesh_tex1, (p - m.centre) / m.scale, m.origin, m.voxel) * m.scale; }
fn mesh_dist_2(p: vec3f) -> f32 { let m = meshes[2]; if (m.scale < 0.0) { return land_dist(mesh_tex2, p, m); } return mesh_sdf_at(mesh_tex2, (p - m.centre) / m.scale, m.origin, m.voxel) * m.scale; }
fn mesh_dist_3(p: vec3f) -> f32 { let m = meshes[3]; if (m.scale < 0.0) { return land_dist(mesh_tex3, p, m); } return mesh_sdf_at(mesh_tex3, (p - m.centre) / m.scale, m.origin, m.voxel) * m.scale; }
fn field_at(p: vec3f) -> f32 { return graph_density(p); }
";

impl MeshEvaluator {
    /// `mesh` is the baked distance grid (see `MeshSdf::upload`); `origin` and `voxel` describe it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, graph: &Graph, seed: u32, mesh: wgpu::TextureView, origin: V3, voxel: f32, bounds: (V3, V3), mask: bool) -> Result<Self, String> {
        let c = codegen::compile(graph)?;
        // The kernels come after the field function, which the classic displaced-mesh field defines here.
        let src = format!("{NOISE_LIB}\n{}\n{}\n{EVAL_BINDINGS}\n{CLASSIC_FIELD_WGSL}\n{MESH_EVAL_WGSL}", codegen::mesh_mask_lib(mask), c.wgsl);
        let info = [[origin[0], origin[1], origin[2], voxel, 0.0, 0.0, 0.0, 1.0]; MAX_GRAPH_MESHES];
        Ok(Self::build(device, queue, &src, &c.params, seed, mesh, origin, voxel, bounds, Vec::new(), info))
    }

    /// An evaluator for a mesh graph's field: `graph` is the lowered graph (see `meshgraph::lowered_mesh`) with the
    /// output node whose source is the field to evaluate; the loaded-mesh nodes read `meshes` in the order of
    /// `Compiled::mesh_slots`. The field is in the units of the mesh files.
    pub fn new_graph(device: &wgpu::Device, queue: &wgpu::Queue, graph: &Graph, seed: u32, mut meshes: Vec<GraphMesh>) -> Result<Self, String> {
        // The Normal node steps over half a voxel of the finest loaded mesh, in the units of the mesh files.
        let eps = meshes.iter().map(|m| 0.5 * m.voxel * m.scale.abs()).fold(f32::MAX, f32::min);
        let eps = if eps.is_finite() && eps > 0.0 && eps < f32::MAX { eps } else { codegen::NORMAL_EPS };
        let c = codegen::compile_eps(graph, &Default::default(), eps)?;
        if meshes.len() > MAX_GRAPH_MESHES {
            return Err(format!("a mesh graph can load at most {MAX_GRAPH_MESHES} meshes"));
        }
        let src = format!("{NOISE_LIB}\n{}\n{}\n{EVAL_BINDINGS}\n{GRAPH_FIELD_WGSL}\n{MESH_EVAL_WGSL}", codegen::mesh_mask_lib(false), c.wgsl);
        let dummy = |device: &wgpu::Device, queue: &wgpu::Queue| {
            let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
            let t = device.create_texture(&wgpu::TextureDescriptor { label: Some("unused mesh"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3, format: wgpu::TextureFormat::R32Float, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] });
            queue.write_texture(wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, bytemuck::bytes_of(&1000.0f32), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) }, size);
            t.create_view(&Default::default())
        };
        while meshes.len() < MAX_GRAPH_MESHES {
            meshes.push(GraphMesh { view: dummy(device, queue), origin: [0.0; 3], voxel: 1.0, centre: [0.0; 3], scale: 1.0 });
        }
        let info: [[f32; 8]; MAX_GRAPH_MESHES] = std::array::from_fn(|i| {
            let m = &meshes[i];
            [m.origin[0], m.origin[1], m.origin[2], m.voxel, m.centre[0], m.centre[1], m.centre[2], m.scale]
        });
        let mut views = meshes.into_iter().map(|m| (m.view, m.origin, m.voxel));
        let (first, origin, voxel) = views.next().unwrap();
        let extra = views.map(|(v, _, _)| v).collect();
        Ok(Self::build(device, queue, &src, &c.params, seed, first, origin, voxel, ([0.0; 3], [0.0; 3]), extra, info))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(device: &wgpu::Device, queue: &wgpu::Queue, src: &str, params: &[f32], seed: u32, mesh: wgpu::TextureView, origin: V3, voxel: f32, bounds: (V3, V3), mut extra: Vec<wgpu::TextureView>, info: [[f32; 8]; MAX_GRAPH_MESHES]) -> Self {
        use wgpu::util::DeviceExt;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mesh export"), source: wgpu::ShaderSource::Wgsl(src.to_string().into()) });
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                buffer(1, ro),
                buffer(3, ro),
                buffer(4, wgpu::BufferBindingType::Uniform),
                buffer(5, wgpu::BufferBindingType::Storage { read_only: false }),
                texture(6),
                buffer(9, wgpu::BufferBindingType::Uniform),
                texture(10),
                texture(11),
                texture(12),
            ],
        });
        let points_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                buffer(1, ro),
                buffer(3, ro),
                buffer(4, wgpu::BufferBindingType::Uniform),
                texture(6),
                buffer(7, ro),
                buffer(8, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer(9, wgpu::BufferBindingType::Uniform),
                texture(10),
                texture(11),
                texture(12),
            ],
        });
        let points_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&points_bgl)], immediate_size: 0 });
        let points_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh points"),
            layout: Some(&points_layout),
            module: &module,
            entry_point: Some("points"),
            compilation_options: Default::default(),
            cache: None,
        });
        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("field march"),
            layout: Some(&points_layout),
            module: &module,
            entry_point: Some("march"),
            compilation_options: Default::default(),
            cache: None,
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
        while extra.len() < MAX_GRAPH_MESHES - 1 {
            extra.push(mesh.clone());
        }
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bgl,
            points_pipeline,
            points_bgl,
            march_pipeline,
            uniforms: init(bytemuck::bytes_of(&[seed, 0, 0, 0]), wgpu::BufferUsages::UNIFORM),
            params: init(bytemuck::cast_slice(params), wgpu::BufferUsages::STORAGE),
            blue: init(bytemuck::cast_slice(&crate::bluenoise::tables()), wgpu::BufferUsages::STORAGE),
            mesh,
            mesh_origin: origin,
            mesh_voxel: voxel,
            mesh_bounds: bounds,
            extra,
            info: init(bytemuck::cast_slice(&info), wgpu::BufferUsages::UNIFORM),
        }
    }

    /// The field at layers `z0..z0 + count` of the grid: `nx * ny` values per layer, x fastest.
    pub fn layers(&self, grid: &Grid, d: Displace, band: f32, z0: usize, count: usize) -> Vec<f32> {
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
            mesh_lo: self.mesh_bounds.0,
            mask_n: d.mask.count,
            mesh_hi: self.mesh_bounds.1,
            band,
            mask: std::array::from_fn(|i| d.mask.terms[i]),
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
                entry(9, &self.info),
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.extra[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.extra[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.extra[2]) },
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

    /// The displaced field at arbitrary points of the preview's space: (distance, gradient x, y, z) for each,
    /// the gradient by central differences over `eps`. The noise is never skipped, so any point is exact
    /// (unlike `layers`, which trusts the band). Done in chunks so any number of points works.
    pub fn points(&self, d: Displace, points: &[V3], eps: f32) -> Vec<[f32; 4]> {
        self.sample_points(d, points, eps, true)
    }

    /// The displaced field's distance alone at each point (see `points`).
    pub fn distances(&self, d: Displace, points: &[V3]) -> Vec<f32> {
        self.sample_points(d, points, 1.0, false).iter().map(|s| s[0]).collect()
    }

    /// A picture of the field's zero surface seen by an orthographic camera (`cam`), `w` by `h` pixels as RGBA in 0 to 1
    /// (alpha 0 where nothing was hit).
    pub fn march(&self, d: Displace, cam: &MarchCamera, w: u32, h: u32) -> Vec<[f32; 4]> {
        use wgpu::util::DeviceExt;
        let dev = &self.device;
        let block = Block {
            grid_min: [0.0; 3],
            h: cam.hit,
            mesh_origin: self.mesh_origin,
            mesh_voxel: self.mesh_voxel,
            nx: w,
            ny: h,
            z0: 0,
            count: w * h,
            amount: d.amount,
            tile_size: d.tile_size,
            prim_size: d.prim_size,
            _pad: 0.0,
            mesh_lo: self.mesh_bounds.0,
            mask_n: d.mask.count,
            mesh_hi: self.mesh_bounds.1,
            band: f32::MAX,
            mask: std::array::from_fn(|i| d.mask.terms[i]),
        };
        let block = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&block), usage: wgpu::BufferUsages::UNIFORM });
        let v4 = |v: V3, w: f32| [v[0], v[1], v[2], w];
        let (clip_lo, clip_hi, clipped) = match cam.clip {
            Some((lo, hi)) => (lo, hi, 1.0),
            None => ([0.0; 3], [0.0; 3], 0.0),
        };
        let input = [v4(cam.origin, 0.0), v4(cam.step_x, 0.0), v4(cam.step_y, 0.0), v4(cam.direction, 0.0), v4(cam.light, cam.t_max), [0.8, cam.hit, cam.gradient_step, 0.0], v4(clip_lo, clipped), v4(clip_hi, 0.0)];
        let input = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
        let bytes = (w as u64) * (h as u64) * 16;
        let out = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
        }
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.points_bgl,
            entries: &[
                entry(0, &self.uniforms),
                entry(1, &self.params),
                entry(3, &self.blue),
                entry(4, &block),
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh) },
                entry(7, &input),
                entry(8, &out),
                entry(9, &self.info),
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.extra[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.extra[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.extra[2]) },
            ],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.march_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        self.queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map march buffer"));
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = read.get_mapped_range(..).expect("mapped");
        bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec()
    }

    fn sample_points(&self, d: Displace, points: &[V3], eps: f32, gradient: bool) -> Vec<[f32; 4]> {
        use wgpu::util::DeviceExt;
        const CHUNK: usize = 1 << 20;
        let dev = &self.device;
        let mut all = Vec::with_capacity(points.len());
        for chunk in points.chunks(CHUNK) {
            let block = Block {
                grid_min: [0.0; 3],
                h: eps,
                mesh_origin: self.mesh_origin,
                mesh_voxel: self.mesh_voxel,
                nx: 0,
                ny: u32::from(!gradient),
                z0: 0,
                count: chunk.len() as u32,
                amount: d.amount,
                tile_size: d.tile_size,
                prim_size: d.prim_size,
                _pad: 0.0,
                mesh_lo: self.mesh_bounds.0,
                mask_n: d.mask.count,
                mesh_hi: self.mesh_bounds.1,
                band: f32::MAX,
                mask: std::array::from_fn(|i| d.mask.terms[i]),
            };
            let block = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&block), usage: wgpu::BufferUsages::UNIFORM });
            let input: Vec<[f32; 4]> = chunk.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
            let input = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&input), usage: wgpu::BufferUsages::STORAGE });
            let bytes = (chunk.len() * 16) as u64;
            let out = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
            }
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.points_bgl,
                entries: &[
                    entry(0, &self.uniforms),
                    entry(1, &self.params),
                    entry(3, &self.blue),
                    entry(4, &block),
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh) },
                    entry(7, &input),
                    entry(8, &out),
                    entry(9, &self.info),
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.extra[0]) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.extra[1]) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.extra[2]) },
                ],
            });
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.points_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((chunk.len() as u32).div_ceil(64), 1, 1);
            }
            enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
            self.queue.submit([enc.finish()]);
            read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map point samples"));
            dev.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            let view = read.get_mapped_range(..).expect("mapped");
            all.extend_from_slice(bytemuck::cast_slice::<u8, [f32; 4]>(&view));
        }
        all
    }
}

/// An orthographic camera for `MeshEvaluator::march`: the first pixel's ray origin, the world step per pixel along the
/// picture's x and y, the ray direction, the light direction, how far to trace, and the hit and gradient distances.
pub struct MarchCamera {
    pub origin: V3,
    pub step_x: V3,
    pub step_y: V3,
    pub direction: V3,
    pub light: V3,
    pub t_max: f32,
    pub hit: f32,
    pub gradient_step: f32,
    /// The box the surface is inside, if known: each ray is traced only where it crosses it. A terrain's field outside
    /// its square is many times the true distance (see `land_dist`), so a ray tracing in from beside the square would
    /// jump straight through it.
    pub clip: Option<(V3, V3)>,
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

/// One terrain chunk cut from a lattice every chunk shares: sample (i, j, k) is at `(x0 + (i - 1) h, y0 + (j - 1) h,
/// z_origin + (k0 + k) h)` for the chunk's `x0`, `y0`, `h = size / q`.
pub struct ChunkWindow {
    pub x0: f32,
    pub y0: f32,
    pub size: f32,
    /// Quads along each side of the chunk.
    pub q: usize,
    /// The lattice's lowest z, the same for every chunk, and the z range this chunk needs.
    pub z_origin: f32,
    pub z: (f32, f32),
}

impl MeshEvaluator {
    /// The surface of the field inside the chunk's window as surface nets on the shared lattice: the same vertices
    /// along a border as the neighbour's, normals from the field's gradient (so lighting has no seam), no remeshing.
    /// Coordinates are the field's own. About `2 q^2` triangles for a terrain.
    pub fn chunk_surface(&self, d: Displace, w: &ChunkWindow, progress: &Progress) -> RawMesh {
        let h = w.size / w.q as f32;
        let k0 = ((w.z.0 - w.z_origin) / h).floor().max(0.0) as usize;
        let k1 = (((w.z.1 - w.z_origin) / h).ceil() as usize).max(k0 + 2);
        let (nx, nz) = (w.q + 2, k1 - k0 + 1);
        let grid = Grid { min: [w.x0 - h, w.y0 - h, w.z_origin + k0 as f32 * h], h, dims: [nx, nx, nz] };
        let mut slab: (usize, Vec<f32>) = (usize::MAX, Vec::new());
        let layer = |k: usize| {
            if slab.0 == usize::MAX || k < slab.0 || k >= slab.0 + SLAB {
                let count = SLAB.min(nz - k);
                slab = (k, self.layers(&grid, d, f32::MAX, k, count));
            }
            progress.store((k * 1000 / nz) as u32, Ordering::Relaxed);
            let at = (k - slab.0) * nx * nx;
            slab.1[at..at + nx * nx].to_vec()
        };
        let mut mesh = crate::surface::surface_nets_in(nx, nx, nz, layer, Some((1, w.q + 1, 1, w.q + 1)));
        for p in &mut mesh.positions {
            *p = [grid.min[0] + p[0] * h, grid.min[1] + p[1] * h, grid.min[2] + p[2] * h];
        }
        mesh
    }
}

impl Job<'_> {
    pub fn grid(&self) -> Grid {
        Grid::plan(self.lo, self.hi, self.d, self.reach, self.settings.res)
    }

    /// The displaced surface in the preview's space (before the shape size and normalization are undone).
    pub fn extract(&self, progress: &Progress) -> RawMesh {
        self.extract_with(progress, false)
    }

    /// `extract`, by marching tetrahedra when `manifold` (a closed manifold surface, with poor triangles).
    fn extract_with(&self, progress: &Progress, manifold: bool) -> RawMesh {
        let grid = self.grid();
        let [nx, ny, nz] = grid.dims;
        let band = self.reach + 3.0 * grid.h;
        let mut slab: (usize, Vec<f32>) = (usize::MAX, Vec::new());
        let mut layer = |k: usize| {
            if slab.0 == usize::MAX || k < slab.0 || k >= slab.0 + SLAB {
                let count = SLAB.min(nz - k);
                slab = (k, self.ev.layers(&grid, self.d, band, k, count));
            }
            progress.store((k * 900 / nz) as u32, Ordering::Relaxed);
            let at = (k - slab.0) * nx * ny;
            slab.1[at..at + nx * ny].to_vec()
        };
        let mut mesh = if manifold { marching_tets(nx, ny, nz, &mut layer) } else { surface_nets(nx, ny, nz, &mut layer) };
        for p in &mut mesh.positions {
            *p = [grid.min[0] + p[0] * grid.h, grid.min[1] + p[1] * grid.h, grid.min[2] + p[2] * grid.h];
        }
        mesh
    }

    /// This job with another extraction resolution.
    fn with_res(&self, res: u32) -> Job<'_> {
        Job { ev: self.ev, d: self.d, lo: self.lo, hi: self.hi, reach: self.reach, norm: self.norm, settings: Settings { res, ..self.settings.clone() } }
    }

    /// The displaced surface remeshed to about `target` triangles, in the preview's space: a coarse uniform
    /// extraction (fine enough to keep the shape's topology), then adaptive remeshing onto the exact field.
    pub fn remeshed(&self, target: usize, progress: &Progress) -> RawMesh {
        self.remeshed_with(target, 1.0, progress)
    }

    pub fn remeshed_with(&self, target: usize, adaptivity: f32, progress: &Progress) -> RawMesh {
        let start_res = ((target as f32 / 8.0).sqrt() * 1.5).clamp(64.0, 192.0) as u32;
        let coarse = self.with_res(start_res);
        let mut mesh = coarse.extract_with(progress, true);
        mesh.remove_small_parts(self.settings.cull);
        if std::env::var_os("NOISER_REMESH_DEBUG").is_some() {
            eprintln!("start mesh:
{}", crate::meshquality::report(&mesh));
        }
        let grid = coarse.grid();
        let diagonal = (0..3).map(|a| (grid.dims[a] as f32 * grid.h).powi(2)).sum::<f32>().sqrt();
        let mut params = crate::remesh::Params::new(target, diagonal);
        params.adaptivity = adaptivity;
        let eps = 0.5 * params.min_edge;
        let d = self.d;
        // The field averaged over a seven point stencil of the requested radius (distance and gradient alike).
        let mut field = |pts: &[V3], radii: &[f32]| -> Vec<[f32; 4]> {
            if radii.iter().all(|&r| r <= 0.0) {
                return self.ev.points(d, pts, eps);
            }
            let mut stencil = Vec::with_capacity(pts.len() * 7);
            for (p, &r) in pts.iter().zip(radii) {
                stencil.push(*p);
                for axis in 0..3 {
                    for sign in [-1.0, 1.0] {
                        let mut q = *p;
                        q[axis] += sign * r;
                        stencil.push(q);
                    }
                }
            }
            self.ev.points(d, &stencil, eps).chunks(7).map(|c| std::array::from_fn(|i| c.iter().map(|s| s[i]).sum::<f32>() / 7.0)).collect()
        };
        crate::remesh::remesh(mesh, &mut field, &params, &|round, rounds| progress.store(900 + (round * 100 / rounds.max(1)) as u32, Ordering::Relaxed))
    }

    /// Extracts the surface, drops small parts, puts it back in the mesh's own units and writes it.
    pub fn run(&self, path: &Path, progress: &Progress) -> Result<String, String> {
        let mut mesh = match self.settings.remesh {
            Some(target) => self.remeshed(target, progress),
            None => self.extract(progress),
        };
        if mesh.triangles.is_empty() {
            return Err("the displaced surface is empty (is the mesh outside the exported region?)".into());
        }
        let removed = if self.settings.remesh.is_some() { 0 } else { mesh.remove_small_parts(self.settings.cull) };
        if self.settings.remesh.is_some() {
            mesh.recompute_normals();
        }
        let mut quality = String::new();
        if self.settings.report {
            let (lo, hi) = mesh.positions.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| ([0, 1, 2].map(|a| lo[a].min(p[a])), [0, 1, 2].map(|a| hi[a].max(p[a]))));
            let diagonal = (0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f32>().sqrt();
            let (d, eps) = (self.d, 0.0005 * diagonal);
            let mut r = crate::meshquality::report(&mesh).with_surface_error(&mesh, &mut |pts| self.ev.points(d, pts, eps)).with_intersections(&mesh);
            // Distances in the mesh's own units, like the file.
            let to_object = self.norm.scale / self.d.prim_size;
            (r.mean_error, r.max_error) = (r.mean_error.map(|v| v * to_object), r.max_error.map(|v| v * to_object));
            quality = format!("
{r}");
        }
        let wants_uv = self.settings.uv.is_some() || self.settings.bake.is_some();
        if wants_uv && self.settings.format != Format::Obj {
            return Err("UVs and maps are written with OBJ meshes only; choose the OBJ format".into());
        }
        // UVs and maps are made in the preview's space, where the field is; only then is the mesh put back in the
        // mesh's own units.
        let mut uv = None;
        let mut baked = String::new();
        if wants_uv {
            let params = self.settings.uv.clone().unwrap_or_default();
            let mut u = crate::uvgen::generate(&mesh, &params)?;
            if self.settings.report {
                quality += &format!("
{}", crate::meshquality::uv_report(&u, 1024));
            }
            if let Some(bake) = &self.settings.bake {
                let (lo, hi) = u.positions.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| ([0, 1, 2].map(|a| lo[a].min(p[a])), [0, 1, 2].map(|a| hi[a].max(p[a]))));
                let diagonal = (0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f32>().sqrt();
                let mut field = EvalField { ev: self.ev, d: self.d, eps: 0.0005 * diagonal };
                let images = crate::bake::bake(&u, &mut field, &mut [], bake, &|done, total| progress.store(900 + (done * 100 / total.max(1)) as u32, Ordering::Relaxed))?;
                let stem = path.file_stem().map_or("mesh".into(), |s| s.to_string_lossy().into_owned());
                let dir = path.parent().unwrap_or(Path::new("."));
                for image in &images {
                    let file = dir.join(format!("{stem}_{}.png", image.label));
                    image.write_png(&file)?;
                }
                baked = format!(", {} map(s) of {} texels", images.len(), bake.size);
            }
            for p in &mut u.positions {
                *p = self.norm.to_object(p.map(|v| v / self.d.prim_size));
            }
            uv = Some(u);
        }
        for p in &mut mesh.positions {
            *p = self.norm.to_object(p.map(|v| v / self.d.prim_size));
        }
        let written = match (&uv, self.settings.format) {
            (Some(u), _) => u.write_obj(path),
            (None, Format::Obj) => mesh.write_obj(path),
            (None, Format::Ply) => mesh.write_ply(path),
        };
        written.map_err(|e| format!("could not write {}: {e}", path.display()))?;
        progress.store(1000, Ordering::Relaxed);
        let parts = if removed > 0 { format!(", {removed} small parts removed") } else { String::new() };
        let (vertices, triangles) = uv.as_ref().map_or((mesh.positions.len(), mesh.triangles.len()), |u| (u.positions.len(), u.triangles.len()));
        Ok(format!("Exported {vertices} vertices, {triangles} triangles{parts}{baked} to {}{quality}", path.display()))
    }
}

/// The displaced field as a bake source.
pub struct EvalField<'a> {
    ev: &'a MeshEvaluator,
    d: Displace,
    eps: f32,
}

impl<'a> EvalField<'a> {
    pub fn new(ev: &'a MeshEvaluator, d: Displace, eps: f32) -> Self {
        EvalField { ev, d, eps }
    }
}

impl crate::bake::Field for EvalField<'_> {
    fn gradient(&mut self, points: &[V3]) -> Vec<[f32; 4]> {
        self.ev.points(self.d, points, self.eps)
    }
    fn distance(&mut self, points: &[V3]) -> Vec<f32> {
        self.ev.distances(self.d, points)
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

/// A repeatable per-mesh factor for the displacement amount: 1 plus or minus up to `spread` (0.2 is 20 %),
/// from the global seed and the mesh's index. 1 exactly when `spread` is 0.
pub fn amount_factor(base: u32, index: usize, spread: f32) -> f32 {
    let mut z = ((base as u64) << 32 | index as u64).wrapping_add(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let unit = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
    1.0 + spread * (unit as f32 * 2.0 - 1.0)
}

/// How far the surface can move: the displacement's reach, limited to what a distance grid padded
/// by `grid_pad` covers (it keeps 0.05 of margin).
pub fn reach(graph: &Graph, amount: f32, mask: (f32, f32), grid_pad: f32) -> f32 {
    let push = crate::view::push_range(graph.density_range(), amount, mask);
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
    pub combine: crate::meshsdf::Combine,
    /// Keep the detail the same size in object units: the size, in object units, of one normalized unit that
    /// `d.tile_size` and `d.amount` were set for (a mesh scaled by 2.5 then gets tile and amount divided by 2.5).
    /// None keeps them relative to each mesh's own size.
    pub object_scale: Option<f32>,
}

impl FileJob<'_> {
    /// Loads `input`, displaces it like the preview would and writes the result to `output`.
    pub fn run(&self, input: &Path, output: &Path, progress: &Progress) -> Result<String, String> {
        let (mesh, norm) = crate::meshformats::load(input)?.normalized();
        let (mut d, mut grid_pad) = (self.d, self.grid_pad);
        if let Some(reference) = self.object_scale {
            let k = reference / norm.scale;
            d.tile_size *= k;
            d.amount *= k;
            // A small mesh displaced by a fixed amount moves far in its own units: the grid has to reach that far.
            let push = crate::view::push_range(self.graph.density_range(), d.amount, d.mask.range);
            grid_pad = grid_pad.max((push.0.abs().max(push.1.abs()) + 0.05).min(8.0));
        }
        let sdf = crate::meshsdf::MeshSdf::build_with(&mesh, self.grid_res, grid_pad, &Default::default(), self.combine);
        let (_texture, view) = sdf.upload(self.device, self.queue); // keep the texture alive for the whole job
        let (lo, hi) = mesh.bounds();
        let ev = MeshEvaluator::new(self.device, self.queue, self.graph, self.seed, view, sdf.origin, sdf.voxel, (lo, hi), self.d.mask.count > 0)?;
        let job = Job { ev: &ev, d, lo, hi, reach: reach(self.graph, d.amount, d.mask.range, grid_pad), norm, settings: self.settings.clone() };
        job.run(output, progress)
    }
}
