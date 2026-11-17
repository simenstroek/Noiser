// Bindings and fullscreen vertex stage shared by the preview shaders.
// Must match `Uniforms` in view.rs field for field.

struct Uniforms {
    cam_pos: vec3f, aspect: f32,
    cam_fwd: vec3f, amount: f32,
    cam_right: vec3f, step_scale: f32,
    cam_up: vec3f, seed: u32,
    tile_size: f32, prim_size: f32, pixel: f32, cloud_density: f32,
    mode: u32, prim: u32, slice_axis: u32, slice_mode: u32,
    slice_pos: f32, slice_aspect: f32, slice_tiles: f32, slice_res: u32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> P: array<f32>;

// Slice values plus their min/max, as order-preserving u32 bit patterns so atomics can compare them.
struct Slice { lo: atomic<u32>, hi: atomic<u32>, vals: array<f32> };
@group(0) @binding(2) var<storage, read_write> slice: Slice;

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle covering the viewport; uv is -1..1 with +y up.
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p * 2.0 - 1.0;
    return o;
}
