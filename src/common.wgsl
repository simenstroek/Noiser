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
    // Range of amount × density (Graph::density_range): how far the surface can move in / out.
    push_min: f32, push_max: f32, max_steps: f32, _pad1: f32, // max_steps: ray march limit, 0 = default
    // The baked mesh distance grid (see meshsdf.rs): its min corner and texel size.
    mesh_origin: vec3f, mesh_voxel: f32,
    // The mesh's bounds, and the displacement mask (meshmask.rs): its term count and two vec4 per term.
    mesh_lo: vec3f, mask_n: u32,
    mesh_hi: vec3f, _pm: f32,
    mask: array<vec4f, 12>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> P: array<f32>;

// Slice values plus their min/max, as order-preserving u32 bit patterns so atomics can compare them.
struct Slice { lo: atomic<u32>, hi: atomic<u32>, vals: array<f32> };
@group(0) @binding(2) var<storage, read_write> slice: Slice;
@group(0) @binding(3) var<storage, read> BN: array<f32>;

// ---- mesh distance grid (group 2; a 1³ dummy until a mesh is loaded) ----
@group(2) @binding(0) var mesh_tex: texture_3d<f32>;

// ---- node caches (see cache.rs) ----
// Up to 8 baked node values, one tile each, sampled with wrap-around. Unused slots hold a 1³ dummy.
@group(1) @binding(0) var cache_smp: sampler;
@group(1) @binding(1) var cache0: texture_3d<f32>;
@group(1) @binding(2) var cache1: texture_3d<f32>;
@group(1) @binding(3) var cache2: texture_3d<f32>;
@group(1) @binding(4) var cache3: texture_3d<f32>;
@group(1) @binding(5) var cache4: texture_3d<f32>;
@group(1) @binding(6) var cache5: texture_3d<f32>;
@group(1) @binding(7) var cache6: texture_3d<f32>;
@group(1) @binding(8) var cache7: texture_3d<f32>;

// Cubic B-spline filtering from 8 trilinear fetches (Sigg & Hadwiger, GPU Gems 2 ch. 20).
// Smooth (C2), so normals of cached fields aren't faceted; it slightly softens detail.
fn tricubic(t: texture_3d<f32>, p: vec3f) -> f32 {
    let n = vec3f(textureDimensions(t));
    let c = p * n - 0.5;
    let i = floor(c);
    let f = c - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let h0 = (i - 0.5 + w1 / g0) / n; // texel centres are at (k + 0.5) / n
    let h1 = (i + 1.5 + w3 / g1) / n;
    var r = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let b = vec3<bool>((k & 1u) == 1u, (k & 2u) == 2u, (k & 4u) == 4u);
        let at = select(h0, h1, b);
        let w = select(g0, g1, b);
        r += w.x * w.y * w.z * textureSampleLevel(t, cache_smp, at, 0.0).x;
    }
    return r;
}

fn cache_sample(slot: u32, p: vec3f) -> f32 {
    switch slot {
        case 0u: { return tricubic(cache0, p); }
        case 1u: { return tricubic(cache1, p); }
        case 2u: { return tricubic(cache2, p); }
        case 3u: { return tricubic(cache3, p); }
        case 4u: { return tricubic(cache4, p); }
        case 5u: { return tricubic(cache5, p); }
        case 6u: { return tricubic(cache6, p); }
        default: { return tricubic(cache7, p); }
    }
}

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
