// Draws the Composer's off-screen picture into the viewport, stretched when it was rendered at a lower resolution.

@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var picture_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return VsOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    return textureSampleLevel(picture, picture_sampler, in.uv, 0.0);
}
