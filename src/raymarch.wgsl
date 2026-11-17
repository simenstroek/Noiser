// 3D preview: raymarches the base shape displaced by the graph's density.
// Concatenated after noise.wgsl and the generated graph code.

struct Uniforms {
    cam_pos: vec3f, aspect: f32,
    cam_fwd: vec3f, amount: f32,
    cam_right: vec3f, step_scale: f32,
    cam_up: vec3f, seed: u32,
    tile_size: f32, _pad0: f32, _pad1: f32, _pad2: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> P: array<f32>;

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle covering the viewport.
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p * 2.0 - 1.0;
    return o;
}

// The base shape slot: a primitive now, a sampled mesh SDF later.
fn base_sdf(p: vec3f) -> f32 { return length(p) - 1.0; }

fn field(p: vec3f) -> f32 {
    return base_sdf(p) - u.amount * graph_density(p / u.tile_size);
}

fn normal_at(p: vec3f) -> vec3f {
    // IQ tetrahedral normal.
    let e = 0.0015;
    let k = vec2f(1.0, -1.0);
    return normalize(
        k.xyy * field(p + k.xyy * e) + k.yyx * field(p + k.yyx * e) +
        k.yxy * field(p + k.yxy * e) + k.xxx * field(p + k.xxx * e));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    let rd = normalize(u.cam_fwd * 1.8 + u.cam_right * in.uv.x * u.aspect + u.cam_up * in.uv.y);
    let ro = u.cam_pos;
    let bg = mix(vec3f(0.10, 0.10, 0.12), vec3f(0.18, 0.19, 0.22), in.uv.y * 0.5 + 0.5);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < 256; i++) {
        let d = field(ro + rd * t);
        if (abs(d) < 0.0005 * t) { hit = true; break; }
        t += d * u.step_scale; // displacement breaks the distance bound, so step conservatively
        if (t > 20.0) { break; }
    }
    if (!hit) { return vec4f(pow(bg, vec3f(1.0 / 2.2)), 1.0); }

    let p = ro + rd * t;
    let n = normal_at(p);
    let l = normalize(vec3f(0.6, 0.8, 0.4));
    let diff = max(dot(n, l), 0.0);
    let sky = 0.5 + 0.5 * n.y;
    let col = vec3f(0.8, 0.78, 0.74) * (diff * 0.9 + sky * 0.25);
    return vec4f(pow(col, vec3f(1.0 / 2.2)), 1.0);
}
