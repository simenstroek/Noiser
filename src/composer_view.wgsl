// The SDF Composer's viewport: a ray marcher over the interpreter in composer_sdf_lib.wgsl (concatenated by
// composer_gpu.rs). The maths is the CPU evaluator's (sdf_model.rs), line for line.

struct U {
    eye: vec4f,
    forward: vec4f,
    right: vec4f,
    up: vec4f,
    light: vec4f,
    // aspect, tan(fov / 2), far, selected shapes tint strength
    params: vec4f,
    // operation count, shape count, test point count, 0
    counts: vec4u,
    // the box around the drawn shapes: rays that miss it are not marched
    box_lo: vec4f,
    box_hi: vec4f,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> shapes: array<vec4f>;
@group(0) @binding(2) var<storage, read> ops: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> test_points: array<vec4f>;

fn sdf_op_count() -> u32 {
    return u.counts.x;
}

// ---- tests: distances at given points ----

@compute @workgroup_size(64)
fn cs_eval(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u.counts.z) { return; }
    test_points[id.x].w = map(test_points[id.x].xyz);
}

// ---- drawing ----

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return VsOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

fn normal_at(p: vec3f, t: f32) -> vec3f {
    let h = 0.0008 * (1.0 + t);
    let k0 = vec3f(1.0, -1.0, -1.0);
    let k1 = vec3f(-1.0, -1.0, 1.0);
    let k2 = vec3f(-1.0, 1.0, -1.0);
    let k3 = vec3f(1.0, 1.0, 1.0);
    return normalize(k0 * map(p + k0 * h) + k1 * map(p + k1 * h) + k2 * map(p + k2 * h) + k3 * map(p + k3 * h));
}

fn selected_distance(p: vec3f) -> f32 {
    var best = 1e30;
    let count = u.counts.y;
    for (var i = 0u; i < count; i++) {
        if (shapes[i * STRIDE + 11u].x > 0.5) {
            best = min(best, abs(shape_distance(i, p)));
        }
    }
    return best;
}

// The colour of the shapes nearest to a point on the surface, blended by how close each is.
fn surface_color(p: vec3f) -> vec3f {
    var sum = vec3f(0.0);
    var total = 0.0;
    let count = u.counts.y;
    for (var i = 0u; i < count; i++) {
        let d = abs(shape_distance(i, p));
        let w = 1.0 / (d * d * d * d + 1e-6);
        sum += shapes[i * STRIDE + 11u].yzw * w;
        total += w;
    }
    return sum / max(total, 1e-9);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let hs = u.params.y;
    let direction = normalize(u.forward.xyz + u.right.xyz * (ndc.x * hs * u.params.x) + u.up.xyz * (ndc.y * hs));
    let origin = u.eye.xyz;
    let background = 0.20 + 0.10 * (1.0 - in.uv.y);
    let inverse = 1.0 / select(direction, vec3f(1e-9), abs(direction) < vec3f(1e-9));
    let t0 = (u.box_lo.xyz - origin) * inverse;
    let t1 = (u.box_hi.xyz - origin) * inverse;
    let near_t = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
    let far_t = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (near_t > far_t) { return vec4f(vec3f(background), 1.0); }
    let far = min(u.params.z, far_t);
    var t = near_t;
    var hit = false;
    for (var i = 0; i < 260; i++) {
        let d = map(origin + direction * t);
        if (d < 0.0004 * (1.0 + t)) {
            hit = true;
            break;
        }
        t += max(d * 0.85, 0.0002);
        if (t > far) { break; }
    }
    if (!hit) {
        return vec4f(vec3f(background), 1.0);
    }
    let p = origin + direction * t;
    let n = normal_at(p, t);
    let diffuse = max(dot(n, u.light.xyz), 0.0);
    let rim = pow(1.0 - max(dot(n, -direction), 0.0), 3.0) * 0.15;
    // Ambient occlusion: how much the distance field closes in along the normal, five steps out to a few percent of
    // the scene's size.
    let reach = length(u.box_hi.xyz - u.box_lo.xyz) * 0.05;
    var occlusion = 0.0;
    var weight = 1.0;
    for (var i = 1; i <= 5; i++) {
        let h = reach * f32(i) / 5.0;
        occlusion += max(h - map(p + n * h), 0.0) / reach * weight;
        weight *= 0.75;
    }
    let ao = clamp(1.0 - 0.9 * occlusion, 0.25, 1.0);
    var color = vec3f(0.74, 0.75, 0.78);
    if (u.counts.w == 1u) {
        color = surface_color(p) * 0.92;
    }
    if (u.params.w > 0.5 && selected_distance(p) < 0.0025 * (1.0 + t)) {
        color = vec3f(0.98, 0.58, 0.18);
    }
    let light_amount = 0.28 * ao + 0.72 * diffuse * (0.5 + 0.5 * ao) + rim;
    return vec4f(min(color * light_amount, vec3f(1.0)), 1.0);
}
