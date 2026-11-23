// The SDF Composer's viewport: an interpreter for the flattened shape tree of `sdf_model::GpuScene`, and a ray marcher
// that draws it. The maths is the CPU evaluator's (sdf_model.rs), line for line; the layout of `shapes` and `ops` is
// described there too.

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
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> shapes: array<vec4f>;
@group(0) @binding(2) var<storage, read> ops: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> test_points: array<vec4f>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const STRIDE: u32 = 12u;

// ---- blends ----

fn union_smooth(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return b * (1.0 - h) + a * h - k * h * (1.0 - h);
}

fn subtract_smooth(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return max(b, -a); }
    let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    return b * (1.0 - h) - a * h + k * h * (1.0 - h);
}

fn intersect_smooth(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return max(a, b); }
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return b * (1.0 - h) + a * h + k * h * (1.0 - h);
}

fn union_chamfer(a: f32, b: f32, k: f32) -> f32 {
    return min(min(a, b), (a - 0.5 * k + b) * 0.5);
}

fn intersect_chamfer(a: f32, b: f32, k: f32) -> f32 {
    return max(max(a, b), (b + 0.5 * k + a) * 0.5);
}

fn uni(a: f32, b: f32, k: f32, chamfer: bool) -> f32 {
    if (chamfer) { return union_chamfer(a, b, k); }
    return union_smooth(a, b, k);
}

fn sub(a: f32, b: f32, k: f32, chamfer: bool) -> f32 {
    if (chamfer) { return intersect_chamfer(-a, b, k); }
    return subtract_smooth(a, b, k);
}

fn inter(a: f32, b: f32, k: f32, chamfer: bool) -> f32 {
    if (chamfer) { return intersect_chamfer(a, b, k); }
    return intersect_smooth(a, b, k);
}

fn combine(mode: u32, a: f32, b: f32, k: f32, r: f32, chamfer: bool) -> f32 {
    switch mode {
        case 0u: { return uni(a, b, k, chamfer); }
        case 1u: { return sub(a, b, k, chamfer); }
        case 2u: { return inter(a, b, k, chamfer); }
        case 3u, 9u: { return b; }
        case 4u: { return min(sub(a - r, b, k, chamfer), a); }
        case 5u: { return min(sub(b - r, a, k, chamfer), b); }
        case 6u: { return uni(b, inter(a, b - r, k, chamfer), k, chamfer); }
        case 7u: { return sub(sub(b + r, a, k, chamfer), b, k, chamfer); }
        default: {
            let radius = r + 0.01;
            return inter(a, abs(b + radius) - radius, k, chamfer);
        }
    }
}

// ---- primitives ----

fn smooth_abs(value: f32, width: f32) -> f32 {
    let a = abs(value);
    if (width <= 0.0 || a > width) { return a; }
    let n = width * 0.25;
    let k0 = 2.0 * n - width;
    let k1 = 2.0 * width - 3.0 * n;
    let t = a / width;
    return (k0 * t + k1) * t * t + n;
}

fn glsl_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn uber(p: vec3f, size: vec4f, radii: vec3f) -> f32 {
    var s = size;
    var r = radii;
    s.x -= r.x;
    s.z -= r.x;
    r.x -= s.w;
    s.w -= r.y;
    s.y -= r.y;
    let b = vec2f(r.z, -2.0 * s.y);
    let ba = b / max(dot(b, b), 1e-30);
    let d = abs(p) - s.xyz;
    var q = length(max(d.xz, vec2f(0.0))) + min(0.0, max(d.x, d.z)) - r.x;
    q = abs(q) - s.w;
    let pa = vec2f(q, p.y - s.y);
    let diagonal = pa - vec2f(r.z, b.y) * clamp(dot(pa, ba), 0.0, 1.0);
    let h0 = vec2f(max(q - r.z, 0.0), p.y + s.y);
    let h1 = vec2f(max(q, 0.0), p.y - s.y);
    let nearest = min(dot(diagonal, diagonal), min(dot(h0, h0), dot(h1, h1)));
    let side = sign(max(dot(pa, vec2f(-ba.y, ba.x)), d.y));
    return sqrt(nearest) * side - r.y;
}

fn ellipse_distance(point: vec2f, ab: vec2f) -> f32 {
    let p = abs(point);
    let q = ab * (p - ab);
    var cs = normalize(select(vec2f(1.0, 0.01), vec2f(0.01, 1.0), q.x < q.y));
    for (var i = 0; i < 5; i++) {
        let uu = ab * cs;
        let vv = ab * vec2f(-cs.y, cs.x);
        let a = dot(p - uu, vv);
        let c = dot(p - uu, uu) + dot(vv, vv);
        let b = sqrt(max(c * c - a * a, 0.0));
        cs = vec2f(cs.x * b - cs.y * a, cs.y * b + cs.x * a) / c;
    }
    let d = length(p - ab * cs);
    return select(-d, d, dot(p / ab, p / ab) > 1.0);
}

fn cylinder(p: vec3f, r: vec3f) -> f32 {
    let d = ellipse_distance(p.xz, r.xz);
    let w = vec2f(d, abs(p.y) - r.y);
    return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
}

fn polygon(p: vec3f, r: vec2f, points: f32, star: f32, bevel: f32) -> f32 {
    let m = points + star * (2.0 - points);
    let an = PI / points;
    let en = PI / m;
    let racs = r.x * vec2f(cos(an), sin(an));
    let ecs = vec2f(cos(en), sin(en));
    let x = abs(p.x);
    let y = abs(p.y);
    let bn = glsl_mod(atan2(x, p.z), 2.0 * an) - an;
    let radius = length(vec2f(x, p.z));
    var q = vec2f(radius * cos(bn) - racs.x, radius * abs(sin(bn)) - racs.y);
    let t = clamp(-dot(q, ecs), 0.0, racs.y / ecs.y);
    q = q + ecs * t;
    let d1 = length(q) * sign(q.x) - bevel;
    let d2 = y - r.y;
    let d = vec2f(d1, d2);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn prism(p: vec3f, h: vec3f, bevel: f32) -> f32 {
    let x = abs(p.x) - h.x;
    let y = p.y + h.y;
    let e = vec2f(-h.x, h.y * 2.0);
    let t = clamp((x * e.x + y * e.y) / dot(e, e), 0.0, 1.0);
    let segment = vec2f(x, y) - e * t;
    var d1 = length(segment);
    if (max(segment.x, segment.y) < 0.0) { d1 = -min(d1, y); }
    d1 = d1 - bevel;
    let d2 = abs(p.z) - h.z - bevel;
    return length(max(vec2f(d1, d2), vec2f(0.0))) + min(max(d1, d2), 0.0);
}

fn brush_distance(kind: u32, hs: vec3f, b: vec4f, p: vec3f) -> f32 {
    switch kind {
        case 0u: {
            let length_scaled = length(p / hs);
            return (length_scaled - 1.0) * min(hs.x, min(hs.y, hs.z));
        }
        case 1u: { return uber(vec3f(p.x, -p.y, p.z), vec4f(hs, b.x), vec3f(b.y, b.z, b.w)); }
        case 2u: { return cylinder(p, hs - vec3f(b.x)) - b.x; }
        case 3u: { return prism(p, hs - vec3f(b.y), max(b.y - b.x, 0.0)) - b.x; }
        default: { return polygon(p, vec2f(hs.x - b.y, hs.y - b.x), b.z, b.w, max(b.y - b.x, 0.0)) - b.x; }
    }
}

// ---- shapes ----

fn affine(base: u32, p: vec3f) -> vec3f {
    let r0 = shapes[base];
    let r1 = shapes[base + 1u];
    let r2 = shapes[base + 2u];
    return vec3f(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}

fn shape_distance(index: u32, point: vec3f) -> f32 {
    let base = index * STRIDE;
    var p = affine(base, point);
    let head = shapes[base + 6u];
    let mirror = u32(head.y);
    if ((mirror & 1u) != 0u) { p.x = smooth_abs(p.x, head.z); }
    if ((mirror & 2u) != 0u) { p.y = smooth_abs(p.y, head.z); }
    if ((mirror & 4u) != 0u) { p.z = smooth_abs(p.z, head.z); }
    let local = affine(base + 3u, p);
    let kind = u32(head.x);
    let hs = shapes[base + 7u].xyz;
    let b = shapes[base + 8u];
    let repeat = u32(head.w);
    if (repeat == 1u) {
        let steps = shapes[base + 9u].xyz;
        let counts = shapes[base + 10u].xyz;
        let last = counts - vec3f(1.0);
        let shifted = local + 0.5 * steps * last;
        let id = clamp(round(shifted / steps), vec3f(0.0), last);
        var cell = shifted - steps * id;
        let odd = vec3<i32>(id) & vec3<i32>(1);
        cell = select(cell, -cell, odd == vec3<i32>(1));
        return brush_distance(kind, hs, b, cell);
    }
    if (repeat == 2u) {
        let a = shapes[base + 9u];
        let radius = a.x;
        let count = a.y;
        let axis = u32(a.z);
        var r = local.zy;
        if (axis == 1u) { r = local.zx; }
        if (axis == 2u) { r = local.yx; }
        let angle_step = TAU / count;
        let id = floor(atan2(r.y, r.x) / angle_step);
        var best = 1e30;
        for (var k = 0; k < 2; k++) {
            let angle = angle_step * (id + f32(k));
            let s = sin(angle);
            let c = cos(angle);
            let turned = vec2f(c * r.x + s * r.y, -s * r.x + c * r.y);
            var q = vec3f(local.x, turned.y, turned.x - radius);
            if (axis == 1u) { q = vec3f(turned.y, local.y, turned.x - radius); }
            if (axis == 2u) { q = vec3f(turned.y, turned.x - radius, local.z); }
            best = min(best, brush_distance(kind, hs, b, q));
        }
        return best;
    }
    return brush_distance(kind, hs, b, local);
}

fn map(p: vec3f) -> f32 {
    let count = u.counts.x;
    if (count == 0u) { return 1000.0; }
    var stack: array<f32, 32>;
    var top = 0;
    for (var i = 0u; i < count; i++) {
        let op = ops[i];
        let kind = u32(op.x);
        if (kind == 0u) {
            stack[top] = shape_distance(u32(op.y), p);
            top++;
        } else if (kind == 3u) {
            stack[top - 1] = stack[top - 1] * op.y;
        } else {
            let a = stack[top - 1];
            let b = stack[top - 2];
            top--;
            stack[top - 1] = combine(u32(op.y), a, b, op.z, op.w, kind == 2u);
        }
    }
    return stack[0];
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

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let hs = u.params.y;
    let direction = normalize(u.forward.xyz + u.right.xyz * (ndc.x * hs * u.params.x) + u.up.xyz * (ndc.y * hs));
    let origin = u.eye.xyz;
    let far = u.params.z;
    var t = 0.0;
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
        let shade = 0.20 + 0.10 * (1.0 - in.uv.y);
        return vec4f(vec3f(shade), 1.0);
    }
    let p = origin + direction * t;
    let n = normal_at(p, t);
    let diffuse = max(dot(n, u.light.xyz), 0.0);
    let rim = pow(1.0 - max(dot(n, -direction), 0.0), 3.0) * 0.15;
    var color = vec3f(0.74, 0.75, 0.78);
    if (u.params.w > 0.5 && selected_distance(p) < 0.0025 * (1.0 + t)) {
        color = vec3f(0.98, 0.58, 0.18);
    }
    let light_amount = 0.28 + 0.72 * diffuse + rim;
    return vec4f(min(color * light_amount, vec3f(1.0)), 1.0);
}
