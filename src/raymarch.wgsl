// 3D preview: the base shape displaced by the graph's density, drawn as a solid surface or a cloud.

// ---- base shape (IQ, https://iquilezles.org/articles/distfunctions/) ----

fn sd_box(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn unit_prim(p: vec3f) -> f32 {
    switch u.prim {
        case 1u: { return sd_box(p, vec3f(0.8)); }
        case 2u: { return sd_box(p, vec3f(0.65)) - 0.15; } // round box
        case 3u: { return length(vec2f(length(p.xz) - 0.9, p.y)) - 0.35; } // torus
        case 4u: { // capped cylinder
            let d = abs(vec2f(length(p.xz), p.y)) - vec2f(0.8, 0.8);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
        }
        case 5u: { // capsule
            let q = vec3f(p.x, p.y - clamp(p.y, -0.6, 0.6), p.z);
            return length(q) - 0.5;
        }
        case 6u: { // octahedron (bound)
            let q = abs(p);
            return (q.x + q.y + q.z - 1.2) * 0.57735027;
        }
        default: { return length(p) - 1.0; } // sphere
    }
}

// The base shape slot: a primitive, or (prim 7) the baked distance to a loaded mesh. The mesh is
// normalized to a bounding sphere of radius 1 like the unit primitives, and scales with prim_size.
fn base_sdf(p: vec3f) -> f32 {
    if (u.prim == 7u) { return mesh_sdf_at(mesh_tex, p / u.prim_size, u.mesh_origin, u.mesh_voxel) * u.prim_size; }
    return unit_prim(p / u.prim_size) * u.prim_size;
}

// The displaced surface lies where base_sdf is between push_min and push_max. Beyond that shell
// (plus a margin, so the bound never looks like a hit) a bound is enough and the graph is skipped.
// Must stay above the hit threshold (0.5 px × t), which is ~0.01 at the far end of the bounds.
const SHELL_MARGIN: f32 = 0.02;

fn displaced(p: vec3f, b: f32) -> f32 {
    return b - u.amount * graph_density(p / u.tile_size);
}

// Signed distance bound for marching: exact near the surface, conservative elsewhere.
fn field(p: vec3f) -> f32 {
    let b = base_sdf(p);
    let outside = b - u.push_max; // the field is at least this
    if (outside > SHELL_MARGIN) { return outside; }
    let inside = b - u.push_min; // the field is at most this
    if (inside < -SHELL_MARGIN) { return inside; }
    return displaced(p, b);
}

// Exact inside the shape (clouds need the real interior values), zero outside it.
fn cloud_sigma(p: vec3f) -> f32 {
    let b = base_sdf(p);
    if (b - u.push_max > 0.0) { return 0.0; }
    return u.cloud_density * max(-displaced(p, b), 0.0);
}

// Ray interval inside a sphere around the origin that holds the displaced shape; x > y means a miss.
fn bounds(ro: vec3f, rd: vec3f) -> vec2f {
    let r = 1.4 * u.prim_size + max(u.push_max, 0.0) + SHELL_MARGIN; // 1.4 ≥ every unit primitive's extent
    let b = dot(ro, rd);
    let h = b * b - dot(ro, ro) + r * r;
    if (h < 0.0) { return vec2f(1.0, 0.0); }
    return vec2f(max(-b - sqrt(h), 0.0), -b + sqrt(h));
}

fn normal_at(p: vec3f, e: f32) -> vec3f {
    // IQ tetrahedral normal, as a loop so the graph is inlined once (driver compile time).
    var n = vec3f(0.0);
    for (var k = 0u; k < 4u; k++) {
        let o = tetra(k);
        n += o * field(p + o * e);
    }
    return normalize(n);
}

const FOCAL: f32 = 1.8;
const SUN: vec3f = vec3f(0.5345, 0.7127, 0.4543); // normalize(0.6, 0.8, 0.51)

fn background(uv: vec2f) -> vec3f {
    return mix(vec3f(0.10, 0.10, 0.12), vec3f(0.18, 0.19, 0.22), uv.y * 0.5 + 0.5);
}

fn solid(ro: vec3f, rd: vec3f, bg: vec3f) -> vec3f {
    // Pixel footprint per unit distance. Stopping and differentiating at that size
    // filters out sub-pixel noise detail instead of aliasing it into speckles.
    let pix = u.pixel / FOCAL;
    let span = bounds(ro, rd);
    if (span.x > span.y) { return bg; }
    var t = span.x;
    var t_out = t; // last t known to be outside, and the field there
    var d_out = 0.0;
    var hit = false;
    for (var i = 0; i < 300; i++) {
        let d = field(ro + rd * t);
        if (d < 0.0 && i > 0) {
            // Overshot into the surface (displacement breaks the distance bound). The field is
            // close to linear over one step, so a few false-position steps land on it.
            var a = t_out;
            var fa = d_out;
            var b = t;
            var fb = d;
            for (var k = 0; k < 4; k++) {
                let m = a + (b - a) * fa / (fa - fb);
                let fm = field(ro + rd * m);
                if (fm > 0.0) { a = m; fa = fm; } else { b = m; fb = fm; }
            }
            t = a + (b - a) * fa / (fa - fb);
            hit = true;
            break;
        }
        if (d < 0.5 * pix * t) { hit = true; break; }
        t_out = t;
        d_out = d;
        t += d * u.step_scale;
        if (t > span.y) { break; }
    }
    if (!hit) { return bg; }
    let p = ro + rd * t;
    let n = normal_at(p, max(pix * t, 1e-4));
    let diff = max(dot(n, SUN), 0.0);
    let sky = 0.5 + 0.5 * n.y;
    return vec3f(0.8, 0.78, 0.74) * (diff * 0.9 + sky * 0.25);
}

fn cloud(ro: vec3f, rd: vec3f, bg: vec3f) -> vec3f {
    // March only through a sphere that bounds the displaced shape.
    let span = bounds(ro, rd);
    if (span.x > span.y) { return bg; }
    let t0 = span.x;
    let t1 = span.y;
    let steps = 128;
    let dt = (t1 - t0) / f32(steps);
    var trans = 1.0;
    var light = vec3f(0.0);
    for (var i = 0; i < steps; i++) {
        let p = ro + rd * (t0 + (f32(i) + 0.5) * dt);
        let sigma = cloud_sigma(p);
        if (sigma <= 0.0) { continue; }
        // Short shadow ray toward the sun.
        var tau = 0.0;
        for (var j = 1; j < 6; j++) {
            tau += cloud_sigma(p + SUN * f32(j) * 0.12) * 0.12;
        }
        let lit = vec3f(1.0, 0.97, 0.9) * exp(-tau) + vec3f(0.25, 0.28, 0.35);
        let a = 1.0 - exp(-sigma * dt);
        light += trans * a * lit;
        trans *= 1.0 - a;
        if (trans < 0.01) { break; }
    }
    return light + bg * trans;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    let rd = normalize(u.cam_fwd * FOCAL + u.cam_right * in.uv.x * u.aspect + u.cam_up * in.uv.y);
    let bg = background(in.uv);
    var col: vec3f;
    if (u.mode == 1u) { col = cloud(u.cam_pos, rd, bg); } else { col = solid(u.cam_pos, rd, bg); }
    return vec4f(pow(col, vec3f(1.0 / 2.2)), 1.0);
}
