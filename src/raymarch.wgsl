// Milestone 0: fullscreen raymarch of a sphere displaced by tiling Perlin fBm.

struct Uniforms {
    cam_pos: vec3f, aspect: f32,
    cam_fwd: vec3f, amount: f32,
    cam_right: vec3f, step_scale: f32,
    cam_up: vec3f, gain: f32,
    period: u32, octaves: u32, lacunarity: u32, seed: u32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

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

// ---- noise -------------------------------------------------------------

// pcg3d (Jarzynski & Olano 2020): integer-only, so every sampler can reproduce it exactly.
fn pcg3d(v_in: vec3u) -> vec3u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn grad(cell: vec3i, period: i32) -> vec3f {
    // Wrap the lattice so the noise repeats every `period` cells (= once per unit tile).
    let c = vec3u(((cell % period) + period) % period);
    let h = pcg3d(c + vec3u(u.seed * 0x9E3779B9u, 0u, 0u));
    let g = vec3f(h) / 4294967295.0 * 2.0 - 1.0;
    return normalize(g + 1e-6);
}

fn fade(t: vec3f) -> vec3f { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// Gradient noise in roughly [-1, 1]; tiles with `period` cells.
fn perlin(p: vec3f, period: i32) -> f32 {
    let i = vec3i(floor(p));
    let f = fract(p);
    let w = fade(f);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(grad(i + o, period), f - vec3f(o));
    }
    let x0 = mix(mix(n[0], n[1], w.x), mix(n[2], n[3], w.x), w.y);
    let x1 = mix(mix(n[4], n[5], w.x), mix(n[6], n[7], w.x), w.y);
    return mix(x0, x1, w.z) * 1.5;
}

fn fbm(p: vec3f) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var norm = 0.0;
    var period = i32(u.period);
    for (var o = 0u; o < u.octaves; o++) {
        sum += amp * perlin(p * f32(period), period);
        norm += amp;
        amp *= u.gain;
        period *= i32(u.lacunarity); // integer lacunarity keeps every octave tiling
    }
    return sum / norm;
}

// ---- scene -------------------------------------------------------------

fn sd_sphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }

fn field(p: vec3f) -> f32 {
    return sd_sphere(p, 1.0) - u.amount * fbm(p);
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
