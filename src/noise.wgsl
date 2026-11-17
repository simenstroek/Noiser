// Noise library shared by every shader that evaluates a graph.
// Expects the including shader to declare `u.seed` (global seed), `P: array<f32>` (node params)
// and `BN: array<f32>` (blue noise tables, see bluenoise.rs).
//
// Every noise works in lattice units: x = position * scale. `period` > 0 wraps the lattice every
// `period` cells so the result repeats once per unit tile.

fn safe_normalize(v: vec3f) -> vec3f { return v / max(length(v), 1e-12); }

// pcg3d (Jarzynski & Olano 2020): integer-only, so every sampler can reproduce it exactly.
fn pcg3d(v_in: vec3u) -> vec3u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn wrap_cell(c: vec3i, period: i32) -> vec3i {
    if (period > 0) { return ((c % period) + period) % period; }
    return c;
}

// Three uniform randoms in [0, 1] for a (wrapped) lattice cell.
fn hash01(c: vec3i, seed: u32) -> vec3f {
    let h = pcg3d(bitcast<vec3u>(c) + vec3u(seed * 0x9E3779B9u, 0u, 0u));
    return vec3f(h) / 4294967295.0;
}

fn rand_dir(c: vec3i, seed: u32) -> vec3f { return safe_normalize(hash01(c, seed) * 2.0 - 1.0); }

fn fade(t: vec3f) -> vec3f { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// ---- Perlin ----------------------------------------------------------------

fn perlin(x: vec3f, period: i32, seed: u32) -> f32 {
    let i = vec3i(floor(x));
    let f = fract(x);
    let w = fade(f);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(rand_dir(wrap_cell(i + o, period), seed), f - vec3f(o));
    }
    let x0 = mix(mix(n[0], n[1], w.x), mix(n[2], n[3], w.x), w.y);
    let x1 = mix(mix(n[4], n[5], w.x), mix(n[6], n[7], w.x), w.y);
    return mix(x0, x1, w.z) * 1.5;
}

// ---- Simplex (tiling) --------------------------------------------------------
// Gustavson & McEwan's psrdnoise approach (JCGT 2022): simplices on the BCC lattice
// uvw = M x, M = [[0,1,1],[1,0,1],[1,1,0]]. Whole-number periods along x, y, z are lattice
// vectors there, so wrapping each simplex corner's position wraps the noise.

fn simplex(x: vec3f, period: i32, seed: u32) -> f32 {
    let uvw = vec3f(x.y + x.z, x.x + x.z, x.x + x.y);
    let i0 = floor(uvw);
    let f0 = uvw - i0;
    // Which of the six tetrahedra in this cube (sort the fractional parts).
    let g = step(f0.yzx, f0.xyz);
    let l = 1.0 - g;
    let o1 = min(g, l.zxy);
    let o2 = max(g, l.zxy);
    let corners = array<vec3f, 4>(i0, i0 + o1, i0 + o2, i0 + 1.0);
    let pf = f32(period);
    var n = 0.0;
    for (var k = 0; k < 4; k++) {
        let c = corners[k];
        let v = 0.5 * vec3f(-c.x + c.y + c.z, c.x - c.y + c.z, c.x + c.y - c.z); // M⁻¹ c
        let d = x - v;
        // 0.5 is the distance from a corner to the opposite face, squared: the kernel ends at the simplex.
        let w = max(0.5 - dot(d, d), 0.0);
        if (w > 0.0) {
            var vw = v;
            if (period > 0) { vw = v - floor(v / pf) * pf; }
            let cw = vec3i(round(vec3f(vw.y + vw.z, vw.x + vw.z, vw.x + vw.y)));
            let w2 = w * w;
            n += w2 * w2 * dot(rand_dir(cw, seed), d);
        }
    }
    return n * SIMPLEX_GAIN;
}

// Brings simplex to roughly [-1, 1]; measured by the range test in gpu_tests.rs.
const SIMPLEX_GAIN: f32 = 90.0;

// ---- Cellular (Voronoi / Worley) ------------------------------------------

// (F1, F2, random value of the nearest cell in [-1, 1]); feature points jittered over the whole cell.
fn cells(x: vec3f, period: i32, seed: u32) -> vec3f {
    let i = vec3i(floor(x));
    let f = x - floor(x);
    var f1 = 1e9;
    var f2 = 1e9;
    var val = 0.0;
    for (var k = 0; k < 27; k++) {
        let o = vec3i(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1);
        let c = wrap_cell(i + o, period);
        let d = length(vec3f(o) + hash01(c, seed) - f);
        if (d < f1) {
            f2 = f1;
            f1 = d;
            val = hash01(c, seed + 0x51ED27u).x * 2.0 - 1.0;
        } else if (d < f2) {
            f2 = d;
        }
    }
    return vec3f(f1, f2, val);
}

fn worley(x: vec3f, period: i32, seed: u32, feature: u32) -> f32 {
    let c = cells(x, period, seed);
    // Typical range of each feature, used to map it to signed [-1, 1].
    switch feature {
        case 1u: { return clamp(c.y / 1.4 * 2.0 - 1.0, -1.0, 1.0); }
        case 2u: { return clamp((c.y - c.x) * 2.0 - 1.0, -1.0, 1.0); }
        default: { return clamp(c.x * 2.0 - 1.0, -1.0, 1.0); }
    }
}

// ---- White / blue ------------------------------------------------------------

fn white(x: vec3f, period: i32, seed: u32) -> f32 {
    return hash01(wrap_cell(vec3i(floor(x)), period), seed).x * 2.0 - 1.0;
}

// Blue noise tables of size 4³, 8³, 16³, 32³ stored back to back in BN.
fn blue(x: vec3f, period: i32, seed: u32) -> f32 {
    // Largest table that divides the period, so whole tables fill the tile.
    var t = 32;
    var off = 4u * 4u * 4u + 8u * 8u * 8u + 16u * 16u * 16u;
    if (period > 0) {
        if (period % 32 != 0) { t = 16; off = 64u + 512u; }
        if (period % 16 != 0) { t = 8; off = 64u; }
        if (period % 8 != 0) { t = 4; off = 0u; }
    }
    // The seed shifts the lookup by whole cells, which keeps tiling and the blue spectrum.
    let shift = vec3i(hash01(vec3i(0), seed) * f32(t));
    let c = wrap_cell(vec3i(floor(x)) + shift, t);
    let ut = u32(t);
    return BN[off + (u32(c.z) * ut + u32(c.y)) * ut + u32(c.x)] * 2.0 - 1.0;
}

// ---- Dispatch and fBm ------------------------------------------------------

fn noise_base(x: vec3f, period: i32, seed: u32, kind: u32, feature: u32) -> f32 {
    // Every kind stays within [-1, 1]; Graph::density_range relies on it.
    switch kind {
        case 1u: { return clamp(simplex(x, period, seed), -1.0, 1.0); }
        case 2u, 3u: { return cells(x, period, seed).z; }
        case 4u: { return worley(x, period, seed, feature); }
        case 5u: { return white(x, period, seed); }
        case 6u: { return blue(x, period, seed); }
        default: { return clamp(perlin(x, period, seed), -1.0, 1.0); }
    }
}

// A noise node with Blender's Noise Texture parameters. Params at P[s..s+8]:
// scale, normalize, octaves, lacunarity, roughness, distortion, seed (-1 = global), worley feature.
// `kind` is a NoiseKind discriminant; kinds without octaves always use one.
fn noise(q: vec3f, s: u32, kind: u32) -> f32 {
    let scale = P[s];
    let normalize = P[s + 1u] > 0.5;
    let has_octaves = kind == 0u || kind == 1u || kind == 3u || kind == 4u;
    let octaves = select(1u, u32(P[s + 2u]), has_octaves);
    let lacunarity = P[s + 3u];
    let roughness = P[s + 4u];
    let distortion = P[s + 5u];
    let seed = select(u.seed, u32(P[s + 6u]), P[s + 6u] >= 0.0);
    let feature = u32(P[s + 7u]);

    // Whole-number scale and lacunarity tile; anything else samples the unwrapped lattice.
    let tiles = fract(scale) == 0.0 && (octaves <= 1u || fract(lacunarity) == 0.0);
    var period = select(0, i32(scale), tiles);
    var x = q * scale;
    if (distortion != 0.0) {
        x += vec3f(perlin(x, period, seed + 1u), perlin(x, period, seed + 2u), perlin(x, period, seed + 3u)) * distortion;
    }
    var sum = 0.0;
    var amp = 1.0;
    var total = 0.0;
    for (var o = 0u; o < max(octaves, 1u); o++) {
        sum += amp * noise_base(x, period, seed + o, kind, feature);
        total += amp;
        amp *= roughness;
        x *= lacunarity;
        period *= i32(lacunarity);
    }
    let v = sum / max(total, 1e-6); // signed, roughly [-1, 1]
    return select(v, v * 0.5 + 0.5, normalize);
}
