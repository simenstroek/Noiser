// Noise library shared by every shader that evaluates a graph.
// Expects the including shader to declare `u.seed` (global seed) and `P: array<f32>` (node params).

fn safe_normalize(v: vec3f) -> vec3f { return v / max(length(v), 1e-12); }

// pcg3d (Jarzynski & Olano 2020): integer-only, so every sampler can reproduce it exactly.
fn pcg3d(v_in: vec3u) -> vec3u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// Random unit gradient for a lattice cell. period > 0 wraps the lattice so the noise tiles.
fn lattice_grad(cell: vec3i, period: i32, seed: u32) -> vec3f {
    var c = cell;
    if (period > 0) { c = ((cell % period) + period) % period; }
    let h = pcg3d(bitcast<vec3u>(c) + vec3u(seed * 0x9E3779B9u, 0u, 0u));
    return safe_normalize(vec3f(h) / 4294967295.0 * 2.0 - 1.0);
}

fn fade(t: vec3f) -> vec3f { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// Gradient noise in roughly [-1, 1].
fn perlin(x: vec3f, period: i32, seed: u32) -> f32 {
    let i = vec3i(floor(x));
    let f = fract(x);
    let w = fade(f);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(lattice_grad(i + o, period, seed), f - vec3f(o));
    }
    let x0 = mix(mix(n[0], n[1], w.x), mix(n[2], n[3], w.x), w.y);
    let x1 = mix(mix(n[4], n[5], w.x), mix(n[6], n[7], w.x), w.y);
    return mix(x0, x1, w.z) * 1.5;
}

// Perlin fBm with Blender's Noise Texture parameters. Params at P[s..s+7]:
// scale, normalize, octaves, lacunarity, roughness, distortion, seed (-1 = global).
fn noise_perlin(q: vec3f, s: u32) -> f32 {
    let scale = P[s];
    let normalize = P[s + 1u] > 0.5;
    let octaves = u32(P[s + 2u]);
    let lacunarity = P[s + 3u];
    let roughness = P[s + 4u];
    let distortion = P[s + 5u];
    let seed = select(u.seed, u32(P[s + 6u]), P[s + 6u] >= 0.0);

    // Whole-number scale and lacunarity tile; anything else samples the unwrapped lattice.
    let tiles = fract(scale) == 0.0 && fract(lacunarity) == 0.0;
    var period = select(0, i32(scale), tiles);
    var x = q * scale;
    if (distortion != 0.0) {
        x += vec3f(perlin(x, period, seed + 1u), perlin(x, period, seed + 2u), perlin(x, period, seed + 3u)) * distortion;
    }
    var sum = 0.0;
    var amp = 1.0;
    var total = 0.0;
    for (var o = 0u; o < max(octaves, 1u); o++) {
        sum += amp * perlin(x, period, seed);
        total += amp;
        amp *= roughness;
        x *= lacunarity;
        period *= i32(lacunarity);
    }
    let v = sum / max(total, 1e-6); // signed, roughly [-1, 1]
    return select(v, v * 0.5 + 0.5, normalize);
}
