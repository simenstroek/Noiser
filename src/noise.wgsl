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

// Wraps a cell that is at most one period outside [0, period). Integer % is slow on GPUs, so
// noises wrap their base cell once with wrap_cell and step to neighbours with this.
fn wrap_near(c: vec3i, period: i32) -> vec3i {
    if (period <= 0) { return c; }
    let p = vec3i(period);
    return select(select(c, c - p, c >= p), c + p, c < vec3i(0));
}

// Three uniform randoms in [0, 1] for a (wrapped) lattice cell.
fn hash01(c: vec3i, seed: u32) -> vec3f {
    let h = pcg3d(bitcast<vec3u>(c) + vec3u(seed * 0x9E3779B9u, 0u, 0u));
    return vec3f(h) / 4294967295.0;
}

fn rand_dir(c: vec3i, seed: u32) -> vec3f { return safe_normalize(hash01(c, seed) * 2.0 - 1.0); }

fn fade(t: vec3f) -> vec3f { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// ---- Perlin ----------------------------------------------------------------

// Gradient at wrapped base cell `i` + corner `o`, dotted with the offset to that corner.
fn corner(i: vec3i, o: vec3i, f: vec3f, period: i32, seed: u32) -> f32 {
    return dot(rand_dir(wrap_near(i + o, period), seed), f - vec3f(o));
}

// Written out per corner: a loop over a local array spills it out of registers.
fn perlin(x: vec3f, period: i32, seed: u32) -> f32 {
    let i = wrap_cell(vec3i(floor(x)), period);
    let f = fract(x);
    let w = fade(f);
    let n000 = corner(i, vec3i(0, 0, 0), f, period, seed);
    let n100 = corner(i, vec3i(1, 0, 0), f, period, seed);
    let n010 = corner(i, vec3i(0, 1, 0), f, period, seed);
    let n110 = corner(i, vec3i(1, 1, 0), f, period, seed);
    let n001 = corner(i, vec3i(0, 0, 1), f, period, seed);
    let n101 = corner(i, vec3i(1, 0, 1), f, period, seed);
    let n011 = corner(i, vec3i(0, 1, 1), f, period, seed);
    let n111 = corner(i, vec3i(1, 1, 1), f, period, seed);
    let x0 = mix(mix(n000, n100, w.x), mix(n010, n110, w.x), w.y);
    let x1 = mix(mix(n001, n101, w.x), mix(n011, n111, w.x), w.y);
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
    let n = simplex_corner(x, i0, period, seed) + simplex_corner(x, i0 + o1, period, seed)
          + simplex_corner(x, i0 + o2, period, seed) + simplex_corner(x, i0 + 1.0, period, seed);
    return n * SIMPLEX_GAIN;
}

// One simplex corner c (in lattice space uvw).
fn simplex_corner(x: vec3f, c: vec3f, period: i32, seed: u32) -> f32 {
    let v = 0.5 * vec3f(-c.x + c.y + c.z, c.x - c.y + c.z, c.x + c.y - c.z); // M⁻¹ c
    let d = x - v;
    // 0.5 is the distance from a corner to the opposite face, squared: the kernel ends at the simplex.
    let w = max(0.5 - dot(d, d), 0.0);
    if (w <= 0.0) { return 0.0; }
    var vw = v;
    if (period > 0) { let pf = f32(period); vw = v - floor(v / pf) * pf; }
    let cw = vec3i(round(vec3f(vw.y + vw.z, vw.x + vw.z, vw.x + vw.y)));
    let w2 = w * w;
    return w2 * w2 * dot(rand_dir(cw, seed), d);
}

// Brings simplex to roughly [-1, 1]; measured by the range test in gpu_tests.rs.
const SIMPLEX_GAIN: f32 = 90.0;

// ---- Cellular (Voronoi / Worley) ------------------------------------------

// (F1, F2, random value of the nearest cell in [-1, 1]); feature points jittered over the whole cell.
// The cell value costs one more hash, so only Voronoi asks for it.
fn cells(x: vec3f, period: i32, seed: u32, want_value: bool) -> vec3f {
    let i = wrap_cell(vec3i(floor(x)), period);
    let f = fract(x);
    var f1 = 1e9; // squared distances until the end
    var f2 = 1e9;
    var nearest = i;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let o = vec3i(xx, y, z);
                let c = wrap_near(i + o, period);
                let d = vec3f(o) + hash01(c, seed) - f;
                let d2 = dot(d, d);
                if (d2 < f1) {
                    f2 = f1;
                    f1 = d2;
                    nearest = c;
                } else if (d2 < f2) {
                    f2 = d2;
                }
            }
        }
    }
    var val = 0.0;
    if (want_value) { val = hash01(nearest, seed + 0x51ED27u).x * 2.0 - 1.0; }
    return vec3f(sqrt(f1), sqrt(f2), val);
}

fn worley(x: vec3f, period: i32, seed: u32, feature: u32) -> f32 {
    let c = cells(x, period, seed, false);
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

// ---- fBm -------------------------------------------------------------------
// The per-node fBm wrapper lives in noise_fbm.wgsl and is instantiated per noise kind by
// codegen.rs, so each call only contains its own noise (the driver compiles every inlined path).

// Tetrahedral sample offsets (IQ), as a function of k = 0..3 so callers can loop with a single
// call site instead of four inlined copies of the graph.
fn tetra(k: u32) -> vec3f {
    return vec3f(select(-1.0, 1.0, k == 0u || k == 3u), select(-1.0, 1.0, k >= 2u), select(-1.0, 1.0, (k & 1u) == 1u));
}
