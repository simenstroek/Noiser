// Noise library shared by every shader that evaluates a graph.
// Expects the including shader to declare `u.seed` (global seed), `P: array<f32>` (node params)
// and `BN: array<f32>` (blue noise tables, see bluenoise.rs).
//
// Every noise works in lattice units: x = position * scale. `period` > 0 wraps the lattice every
// `period` cells so the result repeats once per unit tile.

fn safe_normalize(v: vec3f) -> vec3f { return v / max(length(v), 1e-12); }

// The slope in degrees of a field with gradient g (over x and y), for a height of `k` per unit of the field.
fn slope_degrees(g: vec2f, k: f32) -> f32 { return degrees(atan(k * length(g))); }

// A terrain map (the Landscape tab: painted masks and the other maps, see terrain_maps.rs): `side` x `side` values in P
// starting `offset` x 256 floats after MASK_BASE (offsets are in those units so f32 holds them exactly); q.xy across the
// terrain, 0 to 1. Bilinear.
const MASK_BASE: u32 = 1048576u;
const MAP_ALIGN: u32 = 256u;
fn terrain_map(offset: f32, side: f32, q: vec3f) -> f32 {
    let res = max(side, 2.0);
    let n = u32(res);
    let base = MASK_BASE + u32(offset) * MAP_ALIGN;
    let g = clamp(q.xy, vec2f(0.0), vec2f(1.0)) * (res - 1.0);
    let i = min(vec2u(floor(g)), vec2u(n - 2u, n - 2u));
    let f = g - vec2f(i);
    let a = P[base + i.y * n + i.x];
    let b = P[base + i.y * n + i.x + 1u];
    let c = P[base + (i.y + 1u) * n + i.x];
    let d = P[base + (i.y + 1u) * n + i.x + 1u];
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// A node's numeric parameters come from P[i], except those a wired input pin replaces: the generated code lists
// them in ov_i / ov_v (ov_n entries) around the node's call.
var<private> ov_n: u32;
var<private> ov_i: array<u32, 12>;
var<private> ov_v: array<f32, 12>;
fn pp(i: u32) -> f32 {
    for (var k = 0u; k < ov_n; k = k + 1u) {
        if (ov_i[k] == i) { return ov_v[k]; }
    }
    return P[i];
}

// A position in cylindrical (radius from the axis, angle around it, height along it) or spherical (radius from the
// centre, angle around the axis, angle from the axis) coordinates; the angles are in radians.
fn radial_coords(q: vec3f, center: vec3f, axis: vec3f, spherical: bool) -> vec3f {
    let a = select(vec3f(0.0, 0.0, 1.0), axis / max(length(axis), 1e-12), length(axis) > 1e-9);
    // The angle is measured from X (or from Y when the axis is X): atan2(y, x) about Z, atan2(z, y) about X.
    let reference = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(a.x) > 0.9);
    let u = normalize(reference - a * dot(reference, a));
    let w = cross(a, u);
    let d = q - center;
    let h = dot(d, a);
    let rv = d - a * h;
    let angle = atan2(dot(rv, w), dot(rv, u));
    if (spherical) { return vec3f(length(d), angle, atan2(length(rv), h)); }
    return vec3f(length(rv), angle, h);
}

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

// The Random node: one number in [lo, hi] from the global seed and the node's id, or from the node's own seed.
fn random_range(lo: f32, hi: f32, seed_override: f32, id: u32) -> f32 {
    let seed = select(u.seed * 0x9E3779B9u + id * 0x85EBCA6Bu, u32(seed_override), seed_override >= 0.0);
    return mix(lo, hi, hash01(vec3i(0), seed).x);
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
    var g = step(f0.yzx, f0.xyz);
    // With all three fractions equal (points on the cube's main diagonals, which regular grids hit) every
    // comparison is true and two corners would be counted twice; the last one is made strict so that exactly
    // one of the six orderings is chosen.
    g.z = 1.0 - step(f0.z, f0.x);
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

// ---- Cellular (Worley) ------------------------------------------

// Euclidean distance with fully random points: the original loop, kept as the fast path for the default
// settings (the general one below does more work per feature point). Must agree with `cells`.
fn cells_classic(x: vec3f, period: i32, seed: u32, want_value: bool) -> vec3f {
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

// The Worley node's function when only the classic settings are used (Euclidean, jitter 1, no cell edges):
// the features of `cells_classic`, mapped to [-1, 1] with the constants the first Worley shipped with.
fn worley_classic(x: vec3f, period: i32, seed: u32, feature: u32) -> f32 {
    let c = cells_classic(x, period, seed, feature == 3u);
    switch feature {
        case 1u: { return clamp(c.y / 1.4 * 2.0 - 1.0, -1.0, 1.0); }
        case 2u: { return clamp((c.y - c.x) * 2.0 - 1.0, -1.0, 1.0); }
        case 3u: { return c.z; }
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

// ---- Math and remap helpers used by generated graph code ------------------------

// Power that never produces NaN: a negative base only takes whole exponents, 0^x = 0 and x^0 = 1.
fn pow_safe(a: f32, b: f32) -> f32 {
    if (b == 0.0) { return 1.0; }
    if (a > 0.0) { return pow(a, b); }
    if (a == 0.0 || fract(b) != 0.0) { return 0.0; }
    let m = pow(-a, b);
    return select(m, -m, fract(b * 0.5) != 0.0); // odd exponent keeps the sign
}
fn pow_safe3(a: vec3f, b: vec3f) -> vec3f { return vec3f(pow_safe(a.x, b.x), pow_safe(a.y, b.y), pow_safe(a.z, b.z)); }

// Blender's polynomial smooth minimum; k <= 0 is a plain min.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * k * (1.0 / 6.0);
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
fn smin3(a: vec3f, b: vec3f, k: f32) -> vec3f { return vec3f(smin(a.x, b.x, k), smin(a.y, b.y, k), smin(a.z, b.z, k)); }
fn smax3(a: vec3f, b: vec3f, k: f32) -> vec3f { return -smin3(-a, -b, k); }

// Linear map of x from [a, b] to [c, d]; an empty input range gives c.
// A curve through the points stored at P[base...] as (x, y) pairs, sorted by x; n points.
// mode: 0 linear, 1 smooth (smoothstep), 2 step, 3 cubic (monotone cubic Hermite, Fritsch-Butland
// slopes, so the curve never overshoots its points).
fn curve_secant(base: u32, i: u32) -> f32 {
    return (P[base + 2u * i + 3u] - P[base + 2u * i + 1u]) / max(P[base + 2u * i + 2u] - P[base + 2u * i], 1e-12);
}

fn curve_slope(base: u32, n: u32, k: u32) -> f32 {
    if (n < 2u) { return 0.0; }
    if (k == 0u) { return curve_secant(base, 0u); }
    if (k == n - 1u) { return curve_secant(base, n - 2u); }
    let d0 = curve_secant(base, k - 1u);
    let d1 = curve_secant(base, k);
    if (d0 * d1 <= 0.0) { return 0.0; }
    let h0 = P[base + 2u * k] - P[base + 2u * (k - 1u)];
    let h1 = P[base + 2u * (k + 1u)] - P[base + 2u * k];
    let w0 = 2.0 * h1 + h0;
    let w1 = h1 + 2.0 * h0;
    return (w0 + w1) / (w0 / d0 + w1 / d1);
}

fn curve_eval(x: f32, base: u32, n: u32, mode: u32) -> f32 {
    if (x <= P[base]) { return P[base + 1u]; }
    for (var i = 1u; i < n; i++) {
        let x1 = P[base + 2u * i];
        if (x < x1) {
            let x0 = P[base + 2u * (i - 1u)];
            let y0 = P[base + 2u * (i - 1u) + 1u];
            let y1 = P[base + 2u * i + 1u];
            if (mode == 2u) { return y0; }
            var t = (x - x0) / max(x1 - x0, 1e-12);
            if (mode == 1u) { t = t * t * (3.0 - 2.0 * t); }
            if (mode == 3u) {
                let h = x1 - x0;
                let t2 = t * t;
                let t3 = t2 * t;
                return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * curve_slope(base, n, i - 1u)
                     + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * h * curve_slope(base, n, i);
            }
            return mix(y0, y1, t);
        }
    }
    return P[base + 2u * (n - 1u) + 1u];
}

// a + (b - a) * t, with t limited to 0..1 when `clamped`.
fn mix_ab(t: f32, a: f32, b: f32, clamped: bool) -> f32 {
    return a + (b - a) * select(t, clamp(t, 0.0, 1.0), clamped);
}

// 1 for x from lo to hi, falling smoothly to 0 over `soft` beyond either end.
fn select_range(x: f32, lo: f32, hi: f32, soft: f32) -> f32 {
    let s = max(soft, 1e-6);
    return smoothstep(lo - s, lo, x) * (1.0 - smoothstep(hi, hi + s, x));
}

fn remap(x: f32, a: f32, b: f32, c: f32, d: f32, clamped: bool) -> f32 {
    let t = select(0.0, (x - a) / (b - a), b != a);
    let r = c + t * (d - c);
    return select(r, clamp(r, min(c, d), max(c, d)), clamped);
}

// Distance to the centre of the grid cell containing q (or, with `corners`, to the nearest cell
// corner), for n cells per tile on each axis, scaled to [-1, 1]: -1 at the reference point, +1 as far
// from it as the grid gets. Tiles when every count is a whole number.
// With `cubic` the offset is the largest one along any axis, in cell units (the cell is a box with sharp corners):
// max(abs(0.5 - fract(q * n))) * 2 - 1.
fn grid_dist(q: vec3f, n_in: vec3f, corners: bool, cubic: bool) -> f32 {
    let n = max(n_in, vec3f(1e-3));
    let f = fract(q * n);
    if (cubic) {
        let e = abs(select(f - 0.5, min(f, 1.0 - f), corners));
        return max(e.x, max(e.y, e.z)) * 4.0 - 1.0;
    }
    let d = select(f - 0.5, min(f, 1.0 - f), corners) / n;
    return length(d) / (0.5 * length(1.0 / n)) * 2.0 - 1.0;
}

// ---- Mesh distance grid ----------------------------------------------------------------

// Distance to a baked mesh (meshsdf.rs): trilinear between texel centres, and beyond the grid the
// nearest texel plus the distance to it. `origin` is the grid's min corner and `voxel` a texel's size.
// Takes the texture as an argument so the preview and the export can bind it in different places.
fn mesh_sdf_at(t: texture_3d<f32>, p: vec3f, origin: vec3f, voxel: f32) -> f32 {
    let dims = vec3i(textureDimensions(t));
    let g = (p - origin) / voxel - 0.5;
    let gc = clamp(g, vec3f(0.0), vec3f(dims) - vec3f(1.0));
    let i0 = vec3i(floor(gc));
    let f = gc - vec3f(i0);
    let i1 = min(i0 + vec3i(1), dims - vec3i(1));
    let c00 = mix(textureLoad(t, vec3i(i0.x, i0.y, i0.z), 0).x, textureLoad(t, vec3i(i1.x, i0.y, i0.z), 0).x, f.x);
    let c10 = mix(textureLoad(t, vec3i(i0.x, i1.y, i0.z), 0).x, textureLoad(t, vec3i(i1.x, i1.y, i0.z), 0).x, f.x);
    let c01 = mix(textureLoad(t, vec3i(i0.x, i0.y, i1.z), 0).x, textureLoad(t, vec3i(i1.x, i0.y, i1.z), 0).x, f.x);
    let c11 = mix(textureLoad(t, vec3i(i0.x, i1.y, i1.z), 0).x, textureLoad(t, vec3i(i1.x, i1.y, i1.z), 0).x, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z) + length(g - gc) * voxel;
}
