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

// ---- Cellular (Worley) ------------------------------------------

// (F1, F2, random value of the nearest cell in [-1, 1]); feature points jittered over the whole cell.
// The cell value costs one more hash, so only the cell-value feature asks for it.
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
    let c = cells(x, period, seed, feature == 3u);
    // Typical range of each feature, used to map it to signed [-1, 1].
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
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32, clamped: bool) -> f32 {
    let t = select(0.0, (x - a) / (b - a), b != a);
    let r = c + t * (d - c);
    return select(r, clamp(r, min(c, d), max(c, d)), clamped);
}

// Distance to the centre of the grid cell containing q (or, with `corners`, to the nearest cell
// corner), for n cells per tile on each axis, scaled to [-1, 1]: -1 at the reference point, +1 as far
// from it as the grid gets. Tiles when every count is a whole number.
fn grid_dist(q: vec3f, n_in: vec3f, corners: bool) -> f32 {
    let n = max(n_in, vec3f(1e-3));
    let f = fract(q * n);
    let d = select(f - 0.5, min(f, 1.0 - f), corners) / n;
    return length(d) / (0.5 * length(1.0 / n)) * 2.0 - 1.0;
}

// ---- Patterns ------------------------------------------------------------------------

// Smooth ramp from 0 at lo to 1 at hi. The width has a floor so lo == hi (no bevel) is a hard
// step instead of a NaN from dividing by zero.
fn ramp01(lo: f32, hi: f32, x: f32) -> f32 {
    let t = clamp((x - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// A brick wall. Params at P[s..s+10]: counts (3), steps, gap, bevel, output (0 distance, 1 shape,
// 2 cell value), mode2d, axis, seed (-1 = global). In 2D mode the wall is extruded along `axis`;
// the other two axes, in order, are the row axis and the stacking axis (see PatternParams::axes).
fn bricks(q: vec3f, s: u32) -> f32 {
    let counts = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let steps = max(i32(P[s + 3u]), 1);
    let gap = P[s + 4u];
    let bevel = P[s + 5u];
    let mode = u32(P[s + 6u]);
    let flat2d = P[s + 7u] > 0.5;
    let axis = u32(P[s + 8u]);
    let seed = select(u.seed, u32(P[s + 9u]), P[s + 9u] >= 0.0);

    var r = 0u;  // along a row
    var st = 1u; // rows stack along this axis
    var ly = 2u; // layers (3D), or the extrusion axis (2D)
    if (flat2d && axis == 0u) { r = 1u; st = 2u; ly = 0u; }
    else if (flat2d && axis == 1u) { r = 0u; st = 2u; ly = 1u; }

    // Brick units: one brick is 1 wide in every axis. Row k is shifted along the row by
    // (k mod steps) / steps of a brick.
    var x = q * counts;
    let row = i32(floor(x[st]));
    x[r] = x[r] - f32(((row % steps) + steps) % steps) / f32(steps);
    let cell = floor(x);
    var from_c = fract(x) - 0.5; // offset from the brick centre, in brick units
    var inv = 1.0 / counts;
    if (flat2d) { from_c[ly] = 0.0; inv[ly] = 0.0; }

    if (mode == 2u) {
        // Every brick has one id; wrapping it by the counts makes bricks across the tile edge agree.
        let n = vec3i(max(counts, vec3f(1.0)) + 0.5);
        var id = vec3i(cell);
        if (flat2d) { id[ly] = 0; }
        return hash01(((id % n) + n) % n, seed).x * 2.0 - 1.0;
    }
    if (mode == 1u) {
        let e = 2.0 * max(abs(from_c.x), max(abs(from_c.y), abs(from_c.z))); // 0 at the centre, 1 at the border
        return mix(-1.0, 1.0, ramp01(1.0 - gap - bevel, 1.0 - gap, e));
    }
    // Same scaling as grid_dist: tile-space distance over the covering radius, as -1..1.
    return length(from_c * inv) / (0.5 * length(inv)) * 2.0 - 1.0;
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

// Displacement strength at `p` (mesh space) from the mask terms (meshmask.rs); 1 with no terms.
// `terms` holds two vec4 per term: (kind, weight, lo, hi) and (direction xyz, invert). Every term is
// measured on the mesh's own distance field, at the nearest surface point, so the mask is a property
// of the surface and does not change along the normal. `blo`/`bhi` are the mesh's bounds.
fn mesh_mask(t: texture_3d<f32>, p: vec3f, origin: vec3f, voxel: f32, blo: vec3f, bhi: vec3f, terms: array<vec4f, 12>, count: u32) -> f32 {
    if (count == 0u) { return 1.0; }
    let e = 1.5 * voxel;
    let d0 = mesh_sdf_at(t, p, origin, voxel);
    let xp = mesh_sdf_at(t, p + vec3f(e, 0.0, 0.0), origin, voxel);
    let xm = mesh_sdf_at(t, p - vec3f(e, 0.0, 0.0), origin, voxel);
    let yp = mesh_sdf_at(t, p + vec3f(0.0, e, 0.0), origin, voxel);
    let ym = mesh_sdf_at(t, p - vec3f(0.0, e, 0.0), origin, voxel);
    let zp = mesh_sdf_at(t, p + vec3f(0.0, 0.0, e), origin, voxel);
    let zm = mesh_sdf_at(t, p - vec3f(0.0, 0.0, e), origin, voxel);
    let n = safe_normalize(vec3f(xp - xm, yp - ym, zp - zm));
    let s = p - n * d0; // the nearest surface point (one Newton step on a distance field)
    var sum = 0.0;
    for (var i = 0u; i < count; i++) {
        let a = terms[2u * i];
        let b = terms[2u * i + 1u];
        let kind = u32(a.x);
        if (kind == 1u) { sum += a.y; continue; }
        var raw = 0.0;
        if (kind == 2u) {
            let u01 = (s - blo) / max(bhi - blo, vec3f(1e-6));
            raw = dot(select(u01, vec3f(1.0) - u01, b.xyz < vec3f(0.0)), abs(b.xyz));
        } else if (kind == 3u) {
            raw = dot(n, b.xyz);
        } else if (kind == 4u) {
            // Ambient occlusion from the distance field (IQ): how far the field falls short of the
            // distance moved along the normal, summed over a few steps.
            var occ = 0.0;
            var sca = 1.0;
            for (var k = 1; k <= 5; k++) {
                let h = 0.01 + 0.07 * f32(k);
                occ += (h - mesh_sdf_at(t, s + n * h, origin, voxel)) * sca;
                sca *= 0.85;
            }
            raw = clamp(1.0 - 2.0 * occ, 0.0, 1.0);
        } else if (kind == 5u) {
            // March inward along the normal until the field turns positive again.
            var q = s - n * 0.01;
            var depth = 0.01;
            for (var k = 0; k < 24; k++) {
                let dd = mesh_sdf_at(t, q, origin, voxel);
                if (dd >= 0.0) { break; }
                let step = max(-dd, 0.5 * voxel);
                depth += step;
                q -= n * step;
            }
            raw = depth;
        } else {
            raw = 0.5 * (xp + xm + yp + ym + zp + zm - 6.0 * d0) / (e * e); // mean curvature
        }
        var v = clamp((raw - a.z) / max(a.w - a.z, 1e-6), 0.0, 1.0);
        if (b.w > 0.5) { v = 1.0 - v; }
        sum += a.y * v;
    }
    return sum;
}
