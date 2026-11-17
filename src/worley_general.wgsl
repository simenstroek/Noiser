// The general Worley: any metric, any jitter, and the exact cell-edge distance. Emitted into a graph's
// shader only when a Worley node needs it (otherwise every graph paid about 0.15 s of compile time
// for a second 27-cell loop). Needs noise.wgsl (cells_classic, worley_classic, hash01).

// How close a point is, for ranking feature points: squared Euclidean distance (compared before the
// square root), Manhattan distance, or Chebyshev distance.
fn measure(d: vec3f, metric: u32) -> f32 {
    let a = abs(d);
    if (metric == 1u) { return a.x + a.y + a.z; }
    if (metric == 2u) { return max(a.x, max(a.y, a.z)); }
    return dot(d, d);
}

// A cell's feature point, relative to the cell's corner: `jitter` 1 puts it anywhere in the cell,
// 0 in the middle.
fn feature_point(c: vec3i, seed: u32, jitter: f32) -> vec3f {
    return mix(vec3f(0.5), hash01(c, seed), jitter);
}

// (F1, F2, random value of the nearest cell in [-1, 1]); the cell value costs one more hash, so only
// the cell-value feature asks for it.
fn cells(x: vec3f, period: i32, seed: u32, want_value: bool, metric: u32, jitter: f32) -> vec3f {
    let i = wrap_cell(vec3i(floor(x)), period);
    let f = fract(x);
    var f1 = 1e9; // squared distances (Euclidean) until the end
    var f2 = 1e9;
    var nearest = i;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let o = vec3i(xx, y, z);
                let c = wrap_near(i + o, period);
                let d = vec3f(o) + feature_point(c, seed, jitter) - f;
                let d2 = measure(d, metric);
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
    let euclidean = metric == 0u;
    return vec3f(select(f1, sqrt(f1), euclidean), select(f2, sqrt(f2), euclidean), val);
}

// The exact distance to the nearest cell wall (Euclidean): find the nearest feature point, then the
// bisector with each point around it.
fn cell_wall_distance(x: vec3f, period: i32, seed: u32, jitter: f32) -> f32 {
    let i = wrap_cell(vec3i(floor(x)), period);
    let f = fract(x);
    var best = 1e9;
    var a = vec3f(0.0);
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let o = vec3i(xx, y, z);
                let p = vec3f(o) + feature_point(wrap_near(i + o, period), seed, jitter);
                let d = p - f;
                let d2 = dot(d, d);
                if (d2 < best) { best = d2; a = p; }
            }
        }
    }
    var wall = 1e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let o = vec3i(xx, y, z);
                let b = vec3f(o) + feature_point(wrap_near(i + o, period), seed, jitter);
                let ab = b - a;
                if (dot(ab, ab) < 1e-12) { continue; }
                wall = min(wall, dot(0.5 * (a + b) - f, normalize(ab)));
            }
        }
    }
    return wall;
}

// The largest value each feature typically takes, for mapping it to [-1, 1]: (F1, F2, F2 - F1, wall
// distance). Measured on the GPU (gpu_tests.rs, measure_worley_ranges) for a perfect lattice (jitter 0)
// and random points (jitter 1) and interpolated in between.
fn worley_scale(metric: u32, jitter: f32) -> vec4f {
    // Euclidean with random points keeps the constants the first Worley node shipped with (F2's 1.4 is a
    // little generous), so existing graphs look the same.
    var lattice = vec4f(0.866, 1.0, 0.95, 0.5);
    var random = vec4f(1.0, 1.4, 1.0, 0.5);
    if (metric == 1u) { lattice = vec4f(1.45, 1.5, 0.95, 0.5); random = vec4f(1.5, 1.65, 1.2, 0.5); }
    if (metric == 2u) { lattice = vec4f(0.5, 0.95, 0.95, 0.5); random = vec4f(0.81, 0.91, 0.66, 0.5); }
    return mix(lattice, random, jitter);
}

fn worley(x: vec3f, period: i32, seed: u32, feature: u32, metric: u32, jitter: f32) -> f32 {
    let k = worley_scale(metric, jitter);
    if (feature == 4u) { return clamp(cell_wall_distance(x, period, seed, jitter) / k.w * 2.0 - 1.0, -1.0, 1.0); }
    if (metric == 0u && jitter >= 1.0 && feature != 4u) { return worley_classic(x, period, seed, feature); }
    let c = cells(x, period, seed, feature == 3u, metric, jitter);
    switch feature {
        case 1u: { return clamp(c.y / k.y * 2.0 - 1.0, -1.0, 1.0); }
        case 2u: { return clamp((c.y - c.x) / k.z * 2.0 - 1.0, -1.0, 1.0); }
        case 3u: { return c.z; }
        default: { return clamp(c.x / k.x * 2.0 - 1.0, -1.0, 1.0); }
    }
}

