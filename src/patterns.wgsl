// Pattern node kernels (see PatternParams). Emitted into a graph's shader only when it has a Pattern
// node: compiling them for every graph added about 0.2 s to each pipeline build. Needs noise.wgsl
// (hash01, and `u.seed` and `P` from the including shader).

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

// Pattern params are at P[s..s+19]: counts (3), steps, gap, bevel, output, 2D, axis, seed (-1 = global),
// variant, radius, profile, direction (3), width, softness, iterations. See PatternParams.

// A checkerboard. counts are cells per tile (one period of the wave spans two cells, so tiling needs
// even counts); softness widens the edges between cells. In 2D the pattern is extruded along `axis`.
fn checker(q: vec3f, s: u32) -> f32 {
    let counts = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let flat2d = P[s + 7u] > 0.5;
    let axis = u32(P[s + 8u]);
    let soft = max(P[s + 17u], 1e-4);
    let f = fract(q * counts * 0.5);
    // A triangle wave: 0 at the cell borders, +1 and -1 at the cell centres.
    let tri = 4.0 * abs(fract(f - vec3f(0.25)) - vec3f(0.5)) - vec3f(1.0);
    var w = clamp(tri / soft, vec3f(-1.0), vec3f(1.0));
    if (flat2d) { w[axis] = 1.0; }
    return w.x * w.y * w.z;
}

// Parallel bands: phase = dot(q, dir) * count (tiles when both are whole). Waves: 0 sine, 1 triangle,
// 2 sawtooth, 3 square (`width` is the share of each period that is +1, softened by `softness`).
fn stripes(q: vec3f, s: u32) -> f32 {
    let count = P[s];
    let wave = u32(P[s + 10u]);
    let dir = vec3f(P[s + 13u], P[s + 14u], P[s + 15u]);
    let width = P[s + 16u];
    let soft = max(P[s + 17u], 1e-4);
    let f = fract(dot(q, dir) * count);
    switch wave {
        case 1u: { return 1.0 - 4.0 * abs(f - 0.5); }
        case 2u: { return 2.0 * f - 1.0; }
        case 3u: {
            let d = abs(f - 0.5); // 0 at the middle of the period
            return 1.0 - 2.0 * ramp01(0.5 * width - 0.25 * soft, 0.5 * width + 0.25 * soft, d);
        }
        default: { return sin(6.2831853 * f); }
    }
}

// Round dimples on a lattice. Params: counts (cubic cells per tile), variant (0 cubic, 1 body-centred,
// 2 face-centred), output (0 distance, 1 dimple profile), radius, profile (0 sphere cap, 1 cone,
// 2 smooth bowl, 3 flat disc). The nearest lattice point is searched among the 27 surrounding cells.
fn dimples(q: vec3f, s: u32) -> f32 {
    let n = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let lattice = u32(P[s + 10u]);
    let mode = u32(P[s + 6u]);
    let radius = max(P[s + 11u], 1e-3);
    let profile = u32(P[s + 12u]);
    let base = vec3i(floor(q * n));
    var points = array<vec3f, 4>(vec3f(0.5), vec3f(0.0), vec3f(0.0, 0.5, 0.5), vec3f(0.5, 0.0, 0.5));
    var count = 1u;
    if (lattice == 1u) { points[0] = vec3f(0.0); points[1] = vec3f(0.5); count = 2u; }
    if (lattice == 2u) { points[0] = vec3f(0.0); points[1] = vec3f(0.0, 0.5, 0.5); points[2] = vec3f(0.5, 0.0, 0.5); points[3] = vec3f(0.5, 0.5, 0.0); count = 4u; }
    var best = 1e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let cell = vec3f(base + vec3i(x, y, z));
                for (var k = 0u; k < count; k++) {
                    let d = q - (cell + points[k]) / n;
                    best = min(best, dot(d, d));
                }
            }
        }
    }
    let dist = sqrt(best);
    let inv = 1.0 / n;
    let edge = min(inv.x, min(inv.y, inv.z));
    if (mode == 1u) {
        // How much room each dimple has: half the distance to its nearest neighbour.
        var nearest = edge;
        if (lattice == 1u) { nearest = 0.8660254 * edge; }
        if (lattice == 2u) { nearest = 0.7071068 * edge; }
        let t = dist / (radius * 0.5 * nearest);
        var depth = 0.0;
        if (t < 1.0) {
            switch profile {
                case 1u: { depth = 1.0 - t; }
                case 2u: { depth = 1.0 - ramp01(0.0, 1.0, t); }
                case 3u: { depth = 1.0; }
                default: { depth = sqrt(1.0 - t * t); }
            }
        }
        return 1.0 - 2.0 * depth;
    }
    // Distance over how far the lattice's farthest point is from its nearest point (the covering radius).
    var cover = 0.5 * length(inv);
    let isotropic = length(inv) * 0.57735027; // the cubic edge of a lattice with the same cell volume
    if (lattice == 1u) { cover = 0.5590170 * isotropic; }
    if (lattice == 2u) { cover = 0.5 * isotropic; }
    return clamp(dist / cover * 2.0 - 1.0, -1.0, 1.0);
}

// Triply periodic minimal surfaces: variant 0 gyroid, 1 Schwarz P, 2 diamond. `counts` are periods per
// tile per axis (whole numbers tile). Output 0 is the field, scaled by its largest value to [-1, 1];
// output 1 is a thin shell around its zero level (`width` is the thickness): -1 in the sheet, +1 away.
fn gyroid(q: vec3f, s: u32) -> f32 {
    let n = vec3f(P[s], P[s + 1u], P[s + 2u]);
    let variant = u32(P[s + 10u]);
    let mode = u32(P[s + 6u]);
    let width = clamp(P[s + 16u], 0.0, 0.999);
    let a = 6.2831853 * q * n;
    let sn = sin(a);
    let cs = cos(a);
    var f = sn.x * cs.y + sn.y * cs.z + sn.z * cs.x; // gyroid
    var peak = GYROID_PEAK;
    if (variant == 1u) { f = cs.x + cs.y + cs.z; peak = 3.0; }
    if (variant == 2u) { f = sn.x * sn.y * sn.z + sn.x * cs.y * cs.z + cs.x * sn.y * cs.z + cs.x * cs.y * sn.z; peak = DIAMOND_PEAK; }
    let field = clamp(f / peak, -1.0, 1.0);
    if (mode == 1u) { return clamp((abs(field) - width) / (1.0 - width) * 2.0 - 1.0, -1.0, 1.0); }
    return field;
}
const GYROID_PEAK: f32 = 1.5;
const DIAMOND_PEAK: f32 = 1.4142136;

// Menger sponge (IQ) in the cube [-1, 1]³: negative inside the solid, positive in the holes.
fn menger(p: vec3f, iterations: u32) -> f32 {
    let q = abs(p) - vec3f(1.0);
    var d = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0); // the cube
    var s = 1.0;
    for (var m = 0u; m < iterations; m++) {
        let a = p * s - 2.0 * floor(p * s / 2.0) - vec3f(1.0); // mod(p * s, 2) - 1
        s *= 3.0;
        let r = abs(vec3f(1.0) - 3.0 * abs(a));
        let hole = (min(max(r.x, r.y), min(max(r.y, r.z), max(r.z, r.x))) - 1.0) / s;
        d = max(d, hole);
    }
    return d;
}

// Sierpinski tetrahedron by tetrahedral folding (Syntopia): each iteration folds the point into the
// region of one vertex of the tetrahedron with vertices (1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1) and
// scales by 2 about that vertex; what is left is tested against the tetrahedron itself. The distance
// is in the folded space, so edges stay crisp at any depth.
fn sierpinski(p_in: vec3f, iterations: u32) -> f32 {
    var z = p_in;
    for (var n = 0u; n < iterations; n++) {
        if (z.x + z.y < 0.0) { z = vec3f(-z.y, -z.x, z.z); }
        if (z.x + z.z < 0.0) { z = vec3f(-z.z, z.y, -z.x); }
        if (z.y + z.z < 0.0) { z = vec3f(z.x, -z.z, -z.y); }
        z = z * 2.0 - vec3f(1.0);
    }
    let faces = max(max(dot(z, vec3f(-1.0, -1.0, -1.0)), dot(z, vec3f(1.0, 1.0, -1.0))), max(dot(z, vec3f(-1.0, 1.0, 1.0)), dot(z, vec3f(1.0, -1.0, 1.0))));
    return (faces - 1.0) * 0.57735027;
}

// A repeating fractal in each cell: variant 0 Menger sponge, 1 Sierpinski tetrahedron; counts are
// cells per tile (whole numbers tile), iterations how deep it goes. Output 1 is the shape (-1 solid,
// +1 empty) from a distance estimate scaled to the cell; output 0 is that estimate unclamped by
// the shape threshold, i.e. the signed distance in units of the cell.
fn fractal(q: vec3f, s: u32) -> f32 {
    let n = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let variant = u32(P[s + 10u]);
    let iterations = min(u32(max(P[s + 18u], 1.0)), 6u);
    let p = (fract(q * n) - vec3f(0.5)) * 2.0; // the cell as [-1, 1]³
    var d = menger(p, iterations);
    // Features are 3^-iterations across, so scale the estimate to keep edges crisp at any depth.
    var scale = pow(3.0, f32(iterations));
    if (variant == 1u) { d = sierpinski(p, iterations); scale = 3.0; }
    return clamp(d * scale, -1.0, 1.0);
}

// 3D Truchet pipes. Each cell joins its six face centres in three pairs of neighbouring faces, by
// quarter circles of radius 1/2 around the edge shared by the two faces; one of the 8 possible
// pairings is picked per cell. Every face centre gets exactly one pipe in each of the two cells
// sharing it, and both meet the face at a right angle, so the pipes continue across cells.
// Faces: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z.
fn truchet_pipes(x: vec3f, counts: vec3f, width: f32, soft: f32, seed: u32) -> f32 {
    var pairs = array<u32, 48>(0u,2u,1u,4u,3u,5u, 0u,2u,1u,5u,3u,4u, 0u,3u,1u,4u,2u,5u, 0u,3u,1u,5u,2u,4u,
                               0u,4u,1u,2u,3u,5u, 0u,4u,1u,3u,2u,5u, 0u,5u,1u,2u,3u,4u, 0u,5u,1u,3u,2u,4u);
    let n = vec3i(max(counts, vec3f(1.0)) + 0.5);
    let h = hash01(((vec3i(floor(x)) % n) + n) % n, seed).x;
    let m = min(u32(h * 8.0), 7u) * 6u;
    let p = fract(x) - vec3f(0.5); // from the cell centre, in cell units
    var d = 1e9;
    for (var k = 0u; k < 3u; k++) {
        let fa = pairs[m + 2u * k];
        let fb = pairs[m + 2u * k + 1u];
        var da = vec3f(0.0);
        var db = vec3f(0.0);
        da[fa / 2u] = select(-1.0, 1.0, (fa & 1u) == 0u);
        db[fb / 2u] = select(-1.0, 1.0, (fb & 1u) == 0u);
        let v = p - 0.5 * (da + db);
        let a = dot(v, da);
        let b = dot(v, db);
        let w = v - a * da - b * db;
        let e = length(vec2f(a, b)) - 0.5;
        d = min(d, sqrt(e * e + dot(w, w)));
    }
    let half_width = 0.5 * width;
    return -1.0 + 2.0 * ramp01(half_width * (1.0 - soft), half_width * (1.0 + soft), d);
}

// Truchet tiles in the plane of two axes (extruded along the third in 2D mode): each cell has one
// of two orientations at random, and the lines join up across cell borders. Variant 0 draws quarter
// circles of radius 1/2 around two opposite corners, 1 draws a diagonal. `width` is the line's
// width as a share of a cell, `softness` how soft its edges are. -1 on the line, +1 away from it.
fn truchet(q: vec3f, s: u32) -> f32 {
    let counts = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let flat2d = P[s + 7u] > 0.5;
    let axis = u32(P[s + 8u]);
    let variant = u32(P[s + 10u]);
    let width = P[s + 16u];
    let soft = P[s + 17u];
    let seed = select(u.seed, u32(P[s + 9u]), P[s + 9u] >= 0.0);
    if (variant == 2u) { return truchet_pipes(q * counts, counts, width, soft, seed); }
    var r = 0u;
    var st = 1u;
    var ly = 2u;
    if (flat2d && axis == 0u) { r = 1u; st = 2u; ly = 0u; }
    else if (flat2d && axis == 1u) { r = 0u; st = 2u; ly = 1u; }
    let x = q * counts;
    let cell = floor(x);
    var id = vec3i(cell);
    if (flat2d) { id[ly] = 0; }
    let n = vec3i(max(counts, vec3f(1.0)) + 0.5);
    let flip = hash01(((id % n) + n) % n, seed).x > 0.5;
    var uv = vec2f(fract(x[r]), fract(x[st]));
    if (flip) { uv.x = 1.0 - uv.x; }
    var d = 0.0;
    if (variant == 1u) {
        d = abs(uv.x - uv.y) * 0.70710678; // distance to the diagonal
    } else {
        d = min(abs(length(uv) - 0.5), abs(length(uv - vec2f(1.0)) - 0.5)); // to the nearer quarter circle
    }
    let half_width = 0.5 * width;
    return -1.0 + 2.0 * ramp01(half_width * (1.0 - soft), half_width * (1.0 + soft), d);
}

// Triangles (honeycomb layout 3): the plane of the two axes left by `axis`, extruded along it.
// In units where the row axis is 2 per count and the stacking axis 1 per count, the three line
// families are Y = whole, X + Y = even and X - Y = even, which cut the plane into right isosceles
// triangles of base 2 and height 1 (inradius 1/(1 + sqrt 2)); with counts rows = 1.155 x columns
// they are equilateral. Whole counts tile when the number of rows is even.
fn triangles(q: vec3f, n: vec3f, gap: f32, bevel: f32, mode: u32, axis: u32, seed: u32) -> f32 {
    var r = 0u;
    var st = 1u;
    if (axis == 0u) { r = 1u; st = 2u; }
    else if (axis == 1u) { r = 0u; st = 2u; }
    let X = q[r] * n[r] * 2.0;
    let Y = q[st] * n[st];
    let a = (X + Y) * 0.5;
    let b = (X - Y) * 0.5;
    let dy = abs(Y - round(Y));
    let da = abs(a - round(a)) * 1.41421356;
    let db = abs(b - round(b)) * 1.41421356;
    let depth = clamp(min(dy, min(da, db)) * 2.41421356, 0.0, 1.0); // 0 on a wall, 1 at the incentre
    if (mode == 2u) {
        let rows = max(i32(n[st] + 0.5), 1);
        let cols = max(i32(n[r] + 0.5) * 2, 1);
        let ia = i32(floor(a));
        let ib = i32(floor(b));
        let j = i32(floor(Y));
        let id = vec3i(((j % rows) + rows) % rows, (((ia + ib) % cols) + cols) % cols, ia - ib - j); // row, column, up or down
        return hash01(id, seed).x * 2.0 - 1.0;
    }
    if (mode == 1u) { return 1.0 - 2.0 * ramp01(gap, gap + bevel, depth); }
    return depth * 2.0 - 1.0;
}

// The cells of a lattice. variant 0 hexagons (the plane of the two axes left by `axis`, extruded along
// it), 1 the cells of a body-centred lattice, 2 of a face-centred one (the points are the ones in
// `dimples`; in 2D the second point is (1/2, 1/2) in the plane). The wall distance is exact: find the
// nearest lattice point, then the distance to the bisector plane of each other point near it.
// Outputs: 0 distance (-1 at a wall, +1 at a cell's centre), 1 shape (+1 on the wall, -1 on the
// cell; `gap` and `bevel` are shares of the cell's radius), 2 a flat random value per cell.
fn honeycomb(q: vec3f, s: u32) -> f32 {
    let n = max(vec3f(P[s], P[s + 1u], P[s + 2u]), vec3f(1e-3));
    let gap = P[s + 4u];
    let bevel = P[s + 5u];
    let mode = u32(P[s + 6u]);
    let flat2d = P[s + 7u] > 0.5;
    let axis = u32(P[s + 8u]);
    let seed = select(u.seed, u32(P[s + 9u]), P[s + 9u] >= 0.0);
    let lattice = u32(P[s + 10u]);
    if (lattice == 3u) { return triangles(q, n, gap, bevel, mode, axis, seed); }
    var ly = 2u; // the extruded axis in 2D
    if (flat2d && axis == 0u) { ly = 0u; }
    else if (flat2d && axis == 1u) { ly = 1u; }
    var basis = array<vec3f, 4>(vec3f(0.0), vec3f(0.5), vec3f(0.0), vec3f(0.0));
    var count = 2u;
    if (flat2d) { basis[1][ly] = 0.0; }
    if (lattice == 2u) { basis[1] = vec3f(0.0, 0.5, 0.5); basis[2] = vec3f(0.5, 0.0, 0.5); basis[3] = vec3f(0.5, 0.5, 0.0); count = 4u; }
    var x = q * n;
    if (flat2d) { x[ly] = 0.0; }
    let base = vec3i(floor(x));
    // Pass 1: the nearest lattice point.
    var best = 1e9;
    var best_cell = vec3f(0.0);
    var best_k = 0u;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var w = -1; w <= 1; w++) {
                let o = vec3i(w, y, z);
                if (flat2d && o[ly] != 0) { continue; }
                let cell = vec3f(base + o);
                for (var k = 0u; k < count; k++) {
                    let d = (x - (cell + basis[k])) / n;
                    let d2 = dot(d, d);
                    if (d2 < best) { best = d2; best_cell = cell; best_k = k; }
                }
            }
        }
    }
    let a = (best_cell + basis[best_k]) / n;
    let p = x / n;
    // Pass 2: the nearest wall, which is the bisector of the nearest point and one of its neighbours.
    var wall = 1e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var w = -1; w <= 1; w++) {
                let o = vec3i(w, y, z);
                if (flat2d && o[ly] != 0) { continue; }
                let cell = vec3f(base + o);
                for (var k = 0u; k < count; k++) {
                    let b = (cell + basis[k]) / n;
                    let ab = b - a;
                    if (dot(ab, ab) < 1e-12) { continue; }
                    wall = min(wall, dot(0.5 * (a + b) - p, normalize(ab)));
                }
            }
        }
    }
    // A cell's radius is half the distance to its nearest neighbour.
    let inv = 1.0 / n;
    var nearest = 1e9;
    for (var c = 0u; c < 3u; c++) { if (!(flat2d && c == ly)) { nearest = min(nearest, inv[c]); } }
    for (var k = 1u; k < count; k++) { nearest = min(nearest, length(basis[k] * inv)); }
    let radius = 0.5 * nearest;
    let depth = clamp(wall / radius, 0.0, 1.0); // 0 on a wall, 1 at a centre
    if (mode == 2u) {
        let cells = vec3i(max(n, vec3f(1.0)) + 0.5);
        var id = vec3i(best_cell);
        if (flat2d) { id[ly] = 0; }
        return hash01(((id % cells) + cells) % cells, seed + best_k).x * 2.0 - 1.0;
    }
    if (mode == 1u) { return 1.0 - 2.0 * ramp01(gap, gap + bevel, depth); }
    return depth * 2.0 - 1.0;
}
