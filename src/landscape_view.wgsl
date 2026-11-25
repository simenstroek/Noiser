// The Landscape tab's 3D view: the height grid (metres, n x n, rows along y) ray marched from an orbiting camera, drawn
// straight into egui's pass for the view's rectangle. A port of the CPU renderer in landscape.rs (`render_steps`), which
// the tests compare it with: box clipping, slope-safe steps, a bisection refine, and the same colours.

struct U {
    eye: vec4f,      // w: tan(half the vertical field of view)
    fwd: vec4f,      // w: width / height of the picture
    right: vec4f,    // w: side of the terrain in metres
    up: vec4f,       // w: metres between samples
    light: vec4f,    // w: steepest slope (metres up per metre across, with margin)
    range: vec4f,    // lowest and highest ground, water level (very low for none), 1 when the target is sRGB
    extra: vec4f,    // samples per side, box floor, box top, smallest step in cells
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var heights: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return VsOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

fn ground(x: f32, y: f32) -> f32 {
    let n = u.extra.x;
    let cell = u.up.w;
    let gx = clamp(x / cell, 0.0, n - 1.0 - 1e-3);
    let gy = clamp(y / cell, 0.0, n - 1.0 - 1e-3);
    let i = i32(gx);
    let j = i32(gy);
    let fx = gx - f32(i);
    let fy = gy - f32(j);
    let h00 = textureLoad(heights, vec2i(i, j), 0).x;
    let h10 = textureLoad(heights, vec2i(i + 1, j), 0).x;
    let h01 = textureLoad(heights, vec2i(i, j + 1), 0).x;
    let h11 = textureLoad(heights, vec2i(i + 1, j + 1), 0).x;
    return (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fy) + (h01 * (1.0 - fx) + h11 * fx) * fy;
}

// Water: a flat surface at its level wherever the ground is below it.
fn surface(x: f32, y: f32) -> f32 {
    return max(ground(x, y), u.range.z);
}

fn to_target(c: vec3f) -> vec4f {
    // The colours are written as sRGB values (as the CPU picture's bytes were); an sRGB target wants them linear.
    if (u.range.w > 0.5) {
        return vec4f(pow((c + 0.055) / 1.055, vec3f(2.4)), 1.0);
    }
    return vec4f(c, 1.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
    let world = u.right.w;
    let cell = u.up.w;
    let lo = u.range.x;
    let hi = u.range.y;
    let level = u.range.z;
    let s = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let c = vec2f(s.x * u.eye.w * u.fwd.w, s.y * u.eye.w);
    let d = normalize(u.fwd.xyz + u.right.xyz * c.x + u.up.xyz * c.y);
    let eye = u.eye.xyz;
    // Sky: lighter at the horizon.
    let k = clamp(d.z * 2.0, 0.0, 1.0);
    let sky = mix(vec3f(150.0, 180.0, 215.0), vec3f(70.0, 110.0, 170.0), k) / 255.0;

    // March only where the ray is inside the terrain's box.
    let box_lo = vec3f(0.0, 0.0, u.extra.y);
    let box_hi = vec3f(world, world, u.extra.z);
    let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
    let a = (box_lo - eye) * inv;
    let b = (box_hi - eye) * inv;
    let t_in = max(max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z)), 0.0);
    let t_out = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    if (t_in > t_out) { return to_target(sky); }

    let min_step = cell * u.extra.w;
    // The surface rises at most `steepest` per metre across, so a ray `gap` above it can go `gap / closing` safely.
    let closing = abs(d.z) + u.light.w * length(d.xy);
    var t = t_in;
    var prev = t_in;
    for (var i = 0; i < 8192; i = i + 1) {
        if (t > t_out) { break; }
        let p = eye + d * t;
        let gap = p.z - surface(p.x, p.y);
        if (gap < 0.0) {
            // Refine between the last two steps.
            var lo_t = prev;
            var hi_t = t;
            for (var r = 0; r < 5; r = r + 1) {
                let m = (lo_t + hi_t) * 0.5;
                let q = eye + d * m;
                if (q.z < surface(q.x, q.y)) { hi_t = m; } else { lo_t = m; }
            }
            let q = eye + d * hi_t;
            let gx = (surface(q.x + cell, q.y) - surface(q.x - cell, q.y)) / (2.0 * cell);
            let gy = (surface(q.x, q.y + cell) - surface(q.x, q.y - cell)) / (2.0 * cell);
            let nrm = normalize(vec3f(-gx, -gy, 1.0));
            let diffuse = max(dot(nrm, u.light.xyz), 0.0);
            let height = clamp((q.z - lo) / max(hi - lo, 1e-3), 0.0, 1.0);
            // Low ground green, higher ground rock, the top snow; steep slopes always rock.
            let green = vec3f(70.0, 105.0, 60.0);
            let rock = vec3f(125.0, 115.0, 100.0);
            let snow = vec3f(235.0, 238.0, 245.0);
            let low = mix(green, rock, clamp((height - 0.25) * 3.0, 0.0, 1.0));
            var base = mix(low, snow, clamp((height - 0.78) * 8.0, 0.0, 1.0));
            base = mix(base, rock, clamp((1.0 - nrm.z) * 3.5 - 0.35, 0.0, 1.0) * (1.0 - clamp((height - 0.78) * 8.0, 0.0, 0.6)));
            let shade = 0.25 + 0.85 * diffuse;
            if (ground(q.x, q.y) < level) { base = vec3f(45.0, 95.0, 165.0); }
            return to_target(min(base * shade, vec3f(255.0)) / 255.0);
        }
        prev = t;
        t = t + max(gap / max(closing, 1e-6), min_step);
    }
    return to_target(sky);
}
