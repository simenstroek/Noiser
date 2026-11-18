// 2D slice view: evaluates the graph on one slice of the unit tile, then displays it normalized.

// f32 → u32 that sorts the same way as the float, so atomicMin/Max work on it.
fn ord(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
fn unord(o: u32) -> f32 {
    return bitcast<f32>(select(~o, o & 0x7fffffffu, (o & 0x80000000u) != 0u));
}

// Point in tile space for slice texel coordinates uv in [0, 1].
fn slice_point(uv: vec2f) -> vec3f {
    let s = u.slice_pos;
    switch u.slice_axis {
        case 0u: { return vec3f(s, uv.x, uv.y); }
        case 1u: { return vec3f(uv.x, s, uv.y); }
        default: { return vec3f(uv.x, uv.y, s); }
    }
}

var<workgroup> wg_lo: atomic<u32>;
var<workgroup> wg_hi: atomic<u32>;

// Min/max is reduced per workgroup first, so only one thread in 64 touches the global atomics.
@compute @workgroup_size(8, 8)
fn cs_slice(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&wg_lo, 0xffffffffu);
        atomicStore(&wg_hi, 0u);
    }
    workgroupBarrier();
    let n = u.slice_res;
    if (id.x < n && id.y < n) {
        let v = graph_density(slice_point((vec2f(id.xy) + 0.5) / f32(n)));
        slice.vals[id.y * n + id.x] = v;
        if (v == v) { // skip NaN
            atomicMin(&wg_lo, ord(v));
            atomicMax(&wg_hi, ord(v));
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicMin(&slice.lo, atomicLoad(&wg_lo));
        atomicMax(&slice.hi, atomicLoad(&wg_hi));
    }
}

fn heat(t: f32) -> vec3f {
    // Black → purple → red → orange → near white.
    let c = array<vec3f, 5>(vec3f(0.0, 0.0, 0.08), vec3f(0.33, 0.05, 0.5), vec3f(0.85, 0.15, 0.3), vec3f(1.0, 0.6, 0.1), vec3f(1.0, 1.0, 0.85));
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let i = min(u32(x), 3u);
    return mix(c[i], c[i + 1u], x - f32(i));
}

@fragment
fn fs_slice(in: VsOut) -> @location(0) vec4f {
    // Fit a square into the view; uv runs 0..1 left→right, top→bottom.
    let q = in.uv * vec2f(u.slice_aspect, 1.0) / min(u.slice_aspect, 1.0);
    if (any(abs(q) > vec2f(1.0))) { return vec4f(0.09, 0.09, 0.1, 1.0); }
    let uv = fract(vec2f(q.x, -q.y) * 0.5 * u.slice_tiles + 0.5 * u.slice_tiles);
    let n = u.slice_res;
    let texel = min(vec2u(uv * f32(n)), vec2u(n - 1u));
    if (VECTOR_PREVIEW) {
        return vec4f(graph_color(slice_point((vec2f(texel) + 0.5) / f32(n))), 1.0);
    }
    let v = slice.vals[texel.y * n + texel.x];
    let lo = unord(atomicLoad(&slice.lo));
    let hi = unord(atomicLoad(&slice.hi));
    let t = (v - lo) / max(hi - lo, 1e-9);
    // The target is not sRGB, so these values show as-is: 0.5 is mid grey.
    let c = select(vec3f(t), heat(t), u.slice_mode == 1u);
    return vec4f(c, 1.0);
}
