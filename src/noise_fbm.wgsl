// Template, not WGSL on its own: codegen.rs instantiates it once per noise kind used in a graph,
// replacing KIND, HAS_OCTAVES and BASE (the kind's noise at x for octave seed `seed + o`).
// A noise node with Blender's Noise Texture parameters. Params at P[s..s+8]:
// scale, normalize, octaves, lacunarity, roughness, distortion, seed (-1 = global), worley feature,
// worley metric, worley jitter (P[s..s+10]).
// Every BASE stays within [-1, 1]; Graph::density_range relies on it.
fn noise_kKIND(q: vec3f, s: u32) -> f32 {
    let scale = pp(s);
    let normalize = pp(s + 1u) > 0.5;
    let octaves = select(1u, u32(pp(s + 2u)), HAS_OCTAVES);
    let lacunarity = pp(s + 3u);
    let roughness = pp(s + 4u);
    let distortion = pp(s + 5u);
    let seed = select(u.seed, u32(pp(s + 6u)), pp(s + 6u) >= 0.0);
    let feature = u32(pp(s + 7u));
    let metric = u32(pp(s + 8u));
    let jitter = pp(s + 9u);

    // Whole-number scale and lacunarity tile; anything else samples the unwrapped lattice.
    let tiles = fract(scale) == 0.0 && (octaves <= 1u || fract(lacunarity) == 0.0);
    var period = select(0, i32(scale), tiles);
    var x = q * scale;
    if (distortion != 0.0) {
        var d = vec3f(0.0);
        for (var c = 0u; c < 3u; c++) { d[c] = perlin(x, period, seed + 1u + c); }
        x += d * distortion;
    }
    var sum = 0.0;
    var amp = 1.0;
    var total = 0.0;
    for (var o = 0u; o < max(octaves, 1u); o++) {
        sum += amp * BASE;
        total += amp;
        amp *= roughness;
        x *= lacunarity;
        period *= i32(lacunarity);
    }
    let v = sum / max(total, 1e-6); // signed, roughly [-1, 1]
    return select(v, v * 0.5 + 0.5, normalize);
}
