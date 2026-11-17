# `.noiser` math-function format, version 1

A `.noiser` file describes a Noiser graph as a function `density(p)` of a 3D point, so
other software can evaluate the same noise natively at any resolution. Each host gets a
**sampler**: a plugin that reads this file and implements the ops below in the host's own
language (Blender node groups, Houdini VEX, HLSL/GLSL, ...). Where a host or an op can't be
implemented, the fallback is a baked texture (Noiser's VDB or slice-grid export).

The reference implementation is Noiser's WGSL (`src/noise.wgsl`, `src/noise_fbm.wgsl`,
`src/codegen.rs`). Every file carries test vectors, and a sampler is correct when it reproduces
them (see [Test vectors](#test-vectors)).

## File structure

JSON, UTF-8.

```json
{
  "format": "noiser",
  "version": 1,
  "ops": [ { "id": 0, "type": "vector", "op": "position" }, ... ],
  "outputs": { "density": 4, "normal": null },
  "tests": { "points": [[x, y, z], ...], "density": [d, ...] }
}
```

- `ops` is in evaluation order: every op's inputs appear before it.
- An input that references another op is `{"node": id}` (Math operands) or a bare id
  (`position`, `field`). `null` means unconnected.
- `outputs.density` is the op whose value is the density. `null` means density is `0`.
- `outputs.normal` (optional output) is the op for the Normal output. Samplers may ignore it.
- Readers must reject files whose `version` they don't know.

### Space

`p` is in **noise space**: one tile is the unit cube, so a tiling graph repeats with period 1
on every axis. Samplers map their own coordinates (object space, UVW, ...) into noise space.

## Types

Every op has a `type`: `"float"` or `"vector"` (3 floats). Mixing follows Blender's implicit
conversions:

- float → vector: `(f, f, f)`
- vector → float: the average, `(x + y + z) / 3`

## Ops

### `position`
`p`. Type vector.

### `value`
A constant: `value` is a number (float) or `[x, y, z]` (vector).

### `math`
`fn` is one of `add`, `subtract`, `multiply`, `divide`, `min`, `max`. Operands `a` and `b` are
either `{"node": id}` or a number. The result is a vector if either operand is a vector, and
then both operands are converted to vectors and the op applies per component.

- `divide` is Blender's safe divide: `x / 0 = 0` (per component).

### `noise`
Fields: `kind`, `position` (op id or `null` for `p`), `scale`, `normalize`, `octaves`,
`lacunarity`, `roughness`, `distortion`, `seed` (already resolved; a non-negative integer), and `feature` for Worley (`f1`, `f2`, `f2_minus_f1`, `cell_value`; Worley is fractal when `octaves` > 1). Type float.

`kind` is one of `perlin`, `simplex`, `worley`, `white`, `blue`.
Version 1 specifies `perlin` exactly below. The others follow the same lattice, hash and fBm
rules; until each gets its own section, `src/noise.wgsl` is their definition.

### `normal`
The normalized gradient of the float field `field`, negated if `flip`. Type vector. Tetrahedral
central differences with `e = 1e-3`:
`g = Σ k_i · field(p + k_i·e)` over `k = (1,−1,−1), (−1,−1,1), (−1,1,−1), (1,1,1)`, then
`g / max(|g|, 1e-12)`. `field = null` gives `(0, 0, 0)`.

## The hash

All randomness comes from **pcg3d** (Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT
2020): plain 32-bit unsigned integer arithmetic, so any host with u32 add, multiply, xor and shift
can reproduce it bit for bit.

```c
uint3 pcg3d(uint3 v) {                 /* every operation wraps mod 2^32 */
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;  v.y += v.z * v.x;  v.z += v.x * v.y;
  v ^= v >> 16;                        /* logical shift */
  v.x += v.y * v.z;  v.y += v.z * v.x;  v.z += v.x * v.y;
  return v;
}
```

The per-cell random vector for integer cell `c` (int32, reinterpreted as u32) and seed `s` (u32) is

```c
float3 hash3(int3 c, uint s) {
  uint3 h = pcg3d(uint3(c.x + s * 0x9E3779B9u, c.y, c.z));   /* wraps */
  return float3(h) / 4294967295.0;     /* float(u32) rounds to nearest f32; divide in f32 */
}
```

Results are in [0, 1]. Cells are always exact integers (the lattice is wrapped and stepped as
integers), so there are no float-to-bit cast concerns.

Host notes:
- **Blender (Geometry Nodes, 5.3)**: Integer Math add/multiply wrap like u32, and Bit Math `SHIFT`
  with a negative `Shift` is a logical right shift. Blender ints are int32, which carry the same bits.
  `float(u32)` is built from the two 16-bit halves (`hi * 65536 + lo`, one rounding). Shader node
  trees have no Bit Math node, so shaders need the texture-bake fallback.
- Hosts without integer ops in their graph language (some shader editors) use the bake fallback.

## Lattice and tiling

Noises work in lattice units `x = q · scale`, where `q` is the noise's position input.

A noise **tiles** when `scale` is a whole number and, if `octaves > 1`, `lacunarity` is too.
Then each octave has a `period` (cells per tile): `scale` for octave 0, times `lacunarity` for
each later octave. Otherwise `period = 0` (no wrapping).

Wrapping a cell coordinate: `wrap(c) = c − period · floor(c / period)` per component when
`period > 0`, else `c`.

## Perlin

```
perlin(x, period, seed):
  i = wrap(floor(x))
  f = x − floor(x)
  w = f³ (f (6f − 15) + 10)                      (per component)
  for each corner o in {0,1}³:
    c = wrap(i + o)
    g = normalize(hash3(c, seed) · 2 − 1)        (normalize: v / max(|v|, 1e-12))
    n_o = dot(g, f − o)
  trilinear blend of n_o with weights w, where mix(a, b, t) = a·(1 − t) + b·t,
  x first (pairs differing in o.x), then y, then z
  return clamp(1.5 · result, −1, 1)
```

## fBm wrapper (every noise kind)

```
noise(q):
  x = q · scale
  period = scale if the noise tiles, else 0
  if distortion ≠ 0:
    x += (perlin(x, period, seed+1), perlin(x, period, seed+2), perlin(x, period, seed+3)) · distortion
  sum = 0, amp = 1, total = 0
  for o in 0 .. max(octaves, 1) − 1:          (octaves is 1 for kinds without octaves)
    sum += amp · BASE(x, period, seed + o)
    total += amp
    amp *= roughness
    x *= lacunarity
    period *= lacunarity                       (stays 0 when not tiling)
  v = sum / max(total, 1e-6)
  return normalize ? v · 0.5 + 0.5 : v
```

`BASE` for `perlin` is `perlin(...)` above. Kinds without octaves: `white`, `blue`. `worley` has
octaves only when fractal; the file then stores `octaves` as 1.

## Test vectors

`tests.points` is an explicit list of points and `tests.density` is Noiser's GPU result at each. The
list covers one tile's 6×6×6 grid centres, exact seam and lattice coordinates (combinations of
−1, 0, 0.5 and 1, so the tile edges and their neighbours are hit), and scattered points over
[−2, 3]³ (negative and above-one coordinates, as object-space samplers produce). A sampler passes when
every value is within **1e-4** of the vector. Float rounding differs between hosts; the hash
itself is exact, so larger differences mean a real mismatch.
