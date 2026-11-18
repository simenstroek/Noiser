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
`fn` is one of `add`, `subtract`, `multiply`, `divide`, `min`, `max`, `power`, `smooth_min`,
`smooth_max`, `dot`, `absolute`, `ridge`, `floor`, `fract`, `sine`, `sqrt`, `modulo`, `greater`. Operands `a` and `b` are
either `{"node": id}` or a number. The result is a vector if either operand is a vector, and
then both operands are converted to vectors and the op applies per component.

- `divide` is Blender's safe divide: `x / 0 = 0` (per component).
- `power` never gives NaN: `x^0 = 1`; for `x > 0` it is `pow(x, y)`; `0^y = 0`; a negative base takes only
  whole exponents (sign kept for odd ones), otherwise 0.
- `smooth_min` has a third operand `k` (a number or `{"node": id}`, always a float). For `k <= 0` it is `min`;
  otherwise `h = max(k − |a − b|, 0) / k` and the result is `min(a, b) − h³·k/6` (Blender's polynomial
  smooth minimum). `smooth_max(a, b, k) = −smooth_min(−a, −b, k)`.
- `dot` converts both operands to vectors and returns the float `a·b`. Its `type` is `float`.
- `absolute` and `ridge` are unary (`b` is ignored): `|a|` and `1 − |a|`, per component.
- `floor`, `fract`, `sine` and `sqrt` are unary, per component: `floor(a)`, `a − floor(a)`, `sin(a)` (radians) and
  `sqrt(max(a, 0))`.
- `modulo` is `a − b·floor(a / b)` (the sign of `b`), 0 where `b` is 0. `greater` is 1 where `a > b`, else 0.

### `remap`
`value`, `from_min`, `from_max`, `to_min`, `to_max` (each a number or `{"node": id}`, converted to float),
and `clamp`. With `t = (value − from_min) / (from_max − from_min)` (0 when the two are equal), the
result is `to_min + t·(to_max − to_min)`, clamped between `min(to_min, to_max)` and `max(to_min, to_max)`
when `clamp` is true. Type float.

### `separate`
`vector` (op id or `null`, converted to a vector; `null` is the zero vector) and `component` (`x`, `y` or `z`).
The result is that component of the vector. Type float.

### `combine`
`x`, `y` and `z` (each a number or `{"node": id}`, converted to float). The result is the vector `(x, y, z)`.
Type vector.

### `mix`
`factor`, `a` and `b` (each a number or `{"node": id}`, floats) and `clamp`. The result is
`a + (b − a)·t`, where `t` is `factor`, limited to 0..1 when `clamp` is true. Type float.

### `curve`
`value` (op id or a number, float), `points` (a list of `[x, y]`, already sorted by x, 1 to 16 of them) and
`mode` (`linear`, `smooth`, `step` or `cubic`). Below the first x the result is the first y and above the last x the
last y. Between points i and i+1, with `t = (value − x_i) / (x_i+1 − x_i)`: `linear` gives `mix(y_i, y_i+1, t)`,
`smooth` the same with `t` replaced by `t²(3 − 2t)`, `step` gives `y_i`, and `cubic` is a cubic Hermite
spline with `h = x_i+1 − x_i`, `t` as above and slopes `m_k` at the points: with secants `d_k`, an end point
has `m = d` of its one segment, an inner point has `m = 0` if `d_k−1·d_k ≤ 0`, else the weighted harmonic mean
`(w1 + w2) / (w1/d_k−1 + w2/d_k)` with `w1 = 2h_k + h_k−1`, `w2 = h_k + 2h_k−1` (so the curve never leaves the
range of its points); the value is `(2t³−3t²+1)·y_i + (t³−2t²+t)·h·m_i + (−2t³+3t²)·y_i+1 + (t³−t²)·h·m_i+1`. Type float.

### `grid`
`position` (op id or `null` for `p`), `counts` `[nx, ny, nz]` (cells per tile on each axis) and `reference`
(`center` or `corner`). With `n = counts` and `f = fract(q·n)`, the offset from the reference point is
`d = (f − 0.5) / n` for `center`, or `d = min(f, 1 − f) / n` (per component) for `corner`. The result is
`|d| / (0.5·|1/n|)·2 − 1`: −1 at the reference point and +1 as far from it as the grid gets (the cell
corners for `center`, the cell centres for `corner`). Tiles when every count is a whole number. Type float.

### `pattern`
`kind` selects the pattern; version 1 has `bricks`, `checker`, `stripes`, `dimples`, `gyroid`, `fractal`, `truchet` and `honeycomb`. Common fields: `position` (op id or `null` for
`p`) and `seed` (resolved, as for `noise`). Output is a float in [-1, 1].

**`bricks`**: a brick wall. Fields `counts` `[nx, ny, nz]`, `mode2d`, `axis`, `steps`, `gap`,
`bevel`, `output` (`distance`, `shape` or `cell_value`).

Axes. With `mode2d` false: along a row is X (r = 0), rows stack along Y (s = 1), layers along Z
(l = 2). With `mode2d` true the wall is extruded along `axis` (l) and the other two axes in
increasing order are r then s: `axis` 0 gives r = Y, s = Z; `axis` 1 gives r = X, s = Z; `axis` 2 gives
r = X, s = Y. `counts[a]` is the number of bricks per tile along axis `a`.

Per point `q` (the noise-space position): `x = q·counts` (per component), `row = floor(x[s])`,
`x[r] −= (row mod steps) / steps` (mod always non-negative, `steps` at least 1), `cell = floor(x)`,
`c = fract(x) − 0.5` (the offset from the brick centre, in brick units) and, in 2D mode, `c[l] = 0`.
- `distance`: `d = c / counts` per component (0 on an unused axis), result `|d| / (0.5·|inv|)·2 − 1`
  with `inv = 1 / counts` (0 on an unused axis): −1 at the centre, +1 at the corners. With `steps` 1 and
  `mode2d` false this equals the `grid` op's centre distance.
- `shape`: `e = 2·max(|c.x|, |c.y|, |c.z|)` (0 at the centre, 1 at the border). With `lo = 1 − gap − bevel`
  and `hi = 1 − gap`, `t = clamp((e − lo) / max(hi − lo, 1e-4), 0, 1)` smoothed as `t²(3 − 2t)`, and
  the result is `−1 + 2t`: −1 on the face, +1 in the mortar.
- `cell_value`: `id = cell` (with `id[l] = 0` in 2D mode) wrapped per axis by `counts` (rounded to a
  whole number, at least 1, non-negative modulo), then `hash3(id, seed).x · 2 − 1`. Bricks that straddle
  the tile edge therefore get the same id on both sides.

Tiles when the counts of the used axes are whole and, if `steps > 1`, `counts[s]` is a multiple of
`steps`.

**`checker`**: fields `counts`, `mode2d`, `axis`, `softness`. `counts` are cells per tile (one period of
the wave spans two cells, so every used count must be even to tile). With `f = fract(q · counts / 2)`
per component, `tri = 4·|fract(f − 0.25) − 0.5| − 1` is a triangle wave (0 at cell borders, ±1 at
cell centres) and `w = clamp(tri / max(softness, 1e-4), −1, 1)`. The result is `w.x · w.y · w.z`; in 2D mode
`w[axis]` is 1. The axes used are those of `mode2d` (see `bricks`).

**`stripes`**: fields `counts[0]` (stripes per tile), `dir`, `variant`, `width`, `softness`. With
`f = fract(dot(q, dir) · counts[0])` (whole `dir` and count tile): `variant` 0 sine `sin(2π f)`; 1 triangle
`1 − 4|f − ½|`; 2 sawtooth `2f − 1`; 3 square `1 − 2·ramp(½·width − ¼·softness, ½·width + ¼·softness, |f − ½|)`
where `ramp(lo, hi, x) = t²(3 − 2t)` with `t = clamp((x − lo) / max(hi − lo, 1e-4), 0, 1)`.

**`dimples`**: fields `counts` (cubic cells per tile), `variant` (lattice: 0 cubic, 1 body-centred, 2
face-centred), `output` (`distance` or `shape`), `radius`, `profile`. Lattice points are at
`(cell + b) / counts` for cells `floor(q·counts) + (−1..1)³`, with `b` in cell units: cubic `(½, ½, ½)`;
body-centred `(0,0,0)`, `(½,½,½)`; face-centred `(0,0,0)`, `(0,½,½)`, `(½,0,½)`, `(½,½,0)`. `dist` is the
distance from `q` to the nearest point and `inv = 1 / counts`. `distance`: `dist / cover · 2 − 1`, clamped,
with `cover = ½|inv|` (cubic), `0.5590170 · |inv|/√3` (body-centred) or `0.5 · |inv|/√3` (face-centred).
`shape`: with `edge = min(inv)`, `nearest` is `edge`, `0.8660254 · edge` or `0.7071068 · edge`, and
`t = dist / (radius · ½ · nearest)`; the depth is 0 for `t ≥ 1` and otherwise `sqrt(1 − t²)` (profile 0, sphere
cap), `1 − t` (1, cone), `1 − ramp(0, 1, t)` (2, smooth bowl) or 1 (3, flat disc); the result is `1 − 2·depth`.
For the cubic lattice `distance` equals the `grid` op's centre distance.

**`gyroid`**: fields `counts` (periods per tile per axis, whole to tile), `variant`, `output`, `width`. With
`a = 2π · q · counts`, `s = sin a`, `c = cos a`: the field is `s.x c.y + s.y c.z + s.z c.x` (variant 0, peak 1.5),
`c.x + c.y + c.z` (1, Schwarz P, peak 3) or `s.x s.y s.z + s.x c.y c.z + c.x s.y c.z + c.x c.y s.z` (2, diamond, peak √2);
`field = clamp(f / peak, −1, 1)`. `distance` returns the field; `shape` returns
`clamp((|field| − w) / (1 − w) · 2 − 1, −1, 1)` with `w = clamp(width, 0, 0.999)`.

**`fractal`**: fields `counts` (cells per tile, whole to tile), `variant` (0 Menger sponge, 1 Sierpinski
tetrahedron), `iterations` (1 to 6; values above 6 are clamped). With `p = (fract(q · counts) − ½) · 2` (the
cell as [−1, 1]³): `menger(p)` starts from the cube's distance `|max(|p| − 1, 0)| + min(max(q.x, q.y, q.z), 0)` and for each
iteration `m` (with `s` starting at 1) takes `a = (p·s mod 2) − 1`, multiplies `s` by 3, `r = |1 − 3|a||` and sets
`d = max(d, (min(max(r.x, r.y), max(r.y, r.z), max(r.z, r.x)) − 1) / s)`. `sierpinski(p)` folds
`z` each iteration (if `z.x + z.y < 0` then `z.xy = −z.yx`; if `z.x + z.z < 0` then `z.xz = −z.zx`; if `z.y + z.z < 0`
then `z.zy = −z.yz`, applied in that order) and sets `z = 2z − 1`, then returns `(max(−z.x − z.y − z.z, z.x + z.y − z.z,
−z.x + z.y + z.z, z.x − z.y + z.z) − 1) / √3`. The result is `clamp(d · k, −1, 1)` with `k = 3^iterations` for the sponge and
3 for the tetrahedron: −1 in the solid, +1 in the empty space.

**`truchet`**: fields `counts`, `mode2d`, `axis`, `variant` (0 arcs, 1 diagonals), `width`, `softness`, `seed`. The
plane is the row and stacking axes of `mode2d` (see `bricks`), extruded along the third. Per cell,
`flip = hash3(wrapped cell, seed).x > 0.5` (the cell wrapped by `counts` as for `bricks`, the extrusion axis 0),
`uv = fract(x)` on the two axes with `uv.x = 1 − uv.x` when flipped. Arcs: `d = min(|‖uv‖ − ½|, |‖uv − (1,1)‖ − ½|)`;
diagonals: `d = |uv.x − uv.y| / √2`. With `h = ½ · width`, the result is
`−1 + 2 · ramp(h(1 − softness), h(1 + softness), d)`. Version 1 has no 3D pipes variant.

**`honeycomb`**: fields `counts`, `variant` (0 hexagons, 1 body-centred cells, 2 face-centred cells), `mode2d`/`axis`
(hexagons are 2D: `mode2d` true, extruded along `axis`; the others are 3D), `output` (`distance`, `shape`,
`cell_value`), `gap`, `bevel`, `seed`. Lattice points are at `(cell + b) / counts`, `cell = floor(q · counts)` plus offsets in
{−1, 0, 1} on each used axis (just 0 on the extrusion axis in 2D). The offsets `b` (cell units): hexagons `(0,0,0)` and
`(½,½,½)` with the extrusion-axis component 0; body-centred `(0,0,0)`, `(½,½,½)`; face-centred `(0,0,0)`, `(0,½,½)`, `(½,0,½)`,
`(½,½,0)`. Pass 1 finds the nearest point `a` (in tile units, `x = q · counts / counts`); pass 2 takes, over every other point `b` in
the same search, `wall = min dot((a + b)/2 − q, normalize(b − a))` (the exact distance to the cell's boundary). With `inv = 1 / counts` the
cell radius is `r = ½ · min(inv on the used axes, |b · inv| for each basis offset b ≠ 0)` and `depth = clamp(wall / r, 0, 1)`.
`distance` returns `2·depth − 1`; `shape` returns `1 − 2·ramp(gap, gap + bevel, depth)` (+1 on walls); `cell_value` returns
`hash3(wrapped cell, seed + k).x · 2 − 1` for the nearest point's cell and basis index `k` (the cell wrapped by `counts` and, in 2D,
with the extrusion axis 0). Hexagons are regular when the counts are in the ratio 1 : 0.577 (7 × 4, 12 × 7, 26 × 15).

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
