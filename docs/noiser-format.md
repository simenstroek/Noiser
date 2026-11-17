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
`lacunarity`, `roughness`, `distortion`, `seed` (already resolved; a non-negative integer below
2^24), and `feature` for Worley (`f1`, `f2`, `f2_minus_f1`). Type float.

`kind` is one of `perlin`, `simplex`, `voronoi`, `voronoi_fractal`, `worley`, `white`, `blue`.
Version 1 specifies `perlin` exactly below. The others follow the same lattice, hash and fBm
rules; until each gets its own section, `src/noise.wgsl` is their definition.

### `normal`
The normalized gradient of the float field `field`, negated if `flip`. Type vector. Tetrahedral
central differences with `e = 1e-3`:
`g = Σ k_i · field(p + k_i·e)` over `k = (1,−1,−1), (−1,−1,1), (−1,1,−1), (1,1,1)`, then
`g / max(|g|, 1e-12)`. `field = null` gives `(0, 0, 0)`.

## The hash

All randomness comes from one hash, **Blender's `hash_float_to_float(float4)`**: Bob Jenkins'
lookup3 over the IEEE-754 bit patterns of four floats. Blender exposes it as the White Noise
Texture node (4D), so Blender reproduces it with a stock node. Elsewhere it is about 30 lines
of u32 code:

```c
uint32 rotl(uint32 x, uint32 k) { return (x << k) | (x >> (32 - k)); }

uint32 jenkins4(uint32 kx, uint32 ky, uint32 kz, uint32 kw) {
  uint32 a, b, c;
  a = b = c = 0xdeadbeef + (4 << 2) + 13;
  a += kx; b += ky; c += kz;
  /* mix */
  a -= c; a ^= rotl(c, 4);  c += b;
  b -= a; b ^= rotl(a, 6);  a += c;
  c -= b; c ^= rotl(b, 8);  b += a;
  a -= c; a ^= rotl(c, 16); c += b;
  b -= a; b ^= rotl(a, 19); a += c;
  c -= b; c ^= rotl(b, 4);  b += a;
  a += kw;
  /* final */
  c ^= b; c -= rotl(b, 14);
  a ^= c; a -= rotl(c, 11);
  b ^= a; b -= rotl(a, 25);
  c ^= b; c -= rotl(b, 16);
  a ^= c; a -= rotl(c, 4);
  b ^= a; b -= rotl(a, 14);
  c ^= b; c -= rotl(b, 24);
  return c;
}

/* All arithmetic wraps mod 2^32. bits(f) is the float's bit pattern. */
float h(float4 k) { return float(jenkins4(bits(k.x), bits(k.y), bits(k.z), bits(k.w))) / 4294967296.0; }
```

The per-cell random vector for integer cell `c` and seed `s` is
`hash3(c, s) = (h(k), h(k.zxwy), h(k.wzyx))` with `k = (c.x, c.y, c.z, s)` as floats. This is
the White Noise node's 4D Color output with Vector = `c` and W = `s`.

- Cells and seeds must be exact integers in float (|value| < 2^24).
- **Never feed `-0.0`**: its bit pattern differs from `+0.0`. Cells computed as `floor(x) + offset`
  (adding the offset even when it is 0) or by the wrap below are always `+0.0`.

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

`BASE` for `perlin` is `perlin(...)` above. Kinds without octaves: `voronoi`, `worley`,
`white`, `blue`.

## Test vectors

`tests.points` are the centres of a 6×6×6 grid over one tile, `((i + 0.5) / 6)` per axis with
x varying fastest. `tests.density` is Noiser's GPU result at each point. A sampler passes when
every value is within **1e-4** of the vector. Float rounding differs between hosts; the hash
itself is exact, so larger differences mean a real mismatch.
