# Pattern nodes plan

Status: planned, nothing built. Written 2026-11-17 for an implementing agent.

> **Update from the project owner, 2026-11-17 (supersedes the Brick handling below):** the existing
> `Brick` node was renamed **Grid** and kept as its own node (not folded into Dimples). It gained a
> reference option: distance to the nearest cell centre (default) or to the nearest cell corner, both
> scaled to [-1, 1] (-1 at the reference point). So wherever this plan says `Brick` (sections 2, 3
> "Migration of `Brick`", 4 Dimples, 8 step 2, 9), read **Grid**: the function is `grid_dist()` in
> `src/noise.wgsl` (was `brick()`), the test is `grid_spans_minus_one_to_one_and_tiles`, and saved files
> that say `Brick` load as Grid through a serde alias (no migration function needed). Do not remove Grid
> when adding Dimples; Dimples' Cubic/Distance output may reuse `grid_dist`. The staggered **Bricks**
> node in this plan is the node the owner wants for brick patterns. Also new since this was written:
> a Math node now shows its operator as a dropdown on the node (`body_top()` in `src/editor.rs` reserves
> the row), and `NodeKind::Remap`, `MathOp::{Power, SmoothMin, SmoothMax, Dot}` and pin colors exist
> (float blue, vector yellow); see PLAN.md "Nodes added after the first milestones".

Noiser has Perlin, Simplex, Worley, White and Blue noise, plus a `Brick` node that is really a cubic grid of cell-distance fields. Nothing makes regular structure: running-bond bricks, golf-ball dimples, honeycomb, Truchet, stripes, checker, gyroids, sponge fractals. This plan adds those, extends Worley, and adds two Math operators.

## 1. Goals and rules
- Every node tiles when its counts are whole numbers (see the per-node tiling rules), uses the same `Position` input as Noise, has one float output, and is deterministic from `seed`.
- **Output convention for all new patterns: [-1, 1], `-1` on the "face" or inside of a feature, `+1` in the gap, wall or empty space.** This matches `Brick` (centre -1, corners +1). Artists remap with Remap or Math. No per-node "normalize" checkbox.
- Artist-friendly names in the add menu and node title. Options live in the details panel (`node_props` in `src/main.rs`), each with a reset button via `row()`.
- 2D variants are extruded along a chosen axis (`axis` X/Y/Z); 3D variants use all three.

## 2. Node list (add-menu names)
| Node | Options in the details panel |
|---|---|
| **Bricks** | layout, offset rows, gap, bevel, output, 2D/3D |
| **Dimples** | lattice Cubic/BCC/FCC, radius, profile, output |
| **Honeycomb** | layout Hexagons (2D) / BCC cells (3D) / FCC cells (3D) / Triangles (2D), wall, bevel, output |
| **Truchet** | style Arcs / Diagonals (2D), Pipes (3D); width, bevel, output |
| **Stripes** | wave Sine/Triangle/Saw/Square, direction, count, width, softness |
| **Checker** | 2D/3D, counts, softness |
| **Gyroid** | type Gyroid / Schwarz P / Diamond, mode Field / Shell, thickness |
| **Fractal** | type Menger sponge / Sierpinski tetrahedron, iterations |

The existing `Brick` node becomes **Dimples** with lattice = Cubic (see section 3).

Not in this plan: dots/halftone (Dimples with a small radius covers it), spirals, weaves, scales, herringbone, wood, marble, curl noise, domain warp, reaction-diffusion, Mandelbulb.

## 3. Architecture: one `NodeKind::Pattern`
Eight new `NodeKind` variants would each touch about ten places. Instead mirror `Noise(NoiseParams)`:

```rust
NodeKind::Pattern(PatternParams)
pub enum PatternKind { Bricks, Dimples, Honeycomb, Truchet, Stripes, Checker, Gyroid, Fractal }
pub struct PatternParams {              // flat, every field #[serde(default)], like NoiseParams
    kind: PatternKind,
    counts: [f32; 3],                   // cells per tile per axis; whole numbers tile
    axis: u32,                          // extrusion axis for 2D modes (0=X,1=Y,2=Z)
    mode2d: bool,
    variant: u32,                       // lattice / layout / style / wave / type, meaning per kind
    gap: f32, bevel: f32,               // gap/wall/mortar width and edge softness, 0..1
    radius: f32, profile: u32,          // Dimples
    steps: u32,                         // Bricks row offset steps (1 = none, 2 = half, 3 = third)
    dir: [f32; 3],                      // Stripes direction, whole numbers
    iterations: u32,                    // Fractal
    output: u32,                        // 0 = Shape, 1 = Cell value (random per cell)
    seed: Option<u32>,
}
```
Use real Rust enums for `variant`/`profile`/`output` in the struct if preferred; the GPU slot is a float either way (as `feature` is for Worley).

Files touched, once for the whole family (the Brick path is the template; grep `Brick`):
- `src/graph.rs`: enum, `input_count` = 1, `PatternParams`, tiling rule (below), `input_type` Vector, `range_of` = (-1, 1), cache eligibility, and the `finest` resolution estimate (use `max(counts)`; Fractal: `max(counts) * 3^iterations`, capped by the existing 512 limit).
- `src/codegen.rs`: params slots (`params.extend([...])` next to Noise), `node_expr` emits `pattern_k{N}(q, slot)`. One WGSL function per kind, generated only when used (like `kinds` for noise).
- `src/noise.wgsl` / new `src/patterns.wgsl`: the kernels (section 5). Needs a per-axis wrap (`wrap_cell3(c, n: vec3i)`); the existing `wrap_cell` takes one isotropic period.
- `src/editor.rs`: `title`, `input_label` ("Position"), `palette()` entries.
- `src/main.rs`: `node_props` section.
- `src/noiser_file.rs` + `docs/noiser-format.md`: one op `pattern` with `kind` and the fields, like `noise`.
- `src/gpu_tests.rs`: one test per kind (range, seam, reference points).
- Blender sampler: `blender/noiser/build.py` already raises `Unsupported` for unknown ops, which falls back to the texture bake. Leave patterns unsupported there.

**Migration of `Brick`:** write a `migrate_` function in the style of `migrate_noise_kinds` (`src/graph.rs`): `Brick {counts}` becomes `Pattern{kind: Dimples, variant: Cubic, counts, output: Distance}`. The `.noiser` op `brick` stays readable and defined as an alias; new files write `pattern`. Keep `brick_spans_minus_one_to_one_and_tiles` passing unchanged: the Cubic/Distance path must stay bit-identical.

**Tiling rule** (one helper, used by all kinds): Periodic when every used count is whole and the input is None/Coord/Periodic; Const for a Const input; Broken(reason) otherwise. Extra per-kind reasons are listed below. Reasons are shown to the user, so write them in plain words.

**Seeds:** `seed` overrides the global seed as with Noise (slot value -1 = global). Cell randomness reuses `hash01(cell, seed)`.

## 4. Per-node spec

### Bricks
- `counts` X = bricks along the row, Y = rows, Z = layers. `mode2d` + `axis` = extruded bricks.
- **Offset rows** (`steps`): row k is shifted along X by `(k mod steps) / steps` of a brick. 1 = stacked grid, 2 = running bond (default), 3 = thirds. In 3D, do rows first; shifting Z layers (e.g. half a step along X) is optional.
- Tiling: Broken("rows must be a multiple of the offset steps") unless `counts.y % steps == 0`.
- Math: brick-local `u` in [-1,1]³ (offset removed); `e = max(|u.x|,|u.y|,|u.z|)` (drop z in 2D). `t = smoothstep(1 - gap - bevel, 1 - gap, e)`; output `mix(-1, 1, t)`. bevel 0 = hard edge (use a 1e-4 floor to avoid a NaN smoothstep).
- Output option Cell value: `hash01(wrapped brick id, seed).x*2-1`.

### Dimples (golf ball; absorbs Brick)
- `variant` lattice: Cubic (1 point per cell), BCC (2 points: (0,0,0),(½,½,½)), FCC (4 points: (0,0,0),(0,½,½),(½,0,½),(½,½,0)). `counts` = cubic cells per tile per axis; points are in tile units, so BCC/FCC tile with whole counts.
- Output **Distance** (default): distance to the nearest lattice point, in tile units, divided by the lattice's covering radius, mapped to [-1, 1]. Cubic must equal the existing `brick()` formula (`length(d) / (0.5*length(1/n)) * 2 - 1`). For BCC/FCC use the isotropic covering radius (BCC `sqrt(5)/4`, FCC `1/2` of the cubic cell edge) times `length(1/n)/sqrt(3)`; exact for equal counts, approximate otherwise, clamp to [-1, 1].
- Output **Dimples**: `s = dist / (radius * half_nn)`, where `half_nn` is half the nearest-neighbour spacing (Cubic `min(1/n)`, BCC `sqrt(3)/2*min(1/n)`, FCC `min(1/n)/sqrt(2)`, then halved). Profile of `s`: Sphere cap `sqrt(1 - s²)`, Cone `1 - s`, Smooth bowl `smoothstep`, Flat disc step; all 0 for `s >= 1`. Output `1 - 2*profile`: -1 at the dimple centre, +1 on the flat. radius 1 = touching dimples. Golf ball = BCC or FCC, Sphere cap, radius ≈ 0.9, fed into Density.
- Search: check the 27 neighbouring cells (like `cells()` in `noise.wgsl`) with the basis points.

### Honeycomb
- Shared machinery: `border_distance(q, points_for_cell)`: find the nearest lattice point `a` and the runner-up `b` by scanning neighbours, then the wall distance is `(|b-p|² - |a-p|²) / (2|b-a|)` (distance to the bisector; IQ's exact Voronoi border). The same helper serves **Worley Edges** (section 6): factor it into one WGSL function so there is one copy.
- Layouts: **Hexagons (2D)**: rect cell of size `(1/nx, 1/ny)` holding two points, `(0,0)` and `(½,½)` of the cell, with y scaled so the hexagons are regular when `ny/nx ≈ 0.577`. **BCC cells (3D)** (truncated octahedra) and **FCC cells (3D)** (rhombic dodecahedra): the Dimples basis points. Tiling is exact for whole counts; hexagons are stretched when `ny/nx` isn't 0.577 (UI hint: "Y ≈ X × 0.577 gives regular hexagons: 7×4, 12×7, 26×15"). Tiling needs no extra rule for these three.
- **Triangles (2D)** (last, optional): three line families at 0°, 60°, 120° whose phase functions are integer-linear in (x, y), distance = min over the three families of the distance to the nearest line. Verify the seam test passes before shipping it.
- Output **Shape**: `w = wall_distance / half_cell_width`; `t = smoothstep(...)` with `wall`/`bevel` as for Bricks, giving -1 inside the cell, +1 on the wall. Output **Cell value**: random per nearest point.

### Truchet
- Cells: `counts` per axis (whole), random orientation per cell from `hash01(wrapped cell, seed)` (per-axis wrap: `wrap_cell3`).
- **Arcs (2D)**: quarter circles of radius ½ centred on two opposite cell corners, random flip per cell. **Diagonals (2D)**: one diagonal per cell, random flip. Both: distance to the curve → band of half-width `width`, softened by `bevel` (same -1 inside / +1 outside mapping).
- **Pipes (3D)**: each cell connects its six face centres in three pairs of adjacent faces with quarter arcs of radius ½ centred on cell edge midpoints (the tangent at each face centre is perpendicular to the face, so pipes continue across cells). Pick one of the 8 valid pairings via the hash. Distance to the nearest arc centreline minus `width`. Check continuity across faces in a test.
- Output Cell value: random per cell.

### Stripes
- `phase = dot(q, dir) * count + offset`; `dir` whole numbers (default (1,0,0); (1,1,0) is a diagonal; a Direction preset X/Y/Z/Diagonal sets it), `count` whole. Tiles when `dir` and `count` are whole, else Broken("direction or stripe count is not whole").
- Waves: Sine `sin(2π·phase)`, Triangle `1 - 4|fract(phase) - ½|`, Sawtooth `2·fract(phase) - 1`, Square with `width` (duty) and `softness` (smoothstep edge). All in [-1, 1].

### Checker
- `counts` per axis, 2D (extruded, `axis`) or 3D. Per axis a softened square wave `s_i` of `2·fract(q_i·n_i) - 1` (edge width `softness`); output is the product of the used axes' `s_i` (so +1/-1 checks, blended at the edges).
- Tiling: Broken("checker needs even counts") if any used count is odd (parity flips at the tile edge).

### Gyroid
- Phase `x = 2π·q·n` per axis (`counts`, whole). `variant`: Gyroid `sin x cos y + sin y cos z + sin z cos x`; Schwarz P `cos x + cos y + cos z`; Diamond `sin x sin y sin z + sin x cos y cos z + cos x sin y cos z + cos x cos y sin z`.
- Normalize each by its true range (measure it in the range test; do not trust textbook numbers: Gyroid about ±1.5, Schwarz P ±3, Diamond to be measured), clamp to [-1, 1].
- Mode **Field**: the normalized value (the sheet is its zero level; a level shift is a Math add). Mode **Shell**: `abs(f) - thickness` remapped to [-1, 1], giving a thin sheet.

### Fractal (Menger + Sierpinski)
- `variant`: Menger sponge or Sierpinski tetrahedron; `counts` = fractals per tile per axis (whole); `iterations` 1..6 (Detail slider). Work in `fract(q·n)` so it tiles for whole counts.
- Menger: IQ's cross-subtraction SDF on the unit cube, `iterations` passes. Sierpinski: tetrahedron fold IFS, `d = length(p) * 2^-iterations`.
- Output **Shape**: signed distance scaled by the cell size and clamped to [-1, 1] (-1 solid, +1 empty). The distance is a bounded approximation, which is fine for thresholding and for the Normal node (gradient by differences).
- The loop bound is a uniform (`P[]` slot), so changing iterations does not recompile.

## 5. WGSL organisation
New file `src/patterns.wgsl`, included next to `noise.wgsl` the same way (see how `NOISE_LIB` and `FBM_TEMPLATE` are included in `codegen.rs`). One function per kind, `fn pat_bricks(q, s)`, reading its params from `P[s..]`. Share: `wrap_cell3`, the lattice-point basis loop, `border_distance`, `smooth_gap(e, gap, bevel)`. `codegen.rs` collects used kinds in a `BTreeSet<PatternKind>` and emits `pattern_k{N}` wrappers (copy how `kinds`/`fbm_function` work).

## 6. Worley variants (extend `NoiseParams`)
Add three fields, all `#[serde(default)]` so old files and the default output are unchanged:
- `jitter: f32` (default 1.0): feature point = `mix(vec3(0.5), hash01(...), jitter)`. 0 = perfect lattice, 1 = current Worley. The `worley()` range mapping (`/1.4` and similar) assumes jitter 1; rescale it with jitter and clamp, then check the range test.
- `metric` (default Euclidean): Euclidean / Manhattan ("diamond") / Chebyshev ("square"). Replace `dot(d,d)` with the metric. The 3×3×3 search can miss the nearest point for Manhattan/Chebyshev at high jitter; accept and document.
- New `WorleyFeature::Edges` ("Cell edges / cracks"): the exact border distance from `border_distance` (section 4), mapped to [-1, 1] with the same clamp pattern. Costs a second pass over the 27 neighbours.
- UI: in the Worley block of `node_props`, add a `jitter` slider (0..1), `metric` buttons, and "Cell edges" in the feature row. Slot layout: append `metric` and `jitter` after `feature` in `compile_with` (slots 8, 9) and extend the header comment in `noise_fbm.wgsl`. `worley(x, period, seed, feature)` gains `metric` and `jitter` arguments (BASE string in `fbm_function`).
- `.noiser` format: emit `metric`, `jitter` and feature `edges` only when non-default. Samplers must raise Unsupported for fields they don't know (the Blender sampler only does Perlin, so it already does).
- Even cells (Poisson/blue-noise Voronoi): the jitter slider covers most of the look. A true version would bake a relaxed point set per tile like `src/bluenoise.rs`; leave as a follow-up and check how that file builds its tables first.

## 7. Math: Absolute and Ridge
- Add `MathOp::Absolute` (`|a|`, billow) and `MathOp::Ridge` (`1 - |a|`, ridged noise). Both are unary.
- `graph.rs`: add to `ALL` (12), add `fn is_unary(self)`, and make `NodeKind::input_count` return 1 for them. Already general: `prune_edges` drops links into ports that no longer exist, and `main.rs` calls it after every details-panel edit.
- Fix the places that assume two operands: tiling (`tiling_of` Math arm: treat unary b as Const, don't read port 1), `yields_vector` (only port 0), `input_type` (only port 0), `math_range` (Absolute: `[0 if the range straddles 0 else min(|a0|,|a1|), max(|a0|,|a1|)]`; Ridge: `1 - that`, mirrored). Abs of a position (Coord) falls into the existing "does not tile" Broken arm; leave it.
- `codegen.rs`: `abs(a)` and `1.0 - abs(a)` (per component for vectors). `noiser_file.rs`: fn names `absolute`, `ridge`; add both to `docs/noiser-format.md` (b ignored). Blender `MATH_OPS`: Absolute maps to Blender's ABSOLUTE; Ridge needs two nodes (or leave Unsupported and let it bake).
- Details panel hint: "Ridge = 1 − |a|. Ridge of noise gives ridged noise; Absolute gives billow."
- Tests: add to `new_math_ops_match_reference`; a graph test that switching Math from Add to Absolute prunes a link into B.

## 8. Order of work
1. Math Absolute/Ridge (small, isolates the unary-op plumbing).
2. `Pattern` plumbing plus the Brick → Dimples migration, with Cubic/Distance bit-identical (existing tests must pass untouched). Ship **Dimples** first, since it exercises the whole path.
3. Bricks, Stripes, Checker (cheap math, no search).
4. `border_distance` + Worley variants (jitter, metric, Edges) + Honeycomb (they share the helper).
5. Gyroid, Fractal, Truchet.
6. Honeycomb Triangles, Even-cells Voronoi (optional).

## 9. Verification
For each node, in `src/gpu_tests.rs` (copy `brick_spans_minus_one_to_one_and_tiles`): range inside [-1, 1] and filling it, one-tile seam equality on all three axes (`eval` at `p` and `p + axis`, worst difference < 2e-3), and a few hand-computed reference points (centre of a brick, dimple centre, mortar). Plus: Broken-tiling cases in `graph.rs` tests (odd checker count, bricks rows not a multiple of steps, non-whole counts); a save/load test for the Brick migration; a `codegen.rs` naga-validated shader per kind (the `validate` helper). Run `cargo test` (GPU tests skip without an adapter). Finally open the app, add each node, check the preview slice view and the 2×2 tile view show no seams, and a Dimples (BCC, Sphere cap) graph into Density to confirm the golf-ball look.

## 10. Known risks
- Anisotropic counts: Dimples BCC/FCC normalisation is approximate; hexagon regularity depends on the X/Y count ratio.
- Manhattan/Chebyshev Worley can miss the true nearest point at jitter 1.
- Fractal SDF is bounded-error, not exact.
- Adding Pattern to the `.noiser` format is additive (still version 1); old samplers must treat unknown ops as Unsupported (the Blender one does).
