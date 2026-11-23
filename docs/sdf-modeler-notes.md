# SDF Modeler 0.6 files: what is known

Working notes for the SDF Composer. SDF Modeler is closed, so everything here was measured: the Python scripts in
[`tools/sdf_calibration`](../tools/sdf_calibration) evaluate a candidate model of a `.sdf` file and compare it with
the meshes SDF Modeler itself exports (Export > PLY), vertex by vertex. A correct model puts every exported vertex on
the zero set of the distance field, so the median `|d|` is about 1e-5 and the 95th percentile below 1e-3. The exports
are marching-cubes meshes, so errors around 1e-3 to 5e-3 are expected.

```
python tools/sdf_calibration/runlayers.py robot          # every layer of robot.sdf
python tools/sdf_calibration/runlayers.py mech 9,10,11   # chosen layers (0-based)
python tools/sdf_calibration/gallery.py                  # each isolated shape of shapes.sdf
python tools/sdf_calibration/worst.py mech 5             # which shapes sit near the worst vertices of a layer
```

Set `SDF_SAMPLES` to the folder holding the samples and their exported `.ply` files. `shapes.sdf` is a gallery of
isolated shapes on a grid and is the best calibration set. The Rust port is `src/sdf_model.rs`; its tests run the same
comparison when the samples are present.

## Result

Of the 48 layers in the nine 0.6 sample files that are not empty, 44 match with a 95th percentile below 0.01 (most below
0.001). The other four are `horse` Body, Mane and Tail and `mouse` Mouse, which contain splines (not supported). Files
from before 0.6 (`head`, `modern_monkey`, `troll`) use another layout and are not read.

## Rules

- **File.** `assemblies[].shapes[]` is a tree of `shape` and `group` nodes (and `spline` / `control_point`, not covered).
  A layer places one assembly. Exported PLY files: `name.ply` is layer 1, `name - k.ply` is layer k + 1.
- **Coordinates.** PLY `(x, y, z)` = file `(x, -z, y)`. Layer transform: `local = Rᵀ(p − position) / scale`, where `R`
  is the Euler rotation (x, y, z radians; matrix `Rz · Ry · Rx`). Shapes and groups use the same convention.
- **Scale is the full size.** A sphere's radii and a cube's half extents are `scale / 2`.
- **Mirror.** `mirror_x/y/z` replaces that coordinate by a smooth absolute value of width `mirror_blend` (a plain
  absolute value when it is 0), in the parent's frame, before the node's own position and rotation. The shader's
  `smooth_abs` is used; no sample exercises a large `mirror_blend`, so that part is unverified.
- **Combining.** Siblings combine front to back. The shape being added is `a`, what is accumulated is `b`
  (`sub` removes `a` from `b`). `blend` is the smooth (or, with `chamfer`, chamfer) width `k`; `blend_strength` is the
  radius `r` of `push`, `avoid`, `emboss`, `deboss` and `shell` (`shell` uses `r + 0.01`). `paint` leaves the distance
  unchanged. A group is evaluated in its own frame (`/ scale`, distance multiplied back by `|scale.x|`) and then
  combined like a shape. Formulas: `tools/sdf_calibration/sdftree.py` (they are the shader's).
- **Linear repeat** (`linear_repeats`, `steps`, alternate cells mirrored) and **radial repeat** (`radial_repeats`,
  `radius`, `repeat_axis`, the two nearest copies) follow the shader's `repetition_linear` / `repetition_radial`.
- **Sphere** is an ellipsoid with radii `scale / 2`, distance `(|p/r|₂ − 1) · min(r)`. `round`, `bevel`, `cone` and
  `thickness` do not change it.
- **Cube** is the shader's "uber primitive" (`sd_uberprim`, in `sdfcal.py`) with `y` flipped, `s = (half, W)` and
  `r = (Rv, ry, rz)`. With `hxz = min(half.x, half.z)`, `ry = round · min(half)`, and for `thickness` t < 1 a wall
  `wall = ½ · hxz · (0.05 + 0.95 t)`:
  - hollow: `ry = min(ry, wall)` (the edge radius cannot exceed the wall); solid (t = 1): `W` is huge.
  - `Rv = max(ry, bevel · hxz)`: the vertical-edge radius (`round` 1, `bevel` 0 is a capsule).
  - `rz = −cone · max(hxz − ry, 0)`: a linear taper, narrower toward +y.
  - hollow: `W = wall + |rz|`. A hollow cube is a tube open at both ends.
- **Cylinder, prism, polygon** use the shader's `sdf_cylinder`, `sdf_triprism` and `sdf_ngon` with
  `sw = min(half) · (0.01 + 0.99 · round)` and `rx = max(sw, bevel · min(half))`:
  cylinder `sd_cylinder(p, half − sw) − sw`; prism `sd_triprism(p, half − rx, max(rx − sw, 0), 0) − sw`;
  polygon `sd_ngon(p, (half.x − rx, half.y − sw), points, 1 − star, max(rx − sw, 0), 0) − sw`. Solid only (all samples
  have `thickness` 1); `cone` is ignored.

## Open

- Every cone face sits about `0.003 · cone` further out than predicted (under 0.3% of the shape).
- Hollow cubes with a small `round` (gallery #8, #9) want a smaller edge radius than `min(round · min(half), wall)`
  (error about 0.02 on a 1 m shape).
- Not exercised by any sample: group scale other than 1, a large `mirror_blend`, hollow or starred polygons and
  prisms, flips. Splines and the free-form polygon tool are not supported.
- The exported meshes are only as accurate as their grid: large shapes at low `resolution` (a 12 m ground box at
  resolution 6) cannot resolve small radii, so judge small shapes.
