# SDF Modeler 0.6 files: what is known

Working notes for the SDF Composer. SDF Modeler is closed, so everything here was measured: the Python scripts in
[`tools/sdf_calibration`](../tools/sdf_calibration) evaluate a candidate model of a `.sdf` file and compare it with
the meshes SDF Modeler itself exports (Export > PLY), vertex by vertex. A correct model puts every exported vertex on
the zero set of the distance field, so the median `|d|` is about 1e-5 and the 95th percentile below 1e-3.

```
python tools/sdf_calibration/runlayers.py robot          # every layer of robot.sdf
python tools/sdf_calibration/runlayers.py mech 9,10,11   # chosen layers (0-based)
python tools/sdf_calibration/gallery.py                  # each isolated shape of shapes.sdf
python tools/sdf_calibration/worst.py mech 5             # which shapes sit near the worst vertices of a layer
```

Set `SDF_SAMPLES` to the folder holding the samples and their exported `.ply` files. `shapes.sdf` is a gallery of
isolated shapes on a grid and is the best calibration set.

## Verified

- **File.** `assemblies[].shapes[]` is a tree of `shape` and `group` nodes (and `spline` / `control_point`, not covered).
  A layer places one assembly. Exported PLY files: `name.ply` is layer 1, `name - k.ply` is layer k + 1.
- **Coordinates.** PLY `(x, y, z)` = file `(x, -z, y)`. Layer transform: `local = Rᵀ(p − position) / scale`, where `R`
  is the Euler rotation (x, y, z radians; matrix `Rz · Ry · Rx`). Shapes and groups use the same convention.
- **Scale is the full size.** A sphere's radii and a cube's half extents are `scale / 2`.
- **Mirror.** `mirror_x/y/z` takes the absolute value of that coordinate in the parent's frame, before the node's own
  position and rotation. (`mirror_blend`, a smooth absolute value, is not implemented yet.)
- **Combining.** Siblings combine front to back. The shape being added is `a`, what is accumulated is `b`
  (`sub` removes `a` from `b`). `blend` is the smooth (or, with `chamfer`, chamfer) width `k`; `blend_strength` is the
  radius `r` of `push`, `avoid`, `emboss`, `deboss` and `shell`. `paint` leaves the distance unchanged. A group is
  evaluated in its own frame (`/ scale`, distance multiplied back by `|scale.x|`) and then combined like a shape.
  Formulas: `tools/sdf_calibration/sdftree.py` (they are the shader's). Layers of `robot`, `fried_egg`, `icon` and
  most of `mech`, `mouse`, `headphones`, `bricks` match to 1e-3 or better with these rules.
- **Sphere** is an ellipsoid with radii `scale / 2` and exponent 2, distance `(|p/r|₂ − 1) · min(r)`. `round` and
  `thickness` do not change it (gallery spheres: 6e-5).
- **Cube** is the shader's "uber primitive" (`sd_uberprim`, in `sdfcal.py`) with `y` flipped, `s = (half, W)` and
  `r = (Rv, ry, rz)`. With `hxz = min(half.x, half.z)`:
  - `ry = round · min(half)`: the radius of the horizontal edges.
  - `Rv = max(ry, bevel · hxz)`: the radius of the vertical edges (`round` 1, `bevel` 0 is a capsule).
  - `rz = −cone · hxz · (half.y − ry) / half.y`: a linear taper, narrower toward +y (slope exactly
    `cone · hxz / (2 · half.y)`).
  - `W`: `thickness` 1 is solid (`W` huge). Otherwise half the wall thickness is
    `W = ½ · hxz · (0.05 + 0.95 · thickness) + |rz|`. A hollow cube is a tube open at both ends. `ry` is capped at the
    wall: `ry = min(ry, W)`, with `rz` recomputed for the new `ry`.
  Solid and hollow cubes without a cone match to 1e-4. Hollow cubes with a cone (`mech` Arms) match to 7e-4.
- **Linear repeat** (`linear_repeats`, `steps`, alternate cells mirrored) reproduces `fried_egg` and `bricks`.

## Open

- **Cube corner cases.** A cone together with a large `round` (`mech` Head and Legs: p95 0.01 – 0.03) is off by
  about 16% in the taper; a hollow cube with a small `round` wants an edge radius far below `round · min(half)`
  (gallery #8: 0.017 against 0.07). Every cone face also sits about `0.003 · cone` further out than predicted.
- **Not implemented:** cylinder, prism and polygon brushes (a handful of shapes in `headphones`, `mouse`, `mech`,
  `shapes`), `mirror_blend`, radial repeat (`horse` Mane and Tail), the sphere's `bevel` and `cone`
  (`headphones` Ear Cup), group scale (no sample uses it).
- **Splines are out of scope.** `horse` Body, Mane and Tail, `mouse` Mouse and the `Cables` layers contain them,
  so those layers cannot be checked.
- The exported meshes are only as accurate as their grid: large shapes at low `resolution` (a 12 m ground box at
  resolution 6) cannot resolve small radii, so judge small shapes.
