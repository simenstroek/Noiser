# SDF Modeler 0.6 files: what is known

Working notes for the SDF Composer. SDF Modeler is closed, so everything here was measured: the Python scripts in
[`tools/sdf_calibration`](../tools/sdf_calibration) evaluate a candidate model of a `.sdf` file and compare it with
the meshes SDF Modeler itself exports (Export > PLY), vertex by vertex. A correct model puts every exported vertex on
the zero set of the distance field, so the median `|d|` is about 1e-5. Run it with

```
python tools/sdf_calibration/runlayers.py robot          # every layer of robot.sdf
python tools/sdf_calibration/runlayers.py mech 9,10,11   # chosen layers (0-based)
```

Set `SDF_SAMPLES` to the folder holding the samples and their exported `.ply` files.

## Verified

- **File.** `assemblies[].shapes[]` is a tree of `shape` and `group` nodes (and `spline` / `control_point`, not covered).
  A layer places one assembly. Exported PLY files: `name.ply` is layer 1, `name - k.ply` is layer k + 1.
- **Coordinates.** PLY `(x, y, z)` = file `(x, -z, y)`. Layer transform: `local = Rᵀ(p − position) / scale`, where `R`
  is the Euler rotation (x, y, z radians; matrix `Rz · Ry · Rx`). Shapes and groups use the same convention.
- **Scale is the full size.** A sphere's radii and a cube's half extents are `scale / 2`.
- **Mirror.** `mirror_x/y/z` takes the absolute value of that coordinate in the parent's frame, before the node's own
  position and rotation (eyes mirrored about the group origin).
- **Combining.** Siblings combine front to back. The shape being added is `a`, what is accumulated is `b`
  (`sub` removes `a` from `b`). `blend` is the smooth (or, with `chamfer`, chamfer) width `k`; `blend_strength` is the
  radius `r` of `push`, `avoid`, `emboss`, `deboss` and `shell`. `paint` leaves the distance unchanged. A group is
  evaluated in its own frame (`/ scale`, distance multiplied back by `|scale.x|`) and then combined like a shape.
  Formulas: `tools/sdf_calibration/sdftree.py` (they are the shader's).
- **Cube** is the shader's "uber primitive" (`sd_uberprim`) with `y` flipped:
  `s = (half, W)`, `r = (bevel · hxz, round · min(half), −cone · hxz)` where `hxz = min(half.x, half.z)`.
  The vertical-edge radius is `bevel · hxz`, the edge radius `round · min(half)`, and `cone` tapers the top by
  `cone · hxz`. Solid cubes (`thickness` 1) use a very large `W`. Verified to ~1e-5 on cubes with and without cones.
- **Sphere** is an ellipsoid with radii `scale / 2` and exponent 2 (verified on `horse` Eyes: 2e-5).
- **Linear repeat** (`linear_repeats`, `steps`) and layer transforms reproduce `fried_egg` and most of `robot`.

## Open

- `thickness` < 1 on cubes hollows them into a tube: `W` is close to `thickness · hxz / 2` but not equal
  (`bricks`, thickness 0.48, fits best at W ≈ 0.126 where the formula gives 0.12). Needs more data points.
- Cylinder, prism and polygon brushes (a handful of shapes in `headphones`, `mouse`, `mech`, `shapes`).
- Sphere `bevel`, `cone` and `thickness`; `radial` repeat (`horse`); `tint` and colours are not needed for geometry.
- Layers that still disagree: `horse` Body/Mane/Tail/Collar, `mech` Head/Arms/Legs, `mouse` Mouse, `headphones`
  Ear Cup/Headband/Padding, `bricks`. Splines (`Cables` layers) are out of scope for now.
- The exported meshes are only as accurate as their grid: large shapes at low `resolution` (a 12 m ground box at
  resolution 6) cannot resolve small radii, so judge small shapes.
