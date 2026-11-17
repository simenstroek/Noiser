# Noiser

A node-based editor for **tiling 3D noise**. You build a graph of noises, patterns and math, watch it
as a raymarched shape and as a 2D slice while you work, and export it as 2D or 3D textures, an
OpenVDB volume, or a portable `.noiser` math-function file that other software can evaluate.

Everything is evaluated on the GPU (Rust, egui, wgpu). Resolution never limits the detail: the graph
stays a function, and exports sample it at whatever resolution you ask for.

> **Status: early and moving fast.** Windows only for now (a Vulkan or DX12 GPU is required).
> Releases are builds of whatever state the project is in at the time; see
> [Releases](../../releases) and the notes there.

## Running it

Download `noiser-windows-x64.zip` from the latest release, unzip it and run `noiser.exe`. To open a
graph straight away: `noiser.exe SimpleTestGraph.noisegraph` (an example is in the zip).

To build from source you need a recent stable Rust toolchain:

```
cargo run --release
cargo test          # unit tests, plus GPU tests that skip when there is no adapter
```

## What it does

**Tiling by construction.** One tile is the unit cube. Noises wrap their lattice, so a graph repeats
with period 1 on every axis whenever its scales and counts are whole numbers. Every node is checked, and
a node that breaks tiling gets an orange outline and says why in the details panel.

**Nodes**
- **Noise**: Perlin, Simplex, Worley and Blue/White. Worley has F1, F2, F2 − F1, a flat cell value or exact
  *cell edges*, a choice of Euclidean, Manhattan or Chebyshev distance, and a jitter control. Scale, octaves,
  lacunarity, roughness, distortion, seed override, optional position input.
- **Patterns**: *Grid* (distance to the nearest cell centre or corner), *Bricks* (running bond or thirds, 2D
  or 3D), *Checker*, *Stripes*, *Dimples*, *Gyroid*, *Fractal* (Menger, Sierpinski), *Truchet* (arcs,
  diagonals) and *Honeycomb*, with Distance, Shape and Cell value outputs where they make sense.
- **Math**: add, subtract, multiply, divide, min, max, power, smooth min and max, dot product, absolute, ridge. The
  operator is a dropdown on the node (the mouse wheel over it steps through them). A float with a vector
  gives a vector.
- **Remap** (with optional clamp), **Value**, **Position**, **Normal** (gradient of a field).
- **Outputs**: Density and Normal.
- Pins and wires are blue for floats and yellow for vectors.

**Viewports.** A raymarched 3D view (solid, or cloud with a light) of a chosen shape displaced by the
graph, and a 2D slice view (any axis, grayscale or heatmap, with a 2 × 2 tile view to spot seams).
The eye in a node's header previews that node instead of the output.

**Speed.** Pipelines compile on a worker thread, so edits never freeze the UI. Any Noise, Math, Remap,
Grid or Bricks node can have a *cache* (button in its header): everything upstream is baked into a
tiling 3D texture for the preview, and the nodes it covers get a green outline. Exports ignore caches.

**Export**
- 2D image or 3D slice grid: PNG 8 or 16 bit, EXR 32 bit; density or normals.
- 3D OpenVDB (a float grid, written by Noiser's own minimal writer).
- `.noiser` math function: the graph as an op list plus test vectors, so a sampler in another host
  can check itself. The format is specified in [`docs/noiser-format.md`](docs/noiser-format.md). A Blender
  extension that builds Geometry Nodes groups from it is in [`blender/`](blender).

**Crash recovery.** Rolling backups; after a crash you are offered the last one on the next start.

**Meshes.** Import a low-resolution OBJ (File menu, or `noiser.exe --mesh file.obj`) and it appears as the
*Mesh* shape in the 3D view, shaped live by your graph. *Export displaced mesh* then writes a high-resolution
OBJ or PLY of the same surface (as many samples along the longest side as you ask for), with small floating
bits removed and the mesh's own size and position restored. The mesh can be made of overlapping parts. A
*displacement mask* varies how strongly the noise acts on each part of the mesh, as a weighted sum of terms: a ramp
along an axis, which way the surface faces, occlusion, thickness and curvature. See `PLAN.md` for how it works and
its limits.

**Batch.** Once a graph and settings look right on one mesh, *Batch displace meshes* applies them to many OBJ
files at once: pick the files and an output folder, choose the same seed for all, the global seed plus one per
mesh, or a repeatable random seed per mesh (the seed is in the file name), and start. Because every mesh is
scaled to a unit sphere first, the noise is the same size relative to each mesh. A file that fails is
reported and the rest continue; nothing is ever overwritten. From a script:
`noiser.exe graph.noisegraph --batch-out DIR --batch-file a.obj --batch-file b.obj`.

## Controls

| | |
|---|---|
| Node graph | Middle-drag pans, scroll zooms, left-drag on empty space box-selects, right-click adds a node |
| | Drag from a pin to connect; drag a connected input away to disconnect; drop a wire on empty space to pick a node for it |
| | Delete removes the selection, Ctrl+D duplicates it at the cursor |
| 3D view | Drag orbits, scroll zooms |
| Slice view | Scroll moves the slice |

## Project docs

- [`PLAN.md`](PLAN.md): architecture, decisions, measurements and the roadmap.
- [`docs/noiser-format.md`](docs/noiser-format.md): the `.noiser` file format.
- [`docs/pattern-nodes-plan.md`](docs/pattern-nodes-plan.md): the plan for more pattern nodes.

## Roadmap

Next: a Curve node, Truchet pipes, then rendering and material settings for the viewport (march steps,
lighting, material colour and roughness). See `PLAN.md` for the details.
