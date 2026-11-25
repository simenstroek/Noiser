# Terrain maps, erosion and layer nodes, stamps: plan

Status: agreed 2026-11-26. Phases 0 to 2 done (see Progress at the end); defaults decided: imported maps up to 2049 a side, Erosion auto-run on by default. Covers five features picked after looking at TerraForge3D:

1. Erosion output maps (flow, deposition, wear).
2. Unreal paint-layer weightmaps from the Landscape.
3. Shape stamps: `.sdf` shapes pressed into the terrain.
4. Wind erosion (and room for more erosion kinds).
5. Importing existing heightmaps.

Decision: erosion and layers live in the **Landscape graph as nodes**, as in other terrain tools, not in side panels.
The plan is built around that.

## What the user sees

```
Heightmap ──┐
Noise ──────┴─ Add ─ Erosion ─┬─ Height ─ Add ─ Output (height)
Stamp ─────────┘              │         Noise ┘
                              ├─ Deposition ─ Select ─ Layer "sand"
                              ├─ Flow ─────── Select ─ Layer "wet"
                              └─ Wear
Slope ─ Select ─ Layer "rock"
```

- **Erosion** is a node. It goes anywhere in the chain, more than once if wanted. Its outputs are the eroded height and
  its maps (flow, deposition, wear). Noise added after it keeps full detail. Its settings (and presets) are the
  node's; a badge on the node shows *up to date*, *out of date* or *running* with progress. It reruns on demand, or
  automatically after a pause, and its result is cached by everything upstream plus its settings, so editing nodes
  downstream never reruns it.
- **Heightmap** (an imported image) and **Stamp** (an `.sdf` shape) are source nodes. **Mask** (painted) already is.
- **Select** keeps a range of its input with soft edges: the selector artists know from Gaea / World Machine.
- **Layer** is an output node: a name, a preview colour, and a 0..1 weight in. Exports normalise the layers to sum to
  1 per cell and write one weightmap each; the 3D view shows the layer colours.
- The side panel keeps the terrain settings (size, height, working resolution), the selected node's settings and the
  exports. The erosion panel goes away; old projects open with an Erosion node (their settings) in front of the output.
- These nodes are 2D-terrain nodes: the palette offers them in the Landscape graph only. A noise graph that has one,
  opened in the Noise graph tab, says so plainly.

## How it works underneath: terrain maps and grid boundaries

The noise graph is a function the GPU evaluates at any point and resolution. Erosion is a simulation over a whole
grid. So an Erosion node is a **grid boundary**: everything upstream of it is sampled onto the working grid and eroded,
and its outputs become **terrain maps**, named grids that downstream nodes sample bilinearly.

Painted masks are already exactly that: named 1024² grids in the parameter buffer, read by the Mask node
(`codegen::MASK_RES`, `noise.wgsl` `painted_mask`, `Evaluator::with_masks`). **Phase 0 generalises them** so every grid
(painted masks, imported heightmaps, baked stamps, erosion outputs) is one kind of thing:

- A `TerrainMaps` store (name -> grid, side, source) replaces the bare mask map that is passed around today.
- One upload path packs them into the parameter buffer, each starting at its own offset with its own side, so a 4096
  erosion output and a 1024 painted mask sit side by side.
  - Offsets are passed in units of 256 floats (maps padded to that), because `array<f32>` holds whole numbers exactly
    only to 2^24.
  - The total is checked against the device's storage binding limit (128 MB with wgpu's defaults, which the app uses
    today), with a plain message about what to shrink. If needed, the app can request the adapter's real limits.
- One shader sampler (`terrain_map`: bilinear, plus an optional offset/scale/rotation for stamps) used by every
  map-reading node.
- The Mask node keeps its variant and file name (`"Mask": {"name": ...}`): the agent catalogue (`agent_cli.rs`, enforced
  by a test), `--schema`, specs, examples and demo scenes use it. New fields are optional with `serde(default)`.

Evaluating a Landscape graph with grid boundaries: walk the graph from the outputs, find Erosion (and later other grid)
nodes, evaluate each one's upstream into a grid (inner ones first), run it (or take its cached result), register its
outputs as maps (`erosion#<node id>.height`, `.flow`, ...), and rewrite the node in the compiled graph as Map reads.
Downstream code is then an ordinary graph over maps. This lowering step sits next to `meshgraph::lowered_mesh`, which
already rewrites Noise Graph nodes before compiling.

## Phases

Each phase is usable on its own and tested before the next.

| Phase | What | Size |
|---|---|---|
| 0 | `TerrainMaps`: per-map offset and side, binding check, the shared sampler; painted masks moved onto it; Mask shows as "Map" with a grouped menu | small to medium |
| 1 | **Erosion node**: the grid boundary and lowering, a cached run per node, the badge and auto-run, outputs height/flow/deposition/wear (flow as its own small pass: the hydraulic shader already uses 7 of the 8 default storage bindings); the panel goes, old projects get the node; presets move onto the node | medium to large |
| 2 | **Select and Layer nodes**, the weightmap export (8-bit PNG per layer next to the heightmap, listed in the sidecar), the 3D view in layer colours; checked against Unreal in `D:\UnrealProjects\NoiserTesting` before the format is fixed | medium |
| 3 | **Heightmap node**: PNG 8/16-bit or EXR, Noiser heightmaps back in metres via their sidecar, sides up to 2049, path relative to the project | small |
| 4 | **Stamp node**: an `.sdf` baked to its top surface (traced along the shape's −Y, since `.sdf` is Y-up and terrain Z-up; size = widest side in metres, plus a height factor), placed with the shared sampler's transform; add, max or cut | medium |
| 5 | **Wind** and more erosion kinds: a setting or sibling node of Erosion, each kind its own pass and bind group within the binding limit; chaining nodes is the "stack" | medium |
| 6 | **Into the Mesh graph**: maps travel beside the heightmap (as more layers of the terrain's height texture, so no new bindings), Scatter density by a map, Bake writes layer weights per chunk | medium |

## Avoiding duplicate code

- One map store, one upload path, one sampler for masks, imports, stamps and erosion outputs.
- One normalise-and-write path for exporting any map as a PNG.
- One CPU slope helper (today slope is worked out separately in the GPU view and the map hillshade).
- Erosion kinds share the buffer and pipeline setup in `landscape_erosion.rs`; a kind adds entry points, not a second
  simulator.
- The grid-boundary lowering is one step that later grid nodes (blur, flow-based effects) reuse.

## Open questions

- Imported maps up to 2049 a side: enough? (The terrain is still sampled at the working resolution.)
- Stamps only raise or cut height. Caves and overhangs stay in the Mesh graph, where they already work.
- Auto-run for Erosion nodes: on by default, or a Run button only?

## Progress

- **Phase 0, 1:** done (terrain maps store; Erosion node with grid boundary, cache, badge and Run, old projects migrated).
- **Phase 2:** done. *Select* (range with soft edges: height, slope in degrees, or a map) and *Layer* (output; name and
  colour) nodes. `export_terrain` writes the heightmap and one 8-bit weightmap per layer (`<heightmap>_<layer>.png`,
  normalised, adding up to 255 per cell), listed under `layers` in the sidecar. The 3D view paints the layers' colours.
  `noiser --build spec.json out.landscape` (with an optional `"settings"` object) and `--preview X.landscape OUT.png`
  (map and 3D view side by side) work from the command line. Weightmap import into Unreal is checked in the end-to-end
  run (UE 5.8 is at `C:\UnrealEngine\UE_5.8` on this machine, not `E:`).
- **Phase 3:** done. *Heightmap* node (PNG exported by the tab with its sidecar returns in metres; other 8/16-bit PNGs
  are 0..1 of the height; EXR as is; `scale`, `offset`; up to 2049 a side; relative to the project's folder). It is read
  when the graph is prepared and becomes a Map (`heightmap#<id>#<hash>`), so erosion caches see a changed image.
- **Phase 4:** done. *Stamp* node (`.sdf` traced from above along its -Y into a height and a cover map on the working
  grid: size = widest side in metres, height factor, centre, turn; modes Add, Max, Cut; the last 8 bakes are cached).
- **Phase 5:** done. The Erosion node has a *kind*: Water (as before) or Wind (`landscape_wind.rs`: saltation with a
  lee shadow, material hops downwind, dunes build behind crests, windward faces wear, thermal slides after; its own
  shader and bind group; same outputs and maps). Chaining Erosion nodes is the stack.
- **Phase 6:** done. The terrain's Load Mesh node has `maps`: paint layers (from the sidecar's `layers`) loaded as extra
  layers of the heights' 3D texture (no new bindings; each costs as much video memory as the heights). *Terrain layer*
  (Mesh graph node, `TerrainLayer { name }`) reads one at a position: wire it to a Bake node's extra inputs for per-chunk
  layer weights. Scatter has `density_map`, a layer that scales its density.
