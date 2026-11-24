# Landscape tab: research and plan

Status: proposal, 2026-11-24, agreed with these defaults: world 2 km x 2 km, 4 x 4 chunks of 500 m, 1M triangles per chunk, heightfields up to 4k, the Unreal import script as an extra after the first Unreal test (the chunk export writes a manifest for it), Unreal's own Nanite Landscape left as a plain option.

**Progress.** P0 started: the tab, sampling a noise graph over the terrain, a 3D preview, and the Unreal heightmap export (16-bit 4033 PNG plus a sidecar) are in. Authoring is in (painted masks on a map view; Mask, Slope and Noise Graph nodes in the graph; project files that keep the masks), so a canyon or a ridge is a painted mask times a noise graph, and slope or height ranges pick where a different noise graph applies. P2 has started: GPU hydraulic and thermal erosion on a working grid (as a stage between the graph and the exports, with the settings in the panel, not as graph nodes yet); gentle by default, strong settings show the pipe model's axis-aligned gullies, so an eight-neighbour model is the next quality step, then flow, deposit and slope masks. P1's chunk export is in too (regular-grid chunks up to the triangle budget, planar UVs, per-chunk normal / occlusion / curvature maps, the manifest, and the UDIM pack with light stand-ins for the combined mesh). Still open from P1: the *Load Landscape* node for the Mesh graph (3D detail on top of the terrain), adaptive meshing and the Unreal import script. P0 deviates from section 4 on purpose: the terrain is the existing noise graph evaluated over one tile (so every noise, pattern and math node already works), not yet a separate heightfield-pass engine. That engine arrives with erosion in P2, where grid passes are needed.

**3D chunks through the Mesh graph (started).** A *Load Mesh* node given a Landscape heightmap PNG (with its `.json` sidecar) is the ground: field = height above it, in the node's units. Wire it, with Load Mesh / Load SDF nodes and noise graphs combined onto it, into a *Remesh*: with a terrain the Remesh cuts the surface into `chunks` x `chunks` chunks (a node setting) by surface nets on one lattice that every chunk shares, so borders have the same vertices and normals come from the field. The triangle budget is spent on the steepest chunk. *Auto UVs* has a planar-from-above mode (instant; chunk i is UDIM tile 1001 + i), Bake works on the chunk as on any mesh, and a run writes `<name>_<row>_<column>` files and a `terrain_chunks.json` manifest. Measured on a 4033 x 4033 terrain (2 km, 16 chunks): 1M triangles per chunk, 1.2 s each to extract; a full run of 16 about 19 s with the OBJ writes. One 1M-triangle chunk baked at 4096 px: normal 10 s, curvature 14 s, ambient occlusion 50 s. Not yet: adaptive density, skirts, sparse sampling of the empty space above the ground (not needed so far), a pivot at the chunk corner, cracks in Unreal untested.

## 1. Goal

A fifth tab next to *SDF Composer*, *Noise graph*, *Mesh graph* and *Mesh*, for building terrain. Two kinds of output, both for Unreal Engine 5.8:

1. **Heightmap landscape**: a 4k heightfield (up to 4033 x 4033 samples) plus weightmaps, importable with Unreal's Landscape import.
2. **Nanite partitions**: the terrain cut into chunks, each its own mesh of up to about 1 million triangles, textured through one UDIM set, for use as Nanite static meshes. Chunks can carry real 3D detail (overhangs, caves, rock faces) that a heightmap cannot.

It is not a world-scale tool in the first versions: 4k per heightfield is the size limit, bigger worlds are a later tiling step.

## 2. What the other tools do

Sources: [Gaea 2.0 (80.lv)](https://80.lv/articles/rewritten-engine-new-terrain-tools-ue5-bridge-gaea-2-0-unveiled), [Gaea 2.0 (CG Channel)](https://www.cgchannel.com/2024/08/quadspinner-releases-gaea-2-0), [World Machine features](https://world-machine.com/features.php), [World Machine Dragontail Peak (CG Channel)](https://www.cgchannel.com/2026/07/world-machine-software-ships-world-machine-dragontail-peak/), [World Machine tiled-erosion thread](https://forum.world-machine.com/t/better-erosion-support-in-big-maps/4082), [World Creator features](https://www.world-creator.com/en/features.phtml). These are search summaries, not the vendors' full documentation.

| Feature | Seen in | Decision |
|---|---|---|
| Node graph on heightfields (Gaea also offers layers) | Gaea, World Machine | Yes: reuse our node editor |
| Hydraulic erosion, sediment; Gaea 2 "Erosion 2"; a Rivers node | Gaea, World Machine, World Creator | Yes, phase 2: the core of the tab |
| GPU simulation fast enough to see change as you edit | World Creator | Yes: we are GPU-only already |
| Masks from height, slope, curvature, flow, deposits | World Creator, Gaea | Yes: they feed texturing and displacement |
| Macros / custom devices | World Machine | Yes: a `.noisegraph` is already a reusable node |
| Tiled builds beyond memory, with overlap and blending (erosion shows seams if not handled) | World Machine | Later; design for it now |
| Terrain with overhangs and caves (vector displacement, new in World Machine) | World Machine | **Our natural advantage**: our graphs are 3D fields. See section 5 |
| Splatmaps, colour maps, heightmap import and stamping | World Creator, Gaea | Yes: export of weightmaps; import and stamps later |
| Painted masks, brushes | World Creator | Later: large UI cost |
| Terraces, strata, craters, dunes, thermal erosion, snow | all (general knowledge, not from the searches) | Yes, in phases 1 to 4 as nodes |

## 3. Facts from the Unreal 5.8 source (checked on this machine)

Path: `E:\Unreal Engine\UE_5.8\Engine\Source`.

- **Landscape height encoding** (`Runtime/Landscape/Public/LandscapeDataAccess.h`): heights are 16-bit, 32768 is zero, one step is `1/128` of a unit times the actor's Z scale. At the default Z scale of 100 that is about +/-256 m in 0.78 cm steps. The exporter must therefore choose a Z scale that fits the terrain's height range and write it to a sidecar file (location, XY scale, Z scale).
- **Valid landscape sizes** (`LandscapeConfigHelper.cpp`): a component has 1 or 2 sections of 7, 15, 31, 63, 127 or 255 quads, and the landscape is `components x quads + 1` samples across. So 4096 is not valid. 4033 (64 components of 63 quads), 4065 (32 of 127) and 4081 (16 of 255) are. **The 4k export resamples to 4033 by default.**
- **Import formats** (`Editor/LandscapeEditor/Private/LandscapeFileFormat{Png,Raw}.cpp`): 16-bit grayscale PNG (the importer warns about 8-bit or colour PNGs) or RAW 16-bit; weightmaps are one 8-bit grayscale file per layer.
- **Landscape can be Nanite itself** (`LandscapeProxy.h`, `bEnableNanite`): a second path for the heightmap output that needs nothing from us beyond the heightmap.
- **Nanite limits** (`Shaders/Shared/NaniteDefinitions.h`): clusters are at most 128 triangles and 256 vertices; resources address up to 2 GB of root pages or 4 GB of streaming pages. A 1M-triangle chunk is about 8,000 clusters, far below any limit. Nanite also has assemblies (`NaniteAssemblyBuild.cpp`); not needed here.
- **Not yet confirmed:** how Nanite treats open mesh borders when simplifying. The builder tracks external edges per cluster (`Cluster.cpp`), but whether a chunk's outer border stays locked is unverified. Distant cracks between chunks are the main risk of the partition output (section 8); we will test it with a real import before building chunk features on top of it.

## 4. Engine: a second kind of graph

Our graphs compile to one function of a 3D point, `f(p)`, evaluated independently per point. Erosion, flow and thermal passes cannot work like that: they are iterative passes over a grid where each cell reads its neighbours.

So the Landscape tab gets its own engine:

- **Heightfield** image type: an N x N `R32F` texture (N up to 4096), plus mask images (`R32F` or `R8`). A new `PortType::Heightfield` (and `Mask`), coloured like the mesh and image types, in the same graph and editor, as the Mesh graph already does with its own node set.
- Nodes are **compute passes** over textures, with ping-pong buffers for iterative nodes (erosion, thermal, blur). One WGSL pass per node, scheduled in graph order; results are cached per node, like the existing *Cache* node, so editing a late node does not re-run the early ones.
- **Determinism:** every node takes the graph's seed (and the existing seed modes), so a build is repeatable.
- **Axes and units:** the heightfield lies in X/Y with height up. The tab states its up axis (Z-up for Unreal exports) and unit scale (cm), and converts explicitly where it meets the Y-up graphs. This is the part that has already caused trouble elsewhere, so it is an explicit setting from the start.
- **Preview:** a displaced grid in the tab's 3D view (cheap, fine at 4k), with a mask overlay chosen from a menu. The ray-marched view is not needed for heightfields; the Mesh graph's field view already ray-marches 3D detail.

## 5. How it ties into the existing tabs

1. **Noise graph to landscape.** A *Noise Graph* node samples a `.noisegraph` as a 2D height or mask source (a slice at a chosen Z, or the x/y of a 3D noise with the seed on Z). Every noise, pattern, Worley, curve and remap becomes a terrain generator, and the tiling analysis tells you when the result tiles.
2. **SDF Composer to landscape.** A *Stamp* node rasterises a `.sdf` shape from above (top height of the shape) and blends it in: mesas, cliffs, rock formations, building platforms.
3. **Landscape to Mesh graph (the 3D bridge).** A *Load Landscape* node gives the Mesh graph a field `f(p) = height of p above the terrain`. Everything downstream already exists:
   - the Noise graph displaces it in 3D, so cliffs get overhangs and caves the heightmap cannot hold;
   - Remesh makes the adaptive mesh;
   - Bake gives normal, AO and curvature maps;
   - batch runs and UDIM pack handle many chunks.
   This is the feature the other tools do not have in this form.
4. **Masks both ways.** Slope, curvature, flow and deposit maps are exportable as textures, and can drive displacement amplitude in the Mesh graph the way mask options already do in the mesh export.
5. **Instances.** Place the rocks of the existing batch and UDIM pipeline on the terrain by mask, and export transforms (position, rotation, scale).
6. **Shared infrastructure that comes for free:** editor (undo, copy and paste, frames, pins, Random node, uniform-value colours), library, prefs, backup, the headless runner, the seed modes.

## 6. Outputs

### A. Heightmap landscape (up to 4k)

- Internal resolution 4096 x 4096 (or 4097); export profile **Unreal Landscape** resamples to 4033 x 4033 (or 4065 / 4081, selectable).
- `heightmap.png` (16-bit grayscale; RAW 16-bit as an option), plus a sidecar JSON with the values to type into the import dialog: location, X/Y scale in cm per quad, Z scale chosen to fit the range.
- Weightmaps: one 8-bit PNG per layer, named after the layer (slope, height bands, flow, deposit, ...), from the graph's mask outputs.
- Also EXR (32-bit float) for other tools and a baked normal map.

### B. Nanite partitions (up to about 1M triangles per chunk)

- **Partition:** a chunk grid over the terrain; the settings are the chunk size (metres) and the triangle budget per chunk (default 1M, adjustable).
- **Meshing:**
  - Regular-grid chunks first: `sqrt(budget / 2)` quads per side, about 707 for 1M triangles, sampled from the field. The heightmap is interpolated, so a chunk can be finer than the 4k source and get its extra detail from the noise graph.
  - Adaptive chunks next, using the existing Remesh (denser where the surface bends), for chunks with 3D detail.
- **Seams:** every chunk samples one global field at identical border positions, so border vertices match exactly; normals come from the field gradient, not the mesh, so lighting has no seam. The remaining risk is Nanite simplification at the borders (see section 3 and 8); an optional skirt along borders is the fallback.
- **UVs and textures:** each chunk gets planar UVs filling 0..1, so chunk `i` is UDIM tile `1001 + i`. The existing Bake (normal, AO, curvature, plus the terrain masks as custom maps) bakes per chunk, and the existing **UDIM pack** tool puts all chunks and maps into one tileset with one material for Substance and Unreal. No unwrapping is needed.
- **Files:** OBJ and GLB writers exist (GLB is likely faster to import at 1M triangles; to be measured). We cannot write FBX.
- **Unreal side:** a short Python or Blueprint import script (later) that imports the chunks as Nanite static meshes at the right transforms, so the partition drops into the level in one step.

## 7. Phases

Each phase ends with something usable, and only the tests for the code it touches.

| Phase | Deliverable | Done when |
|---|---|---|
| **P0 Foundation** | Tab, `Heightfield` and `Mask` types, compute-pass scheduler with per-node cache, 3D grid preview. Nodes: Noise Graph source, Value, Math, Remap, Curve, Blend, Slope, Height, Output. Export A: 16-bit PNG, sidecar, 4033 resample | A noise graph becomes a heightmap that imports into Unreal with the right size and scale |
| **P1 3D bridge and chunks** | *Load Landscape* node in the Mesh graph; chunk partition with regular-grid meshing and the triangle budget; planar chunk UVs; per-chunk bake; hand-off to UDIM pack | A 4x4 chunk terrain of 1M-triangle chunks, textured through one UDIM set, imports as Nanite meshes; the seam test is run in Unreal |
| **P2 Simulation** | GPU hydraulic erosion (grid based, seeded), thermal erosion, sediment; flow, deposit and curvature masks; weightmap export | Erosion looks plausible at 4k in a few seconds; masks export as weightmaps |
| **P3 Stamps and shaping** | SDF *Stamp*, terraces, strata, craters, dunes, warp, ridged and billowy generators | A mesa from the Composer sits on eroded terrain |
| **P4 Chunks, adaptive** | Adaptive remesh per chunk with border-matching, optional skirts, 3D-detail displacement presets (rock faces, overhangs) | A cliff chunk has overhangs and still matches its neighbours |
| **P5 Extras** | Rivers, instance scatter and export, heightmap import and stamping, Unreal import script, tiled builds beyond 4k, painted masks | As needed |

## 8. Risks and open decisions

- **Erosion quality and speed** are the hard part. The algorithms are public (GPU hydraulic and thermal erosion); making them look good and run in seconds at 4k will take iteration.
- **Nanite chunk borders** (see section 3): possible cracks at distance. Test early (P1); fallbacks are skirts or overlap strips.
- **Memory and time at 1M triangles per chunk:** a 4x4 grid is 16M triangles. Meshing, baking and writing must stream chunk by chunk, and import time in Unreal is untested.
- **Z scale precision:** 16 bits over a large height range gives coarse steps (0.78 cm at scale 100). A tall mountain needs a larger scale and coarser steps; the exporter should report the step size so it is a visible choice.
- **Heightmap vs 3D detail:** where chunk detail comes from the noise graph, the heightmap export and the chunk meshes differ (the heightmap has no overhangs). The tab should say so, and the weightmaps should come from the same masks for both.
- **Open:**
  - Default world size and chunk size (proposal: world 2 km x 2 km, 4 x 4 chunks of 500 m, or tell me the real numbers).
  - Whether to also offer Unreal's own Nanite Landscape as the heightmap-path default (it needs nothing from us).
  - Whether the first version must include the Unreal import script.

## 9. Testing

- One small test per new node (compute pass against a CPU reference on a tiny grid), a golden check for the erosion on a fixed seed, and tests for the export profile (valid sizes, height encoding round trip: encode then decode within one step).
- Chunk tests: border vertices of neighbouring chunks are identical, the triangle count stays within the budget, UDIM tile numbers follow chunk order.
- Per the project's testing policy, only the tests for the changed area run, not the whole suite.
