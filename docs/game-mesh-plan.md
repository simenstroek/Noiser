# Game-resolution mesh pipeline: plan

Goal: from a mesh plus a noise graph, export a game-ready mesh (1k to 100k triangles, good triangles, automatic UVs) and baked textures sampled straight from the SDF function, with no high-poly intermediate. A separate mesh graph comes last.

Decisions (agreed): build order is remesh, UVs, bake, then the mesh graph. Native C/C++ crates are fine (xatlas, meshoptimizer). Organic surfaces only, so no crease preservation. Unreal-style output: tangent-space normals (mikktspace), 16-bit PNG, 1k to 4k, optional packed ORM.

## Foundations (do first)

**F1. Point evaluator.** Today `MeshEvaluator::layers` evaluates the displaced field on a regular grid, slab by slab (`src/meshexport.rs`). The remesher and the baker need arbitrary points in, `(distance, gradient, optional channels)` out, in large GPU batches. This is the "field" abstraction: everything after it sees only `Field::eval(points) -> Vec<Sample>`. The mesh graph later replaces what is behind it and nothing else.

**F2. Quality report** (`src/meshquality.rs`, plus `--report` in headless). Objective numbers for any mesh and any UV set, so every algorithm is scored the same way:
- Mesh: triangle count, min and median angle, share of triangles under 20 degrees, aspect ratio, normal deviation between neighbours, max and mean distance to the true SDF surface (exact, from F1), and a watertight and manifold check.
- UVs, modelled on `nemi_uv_toolkit/tests/uv_quality.py`: coverage of 0..1 (target 70% or more), island count, micro-islands (a size threshold), overlap area (must be zero), stretch and area distortion.

Fixed test set: the rock, the monkey, a sphere and the 5-part rock.

## Phase 1: adaptive remesh (`src/remesh.rs`)

1. Coarse start: the existing surface-nets extraction at low resolution (this is why grid resolution stops limiting detail).
2. Incremental isotropic remeshing: split long edges, collapse short ones, flip toward the Delaunay criterion, relax tangentially, and after every step project each vertex back onto the exact SDF surface with a few gradient steps from F1.
3. Size field: target edge length from local curvature with an error bound (small triangles where the surface bends, large on flats). A single scale knob moves the triangle count; binary search on it hits a requested count within a few percent.
4. Low-poly mode (about 1k): silhouette-driven simplification with meshoptimizer's error-bounded simplifier on the remeshed result, then a cleanup pass. The detail goes into the bake, so this mode does not try to follow the noise.
5. Done when, on the test set at 1k, 5k, 20k and 100k: triangle count within 5% of the request, under 5% of triangles below 20 degrees, max distance to the true surface under a stated fraction of the bounding radius, and zero non-manifold edges. Thresholds get fixed after the first measurements.

## Phase 2: UVs (`src/uvgen.rs`)

1. xatlas through a Rust binding, with chart options tuned (normal deviation weight, maximum chart area, boundary length).
2. Merge pass for micro-islands: fold islands under the size threshold into the neighbour they share the longest border with when stretch stays under a limit; re-pack.
3. Repack at the end for coverage, with padding to match the texture size.
4. Existing UVs (stretch goal): read `vt` / `TEXCOORD_0` in the loaders. For each new triangle, pick one source triangle (nearest to the centroid) and take every corner's UV from it, duplicating vertices along seams so nothing stretches across two islands. Then a relax step. Expect it to work only for mild displacement; the report says when it does not.
5. Done when, on the test set: zero overlap, coverage of 70% or more, island count and micro-island count under limits we set from the first measurements. Compare against the Blender addon's numbers on the same meshes.

## Phase 3: bake from the SDF (`src/bake.rs`)

1. Rasterize the low mesh into its UV atlas, giving for each texel a surface point and an interpolated normal. A GPU compute pass or a CPU rasterizer, whichever is simpler to verify first.
2. Snap each point onto the true surface with F1, starting from the low-poly point. If the gap is larger than a threshold (big silhouette error at 1k), fall back to marching along the low-poly normal (a cage ray). Compare both on the test set before choosing.
3. Channels: normal (tangent space with mikktspace so it matches the engine), occlusion, curvature, depth, thickness, position, and custom. The first six reuse the mask code. Custom is a new graph output node, "Texture output (name)", which any value node can feed.
4. Island padding by dilating outward after the bake; 16-bit PNG output; packed ORM option.
5. Done when: baked normals from the 1k mesh, viewed on the 1k mesh, match the displaced SDF's own normals within a tolerance (checked numerically, by re-evaluating at random texels), there are no seam artefacts beyond the dilation width, and a render with the maps shows no visible gaps.

## Phase 4: the mesh graph (design after phase 3)

A second graph kind with field-valued nodes: load mesh, combine (the union/intersect/smooth modes already built), displace by a noise graph with a mask, and Lerp between fields. Mask terms (normal, curvature and the rest) become value nodes, so "blend a rough and a smooth rock graph by normal" is a Lerp. Texture output nodes feed phase 3. Detailed design waits until the F1 interface is settled, because the graph compiles down to that interface.

Agreed shape:
- Wire types: Field (GPU-evaluable distance plus values), Mesh (with optional UVs), Image (float RGBA, named channels), Value (float, Vec3). Field nodes compile to one shader and stay live; operation nodes (Remesh, Auto UVs, Transfer UVs, Baker) run on a Run button, cache by input and settings, and show progress.
- Baker: one source field, optional named custom inputs, checkboxes for normal, occlusion, curvature, depth, thickness, position. One image output per enabled channel.
- Image nodes between Baker and Output: split and combine channels, per-pixel math (the existing Math nodes broadcast over images), invert, remap, resize, dilate. Use case: store a normal map's R and G, drop B (rebuilt in the shader) and put other data in B and A. Data maps stay linear (no sRGB).
- Output: image or mesh (pin type follows the selector; switching disconnects), naming with `{name}` and suffixes for batches, format options (PNG 8 or 16 bit, EXR, mesh formats).
- Load Noiser Graph: reference to a `.noisegraph` file with an embedded snapshot and an Embed (detach) option. Double-click switches to the Noise graph tab with that graph loaded. The app gets two tabs, Noise graph and Mesh graph; several open noise graphs at once (a tab each) is deliberately left for later.
- Saving: Ctrl+S writes the mesh graph and every dirty noise graph it references, and refreshes every embedded snapshot. Before a noise graph file is overwritten, the old version goes to `<name>.history/` with a timestamp. A referenced file is the source of truth; the snapshot is only the fallback when the file is missing or changed on disk (then ask: use file, use snapshot, or keep both). Embedded (detached) graphs live only in the mesh graph file.

## References in the Unreal 5.8 source (`E:\Unreal Engine\UE_5.8`)

Read for the algorithms; Epic's licence does not allow copying code into this repo, so we reimplement. Paths are under `Engine\Plugins\Runtime\GeometryProcessing\Source\` unless noted.

- Remeshing: `DynamicMesh\Private\Remesher.cpp` (761 lines) and `QueueRemesher.cpp`: split/collapse/flip/smooth/reproject with one uniform target edge length (min 1.0x, max 3.0x of it). No curvature size field, so our adaptive size field is an addition, not a port.
- Simplification: `MeshSimplificationQuadrics.cpp`, `MeshClusterSimplifier.cpp`.
- UV generation: `DynamicMesh\Private\Parameterization\PatchBasedMeshUVGenerator.cpp`. Its steps: split into many small patches (by normal-weighted clustering), merge patches while stretch stays under a threshold and the normal deviation under 45 degrees, unwrap with ExpMap, align, pack. This is the closest match to what you described and gives direct control over island count.
- UV relax and solver: `Solvers\Internal\MeshUVSolver.cpp`.
- Packing: `Engine\Source\Runtime\GeometryCore\Private\Parameterization\UVPacking.cpp` (1076 lines).
- xatlas: UE vendors it at `GeometryAlgorithms\Private\ThirdParty\xatlas` (MIT licence), so it is an Epic-approved fallback.
- Baking: `DynamicMesh\Private\Sampling\MeshMapBaker.cpp`, as the model for evaluator and dilation structure.
- Tangents: `GeometryAlgorithms\Public\DynamicMeshMikkTWrapper.h`.

## Risks

- A hand-written remesher is the largest piece of new code. The metrics from F2 are what make it tractable; build them before the algorithm.
- xatlas gives zero overlap and good packing but many charts on organic meshes; island count is a tuning problem, not a guarantee.
- Bake cost: 4k by 4k texels times about 7 field evaluations each (value plus gradient) per channel. Fine on a GPU, but needs batching and a progress bar.
- At 1k triangles the noise is mostly in the normal map, so snapping from the low mesh to the true surface can be far; that is what the cage-ray fallback in phase 3 is for.

## Status (2026-11-18, evening)

Released as 0.9.0. Done, tested and pushed: the point evaluator (`MeshEvaluator::points` and `distances`), the quality reports (`meshquality.rs`: triangle shape, folds, topology, distance to the true surface, UV coverage, islands, overlap, stretch), the adaptive remesher (`remesh.rs`), UV generation (`uvgen.rs`) and the baker (`bake.rs`). All of it is reachable from the Mesh tab's export window and from headless (`--remesh N --uv --bake all`). Not yet: the mesh graph (phase 4), UV transfer from the input mesh, the custom bake channel.

Against the acceptance numbers the UV work is not finished: coverage is 58 to 66% (70% or more wanted; best packing adds about 5 points at 8 to 10 times the time), and bumpy meshes keep many tiny islands (rock at 10k triangles: 223 islands, 167 tiny; monkey: 53 and 25). Checked that these are not caused by bad triangles: fewer than 1% of faces are slivers or folds and almost none lie in tiny charts, so the cause is xatlas's chart growth, which `max_cost` does not change. Tried and rejected: merging small charts and re-running, segmenting on a heavily smoothed copy, uvunwrap. Also the remesher: about 1.5% of triangles of the monkey at amplitude 0.08 and 5% of the rock's face against the field at their centres (the uniform mesh has about 1%), and it degrades at amplitude 0.2.

What was learned:
- Start the remesher from marching tetrahedra, not surface nets: surface nets leave non-manifold edges that block the edge operations. The extractor is `surface::marching_tets`.
- Noise finer than the triangles makes naive projection fold the surface. Three guards fix it: a vertex moves at most half an edge per step, no operation may create a fold (neighbouring normals more than about 125 degrees apart), and sizes are bounded to 1/4 to 5 times the uniform edge length so a crease does not eat the triangle budget. Blurring the field was tried and made things worse.
- The count converges by re-solving the error budget each round against a measured correction factor (the equilateral estimate is about 1.5 times too low).
- The p95 neighbour-normal angle is a poor metric on hard-edged inputs (a true 90 degree crease is correct); fold edges and the distance to the true surface are the useful numbers.
- xatlas on a bumpy surface cuts hundreds of tiny charts; charting a smoothed copy of the surface (same triangles, Taubin smoothing) removes most of them. A pure Rust unwrapper (uvunwrap) gave the same island counts and was dropped. Tiny islands that remain are mostly in hidden crevices and on undersides.
- The baker was checked by rendering the 4k-triangle monkey with its baked normal map against a 100k-triangle reference: the detail matches, so the tangent frame and green channel convention (green down, Unreal) are right.

## Mesh graph status (2026-11-18, night)

Built: the Mesh graph tab, a second `Graph` with extra node kinds (Load Mesh, Noise Graph, Remesh, Auto UVs, Bake with custom inputs, Mesh and Image outputs, Split and Combine channels, Image levels), typed mesh and image wires, `meshgraph::lowered_mesh` (a Noise Graph node becomes a copy of its graph: unconnected generators and Position nodes read the position times the frequency, the output is multiplied by the amplitude; checked on the GPU against the standalone graphs), `MeshEvaluator::new_graph` (up to four loaded meshes as distance grids in one shader), the executor `meshgraph::Runner` (memoized per node, remesh, UVs and bake on a worker thread, outputs written without overwriting) and headless `--mesh-graph`. A test runs the design's example graph end to end on the GPU.

Not built: a live 3D preview of the field in the tab, transfer of existing UVs, and clicking through the new tab on screen has not been done (no way to drive the window from here).
