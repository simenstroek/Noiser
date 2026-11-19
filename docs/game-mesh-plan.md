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

## Mesh graph status (2026-11-18, night, v0.11.0)

Built: the Mesh graph tab, a second `Graph` with extra node kinds (Load Mesh with Field and Mesh outputs, Noise Graph, Remesh with adaptivity, Auto UVs, Transfer UVs, Bake with custom inputs, Split / Combine channels, Image levels, Mesh and Image outputs in OBJ, PLY, PNG 8/16, EXR), typed mesh and image wires, `meshgraph::lowered_mesh` (a Noise Graph node becomes a copy of its graph: unconnected generators and Position nodes read the position times the frequency, the output is multiplied by the amplitude; checked on the GPU against the standalone graphs), `MeshEvaluator::new_graph` (up to four loaded meshes as distance grids in one shader, the Normal step following the mesh's voxel), the executor `meshgraph::Runner` (memoized per node, worker thread, outputs written without overwriting, batch runs with `{batch}` and `{name}`), the quick Preview (first Remesh at 6,000 triangles, optionally redone after changes, using the noise graph open in the Noise tab as it is on screen), thumbnails of the written maps, undo, save with `.history` versions, drag and drop, and headless `--mesh-graph`. Tests run the design's example graph, a transfer, a preview and the output formats end to end on the GPU.

Not built: a live raymarched preview of the field in the tab, a relax step for transferred UVs, a better answer to tiny UV islands on bumpy meshes, a pass at very large noise amplitudes where the remesher degrades. The window was checked by screenshots (`tools/screenshot.ps1`), never clicked through.

Checked in Blender 4.5 (Cycles, from the command line): the 4,000-triangle monkey with its baked normal map (`--green-up`, tangent space, Non-Color) renders with the detail of a 100,000-triangle reference, where the same mesh without the map is smooth. So the green-up convention, the tangent frame and the UV orientation of the OBJ are right for Blender; the default green-down map is the same image with the green channel inverted (Unreal).

### Where the UV coverage goes (measured 2026-11-19)

Rock, 20k triangles, 351 islands (299 tiny): coverage 64.1% in 10 s. Padding is not the lever: 2 texels gives 63.4%, 4 texels 64.1%, 4096 px 65.9%. Packing efficiency is: xatlas best packing (`brute_force`) gives 70.6% (69.6% with 2 texels) in 35 s, so it is now on by default up to 50k triangles. At 100k triangles (1218 islands) it did not help (64.8% plain, 60.9% best packing, 25 s longer), so it is skipped above 50k. Transfer UVs now relaxes each island's interior (cotan Laplacian, borders pinned, flip-safe): rock mean stretch 1.18 to 1.13, 0 flipped. xatlas sometimes returns a non-square atlas (IslandRock: 2279 x 2449); stretching it to the square when the ratio is under 1.15 gains that much coverage (IslandRock 53.1% to 57.1%, rock 70.6% to 72.9% with best packing). Real assets through the batch run (20k triangles each): IslandRock_Complex_01 57%, vehicle SDF export 65%, leaves sculpt (thin open sheets, not a good fit for a volume) 60%. Chart smoothing was tuned again: with 10 rounds the UVs are made on a copy that differs from the true surface, and the rock's mean stretch was 1.57 (p95 3.2); at 3 rounds it is 1.2 to 1.3 with fewer islands (rock 298 against 351, coverage similar), so 3 is the default now (rock 20k: islands 351 to ~300, stretch 1.57 to ~1.2; 100k rock: stretch 1.44 to 1.26, coverage 64.8% to 68.0%). A Laplacian relax of the interiors after xatlas, against the true surface, did not help (mean stretch unchanged), so only the Transfer UVs path uses relax.

### Measured with the shipped defaults (v0.12.0: smoothing 3, best packing up to 50k triangles, atlas 2048, overlap counted at 2048)

| Mesh (20k triangles unless noted) | Islands (tiny) | Coverage | Mean stretch (p95) | Overlap |
|---|---|---|---|---|
| Rock (RockShape + noise) | 294 (234) | 66.2% | 1.25 (1.88) | 0.000% |
| IslandRock_Complex_01 | 116 (30) | 58.2% | 1.16 (1.39) | 0.000% |
| Vehicle SDF export | 39 (17) | 67.5% | 1.04 (1.07) | 0.000% |
| Leaves sculpt (open sheets) | 872 (705) | 62.1% | 2.47 (2.90) | 0.000% |
| Rock, 100k triangles | 1237 (1077) | 68.0% | 1.26 (1.63) | 0.000% |

The 0.001% overlap seen earlier was a counting artefact: the log counted at 512 px while islands are packed 4 px apart at 2048, so they touched at 512. The log now counts at the atlas resolution. Coverage between runs varies by a few points with any change to the charts, so single numbers are not exact; the 70% target is met only by some meshes. Graphs saved before v0.12.0 keep their stored *chart smoothing* 10 and *best packing* off; press reset on those two fields in the Auto UVs node.

### Tiny islands, second pass (2026-11-19, after v0.12.0)

The absorb step turned out to reject 249 of 290 candidate islands on the rock, almost all on density or winding: a one-triangle island that is welded along one edge had its third vertex pulled onto the line between the two twins (a collapsed triangle). Islands that hang off the neighbour's border (not enclosed by it) are now unfolded from it triangle by triangle with their own 3D shape, on the neighbour's side of the winding; enclosed ones are still spread between their twins. A latent bug showed up with it: merges could chain (an island that had absorbed another was itself absorbed, or used as a target), so undoing one left the other half-restored and overlapping. Merges no longer chain, a merge that leaves the 0 to 1 square is refused, and the overlap check now also refuses faces within one texel of another island's. Result at the shipped defaults: rock 20k 234 to 157 tiny islands (294 to 216 islands), IslandRock 30 to 22, vehicle 17 to 9, rock 100k 1077 to 666 (1237 to 831 islands); overlap 0.000% everywhere, no flipped triangles, coverage and stretch unchanged. What remains are islands whose unfolded shape would fold over itself on a bumpy surface; they stay as they are.

### Measured with the defaults of v0.13.0 (absorb with unfolding; overlap counted at 2048)

| Mesh (20k triangles unless noted) | Islands (tiny) | Coverage | Mean stretch (p95) | Triangles over 100x | Overlap / flipped |
|---|---|---|---|---|---|
| Rock (RockShape + noise) | 209 (150) | 66.2% | 1.25 (1.86) | 0 | 0.000% / 0 |
| IslandRock_Complex_01 | 108 (21) | 58.2% | 1.15 (1.38) | 1 | 0.000% / 0 |
| Vehicle SDF export | 28 (6) | 67.5% | 1.04 (1.07) | 0 | 0.000% / 0 |
| Leaves sculpt (open sheets) | 648 (482) | 62.2% | 2.42 (2.86) | 36 | 0.000% / 0 |
| Rock, 100k triangles | 795 (630) | 68.1% | 1.25 (1.62) | 11 | 0.000% / 0 |
| Rock transferred onto a 20k remesh (Transfer UVs) | 49 (17) | 62.9% | 1.16 (1.57) | 1 | 0.000% / 1 |

The log now reports the number of triangles stretched over 100x instead of the maximum, which was capped at 1000 and looked like an error; those are slivers of the remesh or zero-area UV triangles.

Tried and rejected after v0.13.0: packing the charts a second time after absorbing (xatlas `add_uv_mesh`, first pass without best packing, repack with it; the wrapper counts contiguous UV data in threes, so the UVs have to go in as strided bytes). Rock 20k: coverage 66.2% to 63.5%, overlap 0.000% to 0.130%, one flipped triangle, mean stretch 1.25 to 1.50, time 36 s to 30 s. The merges kept or undone differ when the first layout is not the best-packed one, and the repack does not help what they kept. Not kept.

