# Noiser for agents

Noiser is a toolkit for making 3D content from signed distance fields: SDF shapes (Composer), noise graphs, terrains
(Landscape), and mesh graphs that combine them and write Nanite-ready meshes and texture maps. Everything is a file, and
everything below works without opening the app.

## The loop

1. **Find out what exists:** `noiser --schema` prints every node kind (settings with defaults, input/output counts, what it does),
   the enum values, and the file formats, as JSON. Read it instead of guessing setting names.
2. **Write a graph as a spec**, a short name-based JSON, and build it:
   `noiser --build spec.json out.meshgraph` (or `.noisegraph`). Settings you leave out keep their defaults; unknown kinds, unknown
   settings, bad ports and mismatched pin types are rejected with a message naming the problem. The result is validated.
3. **Check it:** `noiser --validate FILE` (`.noisegraph`, `.meshgraph`, `.sdf`, `.landscape`): unconnected required inputs, missing
   files, cycles, a missing output node. Exit code 1 on errors.
4. **Run it:** `noiser --headless --mesh-graph FILE.meshgraph` writes the meshes and images (log next to the file as `FILE.log`).
   A `.noisegraph` is run in batches with `--headless GRAPH --batch-out DIR --batch-file MESH` (see `--help` text on error).
5. **Look at the result:** `noiser --preview FILE OUT.png [--size N] [--up y]` renders a `.noisegraph` (a slice), a `.sdf` (front, side,
   top, sphere-traced) or a mesh (`.obj`, `.ply`, `.glb`: front, side, top). Then read the PNG. Do this after every step that
   changes geometry; the numbers lie, the picture does not.
6. **Read an existing graph:** `noiser --describe FILE` prints it as the same spec format (nodes named kind+id), so you can edit it
   and `--build` it back.

The spec: `{"nodes": {"name": <node>}, "edges": [["from", "to", input_port], ["from", "to", input_port, output_port]]}`. A node is
`"Separate"` or `{"Noise": {"scale": 4.0}}`. Examples that build and run are in `examples/Specs/` (a test builds them all).
Relative file paths in nodes are relative to the folder of the graph file you build into, so build into the folder the paths
expect (examples/Specs builds beside `../SDFs` and `../Noises`).

## What a mesh graph does

The graph computes one field. `Math Add` of a shape field and a noise field roughens the shape; `SmoothMin` blends shapes;
`LoadSdf` and `Scatter` bring in `.sdf` shapes (Scatter plants copies on a terrain by density, size, slope and elevation, or in a
ring facing the middle); `Remesh` turns the field into triangles (`LoadMesh` with `chunks: 4` and a heightmap makes a terrain in 4x4
Nanite chunks); `AutoUv` (`planar: true` for terrain) unwraps; `Bake` makes normal/occlusion/curvature maps from a field; `OutMesh`
and `OutImage` write files. A graph needs at least one output node to do anything.

## From shape to Unreal

SDF Composer (`.sdf`, made in the app) -> Landscape tab (heightmap, masks, water; `noiser --headless --landscape-export P.landscape
OUT.png`) -> Mesh graph (terrain + `Scatter` + noise + `Remesh` in chunks + `Bake`) -> `tools/unreal_import_terrain.py` builds a
level (chunks, material, lights, water). `tools/demo/run_eiffel.sh` does all of it for the Eiffel/sphinx scene, and
`docs/terrain-workflow.md` explains each step and the pitfalls. `tools/unreal_shots.py` takes screenshots of the level.

## Things that cost time

- **Units.** A mesh graph's field is in the units of its first `LoadMesh` (`unit_cm`: 1 = centimetres, 100 = metres). Lengths on
  `Scatter` and `LoadSdf` are in those units: a 150 m sphinx is `15000` with `unit_cm` 1. The app shows metres next to the number.
- **Up.** Terrains and fields are Z-up; `.sdf` files are Y-up (set `z_up` on `LoadSdf`; `Scatter` handles it). OBJ files from the
  SDF side are Y-up: preview them with `--up y`.
- **Standing on terrain.** `LoadSdf` with `snap` stands the shape on the lowest ground under it, `snap_high` on the highest (a
  tower with legs either side of a river). Without one of them the offset is absolute.
- **Remesh budget.** `triangles` is per output: terrain chunks share it by the steepest chunk. Start small (50k) to find mistakes,
  then raise it. A 250k-triangle, 2048-map terrain exported in a few minutes on the dev machine.
- **GPU backend.** Vulkan is preferred; DX12 compiled the big shader for minutes. Do not force DX12.
- **Material maps.** In Unreal, Occlusion/Curvature/Objects maps are data, not colour: import them linear (the import script does).
- **Do not kill `noiser.exe` by name.** The user may have the app open; find your own process by id.
- **Files are the interface.** Library root: the user's Noiser library folder (SDFs, Noises, Terrains, Mesh Graphs); examples in
  `examples/`. The user curates their own examples by hand, so do not rewrite them; make new ones in a scratch folder.

## Working on the code

`cargo build --release` writes `target/release/noiser.exe` (one build folder, one exe). **Do not run the whole test suite**
(`cargo test --release` with no filter): it takes several minutes at best and has run past 20; it is not needed to check a
change. Run only the tests for the area you changed, by name filter (`cargo test --release agent_cli`,
`cargo test --release landscape`, `cargo test --release gpu_terrain`); GPU tests need an adapter. Known failure, not yet
fixed: `gpu_tests::radial_positions_and_displacement_outputs`. Node kinds live in
`src/graph.rs` (`NodeKind`); adding one means adding it to `catalogue()` in `src/agent_cli.rs` too (a test fails until you do).
`PLAN.md` and `docs/` hold the design notes; `docs/noiser-format.md` is the `.noiser` math-function format.

## MCP

`tools/mcp/noiser_mcp.py` is a small stdio MCP server over the commands above (schema, describe, build, validate, preview, run
mesh graph, landscape export). It needs only Python 3 and a built `noiser.exe`; set `NOISER_EXE` if it is not at
`target/release/noiser.exe`. Register it with `claude mcp add noiser -- python tools/mcp/noiser_mcp.py`.
