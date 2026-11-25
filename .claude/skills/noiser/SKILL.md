---
name: noiser
description: Use when asked to make or change Noiser graphs (.noisegraph, .meshgraph), SDF shapes or terrains, to build 3D content from signed distance fields, to export Nanite terrain chunks to Unreal, or to check Noiser output visually. Drives the noiser.exe command line (schema, build, validate, run, preview).
---

# Noiser

Read `AGENTS.md` at the repo root first: it is the full guide (commands, spec format, units, pitfalls). The short version:

1. `noiser --schema` to see the node kinds and their settings; never guess a setting name.
2. Write a spec (`{"nodes": {...}, "edges": [[from, to, port]]}`, see `examples/Specs/`) and `noiser --build spec.json out.meshgraph`.
3. `noiser --validate out.meshgraph`, then `noiser --headless --mesh-graph out.meshgraph`.
4. `noiser --preview result.obj shot.png [--up y]` and read the PNG after every step that changes geometry.
5. Ask yourself, at each step, whether a human artist would do it this way; note the friction and fix the tool, not just the scene.

Do not kill `noiser.exe` by name (the user may have it open). Do not overwrite the user's hand-made library; use a scratch folder.
