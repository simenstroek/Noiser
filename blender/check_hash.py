"""Checks that the Blender sampler's pcg3d (Integer Math and Bit Math nodes) reproduces Noiser's hash01.

Run headless:
    blender --background --factory-startup --python-exit-code 1 --python blender/check_hash.py -- target/blender/hash.json

The JSON comes from `cargo test hash_vectors_for_blender`: rows of {cell: [x, y, z], seed, h: [3 floats]}.
"""

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noiser import build  # noqa: E402

rows = json.load(open(sys.argv[sys.argv.index("--") + 1]))

mesh = bpy.data.meshes.new("cells")
mesh.from_pydata([r["cell"] for r in rows], [], [])
w = mesh.attributes.new("w", "INT", "POINT")
w.data.foreach_set("value", [int(r["seed"]) for r in rows])
obj = bpy.data.objects.new("cells", mesh)
bpy.context.scene.collection.objects.link(obj)

ng, gin, gout = build.new_group("hash", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")], [("Geometry", "NodeSocketGeometry")])
b = build.Builder(ng)
pos = ng.nodes.new("GeometryNodeInputPosition")
seed = ng.nodes.new("GeometryNodeInputNamedAttribute")
seed.data_type = "INT"
seed.inputs["Name"].default_value = "w"
h = build.pcg_hash3(b, b.separate(pos.outputs["Position"]), b.imath("MULTIPLY", seed.outputs["Attribute"], 0x9E3779B9))
store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
store.data_type = "FLOAT_VECTOR"
store.inputs["Name"].default_value = "h"
ng.links.new(b.combine(*h), store.inputs["Value"])
ng.links.new(gin.outputs[0], store.inputs["Geometry"])
ng.links.new(store.outputs["Geometry"], gout.inputs[0])
obj.modifiers.new("hash", "NODES").node_group = ng

ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
got = [0.0] * (3 * len(rows))
ev.attributes["h"].data.foreach_get("vector", got)
worst = max(abs(got[3 * i + k] - r["h"][k]) for i, r in enumerate(rows) for k in range(3))
print(f"check_hash: {len(rows)} cells, Blender {bpy.app.version_string}, max |diff| = {worst:.3g}")
if worst > 0:
    for i, r in enumerate(rows[:5]):
        print("  cell", r["cell"], "seed", r["seed"], "noiser", r["h"], "blender", got[3 * i : 3 * i + 3])
    sys.exit(1)
