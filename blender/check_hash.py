"""Checks that Blender's White Noise Texture (4D, Color) reproduces Noiser's lattice hash.

Run headless:
    blender --background --factory-startup --python-exit-code 1 --python blender/check_hash.py -- target/blender/hash.json

The JSON comes from `cargo test hash_vectors_for_blender`: rows of {cell: [x, y, z], seed, h: [3 floats]}.
"""

import json
import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
rows = json.load(open(path))

mesh = bpy.data.meshes.new("cells")
mesh.from_pydata([r["cell"] for r in rows], [], [])
w = mesh.attributes.new("w", "FLOAT", "POINT")
w.data.foreach_set("value", [r["seed"] for r in rows])
obj = bpy.data.objects.new("cells", mesh)
bpy.context.scene.collection.objects.link(obj)

ng = bpy.data.node_groups.new("hash", "GeometryNodeTree")
ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
n = ng.nodes
gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
pos = n.new("GeometryNodeInputPosition")
seed = n.new("GeometryNodeInputNamedAttribute")
seed.data_type = "FLOAT"
seed.inputs["Name"].default_value = "w"
white = n.new("ShaderNodeTexWhiteNoise")
white.noise_dimensions = "4D"
store = n.new("GeometryNodeStoreNamedAttribute")
store.data_type = "FLOAT_VECTOR"
store.inputs["Name"].default_value = "h"
link = ng.links.new
link(pos.outputs["Position"], white.inputs["Vector"])
link(seed.outputs["Attribute"], white.inputs["W"])
link(white.outputs["Color"], store.inputs["Value"])
link(gin.outputs[0], store.inputs["Geometry"])
link(store.outputs["Geometry"], gout.inputs[0])
obj.modifiers.new("hash", "NODES").node_group = ng

ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
got = [0.0] * (3 * len(rows))
ev.attributes["h"].data.foreach_get("vector", got)

worst = max(abs(got[3 * i + k] - r["h"][k]) for i, r in enumerate(rows) for k in range(3))
print(f"check_hash: {len(rows)} cells, Blender {bpy.app.version_string}, max |diff| = {worst:.3g}")
if worst > 1e-6:
    for i, r in enumerate(rows[:5]):
        print("  cell", r["cell"], "seed", r["seed"], "noiser", r["h"], "blender", got[3 * i : 3 * i + 3])
    sys.exit(1)
