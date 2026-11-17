"""Checks the Noiser add-on against a .noiser file's own test vectors (Geometry Nodes).

Run headless (Blender 5.2+):
    blender --background --factory-startup --python-exit-code 1 --python blender/check_file.py -- a.noiser [b.noiser ...]

The group is evaluated at a vertex per test point.
"""

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noiser import build  # noqa: E402

TOLERANCE = 1e-4
failed = False
for path in sys.argv[sys.argv.index("--") + 1 :]:
    data = json.load(open(path, encoding="utf-8"))
    name = os.path.basename(path)
    group = build.build(data, name)

    pts, want = data["tests"]["points"], data["tests"]["density"]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(pts, [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    ng, gin, gout = build.new_group("check " + name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")], [("Geometry", "NodeSocketGeometry")])
    call = ng.nodes.new("GeometryNodeGroup")
    call.node_tree = group
    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.inputs["Name"].default_value = "d"
    ng.links.new(call.outputs["Density"], store.inputs["Value"])
    ng.links.new(gin.outputs[0], store.inputs["Geometry"])
    ng.links.new(store.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("check", "NODES").node_group = ng

    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    got = [0.0] * len(pts)
    ev.attributes["d"].data.foreach_get("value", got)
    errs = [abs(g - w) for g, w in zip(got, want)]
    worst = max(errs)
    nodes = sum(len(g.nodes) for g in bpy.data.node_groups if g.name in (name, build.PERLIN_GROUP))
    print(f"check_file: {name}: {len(pts)} points, max |diff| = {worst:.3g}, {nodes} nodes")
    if worst > TOLERANCE:
        failed = True
        i = errs.index(worst)
        print(f"  worst at {pts[i]}: noiser {want[i]:.6f}, blender {got[i]:.6f}")
sys.exit(1 if failed else 0)
