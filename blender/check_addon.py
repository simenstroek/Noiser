"""Runs the add-on's real entry point: register, then File > Import via bpy.ops, in a fresh Blender.

    blender --background --factory-startup --python-exit-code 1 --python blender/check_addon.py -- a.noiser
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noiser  # noqa: E402

noiser.register()
path = sys.argv[sys.argv.index("--") + 1]
result = bpy.ops.import_scene.noiser(filepath=path)
names = sorted(g.name for g in bpy.data.node_groups)
print("check_addon:", result, names)
stem = "Noiser " + os.path.splitext(os.path.basename(path))[0]
assert result == {"FINISHED"}, result
assert stem in names, f"expected a group named from the file, got {names}"
assert not any(n.rsplit(".", 1)[-1].isdigit() for n in names), f"Blender renamed a group on a name clash: {names}"
assert {g.name: g.bl_idname for g in bpy.data.node_groups}[stem] == "GeometryNodeTree"
# A file the add-on can't build is reported to the user as an error naming the noise kind.
import json, tempfile  # noqa: E402
bad = json.load(open(path))
bad["ops"][1]["kind"] = "worley"
tmp = os.path.join(tempfile.mkdtemp(), "bad.noiser")
json.dump(bad, open(tmp, "w"))
try:
    bpy.ops.import_scene.noiser(filepath=tmp)
    raise AssertionError("an unsupported noise kind should fail the import")
except RuntimeError as e:  # bpy.ops raises when an operator reports an ERROR
    assert "worley" in str(e), e
print("check_addon: ok")
