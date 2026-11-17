"""Renders an OBJ with Blender's Workbench engine, to look at a mesh without opening Blender.

    blender -b --factory-startup --python tools/render_obj.py -- mesh.obj out.png [azimuth_degrees]

The mesh is framed automatically; the optional third argument turns the camera around it.
"""

import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
path, out = os.path.abspath(args[0]), os.path.abspath(args[1])
azimuth = math.radians(float(args[2])) if len(args) > 2 else math.radians(35)

# The factory startup scene has a default cube (and light and camera): start from an empty one.
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.obj_import(filepath=path)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
corners = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(c[i] for c in corners) for i in range(3)))
hi = Vector((max(c[i] for c in corners) for i in range(3)))
centre, size = (lo + hi) / 2, max(hi - lo)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.scene.collection.objects.link(cam)
cam.data.lens = 60
cam.data.clip_start = size * 0.01
cam.data.clip_end = size * 20
direction = Vector((math.sin(azimuth) * 0.9, -math.cos(azimuth) * 0.9, 0.55)).normalized()
cam.location = centre + direction * size * 2.6
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = 720
scene.render.filepath = out
scene.render.image_settings.file_format = "PNG"
shading = scene.display.shading
shading.light = "STUDIO"
shading.color_type = "SINGLE"
shading.single_color = (0.78, 0.74, 0.68)
shading.show_cavity = True
scene.world = bpy.data.worlds.new("w")
scene.world.color = (0.12, 0.12, 0.14)
bpy.ops.render.render(write_still=True)
tris = sum(len(o.data.polygons) for o in objs)
print(f"render_obj: {path}: {tris} faces -> {out}")
