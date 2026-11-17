"""Noiser sampler for Blender: File > Import > Noiser Function (.noiser).

Builds a Geometry Nodes group and a shader group, "Noiser <name>", each with a Position input and
a Density output, so the noise evaluates live at any resolution like a stock texture node.
"""

import json
import os

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from . import build


class NOISER_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.noiser"
    bl_label = "Import Noiser Function"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".noiser"
    filter_glob: StringProperty(default="*.noiser", options={"HIDDEN"})

    def execute(self, context):
        try:
            data = json.load(open(self.filepath, encoding="utf-8"))
            name = "Noiser " + os.path.splitext(os.path.basename(self.filepath))[0]
            build.build(data, name, "GeometryNodeTree")
            build.build(data, name, "ShaderNodeTree")
        except (build.Unsupported, ValueError, KeyError) as e:
            self.report({"ERROR"}, f"Can't build this function: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Added node groups \"{name}\"")
        return {"FINISHED"}


def menu(self, context):
    self.layout.operator(NOISER_OT_import.bl_idname, text="Noiser Function (.noiser)")


def register():
    bpy.utils.register_class(NOISER_OT_import)
    bpy.types.TOPBAR_MT_file_import.append(menu)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu)
    bpy.utils.unregister_class(NOISER_OT_import)
