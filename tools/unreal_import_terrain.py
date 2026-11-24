"""Imports the terrain chunks a Noiser Mesh graph run wrote (terrain_chunks.json) into an Unreal Engine 5 level.

In the editor (Edit > Plugins: "Python Editor Script Plugin" on), open Window > Output Log, switch the console to Python,
or use File > Execute Python Script. From the Output Log command line:

    py "E:/path/to/noiser/tools/unreal_import_terrain.py" "E:/path/to/out/terrain_chunks.json" /Game/Terrain

What it does, for every chunk in the manifest:
  * imports the mesh file (OBJ / GLB / glTF / FBX) as a static mesh under <destination>/Meshes and turns Nanite on;
  * places it in the open level at the origin: the meshes already hold their world positions in centimetres (the
    Mesh graph's Load Mesh node for the terrain must be set to 1 unit = 1 cm, which is its default for dropped PNGs);
  * imports the chunk's PNG maps under <destination>/Textures (normal maps get the normal map compression, the
    other maps are linear), ready to plug into one material (chunk i of a UDIM set is tile 1001 + i).
It does not make materials: put the maps into a material of your own.
"""

import json
import os
import sys

import unreal

MESH_EXTENSIONS = (".obj", ".glb", ".gltf", ".fbx")
LINEAR_WORDS = ("normal", "occlusion", "curvature", "height", "depth", "position", "thickness")


def _task(path, destination):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("replace_existing_settings", True)
    task.set_editor_property("save", False)
    return task


def _imported(task):
    return [unreal.load_asset(p) for p in task.get_editor_property("imported_object_paths")]


def import_terrain(manifest_path, destination="/Game/Terrain", place=True):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    folder = os.path.dirname(os.path.abspath(manifest_path))
    mesh_tasks, texture_tasks = [], []
    for chunk in manifest["chunks"]:
        for name in chunk["files"]:
            path = os.path.join(folder, name)
            if name.lower().endswith(MESH_EXTENSIONS):
                mesh_tasks.append(_task(path, destination + "/Meshes"))
            elif name.lower().endswith(".png"):
                texture_tasks.append(_task(path, destination + "/Textures"))
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    tools.import_assets_automated(mesh_tasks + texture_tasks)

    placed = 0
    for task in mesh_tasks:
        for asset in _imported(task):
            if not isinstance(asset, unreal.StaticMesh):
                continue
            settings = asset.get_editor_property("nanite_settings")
            settings.set_editor_property("enabled", True)
            asset.set_editor_property("nanite_settings", settings)
            unreal.EditorAssetLibrary.save_loaded_asset(asset)
            if place:
                actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_object(asset, unreal.Vector(0, 0, 0))
                actor.set_actor_label(asset.get_name())
                placed += 1
    for task in texture_tasks:
        for asset in _imported(task):
            if not isinstance(asset, unreal.Texture2D):
                continue
            name = asset.get_name().lower()
            if "normal" in name:
                asset.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
            if any(word in name for word in LINEAR_WORDS):
                asset.set_editor_property("srgb", False)
            unreal.EditorAssetLibrary.save_loaded_asset(asset)
    unreal.log("Noiser terrain: {} chunk mesh(es) imported, {} placed, {} texture(s) under {}".format(len(mesh_tasks), placed, len(texture_tasks), destination))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        unreal.log_error("usage: py unreal_import_terrain.py <terrain_chunks.json> [/Game/Destination]")
    else:
        import_terrain(args[0], args[1] if len(args) > 1 else "/Game/Terrain")
