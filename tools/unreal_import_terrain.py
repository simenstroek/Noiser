"""Brings the terrain chunks a Noiser Mesh graph run wrote (terrain_chunks.json) into an Unreal Engine 5 project and builds a
level from them.

Run it in the editor (needs the "Python Editor Script Plugin"): in the Output Log set the console to Python, or

    py "E:/path/to/noiser/tools/unreal_import_terrain.py" "E:/path/to/export/terrain_chunks.json" /Game/NoiserTerrain /Game/NoiserTerrain/PUT_TERRAIN_HERE

or from a command line (it quits the editor when done):

    UnrealEditor.exe Project.uproject -ExecutePythonScript="unreal_import_terrain.py <manifest> <destination> <level>" -unattended

Arguments: the manifest, the content folder (default /Game/NoiserTerrain) and the level to fill (default: the open level).

For every chunk in the manifest it
  * imports the mesh (OBJ / GLB / glTF / FBX) as a static mesh under <destination>/Meshes with Nanite on;
  * imports the PNG maps under <destination>/Textures (data maps are linear);
  * makes one terrain material, M_NoiserTerrain, and a material instance per chunk that carries that chunk's occlusion and
    curvature maps. The material colours the ground by slope and height (grass, sand, rock, snow) with no help from the
    meshes' UVs, so chunks match at their borders;
  * lays a water surface at the manifest's water level (the Landscape tab's "water at"), if it has one;
  * places the chunks in the level at the origin (the meshes already hold their world positions), with a sun, sky,
    fog and a post process volume so the level is lit.
Run it again to replace everything it made.
"""

import json
import os
import sys

import unreal

MESH_EXTENSIONS = (".obj", ".glb", ".gltf", ".fbx")
LINEAR_WORDS = ("normal", "occlusion", "curvature", "height", "depth", "position", "thickness")
FOLDER = "NoiserTerrain"


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


def make_material(destination, defaults):
    """The terrain material: colour from slope and height (VertexNormalWS, WorldPosition), times the AO texture parameter,
    lit up a little by the curvature texture parameter. `defaults` maps a parameter name to a linear texture to use as its
    default (a parameter's default must have the sampler's type, or the material does not compile)."""
    name, folder = "M_NoiserTerrain", destination + "/Materials"
    path = folder + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    lib = unreal.MaterialEditingLibrary

    def node(cls, x, y):
        return lib.create_material_expression(material, cls, x, y)

    def link(src, dst, dst_input, src_output=""):
        lib.connect_material_expressions(src, src_output, dst, dst_input)

    def colour(r, g, b, y):
        n = node(unreal.MaterialExpressionConstant3Vector, -1400, y)
        n.set_editor_property("constant", unreal.LinearColor(r, g, b, 1.0))
        return n

    def channel(src, r=False, g=False, b=False, x=-1200, y=0):
        n = node(unreal.MaterialExpressionComponentMask, x, y)
        n.set_editor_property("r", r)
        n.set_editor_property("g", g)
        n.set_editor_property("b", b)
        link(src, n, "")
        return n

    def smooth(value, lo, hi, x, y):
        n = node(unreal.MaterialExpressionSmoothStep, x, y)
        n.set_editor_property("const_min", lo)
        n.set_editor_property("const_max", hi)
        link(value, n, "Value")
        return n

    def lerp(a, b, alpha, x, y):
        n = node(unreal.MaterialExpressionLinearInterpolate, x, y)
        link(a, n, "A")
        link(b, n, "B")
        link(alpha, n, "Alpha")
        return n

    def multiply(a, b, x, y):
        n = node(unreal.MaterialExpressionMultiply, x, y)
        link(a, n, "A")
        link(b, n, "B")
        return n

    # How upward-facing the ground is, and how high it is (centimetres).
    up = channel(node(unreal.MaterialExpressionVertexNormalWS, -1500, -300), b=True, x=-1300, y=-300)
    height = channel(node(unreal.MaterialExpressionWorldPosition, -1500, 100), b=True, x=-1300, y=100)
    flat = smooth(up, 0.62, 0.86, -1100, -300)
    sandy = node(unreal.MaterialExpressionOneMinus, -900, 100)
    link(smooth(height, 800.0, 2500.0, -1100, 100), sandy, "")
    snowy = multiply(smooth(height, 11500.0, 13500.0, -1100, 250), flat, -900, 250)

    grass = lerp(colour(0.07, 0.13, 0.03, -50), colour(0.36, 0.28, 0.17, 0), sandy, -700, -50)
    ground = lerp(grass, colour(0.7, 0.72, 0.75, 300), snowy, -500, 0)
    rock = lerp(colour(0.10, 0.09, 0.08, -450), colour(0.26, 0.22, 0.18, -400), smooth(channel(node(unreal.MaterialExpressionWorldPosition, -1500, -450), b=True, x=-1300, y=-450), 0.0, 14000.0, -1100, -450), -700, -420)
    base = lerp(rock, ground, flat, -300, -200)

    def texture_parameter(parameter, y):
        n = node(unreal.MaterialExpressionTextureSampleParameter2D, -700, y)
        n.set_editor_property("parameter_name", parameter)
        if defaults.get(parameter):
            n.set_editor_property("texture", defaults[parameter])
        # The maps are data, not colours.
        n.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        return n

    # Everything that is not ground (baked from the field: 0 on the terrain, 1 on objects) is stone, whatever its height. Only when
    # the export has such a map (a Bake custom map named Objects).
    if defaults.get("Objects"):
        objects = texture_parameter("Objects", 1100)
        stone = lerp(colour(0.17, 0.14, 0.11, 1000), colour(0.33, 0.27, 0.20, 1050), smooth(channel(node(unreal.MaterialExpressionWorldPosition, -1500, 1100), b=True, x=-1300, y=1100), 0.0, 33000.0, -1100, 1100), -700, 1050)
        link_alpha = node(unreal.MaterialExpressionLinearInterpolate, -450, 1100)
        link(base, link_alpha, "A")
        link(stone, link_alpha, "B")
        link(objects, link_alpha, "Alpha", "R")
        base = link_alpha
    occlusion = texture_parameter("Occlusion", 500)
    curvature = texture_parameter("Curvature", 800)
    shade = node(unreal.MaterialExpressionLinearInterpolate, -300, 700)
    shade.set_editor_property("const_a", 0.8)
    shade.set_editor_property("const_b", 1.2)
    link(curvature, shade, "Alpha", "R")
    lit = multiply(multiply(base, shade, -100, -200), occlusion, 100, -200)
    lib.connect_material_property(lit, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = node(unreal.MaterialExpressionConstant, 100, 100)
    rough.set_editor_property("r", 0.88)
    lib.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def make_water_material(destination):
    """A translucent blue water material: dark where deep, glossy."""
    name, folder = "M_NoiserWater", destination + "/Materials"
    path = folder + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    lib = unreal.MaterialEditingLibrary
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    colour = lib.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -400, 0)
    colour.set_editor_property("constant", unreal.LinearColor(0.03, 0.14, 0.20, 1.0))
    lib.connect_material_property(colour, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = lib.create_material_expression(material, unreal.MaterialExpressionConstant, -400, 200)
    rough.set_editor_property("r", 0.06)
    lib.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    opacity = lib.create_material_expression(material, unreal.MaterialExpressionConstant, -400, 400)
    opacity.set_editor_property("r", 0.85)
    lib.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY)
    lib.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def place_water(manifest, destination, actors):
    """A flat water surface over the whole terrain at the manifest's water level (the ground below it is the river, lake or sea)."""
    level_cm = manifest.get("water_level_cm")
    if level_cm is None or not manifest["chunks"]:
        return
    unit_cm = manifest.get("unit_cm", 1.0)
    side_cm = manifest["chunks_per_side"] * manifest["chunks"][0]["size"] * unit_cm
    plane = unreal.load_asset("/Engine/BasicShapes/Plane.Plane")
    actor = actors.spawn_actor_from_object(plane, unreal.Vector(side_cm / 2, -side_cm / 2, level_cm))
    actor.set_actor_label("Water")
    actor.set_folder_path(FOLDER)
    actor.set_actor_scale3d(unreal.Vector(side_cm / 100.0, side_cm / 100.0, 1.0))
    actor.static_mesh_component.set_material(0, make_water_material(destination))
    actor.static_mesh_component.set_editor_property("cast_shadow", False)


def import_terrain(manifest_path, destination="/Game/NoiserTerrain", level=None, place=True):
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
    tools.import_asset_tasks(mesh_tasks + texture_tasks)

    textures = {}
    for task in texture_tasks:
        for asset in _imported(task):
            if not isinstance(asset, unreal.Texture2D):
                continue
            name = asset.get_name().lower()
            if "normal" in name:
                asset.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
            if any(word in name for word in LINEAR_WORDS):
                asset.set_editor_property("srgb", False)
            # Maps are made for tiles of the terrain: do not wrap at the edges.
            asset.set_editor_property("address_x", unreal.TextureAddress.TA_CLAMP)
            asset.set_editor_property("address_y", unreal.TextureAddress.TA_CLAMP)
            textures[asset.get_name().lower()] = asset
            unreal.EditorAssetLibrary.save_loaded_asset(asset)

    material = make_material(destination, {
        "Occlusion": next((t for n, t in textures.items() if n.endswith("_ambient_occlusion")), None),
        "Curvature": next((t for n, t in textures.items() if n.endswith("_curvature")), None),
        "Objects": next((t for n, t in textures.items() if n.endswith("_objects")), None),
    })

    meshes = []
    for task in mesh_tasks:
        for asset in _imported(task):
            if not isinstance(asset, unreal.StaticMesh):
                continue
            settings = asset.get_editor_property("nanite_settings")
            settings.set_editor_property("enabled", True)
            asset.set_editor_property("nanite_settings", settings)
            stem = asset.get_name()
            instance_name = "MI_" + stem
            instance_path = destination + "/Materials/" + instance_name
            if unreal.EditorAssetLibrary.does_asset_exist(instance_path):
                unreal.EditorAssetLibrary.delete_asset(instance_path)
            instance = tools.create_asset(instance_name, destination + "/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
            instance.set_editor_property("parent", material)
            for parameter, suffix in (("Occlusion", "_ambient_occlusion"), ("Curvature", "_curvature"), ("Objects", "_objects")):
                texture = textures.get((stem + suffix).lower())
                if texture:
                    unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(instance, parameter, texture)
            unreal.MaterialEditingLibrary.update_material_instance(instance)
            asset.set_material(0, instance)
            unreal.EditorAssetLibrary.save_loaded_asset(instance)
            unreal.EditorAssetLibrary.save_loaded_asset(asset)
            meshes.append(asset)

    placed = 0
    if place:
        if level:
            editor_levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            if not unreal.EditorAssetLibrary.does_asset_exist(level):
                # A plain level, not a World Partition one (which Unreal makes by default in 5.8).
                editor_levels.new_level(level, False)
            else:
                editor_levels.load_level(level)
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor in actors.get_all_level_actors():
            if str(actor.get_folder_path()) == FOLDER:
                actors.destroy_actor(actor)
        for asset in meshes:
            actor = actors.spawn_actor_from_object(asset, unreal.Vector(0, 0, 0))
            actor.set_actor_label(asset.get_name())
            actor.set_folder_path(FOLDER)
            placed += 1
        place_water(manifest, destination, actors)
        light_the_level(actors)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    unreal.log("Noiser terrain: {} chunk mesh(es) imported, {} placed, {} texture(s) under {}".format(len(mesh_tasks), placed, len(texture_tasks), destination))


def light_the_level(actors):
    """A sun, sky, sky light, fog and an exposure volume: the ones the level already has are set up, the others made. The
    sky sphere of Unreal's default level is removed (it is a few hundred metres across and hides a bigger terrain)."""
    for actor in list(actors.get_all_level_actors()):
        if actor.get_actor_label() == "SM_SkySphere":
            actors.destroy_actor(actor)
    existing = {}
    for actor in actors.get_all_level_actors():
        existing.setdefault(actor.get_class().get_name(), actor)

    def get(cls, location=(0, 0, 0), rotation=(0, 0, 0)):
        actor = existing.get(cls.static_class().get_name())
        if actor is None:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(cls, unreal.Vector(*location), unreal.Rotator(*rotation))
            actor.set_folder_path(FOLDER)
        return actor

    sun = get(unreal.DirectionalLight, (0, 0, 30000), (0, -38, -35))
    sun.set_actor_rotation(unreal.Rotator(0, -38, -35), False)
    sun.get_component_by_class(unreal.DirectionalLightComponent).set_editor_property("intensity", 6.0)
    # The planet's ground at the terrain's lowest level (the terrain's edges are open), in earth colours.
    atmosphere = get(unreal.SkyAtmosphere, (0, 0, -50))
    atmosphere.set_actor_location(unreal.Vector(0, 0, -50), False, False)
    atmosphere.get_component_by_class(unreal.SkyAtmosphereComponent).set_editor_property("ground_albedo", unreal.Color(110, 96, 70, 255))
    sky = get(unreal.SkyLight, (0, 0, 30000))
    sky.get_component_by_class(unreal.SkyLightComponent).set_editor_property("real_time_capture", True)
    fog = get(unreal.ExponentialHeightFog, (50000, 50000, 5000))
    fog.get_component_by_class(unreal.ExponentialHeightFogComponent).set_editor_property("fog_density", 0.004)
    volume = get(unreal.PostProcessVolume)
    volume.set_editor_property("unbound", True)
    settings = volume.get_editor_property("settings")
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", -0.5)
    volume.set_editor_property("settings", settings)


if __name__ == "__main__":
    # Arguments, or the environment (a path with a space in it is easier to pass that way): NOISER_MANIFEST, NOISER_DESTINATION, NOISER_LEVEL.
    args = sys.argv[1:]
    manifest = args[0] if args else os.environ.get("NOISER_MANIFEST")
    if not manifest:
        unreal.log_error("usage: py unreal_import_terrain.py <terrain_chunks.json> [/Game/Destination] [/Game/Level]  (or set NOISER_MANIFEST, NOISER_DESTINATION, NOISER_LEVEL)")
    else:
        import_terrain(manifest, args[1] if len(args) > 1 else os.environ.get("NOISER_DESTINATION", "/Game/NoiserTerrain"), args[2] if len(args) > 2 else os.environ.get("NOISER_LEVEL"))
        if os.environ.get("NOISER_UE_QUIT"):
            unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
