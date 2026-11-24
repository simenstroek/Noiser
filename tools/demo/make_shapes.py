"""Writes the demo SDF shapes (Y up, as the SDF Composer saves them) into the Noiser library's SDFs folder.

    python tools/demo/make_shapes.py "H:/My Drive/NoiserLibrary/SDFs"

Shapes: Boulder, Crag (a standing slab), Arch, Cave (the empty space of a cave: subtract it from terrain).
The Mushroom shape is the one already in the library.
"""
import json
import os
import sys


def shape(brush, mode="add", pos=(0, 0, 0), scale=(1, 1, 1), blend=0.0, round_=0.1, name=None, **extra):
    s = {
        "name": name or brush.title(), "hidden": False, "type": "shape",
        "position": list(pos), "rotation": [0.0, 0.0, 0.0], "scale": list(scale), "color": [0.8, 0.8, 0.8],
        "mode": mode, "chamfer": False, "blend": blend, "blend_strength": 0.0,
        "mirror_x": False, "mirror_y": False, "mirror_z": False, "mirror_blend": 0,
        "brush": brush, "bevel": 0.0, "round": round_, "thickness": 1, "cone": 0, "star": 0, "points": 6,
        "repeat_mode": "linear", "linear_repeats": [1, 1, 1], "steps": [1.5, 1.5, 1.5],
        "radial_repeats": 1, "radius": 1.5, "repeat_axis": "x",
    }
    s.update(extra)
    return s


def file(shapes):
    return {
        "version": [0, 6, 1],
        "environment": {"use_lightmap": False, "lightmap": "", "lightmap_rotation": 0, "luminance": 1, "color": [0.327, 0.333, 0.34]},
        "assemblies": [{"resolution": 6, "shapes": shapes}],
        "layers": [{"name": "Layer", "assembly": 0, "hidden": False, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0], "image": "", "roughness": 0.3, "metallic": 0, "emissive": 0, "ior": 1,
                    "transparency": 0, "image_opacity": 1}],
    }


SHAPES = {
    # Three overlapping lumps: a boulder that sits on the ground (its flat bottom is cut by the terrain).
    "Boulder": [
        shape("sphere", pos=(0, 0.55, 0), scale=(2.0, 1.4, 1.7), blend=0.3),
        shape("sphere", pos=(0.7, 0.35, 0.3), scale=(1.1, 0.9, 1.0), blend=0.35),
        shape("sphere", pos=(-0.6, 0.3, -0.4), scale=(0.9, 0.7, 0.9), blend=0.35),
    ],
    # A tall rock slab with a smaller one leaning against it.
    "Crag": [
        shape("cube", pos=(0, 1.3, 0), scale=(0.9, 2.6, 0.6), blend=0.2, round_=0.35),
        shape("cube", pos=(0.55, 0.6, 0.15), scale=(0.7, 1.2, 0.5), blend=0.25, round_=0.4),
        shape("cube", pos=(-0.5, 0.35, -0.1), scale=(0.6, 0.7, 0.6), blend=0.25, round_=0.4),
    ],
    # A block with a rounded opening through it and its base opened up: a natural arch.
    "Arch": [
        shape("cube", pos=(0, 0.95, 0), scale=(3.0, 1.9, 0.9), blend=0.1, round_=0.2),
        shape("sphere", mode="sub", pos=(0, 0.55, 0), scale=(1.5, 1.5, 2.5), blend=0.15),
        shape("cube", mode="sub", pos=(0, 0.1, 0), scale=(1.4, 1.2, 2.5), blend=0.1, round_=0.1),
    ],
    # The volume of a cave: a rounded tunnel with a flatter floor and a chamber behind. Subtract it from terrain.
    "Cave": [
        shape("sphere", pos=(0, 0.9, 0.0), scale=(2.2, 1.8, 3.0), blend=0.3),
        shape("sphere", pos=(0.2, 1.0, 2.6), scale=(3.2, 2.4, 3.2), blend=0.5),
    ],
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for name, shapes in SHAPES.items():
        path = os.path.join(out, name + ".sdf")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(file(shapes), f, indent="\t")
        print("wrote", path)
