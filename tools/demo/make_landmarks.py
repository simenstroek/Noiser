"""Writes the landmark shapes (Y up, +Z is the front, as the SDF Composer saves them): an Eiffel tower and a Sphinx.

    python tools/demo/make_landmarks.py "H:/My Drive/NoiserLibrary/SDFs"

The tower is 330 m tall in metres (any scale works: the Mesh graph fits a shape to the size you give it). The sphinx is 73 m
long, its head at +Z.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_shapes import file, shape  # noqa: E402

PI = math.pi


def tower():
    """Four pylons (one, mirrored in x and z), the arches between them, two platforms, the shaft, the top and the spire."""
    s = []
    # The pylons: three hollow, slightly tapering sections stepping inwards (a rotated, mirrored leg is hard to place by hand and
    # to judge from a perspective view). Mirrored in both axes: four legs from one.
    for centre, y, height, size in [(47.0, 11.0, 24.0, 26.0), (41.0, 33.0, 22.0, 22.0), (35.0, 52.0, 18.0, 18.0)]:
        s.append(shape("cube", name="Pylon", pos=(centre, y, centre), scale=(size, height, size), thickness=0.25, cone=0.12, round_=0.02,
                       mirror_x=True, mirror_z=True, blend=3.0))
    # The arches between the pylons: a round arch on each face, and the floor under them stays open.
    s.append(shape("cylinder", name="Arch z", mode="sub", pos=(0, 23.0, 0), scale=(50.0, 140.0, 50.0), rotation=(PI / 2, 0, 0), blend=0.0))
    s.append(shape("cylinder", name="Arch x", mode="sub", pos=(0, 23.0, 0), scale=(50.0, 140.0, 50.0), rotation=(0, 0, PI / 2), blend=0.0))
    s.append(shape("cube", name="Arch floor z", mode="sub", pos=(0, 6.0, 0), scale=(50.0, 12.0, 140.0), blend=0.0))
    s.append(shape("cube", name="Arch floor x", mode="sub", pos=(0, 6.0, 0), scale=(140.0, 12.0, 50.0), blend=0.0))
    # First platform, with a thinner second storey on it.
    s.append(shape("cube", name="First platform", pos=(0, 57.0, 0), scale=(78.0, 5.0, 78.0), round_=0.05, blend=0.0))
    s.append(shape("cube", name="Rail", pos=(0, 61.0, 0), scale=(74.0, 3.0, 74.0), thickness=0.08, round_=0.0, blend=0.0))
    # The middle: a hollow tapering block with windows cut through its sides.
    s.append(shape("cube", name="Middle", pos=(0, 87.0, 0), scale=(46.0, 62.0, 46.0), thickness=0.2, cone=0.45, round_=0.02, blend=0.0))
    s.append(shape("cube", name="Window z", mode="sub", pos=(0, 84.0, 0), scale=(14.0, 38.0, 80.0), blend=0.0))
    s.append(shape("cube", name="Window x", mode="sub", pos=(0, 84.0, 0), scale=(80.0, 38.0, 14.0), blend=0.0))
    s.append(shape("cube", name="Second platform", pos=(0, 115.0, 0), scale=(38.0, 4.0, 38.0), round_=0.05, blend=0.0))
    # The shaft up to the third platform, hollow, with long windows.
    s.append(shape("cube", name="Shaft", pos=(0, 196.0, 0), scale=(24.0, 162.0, 24.0), thickness=0.3, cone=0.6, round_=0.02, blend=0.0))
    for k in range(4):
        y = 140.0 + 32.0 * k
        s.append(shape("cube", name="Shaft window z", mode="sub", pos=(0, y, 0), scale=(6.0, 18.0, 60.0), blend=0.0))
        s.append(shape("cube", name="Shaft window x", mode="sub", pos=(0, y, 0), scale=(60.0, 18.0, 6.0), blend=0.0))
    s.append(shape("cube", name="Third platform", pos=(0, 276.0, 0), scale=(20.0, 4.0, 20.0), round_=0.05, blend=0.0))
    s.append(shape("cube", name="Cabin", pos=(0, 283.0, 0), scale=(14.0, 10.0, 14.0), round_=0.15, blend=0.0))
    s.append(shape("cylinder", name="Spire", pos=(0, 305.0, 0), scale=(4.0, 50.0, 4.0), blend=0.0))
    s.append(shape("cylinder", name="Mast", pos=(0, 325.0, 0), scale=(1.4, 10.0, 1.4), blend=0.0))
    return file(s)


def sphinx():
    """A lion lying along Z with a human head at +Z, front paws stretched out."""
    s = []
    s.append(shape("cube", name="Body", pos=(0, 8.0, -4.0), scale=(18.0, 14.0, 42.0), round_=0.45, blend=0.0))
    s.append(shape("sphere", name="Haunch", pos=(0, 8.0, -20.0), scale=(20.0, 15.0, 22.0), blend=4.0))
    s.append(shape("cube", name="Chest", pos=(0, 10.0, 14.0), scale=(15.0, 17.0, 14.0), round_=0.4, blend=3.0))
    s.append(shape("cube", name="Paw", pos=(5.5, 2.4, 28.0), scale=(5.0, 4.8, 24.0), round_=0.5, mirror_x=True, blend=1.5))
    s.append(shape("cylinder", name="Neck", pos=(0, 20.0, 20.0), scale=(9.0, 14.0, 9.0), blend=3.0))
    s.append(shape("sphere", name="Head", pos=(0, 29.0, 22.0), scale=(10.0, 13.0, 11.0), blend=3.0))
    s.append(shape("cube", name="Headdress", pos=(0, 29.5, 19.0), scale=(16.0, 14.0, 9.0), round_=0.3, blend=2.0))
    s.append(shape("cube", name="Headdress flap", pos=(7.5, 21.0, 20.0), scale=(3.0, 16.0, 6.0), round_=0.3, mirror_x=True, blend=1.5))
    s.append(shape("sphere", name="Nose", pos=(0, 28.0, 27.5), scale=(2.4, 4.0, 3.0), blend=1.0))
    s.append(shape("sphere", name="Eye", mode="sub", pos=(2.6, 31.0, 26.5), scale=(2.0, 1.6, 2.0), mirror_x=True, blend=0.4))
    s.append(shape("sphere", name="Mouth", mode="sub", pos=(0, 24.8, 27.5), scale=(4.4, 1.2, 3.0), blend=0.4))
    s.append(shape("cylinder", name="Tail", pos=(7.0, 3.0, -29.0), scale=(2.4, 16.0, 2.4), rotation=(PI / 2, 0, 0.0), blend=1.0))
    return file(s)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for name, data in {"EiffelTower": tower(), "Sphinx": sphinx()}.items():
        path = os.path.join(out, name + ".sdf")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent="\t")
        print("wrote", path)
