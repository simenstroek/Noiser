"""Turns Gaea heightmap exports into training arrays next to the lidar ones (same npz format, plus the cell size).

    python tools/ml/prep_gaea.py [GAEA_DIR]   default E:/_NoiserML/gaea; writes E:/_NoiserML/lidar/prep/<type>__gaea_<name>.npz
                                             and the 4x-averaged copy into E:/_NoiserML/lidar_4m/prep

Each export is <folder>/*.exr, one float channel of normalised height. Every project here is 5000 m wide with a 2500 m height
range at 4096 px (read from its .terrain file), so cells are 1.22 m (4.88 m for the first stage). Pixels at the exact
minimum or maximum are marked invalid: Gaea clamps there, and clamped flats are not terrain. The type comes from the folder
name (GROUPS); folders not in GROUPS are skipped (canyonWithRiverBed's export is a colour image, not heights).
More variations of a project: tools/ml/gaea_seeds.ps1 builds it with other mutation seeds through Gaea's Build Swarm.
"""

import glob
import os
import re
import sys

import numpy as np
import OpenEXR

from prep_lidar import COARSE, flow_accumulation

GROUPS = {"Craters": "craters", "Fantasy01": "fantasy", "Fantasy02": "fantasy", "icelandic": "icelandic",
          "rockyOutcrops": "rocky", "sharpRocky": "rocky", "snowCoveredCliffs": "alpine", "Alpine": "alpine"}
HOLD = {"Craters004", "Fantasy01016", "Fantasy02020", "icelandic004", "rockyOutcrops004", "sharpRocky004"}  # one per project


def world(folder, group):
    """(width m, height range m) of a Gaea project: the last Width/Height in the .terrain file are the terrain's own. Some
    build folders have no copy of the project; then the one beside them is used."""
    found = glob.glob(os.path.join(folder, "*.terrain")) or [os.path.join(os.path.dirname(folder), group + ".terrain")]
    t = open(found[0], encoding="utf-8").read()
    return float(re.findall(r'"Width":\s*([\d.]+)', t)[-1]), float(re.findall(r'"Height":\s*([\d.]+)', t)[-1])


def save(out, dem, valid, cell):
    n, m = dem.shape
    coarse = dem.reshape(n // COARSE, COARSE, m // COARSE, COARSE).mean(axis=(1, 3))
    acc = np.log1p(flow_accumulation(coarse.astype(np.float64))).astype(np.float32)
    np.savez_compressed(out, dem=dem.astype(np.float32), acc=acc, valid=valid, cell=np.float32(cell))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "E:/_NoiserML/gaea"
    for folder in sorted(glob.glob(os.path.join(src, "*"))):
        name = os.path.basename(folder)
        group = re.sub(r"\d+$", "", name) if not name.startswith("Fantasy") else name[:9]
        if group not in GROUPS or not glob.glob(os.path.join(folder, "*.exr")):
            continue
        width, height = world(folder, group)
        with OpenEXR.File(glob.glob(os.path.join(folder, "*.exr"))[0]) as f:
            ch = f.channels()
            assert list(ch) == ["Y"], f"{name}: channels {list(ch)}, expected one height channel"
            h = ch["Y"].pixels.astype(np.float64)
        dem = (h * height).astype(np.float32)
        valid = (h > h.min()) & (h < h.max())
        cell = width / h.shape[1]
        base = f"{GROUPS[group]}__gaea_{name}" + ("_HOLD" if name in HOLD else "")
        for down, dest in ((1, "E:/_NoiserML/lidar/prep"), (4, "E:/_NoiserML/lidar_4m/prep")):
            out = os.path.join(dest, base + ".npz")
            if os.path.exists(out):
                continue
            n = dem.shape[0] // (COARSE * down) * COARSE
            d = dem[: n * down, : n * down].reshape(n, down, n, down).mean(axis=(1, 3))
            v = valid[: n * down, : n * down].reshape(n, down, n, down).all(axis=(1, 3))
            save(out, d, v, cell * down)
            print(f"{base} @ {cell * down:.3f} m: {d.shape}, {v.mean() * 100:.1f}% valid, relief {np.ptp(d):.0f} m", flush=True)


if __name__ == "__main__":
    main()
