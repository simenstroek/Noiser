"""16x demo: 1 km of held-out 1 m lidar per terrain type, averaged to 16 m cells (64 x 64), enhanced by both flow stages
(16 m -> 4 m -> 1 m) with two seeds, next to a smooth 16x upscale and the real terrain. Optionally a Noiser heightmap too.

    python tools/ml/demo.py OVERVIEW.png ZOOM.png [--heightmap H.png --kind alpine]

OVERVIEW: 1 km panels (input | smooth 16x | seed 1 | seed 2 | real). ZOOM: a 256 m corner of each at full resolution.
A heightmap's world size and height range come from its .json sidecar (Noiser writes one).
"""
import argparse
import json
import os

os.environ.setdefault("NOISER_ML_CELL", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import flow  # noqa: E402
import terrain_ml as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("overview")
ap.add_argument("zoom")
ap.add_argument("--heightmap")
ap.add_argument("--kind", default="alpine")
ap.add_argument("--data", default="E:/_NoiserML/lidar")
ap.add_argument("--stage1", default="E:/_NoiserML/runs/flow_stage1/ckpt.pt")
ap.add_argument("--stage2", default="E:/_NoiserML/runs/flow1/ckpt.pt")
a = ap.parse_args()
n1, s1 = flow.load(a.stage1)
n2, s2 = flow.load(a.stage2)
S = 1024


def smooth16(c):
    return T.upsample(T.upsample(torch.from_numpy(c)[None, None].cuda()))[0, 0].cpu().numpy()


def cascade(c, kind, seed):
    T.CELL = 4.0  # each stage feeds log2(cell / its own training cell) to its scale input
    mid = flow.enhance(c, kind, 4.0, n1, s1, seed=seed)
    T.CELL = 1.0
    return flow.enhance(mid.astype(np.float32), kind, 1.0, n2, s2, seed=seed + 100)


rows, names = [], []
hold = T.load_tiles(a.data, True)
rng = np.random.default_rng(2)
for kind in ["alpine", "volcanic", "desert", "badlands", "hills"]:
    dem, _, integ, _ = hold[kind][0]
    best = None
    for _ in range(40):  # the roughest fully valid 1 km window of a few tries
        y, x = rng.integers(0, dem.shape[0] - S), rng.integers(0, dem.shape[1] - S)
        if T.valid_share(integ, y, x, S) >= 0.99:
            s = float(dem[y : y + S, x : x + S].std())
            if best is None or s > best[0]:
                best = (s, y, x)
    _, y, x = best
    real = dem[y : y + S, x : x + S]
    c = real.reshape(64, 16, 64, 16).mean(axis=(1, 3)).astype(np.float32)
    rows.append((c, smooth16(c), cascade(c, kind, 1), cascade(c, kind, 2), real))
    names.append(kind)
if a.heightmap:
    from skimage.transform import resize

    side = json.load(open(a.heightmap + ".json"))
    lo, hi = side["height_range_m"]
    img = np.array(Image.open(a.heightmap)).astype(np.float64) / 65535.0
    c = (lo + resize(img, (64, 64), anti_aliasing=True) * (hi - lo)).astype(np.float32)  # ~1 km at ~16 m cells
    rows.append((c, smooth16(c), cascade(c, a.kind, 1), cascade(c, a.kind, 2), None))
    names.append(f"heightmap ({a.kind})")


def labelled(img, cols, P):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, t in enumerate(cols):
        d.rectangle([i * P, 0, i * P + 7 * len(t) + 6, 14], fill=(0, 0, 0))
        d.text((i * P + 3, 1), t, fill=(255, 255, 0))
    for j, t in enumerate(names):
        d.rectangle([0, j * P + 16, 7 * len(t) + 6, j * P + 30], fill=(0, 0, 0))
        d.text((3, j * P + 17), t, fill=(0, 255, 255))
    return im


P, Z = 384, 256
over, zoom = [], []
for c, *full in rows:
    tiles = [np.array(Image.fromarray(T.hillshade(c, 16.0)).resize((P, P), Image.NEAREST))]
    tiles += [np.array(Image.fromarray(T.hillshade(h, 1.0)).resize((P, P), Image.LANCZOS)) if h is not None else np.zeros((P, P), np.uint8) for h in full]
    over.append(np.concatenate(tiles, axis=1))
    zoom.append(np.concatenate([T.hillshade(h[384 : 384 + Z, 384 : 384 + Z], 1.0) if h is not None else np.zeros((Z, Z), np.uint8) for h in full], axis=1))
labelled(np.concatenate(over), ["INPUT 16 m", "smooth 16x", "flow seed 1", "flow seed 2", "real 1 m"], P).save(a.overview)
labelled(np.concatenate(zoom), ["smooth 16x", "flow seed 1", "flow seed 2", "real 1 m"], Z).save(a.zoom)
print("wrote", a.overview, a.zoom)
