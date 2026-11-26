"""Shared pieces of the terrain detail model: data sampling, the grid-free upsample and coarse-agreement projection, and the
input features.

The model adds 4x detail to a coarse heightmap: it sees the coarse height (smoothly upsampled), its slope and curvature, the
log flow accumulation of the coarse terrain, and a terrain type + scale, and generates the residual (fine - upsampled coarse).
The residual is made to average to zero over each coarse cell, so the output always agrees with the coarse input.
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

TYPES = ["alpine", "desert", "badlands", "hills", "volcanic", "karst", "coastal", "plains"]
FACTOR = 4  # coarse -> fine
PATCH = 256  # fine patch side in cells
CELL = float(os.environ.get("NOISER_ML_CELL", "1"))  # metres per cell of the stored tiles (1 for the lidar, 4 for its first stage)


def load_tiles(data_dir, holdout):
    """{type: [(dem, acc, valid_integral)]} for the training (holdout=False) or the held-out (True) tiles."""
    out = {t: [] for t in TYPES}
    for path in sorted(glob.glob(os.path.join(data_dir, "prep", "*.npz"))):
        name = os.path.basename(path)
        kind = name.split("__")[0]
        is_hold = "_HOLD" in name  # one whole site per type, never trained on
        if is_hold != holdout:
            continue
        z = np.load(path)
        valid = z["valid"].astype(np.float32)
        # Integral image of validity: the share of valid cells under any window in O(1).
        integral = np.pad(valid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        out[kind].append((z["dem"], z["acc"], integral, name))
    return {k: v for k, v in out.items() if v}


def valid_share(integral, y, x, size):
    return (integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]) / (size * size)


def sample_batch(tiles, n, rng, scale_choices=(1, 2), fixed_type=None):
    """n random fine patches: (fine [n,P,P] m, acc [n,P/F,P/F], type ids [n], cell metres [n])."""
    kinds = list(tiles)
    fines, accs, tids, cells = [], [], [], []
    while len(fines) < n:
        kind = fixed_type if fixed_type else kinds[rng.integers(len(kinds))]
        if kind not in tiles:
            continue
        dem, acc, integral, _ = tiles[kind][rng.integers(len(tiles[kind]))]
        s = int(rng.choice(scale_choices))  # 2: the same terrain read at twice the cell size
        size = PATCH * s
        if dem.shape[0] <= size or dem.shape[1] <= size:
            continue
        y, x = rng.integers(0, dem.shape[0] - size), rng.integers(0, dem.shape[1] - size)
        if valid_share(integral, y, x, size) < 0.97:
            continue
        fine = dem[y : y + size, x : x + size]
        a = acc[y // FACTOR : (y + size) // FACTOR, x // FACTOR : (x + size) // FACTOR]
        if s > 1:
            fine = fine.reshape(PATCH, s, PATCH, s).mean(axis=(1, 3))
            a = a.reshape(PATCH // FACTOR, s, PATCH // FACTOR, s).mean(axis=(1, 3))
        k = rng.integers(4)  # the eight symmetries of the square keep terrain statistics but multiply the data
        fine, a = np.rot90(fine, k), np.rot90(a, k)
        if rng.random() < 0.5:
            fine, a = fine[:, ::-1], a[:, ::-1]
        fines.append(np.ascontiguousarray(fine))
        accs.append(np.ascontiguousarray(a))
        tids.append(TYPES.index(kind))
        cells.append(CELL * s)
    return np.stack(fines), np.stack(accs), np.array(tids), np.array(cells, np.float32)


def block_mean(x, f=FACTOR):
    return F.avg_pool2d(x, f)


def _gauss(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    x = F.pad(x, (k // 2,) * 4, mode="replicate")
    return F.conv2d(F.conv2d(x, g.view(1, 1, 1, k)), g.view(1, 1, k, 1))


def _smooth_up(c):
    # Plain bicubic of block means leaves a 4-cell grid in slope (10x real terrain's); a light blur removes it.
    return _gauss(F.interpolate(c, scale_factor=FACTOR, mode="bicubic", align_corners=False), 1.5)


def zero_block_mean(r, iters=16):
    """Removes r's mean over every coarse cell smoothly (no 4-cell steps), by repeated smooth back-projection."""
    for _ in range(iters):
        r = r - _smooth_up(block_mean(r))
    # The smooth passes leave a little of the mean (8: 1.4%, 32: 0.05%, but 32 overflowed GPU memory in training); an exact
    # (stepped) removal of that rest keeps the hard guarantee, without which training diverged twice.
    return r - F.interpolate(block_mean(r), scale_factor=FACTOR, mode="nearest")


def upsample(c, iters=4):
    """Grid-free 4x upsample of coarse heights [n,1,h,w] whose block means match c (back-projected)."""
    u = _smooth_up(c)
    for _ in range(iters):
        u = u + _smooth_up(c - block_mean(u))
    return u


def degrade(coarse, rng_t):
    """Makes the coarse input less like clean block averages, since an artist's input is not terrain: random blur and smooth noise."""
    n = coarse.shape[0]
    out = coarse.clone()
    blur = torch.rand(n, device=coarse.device) < 0.7
    if blur.any():
        sigma = torch.rand(n, device=coarse.device) * 1.2 + 0.4
        heavy = torch.rand(n, device=coarse.device) < 0.3  # very smooth input, like low-frequency noise terrain
        sigma = torch.where(heavy, sigma * 4, sigma)
        for i in torch.nonzero(blur).flatten().tolist():
            k = int(2 * round(3 * float(sigma[i])) + 1)
            ax = torch.arange(k, device=coarse.device, dtype=torch.float32) - k // 2
            g = torch.exp(-(ax**2) / (2 * sigma[i] ** 2))
            g = (g / g.sum())
            x = F.pad(out[i : i + 1], (k // 2,) * 4, mode="replicate")
            x = F.conv2d(x, g.view(1, 1, 1, k))
            out[i : i + 1] = F.conv2d(x, g.view(1, 1, k, 1))
    noisy = torch.rand(n, 1, 1, 1, device=coarse.device) < 0.4
    if noisy.any():
        sm = F.interpolate(torch.randn(n, 1, 8, 8, device=coarse.device), size=coarse.shape[-2:], mode="bicubic", align_corners=False)
        amp = (coarse.flatten(1).std(dim=1).view(n, 1, 1, 1) * 0.1)
        out = out + noisy * sm * amp
    return out


def features(coarse_m, acc, cell_m):
    """The model's inputs from the coarse height in metres [n,1,c,c] and log flow accumulation [n,1,c,c].

    Returns (x [n,4,P,P] input channels, sigma [n] metres, mean [n] metres). cell_m [n] is the fine cell size in metres.
    """
    n = coarse_m.shape[0]
    up = upsample(coarse_m)
    mean = up.mean(dim=(1, 2, 3), keepdim=True)
    sigma = torch.maximum(up.flatten(1).std(dim=1), 0.1 * cell_m).view(n, 1, 1, 1)  # floor: 0.1 m at 1 m cells
    h = (up - mean) / sigma
    cm = cell_m.view(n, 1, 1, 1)
    gy, gx = torch.gradient(up, dim=(2, 3))
    slope = torch.sqrt(gx**2 + gy**2) / cm  # rise over run
    # Laplacian of a lightly smoothed copy: the plain 5-point Laplacian amplifies what little 4-cell ripple the base has.
    lap = F.conv2d(F.pad(_gauss(up, 1.0), (1, 1, 1, 1), mode="replicate"), torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=up.device)) / cm
    a = _smooth_up(acc)  # bilinear 4x left a 4-cell grid in the drainage input (30x real terrain's)
    x = torch.cat([h, slope.clamp(max=3.0), (lap * 10.0).clamp(-3, 3), (a - 4.0) / 3.0], dim=1)
    return x, sigma.flatten(), mean.flatten()


def hillshade(h, cell):
    """uint8 hillshade of h [n, m] metres, light from the north-west."""
    gy, gx = np.gradient(h)
    nx, ny = -gx / cell, -gy / cell
    lit = np.clip((nx * -0.5 + ny * 0.5 + 0.7) / np.sqrt(nx**2 + ny**2 + 1) / 1.1, 0, 1)
    return (lit * 255).astype(np.uint8)
