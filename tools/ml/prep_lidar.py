"""Turns the downloaded 1 m lidar tiles into training arrays (same format as prep_tiles.py, 1 m cells).

    python tools/ml/prep_lidar.py [DATA_DIR] [DOWN OUT_DIR]   default E:/_NoiserML/lidar; writes DATA_DIR/prep/<name>.npz
                                                             (or OUT_DIR/prep, at DOWN-metre cells)

Each tile is cropped to 5000 x 5000 cells (the window with the most valid data), nodata is marked invalid, and the log flow
accumulation is computed on the 4x-coarser (4 m) terrain. Water is not masked: lidar bare earth has voids there.
"""

import heapq
import os
import sys

import numba
import numpy as np

COARSE = 4
CROP = 5000


@numba.njit(cache=True)
def flow_accumulation(h):
    """Priority-flood depression filling, then multiple-flow-direction accumulation (cells draining through each cell)."""
    n, m = h.shape
    filled = h.copy()
    done = np.zeros((n, m), np.bool_)
    heap = [(np.float64(0.0), 0, 0)]
    heap.pop()
    for i in range(n):
        for j in range(m):
            if i == 0 or j == 0 or i == n - 1 or j == m - 1:
                heapq.heappush(heap, (np.float64(h[i, j]), i, j))
                done[i, j] = True
    order = np.empty(n * m, np.int64)
    k = 0
    while len(heap) > 0:
        z, i, j = heapq.heappop(heap)
        order[k] = i * m + j
        k += 1
        for di in range(-1, 2):
            for dj in range(-1, 2):
                a, b = i + di, j + dj
                if (di == 0 and dj == 0) or a < 0 or b < 0 or a >= n or b >= m or done[a, b]:
                    continue
                done[a, b] = True
                # Nudge so flats drain in the order they were reached.
                filled[a, b] = max(h[a, b], z + 1e-4)
                heapq.heappush(heap, (np.float64(filled[a, b]), a, b))
    acc = np.ones((n, m), np.float32)
    # Cells popped later are higher or equal along each flow path: walk from the last popped back, sending flow downhill.
    # Multiple flow directions (Freeman 1991, exponent 1.1): flow splits over all lower neighbours by slope, so channels
    # are not forced onto the 8 compass lines that single-direction D8 draws.
    for idx in range(n * m - 1, -1, -1):
        c = order[idx]
        i, j = c // m, c % m
        w = np.zeros(8)
        total = 0.0
        k = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if a >= 0 and b >= 0 and a < n and b < m:
                    drop = (filled[i, j] - filled[a, b]) / (1.4142 if di != 0 and dj != 0 else 1.0)
                    if drop > 0:
                        w[k] = drop**1.1
                        total += w[k]
                k += 1
        if total > 0:
            k = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    if w[k] > 0:
                        acc[i + di, j + dj] += acc[i, j] * w[k] / total
                    k += 1
    return acc


def prep(path, out, down=1):
    import rasterio  # only for data prep; inference imports flow_accumulation from here

    with rasterio.open(path) as src:
        res = abs(src.transform.a)
        if abs(res - 1.0) > 0.01:
            raise ValueError(f"cell size {res} m, expected 1 m")
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    valid = np.isfinite(dem) & (dem > -1000)
    if nodata is not None:
        valid &= dem != nodata
    n, m = dem.shape
    best = None
    for y in sorted({0, max(n - CROP, 0) // 2, max(n - CROP, 0)}):
        for x in sorted({0, max(m - CROP, 0) // 2, max(m - CROP, 0)}):
            share = valid[y : y + CROP, x : x + CROP].mean()
            if best is None or share > best[0]:
                best = (share, y, x)
    _, y, x = best
    dem, valid = dem[y : y + CROP, x : x + CROP], valid[y : y + CROP, x : x + CROP]
    n, m = (dem.shape[0] // COARSE) * COARSE, (dem.shape[1] // COARSE) * COARSE
    dem, valid = dem[:n, :m], valid[:n, :m]
    fill = float(dem[valid].mean()) if valid.any() else 0.0
    dem = np.where(valid, dem, fill).astype(np.float32)
    if down > 1:  # the same terrain at down-metre cells, for the first stage of a two-stage (16x) upscale
        n, m = n // (COARSE * down) * COARSE, m // (COARSE * down) * COARSE
        dem = dem[: n * down, : m * down].reshape(n, down, m, down).mean(axis=(1, 3))
        valid = valid[: n * down, : m * down].reshape(n, down, m, down).all(axis=(1, 3))
    coarse = dem.reshape(n // COARSE, COARSE, m // COARSE, COARSE).mean(axis=(1, 3))
    acc = np.log1p(flow_accumulation(coarse.astype(np.float64))).astype(np.float32)
    np.savez_compressed(out, dem=dem, acc=acc, valid=valid)
    return dem.shape, float(valid.mean()), float(dem[valid].std()) if valid.any() else 0.0


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "E:/_NoiserML/lidar"
    down = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dest = sys.argv[3] if len(sys.argv) > 3 else data
    os.makedirs(os.path.join(dest, "prep"), exist_ok=True)
    for f in sorted(os.listdir(data)):
        if not f.endswith(".tif"):
            continue
        out = os.path.join(dest, "prep", f[:-4] + ".npz")
        if os.path.exists(out):
            continue
        try:
            shape, valid, std = prep(os.path.join(data, f), out, down)
            print(f"{f}: {shape}, {valid * 100:.0f}% valid, height std {std:.0f} m", flush=True)
        except Exception as e:
            print(f"{f}: skipped ({e})", flush=True)


if __name__ == "__main__":
    main()
