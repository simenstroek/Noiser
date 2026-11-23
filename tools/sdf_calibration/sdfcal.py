"""Calibration harness: compare candidate SDF maths with meshes exported from SDF Modeler."""
import json
import os
import numpy as np

# The folder with SDF Modeler's sample .sdf files and the .ply files exported from them (Export > PLY in SDF Modeler).
# `name.ply` is layer 1 and `name - k.ply` is layer k + 1. PLY coordinates are (x, -z, y) of the file's.
SDFS = os.environ.get("SDF_SAMPLES", os.path.expanduser(r"~\Documents\SDF Modeler\sdfs"))


def load_ply(path, max_vertices=150000, seed=0):
    """Vertex positions and colours (subsampled) of an ASCII PLY."""
    with open(path, "r") as f:
        count = 0
        header = 0
        for line in f:
            header += 1
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            if line.startswith("end_header"):
                break
        data = np.loadtxt(f, max_rows=count)
    if len(data) > max_vertices:
        rng = np.random.default_rng(seed)
        data = data[rng.choice(len(data), max_vertices, replace=False)]
    return data[:, :3], data[:, 3:6]


def load_sdf(name):
    return json.load(open(os.path.join(SDFS, f"{name}.sdf")))


def report(label, d):
    d = np.asarray(d)
    a = np.abs(d)
    print(f"{label}: n={len(d)} mean|d|={a.mean():.5f} median={np.median(a):.5f} p95={np.percentile(a, 95):.5f} max={a.max():.5f} signed mean={d.mean():+.5f}")


def rot_xyz(rot):
    """Rotation matrix for Euler angles (radians), applied x then y then z extrinsically (guess; calibrated below)."""
    cx, cy, cz = np.cos(rot); sx, sy, sz = np.sin(rot)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sd_uberprim(p, s, r):
    """The shader's uber primitive: s = (x, y, z, w) sizes, r = (corner, edge, cone) radii. p is (N, 3)."""
    s = np.array(s, float); r = np.array(r, float)
    s[[0, 2]] -= r[0]
    r[0] -= s[3]
    s[3] -= r[1]
    s[1] -= r[1]
    b = np.array([r[2], -2.0 * s[1]])
    ba = b / max(b @ b, 1e-30)
    sz2 = b[1]
    d = np.abs(p) - s[:3]
    q = np.linalg.norm(np.maximum(d[:, [0, 2]], 0), axis=1) + np.minimum(0, np.maximum(d[:, 0], d[:, 2])) - r[0]
    q = np.abs(q) - s[3]
    pa = np.stack([q, p[:, 1] - s[1]], 1)
    t = np.clip(pa @ ba, 0, 1)
    diag = pa - np.outer(t, [r[2], sz2])
    h0 = np.stack([np.maximum(q - r[2], 0), p[:, 1] + s[1]], 1)
    h1 = np.stack([np.maximum(q, 0), p[:, 1] - s[1]], 1)
    dd = np.minimum((diag ** 2).sum(1), np.minimum((h0 ** 2).sum(1), (h1 ** 2).sum(1)))
    sign = np.sign(np.maximum(pa @ np.array([-ba[1], ba[0]]), d[:, 1]))
    return np.sqrt(dd) * sign - r[1]
