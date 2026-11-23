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


def _glsl_mod(x, y):
    return x - y * np.floor(x / y)


def sd_ellipse(p, ab):
    """Distance to an axis-aligned ellipse (5 Newton steps). p is (N, 2)."""
    p = np.abs(p)
    ab = np.asarray(ab, float)
    q = ab * (p - ab)
    cs = np.where((q[:, 0] < q[:, 1])[:, None], np.array([0.01, 1.0]), np.array([1.0, 0.01]))
    cs = cs / np.linalg.norm(cs, axis=1, keepdims=True)
    for _ in range(5):
        u = ab * cs
        v = ab * np.stack([-cs[:, 1], cs[:, 0]], 1)
        a = ((p - u) * v).sum(1)
        c = ((p - u) * u).sum(1) + (v * v).sum(1)
        b = np.sqrt(np.maximum(c * c - a * a, 0))
        cs = np.stack([cs[:, 0] * b - cs[:, 1] * a, cs[:, 1] * b + cs[:, 0] * a], 1) / c[:, None]
    d = np.linalg.norm(p - ab * cs, axis=1)
    return np.where(((p / ab) ** 2).sum(1) > 1.0, d, -d)


def sd_cylinder(p, r):
    d = sd_ellipse(p[:, [0, 2]], r[[0, 2]])
    w = np.stack([d, np.abs(p[:, 1]) - r[1]], 1)
    return np.minimum(np.maximum(w[:, 0], w[:, 1]), 0) + np.linalg.norm(np.maximum(w, 0), axis=1)


def sd_ngon(p, r, n, w, bevel, thickness):
    m = n + w * (2.0 - n)
    an = np.pi / n
    en = np.pi / m
    racs = r[0] * np.array([np.cos(an), np.sin(an)])
    ecs = np.array([np.cos(en), np.sin(en)])
    x, y, z = np.abs(p[:, 0]), np.abs(p[:, 1]), p[:, 2]
    bn = _glsl_mod(np.arctan2(x, z), 2 * an) - an
    rad = np.hypot(x, z)
    px = rad * np.cos(bn) - racs[0]
    pz = rad * np.abs(np.sin(bn)) - racs[1]
    t = np.clip(-(px * ecs[0] + pz * ecs[1]), 0, racs[1] / ecs[1])
    px = px + ecs[0] * t
    pz = pz + ecs[1] * t
    d1 = np.hypot(px, pz) * np.sign(px) - bevel
    if thickness != 0.0:
        d1 = np.maximum(d1, np.abs(d1) - thickness)
    d2 = y - r[1]
    dd = np.stack([d1, d2], 1)
    return np.minimum(np.maximum(dd[:, 0], dd[:, 1]), 0) + np.linalg.norm(np.maximum(dd, 0), axis=1)


def sd_triprism(p, h, bevel, thickness):
    x = np.abs(p[:, 0]) - h[0]
    y = p[:, 1] + h[1]
    z = p[:, 2]
    end = np.array([-h[0], h[1] * 2.0])
    t = np.clip((x * end[0] + y * end[1]) / (end @ end), 0, 1)
    sx = x - end[0] * t
    sy = y - end[1] * t
    d1 = np.hypot(sx, sy)
    d1 = np.where(np.maximum(sx, sy) < 0, -np.minimum(d1, y), d1)
    d1 = d1 - bevel
    if thickness != 0.0:
        d1 = np.maximum(d1, np.abs(d1) - thickness)
    d2 = np.abs(z) - h[2] - bevel
    dd = np.stack([d1, d2], 1)
    return np.linalg.norm(np.maximum(dd, 0), axis=1) + np.minimum(np.maximum(d1, d2), 0)
