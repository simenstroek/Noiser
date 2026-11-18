"""Reference implementations of the white, blue, worley and simplex noises, written only from docs/noiser-format.md.

Usage: python tools/noiser_reference.py file.noiser [...]
Each file must be a graph of one noise node (whole scale and lacunarity); the script evaluates the noise
at the file's test points and prints the largest difference from the GPU values stored in the file. It is meant as
starting point and cross-check for ports to other hosts (Houdini VEX, shaders): the same arithmetic, in Python.
"""
import json, math, subprocess, sys, os
import numpy as np

M32 = 0xFFFFFFFF


def pcg3d(x, y, z):
    x = (x * 1664525 + 1013904223) & M32
    y = (y * 1664525 + 1013904223) & M32
    z = (z * 1664525 + 1013904223) & M32
    x = (x + y * z) & M32; y = (y + z * x) & M32; z = (z + x * y) & M32
    x ^= x >> 16; y ^= y >> 16; z ^= z >> 16
    x = (x + y * z) & M32; y = (y + z * x) & M32; z = (z + x * y) & M32
    return x, y, z


def hash3(c, s):
    x, y, z = pcg3d((c[0] + s * 0x9E3779B9) & M32, c[1] & M32, c[2] & M32)
    return np.array([np.float32(x), np.float32(y), np.float32(z)], dtype=np.float32) / np.float32(4294967295.0)


def wrap(c, period):
    return tuple(int(v - period * math.floor(v / period)) for v in c) if period > 0 else tuple(int(v) for v in c)


def white(x, period, seed):
    return float(hash3(wrap([math.floor(v) for v in x], period), seed)[0]) * 2 - 1


def worley(x, period, seed, feature):
    fl = [math.floor(v) for v in x]
    i = wrap(fl, period)
    f = [x[k] - fl[k] for k in range(3)]
    best = [(1e9, None), (1e9, None)]
    for oz in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                o = (ox, oy, oz)
                c = tuple(i[k] + o[k] for k in range(3))
                if period > 0:
                    c = tuple(c[k] - period if c[k] >= period else (c[k] + period if c[k] < 0 else c[k]) for k in range(3))
                h = hash3(c, seed)
                d = [o[k] + float(h[k]) - f[k] for k in range(3)]
                d2 = sum(v * v for v in d)
                if d2 < best[0][0]:
                    best = [(d2, c), best[0]]
                elif d2 < best[1][0]:
                    best[1] = (d2, c)
    f1, f2 = math.sqrt(best[0][0]), math.sqrt(best[1][0])
    clamp = lambda v: max(-1.0, min(1.0, v))
    if feature == "f1": return clamp(f1 * 2 - 1)
    if feature == "f2": return clamp(f2 / 1.4 * 2 - 1)
    if feature == "f2_minus_f1": return clamp((f2 - f1) * 2 - 1)
    return float(hash3(best[0][1], seed + 0x51ED27)[0]) * 2 - 1


def normalize(v):
    n = max(math.sqrt(sum(a * a for a in v)), 1e-12)
    return [a / n for a in v]


def simplex(x, period, seed):
    u = [x[1] + x[2], x[0] + x[2], x[0] + x[1]]
    i0 = [math.floor(v) for v in u]
    f0 = [u[k] - i0[k] for k in range(3)]
    step = lambda edge, v: 1.0 if v >= edge else 0.0   # step(edge, x): 0 if x < edge else 1
    # g = step(f0.yzx, f0.xyz): component k is step(f0[(k+1)%3], f0[k])
    g = [step(f0[(k + 1) % 3], f0[k]) for k in range(3)]
    g[2] = 1.0 if f0[2] > f0[0] else 0.0
    l = [1 - v for v in g]
    lzxy = [l[2], l[0], l[1]]
    o1 = [min(g[k], lzxy[k]) for k in range(3)]
    o2 = [max(g[k], lzxy[k]) for k in range(3)]
    total = 0.0
    for c in ([i0[k] for k in range(3)], [i0[k] + o1[k] for k in range(3)], [i0[k] + o2[k] for k in range(3)], [i0[k] + 1 for k in range(3)]):
        v = [0.5 * (-c[0] + c[1] + c[2]), 0.5 * (c[0] - c[1] + c[2]), 0.5 * (c[0] + c[1] - c[2])]
        d = [x[k] - v[k] for k in range(3)]
        w = max(0.5 - sum(a * a for a in d), 0.0)
        if w == 0.0:
            continue
        vw = [v[k] - math.floor(v[k] / period) * period for k in range(3)] if period > 0 else v
        cell = [round(vw[1] + vw[2]), round(vw[0] + vw[2]), round(vw[0] + vw[1])]
        grad = normalize([float(a) * 2 - 1 for a in hash3(cell, seed)])
        total += w ** 4 * sum(grad[k] * d[k] for k in range(3))
    return max(-1.0, min(1.0, 90.0 * total))


_BLUE = None


def blue(x, period, seed):
    global _BLUE
    if _BLUE is None:
        data = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "bluenoise.bin"), "rb").read()
        _BLUE, at = {}, 0
        for n in (4, 8, 16, 32):
            _BLUE[n] = np.frombuffer(data[at:at + 2 * n ** 3], dtype="<u2").astype(np.float64) / (n ** 3 - 1)
            at += 2 * n ** 3
    t = 32
    if period > 0:
        if period % 32 != 0: t = 16
        if period % 16 != 0: t = 8
        if period % 8 != 0: t = 4
    shift = [int(math.floor(float(h) * t)) for h in hash3((0, 0, 0), seed)]
    c = [(math.floor(x[k]) + shift[k]) % t for k in range(3)]
    return float(_BLUE[t][c[0] + t * (c[1] + t * c[2])]) * 2 - 1


def perlin(x, period, seed):
    fl = [math.floor(v) for v in x]
    i = wrap(fl, period)
    f = [x[k] - fl[k] for k in range(3)]
    w = [v * v * v * (v * (v * 6 - 15) + 10) for v in f]
    n = {}
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                o = (ox, oy, oz)
                c = wrap([i[k] + o[k] for k in range(3)], period)
                g = normalize([float(a) * 2 - 1 for a in hash3(c, seed)])
                n[o] = sum(g[k] * (f[k] - o[k]) for k in range(3))
    mix = lambda a, b, t: a * (1 - t) + b * t
    x0 = lambda z: mix(mix(n[0, 0, z], n[1, 0, z], w[0]), mix(n[0, 1, z], n[1, 1, z], w[0]), w[1])
    return max(-1.0, min(1.0, 1.5 * mix(x0(0), x0(1), w[2])))


def noise_op(op, q):
    """The fBm wrapper of the format document (whole scale and lacunarity, so the noise tiles)."""
    scale = op["scale"]
    # The GPU multiplies in float32: points on cell seams (k / 12 at scale 24) land on either side of the seam
    # depending on that rounding, so the reference rounds the same way.
    x = [float(np.float32(v) * np.float32(scale)) for v in q]
    period = int(scale)
    kind = op["kind"]
    if kind == "white": base = white
    elif kind == "blue": base = blue
    elif kind == "perlin": base = perlin
    elif kind == "worley": base = lambda x, p, s: worley(x, p, s, op["feature"])
    else: base = simplex
    if op["distortion"] != 0:
        d = [perlin(x, period, op["seed"] + 1 + c) for c in range(3)]
        x = [x[k] + d[k] * op["distortion"] for k in range(3)]
    total = acc = 0.0
    amp = 1.0
    for o in range(max(op["octaves"], 1)):
        acc += amp * base(x, period, op["seed"] + o)
        total += amp
        amp *= op["roughness"]
        x = [v * op["lacunarity"] for v in x]
        period = int(period * op["lacunarity"])
    v = acc / max(total, 1e-6)
    return v * 0.5 + 0.5 if op["normalize"] else v


def check(path):
    d = json.load(open(path))
    op = [o for o in d["ops"] if o["op"] == "noise"][0]
    worst = 0.0
    for q, want in zip(d["tests"]["points"], d["tests"]["density"]):
        worst = max(worst, abs(noise_op(op, q) - want))
    print(os.path.basename(path), op["kind"], op.get("feature"), "max diff", worst)
    return worst


if __name__ == "__main__":
    worst = max(check(p) for p in sys.argv[1:])
    sys.exit(0 if worst < 2e-3 else 1)
