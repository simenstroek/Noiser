"""Reference evaluator for SDF Modeler 0.6 assemblies (no splines), calibrated against exported meshes."""
import numpy as np
from sdfcal import rot_xyz, sd_uberprim

# ---- blends (the shader's maths; a = the shape being added, b = what is accumulated so far) ---------------------


def union_smooth(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def sub_smooth(a, b, k):  # b minus a
    if k <= 0:
        return np.maximum(b, -a)
    h = np.clip(0.5 - 0.5 * (b + a) / k, 0, 1)
    return b * (1 - h) + (-a) * h + k * h * (1 - h)


def intersect_smooth(a, b, k):
    if k <= 0:
        return np.maximum(a, b)
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h + k * h * (1 - h)


def union_chamfer(a, b, k):
    k2 = 0.5 * k
    return np.minimum(np.minimum(a, b), (a - k2 + b) * 0.5)


def intersect_chamfer(a, b, k):
    k2 = 0.5 * k
    return np.maximum(np.maximum(a, b), (b + k2 + a) * 0.5)


def sub_chamfer(a, b, k):
    return intersect_chamfer(-a, b, k)


def blend(a, b, mode, k, r, chamfer):
    U = union_chamfer if chamfer else union_smooth
    S = sub_chamfer if chamfer else sub_smooth
    I = intersect_chamfer if chamfer else intersect_smooth
    if mode == 'add':
        return U(a, b, k)
    if mode == 'sub':
        return S(a, b, k)
    if mode == 'intersect':
        return I(a, b, k)
    if mode == 'paint':
        return b
    if mode == 'push':
        return np.minimum(S(a - r, b, k), a)
    if mode == 'avoid':
        return np.minimum(S(b - r, a, k), b)
    if mode == 'emboss':
        return U(b, I(a, b - r, k), k)
    if mode == 'deboss':
        return S(S(b + r, a, k), b, k)
    if mode == 'shell':
        return I(a, np.abs(b + r) - r, k)
    if mode == 'none':
        return b
    raise ValueError(mode)


# ---- primitives -------------------------------------------------------------------------------------------------


def sd_ellipsoid(p, r, n=2.0):
    k = np.abs(p / r) ** n
    return (k.sum(1) ** (1.0 / n) - 1.0) * r.min()


import os
RV_MODE = os.environ.get('RV', 'max')
CAP_RY = os.environ.get('CAP', '1') == '1'


def cube_params(node, half):
    hxz = min(half[0], half[2])
    ry = node['round'] * half.min()
    if RV_MODE == 'max':
        rv = max(ry, node['bevel'] * hxz)
    elif RV_MODE == 'lerp':
        rv = ry + node['bevel'] * (hxz - ry)
    else:
        rv = node['bevel'] * hxz
    hc = os.environ.get('HC', 'height')
    if hc == 'height':
        rz = -node['cone'] * hxz * (half[1] - ry) / half[1]
    elif hc == 'plain':
        rz = -node['cone'] * hxz
    else:  # taper measured on the straight part only, then scaled to the full height
        rz = -node['cone'] * hxz * (half[1] - ry) / half[1] * (1.0 + float(hc) * ry / half[1])
    return rv, ry, rz


def cube_wall(node, half):
    """The uber primitive's `W`: half the wall thickness. Thickness 1 is solid."""
    t = node['thickness']
    if t >= 0.999:
        return 1e3
    return 0.5 * min(half[0], half[2]) * (0.05 + 0.95 * t)


def prim_cube(p, node):
    half = np.array(node['scale']) / 2
    rv, ry, rz = cube_params(node, half)
    wall = cube_wall(node, half)
    if wall < 1e2:
        wall += abs(rz)
    if CAP_RY:
        ry_new = min(ry, wall)
        if ry > 0:
            rz = rz * (half[1] - ry_new) / (half[1] - ry) if half[1] > ry else rz
        ry = ry_new
    q = p.copy()
    q[:, 1] = -q[:, 1]
    return sd_uberprim(q, [*half, wall], [rv, ry, rz])


def prim_sphere(p, node):
    half = np.array(node['scale']) / 2
    return sd_ellipsoid(p, half)


PRIMS = {'cube': prim_cube, 'sphere': prim_sphere}


# ---- tree ---------------------------------------------------------------------------------------------------------


def repeat_linear(p, steps, counts):
    n = np.array(counts, float) - 1
    steps = np.array(steps, float)
    p = p + 0.5 * steps * n
    ids = np.clip(np.round(p / steps), 0, n)
    r = p - steps * ids
    odd = (ids.astype(int) & 1) == 1
    return np.where(odd, -r, r)


def local_point(p, node, scale_to_local=1.0):
    """World (parent-space) points -> the shape's own frame, mirror and flip included."""
    return p


def shape_distance(p, node, mirror_blend=0.0):
    brush = node['brush']
    q = p - np.array(node['position'])
    q = q @ rot_xyz(np.array(node['rotation']))          # local = R^T (p - pos)
    if node['repeat_mode'] == 'linear' and any(c > 1 for c in node['linear_repeats']):
        q = repeat_linear(q, node['steps'], node['linear_repeats'])
    return PRIMS[brush](q, node)


def eval_nodes(p, nodes, ctx):
    """Combine a list of sibling nodes front to back; returns the accumulated distance (or None)."""
    acc = None
    for node in nodes:
        if node.get('hidden'):
            continue
        if node['type'] == 'shape':
            q = p
            for axis, key in enumerate(('mirror_x', 'mirror_y', 'mirror_z')):
                if node.get(key):
                    q = q.copy()
                    q[:, axis] = np.abs(q[:, axis])
            d = shape_distance(q, node)
            k = node['blend']
            r = node['blend_strength']
        elif node['type'] == 'group':
            t = np.array(node['position']); s = node['scale'][0]
            q = (p - t) @ rot_xyz(np.array(node['rotation'])) / s
            d = eval_nodes(q, node['children'], ctx)
            if d is None:
                continue
            d = d * abs(s)
            k = node['blend']; r = node['blend_strength']
        else:
            continue
        if acc is None:
            acc = d
        else:
            acc = blend(d, acc, node['mode'], k, r, node['chamfer'])
    return acc
