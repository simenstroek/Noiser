"""Reference evaluator for SDF Modeler 0.6 assemblies (no splines), calibrated against exported meshes."""
import numpy as np
from sdfcal import rot_xyz, sd_uberprim, sd_cylinder, sd_ngon, sd_triprism

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
        r = r + 0.01
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
MIRROR_BLEND = os.environ.get('MB', '1') == '1'
CAP_RY = os.environ.get('CAP', '1') == '1'


def cube_inputs(node, half):
    """The uber primitive's inputs (W, Rv, ry, rz) for a cube; see docs/sdf-modeler-notes.md."""
    hxz = min(half[0], half[2])
    ry = node['round'] * half.min()
    t = node['thickness']
    hollow = t < 0.999
    wall = 0.5 * hxz * (0.05 + 0.95 * t)
    if hollow:
        ry = min(ry, wall)
    rv = max(ry, node['bevel'] * hxz)
    rz = -node['cone'] * max(hxz - ry, 0.0)
    w = wall + abs(rz) if hollow else 1e3
    return w, rv, ry, rz


def prim_cube(p, node):
    half = np.array(node['scale']) / 2
    w, rv, ry, rz = cube_inputs(node, half)
    q = p.copy()
    q[:, 1] = -q[:, 1]
    return sd_uberprim(q, [*half, w], [rv, ry, rz])


def prim_sphere(p, node):
    half = np.array(node['scale']) / 2
    return sd_ellipsoid(p, half)



SWK = float(os.environ.get('SWK', '1.1'))
RXK = float(os.environ.get('RXK', '1.0'))


def round_bevel(node, half):
    sw = half.min() * (0.01 + 0.99 * node['round'])
    rx = max(sw, RXK * node['bevel'] * half.min())
    return sw, rx


def prim_cylinder(p, node):
    half = np.array(node['scale']) / 2
    sw, rx = round_bevel(node, half)
    return sd_cylinder(p, half - sw) - sw


def prim_prism(p, node):
    half = np.array(node['scale']) / 2
    sw, rx = round_bevel(node, half)
    return sd_triprism(p, half - rx, max(rx - sw, 0.0), 0.0) - sw


def prim_polygon(p, node):
    half = np.array(node['scale']) / 2
    sw, rx = round_bevel(node, half)
    w = 1.0 - node['star']
    return sd_ngon(p, np.array([half[0] - rx, half[1] - sw]), float(node['points']), w, max(rx - sw, 0.0), 0.0) - sw


PRIMS = {'cube': prim_cube, 'sphere': prim_sphere, 'cylinder': prim_cylinder, 'prism': prim_prism, 'polygon': prim_polygon}



# ---- tree ---------------------------------------------------------------------------------------------------------



def smooth_abs(a, k):
    a = np.abs(a)
    if k <= 0:
        return a
    m = k
    n = k * 0.25
    k0 = 2.0 * n - m
    k1 = 2.0 * m - 3.0 * n
    t = np.minimum(a / m, 1.0)
    return np.where(a > m, a, (k0 * t + k1) * t * t + n)


def repeat_radial(p, radius, count, axis):
    """Copies of the shape around a local axis; each point looks at the two nearest copies. Returns two frames."""
    if axis == 'x':
        r = p[:, [2, 1]]
    elif axis == 'y':
        r = p[:, [2, 0]]
    else:
        r = p[:, [1, 0]]
    step = 2.0 * np.pi / count
    ident = np.floor(np.arctan2(r[:, 1], r[:, 0]) / step)
    frames = []
    for a in (step * ident, step * (ident + 1.0)):
        c, s = np.cos(a), np.sin(a)
        r1 = np.stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1]], 1)
        if axis == 'x':
            q = np.stack([p[:, 0], r1[:, 1], r1[:, 0] - radius], 1)
        elif axis == 'y':
            q = np.stack([r1[:, 1], p[:, 1], r1[:, 0] - radius], 1)
        else:
            q = np.stack([r1[:, 1], r1[:, 0] - radius, p[:, 2]], 1)
        frames.append(q)
    return frames


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


def shape_distance(p, node):
    brush = node['brush']
    q = p - np.array(node['position'])
    q = q @ rot_xyz(np.array(node['rotation']))          # local = R^T (p - pos)
    if node['repeat_mode'] == 'radial' and node['radial_repeats'] > 1:
        a, b = repeat_radial(q, node['radius'], node['radial_repeats'], node['repeat_axis'])
        return np.minimum(PRIMS[brush](a, node), PRIMS[brush](b, node))
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
                    q[:, axis] = smooth_abs(q[:, axis], node['mirror_blend']) if MIRROR_BLEND else np.abs(q[:, axis])
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
