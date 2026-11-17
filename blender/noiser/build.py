"""Builds a node group from a `.noiser` file (docs/noiser-format.md) out of stock Blender nodes.

The same builder makes Geometry Nodes and shader groups: every node used exists in both trees.
Version 1 supports position, value, math and Perlin noise; other ops raise Unsupported, and the
fallback for those is Noiser's VDB export.
"""

import struct

import bpy

PERLIN_GROUP = "Noiser Perlin"


class Unsupported(Exception):
    pass


def f32(x):
    """Rounds to float32, so constants accumulate like Noiser's GPU code."""
    return struct.unpack("f", struct.pack("f", x))[0]


class Builder:
    def __init__(self, tree):
        self.tree = tree
        self.col = 0

    def node(self, kind, **props):
        n = self.tree.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.col * 40 % 4000, -(self.col // 100) * 300)
        self.col += 1
        return n

    def feed(self, socket, value):
        """Links a socket into `socket`, or sets a constant (number or 3-tuple)."""
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, op, a, b=0.0):
        n = self.node("ShaderNodeMath", operation=op)
        self.feed(n.inputs[0], a)
        self.feed(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=(0.0, 0.0, 0.0), out=0):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        if op == "SCALE":
            self.feed(n.inputs["Scale"], b)
        else:
            self.feed(n.inputs[1], b)
        return n.outputs[out]

    def mix(self, a, b, t):
        n = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=False)
        by_id = {s.identifier: s for s in n.inputs}
        self.feed(by_id["Factor_Float"], t)
        self.feed(by_id["A_Float"], a)
        self.feed(by_id["B_Float"], b)
        return next(s for s in n.outputs if s.identifier == "Result_Float")

    def combine(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for s, v in zip(n.inputs, (x, y, z)):
            self.feed(s, v)
        return n.outputs[0]

    def separate(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.feed(n.inputs[0], v)
        return n.outputs


def new_group(name, tree_type, inputs, outputs):
    g = bpy.data.node_groups.new(name, tree_type)
    for sname, stype in inputs:
        g.interface.new_socket(sname, in_out="INPUT", socket_type=stype)
    for sname, stype in outputs:
        g.interface.new_socket(sname, in_out="OUTPUT", socket_type=stype)
    gin = g.nodes.new("NodeGroupInput")
    gout = g.nodes.new("NodeGroupOutput")
    gout.location = (600, 0)
    return g, gin, gout


def perlin_group(tree_type):
    """perlin(x, period, seed) from the spec, as a reusable group (one per tree type)."""
    name = f"{PERLIN_GROUP} ({'GN' if tree_type == 'GeometryNodeTree' else 'Shader'})"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    g, gin, gout = new_group(
        name,
        tree_type,
        [("X", "NodeSocketVector"), ("Period", "NodeSocketFloat"), ("Seed", "NodeSocketFloat")],
        [("Value", "NodeSocketFloat")],
    )
    b = Builder(g)
    x, period, seed = gin.outputs["X"], gin.outputs["Period"], gin.outputs["Seed"]
    i = b.vmath("FLOOR", x)
    f = b.vmath("SUBTRACT", x, i)
    # w = f³ (f (6f − 15) + 10)
    inner = b.vmath("MULTIPLY_ADD", f, (6.0, 6.0, 6.0))
    b.feed(inner.node.inputs[2], (-15.0, -15.0, -15.0))
    poly = b.vmath("MULTIPLY_ADD", f, inner)
    b.feed(poly.node.inputs[2], (10.0, 10.0, 10.0))
    f3 = b.vmath("MULTIPLY", b.vmath("MULTIPLY", f, f), f)
    w = b.separate(b.vmath("MULTIPLY", f3, poly))
    # wrap(c) = c − period·floor(c / period); Vector Math divide is safe, so period 0 leaves c.
    pv = b.combine(period, period, period)

    def corner(o):
        c = b.vmath("ADD", i, o)  # always add, so a floor of −0.0 becomes +0.0
        c = b.vmath("SUBTRACT", c, b.vmath("MULTIPLY", pv, b.vmath("FLOOR", b.vmath("DIVIDE", c, pv))))
        white = b.node("ShaderNodeTexWhiteNoise", noise_dimensions="4D")
        b.feed(white.inputs["Vector"], c)
        b.feed(white.inputs["W"], seed)
        grad = b.vmath("MULTIPLY_ADD", white.outputs["Color"], (2.0, 2.0, 2.0))
        b.feed(grad.node.inputs[2], (-1.0, -1.0, -1.0))
        grad = b.vmath("NORMALIZE", grad)
        return b.vmath("DOT_PRODUCT", grad, b.vmath("SUBTRACT", f, o), out="Value")

    n = {o: corner(o) for o in [(ox, oy, oz) for oz in (0.0, 1.0) for oy in (0.0, 1.0) for ox in (0.0, 1.0)]}
    y0 = b.mix(b.mix(n[0, 0, 0], n[1, 0, 0], w[0]), b.mix(n[0, 1, 0], n[1, 1, 0], w[0]), w[1])
    y1 = b.mix(b.mix(n[0, 0, 1], n[1, 0, 1], w[0]), b.mix(n[0, 1, 1], n[1, 1, 1], w[0]), w[1])
    v = b.math("MULTIPLY", b.mix(y0, y1, w[2]), 1.5)
    clamp = b.node("ShaderNodeClamp")
    b.feed(clamp.inputs["Value"], v)
    clamp.inputs["Min"].default_value, clamp.inputs["Max"].default_value = -1.0, 1.0
    g.links.new(clamp.outputs[0], gout.inputs["Value"])
    return g


def perlin_call(b, tree_type, x, period, seed):
    n = b.node("GeometryNodeGroup" if tree_type == "GeometryNodeTree" else "ShaderNodeGroup")
    n.node_tree = perlin_group(tree_type)
    b.feed(n.inputs["X"], x)
    n.inputs["Period"].default_value = float(period)
    n.inputs["Seed"].default_value = float(seed)
    return n.outputs[0]


def noise(b, tree_type, op, q):
    if op["kind"] != "perlin":
        raise Unsupported(f"{op['kind']} noise (use Noiser's VDB export for now)")
    scale, octaves, lac = op["scale"], max(op["octaves"], 1), op["lacunarity"]
    seed = op["seed"]
    tiles = float(scale).is_integer() and (octaves <= 1 or float(lac).is_integer())
    period = int(scale) if tiles else 0
    x = b.vmath("SCALE", q, scale)
    if op["distortion"] != 0.0:
        d = b.combine(*(perlin_call(b, tree_type, x, period, seed + 1 + c) for c in range(3)))
        x = b.vmath("ADD", x, b.vmath("SCALE", d, op["distortion"]))
    total, amp, acc = 0.0, 1.0, None
    for o in range(octaves):
        term = b.math("MULTIPLY", perlin_call(b, tree_type, x, period, seed + o), amp)
        acc = term if acc is None else b.math("ADD", acc, term)
        total = f32(total + amp)
        amp = f32(amp * op["roughness"])
        x = b.vmath("SCALE", x, lac)
        period *= int(lac) if tiles else 0
    v = b.math("DIVIDE", acc, max(total, 1e-6))
    if op["normalize"]:
        v = b.math("MULTIPLY_ADD", v, 0.5)
        b.feed(v.node.inputs[2], 0.5)
    return v


MATH_OPS = {"add": "ADD", "subtract": "SUBTRACT", "multiply": "MULTIPLY", "divide": "DIVIDE", "min": "MINIMUM", "max": "MAXIMUM"}


def build(data, name, tree_type="GeometryNodeTree"):
    """Returns a node group with a Position (vector) input and a Density (float) output."""
    if data.get("format") != "noiser" or data.get("version") != 1:
        raise Unsupported("not a version 1 .noiser file")
    g, gin, gout = new_group(name, tree_type, [("Position", "NodeSocketVector")], [("Density", "NodeSocketFloat")])
    if tree_type == "GeometryNodeTree":
        sock = g.interface.items_tree["Position"]
        sock.default_input = "POSITION"  # behaves like a texture node: unconnected means the point's position
        sock.hide_value = True
    b = Builder(g)
    out = {}
    for op in data["ops"]:
        kind = op["op"]
        if kind == "position":
            v = gin.outputs["Position"]
        elif kind == "value":
            if op["type"] == "vector":
                v = b.combine(*op["value"])
            else:
                vn = b.node("ShaderNodeValue")
                vn.outputs[0].default_value = op["value"]
                v = vn.outputs[0]
        elif kind == "math":
            operand = lambda x: out[x["node"]] if isinstance(x, dict) else (float(x),) * 3 if op["type"] == "vector" else float(x)
            fn = MATH_OPS[op["fn"]]
            # Links convert like Blender's implicit conversions, which the format adopts.
            v = (b.vmath if op["type"] == "vector" else b.math)(fn, operand(op["a"]), operand(op["b"]))
        elif kind == "noise":
            q = gin.outputs["Position"] if op["position"] is None else out[op["position"]]
            v = noise(b, tree_type, op, q)
        else:
            raise Unsupported(f"{kind} op")
        out[op["id"]] = v
    density = data["outputs"]["density"]
    if density is None:
        gout.inputs["Density"].default_value = 0.0
    else:
        g.links.new(out[density], gout.inputs["Density"])
    return g
