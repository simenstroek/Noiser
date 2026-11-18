"""Builds a node group from a `.noiser` file (docs/noiser-format.md) out of stock Blender nodes.

Geometry Nodes only: the pcg3d hash needs Integer Math and Bit Math, and Blender has no Bit Math
node in shader trees. Version 1 supports position, value, math (all functions; some only on floats), remap, mix, separate, combine and
Perlin noise; other ops raise Unsupported, and the fallback for those (and for shaders) is Noiser's VDB or slice-grid export.
"""

import struct

import bpy

PERLIN_GROUP = "Noiser Perlin"
GN = "GeometryNodeTree"


class Unsupported(Exception):
    pass


def f32(x):
    """Rounds to float32, so constants accumulate like Noiser's GPU code."""
    return struct.unpack("f", struct.pack("f", x))[0]


def s32(v):
    """A u32 constant as the signed int32 with the same bits (Blender ints are signed)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


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

    def mix(self, a, b, t, clamp=False):
        n = self.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=clamp)
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

    # ---- integer ops (Blender ints are int32; add and multiply wrap, matching u32 bit for bit) ----

    def imath(self, op, a, b):
        n = self.node("FunctionNodeIntegerMath", operation=op)
        self.feed(n.inputs[0], s32(a) if isinstance(a, int) else a)
        self.feed(n.inputs[1], s32(b) if isinstance(b, int) else b)
        return n.outputs[0]

    def shr16(self, v):
        """Logical right shift by 16 (a negative Shift is a right shift that fills with zeros)."""
        n = self.node("FunctionNodeBitMath", operation="SHIFT")
        self.feed(n.inputs["A"], v)
        n.inputs["Shift"].default_value = -16
        return n.outputs[0]

    def bit(self, op, a, b):
        n = self.node("FunctionNodeBitMath", operation=op)
        self.feed(n.inputs["A"], a)
        self.feed(n.inputs["B"], s32(b) if isinstance(b, int) else b)
        return n.outputs[0]

    def float_to_int(self, f):
        n = self.node("FunctionNodeFloatToInt", rounding_mode="ROUND")
        self.feed(n.inputs[0], f)
        return n.outputs[0]

    def u32_to_float(self, v):
        """f32(u32 v) exactly: both 16-bit halves are exact in float, so one multiply-add rounds once.
        Links convert int to float implicitly."""
        n = self.node("ShaderNodeMath", operation="MULTIPLY_ADD")
        self.feed(n.inputs[0], self.shr16(v))
        n.inputs[1].default_value = 65536.0
        self.feed(n.inputs[2], self.bit("AND", v, 0xFFFF))
        return n.outputs[0]


def pcg_hash3(b, cell, seed_mul):
    """Noiser's hash01: three floats in [0, 1] for an integer cell (3 float sockets) and
    `seed_mul` = seed * 0x9E3779B9 (an int socket). pcg3d, Jarzynski and Olano 2020."""
    x, y, z = (b.float_to_int(c) for c in cell)
    x = b.imath("ADD", x, seed_mul)
    lcg = lambda v: b.imath("ADD", b.imath("MULTIPLY", v, 1664525), 1013904223)
    x, y, z = lcg(x), lcg(y), lcg(z)

    def step(x, y, z):
        x = b.imath("ADD", x, b.imath("MULTIPLY", y, z))
        y = b.imath("ADD", y, b.imath("MULTIPLY", z, x))
        z = b.imath("ADD", z, b.imath("MULTIPLY", x, y))
        return x, y, z

    x, y, z = step(x, y, z)
    x, y, z = (b.bit("XOR", v, b.shr16(v)) for v in (x, y, z))
    x, y, z = step(x, y, z)
    return [b.math("DIVIDE", b.u32_to_float(v), 4294967295.0) for v in (x, y, z)]


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


def perlin_group():
    """perlin(x, period, seed) from the spec, as a reusable group."""
    if PERLIN_GROUP in bpy.data.node_groups:
        return bpy.data.node_groups[PERLIN_GROUP]
    g, gin, gout = new_group(
        PERLIN_GROUP,
        GN,
        [("X", "NodeSocketVector"), ("Period", "NodeSocketFloat"), ("Seed", "NodeSocketInt")],
        [("Value", "NodeSocketFloat")],
    )
    b = Builder(g)
    x, period, seed = gin.outputs["X"], gin.outputs["Period"], gin.outputs["Seed"]
    seed_mul = b.imath("MULTIPLY", seed, 0x9E3779B9)
    i = b.vmath("FLOOR", x)
    f = b.vmath("SUBTRACT", x, i)
    # w = f³ (f (6f − 15) + 10)
    inner = b.vmath("MULTIPLY_ADD", f, (6.0, 6.0, 6.0))
    inner.node.inputs[2].default_value = (-15.0, -15.0, -15.0)
    poly = b.vmath("MULTIPLY_ADD", f, inner)
    poly.node.inputs[2].default_value = (10.0, 10.0, 10.0)
    f3 = b.vmath("MULTIPLY", b.vmath("MULTIPLY", f, f), f)
    w = b.separate(b.vmath("MULTIPLY", f3, poly))
    # wrap(c) = c − period·floor(c / period); Vector Math divide is safe, so period 0 leaves c.
    pv = b.combine(period, period, period)

    def corner(o):
        c = b.vmath("ADD", i, o)
        c = b.vmath("SUBTRACT", c, b.vmath("MULTIPLY", pv, b.vmath("FLOOR", b.vmath("DIVIDE", c, pv))))
        h = pcg_hash3(b, b.separate(c), seed_mul)
        grad = b.vmath("MULTIPLY_ADD", b.combine(*h), (2.0, 2.0, 2.0))
        grad.node.inputs[2].default_value = (-1.0, -1.0, -1.0)
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


def perlin_call(b, x, period, seed):
    n = b.node("GeometryNodeGroup")
    n.node_tree = perlin_group()
    b.feed(n.inputs["X"], x)
    n.inputs["Period"].default_value = float(period)
    n.inputs["Seed"].default_value = s32(seed)
    return n.outputs[0]


def noise(b, op, q):
    if op["kind"] != "perlin":
        raise Unsupported(f"{op['kind']} noise (use Noiser's VDB export for now)")
    scale, octaves, lac = op["scale"], max(op["octaves"], 1), op["lacunarity"]
    seed = op["seed"]
    tiles = float(scale).is_integer() and (octaves <= 1 or float(lac).is_integer())
    period = int(scale) if tiles else 0
    x = b.vmath("SCALE", q, scale)
    if op["distortion"] != 0.0:
        d = b.combine(*(perlin_call(b, x, period, seed + 1 + c) for c in range(3)))
        x = b.vmath("ADD", x, b.vmath("SCALE", d, op["distortion"]))
    total, amp, acc = 0.0, 1.0, None
    for o in range(octaves):
        term = b.math("MULTIPLY", perlin_call(b, x, period, seed + o), amp)
        acc = term if acc is None else b.math("ADD", acc, term)
        total = f32(total + amp)
        amp = f32(amp * op["roughness"])
        x = b.vmath("SCALE", x, lac)
        period *= int(lac) if tiles else 0
    v = b.math("DIVIDE", acc, max(total, 1e-6))
    if op["normalize"]:
        v = b.math("MULTIPLY_ADD", v, 0.5)
        v.node.inputs[2].default_value = 0.5
    return v


MATH_OPS = {
    "add": "ADD", "subtract": "SUBTRACT", "multiply": "MULTIPLY", "divide": "DIVIDE", "min": "MINIMUM", "max": "MAXIMUM",
    "power": "POWER", "absolute": "ABSOLUTE", "floor": "FLOOR", "fract": "FRACT", "sine": "SINE", "sqrt": "SQRT",
    "modulo": "FLOORED_MODULO", "greater": "GREATER_THAN",
}
# The same ops on vectors (Vector Math has fewer; the others are Unsupported for vectors).
VECTOR_MATH_OPS = {
    "add": "ADD", "subtract": "SUBTRACT", "multiply": "MULTIPLY", "divide": "DIVIDE", "min": "MINIMUM", "max": "MAXIMUM",
    "absolute": "ABSOLUTE", "floor": "FLOOR", "fract": "FRACTION", "sine": "SINE",
}


def math_op(b, op, operand):
    """One `math` op of the file as stock nodes (float or vector, as the op's type says)."""
    fn, vector = op["fn"], op["type"] == "vector"
    a, c = operand(op["a"]), operand(op["b"])
    if fn == "dot":
        # Both operands are vectors here; a float constant is repeated (linked floats convert on their own).
        a, c = [(v, v, v) if isinstance(v, float) else v for v in (a, c)]
        return b.vmath("DOT_PRODUCT", a, c, out=1)
    if fn in ("smooth_min", "smooth_max"):
        if vector:
            raise Unsupported(f"{fn} on vectors")
        n = b.node("ShaderNodeMath", operation="SMOOTH_MIN" if fn == "smooth_min" else "SMOOTH_MAX")
        b.feed(n.inputs[0], a)
        b.feed(n.inputs[1], c)
        b.feed(n.inputs[2], operand(op["k"]))
        return n.outputs[0]
    if fn == "ridge":
        if vector:
            return b.vmath("SUBTRACT", (1.0, 1.0, 1.0), b.vmath("ABSOLUTE", a))
        return b.math("SUBTRACT", 1.0, b.math("ABSOLUTE", a))
    if vector:
        if fn not in VECTOR_MATH_OPS:
            raise Unsupported(f"{fn} on vectors")
        return b.vmath(VECTOR_MATH_OPS[fn], a, c)
    return b.math(MATH_OPS[fn], a, c)


def build(data, name, tree_type=GN):
    """Returns a node group with a Position (vector) input and a Density (float) output."""
    if data.get("format") != "noiser" or data.get("version") != 1:
        raise Unsupported("not a version 1 .noiser file")
    if tree_type != GN:
        raise Unsupported("shader node trees have no Bit Math node, which the hash needs; bake to a texture instead")
    g, gin, gout = new_group(name, tree_type, [("Position", "NodeSocketVector")], [("Density", "NodeSocketFloat")])
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
            # Links convert like Blender's implicit conversions, which the format adopts.
            v = math_op(b, op, operand)
        elif kind == "remap":
            fl = lambda x: out[x["node"]] if isinstance(x, dict) else float(x)
            n = b.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="LINEAR", clamp=bool(op["clamp"]))
            for name, key in [("Value", "value"), ("From Min", "from_min"), ("From Max", "from_max"), ("To Min", "to_min"), ("To Max", "to_max")]:
                b.feed(n.inputs[name], fl(op[key]))
            v = n.outputs[0]
        elif kind == "mix":
            fl = lambda x: out[x["node"]] if isinstance(x, dict) else float(x)
            v = b.mix(fl(op["a"]), fl(op["b"]), fl(op["factor"]), clamp=bool(op["clamp"]))
        elif kind == "curve":
            if op["mode"] != "linear":
                raise Unsupported(f"{op['mode']} curves (Blender's Float Curve is a spline; only linear segments match)")
            xs0 = [p[0] for p in op["points"]]
            if len(xs0) < 2 or len(set(xs0)) < len(xs0):
                raise Unsupported("a curve needs two or more points with different x")
            fl = lambda x: out[x["node"]] if isinstance(x, dict) else float(x)
            n = b.node("ShaderNodeFloatCurve")
            xs = [p[0] for p in op["points"]]
            span = max(xs) - min(xs) or 1.0
            m = n.mapping
            m.use_clip = False
            m.extend = "HORIZONTAL"  # hold the end values, as the format says
            curve = m.curves[0]
            while len(curve.points) > len(op["points"]):
                curve.points.remove(curve.points[-1])
            while len(curve.points) < len(op["points"]):
                curve.points.new(0.5, 0.5)
            for p, (x, y) in zip(curve.points, op["points"]):
                p.location = (x, y)
                p.handle_type = "VECTOR"
            m.update()
            b.feed(n.inputs[0], 1.0)
            b.feed(n.inputs[1], fl(op["value"]))
            v = n.outputs[0]
        elif kind == "separate":
            n = b.node("ShaderNodeSeparateXYZ")
            if op["vector"] is not None:
                g.links.new(out[op["vector"]], n.inputs[0])
            v = n.outputs["XYZ".index(op["component"].upper())]
        elif kind == "combine":
            fl = lambda x: out[x["node"]] if isinstance(x, dict) else float(x)
            v = b.combine(fl(op["x"]), fl(op["y"]), fl(op["z"]))
        elif kind == "noise":
            q = gin.outputs["Position"] if op["position"] is None else out[op["position"]]
            v = noise(b, op, q)
        else:
            raise Unsupported(f"{kind} op")
        out[op["id"]] = v
    density = data["outputs"]["density"]
    if density is None:
        gout.inputs["Density"].default_value = 0.0
    else:
        g.links.new(out[density], gout.inputs["Density"])
    return g
