"""A stdio MCP server over the noiser.exe command line: schema, describe, build, validate, preview, run a mesh graph.

    claude mcp add noiser -- python tools/mcp/noiser_mcp.py

Needs Python 3 only. NOISER_EXE names the executable (default: target/release/noiser.exe next to this repo).
"""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXE = os.environ.get("NOISER_EXE") or os.path.join(ROOT, "target", "release", "noiser.exe" if os.name == "nt" else "noiser")

TOOLS = {
    "noiser_schema": ("Every Noiser node kind with default settings, enum values and file formats (JSON). Read this before writing a graph.", {}, lambda a: ["--schema"]),
    "noiser_describe": ("A .noisegraph/.meshgraph/.landscape as a name-based spec (the format noiser_build takes; a .landscape also shows its settings).", {"file": "path of the graph"}, lambda a: ["--describe", a["file"]]),
    "noiser_build": (
        "Build a graph file from a spec: {\"nodes\": {name: node}, \"edges\": [[from, to, input_port, (output_port)]]}. Settings left out keep defaults. Relative file paths in nodes are relative to the output file's folder. The result is validated. For a .landscape output the spec may also have \"settings\": {seed, world_m, height_m, resolution, fit, water_m}; Landscape nodes (Erosion, Stamp, Heightmap, Select, Layer, Terrace, Craters, Map, Slope) are listed by noiser_schema.",
        {"spec": "the spec as a JSON object", "output": "path of the .noisegraph or .meshgraph to write"},
        None,
    ),
    "noiser_validate": ("Check a .noisegraph, .meshgraph, .sdf or .landscape file: errors and warnings.", {"file": "path"}, lambda a: ["--validate", a["file"]]),
    "noiser_preview": (
        "Render a PNG of a .noisegraph (slice), .sdf (front/side/top), mesh (.obj/.ply/.glb) or .landscape (map from above and 3D view side by side, painted in the Layer colours, Erosion nodes run) so you can look at it (read the PNG afterwards).",
        {"file": "input path", "output": "PNG path", "size": "optional pixels per view", "up": "optional: 'y' when the mesh is Y-up (OBJ from the SDF side)"},
        lambda a: ["--preview", a["file"], a["output"]] + (["--size", str(a["size"])] if a.get("size") else []) + (["--up", a["up"]] if a.get("up") else []),
    ),
    "noiser_terrain_graph": (
        "Write the Mesh graph that cuts an exported heightmap into Nanite chunks (Load Mesh in cm with the paint layers loaded, Remesh 250k per chunk, planar UVs, normal/occlusion/curvature maps, the layers' weights baked per chunk and packed as terrain_splat). Edit it with noiser_describe / noiser_build, run it with noiser_run_meshgraph.",
        {"heightmap": "the .png written by noiser_landscape_export (its .json beside it)", "output": "path of the .meshgraph to write"},
        lambda a: ["--terrain-graph", a["heightmap"], a["output"]],
    ),
    "noiser_run_meshgraph": ("Run a .meshgraph headless: writes its meshes and images. Can take minutes for big terrains.", {"file": "path of the .meshgraph"}, lambda a: ["--headless", "--mesh-graph", a["file"]]),
    "noiser_landscape_export": ("Export a Landscape project's 4033 x 4033 heightmap PNG (with a .json sidecar), and one 8-bit weightmap PNG per Layer node (<name>_<layer>.png, listed in the sidecar), headless. Erosion nodes run first.", {"project": "a .landscape file", "output": "PNG path"}, lambda a: ["--headless", "--landscape-export", a["project"], a["output"]]),
}


def run(name, args):
    if name == "noiser_build":
        spec_path = args["output"] + ".spec.tmp.json"
        with open(spec_path, "w", encoding="utf8") as f:
            json.dump(args["spec"], f)
        try:
            argv = ["--build", spec_path, args["output"]]
            p = subprocess.run([EXE, *argv], capture_output=True, text=True, timeout=600)
        finally:
            os.remove(spec_path)
    else:
        p = subprocess.run([EXE, *TOOLS[name][2](args)], capture_output=True, text=True, timeout=3600)
    text = (p.stdout + p.stderr).strip()
    if len(text) > 6000:  # a long run logs a line per chunk and map: keep the start and the end, which has the result
        omitted = len(text) - 5500
        text = text[:1500] + f"\n... ({omitted} characters left out; the full log is next to the graph file as .log) ...\n" + text[-4000:]
    return (text or "(no output)") + ("" if p.returncode == 0 else f"\n[exit code {p.returncode}]"), p.returncode != 0


def tool_list():
    out = []
    for name, (doc, props, _) in TOOLS.items():
        schema = {"type": "object", "properties": {k: ({"type": "object"} if k == "spec" else {"type": "string" if k != "size" else "integer"}) | {"description": v} for k, v in props.items()}}
        schema["required"] = [k for k, v in props.items() if not v.startswith("optional")]
        out.append({"name": name, "description": doc, "inputSchema": schema})
    return out


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg["error" if error else "result"] = error or result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        method, id_ = req.get("method"), req.get("id")
        if id_ is None:
            continue  # a notification (initialized, cancelled): nothing to answer
        if method == "initialize":
            reply(id_, {"protocolVersion": req["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "noiser", "version": "1.0"}})
        elif method == "tools/list":
            reply(id_, {"tools": tool_list()})
        elif method == "tools/call":
            name, args = req["params"]["name"], req["params"].get("arguments", {})
            if name not in TOOLS:
                reply(id_, error={"code": -32602, "message": f"unknown tool {name}"})
                continue
            try:
                text, failed = run(name, args)
            except Exception as e:  # a missing exe, a missing argument, a timeout
                text, failed = f"{type(e).__name__}: {e}", True
            reply(id_, {"content": [{"type": "text", "text": text}], "isError": failed})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"unknown method {method}"})


if __name__ == "__main__":
    main()
