"""Copies a Gaea 2 project with an Export node added on its final height (the node feeding the colour nodes, or the last
node), so Gaea's Build Swarm writes a heightmap (Export.exr, one height channel) for it.

    python tools/ml/gaea_add_export.py IN.terrain OUT.terrain

Gaea's examples (E:/Gaea 2.0/Examples) have no Export node; the user's own projects do. The file is Newtonsoft JSON with
$id/$ref references: the new node gets $ids above every existing one.
"""

import json
import re
import sys

# Nodes that make colour or masks rather than height: the height to export is what feeds them.
NOT_HEIGHT = {"SatMap", "TextureBase", "Texture", "CLUTer", "SuperColor", "Mixer", "RGBMix", "RGBSplit", "RGBMerge",
              "Weathering", "Synth", "ColorErosion", "Export", "Unity", "Unreal", "Height", "Slope", "Curvature",
              "Occlusion", "Normals", "FlowMap", "Soil", "Data", "Mask", "Combine2", "Peaks", "Angle", "Splat"}


def kind(node):
    return node["$type"].split(",")[0].rsplit(".", 1)[-1]


def edges(nodes):
    """(from, to) node ids of every connection."""
    out = []
    for n in nodes.values():
        if not isinstance(n, dict):
            continue
        for port in n.get("Ports", {}).get("$values", []):
            r = port.get("Record")
            if r and r.get("IsValid", True):
                out.append((r["From"], r["To"]))
    return out


def final_height(nodes):
    ids = {int(k): v for k, v in nodes.items() if isinstance(v, dict) and "$type" in v}
    e = edges(nodes)
    height = {i for i, n in ids.items() if kind(n) not in NOT_HEIGHT}
    # A height node none of whose readers is a height node: an end of the height chain.
    ends = [i for i in height if not any(f == i and t in height for f, t in e)]
    if not ends:
        raise SystemExit("no height node")
    # The end with the most nodes upstream: the main terrain, not a side branch.
    def upstream(i, seen=None):
        seen = seen if seen is not None else set()
        for f, t in e:
            if t == i and f not in seen:
                seen.add(f)
                upstream(f, seen)
        return seen
    return max(ends, key=lambda i: (len(upstream(i)), ids[i]["Position"]["X"]))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8-sig").read()
    doc = json.loads(text)
    terrain = doc["Assets"]["$values"][0]["Terrain"]
    nodes = terrain["Nodes"]
    if any(isinstance(v, dict) and kind(v) == "Export" for v in nodes.values() if isinstance(v, dict) and "$type" in v):
        open(dst, "w", encoding="utf-8").write(text)
        return
    src_id = final_height(nodes)
    nid = max(int(k) for k in nodes if k.isdigit()) + 1
    base = max(int(m) for m in re.findall(r'"\$id": "(\d+)"', text)) + 1
    pos = nodes[str(src_id)]["Position"]
    ref = str(base)
    nodes[str(nid)] = {
        "$id": ref, "$type": "QuadSpinner.Gaea.Nodes.Export, Gaea.Nodes", "Id": nid, "Name": "Export",
        "Position": {"$id": str(base + 1), "X": pos["X"] + 400.0, "Y": pos["Y"]},
        "Ports": {"$id": str(base + 2), "$values": [
            {"$id": str(base + 3), "Name": "In", "Type": "PrimaryIn, Required",
             "Record": {"$id": str(base + 4), "From": src_id, "To": nid, "FromPort": "Out", "ToPort": "In", "IsValid": True},
             "IsExporting": True, "Parent": {"$ref": ref}},
            {"$id": str(base + 5), "Name": "Out", "Type": "PrimaryOut", "IsExporting": True, "Parent": {"$ref": ref}}]},
        "Modifiers": {"$id": str(base + 6), "$values": []},
    }
    # The examples' build settings have no build type or destination; without them Build Swarm builds nothing.
    build = doc["Assets"]["$values"][0].setdefault("BuildDefinition", {"$id": str(base + 7)})
    build.setdefault("Type", "Standard")
    build.setdefault("Destination", "[Filename][Mutation]")
    build.setdefault("PostBuildScript", "")
    build.setdefault("CopyTerrain", True)
    json.dump(doc, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"{src}: Export on node {src_id} ({kind(nodes[str(src_id)])})")


if __name__ == "__main__":
    main()
