"""Writes a flow model for the app (src/landscape_detail.rs), and the test fixtures the port is checked against.

    python tools/ml/export_weights.py weights CKPT OUT.nmlw --cell 1     (stage 2: 4 m -> 1 m; --cell 4 for stage 1)
    python tools/ml/export_weights.py fixtures CKPT OUT.nmlw --cell 1    (small inputs and flow.enhance's answers, fp32 on the CPU)

File format: a little-endian u32 header length, the header (JSON: "meta" and "tensors", each tensor with its name, shape,
dtype f16 or f32, and byte offset into the data), then the data, which starts on a 4-byte boundary.
"""
import argparse
import json
import os
import struct

import numpy as np
import torch


def write(path, meta, tensors):
    """tensors: [(name, array, dtype)] with dtype 'f16' or 'f32'."""
    blobs, entries, off = [], [], 0
    for name, a, dt in tensors:
        b = np.ascontiguousarray(a, np.float16 if dt == "f16" else np.float32).tobytes()
        entries.append({"name": name, "shape": list(np.shape(a)), "dtype": dt, "offset": off})
        blobs.append(b + b"\0" * (-len(b) % 4))
        off += len(blobs[-1])
    header = json.dumps({"meta": meta, "tensors": entries}).encode()
    header += b" " * (-(4 + len(header)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(header)) + header + b"".join(blobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weights", "fixtures"])
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--cell", type=float, required=True, help="the cell size in metres the model was trained at (NOISER_ML_CELL)")
    a = ap.parse_args()
    os.environ["NOISER_ML_CELL"] = str(a.cell)
    import flow
    import terrain_ml as T

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = {"format": 1, "scale": ck["scale"], "res": flow.RES, "cell": a.cell, "factor": T.FACTOR, "tile": T.PATCH, "types": T.TYPES, "chs": list(ck.get("chs", (32, 64, 96, 128))),
            "mode": ck.get("mode", "4x"), "lam": bool(ck.get("lam", False))}
    if a.cmd == "weights":
        write(a.out, meta, [(k, v.float().numpy(), "f16") for k, v in ck["ema"].items()])
        print("wrote", a.out, os.path.getsize(a.out), "bytes")
        return
    net, scale = flow.load(a.ckpt, "cpu")
    # Weights as stored (f16), so the fixture tests the arithmetic, not the rounding of the weights.
    net.load_state_dict({k: v.half().float() for k, v in ck["ema"].items()})
    rng = np.random.default_rng(5)
    dem = T.load_tiles("E:/_NoiserML/lidar" if a.cell == 1 else "E:/_NoiserML/lidar_4m", True)["alpine"][0][0]
    fine = dem[400:528, 600:728].astype(np.float32)  # 128 x 128 cells of held-out terrain
    coarse = fine.reshape(32, 4, 32, 4).mean(axis=(1, 3)).astype(np.float32)
    from prep_lidar import flow_accumulation

    if meta["mode"] == "sharpen":
        # A softened 128 x 128 crop and flow.sharpen's answer (3 steps, guidance 2, four 64-cell tiles).
        soft = T._gauss(torch.from_numpy(fine)[None, None], 1.5)[0, 0].numpy()
        x0 = rng.standard_normal((128, 128)).astype(np.float32)
        sharp = flow.sharpen(soft, "alpine", a.cell, net, scale, steps=3, dev="cpu", tile=64, overlap=16, guidance=2.0, x0=x0)
        meta.update({"tile": 64, "overlap": 16, "steps": 3, "kind": "alpine", "guidance": 2.0})
        write(a.out, meta, [("soft", soft, "f32"), ("x0", x0, "f32"), ("sharp", sharp, "f32")])
        print("wrote", a.out, os.path.getsize(a.out), "bytes; change std", float((sharp - soft).std()))
        return
    acc = np.log1p(flow_accumulation(coarse.astype(np.float64))).astype(np.float32)
    c = torch.from_numpy(coarse)[None, None]
    up = T.upsample(c)[0, 0].numpy()
    cell = torch.tensor([a.cell])
    tile = 64
    feats, sigma, _ = T.features(c[..., : tile // 4, : tile // 4], torch.from_numpy(acc)[None, None, : tile // 4, : tile // 4], cell)
    x0 = rng.standard_normal((128, 128)).astype(np.float32)
    xt = torch.from_numpy(rng.standard_normal((1, 1, tile, tile)).astype(np.float32))
    tid = torch.tensor([T.TYPES.index("alpine")])
    with torch.no_grad():
        v = net(xt, feats, torch.tensor([0.375]), tid, sigma, cell)
    r = torch.from_numpy(rng.standard_normal((1, 1, 64, 64)).astype(np.float32))
    zbm = T.zero_block_mean(r)[0, 0].numpy()
    out = flow.enhance(coarse, "alpine", a.cell, net, scale, steps=3, dev="cpu", tile=tile, overlap=16, x0=x0)
    guided = flow.enhance(coarse, "alpine", a.cell, net, scale, steps=3, dev="cpu", tile=tile, overlap=16, x0=x0, guidance=2.0)
    meta.update({"tile": tile, "overlap": 16, "steps": 3, "kind": "alpine", "t": 0.375, "sigma": float(sigma[0]), "guidance": 2.0})
    extra = []
    if meta["lam"]:  # adherence weights: also the loosened answer (lam 0.5), where the coarse shape may move
        extra = [("loose", flow.enhance(coarse, "alpine", a.cell, net, scale, steps=3, dev="cpu", tile=tile, overlap=16, x0=x0, guidance=2.0, lam=0.5), "f32")]
        meta["lam_value"] = 0.5
    write(a.out, meta, [("coarse", coarse, "f32"), ("acc", acc, "f32"), ("up", up, "f32"), ("feats", feats[0].numpy(), "f32"),
                        ("xt", xt[0, 0].numpy(), "f32"), ("v", v[0, 0].numpy(), "f32"), ("r", r[0, 0].numpy(), "f32"),
                        ("zbm", zbm, "f32"), ("x0", x0, "f32"), ("out", out, "f32"), ("guided", guided, "f32")] + extra)
    print("wrote", a.out, os.path.getsize(a.out), "bytes; out range", float(out.min()), float(out.max()))


if __name__ == "__main__":
    main()
