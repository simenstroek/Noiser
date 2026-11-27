"""Small conditional flow-matching model for terrain detail: samples one plausible 4x detail field instead of predicting the
average (an earlier L1/spectrum/GAN model did that, and looked like blur plus noise).

    NOISER_ML_CELL=1 python tools/ml/flow.py train --data E:/_NoiserML/lidar    --out E:/_NoiserML/runs/flow1_v3       --steps 16000  (4 m -> 1 m)
    NOISER_ML_CELL=4 python tools/ml/flow.py train --data E:/_NoiserML/lidar_4m --out E:/_NoiserML/runs/flow_stage1_v3 --steps 16000  (16 m -> 4 m)

Inputs and residual come from terrain_ml (smooth base, features, zero mean over every coarse cell). The network is a small
U-Net (3.0 M parameters) that predicts the velocity from noise to the residual; sampling is a few Euler steps from pure noise.
`enhance` is the whole-map reference implementation for a port.
"""

import argparse
import copy
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import terrain_ml as T

RES = 10.0  # the within-cell residual is small next to the terrain's relief; scaled up before the unit-variance scale


def prepare(batch, rng, device):
    """From sampled patches: model inputs and the real within-cell residual (in units of the patch's sigma)."""
    fine, acc, tid, cell, add = batch
    fine = torch.from_numpy(fine).to(device).unsqueeze(1)
    acc = torch.from_numpy(acc).to(device).unsqueeze(1)
    tid = torch.from_numpy(tid).long().to(device)
    cell = torch.from_numpy(cell).to(device)
    coarse = T.block_mean(fine)
    # The input is degraded (blur, smooth noise, noise-made shapes) so it is less like clean block means, as an artist's
    # terrain is not. The target below is still the real residual: the upsample is linear, so the added shape has none.
    x, sigma, _ = T.features(T.degrade(coarse + torch.from_numpy(add).to(device).unsqueeze(1), None), acc, cell)
    real = T.zero_block_mean((fine - T.upsample(coarse)) / sigma.view(-1, 1, 1, 1))
    return x, real, tid, sigma, cell


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(c).chunk(2, dim=1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(h))


class FlowNet(nn.Module):
    """U-Net over 4 levels (receptive field a few hundred cells), FiLM-conditioned on time, terrain type, relief and cell size."""

    def __init__(self, chs=(32, 64, 96, 128), cond=64):
        super().__init__()
        self.embed = nn.Embedding(len(T.TYPES), cond)
        self.mlp = nn.Sequential(nn.Linear(cond + 2 + 32, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inp = nn.Conv2d(4 + 1, chs[0], 3, padding=1)  # features + the noisy residual
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for ch in chs:
            self.down.append(nn.ModuleList([ResBlock(c, ch, cond), ResBlock(ch, ch, cond)]))
            c = ch
        self.mid = ResBlock(c, c, cond)
        for ch in reversed(chs):
            self.up.append(nn.ModuleList([ResBlock(c + ch, ch, cond), ResBlock(ch, ch, cond)]))
            c = ch
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, xt, feats, t, tid, sigma, cell_m):
        f = torch.arange(16, device=t.device, dtype=torch.float32)
        ang = t[:, None] * 1000.0 * torch.exp(-math.log(10000.0) * f / 16)[None]
        temb = torch.cat([ang.sin(), ang.cos()], dim=1)
        c = self.mlp(torch.cat([self.embed(tid), torch.log(sigma)[:, None] / 3.0, torch.log2(cell_m / T.CELL)[:, None], temb], dim=1))
        h = self.inp(torch.cat([feats, xt], dim=1))
        skips = []
        for i, (a, b) in enumerate(self.down):
            h = b(a(h, c), c)
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)
        h = self.mid(h, c)
        for i, (a, b) in enumerate(self.up):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = b(a(torch.cat([h, skips.pop()], dim=1), c), c)
        return self.out(h)


@torch.no_grad()
def sample(net, feats, tid, sigma, cell_m, scale, steps=8, gen=None):
    """Euler integration from noise (t=0) to the residual (t=1). Returns the residual in units of sigma, block means zero."""
    x = torch.randn(feats.shape[0], 1, *feats.shape[-2:], device=feats.device, generator=gen)
    for i in range(steps):
        t = torch.full((feats.shape[0],), i / steps, device=feats.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feats.is_cuda):
            v = net(x, feats, t, tid, sigma, cell_m).float()
        x = x + v / steps
    return T.zero_block_mean(x * scale / RES)


@torch.no_grad()
def enhance(coarse, kind, cell_m, net, scale, steps=8, seed=0, dev="cuda", tile=T.PATCH, overlap=64, x0=None):
    """Whole map: coarse [H, W] metres -> [4H, 4W]. Tiles of the training size share one noise field and their velocities are
    blended every step, so there are no seams and no averaging of unrelated samples. A map smaller than a tile is padded with
    its edge values to one tile and cropped after. `x0`: the starting noise [4H, 4W] (default: from `seed`). This is the
    reference the app's port (src/landscape_detail.rs) is tested against."""
    from prep_lidar import flow_accumulation

    H, W = coarse.shape
    ct = tile // T.FACTOR
    if H < ct or W < ct:
        assert x0 is None, "pass x0 only for maps of at least one tile"
        padded = np.pad(coarse, ((0, max(ct - H, 0)), (0, max(ct - W, 0))), mode="edge")
        return enhance(padded, kind, cell_m, net, scale, steps, seed, dev, tile, overlap)[: 4 * H, : 4 * W]
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    acc = np.log1p(flow_accumulation(coarse.astype(np.float64))).astype(np.float32)  # whole map: real watersheds
    c_all = torch.from_numpy(coarse.astype(np.float32))[None, None].to(dev)
    a_all = torch.from_numpy(acc)[None, None].to(dev)
    FH, FW = H * T.FACTOR, W * T.FACTOR

    def starts(n):
        s = list(range(0, n - tile + 1, tile - overlap))
        return s + [n - tile] if s[-1] != n - tile else s

    win1 = torch.hann_window(tile, periodic=False, device=dev).clamp(min=1e-3)
    win = (win1[:, None] * win1[None, :])[None, None]
    cell = torch.tensor([cell_m], device=dev)
    tid = torch.tensor([T.TYPES.index(kind)], device=dev)
    tiles = []
    for y in starts(FH):
        for x in starts(FW):
            cs = (slice(None), slice(None), slice(y // 4, (y + tile) // 4), slice(x // 4, (x + tile) // 4))
            feats, sigma, _ = T.features(c_all[cs], a_all[cs], cell)
            tiles.append((y, x, feats, sigma))
    wsum = torch.zeros(1, 1, FH, FW, device=dev)
    for y, x, _, _ in tiles:
        wsum[..., y : y + tile, x : x + tile] += win
    if x0 is None:
        xs = torch.randn(1, 1, FH, FW, device=dev, generator=torch.Generator(dev).manual_seed(seed))
    else:
        xs = torch.from_numpy(np.asarray(x0, np.float32))[None, None].to(dev)
    for i in range(steps):
        t = torch.full((1,), i / steps, device=dev)
        v = torch.zeros_like(xs)
        for y, x, feats, sigma in tiles:
            with torch.autocast(**amp):
                v[..., y : y + tile, x : x + tile] += net(xs[..., y : y + tile, x : x + tile], feats, t, tid, sigma, cell).float() * win
        xs = xs + v / wsum / steps
    res = torch.zeros_like(xs)
    for y, x, _, sigma in tiles:  # per-tile relief scale, blended
        res[..., y : y + tile, x : x + tile] += xs[..., y : y + tile, x : x + tile] * (scale / RES) * sigma * win
    # Zero mean over every coarse cell, once for the whole map: the output agrees exactly with the input.
    return (T.upsample(c_all) + T.zero_block_mean(res / wsum))[0, 0].cpu().numpy()


def load(path, dev="cuda"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = FlowNet().to(dev)
    net.load_state_dict(ck["ema"])
    return net.eval(), ck["scale"]


def train(args):
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    tiles = T.load_tiles(args.data, False)
    noise = T.load_noise(args.noise) if args.noise else None
    print("tiles:", {k: len(v) for k, v in tiles.items()}, "noise maps:", len(noise or []), flush=True)
    q = queue.Queue(maxsize=6)

    def producer(seed):
        rng = np.random.default_rng(seed)
        while True:
            q.put(T.sample_batch(tiles, args.batch, rng, noise=noise))

    for s in range(2):
        threading.Thread(target=producer, args=(s,), daemon=True).start()
    rng = np.random.default_rng(1234)
    # The residual (x RES) is scaled to unit variance so it matches the noise it is generated from.
    scale = float(torch.stack([prepare(q.get(), rng, dev)[1].std() * RES for _ in range(20)]).mean())
    print(f"residual std x RES: {scale:.3f}", flush=True)
    net = FlowNet().to(dev)
    ema = copy.deepcopy(net).eval()
    print("parameters:", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=0.0)
    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,loss,sec\n")
    t0, avg = time.time(), 0.0
    for step in range(1, args.steps + 1):
        feats, real, tid, sigma, cell = prepare(q.get(), rng, dev)
        x1 = real * RES / scale
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(x1.shape[0], device=dev))  # logit-normal: more weight on the middle of the path
        xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(xt, feats, t, tid, sigma, cell).float()
        loss = F.mse_loss(v, x1 - x0)
        if not torch.isfinite(loss):
            raise SystemExit(f"loss not finite at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - 0.999)
        avg += loss.item()
        if step % 250 == 0:
            line = f"{step},{avg / 250:.4f},{time.time() - t0:.0f}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            avg = 0.0
        if step % 2000 == 0 or step == args.steps:
            ck = {"net": net.state_dict(), "ema": ema.state_dict(), "scale": scale, "step": step}
            torch.save(ck, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data", default="E:/_NoiserML/lidar")
    ap.add_argument("--out", default="E:/_NoiserML/runs/flow1")
    ap.add_argument("--noise", default="", help="folder of noise heightmaps mixed into inputs (tried: less detail, off by default)")
    train(ap.parse_args())
