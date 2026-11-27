//! Learned terrain detail: the Erosion node's third kind. A small flow-matching model (a U-Net, 3 M parameters per stage)
//! adds plausible detail of real terrain to a coarse heightmap, 4x per stage; two stages give 16x. It is the port of
//! `tools/ml/flow.py` (`enhance`), and is tested against that file's answers (`tests/fixtures/detail_fixture.nmlw`).
//!
//! The coarse input's mean over every coarse cell is kept exactly: only detail within a cell is generated. The inputs
//! (a smooth upsample, slope, curvature, flow accumulation) are made on the CPU; the network runs on the GPU in tiles of
//! 256 cells that share one noise field, their velocities blended at every step, so there are no seams. See
//! docs/ml-terrain-results.md.

use crate::export::Progress;
use crate::landscape_erosion::{DetailStage, Erosion, ErosionResult};
use eframe::wgpu;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The terrain types the model knows (the conditioning label), in the order it was trained with.
pub const TYPES: [&str; 11] = ["alpine", "desert", "badlands", "hills", "volcanic", "karst", "coastal", "craters", "fantasy", "rocky", "icelandic"];
const FACTOR: usize = 4;
const TILE: usize = 256;
const OVERLAP: usize = 64;
/// Weight files: stage 1 (16 m -> 4 m cells) and stage 2 (4 m -> 1 m).
const STAGE1: &str = "detail_16m_4m.nmlw";
const STAGE2: &str = "detail_4m_1m.nmlw";
const SHARPEN: &str = "detail_sharpen.nmlw";

// ---------------------------------------------------------------------------------------------------------------------
// Weight files (`tools/ml/export_weights.py`): a u32 header length, a JSON header, then f16/f32 data.

pub struct Tensors {
    pub meta: serde_json::Value,
    pub t: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn get(&self, name: &str) -> Result<&[f32], String> {
        self.t.get(name).map(|(_, v)| v.as_slice()).ok_or_else(|| format!("the weights have no tensor {name}"))
    }
    /// The U-Net's channels per level (`chs` in the weights; older files are the first model's 32, 64, 96, 128).
    fn chs(&self) -> [usize; 4] {
        let read = self.meta["chs"].as_array().and_then(|a| a.iter().map(|v| v.as_u64().map(|c| c as usize)).collect::<Option<Vec<_>>>());
        read.and_then(|v| v.try_into().ok()).unwrap_or([32, 64, 96, 128])
    }

    /// Weights of the sharpen stage (same grid in and out) rather than a 4x stage.
    fn lam(&self) -> bool {
        self.meta.get("lam").and_then(|m| m.as_bool()) == Some(true)
    }

    fn sharpen(&self) -> bool {
        self.meta.get("mode").and_then(|m| m.as_str()) == Some("sharpen")
    }

    fn meta_f32(&self, key: &str) -> Result<f32, String> {
        self.meta.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("the weights have no {key}"))
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s,
        0 => {
            // Subnormal: normalise the mantissa.
            let (mut m2, mut e2) = (m, 113u32);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
        31 => s | 0x7f80_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn read_tensors(bytes: &[u8]) -> Result<Tensors, String> {
    let bad = || "not a Noiser model file".to_string();
    let len = u32::from_le_bytes(bytes.get(..4).ok_or_else(bad)?.try_into().map_err(|_| bad())?) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(4..4 + len).ok_or_else(bad)?).map_err(|e| e.to_string())?;
    let data = &bytes[4 + len..];
    let mut t = HashMap::new();
    for e in header["tensors"].as_array().ok_or_else(bad)? {
        let name = e["name"].as_str().ok_or_else(bad)?.to_string();
        let shape: Vec<usize> = e["shape"].as_array().ok_or_else(bad)?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        let count: usize = shape.iter().product();
        let off = e["offset"].as_u64().ok_or_else(bad)? as usize;
        let values = match e["dtype"].as_str() {
            Some("f16") => data.get(off..off + count * 2).ok_or_else(bad)?.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect(),
            Some("f32") => data.get(off..off + count * 4).ok_or_else(bad)?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            _ => return Err(bad()),
        };
        t.insert(name, (shape, values));
    }
    Ok(Tensors { meta: header["meta"].clone(), t })
}

/// Where the weight files are: the folder in NOISER_DETAIL_WEIGHTS, else `ml` next to the program.
fn weights_path(name: &str) -> PathBuf {
    if let Some(dir) = std::env::var_os("NOISER_DETAIL_WEIGHTS") {
        return PathBuf::from(dir).join(name);
    }
    let exe = std::env::current_exe().unwrap_or_default();
    exe.parent().map_or_else(|| PathBuf::from("ml"), |d| d.join("ml")).join(name)
}

/// The weights, read once per file.
fn load_weights(name: &str) -> Result<Arc<Tensors>, String> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, Arc<Tensors>>>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(t) = cache.lock().map_err(|_| "weights cache unusable")?.get(name) {
        return Ok(t.clone());
    }
    let path = weights_path(name);
    let bytes = std::fs::read(&path).map_err(|e| format!("Detail needs its model files in the ml folder next to noiser.exe (the release zip has them): {} ({e})", path.display()))?;
    let t = Arc::new(read_tensors(&bytes)?);
    if t.meta["types"].as_array().is_none_or(|a| a.iter().map(|v| v.as_str()).ne(TYPES.iter().map(|s| Some(*s)))) {
        return Err(format!("{} is for another version of Noiser (its terrain types differ); use the ml folder from this release", path.display()));
    }
    cache.lock().map_err(|_| "weights cache unusable")?.insert(name.to_string(), t.clone());
    Ok(t)
}

// ---------------------------------------------------------------------------------------------------------------------
// The model's inputs, on the CPU. Grids are row-major [h][w]; each function matches its PyTorch counterpart in
// tools/ml/terrain_ml.py (the fixture test checks).

/// Gaussian blur with edge replication; the kernel has 2 round(3 sigma) + 1 taps (Python rounds halves to even).
pub(crate) fn gauss(x: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let k = 2 * (3.0 * sigma).round_ties_even() as usize + 1;
    let r = (k / 2) as isize;
    let mut g: Vec<f32> = (0..k).map(|i| (-((i as f32 - r as f32).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= sum);
    let mut tmp = vec![0.0f32; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (xo, out) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (i, gi) in g.iter().enumerate() {
                let xx = (xo as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                s += gi * x[y * w + xx];
            }
            *out = s;
        }
    });
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (xo, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (i, gi) in g.iter().enumerate() {
                let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
                s += gi * tmp[yy * w + xo];
            }
            *o = s;
        }
    });
    out
}

/// The 4 source indices and weights of PyTorch's bicubic (A = -0.75, align_corners false) for output index `o`.
fn cubic_taps(o: usize, n: usize) -> ([usize; 4], [f32; 4]) {
    const A: f32 = -0.75;
    let src = (o as f32 + 0.5) / FACTOR as f32 - 0.5;
    let i0 = src.floor();
    let t = src - i0;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let idx = |d: isize| (i0 as isize + d).clamp(0, n as isize - 1) as usize;
    ([idx(-1), idx(0), idx(1), idx(2)], [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)])
}

/// Bicubic 4x upsample [h][w] -> [4h][4w].
fn bicubic4(c: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (fh, fw) = (h * FACTOR, w * FACTOR);
    let xt: Vec<_> = (0..fw).map(|o| cubic_taps(o, w)).collect();
    let mut tmp = vec![0.0f32; h * fw];
    tmp.par_chunks_mut(fw).enumerate().for_each(|(y, row)| {
        for (xo, out) in row.iter_mut().enumerate() {
            let (i, wt) = xt[xo];
            *out = (0..4).map(|k| c[y * w + i[k]] * wt[k]).sum();
        }
    });
    let mut out = vec![0.0f32; fh * fw];
    out.par_chunks_mut(fw).enumerate().for_each(|(yo, row)| {
        let (i, wt) = cubic_taps(yo, h);
        for (xo, o) in row.iter_mut().enumerate() {
            *o = (0..4).map(|k| tmp[i[k] * fw + xo] * wt[k]).sum();
        }
    });
    out
}

/// Mean over each 4x4 block: [h][w] -> [h/4][w/4].
fn block_mean(x: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (ch, cw) = (h / FACTOR, w / FACTOR);
    let mut out = vec![0.0f32; ch * cw];
    out.par_chunks_mut(cw).enumerate().for_each(|(y, row)| {
        for (xo, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for dy in 0..FACTOR {
                for dx in 0..FACTOR {
                    s += x[(y * FACTOR + dy) * w + xo * FACTOR + dx];
                }
            }
            *o = s / (FACTOR * FACTOR) as f32;
        }
    });
    out
}

fn smooth_up(c: &[f32], h: usize, w: usize) -> Vec<f32> {
    gauss(&bicubic4(c, h, w), h * FACTOR, w * FACTOR, 1.5)
}

/// Grid-free 4x upsample of coarse heights whose block means match `c` (four back-projection passes).
pub fn upsample(c: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut u = smooth_up(c, h, w);
    for _ in 0..4 {
        let d: Vec<f32> = c.iter().zip(block_mean(&u, h * FACTOR, w * FACTOR)).map(|(a, b)| a - b).collect();
        u.iter_mut().zip(smooth_up(&d, h, w)).for_each(|(a, b)| *a += b);
    }
    u
}

/// Removes `r`'s mean over every 4x4 block: smoothly (16 passes), then the small rest exactly.
fn zero_block_mean(r: &mut [f32], h: usize, w: usize) {
    for _ in 0..16 {
        let s = smooth_up(&block_mean(r, h, w), h / FACTOR, w / FACTOR);
        r.iter_mut().zip(s).for_each(|(a, b)| *a -= b);
    }
    let m = block_mean(r, h, w);
    let cw = w / FACTOR;
    r.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v -= m[(y / FACTOR) * cw + x / FACTOR];
        }
    });
}

/// Priority-flood depression filling, then multiple-flow-direction accumulation (Freeman, exponent 1.1): the cells
/// draining through each cell. `prep_lidar.flow_accumulation`, including its tie order and f32 accumulator.
pub fn flow_accumulation(h: &[f64], n: usize, m: usize) -> Vec<f32> {
    use std::cmp::Reverse;
    #[derive(PartialEq)]
    struct Key(f64, usize, usize);
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Key {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0).then(self.1.cmp(&o.1)).then(self.2.cmp(&o.2))
        }
    }
    let mut filled = h.to_vec();
    let mut done = vec![false; n * m];
    let mut heap = std::collections::BinaryHeap::new();
    for i in 0..n {
        for j in 0..m {
            if i == 0 || j == 0 || i == n - 1 || j == m - 1 {
                heap.push(Reverse(Key(h[i * m + j], i, j)));
                done[i * m + j] = true;
            }
        }
    }
    let mut order = Vec::with_capacity(n * m);
    while let Some(Reverse(Key(z, i, j))) = heap.pop() {
        order.push(i * m + j);
        for di in -1isize..=1 {
            for dj in -1isize..=1 {
                let (a, b) = (i as isize + di, j as isize + dj);
                if (di == 0 && dj == 0) || a < 0 || b < 0 || a >= n as isize || b >= m as isize || done[a as usize * m + b as usize] {
                    continue;
                }
                let k = a as usize * m + b as usize;
                done[k] = true;
                // Nudge so flats drain in the order they were reached.
                filled[k] = h[k].max(z + 1e-4);
                heap.push(Reverse(Key(filled[k], a as usize, b as usize)));
            }
        }
    }
    let mut acc = vec![1.0f32; n * m];
    for &c in order.iter().rev() {
        let (i, j) = (c / m, c % m);
        let mut w = [0.0f64; 8];
        let mut total = 0.0;
        let mut k = 0;
        for di in -1isize..=1 {
            for dj in -1isize..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let (a, b) = (i as isize + di, j as isize + dj);
                if a >= 0 && b >= 0 && a < n as isize && b < m as isize {
                    // 1.4142, not SQRT_2: the same constant as the training data's (the fixture test compares).
                    #[allow(clippy::approx_constant)]
                    let diagonal = 1.4142;
                    let drop = (filled[c] - filled[a as usize * m + b as usize]) / if di != 0 && dj != 0 { diagonal } else { 1.0 };
                    if drop > 0.0 {
                        w[k] = drop.powf(1.1);
                        total += w[k];
                    }
                }
                k += 1;
            }
        }
        if total > 0.0 {
            let mut k = 0;
            for di in -1isize..=1 {
                for dj in -1isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    if w[k] > 0.0 {
                        let t = (i as isize + di) as usize * m + (j as isize + dj) as usize;
                        acc[t] = (acc[t] as f64 + acc[c] as f64 * w[k] / total) as f32;
                    }
                    k += 1;
                }
            }
        }
    }
    acc
}

/// The network's four input channels for one tile, from its coarse heights and log flow accumulation (both [h][w]), and
/// the tile's relief scale sigma in metres. `cell_m` is the output cell size.
pub fn features(c: &[f32], acc: &[f32], h: usize, w: usize, cell_m: f32) -> (Vec<f32>, f32) {
    features_up(&upsample(c, h, w), acc, h, w, cell_m)
}

/// features() from heights already on the fine grid ([4h][4w]; the sharpen stage's soft input), acc still [h][w].
fn features_up(up: &[f32], acc: &[f32], h: usize, w: usize, cell_m: f32) -> (Vec<f32>, f32) {
    let (fh, fw) = (h * FACTOR, w * FACTOR);
    let n = fh * fw;
    let mean = (up.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32;
    let var = up.iter().map(|&v| ((v - mean) as f64).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sigma = (var.sqrt() as f32).max(0.1 * cell_m);
    let mut x = vec![0.0f32; 4 * n];
    let at = |y: usize, xx: usize| up[y * fw + xx];
    let smooth = gauss(&up, fh, fw, 1.0);
    let sm = |y: isize, xx: isize| smooth[y.clamp(0, fh as isize - 1) as usize * fw + xx.clamp(0, fw as isize - 1) as usize];
    let a = smooth_up(acc, h, w);
    for y in 0..fh {
        for xx in 0..fw {
            let i = y * fw + xx;
            x[i] = (up[i] - mean) / sigma;
            // torch.gradient: central differences inside, one-sided at the edges.
            let gy = if y == 0 { at(1, xx) - at(0, xx) } else if y == fh - 1 { at(fh - 1, xx) - at(fh - 2, xx) } else { (at(y + 1, xx) - at(y - 1, xx)) / 2.0 };
            let gx = if xx == 0 { at(y, 1) - at(y, 0) } else if xx == fw - 1 { at(y, fw - 1) - at(y, fw - 2) } else { (at(y, xx + 1) - at(y, xx - 1)) / 2.0 };
            x[n + i] = ((gx * gx + gy * gy).sqrt() / cell_m).min(3.0);
            let (yi, xi) = (y as isize, xx as isize);
            let lap = (sm(yi - 1, xi) + sm(yi + 1, xi) + sm(yi, xi - 1) + sm(yi, xi + 1) - 4.0 * sm(yi, xi)) / cell_m;
            x[2 * n + i] = (lap * 10.0).clamp(-3.0, 3.0);
            x[3 * n + i] = (a[i] - 4.0) / 3.0;
        }
    }
    (x, sigma)
}

// ---------------------------------------------------------------------------------------------------------------------
// The network on the GPU.

const SHADER: &str = "
struct P {
    cin: u32, cout: u32, h: u32, w: u32,
    wo: u32, bo: u32, go: u32, beo: u32,
    fmul: u32, fadd: u32, flags: u32, dst_off: u32,
    src_off: u32, ty: u32, tx: u32, gw: u32,
    factor: f32, aux_off: u32, count: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> film: array<f32>;
@group(0) @binding(3) var<storage, read_write> src: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<storage, read_write> aux: array<f32>;

var<workgroup> red: array<f32, 256>;

fn flat(wid: vec3u, nw: vec3u, li: u32) -> u32 { return (wid.y * nw.x + wid.x) * 256u + li; }

fn reduce(li: u32) -> f32 {
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (li < k) { red[li] = red[li] + red[li + k]; }
        workgroupBarrier();
    }
    return red[0];
}

// GroupNorm statistics, two levels: 32 partial (count, mean, M2) per group, then combined (Chan et al.).
@compute @workgroup_size(256)
fn gn_partial(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let cpg = p.cin / 8u; let hw = p.h * p.w; let n = cpg * hw;
    let base = p.src_off + wid.y * n;
    let lo = n * wid.x / 32u; let hi = n * (wid.x + 1u) / 32u;
    var s = 0.0;
    for (var i = lo + li; i < hi; i = i + 256u) { s = s + src[base + i]; }
    red[li] = s; workgroupBarrier();
    let mean = reduce(li) / f32(max(hi - lo, 1u));
    workgroupBarrier();
    var q = 0.0;
    for (var i = lo + li; i < hi; i = i + 256u) { let d = src[base + i] - mean; q = q + d * d; }
    red[li] = q; workgroupBarrier();
    let m2 = reduce(li);
    if (li == 0u) { let k = 16u + (wid.y * 32u + wid.x) * 3u; aux[k] = f32(hi - lo); aux[k + 1u] = mean; aux[k + 2u] = m2; }
}

@compute @workgroup_size(8)
fn gn_final(@builtin(local_invocation_index) g: u32) {
    var n = 0.0; var mean = 0.0; var m2 = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        let k = 16u + (g * 32u + j) * 3u;
        let nb = aux[k]; if (nb == 0.0) { continue; }
        let d = aux[k + 1u] - mean; let nt = n + nb;
        mean = mean + d * nb / nt;
        m2 = m2 + aux[k + 2u] + d * d * n * nb / nt;
        n = nt;
    }
    aux[g * 2u] = mean;
    aux[g * 2u + 1u] = inverseSqrt(m2 / n + 1e-5);
}

// GroupNorm with its affine, optionally FiLM (flags & 2), then SiLU.
@compute @workgroup_size(256)
fn norm_act(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li); let hw = p.h * p.w;
    if (i >= p.cin * hw) { return; }
    let c = i / hw; let g = c / (p.cin / 8u);
    var v = (src[p.src_off + i] - aux[g * 2u]) * aux[g * 2u + 1u] * wt[p.go + c] + wt[p.beo + c];
    if ((p.flags & 2u) != 0u) { v = v * film[p.fmul + c] + film[p.fadd + c]; }
    dst[p.dst_off + i] = v / (1.0 + exp(-v));
}

// 3x3 convolution, zero padding. A workgroup makes 16 x 16 pixels by 16 output channels; input channels go through
// workgroup memory 8 at a time (an 18 x 18 patch each) with their weights. flags & 1: add aux (a residual).
const CI: u32 = 8u;
var<workgroup> s_in: array<f32, 2592>;  // CI * 18 * 18
var<workgroup> s_w: array<vec4f, 288>;  // [ci * 9 + tap][4]: the 16 output channels as 4 vec4s

fn bias4(c: u32) -> vec4f {
    var b = vec4f(0.0);
    for (var j = 0u; j < 4u; j = j + 1u) { if (c + j < p.cout) { b[j] = wt[p.bo + c + j]; } }
    return b;
}

@compute @workgroup_size(16, 16)
fn conv3(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let hw = p.h * p.w; let co0 = wid.z * 16u;
    let x0 = i32(wid.x * 16u) - 1; let y0 = i32(wid.y * 16u) - 1;
    let li = lid.y * 16u + lid.x;
    var a0 = bias4(co0); var a1 = bias4(co0 + 4u); var a2 = bias4(co0 + 8u); var a3 = bias4(co0 + 12u);
    for (var c0 = 0u; c0 < p.cin; c0 = c0 + CI) {
        for (var e = li; e < CI * 324u; e = e + 256u) {
            let ci = c0 + e / 324u; let r = e % 324u;
            let yy = y0 + i32(r / 18u); let xx = x0 + i32(r % 18u);
            var v = 0.0;
            if (ci < p.cin && yy >= 0 && yy < i32(p.h) && xx >= 0 && xx < i32(p.w)) { v = src[p.src_off + ci * hw + u32(yy) * p.w + u32(xx)]; }
            s_in[e] = v;
        }
        for (var e = li; e < CI * 9u * 4u; e = e + 256u) {
            let q = e % 4u; let rem = e / 4u; let ci = c0 + rem / 9u; let tap = rem % 9u;
            var v = vec4f(0.0);
            for (var j = 0u; j < 4u; j = j + 1u) {
                let co = co0 + q * 4u + j;
                if (co < p.cout && ci < p.cin) { v[j] = wt[p.wo + (co * p.cin + ci) * 9u + tap]; }
            }
            s_w[e] = v;
        }
        workgroupBarrier();
        for (var ci = 0u; ci < CI; ci = ci + 1u) {
            for (var ky = 0u; ky < 3u; ky = ky + 1u) {
                for (var kx = 0u; kx < 3u; kx = kx + 1u) {
                    let v = s_in[ci * 324u + (lid.y + ky) * 18u + lid.x + kx];
                    let b = (ci * 9u + ky * 3u + kx) * 4u;
                    a0 = a0 + v * s_w[b]; a1 = a1 + v * s_w[b + 1u]; a2 = a2 + v * s_w[b + 2u]; a3 = a3 + v * s_w[b + 3u];
                }
            }
        }
        workgroupBarrier();
    }
    let x = wid.x * 16u + lid.x; let y = wid.y * 16u + lid.y;
    if (x >= p.w || y >= p.h) { return; }
    let px = y * p.w + x;
    let add = (p.flags & 1u) != 0u;
    let m = mat4x4f(a0, a1, a2, a3);
    for (var k = 0u; k < 16u; k = k + 1u) {
        let co = co0 + k;
        if (co >= p.cout) { break; }
        var r = m[k / 4u][k % 4u];
        if (add) { r = r + aux[p.aux_off + co * hw + px]; }
        dst[p.dst_off + co * hw + px] = r;
    }
}

// 1x1 convolution (the residual path when the channel count changes).
@compute @workgroup_size(8, 8)
fn conv1(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) { return; }
    let hw = p.h * p.w; let co0 = id.z * 8u; let px = id.y * p.w + id.x;
    var acc: array<f32, 8>;
    for (var k = 0u; k < 8u; k = k + 1u) { acc[k] = select(0.0, wt[p.bo + co0 + k], co0 + k < p.cout); }
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        let v = src[p.src_off + ci * hw + px];
        for (var k = 0u; k < 8u; k = k + 1u) {
            if (co0 + k < p.cout) { acc[k] = acc[k] + v * wt[p.wo + (co0 + k) * p.cin + ci]; }
        }
    }
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (co0 + k < p.cout) { dst[p.dst_off + (co0 + k) * hw + px] = acc[k]; }
    }
}

// 2x2 average (h, w: the output size).
@compute @workgroup_size(256)
fn pool(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li); let hw = p.h * p.w;
    if (i >= p.cin * hw) { return; }
    let c = i / hw; let y = (i % hw) / p.w; let x = i % p.w;
    let b = p.src_off + c * hw * 4u + y * 2u * p.w * 2u + x * 2u;
    dst[p.dst_off + i] = 0.25 * (src[b] + src[b + 1u] + src[b + p.w * 2u] + src[b + p.w * 2u + 1u]);
}

// Nearest 2x upsample (h, w: the output size).
@compute @workgroup_size(256)
fn up2(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li); let hw = p.h * p.w;
    if (i >= p.cin * hw) { return; }
    let c = i / hw; let y = (i % hw) / p.w; let x = i % p.w;
    dst[p.dst_off + i] = src[p.src_off + c * hw / 4u + (y / 2u) * (p.w / 2u) + x / 2u];
}

// The tile's noisy residual from the whole map's (src: the map, gw wide; dst: the network input's channel at dst_off).
@compute @workgroup_size(256)
fn gather(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li);
    if (i >= p.h * p.w) { return; }
    dst[p.dst_off + i] = src[(p.ty + i / p.w) * p.gw + p.tx + i % p.w];
}

// The tile's velocity into the map's, windowed (aux: the 1D window) and weighted (guidance).
@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li);
    if (i >= p.h * p.w) { return; }
    let y = i / p.w; let x = i % p.w;
    let j = (p.ty + y) * p.gw + p.tx + x;
    dst[j] = dst[j] + src[i] * aux[y] * aux[x] * p.factor;
}

// One Euler step for the whole map (src: the blended velocity, cleared; dst: the residual; aux: the window sum).
@compute @workgroup_size(256)
fn euler(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nw, li);
    if (i >= p.count) { return; }
    dst[i] = dst[i] + src[i] / aux[i] * p.factor;
    src[i] = 0.0;
}
";

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    wo: u32,
    bo: u32,
    go: u32,
    beo: u32,
    fmul: u32,
    fadd: u32,
    flags: u32,
    dst_off: u32,
    src_off: u32,
    ty: u32,
    tx: u32,
    gw: u32,
    /// Euler: 1 / steps; scatter: the weight of the tile's velocity (guidance).
    factor: f32,
    aux_off: u32,
    count: u32,
    pad: u32,
}
const SLOT: u64 = 256; // uniform offset alignment

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Pipe {
    GnPartial,
    GnFinal,
    NormAct,
    Conv3,
    Conv1,
    Pool,
    Up2,
    Gather,
    Scatter,
    Euler,
}
const PIPES: [(Pipe, &str); 10] = [
    (Pipe::GnPartial, "gn_partial"),
    (Pipe::GnFinal, "gn_final"),
    (Pipe::NormAct, "norm_act"),
    (Pipe::Conv3, "conv3"),
    (Pipe::Conv1, "conv1"),
    (Pipe::Pool, "pool"),
    (Pipe::Up2, "up2"),
    (Pipe::Gather, "gather"),
    (Pipe::Scatter, "scatter"),
    (Pipe::Euler, "euler"),
];

/// A buffer of the plan, by index.
type Buf = usize;

enum Op {
    Run { pipe: Pipe, slot: u32, bind: usize, groups: [u32; 3] },
    Copy { src: Buf, src_off: u64, dst: Buf, dst_off: u64, size: u64 },
}

/// One residual block's shape and where its weights are.
struct Block {
    name: String,
    cin: usize,
    cout: usize,
}

fn blocks(chs: [usize; 4]) -> Vec<Block> {
    let mut out = Vec::new();
    let mut c = chs[0];
    for (l, &ch) in chs.iter().enumerate() {
        out.push(Block { name: format!("down.{l}.0"), cin: c, cout: ch });
        out.push(Block { name: format!("down.{l}.1"), cin: ch, cout: ch });
        c = ch;
    }
    out.push(Block { name: "mid".into(), cin: c, cout: c });
    for (i, &ch) in chs.iter().rev().enumerate() {
        out.push(Block { name: format!("up.{i}.0"), cin: c + ch, cout: ch });
        out.push(Block { name: format!("up.{i}.1"), cin: ch, cout: ch });
        c = ch;
    }
    out
}

/// A loaded model on a device, with its buffers for one tile size, and the operations of one network call.
struct Net<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    w: Arc<Tensors>,
    scale: f32,
    res: f32,
    cell: f32,
    tile: usize,
    pipes: HashMap<Pipe, wgpu::ComputePipeline>,
    bgl: wgpu::BindGroupLayout,
    weights: wgpu::Buffer,
    offsets: HashMap<String, u32>,
    film: wgpu::Buffer,
    film_offsets: Vec<(u32, u32)>,
    film_len: usize,
    bufs: Vec<wgpu::Buffer>,
    params: Vec<Params>,
    uniform: Option<wgpu::Buffer>,
    binds: Vec<(Buf, Buf, Buf)>,
    bind_groups: Vec<wgpu::BindGroup>,
    ops: Vec<Op>,
    /// The network's input (5 channels) and output (1 channel).
    input: Buf,
    output: Buf,
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);

impl<'a> Net<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, w: Arc<Tensors>, tile: usize) -> Result<Net<'a>, String> {
        if !tile.is_multiple_of(8) {
            return Err("the tile size must be a multiple of 8".into());
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("terrain detail"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain detail"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }, count: None },
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipes = PIPES.iter().map(|(k, entry)| (*k, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None }))).collect();
        // All tensors in one buffer.
        let mut all = Vec::new();
        let mut offsets = HashMap::new();
        let mut names: Vec<&String> = w.t.keys().collect();
        names.sort();
        for name in names {
            offsets.insert(name.clone(), all.len() as u32);
            all.extend_from_slice(&w.t[name].1);
        }
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("detail weights"), contents: bytemuck::cast_slice(&all), usage: wgpu::BufferUsages::STORAGE });
        let mut film_offsets = Vec::new();
        let mut film_len = 0usize;
        for b in blocks(w.chs()) {
            film_offsets.push((film_len as u32, (film_len + b.cout) as u32));
            film_len += 2 * b.cout;
        }
        let film = device.create_buffer(&wgpu::BufferDescriptor { label: Some("detail film"), size: (film_len * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (scale, res, cell) = (w.meta_f32("scale")?, w.meta_f32("res")?, w.meta_f32("cell")?);
        let mut net = Net { device, queue, w, scale, res, cell, tile, pipes, bgl, weights, offsets, film, film_offsets, film_len, bufs: Vec::new(), params: Vec::new(), uniform: None, binds: Vec::new(), bind_groups: Vec::new(), ops: Vec::new(), input: 0, output: 0 };
        net.plan()?;
        Ok(net)
    }

    fn off(&self, name: &str) -> Result<u32, String> {
        self.offsets.get(name).copied().ok_or_else(|| format!("the weights have no tensor {name}"))
    }

    fn buffer(&mut self, floats: usize) -> Buf {
        self.bufs.push(self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (floats.max(4) * 4) as u64, usage: STORAGE, mapped_at_creation: false }));
        self.bufs.len() - 1
    }

    /// Adds a dispatch; `bind` = (src, dst, aux).
    fn run(&mut self, pipe: Pipe, p: Params, bind: (Buf, Buf, Buf), groups: [u32; 3]) -> u32 {
        let slot = self.params.len() as u32;
        self.params.push(p);
        let b = match self.binds.iter().position(|x| *x == bind) {
            Some(b) => b,
            None => {
                self.binds.push(bind);
                self.binds.len() - 1
            }
        };
        self.ops.push(Op::Run { pipe, slot, bind: b, groups });
        slot
    }

    fn groups_1d(count: usize) -> [u32; 3] {
        let n = count.div_ceil(256) as u32;
        [n.min(65535), n.div_ceil(65535), 1]
    }

    #[allow(clippy::too_many_arguments)]
    fn conv3(&mut self, name: &str, cin: usize, cout: usize, s: usize, src: Buf, dst: Buf, add: Option<Buf>) -> Result<(), String> {
        let p = Params { cin: cin as u32, cout: cout as u32, h: s as u32, w: s as u32, wo: self.off(&format!("{name}.weight"))?, bo: self.off(&format!("{name}.bias"))?, flags: u32::from(add.is_some()), ..Default::default() };
        let g = (s as u32).div_ceil(16);
        self.run(Pipe::Conv3, p, (src, dst, add.unwrap_or(self.dummy())), [g, g, (cout as u32).div_ceil(16)]);
        Ok(())
    }

    fn dummy(&self) -> Buf {
        0
    }

    /// GroupNorm (+ FiLM of block `film`) + SiLU of `src` (c channels, s x s) into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn norm(&mut self, name: &str, c: usize, s: usize, src: Buf, dst: Buf, stats: Buf, film: Option<usize>) -> Result<(), String> {
        let base = Params { cin: c as u32, h: s as u32, w: s as u32, ..Default::default() };
        self.run(Pipe::GnPartial, base, (src, self.dummy(), stats), [32, 8, 1]);
        self.run(Pipe::GnFinal, base, (src, self.dummy(), stats), [1, 1, 1]);
        let (fmul, fadd) = film.map_or((0, 0), |k| self.film_offsets[k]);
        let p = Params { go: self.off(&format!("{name}.weight"))?, beo: self.off(&format!("{name}.bias"))?, fmul, fadd, flags: if film.is_some() { 2 } else { 0 }, ..base };
        self.run(Pipe::NormAct, p, (src, dst, stats), Self::groups_1d(c * s * s));
        Ok(())
    }

    /// The operations of one network call (see flow.FlowNet), with buffers for each level.
    fn plan(&mut self) -> Result<(), String> {
        let dummy = self.buffer(4);
        debug_assert_eq!(dummy, 0);
        let s0 = self.tile;
        let chs = self.w.chs();
        // A level's widest buffer is its concatenation: its own channels and the level below's.
        let maxc: Vec<usize> = (0..4).map(|l| chs[l] + chs[(l + 1).min(3)]).collect();
        let sizes: Vec<usize> = (0..4).map(|l| s0 >> l).collect();
        let mut lv = Vec::new(); // per level: a, b, tmp, h, k, skip, cat
        for l in 0..4 {
            let n = maxc[l] * sizes[l] * sizes[l];
            lv.push([self.buffer(n), self.buffer(n), self.buffer(n), self.buffer(n), self.buffer(n), self.buffer(n), self.buffer(n)]);
        }
        let stats = self.buffer(16 + 8 * 32 * 3);
        self.input = self.buffer(5 * s0 * s0);
        self.output = self.buffer(s0 * s0);
        let bl = blocks(chs);
        let mut k = 0usize;
        // A residual block: src -> dst at level l.
        let mut block = |net: &mut Net, l: usize, src: Buf, dst: Buf| -> Result<(), String> {
            let b = &bl[k];
            let (s, [_, _, tmp, h, kb, _, _]) = (sizes[l], lv[l]);
            net.norm(&format!("{}.n1", b.name), b.cin, s, src, tmp, stats, None)?;
            net.conv3(&format!("{}.c1", b.name), b.cin, b.cout, s, tmp, h, None)?;
            net.norm(&format!("{}.n2", b.name), b.cout, s, h, tmp, stats, Some(k))?;
            let residual = if b.cin != b.cout {
                let p = Params { cin: b.cin as u32, cout: b.cout as u32, h: s as u32, w: s as u32, wo: net.off(&format!("{}.skip.weight", b.name))?, bo: net.off(&format!("{}.skip.bias", b.name))?, ..Default::default() };
                let g = (s as u32).div_ceil(8);
                net.run(Pipe::Conv1, p, (src, kb, 0), [g, g, (b.cout as u32).div_ceil(8)]);
                kb
            } else {
                src
            };
            net.conv3(&format!("{}.c2", b.name), b.cout, b.cout, s, tmp, dst, Some(residual))?;
            k += 1;
            Ok(())
        };
        let [a0, ..] = lv[0];
        self.conv3("inp", 5, chs[0], s0, self.input, a0, None)?;
        for l in 0..4 {
            let [a, b, _, _, _, skip, _] = lv[l];
            block(self, l, a, b)?;
            block(self, l, b, skip)?;
            if l < 3 {
                let s = sizes[l + 1];
                let p = Params { cin: chs[l] as u32, h: s as u32, w: s as u32, ..Default::default() };
                self.run(Pipe::Pool, p, (skip, lv[l + 1][0], 0), Self::groups_1d(chs[l] * s * s));
            }
        }
        let [a3, _, _, _, _, skip3, _] = lv[3];
        block(self, 3, skip3, a3)?;
        // Up: the level's input is [h, skip] in its cat buffer.
        let mut cur = a3;
        let mut c = chs[3];
        for (i, l) in (0..4).rev().enumerate() {
            let [a, b, _, _, _, skip, cat] = lv[l];
            let s = sizes[l];
            let plane = (s * s * 4) as u64;
            if i == 0 {
                self.ops.push(Op::Copy { src: cur, src_off: 0, dst: cat, dst_off: 0, size: c as u64 * plane });
            } else {
                let p = Params { cin: c as u32, h: s as u32, w: s as u32, ..Default::default() };
                self.run(Pipe::Up2, p, (cur, cat, 0), Self::groups_1d(c * s * s));
            }
            self.ops.push(Op::Copy { src: skip, src_off: 0, dst: cat, dst_off: c as u64 * plane, size: chs[l] as u64 * plane });
            block(self, l, cat, b)?;
            block(self, l, b, a)?;
            cur = a;
            c = chs[l];
        }
        let [_, _, tmp0, ..] = lv[0];
        self.norm("out.0", chs[0], s0, cur, tmp0, stats, None)?;
        self.conv3("out.2", chs[0], 1, s0, tmp0, self.output, None)?;
        Ok(())
    }

    /// Uploads the parameters and makes the bind groups; `extra` are buffers made after `plan` (the sampler's).
    fn finish(&mut self) {
        let mut bytes = vec![0u8; self.params.len() * SLOT as usize];
        for (i, p) in self.params.iter().enumerate() {
            bytes[i * SLOT as usize..i * SLOT as usize + std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.uniform = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("detail params"), contents: &bytes, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST }));
        let uniform = self.uniform.as_ref().expect("made above");
        self.bind_groups = self
            .binds
            .iter()
            .map(|&(s, d, a)| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: uniform, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }) },
                        wgpu::BindGroupEntry { binding: 1, resource: self.weights.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.film.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.bufs[s].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.bufs[d].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.bufs[a].as_entire_binding() },
                    ],
                })
            })
            .collect();
    }

    fn encode(&self, enc: &mut wgpu::CommandEncoder, ops: &[Op]) {
        let mut i = 0;
        while i < ops.len() {
            match &ops[i] {
                Op::Copy { src, src_off, dst, dst_off, size } => {
                    enc.copy_buffer_to_buffer(&self.bufs[*src], *src_off, &self.bufs[*dst], *dst_off, *size);
                    i += 1;
                }
                Op::Run { .. } => {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    while let Some(Op::Run { pipe, slot, bind, groups }) = ops.get(i) {
                        pass.set_pipeline(&self.pipes[pipe]);
                        pass.set_bind_group(0, &self.bind_groups[*bind], &[*slot * SLOT as u32]);
                        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                        i += 1;
                    }
                }
            }
        }
    }

    /// The conditioning vector (flow.FlowNet.mlp) and every block's FiLM multiplier and offset, for one tile and time.
    /// `tid` None: the mean of the type embeddings, an average terrain (what guidance steers away from).
    fn film_values(&self, tid: Option<usize>, sigma: f32, cell_m: f32, t: f32, lam: f32) -> Result<Vec<f32>, String> {
        let linear = |name: &str, x: &[f32]| -> Result<Vec<f32>, String> {
            let (wt, b) = (self.w.get(&format!("{name}.weight"))?, self.w.get(&format!("{name}.bias"))?);
            Ok((0..b.len()).map(|o| b[o] + x.iter().enumerate().map(|(i, v)| v * wt[o * x.len() + i]).sum::<f32>()).collect())
        };
        let embed = self.w.get("embed.weight")?;
        let cond = embed.len() / TYPES.len();
        let mut x: Vec<f32> = match tid {
            Some(tid) => embed[tid * cond..(tid + 1) * cond].to_vec(),
            None => (0..cond).map(|j| (0..TYPES.len()).map(|k| embed[k * cond + j]).sum::<f32>() / TYPES.len() as f32).collect(),
        };
        x.push(sigma.ln() / 3.0);
        x.push((cell_m / self.cell).log2());
        let ang: Vec<f32> = (0..16).map(|j| t * 1000.0 * (-(10000.0f32).ln() * j as f32 / 16.0).exp()).collect();
        x.extend(ang.iter().map(|a| a.sin()));
        x.extend(ang.iter().map(|a| a.cos()));
        if self.w.lam() {
            x.push(lam);
        }
        let h: Vec<f32> = linear("mlp.0", &x)?.into_iter().map(|v| v / (1.0 + (-v).exp())).collect();
        let c = linear("mlp.2", &h)?;
        let mut out = vec![0.0f32; self.film_len];
        for (k, b) in blocks(self.w.chs()).iter().enumerate() {
            let f = linear(&format!("{}.film", b.name), &c)?;
            let (mul, add) = self.film_offsets[k];
            for j in 0..b.cout {
                out[mul as usize + j] = 1.0 + f[j];
                out[add as usize + j] = f[b.cout + j];
            }
        }
        Ok(out)
    }

    fn read(&self, buf: Buf, floats: usize) -> Result<Vec<f32>, String> {
        let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (floats * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.bufs[buf], 0, &read, 0, (floats * 4) as u64);
        self.queue.submit([enc.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |r| r.expect("map detail result"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
        let view = read.get_mapped_range(..).map_err(|e| format!("{e:?}"))?;
        Ok(bytemuck::cast_slice(&view).to_vec())
    }
}

/// Deterministic standard normal noise for cell `i` of a map (a PCG hash and Box-Muller).
fn noise(seed: u32, i: u32) -> f32 {
    let mix = |v: u32| {
        let v = v.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((v >> ((v >> 28) + 4)) ^ v).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let a = mix(i.wrapping_mul(2654435761) ^ mix(seed));
    let b = mix(a ^ 0x9E37_79B9);
    let u1 = (a as f64 + 0.5) / 4294967296.0;
    let u2 = (b as f64 + 0.5) / 4294967296.0;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// One 4x stage over a whole map: `coarse` [h][w] metres -> [4h][4w] (flow.enhance). With weights trained as a sharpen
/// stage (meta `mode`), the same grid instead: a soft [h][w] (multiples of 4) gets its finest grain (flow.sharpen).
/// `lam` (weights trained for adherence): how loosely the input is followed, 0 (its coarse shape kept exactly) to 1.
/// `x0`: the starting noise (default from `seed`). `report(fraction)` is called as it goes.
#[allow(clippy::too_many_arguments)]
fn enhance(device: &wgpu::Device, queue: &wgpu::Queue, w: Arc<Tensors>, coarse: &[f32], h: usize, wd: usize, tid: usize, cell_m: f32, steps: u32, guidance: f32, lam: f32, seed: u32, x0: Option<&[f32]>, tile: usize, overlap: usize, report: &dyn Fn(f32)) -> Result<Vec<f32>, String> {
    let lam = if w.lam() { lam.clamp(0.0, 1.0) } else { 0.0 };
    let sharpen = w.sharpen();
    let up = if sharpen { 1 } else { FACTOR };
    let ct = tile / FACTOR * if sharpen { FACTOR } else { 1 };
    if h < ct || wd < ct {
        // Smaller than a tile: pad with the edge values to one tile, and crop.
        let (ph, pw) = (h.max(ct), wd.max(ct));
        let padded: Vec<f32> = (0..ph * pw).map(|i| coarse[(i / pw).min(h - 1) * wd + (i % pw).min(wd - 1)]).collect();
        let out = enhance(device, queue, w, &padded, ph, pw, tid, cell_m, steps, guidance, lam, seed, None, tile, overlap, report)?;
        let fw = pw * up;
        return Ok((0..h * up * wd * up).map(|i| out[(i / (wd * up)) * fw + i % (wd * up)]).collect());
    }
    let (fh, fw) = (h * up, wd * up);
    // Drainage on the coarse grid: the input's own, or (sharpen) its 4x block means'.
    let (acc_src, aw) = if sharpen { (block_mean(coarse, h, wd), wd / FACTOR) } else { (coarse.to_vec(), wd) };
    let acc: Vec<f32> = flow_accumulation(&acc_src.iter().map(|&v| v as f64).collect::<Vec<_>>(), acc_src.len() / aw, aw).iter().map(|v| v.ln_1p()).collect();
    let starts = |n: usize| {
        let mut s: Vec<usize> = (0..=n - tile).step_by(tile - overlap).collect();
        if *s.last().expect("n >= tile") != n - tile {
            s.push(n - tile);
        }
        s
    };
    let win1: Vec<f32> = (0..tile).map(|i| (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (tile - 1) as f32).cos()).max(1e-3)).collect();
    let mut net = Net::new(device, queue, w.clone(), tile)?;
    // Per tile: its features on the GPU and its sigma.
    let mut tiles = Vec::new();
    let (cs, ct2) = (tile / FACTOR, (tile / FACTOR) * (tile / FACTOR));
    for &y in &starts(fh) {
        for &x in &starts(fw) {
            let aw = if sharpen { wd / FACTOR } else { wd };
            let crop = |m: &[f32]| -> Vec<f32> { (0..ct2).map(|i| m[(y / FACTOR + i / cs) * aw + x / FACTOR + i % cs]).collect() };
            let (feats, sigma) = if sharpen {
                let fine: Vec<f32> = (0..tile * tile).map(|i| coarse[(y + i / tile) * wd + x + i % tile]).collect();
                features_up(&fine, &crop(&acc), cs, cs, cell_m)
            } else {
                features(&crop(coarse), &crop(&acc), cs, cs, cell_m)
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&feats), usage: STORAGE });
            net.bufs.push(buf);
            tiles.push((y, x, net.bufs.len() - 1, sigma));
        }
    }
    let mut wsum = vec![0.0f32; fh * fw];
    for &(y, x, _, _) in &tiles {
        for i in 0..tile {
            for j in 0..tile {
                wsum[(y + i) * fw + x + j] += win1[i] * win1[j];
            }
        }
    }
    let x0: Vec<f32> = match x0 {
        Some(x) => x.to_vec(),
        None => (0..(fh * fw) as u32).into_par_iter().map(|i| noise(seed, i)).collect(),
    };
    let mk = |net: &mut Net, data: &[f32]| {
        net.bufs.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: STORAGE }));
        net.bufs.len() - 1
    };
    let (xs, v, ws, win) = (mk(&mut net, &x0), mk(&mut net, &vec![0.0; fh * fw]), mk(&mut net, &wsum), mk(&mut net, &win1));
    // The sampler's own dispatches: per tile, gather and scatter (their tile origin is rewritten per tile).
    let n_ops = net.ops.len();
    let plane = (tile * tile) as u64 * 4;
    let tp = Params { h: tile as u32, w: tile as u32, gw: fw as u32, dst_off: (4 * tile * tile) as u32, ..Default::default() };
    let gather_slot = net.run(Pipe::Gather, tp, (xs, net.input, 0), Net::groups_1d(tile * tile));
    let scatter_slot = net.run(Pipe::Scatter, Params { dst_off: 0, factor: 1.0, ..tp }, (net.output, v, win), Net::groups_1d(tile * tile));
    let euler = Params { count: (fh * fw) as u32, factor: 1.0 / steps as f32, ..Default::default() };
    net.run(Pipe::Euler, euler, (v, xs, ws), Net::groups_1d(fh * fw));
    let sampler: Vec<Op> = net.ops.drain(n_ops..).collect();
    net.finish();
    let uniform = net.uniform.as_ref().expect("finished");
    let total = steps as usize * tiles.len();
    for step in 0..steps {
        let t = step as f32 / steps as f32;
        for (k, &(y, x, feats, sigma)) in tiles.iter().enumerate() {
            // Guidance: v_mean + w (v_type - v_mean) = w v_type + (1 - w) v_mean, two weighted passes into the velocity.
            let passes: &[(Option<usize>, f32)] = if guidance == 1.0 { &[(Some(tid), 1.0)] } else { &[(Some(tid), guidance), (None, 1.0 - guidance)] };
            for &(which, weight) in passes {
                queue.write_buffer(&net.film, 0, bytemuck::cast_slice(&net.film_values(which, sigma, cell_m, t, lam)?));
                for slot in [gather_slot, scatter_slot] {
                    let mut p = Params { ty: y as u32, tx: x as u32, ..net.params[slot as usize] };
                    if slot == scatter_slot {
                        p.factor = weight;
                    }
                    queue.write_buffer(uniform, slot as u64 * SLOT, bytemuck::bytes_of(&p));
                }
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&net.bufs[feats], 0, &net.bufs[net.input], 0, 4 * plane);
                net.encode(&mut enc, &sampler[..1]);
                net.encode(&mut enc, &net.ops);
                net.encode(&mut enc, &sampler[1..2]);
                queue.submit([enc.finish()]);
            }
            if k % 8 == 7 {
                device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("{e:?}"))?;
                report((step as usize * tiles.len() + k + 1) as f32 / total as f32);
            }
        }
        let mut enc = device.create_command_encoder(&Default::default());
        net.encode(&mut enc, &sampler[2..]);
        queue.submit([enc.finish()]);
    }
    let xr = net.read(xs, fh * fw)?;
    report(1.0);
    // The residual in metres: each tile's relief scale, blended; then exactly zero over every coarse cell.
    let mut res = vec![0.0f32; fh * fw];
    // flow.gain: the detail's size, and with adherence loosened, the input's noise the model also corrects.
    let k = ((net.scale / net.res).powi(2) + lam * lam).sqrt();
    for &(y, x, _, sigma) in &tiles {
        for i in 0..tile {
            for j in 0..tile {
                let g = (y + i) * fw + x + j;
                res[g] += xr[g] * k * sigma * win1[i] * win1[j];
            }
        }
    }
    res.iter_mut().zip(&wsum).for_each(|(r, s)| *r /= s);
    if sharpen {
        // Only the fine band (flow.high_pass), so the larger shapes stay the input's.
        let low = gauss(&res, fh, fw, 4.0);
        return Ok(coarse.iter().zip(res).zip(low).map(|((a, r), l)| a + r - l).collect());
    }
    if lam == 0.0 {
        zero_block_mean(&mut res, fh, fw);
    }
    Ok(upsample(coarse, h, wd).into_iter().zip(res).map(|(a, b)| a + b).collect())
}

/// The Erosion node's Detail kind: `heights` (`n` x `n` metres, cells `cell_m` apart) averaged to `scale` (4 or 16) times
/// coarser cells, then given learned detail back to `n` x `n`; `amount` blends from the input (0) to the result (1).
/// The flow output is the input's flow accumulation (on the coarsest grid, smoothly upsampled).
#[allow(clippy::too_many_arguments)]
pub fn detail(device: &wgpu::Device, queue: &wgpu::Queue, heights: &[f32], n: usize, cell_m: f32, e: &Erosion, seed: u32, progress: &Progress) -> Result<ErosionResult, String> {
    let tid = TYPES.iter().position(|t| *t == e.detail_type).ok_or_else(|| format!("unknown terrain type {}", e.detail_type))?;
    let [s1, s2, s3] = e.detail_stages;
    // Both stages: 16x coarser in, landforms then fine detail; one stage alone: 4x, with that stage's model.
    let stages: Vec<(&str, DetailStage, f32)> = match (s1.on, s2.on) {
        (true, true) => vec![(STAGE1, s1, cell_m * 4.0), (STAGE2, s2, cell_m)],
        (true, false) => vec![(STAGE1, s1, cell_m)],
        (false, true) => vec![(STAGE2, s2, cell_m)],
        (false, false) => vec![],
    };
    let scale = 4usize.pow(stages.len() as u32);
    let total = stages.len() + s3.on as usize;
    if total == 0 {
        return Ok(ErosionResult { heights: heights.to_vec(), flow: vec![0.0; heights.len()] });
    }
    if heights.len() != n * n || !n.is_multiple_of(scale) {
        return Err(format!("the working grid ({n}) must be a multiple of {scale}"));
    }
    let report = |stage: usize, stages: usize| move |f: f32| progress.store((((stage as f32 + f) / stages as f32) * 1000.0) as u32, Ordering::Relaxed);
    let down = |x: &[f32], side: usize, by: usize| -> Vec<f32> {
        let s = side / by;
        (0..s * s).map(|i| { let (y, xx) = (i / s, i % s); (0..by * by).map(|k| x[(y * by + k / by) * side + xx * by + k % by]).sum::<f32>() / (by * by) as f32 }).collect()
    };
    let coarse = down(heights, n, scale);
    let c = n / scale;
    let (mut cur, mut side) = (coarse.clone(), c);
    for (i, (file, st, cell)) in stages.iter().enumerate() {
        let out = enhance(device, queue, load_weights(file)?, &cur, side, side, tid, *cell, st.steps.clamp(1, 256), st.guidance.clamp(0.0, 8.0), 1.0 - st.adherence, seed.wrapping_add(100 * i as u32), None, TILE, OVERLAP, &report(i, total))?;
        // amount scales what this stage added to the smooth upsample of its input.
        let base = upsample(&cur, side, side);
        cur = base.iter().zip(&out).map(|(a, b)| a + (b - a) * st.amount).collect();
        side *= FACTOR;
    }
    if s3.on {
        if !n.is_multiple_of(FACTOR) {
            return Err(format!("the working grid ({n}) must be a multiple of 4"));
        }
        let out = enhance(device, queue, load_weights(SHARPEN)?, &cur, n, n, tid, cell_m, s3.steps.clamp(1, 256), s3.guidance.clamp(0.0, 8.0), 0.0, seed.wrapping_add(300), None, TILE, OVERLAP, &report(stages.len(), total))?;
        cur = cur.iter().zip(&out).map(|(a, b)| a + (b - a) * s3.amount).collect();
    }
    let (heights_out, flow_side, flow_src) = (cur, c, coarse);
    // Flow: cells draining through each coarse cell, carried up to the grid (on a log scale, smoothly).
    let mut acc: Vec<f32> = flow_accumulation(&flow_src.iter().map(|&v| v as f64).collect::<Vec<_>>(), flow_side, flow_side).iter().map(|v| v.ln_1p()).collect();
    let mut side = flow_side;
    while side < n {
        acc = smooth_up(&acc, side, side);
        side *= FACTOR;
    }
    let flow = acc.iter().map(|v| v.exp_m1() * 0.01).collect();
    Ok(ErosionResult { heights: heights_out, flow })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_tests::gpu;

    fn fixture() -> Tensors {
        read_tensors(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/detail_fixture.nmlw")).unwrap()).unwrap()
    }

    #[test]
    fn learned_detail_stages_switch_off_and_scale_their_detail() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let n = 64;
        let heights: Vec<f32> = (0..n * n).map(|i| ((i % n) as f32 * 0.3).sin() * 20.0 + (i / n) as f32 * 0.5).collect();
        let off = DetailStage { on: false, ..Default::default() };
        let stages = |a: DetailStage, b: DetailStage| Erosion { kind: crate::landscape_erosion::ErosionKind::Detail, detail_stages: [a, b, off], ..Default::default() };
        let run = |e: &Erosion| detail(&gpu.device, &gpu.queue, &heights, n, 1.0, e, 7, &Progress::default()).unwrap().heights;
        assert_eq!(run(&stages(off, off)), heights, "both stages off keep the input");
        if load_weights(STAGE2).is_err() {
            return eprintln!("no detail weights (NOISER_DETAIL_WEIGHTS); skipping the rest");
        }
        // The fine stage alone at amount 0 is the smooth upsample of the 4x block means; at 1 it adds detail.
        let c = n / 4;
        let coarse: Vec<f32> = (0..c * c).map(|i| (0..16).map(|k| heights[((i / c) * 4 + k / 4) * n + (i % c) * 4 + k % 4]).sum::<f32>() / 16.0).collect();
        let fine = DetailStage { steps: 2, ..Default::default() };
        close("amount 0", &run(&stages(off, DetailStage { amount: 0.0, ..fine })), &upsample(&coarse, c, c), 1e-5);
        let full = run(&stages(off, fine));
        let added = full.iter().zip(upsample(&coarse, c, c)).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(added > 0.01, "amount 1 adds detail ({added})");
        if load_weights(SHARPEN).is_ok() {
            // Sharpen alone works on the grid itself: grain added, the overall shape kept.
            let e = Erosion { detail_stages: [off, off, DetailStage { steps: 2, ..Default::default() }], ..stages(off, off) };
            let sharp = run(&e);
            let change: Vec<f32> = sharp.iter().zip(&heights).map(|(a, b)| a - b).collect();
            let worst = change.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let mean = change.iter().sum::<f32>() / change.len() as f32;
            assert!(worst > 1e-3 && mean.abs() < 0.05, "sharpen: largest change {worst} m, mean change {mean} m");
        }
    }

    fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt().max(1e-6);
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(worst <= tol * scale, "{name}: worst difference {worst} (rms of the answer {scale}, tolerance {})", tol * scale);
    }

    #[test]
    fn f16_conversion() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x0001, 5.960_464_5e-8), (0x7bff, 65504.0), (0x3555, 0.333_251_95)] {
            assert_eq!(f16_to_f32(h), f, "{h:#x}");
        }
    }

    #[test]
    fn inputs_match_the_python_reference() {
        let f = fixture();
        let coarse = f.get("coarse").unwrap();
        let acc: Vec<f32> = flow_accumulation(&coarse.iter().map(|&v| v as f64).collect::<Vec<_>>(), 32, 32).iter().map(|v| v.ln_1p()).collect();
        close("flow accumulation", &acc, f.get("acc").unwrap(), 1e-5);
        let up = upsample(coarse, 32, 32);
        let mean = up.iter().sum::<f32>() / up.len() as f32;
        let want_up = f.get("up").unwrap();
        // Heights are around 3000 m: compare the shape, not the offset's rounding.
        close("upsample", &up.iter().map(|v| v - mean).collect::<Vec<_>>(), &want_up.iter().map(|v| v - mean).collect::<Vec<_>>(), 1e-4);
        let crop = |m: &[f32]| -> Vec<f32> { (0..16 * 16).map(|i| m[(i / 16) * 32 + i % 16]).collect() };
        let (feats, sigma) = features(&crop(coarse), &crop(&acc), 16, 16, 1.0);
        assert!((sigma - f.meta_f32("sigma").unwrap()).abs() < 1e-3 * sigma, "sigma {sigma}");
        let want = f.get("feats").unwrap();
        // Curvature is ten times a Laplacian of heights near 3000 m: f32 rounding of the heights shows at about 1%.
        for (c, name, tol) in [(0, "height", 2e-3), (1, "slope", 2e-3), (2, "curvature", 2e-2), (3, "flow", 2e-3)] {
            close(name, &feats[c * 4096..(c + 1) * 4096], &want[c * 4096..(c + 1) * 4096], tol);
        }
        let mut r = f.get("r").unwrap().to_vec();
        zero_block_mean(&mut r, 64, 64);
        close("zero block mean", &r, f.get("zbm").unwrap(), 1e-4);
    }

    #[test]
    fn network_and_sampler_match_the_python_reference() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let Ok(w) = load_weights(STAGE2) else { return eprintln!("no detail weights (NOISER_DETAIL_WEIGHTS); skipping") };
        let f = fixture();
        let tile = f.meta["tile"].as_u64().unwrap() as usize;
        let mut net = Net::new(&gpu.device, &gpu.queue, w.clone(), tile).unwrap();
        net.finish();
        // One call: the fixture's features and noisy residual.
        let mut input = f.get("feats").unwrap().to_vec();
        input.extend_from_slice(f.get("xt").unwrap());
        gpu.queue.write_buffer(&net.bufs[net.input], 0, bytemuck::cast_slice(&input));
        let film = net.film_values(Some(0), f.meta_f32("sigma").unwrap(), 1.0, f.meta_f32("t").unwrap(), 0.0).unwrap();
        gpu.queue.write_buffer(&net.film, 0, bytemuck::cast_slice(&film));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        net.encode(&mut enc, &net.ops);
        gpu.queue.submit([enc.finish()]);
        let v = net.read(net.output, tile * tile).unwrap();
        close("network", &v, f.get("v").unwrap(), 2e-3);
        // The whole sampler: 3 steps, tiles of 64 overlapping by 16, on a 128 x 128 output.
        let coarse = f.get("coarse").unwrap();
        let out = enhance(&gpu.device, &gpu.queue, w.clone(), coarse, 32, 32, 0, 1.0, 3, 1.0, 0.0, 0, Some(f.get("x0").unwrap()), tile, 16, &|_| ()).unwrap();
        let up = upsample(coarse, 32, 32);
        let detail = |o: &[f32]| o.iter().zip(&up).map(|(a, b)| a - b).collect::<Vec<_>>();
        close("enhance", &detail(&out), &detail(f.get("out").unwrap()), 5e-3);
        let guided = enhance(&gpu.device, &gpu.queue, w.clone(), coarse, 32, 32, 0, 1.0, 3, f.meta_f32("guidance").unwrap(), 0.0, 0, Some(f.get("x0").unwrap()), tile, 16, &|_| ()).unwrap();
        close("guided", &detail(&guided), &detail(f.get("guided").unwrap()), 5e-3);
        // The added detail averages to zero over every coarse cell (the base's own agreement is upsample's).
        let worst = block_mean(&detail(&out), 128, 128).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(worst < 1e-3, "detail's mean over a coarse cell {worst} m");
    }

    #[test]
    fn adherence_matches_the_python_reference() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let Ok(w) = load_weights(STAGE1) else { return eprintln!("no detail weights (NOISER_DETAIL_WEIGHTS); skipping") };
        if !w.lam() {
            return eprintln!("stage-1 weights without adherence; skipping");
        }
        let f = read_tensors(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/adhere_fixture.nmlw")).unwrap()).unwrap();
        let (coarse, x0, g, cell) = (f.get("coarse").unwrap(), f.get("x0").unwrap(), f.meta_f32("guidance").unwrap(), f.meta_f32("cell").unwrap());
        let up = upsample(coarse, 32, 32);
        let detail = |o: &[f32]| o.iter().zip(&up).map(|(a, b)| a - b).collect::<Vec<_>>();
        let run = |lam: f32| enhance(&gpu.device, &gpu.queue, w.clone(), coarse, 32, 32, 0, cell, 3, g, lam, 0, Some(x0), 64, 16, &|_| ()).unwrap();
        close("strict", &detail(&run(0.0)), &detail(f.get("guided").unwrap()), 5e-3);
        // Loosened: the coarse shape moves too (no zero block mean), and the residual is scaled by flow.gain.
        let lam = f.meta_f32("lam_value").unwrap();
        close("loose", &detail(&run(lam)), &detail(f.get("loose").unwrap()), 5e-3);
    }

    #[test]
    fn sharpen_matches_the_python_reference() {
        let Some(gpu) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let Ok(w) = load_weights(SHARPEN) else { return eprintln!("no sharpen weights (NOISER_DETAIL_WEIGHTS); skipping") };
        let f = read_tensors(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sharpen_fixture.nmlw")).unwrap()).unwrap();
        let soft = f.get("soft").unwrap();
        let out = enhance(&gpu.device, &gpu.queue, w, soft, 128, 128, 0, 1.0, 3, f.meta_f32("guidance").unwrap(), 0.0, 0, Some(f.get("x0").unwrap()), 64, 16, &|_| ()).unwrap();
        let added = |o: &[f32]| o.iter().zip(soft).map(|(a, b)| a - b).collect::<Vec<_>>();
        let (a, b) = (added(&out), added(f.get("sharp").unwrap()));
        // The input is already smooth at 1 m, so its curvature is mostly f32 rounding of heights near 3000 m (differently
        // summed on the CPU and GPU): the answers agree to under 1% RMS, with isolated cells up to about 4%.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let diff: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        assert!(rms(&diff) < 0.01 * rms(&b), "sharpen: RMS difference {} against {}", rms(&diff), rms(&b));
        close("sharpen", &a, &b, 6e-2);
    }
}
