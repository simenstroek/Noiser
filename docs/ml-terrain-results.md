# Learned terrain detail

Survey of existing work: [ml-terrain-research.md](ml-terrain-research.md). Code: `tools/ml/`. Data and checkpoints live outside
the repository (`E:\_NoiserML`). Everything here was judged on held-out terrain, mostly by eye on full-resolution crops.

## What it does

Given a coarse heightmap, it adds plausible detail of real terrain, 4x per stage. Two stages chain to 16x: 16 m cells to 4 m,
then 4 m to 1 m. It does not try to recover the original terrain, only to generate believable detail that agrees exactly
with the coarse input (each coarse cell keeps its mean height).

- [16x, 1 km panels](ml/flow16_overview.png) and [a 256 m corner at full resolution](ml/flow16_zoom.png): input at 16 m, a
  smooth 16x upscale, two seeds, real 1 m lidar. It draws desert scarps, branching badlands channels and small hillside
  gullies of the kind the real lidar has; the smooth upscale has none. The seeds differ in the small channels: it invents
  them. Up close it is a little smoother than real at the finest scale (rock and soil roughness, tiny channels on hills).
- [4x alone](ml/flow1_compare.png) (256 m panels; the "old" column is the earlier model described below).

## How

- **Data** (`fetch_lidar.py`, `prep_lidar.py`): USGS 3DEP 1 m bare-earth lidar (public domain; OpenTopography restricts 1 m
  to academic keys), 23 tiles of 5 km over eight terrain types: two training sites per type (one for coastal) and one whole
  site per type held out (`_HOLD`). The first stage uses the same tiles averaged to 4 m (`prep_lidar.py E:/_NoiserML/lidar 4
  E:/_NoiserML/lidar_4m`). Flow accumulation (priority-flood filling, multiple flow directions) is computed per whole tile.
  Plains was dropped: its farmland gave a field grid as "detail" on anything labelled plains.
- **Gaea data** (`prep_gaea.py`): 25 terrains made in Gaea (5 x 2.5 km, 4096 px, so 1.22 m cells; 4.88 m for the first
  stage), giving four made-up types: craters, fantasy (two fantasy landscapes), rocky (rocky outcrops and sharp rocks) and
  icelandic; snow-covered cliffs join alpine. One terrain per group is held out (none of the cliffs). Tiles keep their own cell size (`cell` in the
  npz), which the model is conditioned on. The canyon export was a colour image, not heights, and is not used.
  `gaea_seeds.ps1` rebuilds eight projects (Alpine, Craters, both Fantasy, icelandic, rocky outcrops, sharp rocky, cliffs)
  with ten more mutation seeds each through Gaea's Build Swarm: 81 more terrains. Within a type, batches draw lidar and Gaea
  tiles half and half, so alpine (now mostly Gaea) stays real. `gaea_add_export.py` adds an Export node to projects without
  one (Gaea's examples); unused so far (the one build tried timed out).
- **Inputs** (`terrain_ml.py`): the coarse height upsampled smoothly (bicubic, a light blur and 4 back-projection passes:
  plain bicubic of block means leaves a strong 4-cell grid, [before and after](ml/upsample_grid_fix.png)), its slope,
  curvature, log flow accumulation, the terrain type, the relief and the cell size. Training inputs are randomly blurred and
  given smooth noise, so they are less like clean block averages.
- **Model** (`flow.py`): a 4.7 M parameter U-Net (40, 80, 120, 160 channels, `--chs`; 9 MB in fp16, 4 levels, receptive
  field a few hundred cells) trained by rectified flow matching to generate the residual (fine minus the smooth upsample),
  24 000 steps per stage (about 2 h each on an RTX 4070 SUPER), EMA weights. One sample in ten is trained with the mean
  type embedding instead of its own (`--drop_type 0.1`), so guidance has a learned "any terrain" to push away from. The
  channel widths are stored in the weights; 48 to 192 channels did not fit in 12 GB. Inference: 8 Euler steps from noise (32 look the same); the residual is projected to zero mean
  over every coarse cell. About 0.2 M multiply-adds per output pixel per step.
- **Whole maps** (`flow.enhance`): 256-cell tiles share one noise field and their velocities are blended at every step, so
  there are no seams (mean height step at joins 0.45 m against 0.47 m elsewhere). Agreement with the input: 3 cm RMS,
  0.55 m at most on cliffs, limited by the upsample's 4 back-projection passes (16 would give 4 mm, but the models were
  trained on 4).
- **Demo** (`demo.py`): regenerates the 16x images, optionally with a Noiser heightmap.

## In the app

The **Learned Detail** node (`src/landscape_detail.rs`; in files an Erosion node of kind Detail, so it shares erosion's
cached results, Run and outputs): terrain type (alpine, desert, badlands, hills, volcanic, karst, coastal from lidar;
craters, fantasy, rocky, icelandic from Gaea), then each stage on or off with its own steps (default 8), guidance (default
2, up to 8) and amount (how much of that stage's detail is kept, up to 3). Landforms (16x to 4x) and fine detail (4x to the
grid): both on read the input 16x coarser, one alone 4x. Sharpen (the third stage, 24 000 steps) adds the finest grain on
the grid itself, trained on lidar blurred by 1 to 2.5 cells (`flow.py --sharpen`). Against the early 4 000-step model it
drops the pits on volcanic and hills and adds more grain on desert, but it adds less detail on volcanic and hills, and badlands
still shows fine ripples.
**Adherence** (landforms stage): 1 keeps the input's coarse shape exactly; lower lets the model reshape it. Trained
(`flow.py --adhere 1.0`) with smooth noise of random strength lam (up to the patch's relief) added to the coarse input,
the model conditioned on lam, and for lam > 0 the target taken against the noisy input with no zero block mean; each noise
level is generated at unit variance (`flow.gain`). On a noise terrain, rocky at lam 0.3 to 0.6 breaks Simplex swirls into
crags and gullies; alpine gets blander. The training noise reaches shapes up to a quarter of a tile (about 1 km at 62 m coarse cells), so on a 2 km Perlin terrain adherence 0.9 keeps the main hill and roughens the ground, while 0.75 and below replace the hill itself. Strict
(adherence 1) it gives more detail on held-out lidar than v4. Longer training lowered the held-out loss (0.422 at 12 000
steps, 0.407 at 24 000) but made strict output softer (alpine detail 0.93 to 0.61 m against a real 0.98), even at 8
steps, so the app has the 12 000-step model.
Guidance is classifier-free guidance: the model is also run with the mean of the type embeddings (an average terrain,
trained as such by type dropout), and the velocity is pushed away from that one, v_mean + w (v_type - v_mean). On a noise
terrain at 2 m cells the types differ by 34% of the detail at w = 1, 46% at 1.5 and 60% at 2.5, and the detail is stronger
(1.03, 1.13, 1.35 m std); each step then costs two network runs. (The previous, smaller model without type dropout: 42% and
56% at 1 and 1.5, but less detail, 0.86 and 1.00 m.) Tried and not kept: a temperature (starting noise times 1.2 gives the
same pebbly texture for every type) and Heun steps (about what twice the Euler steps give). Detail is generated back to the working grid; the seed is the terrain's. Trained for 1 m output cells: a 1 km
terrain at working resolution 1024. A **Blur** node (an Erosion of kind Blur underneath; Gaussian, radius in metres) softens hard edges of the input before
Learned Detail. Flow is the input's flow accumulation; deposition and wear are where the result rose
and fell.

- A port of `flow.enhance`: the inputs on the CPU (rayon), the U-Net in WGSL compute (GroupNorm in two reduction passes,
  3x3 convolutions through workgroup memory, 16 output channels per workgroup), tiles blended every step. Checked against
  `tools/ml/flow.py`'s answers (`tests/fixtures/detail_fixture.nmlw`, made by `export_weights.py fixtures`): the inputs,
  one network call and a 3-step, 4-tile sampler agree within 0.2 to 0.5% (`cargo test --release landscape_detail`, with
  `NOISER_DETAIL_WEIGHTS` set to the weights' folder).
- Weights: `export_weights.py weights` writes `detail_16m_4m.nmlw` and `detail_4m_1m.nmlw` (9 MB each, f16). The app reads
  them from `ml\` next to the program, or the folder in `NOISER_DETAIL_WEIGHTS`. Weights list their terrain types; files
  with a different list (the first 8-type model) are refused with a message, not run with the wrong labels.
- RTX 4070 SUPER: 1 km at working resolution 1024 (16x, 8 steps) 2.3 s; 4096 42 s (441 tiles in the second stage). Peak GPU
  memory of a whole 4096 export about 1.2 GB.

## Cost

RTX 4070 SUPER, the Python reference, one tile at a time (not optimised):

| output | time | peak GPU memory |
|---|---|---|
| 1024 x 1024 | 3.8 s | 156 MB |
| 4032 x 4032 (an Unreal landscape) | 46 s | 1.06 GB |

The first stage runs on a 16x smaller grid, so it adds little. Most of the memory at 4k is full-map fp32 buffers; fp16
buffers or processing in strips would cut it. Batching tiles and fewer steps should cut the time; not measured.

## Does it generalise

Flow loss per type, held-out sites against training sites: alpine 0.61/0.67, desert 0.88/1.00, badlands 1.70/1.40, hills
0.48/0.41, volcanic 0.60/0.52, karst 0.69/0.34, coastal 0.63/0.50. No sign of memorising the training sites (karst is the
weakest). Plains 6.57/1.15 is the held-out Nebraska tile being nearly flat (2 m height spread), so its detail is huge relative
to its relief. The real risks are elsewhere: 15 training tiles, all US lidar; the type label is the site, not a landform class;
and artist input (one Noiser heightmap tested) differs from real terrain.

## How much detail

Standard deviation of the detail (16x result minus the smooth 16x upsample, metres), held-out lidar sites, three 1 km
crops each, 8 steps:

| site | real | first model (8 types) | v3 (11 types) | current (v4, bigger) |
|---|---|---|---|---|
| alpine | 1.48 | 1.06 | 0.89 | 0.93 |
| desert | 0.58 | 0.39 | 0.31 | 0.34 |
| badlands | 0.35 | 0.22 | 0.17 | 0.17 |
| hills | 0.37 | 0.31 | 0.29 | 0.27 |
| volcanic | 5.24 | 1.57 | 1.34 | 1.75 |
| karst | 0.17 | 0.21 | 0.17 | 0.16 |
| coastal | 0.97 | 0.40 | 0.36 | 0.37 |

The bigger v4 (more channels, longer training, type dropout, the Gaea seeds) gains most on volcanic and a little on
alpine and desert; up close it is still softer than lidar ([held-out sites: real, v3, v4, v4 at guidance 1.5](ml/compare_v4_lidar.png),
[noise terrain as five types, v3 and v4 at several guidance values](ml/compare_v4_types.png)).

Both models make less detail than real terrain (the right kind, in the right places, but weaker), and the current one about
12% less than the first. Most of that 12% is the residual's normalising scale, which the Gaea tiles lowered (0.343 against
0.383 in the first stage, 0.177 against 0.185 in the second): run with the first model's scales, the current one is within
about 15% of it everywhere. Two levers in the app make up for it: amount above 1 (the slider goes to 2), and more steps
(32 steps instead of 8 added 20 to 30% detail to the first model, at 4 times the time). In exchange it has the four Gaea types, and hills no longer
draws plains' field grid. Sheets (256 m corners at 1 m): [one noise heightmap as every type](ml/types16.png), [held-out
lidar: real, first model, current](ml/types16_lidar_vs_first.png), [noise-made inputs: first model, current](ml/noise_inputs16.png).

## What did not work

An earlier model (dilated convolutions, 221 k parameters; first on 30 m Copernicus data, then on the 1 m lidar) predicted the
detail with L1, a radial-spectrum loss and a PatchGAN. Up close it was blur plus noise, never sharper: L1 regresses to the
average answer and the spectrum loss only adds back the right amount of energy. The GAN made dense noise and a strong 4-cell
grid; a batch-averaged spectrum loss made wormy ripples. At 16x it gave a blobby texture. Generating a sample (flow matching)
instead of predicting the average is what fixed it. Also fixed on the way: grids from bilinear and bicubic upsampling of
block means, and straight compass-line channels from single-direction (D8) flow accumulation.

## Limits

1. **Finest scale** is a little too smooth; a 1.6x bigger model trained 1.5x longer helped only a little, so a third stage
   or a different loss weighting at fine scales is the next thing to try.
2. **Data**: 15 training tiles from one country. More sites, and landform labels per patch instead of site labels.
3. **Artist input**: tested on 12 noise-made heightmaps (ridged, Worley, terraces, warped noise) as well as Gaea's. Training
   with noise heightmaps added to a share of the coarse inputs (`flow.py --noise`, 36 maps; the target detail stays the
   real terrain's) gave no more detail on held-out lidar than the same training without it, so it is off.
4. **Metrics**: the per-type loss and visual side-by-sides only.
5. **Port**: `flow.enhance` is the reference to test a WGSL version against.
6. **Licensing**: USGS 3DEP is public domain, so trained weights carry no data licence; the earlier Copernicus data is no
   longer used.

Reproduce: `python tools/ml/fetch_lidar.py && python tools/ml/prep_lidar.py && python tools/ml/prep_lidar.py E:/_NoiserML/lidar 4 E:/_NoiserML/lidar_4m`,
`python tools/ml/prep_gaea.py` (the Gaea exports in `E:/_NoiserML/gaea`; needs OpenEXR), then the two `flow.py train` commands in its docstring, then `python tools/ml/demo.py over.png zoom.png`. Needs PyTorch with
CUDA, numpy, numba, scikit-image, rasterio (data prep only), Pillow.
