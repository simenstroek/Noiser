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
- **Gaea data** (`prep_gaea.py`): 27 terrains made in Gaea (5 x 2.5 km, 4096 px, so 1.22 m cells; 4.88 m for the first
  stage), giving four made-up types: craters, fantasy (two fantasy landscapes), rocky (rocky outcrops and sharp rocks) and
  icelandic; snow-covered cliffs join alpine. One terrain per group is held out. Tiles keep their own cell size (`cell` in the
  npz), which the model is conditioned on. The canyon export was a colour image, not heights, and is not used.
- **Inputs** (`terrain_ml.py`): the coarse height upsampled smoothly (bicubic, a light blur and 4 back-projection passes:
  plain bicubic of block means leaves a strong 4-cell grid, [before and after](ml/upsample_grid_fix.png)), its slope,
  curvature, log flow accumulation, the terrain type, the relief and the cell size. Training inputs are randomly blurred and
  given smooth noise, so they are less like clean block averages.
- **Model** (`flow.py`): a 3.0 M parameter U-Net (6 MB in fp16, 4 levels, receptive field a few hundred cells) trained by
  rectified flow matching to generate the residual (fine minus the smooth upsample), 16 000 steps per stage (62 min on an
  RTX 4070 SUPER; 11 types, so about the same steps per type as the first 8-type model's 12 000), EMA weights. Inference: 8 Euler steps from noise (32 look the same); the residual is projected to zero mean
  over every coarse cell. About 0.2 M multiply-adds per output pixel per step.
- **Whole maps** (`flow.enhance`): 256-cell tiles share one noise field and their velocities are blended at every step, so
  there are no seams (mean height step at joins 0.45 m against 0.47 m elsewhere). Agreement with the input: 3 cm RMS,
  0.55 m at most on cliffs, limited by the upsample's 4 back-projection passes (16 would give 4 mm, but the models were
  trained on 4).
- **Demo** (`demo.py`): regenerates the 16x images, optionally with a Noiser heightmap.

## In the app

The Erosion node's third kind, **Detail** (`src/landscape_detail.rs`): settings terrain type (alpine, desert, badlands,
hills, volcanic, karst, coastal from lidar; craters, fantasy, rocky, icelandic from Gaea), 4x or 16x, steps (default 8)
and amount (0 keeps the input, 1 is the full result). The input is averaged to 4 or 16 times coarser cells on the working
grid, and detail is generated back to the working grid; the seed is the terrain's. Trained for 1 m output cells: a 1 km
terrain at working resolution 1024. Flow is the input's flow accumulation; deposition and wear are where the result rose
and fell.

- A port of `flow.enhance`: the inputs on the CPU (rayon), the U-Net in WGSL compute (GroupNorm in two reduction passes,
  3x3 convolutions through workgroup memory, 16 output channels per workgroup), tiles blended every step. Checked against
  `tools/ml/flow.py`'s answers (`tests/fixtures/detail_fixture.nmlw`, made by `export_weights.py fixtures`): the inputs,
  one network call and a 3-step, 4-tile sampler agree within 0.2 to 0.5% (`cargo test --release landscape_detail`, with
  `NOISER_DETAIL_WEIGHTS` set to the weights' folder).
- Weights: `export_weights.py weights` writes `detail_16m_4m.nmlw` and `detail_4m_1m.nmlw` (6 MB each, f16). The app reads
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

| site | real | first model (8 types) | current (11 types) |
|---|---|---|---|
| alpine | 1.48 | 1.06 | 0.89 |
| desert | 0.58 | 0.39 | 0.31 |
| badlands | 0.35 | 0.22 | 0.17 |
| hills | 0.37 | 0.31 | 0.29 |
| volcanic | 5.24 | 1.57 | 1.34 |
| karst | 0.17 | 0.21 | 0.17 |
| coastal | 0.97 | 0.40 | 0.36 |

Both models make less detail than real terrain (the right kind, in the right places, but weaker), and the current one about
12% less than the first. Most of that 12% is the residual's normalising scale, which the Gaea tiles lowered (0.343 against
0.383 in the first stage, 0.177 against 0.185 in the second): run with the first model's scales, the current one is within
about 5 to 10% of it everywhere. Two levers in the app make up for it: amount above 1 (the slider goes to 2), and more steps
(32 steps instead of 8 add 20 to 30% detail, at 4 times the time). In exchange it has the four Gaea types, and hills no longer
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

1. **Finest scale** is a little too smooth; more training, a larger top level, or a third stage could help.
2. **Data**: 15 training tiles from one country. More sites, and landform labels per patch instead of site labels.
3. **Artist input**: tested on 12 noise-made heightmaps (ridged, Worley, terraces, warped noise) as well as Gaea's. Training
   with noise heightmaps added to a share of the coarse inputs (`flow.py --noise`, 36 maps; the target detail stays the
   real terrain's) made the detail about 20% weaker everywhere, noise inputs included, so it is off.
4. **Metrics**: the per-type loss and visual side-by-sides only.
5. **Port**: `flow.enhance` is the reference to test a WGSL version against.
6. **Licensing**: USGS 3DEP is public domain, so trained weights carry no data licence; the earlier Copernicus data is no
   longer used.

Reproduce: `python tools/ml/fetch_lidar.py && python tools/ml/prep_lidar.py && python tools/ml/prep_lidar.py E:/_NoiserML/lidar 4 E:/_NoiserML/lidar_4m`,
`python tools/ml/prep_gaea.py` (the Gaea exports in `E:/_NoiserML/gaea`; needs OpenEXR), then the two `flow.py train` commands in its docstring, then `python tools/ml/demo.py over.png zoom.png`. Needs PyTorch with
CUDA, numpy, numba, scikit-image, rasterio (data prep only), Pillow.
