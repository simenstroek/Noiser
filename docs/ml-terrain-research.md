# Learned terrain detail: what exists, and what we tried

The idea: instead of simulating erosion, learn from real elevation data. Take real terrain, throw its detail away (downsample),
and train a small network to put plausible detail back. In Noiser it would be a heightmap-to-heightmap node next to erosion.
`docs/ml-terrain-results.md` has what a first experiment found; this page is the survey.

## Existing work (from web searches in November 2026; read the papers before relying on a claim)

| Work | What it does | Notes |
|---|---|---|
| Guérin et al., *Interactive Example-Based Terrain Authoring with Conditional GANs*, ACM TOG 2017 ([paper](https://www.cs.purdue.edu/homes/bbenes/papers/Guerin17ToG.pdf)) | cGANs trained on real DEMs: sketch to terrain, and a learned stand-in for erosion ("meta-simulation") | The closest ancestor of this idea. Uses real terrain (SRTM-class data) per terrain type. |
| GATA, *Generative Adversarial Terrain Amplification* | A GAN that adds detail to a low-resolution terrain, conditioned on an embedding for the desired theme | The amplification setting we want. Reported as the state of the art for amplification by StyleDEM. Not read in full. |
| StyleDEM ([arXiv 2304.09626](https://arxiv.org/pdf/2304.09626)) | Sketch/terrain in, authored terrain out, with style manipulation and super-resolution | Says erosion-like features are learned implicitly from real examples. |
| VAE + cGAN ([arXiv 2201.02369](https://www.arxiv.org/pdf/2201.02369)) | Learns a latent space of real terrain; a cGAN makes a DEM from a topographic map | |
| MESA ([arXiv 2504.07210](https://arxiv.org/pdf/2504.07210)) | Text-to-terrain latent diffusion (Stable Diffusion 2.1 base) trained on Copernicus DEM, dataset on Major TOM | Weights and inference code published. A whole-terrain generator, not an amplifier. |
| Terrain Diffusion ([repo](https://github.com/xandergos/terrain-diffusion), [paper](https://arxiv.org/abs/2512.08309)) | "InfiniteDiffusion": lazy, unbounded, seed-consistent diffusion sampling for planetary terrain; 30 m and 90 m pretrained models | The determinism and random-access idea is exactly what a chunked Noiser export wants. Training data not checked. |
| Talus ([weights](https://huggingface.co/osfv/talus), Apache-2.0) | 23 M parameter conditional diffusion model, 64x64 maps of 4 km, ONNX and WebGPU | Trained on its own *procedural* maps, not real data. Useful as a proof that a small model runs in WebGPU. |
| Earthbender, TerraFusion (2025) | Sketch-guided ControlNet heightmaps; joint height and texture latent diffusion | Weights not confirmed. |

No dedicated 2024 to 2026 paper on neural *hydraulic* erosion emulation turned up beyond Guérin et al.

## Data

- **Copernicus DEM GLO-30** (30 m, global, open AWS bucket, no key). A *surface* model: trees and buildings are in it. Licence:
  free, worldwide; when you redistribute the data you must show the Copernicus source notice and a liability disclaimer, whose
  exact wording is in the [licence](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf)
  (the first experiment used it; the current model uses only public-domain USGS lidar). OpenTopography lists the same data as CC BY 4.0, which conflicts; treat the Copernicus licence as governing. A few
  countries' tiles are withheld. Whether trained weights are a derived work is not settled by the licence text: ask before shipping.
- **USGS 3DEP lidar** (1 m, US, bare earth) is what fine detail really needs; its licensing was not confirmed in this survey
  (US federal data is generally public domain, verify). Access through OpenTopography needs a free API key.
- Datasets named in the papers above (SRTM, MESA's Major TOM Copernicus extract) were not downloaded.

## What the survey suggests

1. Amplification, not generation, is the right shape for Noiser: the artist's coarse terrain stays authoritative.
2. Conditioning on terrain type (an embedding) and on scale is standard; one network, many presets.
3. Hydraulic structure is non-local. Feeding analytic drainage (flow accumulation) as an input is a cheap hybrid; diffusion
   models with global context (Terrain Diffusion's hierarchy) are the heavyweight answer.
4. Determinism and tiling matter as much as quality for a chunked pipeline; InfiniteDiffusion is worth reading for that alone.
