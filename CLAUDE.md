# Bachelor Thesis: SAR-to-Optical Image Translation Using cGANs

## Project Overview

This repository contains both the **Python codebase** and **LaTeX source** for a bachelor thesis and its MDPI journal manuscript. The thesis investigates SAR-to-optical (multispectral) image translation using conditional Generative Adversarial Networks (cGANs), with application to cloud removal in remote sensing imagery.

- **Author:** Ahmed Attia
- **Degree:** Bachelor of Science (B.Sc.) in Artificial Intelligence (Kuenstliche Intelligenz)
- **Faculty:** Applied Computer Science (Angewandte Informatik)
- **Supervisor:** Dr. Peter Hofmann
- **Target Journal:** MDPI Remote Sensing

## Repository Structure

```
codebase/                          # Python ML pipeline
  prepare_data_sen_12_ms.py        # Data prep: loads SEN12-MS .tif, crops, normalizes, season-stratified split
  prepare_data_sen_12_ms_60m_excluded.py  # Same but excludes 60m bands (B1, B9, B10)
  prepare_data_sen_12_ms_cr.py     # Data prep for cloud removal (SEN12-MS-CR dataset)
  train_pix2pix_lpips_ssim_generate_*.py  # Training scripts (pix2pix with L1 + SSIM + LPIPS losses)
  evaluate_pix2pix_clips_per_band_*.py    # Per-band evaluation (MAE, RMSE, SSIM, LPIPS, etc.)
  infer_generate_optical*.py       # Inference/generation scripts (standard + cloud removal)
  myutils.py                       # Shared utilities (data loading, clipping configs, plotting)
  mymodels/                        # Model wrappers and submodules
    cp2pwrapper.py                 # Pix2Pix (UNet generator) wrapper
    SiMaVP.py                      # SiMa-UNet: custom U-Net with Mamba bottleneck
    UMambaWrapper.py               # U-Mamba encoder wrapper
    swinunetwrapper.py             # Swin-Unet wrapper
    pytorch-CycleGAN-and-pix2pix/  # Vendored pix2pix/CycleGAN repo (submodule)
  dataset_*/                       # Preprocessed datasets with clip JSONs and split metadata

latex_source_code/       # Full bachelor thesis (KOMA-Script scrbook)
  main.tex               # Thesis entry point
  chapters/              # Chapter .tex files (introduction, methodology, results, etc.)
  img/                   # Thesis figures
  references.bib         # BibTeX bibliography

MDPI_manuscript/         # Journal paper (multi-file version)
  main.tex               # Paper entry point
  Definitions/           # MDPI journal class files and styles
  img/                   # Paper figures (ablation, cloud removal, band panels, etc.)
  *.tex                  # Section files (introduction, related_work, results, etc.)

MDPI_manuscript_single/  # Journal paper (single-file version for submission)
  main.tex               # Self-contained paper
```

## Codebase Details

### Tech Stack
- **Language:** Python 3
- **Framework:** PyTorch (CUDA)
- **Key Libraries:** torchvision, lpips, scikit-image, scikit-learn, rasterio (GDAL), numpy, matplotlib, tqdm
- **Data Format:** `.npy` arrays (preprocessed), `.tif` GeoTIFF (raw SEN12-MS)

### Data Pipeline
1. **Raw data:** SEN12-MS dataset (Sentinel-1 SAR + Sentinel-2 multispectral, paired by location/season)
2. **Preprocessing (`prepare_data_*.py`):** Reads `.tif` files via rasterio, crops to 256x256, clips values, normalizes to [-1, 1] (tanh range), saves as `.npy` with season-stratified train/val/test splits
3. **Clipping config:** Sentinel-1 clips: VV [-25, 0] dB, VH [-32.5, 0] dB; Sentinel-2 clips: [0, 10000] per band. Stored in `S1_clip.json` / `S2_clip.json`

### Models
- **Pix2Pix (primary):** UNet-256 generator via `Pix2PixWrapper`, trained with combined L1 + SSIM + LPIPS loss
- **SiMaVP:** Custom U-Net with a bidirectional Mamba state-space-model bottleneck (~26M params)
- **U-Mamba:** UMambaEnc-based encoder-decoder
- **Swin-Unet:** Vision-transformer-based U-Net

### Training Config
- **Input:** 2-channel SAR (VV + VH) or single channel, 256x256
- **Output:** 13-band (or 10-band with 60m excluded) multispectral, 256x256
- **Loss:** `lambda_L1=100`, `lambda_ssim=50`, `lambda_lpips=50`
- **Optimizer:** Adam, lr=1e-4, with optional ReduceLROnPlateau
- **Warmup:** 20 epochs generator-only before adversarial training
- **Batch size:** 16, epochs: 150, early stopping patience: 20
- **LPIPS:** Computed on RGB bands (B4, B3, B2 — indices 2, 1, 0 in 10-band layout)

### Evaluation Metrics
Per-band: MAE, RMSE, SSIM, LPIPS (where applicable). Band groups: Visible (B2-B4), Red Edge (B5-B7), NIR (B8, B8A), SWIR (B11-B12).

### Running on Remote Server
- Training data lives at paths like `/home/dll0505/AdditionalStorage/thesis/sen12-ms/...`
- Scripts are designed to run on a GPU workstation (single GPU, configurable `gpu_id`)
- `num_workers=16` in data loaders

## LaTeX Details

- **LaTeX engine:** pdfLaTeX (`pdftex` option)
- **Thesis class:** KOMA-Script `scrbook`
- **Journal class:** Custom MDPI class (`Definitions/mdpi.cls`)
- **Bibliography:** `natbib` (MDPI paper), `cite` (thesis) with `.bib` / inline `\bibitem`
- **Language:** English (paper), with German metadata in the thesis

## Working with This Repo

- The active branch for paper work is `paper`; `master` is the main branch.
- The MDPI manuscript exists in two forms: a multi-file version (`MDPI_manuscript/`) for editing and a single-file version (`MDPI_manuscript_single/`) for submission.
- Images are stored as `.png` and `.pdf` files in `img/` subdirectories.
- Compile LaTeX with `pdflatex` followed by `bibtex`.

## Conventions

- Follow MDPI formatting guidelines when editing the journal manuscript.
- Use `\input{}` for section includes in the multi-file manuscript.
- Keep figure paths relative to the document root.
- Model wrappers in `codebase/mymodels/` adapt external repos to a common `nn.Module` interface with `(in_channels, out_channels, img_size)` constructors.
- Data normalization is always to [-1, 1] (tanh range); inverse transform uses the clip JSON files.
