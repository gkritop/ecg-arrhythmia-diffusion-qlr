# Generative ECG Class Imbalance: cVAE + Diffusion + Quantum Latent Refinement

Code accompanying:

> Kritopoulos, G., Neofotistos, G., Barmparis, G.D., Tsironis, G.P. **"Addressing Class Imbalance in ECG Arrhythmia Classification Using Latent Diffusion and Quantum-Enhanced Generative Modeling."** *AI Med* (MDPI), 2026.

A three-stage hybrid generative pipeline for class-imbalanced ECG arrhythmia classification on the MIT-BIH Arrhythmia Database:

1. **Spectral-guided conditional VAE (cVAE)** — compresses each 389-sample ECG beat into a 32-dimensional class-conditional latent space.
2. **Class-conditional latent DDPM** — a denoising diffusion model trained on the cVAE latent space, generating new synthetic minority-class latents.
3. **Quantum Latent Refinement (QLR)** — an 8-qubit parameterized quantum circuit, paired with a lightweight classical gate, that applies a bounded MMD-guided correction to align DDPM-generated latents with the real class-specific latent manifold.

A 1D MobileNetV2 classifier is trained on the augmented dataset and evaluated across ten random seeds and four augmentation ratios, benchmarked against an unaugmented baseline, SMOTE, cVAE-only, and plain DDPM augmentation.

## Summary of what the results show

All three deep generative methods (cVAE, DDPM, DDPM+QLR) **significantly improve Macro F1 over the unaugmented baseline** (large effect sizes, p < 0.02), while **SMOTE's improvement over baseline does not reach significance** (p = 0.64). Across the ten-seed comparison, **DDPM+QLR is not statistically distinguishable from plain DDPM** at any augmentation ratio — this work does not claim a quantum advantage. See the paper for the full statistical protocol, per-class breakdown, and the clinical framing of the results (in particular the reduction in false Ventricular alarms). This README mirrors that framing rather than a more favorable one.

## Repository structure

```
.
├── quantum_ecg_class_imbalance.py   # Full pipeline: data loading, cVAE, DDPM, QLR, classifier, ablation, all figures
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── data/
    ├── mit_bih/                     # MIT-BIH records go here (not included, see below)
    └── outputs/                     # Generated checkpoints, figures, and result CSVs land here
```

## Data

This code uses the **MIT-BIH Arrhythmia Database**, publicly available on PhysioNet and not redistributed in this repository:

- https://physionet.org/content/mitdb/

Download the database and place it so the record files (`100.dat`, `100.hea`, `100.atr`, ...) live at `./data/mit_bih/` relative to wherever you run the script, e.g.:

```bash
pip install wfdb
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mit_bih')"
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for practical training times (the original experiments were run on an NVIDIA RTX 4090). QLR's quantum circuits are simulated classically via PennyLane's `default.qubit` device — no quantum hardware is required or used.

## Usage

All configuration (seeds, epochs, learning rates, augmentation ratios, output paths) is set via constants near the top of `quantum_ecg_class_imbalance.py` rather than command-line arguments. Edit these as needed, then run:

```bash
python quantum_ecg_class_imbalance.py
```

This runs the full pipeline end-to-end: cVAE → DDPM → QLR training per class, per-seed classifier training and evaluation across all augmentation methods and ratios, the classical-refiner ablation (Class V, seed 0 of `GLOBAL_SEEDS`), and all figures/tables reported in the paper. A full ten-seed run takes on the order of tens of hours on a single high-end GPU; reduce `GLOBAL_SEEDS` to a single seed for a quick smoke test.

Outputs (model checkpoints, figures, CSV result tables) are written to `data/outputs/`.

## Trained model checkpoints

Checkpoints (`.pth` files for the VAE/DDPM/QLR modules per seed) are not included in this repository due to size. They are available on request — see contact details in the paper or open an issue.

## Citing this work

If you use this code, please cite the paper (see `CITATION.cff`, or use GitHub's "Cite this repository" button):

```bibtex
@article{kritopoulos2026qlr,
  title   = {Addressing Class Imbalance in ECG Arrhythmia Classification Using Latent Diffusion and Quantum-Enhanced Generative Modeling},
  author  = {Kritopoulos, Georgios and Neofotistos, Georgios and Barmparis, Georgios D. and Tsironis, Giorgos P.},
  journal = {AI Med},
  publisher = {MDPI},
  year    = {2026},
  doi     = {10.3390/aimed1010000}
}
```

## License

MIT — see `LICENSE`.

## Funding

This work was supported by the Department of the Navy award N629092412119 issued by the Office of Naval Research Global, USA.
