# Probe-Guided Gradient Boosting (PGGB) for Multimodal Learning

Code release for **"Probe-Guided Gradient Balancing for Multimodal Learning"**, accepted at NeurIPS 2026 (poster).

PGGB is a method for balanced multimodal learning that amplifies weaker modalities while preserving stronger ones, composing naturally with existing throttling techniques such as OGM-GE.

## Method overview

- **Decoupled probe monitoring** — lightweight linear probes on stop-gradient encoder features yield a per-modality utilization signal independent of the joint training objective.
- **Bounded multiplicative boost** — the probe-detected gap drives an EMA-smoothed scaling factor on the weaker encoder's gradient, with a hard cap and gap-driven decay toward identity.
- **Composability** — acts only on encoder gradients (not loss or architecture), composing multiplicatively with throttling methods to produce two-sided pressure toward modality balance.
- **Provable safety properties** — bounded scaling, self-attenuation, and a standard-SGD descent bound under standard smoothness and bounded-variance assumptions.

## Repository layout

```
.
├── src/
│   ├── datasets/        # Dataset loaders (CREMA-D, AVE, KS, CH-SIMS, MOSI, Twitter15, Sarcasm, Food101)
│   ├── losses/          # PGGB loss/scheduler + CGGM baseline
│   ├── models/          # Encoders, fusion, probes
│   └── utils/           # Metrics, logging
├── configs/             # Per-dataset YAML configs (cremad, ave, ks, chsims, mosi, twitter, sarcasm, food101)
├── scripts/
│   ├── train.py         # Main training script
│   ├── train_brats.py   # BraTS 2021 segmentation
│   ├── train_food101_e2e.py
│   ├── extract_*.{sh,py}        # Feature extraction (audio frames, text+image embeddings)
│   ├── prepare_*.{sh,py}        # Dataset preparation
│   ├── sweep_*.sh                # Hyperparameter / ablation sweeps
│   ├── post_hoc_probe_cremad.py # Post-hoc linear-probe protocol for §4.4
│   ├── probe_stability_plot.py  # Probe-stability figure
│   ├── probe_trajectory_early_plot.py # Early probe-trajectory figure
│   ├── post_hoc_probe_sentiment.py    # Post-hoc probes, sentiment benchmarks
│   ├── alpha_multiseed.sh             # Multi-seed boost-strength sweep
│   ├── hp_sensitivity_plots.py  # App. B.5 hyperparameter-sensitivity figure
│   └── ablation_*.py             # Ablation/figure generators
├── pyproject.toml       # Package metadata + dependencies
└── README.md
```

## Installation

Tested on Python 3.10 with CUDA 12.1 + PyTorch 2.0+, RTX 4090 (24 GB VRAM).

```bash
# Recommended: conda environment
conda create -n pggb python=3.10 -y
conda activate pggb

# Install PyTorch (match your CUDA version)
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0

# Install package + dependencies
pip install -e .
```

Or with the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Datasets

Standard splits used in the paper. Place each dataset under `data/<dataset>/`:

| Dataset      | Modalities             | Splits (train / val / test)  | Source                                          |
|--------------|------------------------|------------------------------|-------------------------------------------------|
| CREMA-D      | audio, video           | 6,698 / — / 744              | https://github.com/CheyneyComputerScience/CREMA-D |
| AVE          | audio, video           | 3,328 / 413 / 402            | https://sites.google.com/view/audiovisualresearch |
| Kinetics-Sounds | audio, video        | 22,408 / — / 3,000           | https://github.com/cvdfoundation/kinetics-dataset |
| CH-SIMS      | text, audio, visual    | 1,368 / 456 / 457            | https://github.com/thuiar/MMSA                  |
| CMU-MOSI     | text, audio, visual    | 1,284 / 229 / 686            | https://github.com/A2Zadeh/CMU-MultimodalSDK    |
| Twitter15    | text, image            | 3,179 / 1,122 / 1,037        | https://github.com/jefferyYu/TomBERT             |
| Sarcasm (MMSD-v1) | text, image       | 19,816 / 2,410 / 2,409       | https://github.com/headacheboy/data-of-multimodal-sarcasm-detection |
| Food101 (UPMC) | text, image          | 67,972 / — / 22,716          | https://visiir.lip6.fr/                         |
| BraTS 2021   | 4 MRI sequences        | 1,000 / 125 / 126            | https://www.synapse.org/brats2021               |

Feature-extraction helpers in `scripts/`:

```bash
# Audio-visual frame extraction
bash scripts/extract_cremad_frames.sh
bash scripts/extract_ave_frames.sh
bash scripts/extract_ks_frames_audio.sh

# Text+image feature pre-extraction (BERT + ResNet-18) for Twitter15, Sarcasm, Food101
python scripts/extract_text_image_features.py --dataset food101
```

## Quick start: train PGGB on CREMA-D

```bash
# Boost-only (PGGB without OGM-GE)
python scripts/train.py \
    --config configs/cremad.yaml \
    --mode adaptive --asgml-mode continuous \
    --continuous-alpha 0.5 \
    --num-frames 3 --fps 3 \
    --seed 42 --epochs 100

# PGGB + OGM-GE composition (paper's headline result on CREMA-D)
python scripts/train.py \
    --config configs/cremad.yaml \
    --mode adaptive --asgml-mode continuous \
    --ogm-ge --alpha 0.8 \
    --continuous-alpha 0.75 \
    --num-frames 3 --fps 3 \
    --seed 42 --epochs 100
```

## Reproducing main paper results

### Table 1 — Main results (8 benchmarks, 5 seeds each)
```bash
bash scripts/sweep_cremad.sh phase2     # CREMA-D 5-seed
bash scripts/sweep_ave.sh                # AVE
bash scripts/sweep_ks.sh                 # Kinetics-Sounds
bash scripts/sweep_chsims.sh             # CH-SIMS
bash scripts/sweep_mosi.sh               # CMU-MOSI
# Twitter15 and Sarcasm have no dedicated sweep script. Run train.py directly
# with configs/twitter.yaml or configs/sarcasm.yaml, varying --seed, as in Quick start.
bash scripts/sweep_brats.sh              # BraTS 2021
```

### Table 2 — 1-frame ablation (CREMA-D, hardest setting)
```bash
bash scripts/sweep_cremad.sh aggregate   # Standard 1-frame ablation
```

### App. B.7 — Composability across balancing methods (Table 7)
```bash
bash scripts/sweep_boost_compose.sh
```

### App. B.8 — Post-hoc probe accuracy (Table 9, Figure 3)
```bash
python scripts/post_hoc_probe_cremad.py
```

### App. B.6 — Probe signal stability (Figure 5)
```bash
python scripts/probe_stability_plot.py
```

### App. B.5 — Hyperparameter sensitivity (Figure 4)
```bash
python scripts/hp_sensitivity_plots.py
```

### Boost-strength sensitivity on the sentiment benchmarks (multi-seed)
```bash
bash scripts/alpha_multiseed.sh          # CH-SIMS and CMU-MOSI, 5 alpha values x 5 seeds
```

### Optimizer dependence of the boost actuation
```bash
# Per-step transmission logging is opt-in through an environment variable
PGGB_UPDATE_NORMS_FILE=norms.jsonl python scripts/train.py --config configs/mosi.yaml \
    --mode adaptive --asgml-mode continuous --continuous-alpha 0.75 --seed 42
bash scripts/rebuttal_mosi_sgd_acc.sh    # CMU-MOSI Adam vs SGD accuracy contrast
```

### Early probe trajectory
```bash
# train.py logs per-modality EMA probe accuracy and the signed gap to tensorboard
python scripts/probe_trajectory_early_plot.py
```

### Post-hoc probes on the sentiment benchmarks
```bash
python scripts/post_hoc_probe_sentiment.py
```

## Key hyperparameters

PGGB uses **identical hyperparameters across all datasets**:

| Param        | Value          | Description                                      |
|--------------|----------------|--------------------------------------------------|
| `K`          | 20             | Probe evaluation interval (training iterations)  |
| `β`          | 0.1            | Probe-accuracy EMA coefficient                    |
| `μ`          | 0.3            | Boost-scale EMA coefficient                       |
| `s_max`      | 2.0            | Cap on multiplicative scale                      |
| `γ`          | 1.0            | Auxiliary unimodal loss coefficient              |
| `α`          | 0.5 (alone)    | Boost strength when used standalone              |
|              | 0.75 (with OGM-GE) | Boost strength composed with OGM-GE          |

Probes use Adam (lr `1e-3`). Main encoders use SGD (lr `1e-3`, weight decay `1e-4`) for audio-visual datasets and Adam (lr `1e-3`) for sentiment / text-image datasets. BraTS uses SGD with cosine scheduling.

## Citation

If you use this code or method in your research, please cite:

```bibtex
@inproceedings{vu2026pggb,
  title={Probe-Guided Gradient Balancing for Multimodal Learning},
  author={Vo, Vu and Fayek, Haytham M. and Nguyen, Thuy},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT. See `LICENSE`.

## Contact

- Vu Vo — `s3575819@rmit.edu.vn`
- Haytham M. Fayek — `haytham.fayek@rmit.edu.au`
- Thuy Nguyen — `thuy.nguyen43@rmit.edu.vn`

School of Science, Engineering & Technology, RMIT University.
