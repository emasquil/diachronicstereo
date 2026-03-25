# Diachronic Stereo Matching for Multi-Date Satellite Imagery

This repository contains the code for the paper *Diachronic Stereo Matching for Multi-Date Satellite Imagery*.

Accepted for presentation at the XXV ISPRS Congress.

- [Project page](https://centreborelli.github.io/diachronic-stereo/)
- [Paper](https://arxiv.org/abs/2601.22808)
- [Diachronic model](https://huggingface.co/emasquil/diachronic-stereo)

## Abstract

Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions. On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations. On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs. However, when the two images are captured months apart, strong seasonal, illumination, and shadow changes violate standard stereoscopic assumptions, causing existing pipelines to fail. This work presents the first Diachronic Stereo Matching method for satellite imagery, enabling reliable 3D reconstruction from temporally distant pairs. Two advances make this possible: (1) fine-tuning a state-of-the-art deep stereo network that leverages monocular depth priors, and (2) exposing it to a dataset specifically curated to include a diverse set of diachronic image pairs. In particular, we start from a pretrained MonSter model, trained initially on a mix of synthetic and real datasets such as SceneFlow and KITTI, and fine-tune it on a set of stereo pairs derived from the DFC2019 remote sensing challenge. This dataset contains both synchronic and diachronic pairs under diverse seasonal and illumination conditions. Experiments on multi-date WorldView-3 imagery demonstrate that our approach consistently surpasses classical pipelines and unadapted deep stereo models on both synchronic and diachronic settings. Fine-tuning on temporally diverse images, together with monocular priors, proves essential for enabling 3D reconstruction from previously incompatible acquisition dates.

## Setup

Create the environment with:

```bash
conda env create -f environment.yml
```

Then install the CUDA toolkit if needed, and `flash-attn`:

```bash
conda install cuda-toolkit -c nvidia
pip install flash-attn
```

We use many utilities from `s2p-hd`. Install it following its instructions:

https://github.com/centreborelli/s2p-hd

See the README files of the third-party projects in `thirdparty` for project-specific details.

## Checkpoints

Download the required pretrained checkpoints:

- Diachronic model (fine-tuned project weights): https://huggingface.co/emasquil/diachronic-stereo
- Depth Anything V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/tree/main
- MonSter (mix of all datasets): https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth?download=true
- StereoAnywhere (`sceneflow.tar`): https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x?usp=sharing
- FoundationStereo (`23-51-11`, biggest model): https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing

## Usage

Run stereo on a single stereo pair for visualization:

```bash
python single_inference.py --help
```

Train MonSter on diachronic data:

```bash
python train_monster.py --help
```

Run evaluations on full datasets:

```bash
python evaluate_all.py --help
python evaluate_all_aerial.py --help
```

## Scripts and Utilities

The `scripts` directory contains utilities developed for this project, including stereo rectification, SIFT match computation, DSM projections, and related preprocessing tools.

## Citation

If you find this project useful, please cite:

```bibtex
@article{masquil2026diachronic,
  title={Diachronic Stereo Matching for Multi-Date Satellite Imagery},
  author={Masquil, El{\'\i}as and Aira, Luca Savant and Mar{\'\i}, Roger and Ehret, Thibaud and Mus{\'e}, Pablo and Facciolo, Gabriele},
  journal={arXiv preprint arXiv:2601.22808},
  year={2026}
}
```
