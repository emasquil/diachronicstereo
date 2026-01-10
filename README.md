# Diachronic Stereo Matching for Multi-Date Satellite Imagery

## setup
`conda env create -f environment.yml`

Then install CUDA toolkit if needed; and flash-attn

```
conda install cuda-toolkit -c nvidia
pip install flash-attn
```

download stereo and depthanything v2 checkpoints:

- depth anything v2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/tree/main
- monster (mix of all datasets): https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth?download=true
- stereoanywhere (sceneflow.tar) https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x?usp=sharing
- foundation stereo (23-51-11, biggest model): https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing

See the READMEs of each project (in `thirdparty`) for specific details.

We use many utilities from s2p-hd, install it following their instructions https://github.com/centreborelli/s2p-hd.

## run stereo on a single stereo pair for visualization
`python single_inference.py --help`

## train monster on diachronic data
`python train_monster.py --help`

## run evaluations on full datasets
`python evaluate_all.py --help`

`python evaluate_all_aerial.py --help`

## scripts and other utilities
In `scripts` you will find many scripts and utilitties that were developed in this project, such as: stereo rectification scripts, sift matches computation, dsm projections, etc.