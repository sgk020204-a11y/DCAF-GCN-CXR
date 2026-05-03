# DCAF-GCN-CXR

Official implementation for the manuscript:

**Graph guided multiscale cross attention for multilabel chest X ray classification**

This repository contains the training code for a ViT DenseNet hybrid model with multiscale bidirectional dual cross attention fusion and graph guided label reasoning for multilabel chest X ray classification.

## Main components

- ViT branch and DenseNet-121 branch for heterogeneous visual encoding
- C3/C4 multiscale DCAF fusion
- ML-GCN style label graph reasoning
- Graph guided Transformer decoder
- Composite multilabel loss including ASL, BCE, graph smoothing and pairwise AUC surrogate
- ChestX-ray14 style multilabel data loader

## Repository structure

```text
DCAF-GCN-CXR/
├── newtrain.py                 # main training entry
├── coco.py                     # dataset loader, retained from ML-GCN style naming
├── models2.py                  # DCAF-GCN-CXR model definition
├── engine2.py                  # training and evaluation engine
├── losses.py                   # composite multilabel losses
├── util.py                     # data transforms and metric utilities
├── layers.py                   # auxiliary GCN/backbone layers
├── contmix.py                  # optional ContMix/OverLoCK module
├── utility/                    # helper functions
├── graph/
│   ├── Xray_adj.pkl            # label graph statistics
│   └── word14.pkl              # 14-label word embeddings
├── data/chestxray14/           # annotation json files only
├── checkpoints/                # place pretrained checkpoints here manually
└── scripts/                    # example launch scripts
```

## Environment

Recommended environment:

```bash
conda create -n dcaf_gcn_cxr python=3.9 -y
conda activate dcaf_gcn_cxr
pip install -r requirements.txt
```

`natten` is optional. If it is not installed, `contmix.py` uses a pure PyTorch fallback implementation, which is slower but avoids a hard dependency.

## Data preparation

The raw ChestX-ray14 images are **not redistributed** in this repository. Please download the dataset from the official NIH source and arrange files as follows:

```text
data/chestxray14/
├── train2014/
│   ├── 00000001_000.png
│   └── ...
├── val2014/
│   ├── 00000003_000.png
│   └── ...
├── train_anno.json
├── val_anno.json
├── category.json
├── train_anno_aligned.json
├── val_anno_aligned.json
└── category_aligned.json
```

The annotation JSON files and label mapping files are already included. The image folders should be placed manually.

## Pretrained checkpoints

The following large checkpoint files are **not included** in this repository:

```text
checkpoints/vit-b_CXR_0.5M_mae.pth
checkpoints/densenet121_imagenet_torchvision.pth.tar
```

To reproduce the pretrained setting used in the manuscript, download or copy these files manually and place them under `checkpoints/`.

If these checkpoint arguments are left empty, the code will run without the manuscript-specific CXR/MAE checkpoint. This is useful for code verification, but it is not the exact pretrained setting reported in the paper.

## Training

### Run with manuscript pretrained checkpoints

```bash
python newtrain.py \
  --data-path ./data/chestxray14 \
  --adj-file ./graph/Xray_adj.pkl \
  --word-emb ./graph/word14.pkl \
  --mae-ckpt ./checkpoints/vit-b_CXR_0.5M_mae.pth \
  --densenet-ckpt ./checkpoints/densenet121_imagenet_torchvision.pth.tar \
  --save-model-path ./outputs/checkpoints/chestxray/ \
  --device-ids 0
```

### Run without large private checkpoints

```bash
python newtrain.py \
  --data-path ./data/chestxray14 \
  --adj-file ./graph/Xray_adj.pkl \
  --word-emb ./graph/word14.pkl \
  --mae-ckpt "" \
  --densenet-ckpt "" \
  --save-model-path ./outputs/checkpoints/chestxray/ \
  --device-ids 0
```

This second command is mainly for checking code integrity when the large checkpoints are unavailable.

## Evaluation

After obtaining a checkpoint, run:

```bash
python newtrain.py \
  --data-path ./data/chestxray14 \
  --adj-file ./graph/Xray_adj.pkl \
  --word-emb ./graph/word14.pkl \
  --resume ./outputs/checkpoints/chestxray/model_best.pth.tar \
  --evaluate \
  --device-ids 0
```

## Notes

- The dataset class is named `COCO2014` for compatibility with the original ML-GCN style codebase, but in this repository it reads the aligned ChestX-ray14 annotations.
- The repository does not redistribute raw medical images or large pretrained checkpoints.
- The default paths in `newtrain.py` have been changed to relative paths for public release.

## Citation

If this repository is useful for your research, please cite the corresponding manuscript after publication.
