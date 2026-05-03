# Checkpoints

Large pretrained checkpoint files are not included in this GitHub repository.

To reproduce the manuscript pretrained setting, place the following files here manually:

```text
vit-b_CXR_0.5M_mae.pth
densenet121_imagenet_torchvision.pth.tar
```

Example training command:

```bash
python newtrain.py \
  --mae-ckpt ./checkpoints/vit-b_CXR_0.5M_mae.pth \
  --densenet-ckpt ./checkpoints/densenet121_imagenet_torchvision.pth.tar
```

If these paths are left empty, the code can still be used for checking the training pipeline, but it will not reproduce the exact pretrained setting reported in the manuscript.
