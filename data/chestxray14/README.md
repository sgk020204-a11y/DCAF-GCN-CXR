# ChestX-ray14 data folder

This folder contains annotation and label-mapping files only. Raw medical images are not redistributed.

Please download ChestX-ray14 from the official NIH source and place images as follows:

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

The class order is defined in `category_aligned.json` and is aligned with `graph/Xray_adj.pkl` and `graph/word14.pkl`.
