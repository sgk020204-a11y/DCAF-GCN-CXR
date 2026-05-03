# Upload checklist

Before uploading to GitHub, confirm the following:

- [x] Absolute local paths in `newtrain.py` were replaced with relative paths.
- [x] Large checkpoint files are not included in the repository.
- [x] `checkpoints/README.md` explains where to place large checkpoints.
- [x] Raw medical images are ignored by `.gitignore`.
- [x] Annotation JSON files are included.
- [x] Label graph and word embedding files are included under `graph/`.


Important: upload the files in this directory to the repository root. Do not upload a nested `DCAF-GCN-CXR/` folder into the GitHub repository.

Recommended GitHub upload command:

```bash
git init
git branch -M main
git remote add origin https://github.com/sgk020204-a11y/DCAF-GCN-CXR.git
git add .
git commit -m "Initial public release"
git push -u origin main
```
