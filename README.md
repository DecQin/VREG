# V-REG supplementary results

## Generation quality on nuScenes

| Method | FID ↓ | FVD ↓ |
| :--- | ---: | ---: |
| MaskGWM (multi-view) | 8.90 | 65.40 |
| UniMLVG | **5.80** | **36.10** |
| MoVieDrive | — | 46.80 |
| V-REG (submitted) | 8.98 | 98.75 |
| V-REG (corrected inference) | 7.59 | 43.38 |

## Scene controllability on nuScenes

| Method | Synthesis resolution | 3D Detection mAP ↑ | BEV Segmentation mIoU ↑ |
| :--- | ---: | ---: | ---: |
| MagicDrive | 224×400 | 9.70 | 15.60 |
| MagicDrive-V2 | 224×400 | 11.50 | 17.40 |
| MoVieDrive | 512×256 | 22.70 | **35.80** |
| V-REG (ours) | 448×800 | **22.74** | 35.31 |
