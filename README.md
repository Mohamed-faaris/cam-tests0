# cam-test

Camera calibration, depth, and computer-vision experiments.

## Package instance segmentation

This repository includes an Ultralytics YOLO instance-segmentation workflow
for the [Package-Seg dataset](https://docs.ultralytics.com/datasets/segment/package-seg).
The dataset contains 2,197 images (1,920 train, 188 validation, 89 test) and
one class: `package`. The dataset archive is downloaded automatically by
Ultralytics when training starts.

### 1. Install dependencies

```bash
uv sync
```

### 2. Train

Run from the repository root:

```bash
uv run python -m cam_test.package_seg train
```

Defaults: YOLO26n-seg, 100 epochs, 640px images, batch size 8. Override them
when needed:

```bash
uv run python -m cam_test.package_seg train --epochs 150 --batch 4 --device 0
```

The best checkpoint is written to:

```
runs/segment/package-seg/weights/best.pt
```

### 3. Validate

```bash
uv run python -m cam_test.package_seg val
```

### 4. Segment an image or video

```bash
uv run python -m cam_test.package_seg predict --source path/to/image.jpg
```

Results are saved under `runs/predict/package-seg/`. Add `--show` to display
the prediction window.

### 5. Segment a live camera stream

```bash
uv run python -m cam_test.package_seg live --camera 0
```

Press `q` or `Esc` to close the camera window. Use `--camera 1` to select
another camera and `--weights /path/to/best.pt` to use a different checkpoint.

**Important:** the pretrained checkpoint is used for training initialization.
Live inference uses the fine-tuned `best.pt` checkpoint because the generic
pretrained model does not contain the Package-Seg class. This dataset has one
broad `package` class; it does not distinguish individual product types.
