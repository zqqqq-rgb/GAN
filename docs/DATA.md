# Data prerequisites (no dataset redistribution)

## Cable dataset

The proprietary cable images are not included. Obtain data and the fixed split/selection manifests from the authors through an authorized channel. Required layout includes:

```text
data/cable_defect_classified/<Chinese defect class>/<image>
data/cable_leakfix_train/<Chinese defect class>/<training image>
data/yolo_dataset/images/all_real/
data/yolo_dataset/labels/all_real/
data/yolo_dataset/splits_leakfix/
data/yolo_dataset/images/leakfix/<candidate pool>/
```

The four Chinese ImageFolder class names are 凹点、划痕、形变、鼓包. `leakfix_prepare.py` prepares train-only ImageFolder inputs and frozen real-data lists from the authorized classified data. It replaces its derived training folder. Its `train.txt` must be provided as `train_baseline.txt` in the same split directory for the final detector pipeline. Preserve the archived membership, not just the total 193/48 counts.

Labels originate from the rule function in `build_dataset.py` and the documented human review; rule repeatability does not establish physical ground-truth accuracy. Do not run the historical full `build_dataset.py` workflow to create a new paper split.

## NEU-DET

Obtain the designated NEU-DET version under its own access conditions. The raw VOC structure is expected under `data/neu_det/NEU-DET/{train,validation}/{images,annotations}`. `neu_det_prepare.py` is the original conversion helper; its header's 1440 training count refers to the original export, not the final audited 1439-image membership.

Restore the **audited** 1439/360 `neu_train.txt` / `neu_val.txt` lists. Do not assume running the historical converter alone restores the final de-duplication decision. Strict Q3 pipelines check the counts and disjoint membership and expect the archived Real-only seed42 teacher `last.pt` at their configured location.

## GC10-DET

Use the Dataset Ninja Supervisely export identified in the paper, not an arbitrary revised GC10 download. Place the authorized raw export under `data/gc10_det_raw/extracted/ds`. `gc10_prepare.py` audits images/rectangles and constructs the fixed 1604/688 split. The audited export has 2300 originals with eight no-box images excluded. Verify counts and membership against the archived manifest before training.

Run the stages in order only when prerequisites are available:

```bash
python implementations/yolov8_det/scripts/gc10_prepare.py
python implementations/yolov8_det/scripts/gc10_external_pipeline.py all
python implementations/yolov8_det/scripts/gc10_copypaste_pipeline.py all
python implementations/wgan_gp/prepare_gc10_gan.py
```

GAN training/generation follows with the GC10 train-only path, seed42, 500 epochs and batch4. Generate ten x 500 candidates using `cgenerate_images_clip.py --dataset gc10 --by_class --n_per_class 500 --batch_size 4 --seed 20260922` and the completed GC10 checkpoint. Only then launch `gc10_gan_external_pipeline.py all` to label, select and train. These are expensive GPU stages, not installation tests.

## Moving archives to another machine

Archived split lists often contain absolute paths. Remap only their directory prefix to the authorized local dataset root; preserve list order, duplicate entries, image identity and all sampling seeds. Real-resampling lists intentionally contain duplicates. Path sorting can differ across systems; regenerate neither frozen selections nor names silently.

No data license, annotation-accuracy guarantee or raw-image distribution permission is conveyed by publishing the code.
