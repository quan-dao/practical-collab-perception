# pointpillar_jr_nomap

## commit: a9d24af
### Features
* Backbone2d modified from AFDetV2 (to have stride 4)
* Predict velocity as well
* NO corrector

### NuScenes mAP (nusc style) & NDS
* num epochs: 20
    * mAP: 0.5330
    * NDS: 0.6037
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_nomap/eval_1224180.err)

* num epochs 25:
    * mAP: 0.5407
    * NDS: 0.6120
    * peak at epoch 24
    * [details](../raw_log/pointpillar_jr_nomap/eval_1260463.err)

## commit: 3a59638
### Features:
* base features := commit `a9d24af`
* correct formula for **calibrate class score** using
    * map predicted iou from [-1, 1] to [0, 1] first

### NuScenes mAP (nusc style) & NDS
* num epochs 25 - alpha = 0.1:
    * mAP: 0.5437
    * NDS: 0.6141
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1266066.err)

* num epochs 25 - alpha = 0.2:
    * mAP: 0.5455
    * NDS: 0.6154
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1268025.err)

* num epochs 25 - alpha = 0.3:
    * mAP: 0.5465
    * NDS: 0.6160
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1285782.err)

* num epochs 25 - alpha = 0.4:
    * mAP: 0.5463
    * NDS: 0.6164
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1314953.err)

* num epochs 25 - alpha = 0.5:
    * mAP: 0.5445
    * NDS: 0.6159
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1315625.err)

* num epochs 25 - alpha = 0.6:
    * mAP: 0.5396
    * NDS: 0.6138
    * peak at epoch 25
    * [details](../raw_log/pointpillar_jr_nomap/eval_1315840.err)


# pointpillar_jr_withmap

## commit: faa49f4
### Features:
* base features := commit `3a59638`
* [new] interpolate points' map features (4 binary, 1 lane direction) for map images cached in hard disk
* map cfg: 
    * range: [-51.2, -51.2, -5, 51.2, 51.2, 3.]
    * resolution: 0.2

### NuScenes mAP (nusc style) & NDS
* num epochs: 20
    * mAP: 0.5360
    * NDS: 0.5980
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_withmap/eval_1481004.err)

## commit: eae1cca
### Features:
* base features := commit `faa49f4`
* [new] increase map resolution to 0.1

### NuScenes mAP (nusc style) & NDS
* num epochs: 20
    * mAP: 0.5369
    * NDS: 0.6013
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_withmap/eval_1581859.err)

* num epochs: 20 - calib alpha 0.3
    * mAP: 0.5436
    * NDS: 0.6057
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_withmap/eval_1584877.err)

* num epochs: 25 - calib alpha 0.3
    * mAP: 0.5495
    * NDS: 0.6148
    * peak at epoch 24
    * [details](../raw_log/pointpillar_jr_withmap/train_1634100.err)


## commit: 9ba653b
### Features
* base features := commit `eae1cca`
* [new] add `OracleCorrector` module

### NuScenes mAP (nusc style) & NDS
* num epochs: 20 - no calib
    * mAP: 0.5422
    * NDS: 0.5273 (no shadow -> bad velo estim)
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_withmap/eval_1830535.err)

* num epochs: 20 - calib alpha 0.3
    * mAP: 0.5482
    * NDS: 0.5314 (no shadow -> bad velo estim)
    * peak at epoch 20
    * [details](../raw_log/pointpillar_jr_withmap/eval_1832359.err)
