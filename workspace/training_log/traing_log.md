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
