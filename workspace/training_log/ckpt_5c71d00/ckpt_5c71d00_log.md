# Commit: 5c71d00

# Cfg: pointpillar_jr_corr_withmap_teacher.yaml

# Features:
    * [base] original PointPillar 
    * [mod] Backbone2D -> SCConvBackbone2dStride (*from AFDetV2*)
    * [new] Data 
        * use HD Map: 
            * binary channels (drivable area, ped crossing, walkway, carpark)
            * float channel - lane direction (*new*)
        * combine HD Map with Ground Truth Sampling (*new*)
    * [mod] DenseHead -> calibrate classification score using predicted IoU (*from AFDetV2*)
    * [new] Point cloud correction (compared to our result submitted to IV'23)
        * @Point Head
            * distilling features of the local group which a foreground point belongs to
        * @correct_bev_image
            * distilling BEV image of the oracle model which is trained on point cloud corrected by ground truth 3d flow

# Ablation study

All models are trained on 1/4th of training set and evaluated on the entire validation set of NuScenes.

Recall results submitted to IV'23
| Module | mAP (10 classes) |
| --- | --- |
| PointPillar | 39.0 |
| + corrector | +3.4 |

Current results (convention from top to bottom, module are cumulatively summed)
| Module | mAP (10 classes) |
| --- | --- |
| PointPillar | 39.0 |
| + SCConvBackbone2dStride | +14.3 |
| + HDMap | +0.4 |
| + Cls score calib | +0.6 |
| + **new** corrector | +1.16 |

Comparison between two correctors on PointPillar with the same backbone2D (BaseBEVBackbone - weaker one)
| Module | mAP (10 classes) |
| --- | --- |
| PointPillar | 39.0 |
| + corrector | +3.4 |
| + **new** corrector (with HDMap, no cls calib) | +6.0 |

# 3D Flow evaluation
Our models are trained on 1/4th of training set while baselines are trained on full. The evaluation is carried on the entire validation set.

| Module | EPE ⬇ | ACC_S ⬆ | ACC_R ⬆ | R_Outliers ⬇ |
| --- | --- | --- | --- | --- |
| FLOT | 1.216 | 3.0 | 10.3 | 63.9 |
| NSFPrior | 0.707 | 19.3 | 37.8 | 32.0 |
| PPWC | 0.661 | 7.6 | 24.2 | 31.9 |
| WsRSF | 0.539 | 17.9 | 37.4 | 22.9 |
| PCAcc | **0.301** | 26.6 | 53.4 | 12.1 |
| *corrector* | 0.547 | 14.5 | 26.2 | 36.9 |
| *new corrector* | 0.616 | **46.4** | **66.6** | **6.8** |
