# Practical Collaborative Percpetion

This is the official code release for **Practical Collaborative Perception: A Framework for Asynchronous and Multi-Agent 3D Object Detection**

We propose a new framework for collaborative 3D object detection named *lately* fusion that takes objects detected by other connected agents, 
including connected autonomous vehicles (CAV) and intelligent roadside units (IRSU), and fuse them with the raw point cloud of the ego vehicle.
Our method is the combination of *late* fusion that exchanges connected agents' output (i.e., detected objects) and ear*ly* fusion that fuses 
exchanged information at the input of the ego vehicle, thus its name *lately*.

<p align="center">
  <img src="docs/media/lately_fusion.png">
</p>

The fusion at the input of the ego vehicle is done using [MoDAR](https://arxiv.org/abs/2306.03206).
In details, objects detected by other connected agents are interpreted into 3D points with addition features (e.g., objects' dim, class, 
confident score).
Then, these interpreted 3D points, which are referred to as *MoDAR* points, are transformed to the ego vehicle's frame where they are concatenated
with the raw point cloud obtained by the ego vehicle.

To account for the asynchronization among connected agents, each detected object is assigned a predicted velocity so that they can be propagated
from the timestep when it was detected to the timestep that is queried by the ego vehicle.
This velocity prediction is based on scene flow predicted by our previous work called [Aligner](https://arxiv.org/abs/2305.02909).

Compared to the other collaboration frameworks, including early fusion, late fusion, mid fusion, our method has the following advantages:
- achieving competitive performance with respect to early fusion (99% of early fusion's mAP in V2X-Sim dataset)
- not requiring that connected agents are in sync
- consuming as much bandwidth as late fusion
- requiring minimal changes made to single-agent detection models
- straightforwardly supporting heterogeneous networks of detection models 

The comparion of our method against other collaboration methods in terms of precision (measured by mAP) and bandwidth consumption using 
the V2X-Sim dataset is shown in the Table below
|Method           | Sync | Async | Bandwidth Usage (MB) | Weight |
| -----           | :-----:| :--------: | :-----: | :-----: | 
| [None](tools/cfgs/v2x_sim_models/v2x_pointpillar_basic_car.yaml)  | 52.84  | -      | 0      | [pillar_car](todo) |
| [Late](tools/cfgs/v2x_sim_models/v2x_late_fusion.yaml)            | 70.48  | 67.80  | 0.01   | - |
| [Mid-DiscoNet](tools/cfgs/v2x_sim_models/v2x_pointpillar_disco.yaml)            | 78.70  | 73.10  | 25.16   | [pillar_mid_sync](todo-v2x_pointpillar_disco.pth) |
| [Early](tools/cfgs/v2x_sim_models/v2x_pointpillar_basic_ego_early.yaml)            | 78.10  | 77.30  | 33.95   | [pillar_early_sync](todo) |
| [Ours](tools/cfgs/v2x_sim_models/v2x_pointpillar_basic_ego.yaml)            | 79.20  | 76.72  | 0.02   | [pillar_colab_async](todo) |
