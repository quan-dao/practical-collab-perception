'''
src: https://github.com/weiyithu/LiDAR-Distillation
'''
import numpy as np
import torch
from fast_pytorch_kmeans import KMeans


def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)
    phi_ = (phi_ / np.pi) * 180

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)
    phi = (phi / np.pi) * 180

    phi[phi_ < 0] = 360 - phi[phi_ < 0]
    phi[phi == 360] = 0

    return theta, phi


def beam_label(theta, beam):
    estimator = KMeans(n_clusters=beam)
    res = estimator.fit_predict(theta.reshape(-1, 1))
    label = estimator.labels_
    centroids = estimator.cluster_centers_
    return label, centroids[:,0]


@torch.no_grad()
def beam_label_gpu(theta, beam, use_cuda=True):
    kmeans = KMeans(n_clusters=beam, mode='euclidean')
    if use_cuda:
        label = kmeans.fit_predict(torch.from_numpy(theta).unsqueeze(1).float().cuda()).cpu().numpy()
    else:
        label = kmeans.fit_predict(torch.from_numpy(theta).unsqueeze(1).float()).cpu().numpy()
    centroids = kmeans.centroids.reshape(-1).cpu().numpy()
    return label, centroids


def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask

