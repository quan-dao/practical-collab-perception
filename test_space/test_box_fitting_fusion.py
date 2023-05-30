import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor, LinearRegression

from workspace.uda_tools_box import BoxFinder
from workspace.box_fusion_utils import kde_fusion


class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


def main():
    points = np.load('artifact/hdbscan_dataset10_cluster31.npy')
    print('points: ', points.shape)
    
    points_sweep_idx = points[:, -2].astype(int)
    unq_sweep_idx  = np.unique(points_sweep_idx)

    box_finder = BoxFinder(return_in_form='box_openpcdet')

    traj_boxes = []
    for _idx, sweep_idx in enumerate(unq_sweep_idx):    
        mask_this_sweep = points_sweep_idx == sweep_idx
        box_bev, mean_z, theta_star = box_finder.fit(points[mask_this_sweep])
        # assembly
        x, y, dx, dy, heading = box_bev
        box3d = np.array([x, y, mean_z, dx, dy, np.max(points[mask_this_sweep, :2]), heading, 1, mask_this_sweep.sum() / float(points.shape[0])])
        traj_boxes.append(box3d)

    traj_boxes = np.stack(traj_boxes, axis=0)
    fused_box = kde_fusion(traj_boxes, src_weights=traj_boxes[:, -1])

    print('traj_boxes:\n', traj_boxes[np.argsort(-traj_boxes[:, -1]), 3:])
    print('---')
    print(fused_box[3: 6])

    # find a curve that fit boxes
    ransac = RANSACRegressor(PolynomialRegression(degree=2), 
                             min_samples=5,
                            #  residual_threshold=.3 * np.std(traj_boxes[:, 1]), 
                             random_state=0)
    ransac.fit(np.expand_dims(traj_boxes[:, 0], axis=1), traj_boxes[:, 1])
    inlier_mask = ransac.inlier_mask_
    pred_y = ransac.predict(np.expand_dims(traj_boxes[:, 0], axis=1))

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    
    sweeps_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_sweep_idx.shape[0]))[:, :3]
    ax.scatter(traj_boxes[:, 0], traj_boxes[:, 1], c=sweeps_color, marker='o')
    ax.scatter(traj_boxes[:, 0], pred_y, c=sweeps_color, marker='x')
    ax.plot(traj_boxes[:, 0], pred_y, 'r-')
    plt.show()
    

if __name__ == '__main__':
    main()

