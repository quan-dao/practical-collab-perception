import numpy as np
import pickle
import matplotlib.pyplot as plt

from workspace.o3d_visualization import print_dict


def main():
    classes_name = ['car', 'car+cyc+motor', 'ped']
    trajs_info_path = [f'../workspace/artifact/good/cluster_info_{name}_15sweeps.pkl' for name in classes_name]
    trajs_color = ['r', 'g', 'b']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for info_path, cls_name, cls_color in zip(trajs_info_path, classes_name, trajs_color):
        with open(info_path, 'rb') as f:
            traj_info = pickle.load(f)
        print_dict(traj_info, f'{cls_name}_traj_info')
        embeddings = traj_info['cluster_top_members_static_embed']  # (N, 3)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=cls_color)

    plt.show()



if __name__ == '__main__':
    main()

