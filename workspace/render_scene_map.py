import numpy as np
import cv2
from nuscenes import NuScenes
from workspace.nuscenes_map_helper import MapMaker
from copy import deepcopy


def main():
    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    map_maker = MapMaker(nusc, np.array([-81.0, -81.0, -5.0, 81.0, 81.0, 3.0]), 0.075)

    scene = nusc.scene[0]
    sample_tk = scene['first_sample_token']
    while sample_tk != '':
        map_layers = map_maker.get_binary_layers_in_lidar_frame(sample_tk, return_channel_last=True)
        
        res = cv2.resize(map_layers, (512, 512), interpolation=cv2.INTER_CUBIC)
        out = np.ones((res.shape[0], res.shape[1], 3)) * 255 * res[..., [0]]

        # ----------------
        # draw lidar frame
        # ----------------
        rows, cols = out.shape[:2]
        img_center = [int((cols - 1) / 2.0), int((rows - 1) / 2.0)]
        # origin
        cv2.circle(out, img_center, radius=10, color=(255, 0, 0), thickness=-1)
        # x-axis
        img_x = deepcopy(img_center)
        img_x[0] += 25
        cv2.line(out, img_center, img_x, (0, 0, 255), 5)
        # y-axis
        img_y = deepcopy(img_center)
        img_y[1] += 25
        cv2.line(out, img_center, img_y, (0, 255, 0), 5)


        cv2.imshow('map @ lidar', out[::-1])
        if cv2.waitKey(500) & 0xFF == ord('q'):  # press "q" to end the program
            break
        
        # move on
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

