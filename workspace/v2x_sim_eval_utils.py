import numpy as np
from nuscenes import NuScenes
from typing import List, Dict, Tuple
import operator
from pyquaternion import Quaternion
import os
from tqdm import tqdm
import pickle

from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, filter_eval_boxes

from pcdet.datasets.nuscenes.nuscenes_utils import cls_attr_dist


def transform_det_annos_to_nusc_annos(det_annos: List[Dict], nusc_annos_: Dict) -> None:
    """
    Mutate nusc_annos
    Args:
        det_annos: each dict is
            {   
                'metadata': {
                    'token': sample token
                    'lidar_token'
                }
                'boxes_lidar': (N, 7) - x, y, z, dx, dy, dz, heading | in LiDAR
                'score': (N,)
                'pred_labels': (N,) | int, start from 1
                'name': (N,) str
            }
    """
    seen_lidar_tokens = set()

    for det in det_annos:
        lidar_tk = det['metadata']['lidar_token']

        if lidar_tk not in seen_lidar_tokens:
            seen_lidar_tokens.add(lidar_tk)
        else:
            # print(f'WARNING @ nuscenes_utils.py | see {sample_tk} more than once')
            continue

        boxes_in_lidar = det['boxes_lidar']
        boxes_name = det['name']
        boxes_score = det['score']

        annos = []
        for k  in range(boxes_in_lidar.shape[0]):
            box = boxes_in_lidar[k]  # (7,)
            name = boxes_name[k]
            attr = max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            
            nusc_anno = {
                'sample_token': lidar_tk,  # NOTE: workaround to eval w.r.t sample_data_token
                'translation': box[:3].tolist(),  # in LiDAR
                'size': box[3: 6].tolist(),  # dx, dy, dz
                'rotation': Quaternion(axis=[0, 0, 1], angle=box[6]).elements.tolist(),
                'velocity': [0., 0.],
                'detection_name': name,
                'detection_score': boxes_score[k],
                'attribute_name': attr
            }
            annos.append(nusc_anno)
        
        nusc_annos_['results'][lidar_tk] = annos  # NOTE: workaround to eval w.r.t sample_data_token
    
    return


def load_gt(nusc: NuScenes, eval_split: str, box_cls, dataset_infos: List[Dict], verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    assert box_cls == DetectionBox, 'Error: Invalid box_cls %s!' % box_cls
    # Init.
    attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    _dummy_attribute = attribute_map[list(attribute_map.keys())[0]]

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    
    all_annotations = EvalBoxes()
    for info in tqdm(dataset_infos, total=len(dataset_infos), leave=verbose):
        attribute_name = _dummy_attribute

        gt_boxes = info['gt_boxes']  # (N_gt, 7)
        gt_names = info['gt_names']  # (N_gt,)
        # gt_num_points = info['num_points_in_boxes'].astype(int)  # (N_gt,)
        
        boxes = list()
        for b_idx in range(gt_boxes.shape[0]):
            boxes.append(
                box_cls(
                    sample_token=info['lidar_token'],  # NOTE: workaround to eval w.r.t sample_data_token
                    translation=gt_boxes[b_idx, :3],
                    size=gt_boxes[b_idx, 3: 6],
                    rotation=Quaternion(axis=[0, 0, 1], angle=gt_boxes[b_idx, 6]).elements.tolist(),
                    velocity=[0., 0.],
                    num_pts=10,  # dummy value
                    detection_name=gt_names[b_idx],
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attribute_name
                )
            )
        
        all_annotations.add_boxes(info['lidar_token'], boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def add_dist_to_lidar(eval_boxes: EvalBoxes):
    """
    in this impl of V2XSimDataset, boxes (pred & gt) are already in LiDAR frame

    Args:
        eval_boxes: A set of boxes, either GT or predictions.
    
    Return:
        eval_boxes: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        for box in eval_boxes[sample_token]:
            box.ego_translation = box.translation
    return eval_boxes


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta


class V2XSimDetectionEval(DetectionEval):
    def __init__(self, nusc: NuScenes, config: DetectionConfig, result_path: str, eval_set: str, output_dir: str = None, verbose: bool = True, dataset_infos: List[Dict] = None):
        assert dataset_infos is not None
        self.dataset_infos = dataset_infos

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)

        self.gt_boxes = load_gt(self.nusc, '', DetectionBox, self.dataset_infos, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_dist_to_lidar(self.pred_boxes)
        self.gt_boxes = add_dist_to_lidar(self.gt_boxes)
