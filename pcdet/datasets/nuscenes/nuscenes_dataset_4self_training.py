import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import pickle

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from workspace.nuscenes_temporal_utils import get_sweeps
from workspace.traj_discovery import TrajectoriesManager


class NuScenesDataset4SelfTraining(NuScenesDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        disco_classes_name = self.dataset_cfg.DISCOVERED_DYNAMIC_CLASSES 
        idx_teacher_round = self.dataset_cfg.PSEUDO_LABELS_BY_ROUND_IDX

        self.pseudo_labels_root = self.dataset.root_path / Path(f"database_round{idx_teacher_round}_pseudo_labels")
        

        self.dict_sampleToken_2_labelsPath, self.dict_class_2_labelsPath = self.parse_pseudo_labels_root()
        dict_cls_2_score_thresh = self.filter_database_pseudo_labels_by_score_(self.dataset_cfg.PSEUDO_DATABASE_CUTOFF_PERCENTAGE)
        
        self.psdlabels_sampling_info: Dict[str, Dict] = dict()
        for cls_name, list_paths in self.dict_class_2_labelsPath.items():
            self.psdlabels_sampling_info[cls_name] = {
                'num_to_sample': dataset_cfg.PSEUDO_LABELS_NUM_TO_SAMPLE,
                'pointer': 0,
                'indices': np.random.permutation(len(list_paths)),
                'num_pseudo_labels': len(list_paths)
            }

    def parse_pseudo_labels_root(self) -> Tuple[Dict[str, np.ndarray]]:
        """
        Returns:
            dict_sampleToken_2_labelsPath: for loading pseudo-labels of each sample given sample token
            dict_class_2_labelsPath: for database sampling
        """
        dict_sampleToken_2_labelsPath = dict()
        dict_class_2_labelsPath = dict([(cls_name, list()) for cls_name in self.dataset_cfg.DISCOVERED_DYNAMIC_CLASSES])

        for path_ in self.pseudo_labels_root.glob("*.pkl"):
            parts = str(path_.parts[-1]).split('_')
            sample_token = parts[0]
            
            class_name = parts[-1].split('.')[0]
            assert class_name in self.dataset_cfg.DISCOVERED_DYNAMIC_CLASSES, f"encounter unknow class {class_name} in pseudo_label of {sample_token}"
            
            dict_class_2_labelsPath[class_name].append(path_)

            if sample_token not in dict_sampleToken_2_labelsPath:
                dict_sampleToken_2_labelsPath[sample_token] = [path_,]
            else:
                dict_sampleToken_2_labelsPath[sample_token].append(path_)

        # convert values of dict_class_2_labelsPath to np.ndarray for convenient indexing
        for cls_name, list_paths in dict_class_2_labelsPath.items():
            dict_class_2_labelsPath[cls_name] = np.array(list_paths)

        return dict_sampleToken_2_labelsPath, dict_class_2_labelsPath
    
    def filter_database_pseudo_labels_by_score_(self, filter_percentage: float = 0.2) -> Dict[str, float]:
        """
        Mutate self.dict_class_2_labelsPath
        Filter bottom {filter_percentage}% of pseudo-labels in the database that is used for gt_sampling

        Args:
            filter_percentage
        
        Returns:
            dict_cls_2_score_thresh
        """

        assert 0 < filter_percentage < 1
        
        dict_cls_2_score_thresh = dict()
        
        for cls_name, array_paths in self.dict_class_2_labelsPath.items():
            all_scores = list()
            for path_ in array_paths:
                with open(path_, 'rb') as f:
                    info = pickle.load(f)
                box = info['box_in_lidar']
                all_scores.append(box[-1])
            
            sorted_ids = np.argsort(all_scores)  # ascending order
            
            # remove bottom {filter_percentage}%
            idx_cutoff = int(filter_percentage * sorted_ids.shape[0])
            kept_ids = sorted_ids[idx_cutoff + 1:]
            self.dict_class_2_labelsPath[cls_name] = array_paths[kept_ids]
            
            # store the cutoff score
            dict_cls_2_score_thresh[cls_name] = all_scores[idx_cutoff]

        return dict_cls_2_score_thresh


    def database_take_pseudo_samples(self, existing_boxes: np.ndarray) -> Tuple[np.ndarray]:
        """
        Take sample from database of pseudo labels

        Args:
            existing_boxes (N, 10) - box-7, sweep_idx, inst_idx, cls_idx. Here existing boxes are pseudo-labels of this sample

        Returns:
            points: (N_pts, 5 + 2) - 5-point, sweep_idx, instance_idx
            boxes: (N_box, 10) - 7-box, sweep_idx (:= 10), instance_idx, class_idx
        """
        def _sample_with_fixed_number(info: dict, list_paths: List[Path]) -> List[Path]:
            if info['pointer'] + info['num_to_sample'] >= info['num_pseudo_labels']:
                info['indices'] = np.random.permutation(info['num_pseudo_labels'])
                info['pointer'] = 0
            
            pointer, num_to_sample = info['pointer'], info['num_to_sample']
            sampled_paths = [list_paths[idx] for idx in info['indices'][pointer: pointer + num_to_sample]]
            
            # update pointer
            info['pointer'] += num_to_sample

            return sampled_paths
        

        existing_max_inst_idx = existing_boxes[:, -2].max()

        for cls_name, list_path_psdlabels in self.dict_class_2_psdlabels.items():
            sampled_paths = _sample_with_fixed_number(self.psdlabels_sampling_info[cls_name], 
                                                      list_path_psdlabels)
            points, boxes = list(), list()
            for _idx, _path in enumerate(sampled_paths):
                with open(_path, 'rb') as f:
                    info = pickle.load(f)
                
                pts = info['points']  # (N, 5 + 2) pt-5, sweep_idx, inst_idx  
                box = info['box']  # (7 + 3) box-7, sweep_idx, inst_idx, cls_idx
                
                # overwrite instance_idx of pts & box
                pts[:, -1] = existing_max_inst_idx + _idx
                box[-2] = existing_max_inst_idx + _idx
                # make sure box's cls_idx is coherent
                assert box[-1] == 0 if cls_name == 'car' else 1, f"class_idx := {box[-1]} is invalid"

                # store
                points.append(pts)
                boxes.append(box)

            points = np.concatenate(points)  # (N_pts, 5 + 2) - pt-5, sweep_idx, inst_idx
            boxes = np.stack(boxes, axis=0)  # (N_box, 7 + 3) - box-7, sweep_idx, inst_idx, cls_idx

        # filtering
        points, boxes = TrajectoriesManager.filter_sampled_boxes_by_iou_with_existing(points, boxes, existing_boxes)
        points, boxes = TrajectoriesManager.filter_sampled_boxes_by_iou_with_themselves(points, boxes)
        return points, boxes


