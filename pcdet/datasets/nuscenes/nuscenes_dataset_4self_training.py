import numpy as np
import torch
import torch_scatter
from typing import List, Dict, Tuple
from pathlib import Path
import pickle
import copy

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from workspace.nuscenes_temporal_utils import get_sweeps
from workspace.traj_discovery import TrajectoriesManager
from workspace.box_fusion_utils import label_fusion


class NuScenesDataset4SelfTraining(NuScenesDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        assert self.dataset_cfg.NUM_SWEEPS_TO_BUILD_DATABASE == self.dataset_cfg.MAX_SWEEPS
        idx_teacher_round = self.dataset_cfg.PSEUDO_LABELS_BY_ROUND_IDX
        self.verbose = self.dataset_cfg.get('VERBOSE', False)

        self.pseudo_labels_root = self.root_path / Path(f"database_round{idx_teacher_round}_pseudo_labels")
        
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
        

        existing_max_inst_idx = existing_boxes[:, -2].max() + 1 if existing_boxes.shape[0] > 0 else 0

        points, boxes = list(), list()
        for cls_name, list_path_psdlabels in self.dict_class_2_labelsPath.items():
            sampled_paths = _sample_with_fixed_number(self.psdlabels_sampling_info[cls_name], 
                                                      list_path_psdlabels)
            for _idx, _path in enumerate(sampled_paths):
                with open(_path, 'rb') as f:
                    info = pickle.load(f)
                
                pts = info['points_in_lidar']  # (N, 5 + 2) pt-5, sweep_idx, inst_idx  
                box = info['box_in_lidar']  # (7 + 3 + 1) box-7, sweep_idx, inst_idx, cls_idx, score
                
                # overwrite instance_idx of pts & box
                pts[:, -1] = existing_max_inst_idx + _idx
                
                box[-4] = pts[:, -2].max()  # sweep idx
                box[-3] = existing_max_inst_idx + _idx
                # make sure box's cls_idx is coherent
                assert box[-2] == 0 if cls_name == 'car' else 1, f"class_idx := {box[-2]} is invalid"

                # store
                points.append(pts)
                boxes.append(box)
            
            # update existing_max_inst_idx for the next class
            existing_max_inst_idx += len(sampled_paths)

        points = np.concatenate(points)  # (N_pts, 5 + 2) - pt-5, sweep_idx, inst_idx
        boxes = np.stack(boxes, axis=0)  # (N_box, 7 + 3) - box-7, sweep_idx, inst_idx, cls_idx

        # filtering
        if existing_boxes.shape[0] > 0:
            points, boxes = TrajectoriesManager.filter_sampled_boxes_by_iou_with_existing(points, boxes, 
                                                                                          existing_boxes)
        points, boxes = TrajectoriesManager.filter_sampled_boxes_by_iou_with_themselves(points, boxes)
        return points, boxes

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        num_sweeps = self.dataset_cfg.NUM_SWEEPS_TO_BUILD_DATABASE
        # accumulate pointclouds for 10 sweeps untils this sample
        
        points, glob_se3_current = get_sweeps(self.nusc, info['token'], num_sweeps)
        # points: (N_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx (place holder, := -1)]
        points_max_sweep_index = points[:, -2].max()

        if self.training:
            # --------------------------------
            # load pseudo-labels
            # --------------------------------
            if info['token'] in self.dict_sampleToken_2_labelsPath:
                pseudo_boxes = list()
                for path_ in self.dict_sampleToken_2_labelsPath[info['token']]:
                    with open(path_, 'rb') as f:
                        pseudo_label_info = pickle.load(f)
                    
                    pseudo_boxes.append(pseudo_label_info['box_in_lidar'])  # (7 + 3 + 1) - box-7, sweep_idx, inst_idx, class_idx, score
                
                pseudo_boxes = np.stack(pseudo_boxes, axis=0)  # (N_pbox, 7 + 3 + 1) - box-7, sweep_idx, inst_idx, class_idx, score
                # filter ped
                if pseudo_boxes.shape[0] > 0:
                    mask_valid_ped = np.logical_and(pseudo_boxes[:, -2].astype(int) == 1, pseudo_boxes[:, -1] > 0.3)
                    mask_keep = np.logical_or(pseudo_boxes[:, -2].astype(int) == 0, mask_valid_ped)
                    pseudo_boxes = pseudo_boxes[mask_keep]
            else:
                pseudo_boxes = np.zeros((0, 11))

            if self.verbose:
                print(f"sample {info['token'][-5:]} | pseudo_boxes: {pseudo_boxes.shape}")

            # pad fused_boxes with sweep_idx and instance_idx
            if pseudo_boxes.shape[0] > 0:
                final_pseudo_boxes = np.concatenate([
                    pseudo_boxes[:, :7],  # box-7
                    np.zeros((pseudo_boxes.shape[0], 1)) + points_max_sweep_index,  # sweep_idx
                    np.arange(pseudo_boxes.shape[0]).reshape(-1, 1),  # instance_idx
                    pseudo_boxes[:, -2].reshape(-1, 1),  # class
                ], axis=1)  # (N_fused, 7 + 3) - box-7, sweep_idx, instance_idx, class_idx

                # re-establish point-to-box corr
                box_idx_of_points = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, :3]).float().unsqueeze(0).cuda(), 
                    torch.from_numpy(final_pseudo_boxes[:, :7]).float().unsqueeze(0).cuda(),
                ).long().squeeze(0).cpu().numpy()  # (N_pts,) to index into (N_b)
                points[:, -1] = final_pseudo_boxes[box_idx_of_points, -2]
                points[box_idx_of_points < 0, -1] = -1
            else:
                final_pseudo_boxes = np.zeros((0, 10))
            
            if self.verbose:
                print(f"sample {info['token'][-5:]} | fused_pseudo_disco_boxes: {final_pseudo_boxes.shape}")

            # --------------------------------
            # sample pseudo-labels
            # --------------------------------
            sampled_pseudo_pts, sampled_pseudo_boxes = self.database_take_pseudo_samples(final_pseudo_boxes)
            # sampled_pseudo_pts: (N_pts, 1 + 5 + 2) - batch_idx, point-5, sweep_idx, inst_idx
            # sampled_pseudo_boxes: (N_box, 7 + 3 + 1) - 7-box, sweep_idx (:= 10), instance_idx, class_idx, score
            if self.verbose:
                print(f"sample {info['token'][-5:]} | sampled_pseudo_boxes: {sampled_pseudo_boxes.shape}")

            # merge
            points = np.concatenate([points, sampled_pseudo_pts[:, 1:]])
            boxes = np.concatenate([final_pseudo_boxes, sampled_pseudo_boxes[:, :10]])  # (N_box, 10) - 7-box, sweep_idx, instance_idx, class_idx

            # --------------------------------
            # sample discovered dynamic trajectories
            # --------------------------------
            num_existing_instances = boxes[:, -2].max() + 1 if boxes.shape[0] > 0 else 0
            sampled_disco_pts, sampled_disco_boxes = list(), list()
            for cls_name in self.dataset_cfg.DISCOVERED_DYNAMIC_CLASSES:
                _pts, _boxes = self.traj_manager.sample_disco_database(cls_name, is_dyn=True, 
                                                                       num_existing_instances=num_existing_instances)
                # _boxes: (N_boxes, 10) - box-7, sweep_idx, instance_idx, class_idx
                sampled_disco_pts.append(_pts)
                sampled_disco_boxes.append(_boxes)
                num_existing_instances = _boxes[:, -2].max() + 1
            
            sampled_disco_pts = np.concatenate(sampled_disco_pts)  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
            sampled_disco_boxes = np.concatenate(sampled_disco_boxes)  # (N_boxes, 10) - box-7, sweep_idx, instance_idx, class_idx

            # filter
            if boxes.shape[0] > 0:
                sampled_disco_pts, sampled_disco_boxes = self.traj_manager.filter_sampled_boxes_by_iou_with_existing(
                    sampled_disco_pts, sampled_disco_boxes, boxes)
            sampled_disco_pts, sampled_disco_boxes = self.traj_manager.filter_sampled_boxes_by_iou_with_themselves(
                sampled_disco_pts, sampled_disco_boxes
            )
            # take last disco_box of each sampled disco traj
            unique_insta_idx, inv_unique_insta_idx = torch.unique(
                torch.from_numpy(sampled_disco_boxes[:, -2]).long(), return_inverse=True)
            per_inst_max_sweepidx, per_inst_idx_of_max_sweepidx = torch_scatter.scatter_max(
                torch.from_numpy(sampled_disco_boxes[:, -3]).long(), 
                inv_unique_insta_idx, 
                dim=0)
            sampled_disco_boxes = sampled_disco_boxes[per_inst_idx_of_max_sweepidx.numpy()]  # (N_sampled_boxes, 10)
            if self.verbose:
                print(f"sample {info['token'][-5:]} | sampled_disco_boxes: {sampled_disco_boxes.shape}")
            
            # ---
            # assemble 
            points = np.concatenate([points, sampled_disco_pts])
            gt_boxes = np.concatenate([boxes, sampled_disco_boxes])[:, :7]
            gt_names = self._uda_class_names[np.concatenate([boxes[:, -1], sampled_disco_boxes[:, -1]]).astype(int)]
            gt_inst_idx = np.concatenate([boxes[:, -2], sampled_disco_boxes[:, -2]])

        else:
            # not training
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token'],
                         'num_sweeps_target': num_sweeps},
            'gt_boxes': gt_boxes,
            'gt_names': gt_names  # str
        }
        if self.verbose:
            input_dict['metadata']['pseudo_boxes'] = pseudo_boxes
            input_dict['metadata']['sampled_pseudo_boxes'] = sampled_pseudo_boxes
            input_dict['metadata']['gt_inst_idx'] = gt_inst_idx
        
        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        if self.verbose:
            print('----\n')

        return data_dict


