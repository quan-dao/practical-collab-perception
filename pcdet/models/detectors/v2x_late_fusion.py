import numpy as np
import torch
from typing import List, Dict
from .detector3d_template import Detector3DTemplate, model_nms_utils
from workspace.box_fusion_utils import label_fusion


class V2XLateFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.post_process_cfg = model_cfg.POST_PROCESSING

    def forward(self, batch_dict):
        assert not self.training, "there is nothing to train"
        metadata: List[Dict[str, np.ndarray]] = batch_dict['metadata']
        
        final_box_dicts = list()
        for b_idx in range(len(metadata)):
            dict_exchange_boxes = metadata[b_idx]['exchange_boxes']
            
            if self.model_cfg.BOX_FUSION_METHOD == 'nms':
                exchange_boxes = np.concatenate([boxes for _, boxes in dict_exchange_boxes.items() if boxes.shape[0] > 0])
                exchange_boxes = torch.from_numpy(exchange_boxes).float().cuda()  # (N, 7 + 2) - box-7, score, label
                # NOTE: class_idx of boxes & points go from 1
                
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=exchange_boxes[:, -2], box_preds=exchange_boxes[:, :7],
                    nms_config=self.post_process_cfg.NMS_CONFIG,
                    score_thresh=self.post_process_cfg.SCORE_THRESH
                )
                this_pred_dict = {
                    'pred_boxes': exchange_boxes[selected, :7],
                    'pred_scores': selected_scores,
                    'pred_labels': exchange_boxes[selected, -1].long()
                }
            elif self.model_cfg.BOX_FUSION_METHOD == 'kde':
                exchange_boxes = list()
                weights = list()
                for lidar_id, boxes in dict_exchange_boxes.items():
                    if boxes.shape[0] == 0:
                        continue
                    boxes = boxes[:, [0, 1, 2, 3, 4, 5, 6, 8, 7]]  # (N, 7 + 2) - box-7, score, label -> box-7, label, score
                    exchange_boxes.append(boxes)
                    
                    w = boxes[:, -1]
                    if lidar_id == 1:
                        w *= 2.0
                    
                    weights.append(w)

                exchange_boxes = np.concatenate(exchange_boxes)
                weights = np.concatenate(weights)
                fused_boxes, _ = label_fusion(exchange_boxes, 'kde_fusion', discard=1, radius=2.0, weights=weights)
                fused_boxes = torch.from_numpy(fused_boxes).float().contiguous().cuda()
                this_pred_dict = {
                    'pred_boxes': fused_boxes[:, :7],
                    'pred_scores': fused_boxes[:, -1],
                    'pred_labels': fused_boxes[:, -2].long()
                }
            else:
                raise NotImplementedError(f"BOX_FUSION_METHOD: {self.model_cfg.BOX_FUSION_METHOD} is not implemented")

            

            
            final_box_dicts.append(this_pred_dict)

        batch_dict['final_box_dicts'] = final_box_dicts

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
