import torch
from .detector3d_template import Detector3DTemplate


class ObjectDiscoverer(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        assert not self.training, "there is nothing to train"
        gt_boxes = batch_dict['gt_boxes']  # (B, N_max, 8) - 7-box, class_idx
        
        final_box_dicts = list()
        for b_idx in range(batch_dict['batch_size']):
            this_gt_boxes = gt_boxes[b_idx]
            # remove padded gt_boxes (for batching purpose)
            mask_real = this_gt_boxes[:, -1] > 0
            this_gt_boxes = this_gt_boxes[mask_real]

            this_pred_dict = {
                'pred_boxes': this_gt_boxes[:, :7],
                'pred_scores': this_gt_boxes.new_zeros(this_gt_boxes.shape[0]) + 1,
                'pred_labels': this_gt_boxes[:, -1]
            }
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
