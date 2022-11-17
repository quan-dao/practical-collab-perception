import torch

from .detector3d_template import Detector3DTemplate
from _dev_space.tail_cutter import PointAligner
import logging


class Aligner(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.aligner = PointAligner(model_cfg)

    def forward(self, batch_dict):
        batch_dict = self.aligner(batch_dict)

        if self.training:
            loss, tb_dict = self.aligner.get_training_loss(batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, {}
        else:
            if self.model_cfg.get('DEBUG', False):
                logger = logging.getLogger()
                logger.warning('DEBUG flag is on, migh not generate boxes')
                return batch_dict
            else:
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

    def correct_point_cloud(self, **kwargs):
        if not self.training:
            with torch.no_grad():
                return self.aligner.correct_point_cloud(**kwargs)
        else:
            return self.aligner.correct_point_cloud(**kwargs)
