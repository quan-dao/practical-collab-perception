from .detector3d_template import Detector3DTemplate


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            # save target_dict & points (before being corrected) for debugging
            if self.model_cfg.get('CORRECTOR', None) is not None and self.model_cfg.CORRECTOR.get('DEBUG', False):
                ret_dict['target_dict'] = batch_dict['target_dict']
                ret_dict['points_original'] = batch_dict['points_original']
                ret_dict['gt_boxes'] = batch_dict['gt_boxes']
                ret_dict['hunter_meta'] = batch_dict['hunter_meta']

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if not self.model_cfg.get('RETURN_BATCH_DICT', False):
                return pred_dicts, recall_dicts
            else:
                return pred_dicts, batch_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        if self.corrector is not None:
            loss_corrector, tb_dict = self.corrector.get_training_loss(tb_dict)
            tb_dict['loss_corrector'] = loss_corrector.item()

            loss = loss_rpn + loss_corrector
        else:
            loss = loss_rpn

        if self.v2x_mid_fusion is not None:
            loss = loss + self.v2x_mid_fusion.loss_dict['loss_distill']
            try:
                tb_dict['loss_mid_fusion_distill'] = self.v2x_mid_fusion.loss_dict['loss_distill'].item()
            except:
                # in case there is no loss_mid_fusion_distill because this option is not used
                tb_dict['loss_mid_fusion_distill'] = 0.

        tb_dict['loss_total'] = loss.item()

        return loss, tb_dict, disp_dict

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
