from .centerpoint import CenterPoint

from _dev_space.distill_bev import KnowledgeDistillationLoss


class DistillBEVCenterPoint(CenterPoint):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        if self.training:
            self.kd_loss_module = KnowledgeDistillationLoss(model_cfg.DISTILL_BEV, num_class, dataset)

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            kd_loss = self.kd_loss_module(batch_dict)
            loss = loss + kd_loss

            ret_dict = {
                'loss': loss,
            }
            tb_dict['kd_loss'] = kd_loss
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


