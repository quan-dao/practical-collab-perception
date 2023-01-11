from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # freeze everything up to (not include) roi_head
        for name, param in self.named_parameters():
            if 'roi_head' not in name:
                param.requires_grad = False

    def forward(self, batch_dict):
        # set every module up to (not include) roi_head to eval
        for name in self.module_topology:
            if 'roi_head' not in name:
                try:
                    self.__getattr__(name).eval()
                except:
                    pass

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        # loss_corrector, tb_dict = self.corrector.get_training_loss()
        # tb_dict['loss_corrector'] = loss_corrector.item()
        # loss = loss_corrector
        #
        # if self.dense_head is not None:
        #     loss_rpn, dense_head_tb_dict = self.dense_head.get_loss()
        #     loss = loss + loss_rpn
        #     tb_dict.update(dense_head_tb_dict)

        loss_rcnn, tb_dict = self.roi_head.get_loss()
        loss = loss_rcnn

        return loss, tb_dict, disp_dict
