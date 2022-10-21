from .detector3d_template import Detector3DTemplate
from _dev_space.tail_cutter import PointAligner


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
            raise NotImplementedError
