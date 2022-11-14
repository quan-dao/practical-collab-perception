from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig

from _dev_space.new_nuscenes_eval.my_nuscenes_eval_utils import load_gt


class MyNuScenesEval(DetectionEval):
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)
        if verbose:
            print('Reloading ground truth according to have car as meta vehicle class')
        self.gt_boxes = load_gt(self.nusc, self.eval_set, verbose=verbose)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        self.sample_tokens = self.gt_boxes.sample_tokens
