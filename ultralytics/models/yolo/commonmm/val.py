# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_target, plot_images


class CommomMMValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml")
        validator = OBBValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "commonmm"

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.nci=1

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            nc=self.nc,
            nci=self.nci,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            multimodal=True
        )
    
    def output_to_target(self, output, max_det=300):
        """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
        targets = []
        for i, o in enumerate(output):
            box, conf, cls, weight = o[:max_det].cpu().split((4, 1, 1, self.nci), 1)
            j = torch.full((conf.shape[0], 1), i)
            targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf, weight), 1))
        targets = torch.cat(targets, 0).numpy()
        return [targets[:, 0], targets[:, 1], targets[:, 2:6], targets[:, 6], targets[:, -1]]

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        tmp = self.output_to_target(preds, max_det=self.args.max_det)
        tmp[-1] *= self.args.wei_norm
         
        plot_images(
            batch["img"],
            *tmp,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred