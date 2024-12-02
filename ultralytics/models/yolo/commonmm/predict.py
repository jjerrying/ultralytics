# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class CommomMMPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolov8n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "commonmm"
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            nci=1,
            classes=self.args.classes,
            multimodal=True,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
