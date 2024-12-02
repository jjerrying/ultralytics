# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import CommonMMModel
from ultralytics.utils import DEFAULT_CFG, RANK


class CommomMMTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml", epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "commonmm"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return CommonMMModel initialized with specified config and weights."""
        model = CommonMMModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "wei_loss"
        return yolo.commonmm.CommomMMValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
