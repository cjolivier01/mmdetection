# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS

from .coco import CocoDataset


@DATASETS.register_module()
class CocoRinkLandmarksDataset(CocoDataset):
    """Roboflow "hockey-rink-landmarks" v2, exported as COCO segmentation.

    Unlike the ice rink datasets -- which segment the single playing surface --
    this one segments the individual painted features on the ice. The Roboflow
    export carries a 12th category, ``rink-landmarks``, which is only the
    supercategory placeholder and never appears on an annotation; leaving it out
    of ``classes`` is what keeps the label indices at 0..10.

    Plain ``CocoDataset`` behaviour is what we want here: it maps
    ``classes`` to category ids positionally and unconditionally, so the label
    order is identical across train/valid/test even for the rare classes
    (``Slot Box`` and ``Trapezoid`` have only a couple of instances in val/test).
    """

    METAINFO = {
        "classes": (
            "Blue Line",
            "Center Ice Circle",
            "Center Line",
            "Crease",
            "Faceoff Dot",
            "Field",
            "Goal",
            "Goal Line",
            "Slot Box",
            "Trapezoid",
            "Zone Circle",
        ),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
        ],
    }
