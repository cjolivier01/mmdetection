from .coco_metric import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class CocoIceRinkMetric(CocoMetric):
    """COCO metric for rink annotations with an unused duplicate category."""

    def compute_metrics(self, results):
        if self.cat_ids is None:
            candidate_cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
            self.cat_ids = [
                cat_id for cat_id in candidate_cat_ids
                if self._coco_api.get_ann_ids(cat_ids=[cat_id])
            ]
        return super().compute_metrics(results)
