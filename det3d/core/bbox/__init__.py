#from . import box_np_ops, box_torch_ops, geometry

from . import box_coders, box_np_ops, box_torch_ops, geometry, region_similarity

# from .region_similarity import (RegionSimilarityCalculator,
#                                 RotateIouSimilarity, NearestIouSimilarity,
#                                 DistanceSimilarity)
from .iou import bbox_overlaps