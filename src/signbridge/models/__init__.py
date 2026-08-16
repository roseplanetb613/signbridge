"""signbridge.models: 骨架时序分类模型组件（可插拔）。"""

from signbridge.models.protocol import SkeletonClassifier
from signbridge.models.stgcn import STGCN
from signbridge.models.stgcn_ctc import STGCNCTC
from signbridge.models.stgcn_fusion import FusionSTGCNCTC

__all__ = ["SkeletonClassifier", "STGCN", "STGCNCTC", "FusionSTGCNCTC"]
