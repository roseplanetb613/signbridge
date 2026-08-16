"""signbridge.models: 骨架时序分类模型组件（可插拔）。"""

from signbridge.models.protocol import SkeletonClassifier
from signbridge.models.stgcn import STGCN

__all__ = ["SkeletonClassifier", "STGCN"]
