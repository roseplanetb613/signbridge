"""SignBridge 异常体系。

原则：检测不到手不是错误（返回空 HandFrame）；打不开源、模型不可用才是错误。
"""


class SignBridgeError(Exception):
    """SignBridge 所有异常的基类。"""


class ModelNotFoundError(SignBridgeError):
    """模型文件缺失。"""


class ModelDownloadError(SignBridgeError):
    """模型下载失败。"""


class SourceOpenError(SignBridgeError):
    """输入源无法打开（摄像头打不开 / 文件不存在或无法解码）。"""


class InvalidArgumentError(SignBridgeError):
    """参数非法。"""
