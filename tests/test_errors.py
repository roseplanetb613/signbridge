from signbridge.core.errors import (
    InvalidArgumentError,
    ModelDownloadError,
    ModelNotFoundError,
    SignBridgeError,
    SourceOpenError,
)


def test_base_is_exception():
    assert issubclass(SignBridgeError, Exception)


def test_derived_errors_inherit_base():
    for cls in (ModelNotFoundError, ModelDownloadError, SourceOpenError, InvalidArgumentError):
        assert issubclass(cls, SignBridgeError)


def test_error_carries_message():
    err = SourceOpenError("webcam 0 not available")
    assert str(err) == "webcam 0 not available"
