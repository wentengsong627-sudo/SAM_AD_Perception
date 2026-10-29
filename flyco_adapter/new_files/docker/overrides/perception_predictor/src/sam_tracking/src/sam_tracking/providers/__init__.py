"""Keep the native public import lazy so HTTP backends need no SAM2 toolkit."""

__all__ = ["Sam2CameraProvider"]


def __getattr__(name):
    if name == "Sam2CameraProvider":
        from .sam2_camera_provider import Sam2CameraProvider
        return Sam2CameraProvider
    raise AttributeError(name)
