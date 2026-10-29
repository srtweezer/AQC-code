"""The Basler camera, opened directly with pypylon.

    with Camera(exposure_us=500) as cam:      # waits for the AWG X2 trigger
        image = cam.grab()

    with Camera(exposure_us=500, hardware_trigger=False) as cam:   # free-run
        image = cam.grab()
"""

from pypylon import pylon

from aqc import config


class Camera:
    def __init__(self, exposure_us, hardware_trigger=True):
        self.exposure_us = exposure_us
        self.hardware_trigger = hardware_trigger

    def __enter__(self):
        info = pylon.DeviceInfo()
        info.SetSerialNumber(config.CAMERA_SERIAL)
        self._cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
        self._cam.Open()
        try:
            self._configure()
        except Exception:
            self._cam.Close()
            raise
        return self

    def __exit__(self, *exc):
        self._cam.Close()

    def _configure(self):
        cam = self._cam
        # The Default user set has every auto function off, gain 0 and gamma 1,
        # so pixel values depend only on the exposure time.
        cam.UserSetSelector.Value = "Default"
        cam.UserSetLoad.Execute()
        cam.PixelFormat.Value = config.CAMERA_PIXEL_FORMAT
        cam.ExposureTime.Value = self.exposure_us
        cam.TriggerSelector.Value = "FrameStart"
        if self.hardware_trigger:
            cam.TriggerSource.Value = config.CAMERA_TRIGGER_LINE
            cam.TriggerActivation.Value = "RisingEdge"
            cam.LineSelector.Value = config.CAMERA_TRIGGER_LINE
            cam.LineDebouncerTime.Value = 0.0   # don't swallow short AWG pulses
        cam.TriggerMode.Value = "On" if self.hardware_trigger else "Off"

    def set_exposure(self, exposure_us):
        self._cam.ExposureTime.Value = exposure_us
        self.exposure_us = exposure_us

    def grab(self):
        """Return the first frame exposed after this call (uint16, Mono12 counts)."""
        # Arming the camera per grab means no frame from before the call (an old
        # exposure time, or an SLM pattern that has since changed) is returned.
        self._cam.StartGrabbingMax(1)
        result = self._cam.RetrieveResult(config.CAMERA_TIMEOUT_MS,
                                          pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Basler grab failed: {result.GetErrorDescription()}")
            return result.Array.copy()
        finally:
            result.Release()
            self._cam.StopGrabbing()
