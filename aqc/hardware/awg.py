"""The Spectrum AWG driving the AODs on CH1 and CH3.

    with aods(digital_outputs=True) as awg:
        awg.load_waveform(samples, -1, digital_data=digital)   # -1 loops forever
        awg.force_trigger()
        ...

`samples` has one row per AOD channel (CH1, CH3) in volts relative to the
channel amplitude. With `digital_outputs=True`, `digital_data` has one row per
digital output: X0 (AOM), then X2 (camera trigger).
"""

from contextlib import contextmanager

from e2tools.awg import AWGReplayController

from aqc import config


@contextmanager
def aods(amplitudes=config.AOD_AMPLITUDES, digital_outputs=False,
         sampling_rate=config.AWG_SAMPLING_RATE):
    awg = AWGReplayController(
        config.AWG_SERIAL,
        channels=list(config.AOD_CHANNELS),
        amplitudes=list(amplitudes),
        digital_channels=list(config.AOD_CHANNELS) if digital_outputs else None,
        digital_channel_outputs=[config.AOM_OUTPUT, config.CAMERA_TRIGGER_OUTPUT]
        if digital_outputs else None,
        sampling_rate=sampling_rate,
        max_amplitude=config.AWG_MAX_AMPLITUDE,
        device_identifier=config.AWG_DEVICE,
    )
    awg.initialize()
    try:
        yield awg
    finally:
        # AWGReplayController.close() leaves the card playing; stop it first.
        awg.stop()
        awg.close()
