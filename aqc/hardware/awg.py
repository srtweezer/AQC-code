"""The Spectrum AWG driving the AODs on CH1 and CH3.

    with aods(digital_outputs=True) as awg:
        awg.load_waveform([ch1, ch3, int_hold(aom_blocked)], -1,   # -1 loops forever
                          digital_data=[aom_blocked, camera_trigger])
        awg.force_trigger()
        ...

The analog rows are CH1 and CH3 in volts relative to the channel amplitude. With
`digital_outputs=True`, `digital_data` has one row per digital output: X0 (AOM), then
X2 (camera trigger), and a third analog row drives CH2, the intensity servo's Int Hold
input; build it with aqc.intensity_lock.int_hold from the X0 row.
"""

from contextlib import contextmanager

from e2tools.awg import AWGReplayController

from aqc import config


@contextmanager
def aods(amplitudes=config.AOD_AMPLITUDES, digital_outputs=False,
         sampling_rate=config.AWG_SAMPLING_RATE):
    channels = list(config.AOD_CHANNELS)
    amplitudes = list(amplitudes)
    max_amplitudes = [config.AWG_MAX_AMPLITUDE] * len(channels)
    if digital_outputs:
        # X0 starves the servo's photodiode whenever it blocks the beam; CH2 holds its integrator.
        channels.append(config.INT_HOLD_CHANNEL)
        amplitudes.append(config.INT_HOLD_V)
        # e2tools' RMS limit reads a constant row as sqrt(2) x its level.
        max_amplitudes.append(2 * config.INT_HOLD_V)
    awg = AWGReplayController(
        config.AWG_SERIAL,
        channels=channels,
        amplitudes=amplitudes,
        digital_channels=list(config.AOD_CHANNELS) if digital_outputs else None,
        digital_channel_outputs=[config.AOM_OUTPUT, config.CAMERA_TRIGGER_OUTPUT]
        if digital_outputs else None,
        sampling_rate=sampling_rate,
        max_amplitude=max_amplitudes,
        device_identifier=config.AWG_DEVICE,
    )
    awg.initialize()
    try:
        yield awg
    finally:
        # AWGReplayController.close() leaves the card playing; stop it first.
        awg.stop()
        awg.close()
