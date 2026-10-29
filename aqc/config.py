"""Rig addresses and device settings. Scripts read these; they never hardcode them."""

from pathlib import Path

# Spectrum AWG (on this machine). Analog CH1 and CH3 drive the two AODs.
AWG_SERIAL = 14539
AWG_DEVICE = "/dev/spcm0"
AWG_SAMPLING_RATE = 625_000_000   # Hz; waveform lengths must be multiples of 32 samples
AOD_CHANNELS = (1, 3)
AOD_AMPLITUDES = (1.8, 1.78)      # V, full-scale peak into 50 ohm (CH1, CH3)
AWG_MAX_AMPLITUDE = 2.1           # V RMS, rejected by AWGReplayController above this
# CH3 step that moves the tweezer array by one site along its rows, measured with the
# aligned pattern. scripts/calibration/aod_site_step.py rewrites the line below.
CH3_SITE_STEP_MHZ = 5.6708        # 2026-10-28, 20261028_182104_wgs_aligned.npy
# CH3 amplitude scale per frequency for flat AOD power; holds for 1.6 V peak on CH3 with
# CH1 at 70 MHz. scripts/calibration/aod_amplitude.py rewrites the line below.
CH3_AMPLITUDE_CAL = Path(__file__).parent / "calibration" / "20261028_193037_ch3_amplitude.npz"  # 2026-10-28

# Digital outputs, driven by bit 15 of the analog channel they ride on.
AOM_OUTPUT = 0                    # X0, from CH1; active-low: high blocks the beam
CAMERA_TRIGGER_OUTPUT = 2         # X2, from CH3; wired to the Basler's Line1

# Meadowlark HDMI SLM, served from the SLM PC.
SLM_MODEL = "meadowlark_hdmi_1920x1200_8bit"
SLM_SHAPE = (1200, 1920)          # rows (y), columns (x)
SLM_HOST = "131.215.112.114"
SLM_PORT = 8080
SLM_SETTLE_S = 0.5                # liquid-crystal settling time after a new pattern
# Measured phase vs grey level (scripts/calibration/lut_calibration.py, 2026-10-27).
# It holds only while slm-server loads the same controller LUT as during that measurement.
SLM_PHASE_RESPONSE = Path(__file__).parent / "calibration" / "20261027_165618_phase_response.npz"

# Basler acA4024-29um (USB, on this machine).
CAMERA_SERIAL = "25173881"
CAMERA_PIXEL_FORMAT = "Mono12"
CAMERA_TRIGGER_LINE = "Line1"     # opto-isolated input
CAMERA_TIMEOUT_MS = 10_000
CAMERA_MIN_EXPOSURE_US = 12.0     # ExposureTime floor; the camera rejects anything shorter
