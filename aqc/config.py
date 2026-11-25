"""Rig addresses and device settings. Scripts read these; they never hardcode them."""

from pathlib import Path

# Spectrum AWG (on this machine). Analog CH1 and CH3 drive the two AODs.
AWG_SERIAL = 14539
AWG_DEVICE = "/dev/spcm0"
AWG_SAMPLING_RATE = 625_000_000   # Hz; waveform lengths must be multiples of 32 samples
AOD_CHANNELS = (1, 3)
AOD_AMPLITUDES = (1.8, 1.78)      # V, full-scale peak into 50 ohm (CH1, CH3)
AWG_MAX_AMPLITUDE = 2.1           # V RMS, rejected by AWGReplayController above this
# Acoustic velocity in the AOD crystal: a chirp of rate r acts as a cylindrical lens of
# focal length v^2 / (lambda r) along the deflection axis.
AOD_ACOUSTIC_VELOCITY_M_S = 616.0
# CH3 step that moves the tweezer array by one site along its rows, measured with the
# aligned pattern. scripts/calibration/aod_site_step.py rewrites the line below.
CH3_SITE_STEP_MHZ = 5.6708        # 2026-10-28, 20261028_182104_wgs_aligned.npy
# CH1 step that moves the vertical 90x10 array by one site along its columns.
# scripts/calibration/aod_site_step.py --vertical rewrites the line below.
CH1_SITE_STEP_MHZ = 5.5743        # 2026-11-23, 20261112_234338_handoff_homogenized.npy, Shepard pairs around 70 MHz
# CH3 amplitude scale per frequency for flat AOD power; holds for 1.6 V peak on CH3 with
# CH1 at 70 MHz. scripts/calibration/aod_amplitude.py rewrites the line below.
CH3_AMPLITUDE_CAL = Path(__file__).parent / "calibration" / "20261028_193037_ch3_amplitude.npz"  # 2026-10-28
# The same for CH1, with CH3 at 70 MHz and unscaled. aod_amplitude.py --channel 1 rewrites the line below.
CH1_AMPLITUDE_CAL = Path(__file__).parent / "calibration" / "20261113_003549_ch1_amplitude.npz"  # 2026-11-13
# Camera row angle per degree of WGS design angle (aqc.wgs angle_deg), measured 2026-10-22
# with andrii-test/calibration/align_rows_to_aod.py. Its sign says the imaging is not mirrored.
ROW_ANGLE_PER_DESIGN_DEG = 0.983
# Camera px per MHz of each AOD: J (rows x, y; columns CH3, CH1) and the per-trap steering,
# with the other channel at 70 MHz. scripts/calibration/aod_steering.py rewrites the line below.
AOD_STEERING_CAL = Path(__file__).parent / "calibration" / "20261111_181039_aod_steering.npz"  # 2026-11-11
# The same for the vertical 90x10 array. scripts/calibration/aod_steering.py --vertical rewrites the line below.
AOD_STEERING_VERTICAL_CAL = Path(__file__).parent / "calibration" / "20261112_195249_aod_steering_vertical.npz"  # 2026-11-12

# Digital outputs, driven by bit 15 of the analog channel they ride on.
AOM_OUTPUT = 0                    # X0, from CH1; active-low: high blocks the beam
CAMERA_TRIGGER_OUTPUT = 2         # X2, from CH3; wired to the Basler's Line1

# LB1005-S intensity servo: analog CH2 drives its Int Hold input (aqc.intensity_lock).
INT_HOLD_CHANNEL = 2
INT_HOLD_V = 2.0                  # V as programmed into 50 ohm; ~4 V measured at Int Hold (high-Z; TTL high > 2.4 V)
INT_HOLD_LEAD_US = 5.0            # hold engages this long before X0 blocks the beam
INT_HOLD_RELEASE_US = 1.0         # and releases this long after X0 opens it

# Meadowlark HDMI SLM, served from the SLM PC.
SLM_MODEL = "meadowlark_hdmi_1920x1200_8bit"
SLM_SHAPE = (1200, 1920)          # rows (y), columns (x)
SLM_HOST = "131.215.112.114"
SLM_PORT = 8080
SLM_SETTLE_S = 0.5                # liquid-crystal settling time after a new pattern
# Measured phase vs grey level (scripts/calibration/lut_calibration.py, 2026-10-27).
# It holds only while slm-server loads the same controller LUT as during that measurement.
SLM_PHASE_RESPONSE = Path(__file__).parent / "calibration" / "20261027_165618_phase_response.npz"
# Field-dependent aberration model (aqc.aberration), measured over the vertical 90x10 array's field
# (blaze x -8 px, y 0) at 0.25 V AOD drive.
# scripts/calibration/aberration_scan.py rewrites the line below.
ABERRATION_CAL = Path(__file__).parent / "calibration" / "20261112_193524_aberration.npz"  # 2026-11-12
# Camera (y, x) px of a first order with phase slope (u, v) waves per SLM px along x and y:
# SLM_ZERO_ORDER_YX + u * SLM_PX_PER_WAVE[0] + v * SLM_PX_PER_WAVE[1]. Affine fit to the
# 50x50 grating array (scripts/calibration/grating_array.py), AODs at 70/70 MHz, 2026-11-16;
# 1.5 px rms residual over the field. Nominal |a| = (f1 f3 / f2) lambda / p = 2876 px.
SLM_ZERO_ORDER_YX = (1629.3, 2334.8)
SLM_PX_PER_WAVE = ((973.5, 2790.7), (2764.2, -969.4))

# Basler acA4024-29um (USB, on this machine).
CAMERA_SERIAL = "25173881"
CAMERA_PIXEL_FORMAT = "Mono12"
# Sensor pixel pitch (Sony IMX226, square pixels, no binning). Lengths measured on the
# camera are reported as px * CAMERA_PIXEL_UM, i.e. um on the sensor, not in the trap plane.
CAMERA_PIXEL_UM = 1.85
CAMERA_TRIGGER_LINE = "Line1"     # opto-isolated input
CAMERA_TIMEOUT_MS = 10_000
CAMERA_MIN_EXPOSURE_US = 12.0     # ExposureTime floor; the camera rejects anything shorter
