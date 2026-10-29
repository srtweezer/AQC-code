"""The Meadowlark HDMI SLM, reached through the slm-server on the SLM PC."""

import time

import numpy as np
from slm_client import SLMClient

from aqc import config


def phase_to_gray(phase):
    """Map radians to grey levels through the measured phase response (config.SLM_PHASE_RESPONSE)."""
    cal = np.load(config.SLM_PHASE_RESPONSE)
    return np.round(np.interp(np.mod(phase, 2 * np.pi), cal['dphi'], cal['gray'])).astype(np.uint8)


def send_phase(phase):
    """Display `phase` (radians, shape (1200, 1920)) and wait for the LC to settle."""
    send_gray(phase_to_gray(phase))


def send_gray(gray):
    """Display raw grey levels (uint8, shape (1200, 1920)), bypassing the phase response.

    The HDMI server handles one message per connection, so each call opens its own.
    """
    with SLMClient(config.SLM_MODEL, config.SLM_HOST, config.SLM_PORT) as slm:
        if gray.shape != (slm.height, slm.width):
            raise ValueError(f"pattern shape {gray.shape} != SLM {(slm.height, slm.width)}")
        if not slm.connected:
            raise ConnectionError(f"SLM server {config.SLM_HOST}:{config.SLM_PORT} unreachable")
        if not slm.send_phase_pattern(gray):
            raise RuntimeError("SLM server did not accept the pattern")
    time.sleep(config.SLM_SETTLE_S)
