"""Constant-frequency AOD drive: one tone, or the sum of several (one deflected beam per tone)."""

import numpy as np


def multitone(freqs, nb_samples, sampling_rate, weights=None):
    """Sum of tones (relative amplitudes weights, default equal) snapped to the buffer grid, normalised to unit peak.

    Each tone is snapped to sampling_rate / nb_samples so it fits an integer number
    of cycles and a looped buffer wraps without a phase jump. Newman phases
    (pi k^2 / N) keep the crest factor low, so each tone keeps more of the channel
    amplitude than with equal phases. Returns the waveform and the snapped frequencies.
    """
    cycles = np.round(np.array(freqs) * nb_samples / sampling_rate)
    freqs = cycles * sampling_rate / nb_samples
    times = np.arange(nb_samples) / sampling_rate
    n = len(freqs)
    weights = np.ones(n) if weights is None else np.asarray(weights, float)
    wave = sum(a * np.cos(2 * np.pi * f * times + np.pi * k**2 / n)
               for k, (a, f) in enumerate(zip(weights, freqs)))
    return wave / np.abs(wave).max(), freqs
