# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

AQC-code manages the lab hardware (SLM, AODs driven by the Spectrum AWG, Basler camera) and holds the calibration, optimization, and experiment scripts built on top of it. Keeping it organized and free of AI slop is the main goal. `~/Documents/e2-tools-aqc` is the older, messy predecessor: use it to see how working scripts talked to the hardware, but don't copy its flat layout or file-per-variant habits.

## Layout

```
aqc/                 importable package
  hardware/          awg.py, slm.py, camera.py: connect, configure, close. Nothing else.
  config.py          hosts, ports, serials, device settings: the only place they are defined
  ...                shared logic (fitting, pattern generation) that more than one script uses
scripts/             rig utilities (e.g. aod_tone.py) sit directly here
  calibration/       produce a calibration: a phase pattern, a config value, a response file
  characterization/  measure and plot the setup without producing a calibration
  optimization/
  experiment/
data/                run outputs, gitignored
```

- Scripts are thin: parse args, get devices from `aqc.hardware`, call package code, save results. Logic that a second script needs moves into `aqc/`.
- Scripts never hardcode device addresses (the old scripts had things like `SLM_HOST = "131.215.112.76"` copied into each file). Read them from `aqc/config.py`.
- Release hardware (close the AWG card, disconnect clients) before plotting or saving, using context managers.

## Environment

Everything runs in the `sk` conda env (`~/miniconda3/envs/sk/bin/python`), where `aqc` is installed editable (`pip install -e . --no-deps`). `e2-tools` and `slm-client` come from their local checkouts (`~/Documents/e2-tools`, `~/Documents/slm-client`), not PyPI.

## Hardware access

The AWG (`/dev/spcm0`) and the Basler (USB) are on this machine. The SLM is on the SLM PC, behind slm-server.

- `aqc.hardware.awg.aods(amplitudes, digital_outputs)`: context manager yielding an initialized `e2tools` `AWGReplayController` on CH1/CH3. Its exit **stops** the card, because `AWGReplayController.close()` alone leaves the card playing. With `digital_outputs=True`, the digital rows are X0 (AOM) and then X2 (camera trigger).
- `aqc.hardware.slm.send_phase(phase_rad)`: quantizes to 8 bit and sends the pattern, then sleeps for the LC settle time. The HDMI server takes one message per connection, so there is no persistent SLM handle.
- `aqc.hardware.camera.Camera(exposure_us, hardware_trigger=True)`: opens the Basler directly with pypylon (camera-remote is not used). `grab()` arms the camera for each call, so it never returns a stale frame. With the hardware trigger it waits for an X2 pulse on Line1.
- Don't import `aqc.hardware.awg` from code that doesn't drive the card. Importing it loads `libspcm`, and when that process exits it unlinks the driver's shared memory out from under other processes using the card (see `e2tools/awg/__init__.py`).
- WGS and SLM pattern tools live in `e2tools.slm.local_zernike` and `e2tools.slm.tweezer_file`.

When asked to operate hardware with code that already works, run it unchanged. If you suspect a bug, mention it once, but don't "fix" it from reading the code alone.

## Outputs

Measurement and calibration scripts save plots (PNG) to `data/<timestamp>_<name>.png` and don't save raw frames or fit arrays unless asked. Never write outputs into the repo root or next to the script.

## Writing code here

- Change the existing script for a new case (add a flag or parameter). Don't create `foo_v2.py`, `foo_blaze.py`, `foo_fix.py`, or `foo_try2.py`.
- Each script starts with a module docstring covering what it does, the physical setup it assumes, and a `Usage:` block. Comments explain physics or hardware quirks, not what the next line does.
- Only write code the task needs. That means no speculative options, fallback code paths, `try/except` that swallows hardware errors, retry loops, or logging frameworks unless asked.
- Don't add README or summary markdown files for a change unless asked.
- Delete one-off scratch scripts once they've served their purpose instead of leaving them in the tree.
