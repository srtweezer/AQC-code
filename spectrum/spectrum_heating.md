# Heating from tweezer intensity and position noise

Reference: T. A. Savard, K. M. O'Hara, J. E. Thomas, "Laser-noise-induced heating in
far-off resonance optical traps", PRA 56, R1095 (1997).

## 1. Trap frequency

A tweezer is a focused Gaussian beam with waist $w_0$ (1/e² intensity radius) and
wavelength $\lambda$. Its potential is

$$
U(r, z) = -\frac{U_0}{1 + z^2/z_R^2}\,\exp\!\left(-\frac{2r^2}{w_0^2\,(1 + z^2/z_R^2)}\right),
\qquad z_R = \frac{\pi w_0^2}{\lambda},
$$

where $U_0 > 0$ is the trap depth. Near the bottom, $U \approx -U_0 + \tfrac12 m\omega_r^2 r^2 + \tfrac12 m\omega_z^2 z^2$ with

$$
\omega_r = \sqrt{\frac{4U_0}{m w_0^2}}, \qquad
\omega_z = \sqrt{\frac{2U_0}{m z_R^2}}, \qquad
\nu = \frac{\omega}{2\pi},
$$

and $m$ is the atom's mass ($m_{\mathrm{Cs}} = 2.207\times10^{-25}$ kg).

The paper writes the same thing with the 1/e intensity radius $a = w_0/\sqrt2$: the
spring constant $k = 2V_0/a^2 = 4U_0/w_0^2$, so $\omega_{tr}^2 = k/M$ is $\omega_r^2$ above.

### Numbers

$$
\nu_r = \frac{1}{2\pi}\sqrt{\frac{4U_0}{m w_0^2}}, \qquad
\nu_z = \frac{\lambda}{2\pi^2 w_0^2}\sqrt{\frac{2U_0}{m}}
$$

| $U_0$ | $w_0$ | $\lambda$ | $\nu_r$ | $\nu_z$ |
|---|---|---|---|---|
| $k_B \times 1$ mK  | 0.73 µm| 1.064 µm | 109 kHz | 36 kHz |

## 2. Noise spectra through the handoff

One-sided spectra of Savard Eqs. (8)–(9), normalised so that $\int_0^\infty S\,d\nu$ is
the variance. They come from the row-4 trajectory (`scripts/characterization/handoff_trajectory.py`)
through `heating/noise_spectra.py`:

- $S_\epsilon$: $\epsilon = I/\langle I\rangle - 1$, with $I$ the Gaussian peak amplitude (1/Hz)
- $S_x$: trap-centre position relative to uniform transport, along and across the row,
  in the trap plane (µm²/Hz)

The trace is interpolated onto a uniform 0.932 µs grid (Nyquist 536 kHz, resolution 213 Hz).

![noise spectra](../data/11_02/20261102_151300_continuous_handoff_trajectory_row4_spectra.png)

$$f_\text{handoff} = 1/55 µs = 18.18 kHz$$
$$f_\text{sample} = 1 /0.93 µs = 1.0747 MHz$$
$$f_\text{beat} = 5.6708\ \mathrm{MHz} - 5 f_s = 297 kHz$$

- **Lines every 18.18 kHz** (1/55 µs) come from the part of the position that repeats every
  cycle, with rms 12 nm along and 16 nm across. About 1/3 of it is smooth (harmonics 1–3),
  most likely real handoff motion. About 2/3 is jagged from frame to frame, most likely
  per-exposure offsets copied into every cycle. Repeating the run would separate the two. 
- **Hump at about 297 kHz**: the 5.67 MHz beat between the crossfading tones, aliased by the
  strobe sampling ($5.6708\ \mathrm{MHz} - 5 f_s$, $f_s = 1.0747$ MHz). The atom feels it
  at 5.67 MHz, far above the trap frequencies.
- **Position floor** of about $10^{-10}$ µm²/Hz is the frame-to-frame measurement jitter (about 0.04 px).
- **Missing:** random cycle-to-cycle laser noise. Each frame averages 500 cycles, so it is
  averaged out; it needs single-shot frames or a photodiode.
- **Trap frequencies sit near the handoff harmonics** (for 1 mK, 0.73 µm): $\nu_z \approx$ 2nd,
  $\nu_r \approx$ 6th, $2\nu_z \approx$ 4th, $2\nu_r \approx$ 12th.

## 3. Heating times

**Intensity noise** leads to exponential heating. The mean
energy rate (Savard Eq. 10) is

$$
\langle\dot E\rangle 
= \frac{\pi}{2}\,\omega_{tr}^2\, S_\epsilon(2\omega_{tr})\,\langle E\rangle .
$$

Here $S_\epsilon(\omega)$ is per unit angular frequency, $S_\epsilon(\omega) = S_\epsilon(\nu)/2\pi$.
Since $\langle\dot E\rangle \propto \langle E\rangle$, the energy grows exponentially with
an e-folding time (Eq. 12)

$$
\frac{1}{T_I} = \Gamma_\epsilon = \pi^2 \nu_{tr}^2\, S_\epsilon(2\nu_{tr}).
$$

**Position noise** (Savard Eqs. 15–16) heats at a constant rate
$$ 
\langle\dot E\rangle = \tfrac{\pi}{2} M\omega_{tr}^4 S_x(\omega_{tr}) 
$$ 
We define an energy-doubling time as the time it takes to increase the average energy by the energy at time $t=0$.
$$
T_x \equiv \frac{\langle E(0)\rangle}{\langle \dot{E} \rangle}
$$
Plugging in $\langle E(0)\rangle = M\omega_{tr}^2\langle x^2\rangle$, where $\langle x^2\rangle = \frac{k_B T}{M\omega_{tr}^2}\ \text{(thermal)}$ gives us an energy-doubling time

$$
\frac{1}{T_x} \equiv \frac{\langle \dot{E}\rangle}{\langle E(0) \rangle} = \pi^2 \nu_{tr}^2\, \frac{S_x(\nu_{tr})}{\langle x^2\rangle}
$$

$S$ is per Hz, as in section 2. Each axis has its own $\nu_{tr}$; in 3-D the mean energy
rate is the mean of the three.

**Both together.** With $H = p^2/2M + \tfrac12 M\omega_{tr}^2[1+\epsilon(t)][x-\delta(t)]^2$ the
two rates add to first order (Gehm et al., PRA 58, 3914 (1998)):

$$
\langle\dot E\rangle = \Gamma_\epsilon\langle E\rangle + P_x,
\qquad P_x = \tfrac{\pi}{2}M\omega_{tr}^4 S_x(\omega_{tr}) = \frac{E_0}{T_x},
$$
yielding a solution:
$$
\langle E(t)\rangle = \left(E_0 + \frac{P_x}{\Gamma_\epsilon}\right)e^{\Gamma_\epsilon t} - \frac{P_x}{\Gamma_\epsilon}.
$$


### Numbers

Inputs: section 1 frequencies ($\nu_r$ = 109.1 kHz, $\nu_z$ = 35.8 kHz), the section 2
spectra, and $T$ = 50 µK, which gives $x_{rms}$ = 82 nm radially. Along and across the
row are the two radial axes. Axial position is not measured. Each $S$ is taken two ways:

- **floor:** the median of $S$ within ±9 kHz (half a harmonic spacing), i.e. the broadband part
- **line:** the nearest handoff harmonic, taken as resonant, with its power spread over 213 Hz

| | frequency | nearest line ($n$, detuning) | $S$ floor | $T$ floor | $S$ line | $T$ line |
|---|---|---|---|---|---|---|
| $T_I$ radial | $2\nu_r$ = 218.1 kHz | 12, −0.06 kHz | 4.7e-8 /Hz | 0.18 ms | 4.4e-7 /Hz | 19 µs |
| $T_I$ axial | $2\nu_z$ = 71.6 kHz | 4, −1.17 kHz | 1.2e-8 /Hz | 6.5 ms | 2.7e-7 /Hz | 0.29 ms |
| $T_x$ along | $\nu_r$ = 109.1 kHz | 6, −0.03 kHz | 1.7e-10 µm²/Hz | 0.34 ms | 1.4e-8 µm²/Hz | 4 µs |
| $T_x$ across | $\nu_r$ = 109.1 kHz | 6, −0.03 kHz | 1.2e-10 µm²/Hz | 0.49 ms | 7.9e-8 µm²/Hz | 0.7 µs |

These are not predicted heating times. They are measurement-limited bounds:

- **The floors are mostly measurement**, not trap noise. The position floor is the
  frame-to-frame fit jitter. The intensity floor at $2\nu_r$ sits on the shoulder of the
  aliased 5.67 MHz beat. The real broadband heating is slower than "$T$ floor".
- **The line values are a worst case.** They hold only if a harmonic is exactly resonant.
  The detunings of 30–60 Hz come from the assumed 1 mK and 0.73 µm, and about 2/3 of the
  line power may be per-exposure artefact (section 2).
- **All but one time are shorter than the 4.7 ms transport**, so the perturbative formulas
  are outside their range. Taken at face value, the atom would be lost.
- **Next:** repeat the strobe run to separate real lines from artefact, and measure $\nu_r$
  and $\nu_z$.
