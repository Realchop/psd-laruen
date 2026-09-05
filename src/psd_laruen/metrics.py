"""Evaluation metrics that the training loss does not cover.

ESR measures how far the output is from the target, but it cannot say what
*kind* of error it is. A distortion generates harmonics above Nyquist which
fold back as inharmonic components, and that folding sounds like harshness or
a metallic ring rather than like ordinary error. Two models at the same ESR
can differ audibly in exactly this way, so it is worth measuring separately.

Follows "Aliasing Reduction in Neural Amp Modeling by Smoothing Activations"
(arXiv:2505.04082): drive the model with a single sine and split the output
spectrum into energy that sits on harmonics of the fundamental and energy that
does not.
"""

from collections.abc import Callable
from typing import cast

import torch

# The reference uses 1249 Hz: high enough that several harmonics land above
# Nyquist and fold, low enough to leave many that do not.
PROBE_HZ = 1249.0
PROBE_AMPLITUDE = 0.5


def largest_prime_at_most(value: int) -> int:
    """A prime DFT length keeps harmonics and their aliases in distinct bins.

    With `analysis_length` prime and the fundamental on bin `k`, the bins
    `m * k mod N` are all distinct until every bin has been visited once, so
    no alias can hide underneath a harmonic and be counted as signal.
    """
    if value < 2:
        raise ValueError("value must be at least 2")

    candidate = value
    while candidate >= 2:
        if all(candidate % d for d in range(2, int(candidate**0.5) + 1)):
            return candidate
        candidate -= 1
    raise ValueError(f"no prime at or below {value}")


def aliasing_to_signal_ratio(
    process: Callable[[torch.Tensor], torch.Tensor],
    sample_rate: int = 44_100,
    frequency: float = PROBE_HZ,
    amplitude: float = PROBE_AMPLITUDE,
    align: int = 1,
    warmup: int = 44_100,
) -> tuple[float, float]:
    """Aliased energy over harmonic energy, for a nonlinearity under sine drive.

    `process` maps (1, samples) to (1, samples). `align` pads the probe up to a
    multiple of that many samples, for models that require a chunk-aligned
    input. Returns (ASR, harmonic energy), the latter as a sanity check that
    the probe actually drove the nonlinearity.

    ASR is level dependent -- a nonlinearity aliases more the harder it is
    driven -- so `amplitude` must be held fixed across anything being compared.
    """
    if not 0 < frequency < sample_rate / 2:
        raise ValueError("frequency must be below Nyquist")

    analysis = largest_prime_at_most(sample_rate)
    # Put the fundamental exactly on a bin, so it does not leak into its
    # neighbours and get counted as aliasing.
    bin_index = max(1, round(frequency * analysis / sample_rate))
    exact = bin_index * sample_rate / analysis

    total = warmup + analysis
    if align > 1:
        total = -(-total // align) * align

    time = torch.arange(total, dtype=torch.float32)
    probe = amplitude * torch.sin(2 * torch.pi * exact * time / sample_rate)

    out = process(probe.unsqueeze(0))
    # Analyse only the steady-state tail, so the model's start-up transient
    # does not read as broadband aliasing.
    tail = out[0, -analysis:]

    # torch.fft is untyped, so pin the type here rather than let it spread.
    spectrum = cast(torch.Tensor, torch.fft.rfft(tail.to(torch.float64)))  # pyright: ignore[reportUnknownMemberType]
    energy: torch.Tensor = spectrum.real**2 + spectrum.imag**2

    harmonics = torch.zeros_like(energy, dtype=torch.bool)
    for multiple in range(1, energy.numel() // bin_index + 1):
        index = multiple * bin_index
        if index < energy.numel():
            harmonics[index] = True

    # DC is not aliasing; a nonlinearity legitimately produces an offset.
    harmonic_energy = float(energy[harmonics].sum())
    aliased_energy = float(energy[1:].sum()) - harmonic_energy

    if harmonic_energy <= 0.0:
        return float("nan"), 0.0

    return max(0.0, aliased_energy) / harmonic_energy, harmonic_energy
