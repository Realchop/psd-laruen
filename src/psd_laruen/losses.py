"""Loss functions for black-box amplifier modelling.

Plain MSE is a poor fit for this task: it is dominated by the high energy
fundamental, while the character of an amp lives in the comparatively quiet
harmonics it generates. The losses here follow the virtual analog literature
instead -- error-to-signal ratio, pre-emphasised to weight the harmonics, plus
a multi-resolution spectral term.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Guards a division by the target energy; matches the value used by OpenAmp.
EPSILON = 1e-5


def _as_batched(x: torch.Tensor) -> torch.Tensor:
    """Normalise (T,), (B, T) and (B, C, T) down to (B * C, T)."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    raise ValueError(f"expected 1, 2 or 3 dimensions, got {x.dim()}")


class ESRLoss(nn.Module):
    """Error-to-signal ratio, normalised per item rather than per batch.

    Dividing by the target energy makes a quietly played segment count as much
    as a loud one, which plain MSE does not.
    """

    def __init__(self, epsilon: float = EPSILON) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _as_batched(prediction), _as_batched(target)
        error = torch.mean((target - prediction) ** 2, dim=-1)
        energy = torch.mean(target**2, dim=-1) + self.epsilon
        return torch.mean(error / energy)


class DCLoss(nn.Module):
    """Penalises a constant offset between prediction and target."""

    def __init__(self, epsilon: float = EPSILON) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _as_batched(prediction), _as_batched(target)
        error = torch.mean(target - prediction, dim=-1) ** 2
        energy = torch.mean(target**2, dim=-1) + self.epsilon
        return torch.mean(error / energy)


class PreEmphasis(nn.Module):
    """First order high-pass, `y[n] = x[n] - coefficient * x[n - 1]`.

    Applied to both prediction and target before the error is taken, so that
    the loss actually cares about the harmonics an amp generates.
    """

    kernel: torch.Tensor

    def __init__(self, coefficient: float = 0.85) -> None:
        super().__init__()
        if not 0.0 <= coefficient < 1.0:
            raise ValueError("coefficient must be in [0, 1)")
        # Cross-correlation reverses the taps, hence [-a, 1] and not [1, -a].
        kernel = torch.tensor([[[-coefficient, 1.0]]])
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batched = _as_batched(x).unsqueeze(1)
        padded = F.pad(batched, (1, 0))
        return F.conv1d(padded, self.kernel.to(dtype=x.dtype)).squeeze(1)


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence plus log-magnitude L1, averaged over FFT sizes.

    Catches spectral envelope errors that a time domain loss misses, and is
    forgiving of the sub-sample misalignment that a time domain loss is not.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        epsilon: float = 1e-7,
        magnitude_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        if not fft_sizes:
            raise ValueError("fft_sizes must not be empty")
        if magnitude_floor <= 0.0:
            raise ValueError("magnitude_floor must be greater than 0")
        self.fft_sizes = tuple(fft_sizes)
        self.epsilon = epsilon
        # Bounds the log-magnitude term, which would otherwise reach for
        # log(0) on an empty bin. It does NOT rescue the convergence term on a
        # silent target -- a relative error is undefined there, and flooring
        # every bin just scales the reference with the bin count. Excluding
        # silent segments is the dataset's job (see data.MIN_RMS).
        self.magnitude_floor = magnitude_floor

    def _magnitude(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        spectrum = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        # abs() has no gradient at zero, so take the norm by hand.
        magnitude = torch.sqrt(spectrum.real**2 + spectrum.imag**2 + self.epsilon)
        return magnitude.clamp_min(self.magnitude_floor)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = _as_batched(prediction), _as_batched(target)

        total = prediction.new_zeros(())
        for n_fft in self.fft_sizes:
            if prediction.shape[-1] < n_fft:
                continue

            predicted_magnitude = self._magnitude(prediction, n_fft)
            target_magnitude = self._magnitude(target, n_fft)

            # Frobenius norm by hand; torch.norm is deprecated and untyped.
            residual = torch.sum(
                (target_magnitude - predicted_magnitude) ** 2, dim=(-2, -1)
            ).sqrt()
            reference = torch.sum(target_magnitude**2, dim=(-2, -1)).sqrt()
            convergence = residual / (reference + self.epsilon)

            log_magnitude = F.l1_loss(
                torch.log(predicted_magnitude), torch.log(target_magnitude)
            )

            total = total + torch.mean(convergence) + log_magnitude

        return total / len(self.fft_sizes)


class AmpLoss(nn.Module):
    """The combined objective, shared by every model in the comparison.

    Returns a dict so Lightning can log the components separately. The `esr`
    entry is the unweighted, un-pre-emphasised ratio -- that is the number to
    report and to checkpoint on, since it is comparable across architectures.
    """

    def __init__(
        self,
        esr_weight: float = 1.0,
        pre_emphasis_weight: float = 1.0,
        stft_weight: float = 1.0,
        dc_weight: float = 1.0,
        pre_emphasis_coefficient: float = 0.85,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        weights = (esr_weight, pre_emphasis_weight, stft_weight, dc_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss weights must not be negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one loss weight must be greater than 0")

        self.esr_weight = esr_weight
        self.pre_emphasis_weight = pre_emphasis_weight
        self.stft_weight = stft_weight
        self.dc_weight = dc_weight

        self.esr = ESRLoss(epsilon)
        self.dc = DCLoss(epsilon)
        self.pre_emphasis = PreEmphasis(pre_emphasis_coefficient)
        self.stft = MultiResolutionSTFTLoss(fft_sizes)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: prediction {tuple(prediction.shape)} "
                f"vs target {tuple(target.shape)}"
            )

        esr = self.esr(prediction, target)
        dc = self.dc(prediction, target)
        pre_emphasis = self.esr(
            self.pre_emphasis(prediction), self.pre_emphasis(target)
        )
        stft = self.stft(prediction, target)

        loss = (
            self.esr_weight * esr
            + self.pre_emphasis_weight * pre_emphasis
            + self.stft_weight * stft
            + self.dc_weight * dc
        )

        return {
            "loss": loss,
            "esr": esr,
            "esr_db": 10.0 * torch.log10(esr + self.esr.epsilon),
            "pre_emphasis": pre_emphasis,
            "stft": stft,
            "dc": dc,
        }
