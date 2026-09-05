"""A block-causal transformer for amplifier modelling.

Attention cannot run at the sample rate: a 3 second segment is 132,300
samples, and comparing every position against every other is ~1.75e10 pairs.
So samples are grouped into fixed chunks first, which brings a 1 second
segment down to ~689 positions.

Grouping has a price. The model now emits one thing per chunk, while audio has
to be exact per sample, and neighbouring chunks are computed independently --
so nothing forces the last sample of one chunk to meet the first sample of the
next. Any mismatch repeats at `sample_rate / chunk_size` Hz, which is audible
as a buzz. This implementation gives the network what it needs to avoid that
(every chunk sees the raw input of its predecessors, so it can work out where
the previous chunk must have ended) and leaves the rest to training. If the
seams turn out to be audible, the fallback is to stop predicting samples from
chunk summaries and predict filter coefficients instead.

The window is quoted in chunks, so with `chunk_size=64`:

    window=32   ->  2048 samples = 46.4 ms, matching WaveNet
    window=128  ->  8192 samples = 185.8 ms, matching WaveNet at depth 12

which keeps the comparison against the convolutional baselines honest.
"""

import math
from typing import cast

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..losses import AmpLoss
from ..optim import build_optimizer

STRUCTURAL_HPARAMS = (
    "in_channels",
    "out_channels",
    "chunk_size",
    "model_dim",
    "feedforward_dim",
    "num_heads",
    "num_layers",
    "window",
)


def rope_cache(
    positions: torch.Tensor, head_dim: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embeddings for the given absolute chunk positions.

    A rotary dot product depends only on the difference between two positions,
    which is what makes streaming exact: a decoder can carry on counting from
    wherever it left off and still reproduce the training-time attention.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    half = head_dim // 2
    exponent = torch.arange(half, device=positions.device, dtype=torch.float32) / half
    frequencies = torch.pow(
        torch.tensor(base, device=positions.device, dtype=torch.float32), -exponent
    )
    angles = torch.outer(positions.to(torch.float32), frequencies)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x`, shaped (batch, heads, positions, head_dim)."""
    first, second = x.chunk(2, dim=-1)
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


def sliding_causal_mask(
    positions: int, window: int, device: torch.device
) -> torch.Tensor:
    """True where a query position may attend to a key position.

    Query `i` sees keys `i - window + 1 .. i`, so a chunk never sees the
    future and never reaches further back than the window allows.
    """
    index = torch.arange(positions, device=device)
    delta = index.unsqueeze(1) - index.unsqueeze(0)
    return (delta >= 0) & (delta < window)


def window_for_context(samples: int, num_layers: int, chunk_size: int = 64) -> int:
    """Smallest per-layer window whose stacked reach covers `samples`.

    Inverts `receptive_field`, so a config can name the context it wants in
    samples and not have to account for the compounding over depth.
    """
    if samples <= 0 or num_layers <= 0 or chunk_size <= 0:
        raise ValueError("samples, num_layers and chunk_size must be positive")

    chunks = math.ceil(samples / chunk_size)
    return math.ceil((chunks - 1) / num_layers) + 1


class WindowedSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, positions, _ = x.shape

        qkv = self.qkv(x).reshape(batch, positions, 3, self.num_heads, self.head_dim)
        # (3, batch, heads, positions, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, positions, -1)
        return self.out(attended)


class AmpFormerLayer(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = WindowedSelfAttention(model_dim, num_heads)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return x + self.feedforward(self.feedforward_norm(x))


class AmpFormer(L.LightningModule):
    # nn.Module's __getattr__ widens plain attributes to Tensor | Module as far
    # as pyright is concerned, so spell these out. Lightning hates having
    # proper types 🥀
    learning_rate: float
    weight_decay: float
    warmup_steps: int

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        chunk_size: int = 64,
        model_dim: int = 64,
        feedforward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        window: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        esr_weight: float = 1.0,
        pre_emphasis_weight: float = 1.0,
        stft_weight: float = 1.0,
        dc_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        for param in STRUCTURAL_HPARAMS:
            if self.hparams[param] <= 0:  # pyright: ignore[reportUnknownMemberType]
                raise ValueError(f"{param} must be greater than 0")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")

        self.chunk_size = chunk_size
        self.window = window
        self.num_layers = num_layers
        self.head_dim = model_dim // num_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # A chunk is one position, so a stride-equals-kernel convolution is
        # exactly the chunking plus its linear projection.
        self.patch = nn.Conv1d(
            in_channels, model_dim, kernel_size=chunk_size, stride=chunk_size
        )
        self.layers = nn.ModuleList(
            AmpFormerLayer(model_dim, feedforward_dim, num_heads)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(model_dim)
        self.unpatch = nn.ConvTranspose1d(
            model_dim, out_channels, kernel_size=chunk_size, stride=chunk_size
        )

        # Start as a passthrough: the residual carries the signal and the
        # network only has to learn the difference the amp makes.
        nn.init.zeros_(self.unpatch.weight)
        if self.unpatch.bias is not None:
            nn.init.zeros_(self.unpatch.bias)

        self.criterion = AmpLoss(
            esr_weight=esr_weight,
            pre_emphasis_weight=pre_emphasis_weight,
            stft_weight=stft_weight,
            dc_weight=dc_weight,
        )

    @property
    def receptive_field(self) -> int:
        """Samples of input that can reach a given output chunk.

        Each layer reaches back `window - 1` chunks, but a layer reads the
        previous layer's output, so reach compounds over depth exactly as it
        does in a dilated convolution stack. Verified empirically against the
        mask, not assumed.
        """
        return (self.num_layers * (self.window - 1) + 1) * self.chunk_size

    @property
    def latency(self) -> int:
        """Samples of delay before an output chunk can be emitted."""
        return self.chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dim = x.dim()
        if input_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif input_dim == 2:
            x = x.unsqueeze(1)

        samples = x.shape[-1]
        # Left-padding to reach a whole number of chunks would move the chunk
        # grid by however much was padded, so the same audio would be cut into
        # chunks differently depending on the length of the window it arrived
        # in -- training one way, block-wise inference another. Refuse instead:
        # a silent phase shift is far harder to notice than an exception.
        if samples % self.chunk_size != 0:
            raise ValueError(
                f"input length {samples} is not a multiple of chunk_size "
                f"{self.chunk_size}; pad or trim to align the chunk grid"
            )

        tokens = self.patch(x).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        cos, sin = rope_cache(positions, self.head_dim)
        mask = sliding_causal_mask(tokens.shape[1], self.window, tokens.device)

        for layer in self.layers:
            tokens = cast(AmpFormerLayer, layer)(tokens, cos, sin, mask)

        out = x + self.unpatch(self.norm(tokens).transpose(1, 2))

        if input_dim == 1:
            return out.squeeze(0).squeeze(0)
        if input_dim == 2:
            return out.squeeze(1)
        return out

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)[..., -y.shape[-1] :]
        losses = self.criterion(y_hat, y)

        on_step = stage == "train"
        self.log_dict(
            {
                f"{stage}_{name}": value
                for name, value in losses.items()
                if name != "loss"
            },
            on_step=on_step,
            on_epoch=True,
        )
        self.log(
            f"{stage}_loss",
            losses["loss"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
        )

        return losses["loss"]

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # estimated_stepping_batches accounts for epochs, batch size and
        # accumulation, so the schedule lands exactly at the end of training.
        total = int(self.trainer.estimated_stepping_batches)
        return build_optimizer(
            self.parameters(),
            self.learning_rate,
            self.weight_decay,
            total,
            self.warmup_steps if self.warmup_steps > 0 else None,
        )
