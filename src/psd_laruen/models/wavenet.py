from typing import cast

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GatedActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, filter_ = x.chunk(2, dim=1)
        return torch.tanh(filter_) * torch.sigmoid(gate)


class WaveNetLayer(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.gated = GatedActivation()
        self.causal_conv = CausalConv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation,
        )
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.gated(self.causal_conv(x))
        residual = x + self.residual_conv(z)
        skip = self.skip_conv(z)
        return residual, skip


class WaveNet(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 128,
        kernel_size: int = 2,
        dilation_depth: int = 10,
        dilation_repeat: int = 2,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # Lightning hates having proper types 🥀
        for param in self.hparams:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if self.hparams[param] <= 0:  # pyright: ignore[reportUnknownMemberType]
                raise ValueError(f"{param} must be greater than 0")

        self.learning_rate = learning_rate

        self.input_conv = CausalConv1d(in_channels, residual_channels, 1)

        layers = [
            WaveNetLayer(
                residual_channels,
                skip_channels,
                kernel_size,
                2**i,
            )
            for _ in range(dilation_repeat)
            for i in range(dilation_depth)
        ]

        self.layers = nn.ModuleList(layers)

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dim = x.dim()
        # (T,)  -> (B, C, T)
        if input_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        # (B, T) -> (B, C, T)
        elif input_dim == 2:
            x = x.unsqueeze(1)

        x = self.input_conv(x)

        skip_sum: torch.Tensor | None = None
        for layer in self.layers:
            x, skip = layer(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        # pyright being dumb; it is impossible for skip_sum to be None
        out = F.relu(skip_sum)  # pyright: ignore[reportArgumentType]
        out = F.relu(self.output_conv1(out))
        out = self.output_conv2(out)

        # (B, C, T) -> (T,)
        if input_dim == 1:
            out = out.squeeze(0).squeeze(0)
        # (B, C, T) -> (B, T)
        elif input_dim == 2:
            out = out.squeeze(1)
        return out

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class StreamingWaveNetLayer(nn.Module):
    # pyright being annoying
    context: torch.Tensor

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.gated = GatedActivation()
        self.causal_conv = CausalConv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation,
        )
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

        # The only difference from the WaveNetLayer
        context = torch.zeros(1, residual_channels, self.causal_conv.padding)
        self.register_buffer("context", context, persistent=False)

    def reset(self) -> None:
        self.context.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.cat([self.context, x], dim=-1)
        z = self.gated(self.causal_conv(ctx))
        z = z[..., -x.shape[-1] :]
        residual = x + self.residual_conv(z)
        skip = self.skip_conv(z)
        if self.causal_conv.padding > 0:
            self.context.copy_(ctx[..., -self.causal_conv.padding :])
        return residual, skip


class StreamingWaveNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 128,
        kernel_size: int = 2,
        dilation_depth: int = 10,
        dilation_repeat: int = 2,
    ) -> None:
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, residual_channels, 1)

        layers = [
            StreamingWaveNetLayer(
                residual_channels,
                skip_channels,
                kernel_size,
                2**i,
            )
            for _ in range(dilation_repeat)
            for i in range(dilation_depth)
        ]
        self.layers = nn.ModuleList(layers)
        # to make pyright happy; functionally identical to self.layers
        self._layers = layers

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(layer.causal_conv.padding for layer in self._layers)

    def reset(self) -> None:
        for layer in self._layers:
            layer.reset()

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        if block.dim() != 2:
            raise ValueError(
                "block must be shaped (in_channels, block_size), "
                f"got {tuple(block.shape)}"
            )

        x = block.unsqueeze(0)
        x = self.input_conv(x)

        # pyright being dumb; it is impossible for skip_sum to be None
        skip_sum: torch.Tensor | None = None
        for layer in self._layers:
            x, skip = layer(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = F.relu(skip_sum)  # pyright: ignore[reportArgumentType]
        out = F.relu(self.output_conv1(out))
        out = self.output_conv2(out)

        return out.squeeze(0)

    # In this case, we want to skip torch hooks
    def process(self, block: torch.Tensor) -> torch.Tensor:
        return self.forward(block)

    def process_signal(self, signal: torch.Tensor, block_size: int) -> torch.Tensor:
        if block_size <= 0:
            raise ValueError("block_size must be greater than 0")

        self.reset()
        chunks = [
            self.process(signal[..., start : start + block_size])
            for start in range(0, signal.shape[-1], block_size)
        ]
        return torch.cat(chunks, dim=-1)

    @classmethod
    def from_wavenet(cls, wavenet: WaveNet) -> "StreamingWaveNet":
        src_layers = [cast(WaveNetLayer, layer) for layer in wavenet.layers]

        dilations = [layer.causal_conv.dilation for layer in src_layers]
        # Dilations are [1, 2, 4, ..., 2**(depth-1)] repeated `repeat` times, so
        # the first 1 after index 0 marks the start of the second stack.
        dilation_depth = dilations.index(1, 1) if 1 in dilations[1:] else len(dilations)
        dilation_repeat = len(dilations) // dilation_depth

        model = cls(
            in_channels=wavenet.input_conv.conv.in_channels,
            out_channels=wavenet.output_conv2.out_channels,
            residual_channels=wavenet.input_conv.conv.out_channels,
            skip_channels=wavenet.output_conv1.in_channels,
            kernel_size=src_layers[0].causal_conv.kernel_size,
            dilation_depth=dilation_depth,
            dilation_repeat=dilation_repeat,
        )
        model.load_state_dict(wavenet.state_dict())
        return model
