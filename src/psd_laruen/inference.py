"""Running a trained model over long audio, and getting it back as files."""

from pathlib import Path

import lightning as L
import torch

from .models import AmpFormer, WaveNet

LEAD_IN_FACTOR = 2


def model_class(state_dict: dict[str, torch.Tensor]) -> type[L.LightningModule]:
    if any(key.startswith("patch.") for key in state_dict):
        return AmpFormer
    if any(key.startswith("input_conv.") for key in state_dict):
        return WaveNet
    raise ValueError(
        "cannot tell which model this checkpoint holds; "
        f"expected a key starting with 'patch.' or 'input_conv.', "
        f"got {sorted(state_dict)[:4]}"
    )


def load_model(checkpoint: str | Path) -> L.LightningModule:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cls = model_class(blob["state_dict"])
    model = cls.load_from_checkpoint(checkpoint, map_location="cpu")  # pyright: ignore[reportUnknownMemberType]
    _ = model.eval()
    return model


def find_checkpoint(run: str, root: str = "lightning_logs") -> Path:
    """The best checkpoint of the newest version of a named run."""
    versions = sorted(
        Path(root).glob(f"{run}/version_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"no runs found under {root}/{run}")

    for version in reversed(versions):
        best = [c for c in version.glob("checkpoints/*.ckpt") if c.stem != "last"]
        if best:
            return best[0]

    raise FileNotFoundError(f"no checkpoints under {root}/{run}")


def receptive_field(model: L.LightningModule) -> int:
    field = getattr(model, "receptive_field", None)
    return int(field) if isinstance(field, int) else 0


def chunk_size(model: L.LightningModule) -> int:
    """The alignment a model needs, or 1 if it does not care."""
    size = getattr(model, "chunk_size", None)
    return int(size) if isinstance(size, int) and size > 0 else 1


@torch.inference_mode()
def render(
    model: L.LightningModule,
    signal: torch.Tensor,
    block: int = 44_100,
    lead_in: int | None = None,
) -> torch.Tensor:
    """Run `model` over a whole signal, block by block, with warm-up context.

    Feeding a long signal in one pass is not an option for the transformer --
    its attention mask is quadratic in chunks, so a minute of audio would need
    gigabytes. Blocking with a lead-in gives the same answer as a single pass
    as long as the lead-in covers the receptive field.
    """
    if signal.dim() != 2:
        raise ValueError(f"expected (channels, samples), got {tuple(signal.shape)}")
    if block <= 0:
        raise ValueError("block must be greater than 0")

    if lead_in is None:
        lead_in = max(LEAD_IN_FACTOR * receptive_field(model), 1024)

    # Every window handed to the model must be a whole number of chunks, and
    # must start on a chunk boundary, or a chunked model sees a different grid
    # per window. Rounding block and lead_in up guarantees both.
    align = chunk_size(model)
    block = -(-block // align) * align
    lead_in = -(-lead_in // align) * align
    usable = (signal.shape[-1] // align) * align
    if usable == 0:
        raise ValueError(
            f"signal of {signal.shape[-1]} samples is shorter than one "
            f"{align}-sample chunk"
        )

    pieces: list[torch.Tensor] = []
    for start in range(0, usable, block):
        end = min(start + block, usable)
        context = max(0, start - lead_in)
        out = model(signal[:, context:end].unsqueeze(0))[0]
        pieces.append(out[:, -(end - start) :])

    rendered = torch.cat(pieces, dim=-1)
    # Hand back the same length that came in; the unaligned tail is copied
    # through rather than silently dropped.
    if usable < signal.shape[-1]:
        rendered = torch.cat((rendered, signal[:, usable:]), dim=-1)
    return rendered
