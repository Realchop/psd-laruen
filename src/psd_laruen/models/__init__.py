import lightning as L

from .wavenet import StreamingWaveNet, WaveNet

MODELS: dict[str, type[L.LightningModule]] = {
    "WaveNet": WaveNet,
}

__all__ = ["MODELS", "StreamingWaveNet", "WaveNet"]
