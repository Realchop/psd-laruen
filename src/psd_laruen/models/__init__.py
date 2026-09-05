import lightning as L

from .amp_former import AmpFormer
from .wavenet import StreamingWaveNet, WaveNet

MODELS: dict[str, type[L.LightningModule]] = {
    "WaveNet": WaveNet,
    "AmpFormer": AmpFormer,
}

__all__ = ["MODELS", "AmpFormer", "StreamingWaveNet", "WaveNet"]
