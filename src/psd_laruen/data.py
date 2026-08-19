import torch
import torchaudio
from torch.utils.data import Dataset


class MyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dry_path: str, wet_path: str, segment_length: int = 3 * 44_100):
        self.dry, sample_rate_dry = torchaudio.load(dry_path)  # pyright: ignore[reportUnknownMemberType]
        self.wet, sample_rate_wet = torchaudio.load(wet_path)  # pyright: ignore[reportUnknownMemberType]

        assert sample_rate_dry == sample_rate_wet

        self.segment_length = segment_length
        self.num_segments = self.dry.shape[-1] // segment_length
        self.sample_rate = sample_rate_dry

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.segment_length
        end = start + self.segment_length

        x = self.dry[:, start:end]
        y = self.dry[:, start:end]

        return x, y
