import torch
import torchaudio  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import Dataset, Subset

SAMPLE_RATE = 44_100
# Multiples of 256, so that a chunked model's grid lines up the same way in
# training as it does during block-wise inference, for any chunk size up to
# 256. AmpFormer rejects anything that does not divide evenly.
SEGMENT_LENGTH = 44_032  # ~0.999 seconds
LEAD_IN = 22_016  # ~0.499 seconds
MIN_RMS = 1e-2  # -40 dBFS


class AmpDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dry_path: str,
        wet_path: str,
        segment_length: int = SEGMENT_LENGTH,
        lead_in: int = LEAD_IN,
        min_rms: float = MIN_RMS,
    ):
        if segment_length <= 0:
            raise ValueError("segment_length must be greater than 0")
        if lead_in < 0:
            raise ValueError("lead_in must not be negative")

        dry, sample_rate_dry = torchaudio.load(dry_path)  # pyright: ignore[reportUnknownMemberType]
        wet, sample_rate_wet = torchaudio.load(wet_path)  # pyright: ignore[reportUnknownMemberType]

        if sample_rate_dry != sample_rate_wet:
            raise ValueError(
                f"sample rate mismatch: {dry_path} is {sample_rate_dry} Hz, "
                f"{wet_path} is {sample_rate_wet} Hz"
            )
        if dry.shape[0] != wet.shape[0]:
            raise ValueError(
                f"channel count mismatch: {dry.shape[0]} vs {wet.shape[0]}"
            )

        frames = min(dry.shape[-1], wet.shape[-1])
        self.dry = dry[..., :frames]
        self.wet = wet[..., :frames]

        self.segment_length = segment_length
        self.lead_in = lead_in
        self.sample_rate = sample_rate_dry

        candidates = max(0, (frames - lead_in) // segment_length)
        if candidates == 0:
            raise ValueError(
                f"{frames} frames is too short for segment_length "
                f"{segment_length} plus lead_in {lead_in}"
            )

        self.segments = self._loud_segments(candidates, min_rms)
        if not self.segments:
            raise ValueError(f"every segment is quieter than min_rms {min_rms}")

    def _loud_segments(self, candidates: int, min_rms: float) -> list[int]:
        if min_rms <= 0.0:
            return list(range(candidates))

        kept: list[int] = []
        for idx in range(candidates):
            split = idx * self.segment_length + self.lead_in
            target = self.wet[:, split : split + self.segment_length]
            if torch.sqrt(torch.mean(target**2)) >= min_rms:
                kept.append(idx)
        return kept

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.segments[idx] * self.segment_length
        split = start + self.lead_in
        end = split + self.segment_length

        x = self.dry[:, start:end]
        y = self.wet[:, split:end]

        return x, y


def contiguous_split(
    dataset: AmpDataset, train_fraction: float = 0.8
) -> tuple[
    Subset[tuple[torch.Tensor, torch.Tensor]], Subset[tuple[torch.Tensor, torch.Tensor]]
]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    split = int(len(dataset) * train_fraction)
    if split == 0 or split == len(dataset):
        raise ValueError(
            f"train_fraction {train_fraction} leaves one side of the split empty "
            f"for {len(dataset)} segments"
        )

    return Subset(dataset, range(split)), Subset(dataset, range(split, len(dataset)))
