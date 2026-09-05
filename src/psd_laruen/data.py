"""Paired dry/wet audio, served in segments with a warm-up lead-in."""

import torch
import torchaudio  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import Dataset, Subset

SAMPLE_RATE = 44_100
# Multiples of 256, so that a chunked model's grid lines up the same way in
# training as it does during block-wise inference, for any chunk size up to
# 256. AmpFormer rejects anything that does not divide evenly.
SEGMENT_LENGTH = 44_032  # ~0.999 seconds
LEAD_IN = 22_016  # ~0.499 seconds
# Every loss here is relative to the target, so a target with no signal in it
# divides by almost nothing. -40 dBFS drops 1.3% of segments and cuts the
# spread of per-segment loss from ~950x to ~10x. Measured, not guessed.
MIN_RMS = 1e-2  # -40 dBFS


class AmpDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Segments of dry input paired with the matching wet target.

    Each item hands back `lead_in + segment_length` samples of dry audio but
    only `segment_length` samples of wet audio. The extra dry audio at the
    front is history: without it the first samples of every segment would be
    predicted from an all-zero past, which no model can get right and which
    therefore only adds noise to the gradient. Models are expected to trim
    their output to the target length, keeping the tail.
    """

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

        # Some processors return a sample or two more than they were given.
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

        # A segment with no signal in it says nothing about the amp, and every
        # loss here is relative to the target, so an all-but-silent target
        # divides by almost nothing and swamps the batch.
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
    """Split off a validation tail without interleaving it with training data.

    A random split over segments of one continuous recording puts validation
    segments physically between training segments -- same take, same note, same
    tone -- so validation loss reads as interpolation rather than
    generalisation. Holding out the tail avoids that.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    split = int(len(dataset) * train_fraction)
    if split == 0 or split == len(dataset):
        raise ValueError(
            f"train_fraction {train_fraction} leaves one side of the split empty "
            f"for {len(dataset)} segments"
        )

    return Subset(dataset, range(split)), Subset(dataset, range(split, len(dataset)))
