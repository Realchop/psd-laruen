from collections.abc import Callable
from typing import cast

from pedalboard import (
    Convolution,
    Distortion,
    HighpassFilter,
    HighShelfFilter,
    LowShelfFilter,
    NoiseGate,
    PeakFilter,
)
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile, AudioStream

metal: Callable[[str], Pedalboard] = lambda cabinet: Pedalboard(
    [
        NoiseGate(threshold_db=-45.0, ratio=10.0, release_ms=50.0),
        HighpassFilter(cutoff_frequency_hz=200.0),
        Distortion(drive_db=35.0),
        LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=6.0),
        PeakFilter(cutoff_frequency_hz=750.0, gain_db=-12.0, q=1.5),
        PeakFilter(cutoff_frequency_hz=2200.0, gain_db=3.0, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=5000.0, gain_db=6.0),
        Convolution(
            impulse_response_filename=cabinet,
            mix=1.0,
        ),
    ]
)

cabinet: Callable[[str], Pedalboard] = lambda cabinet: Pedalboard(
    [
        Convolution(
            impulse_response_filename=cabinet,
            mix=1.0,
        )
    ]
)

clean: Callable[[str], Pedalboard] = lambda _: Pedalboard()

AMPS: dict[str, Callable[[str], Pedalboard]] = {
    "metal": metal,
    "clean": clean,
    "cabinet": cabinet,
}


def process(amp: Pedalboard, input: str, output: str) -> int:
    try:
        with AudioFile(input) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        processed_audio = amp(audio, samplerate)

        with AudioFile(output, "w", samplerate, processed_audio.shape[0]) as f:
            f.write(processed_audio)  # pyright: ignore[reportUnknownMemberType]

        return 0

    # Cause AudioFile is quirky like that; throws if file not found
    except ValueError:
        print(f"No such input file: {input}")
        return 1


def stream(amp: Pedalboard, input: str, output: str | None = None) -> int:
    device = output or cast(str | None, AudioStream.default_output_device_name)
    if device is None:
        print("No output device found")
        return 1

    try:
        with AudioStream(output_device_name=device) as stream:
            try:
                with AudioFile(input) as f:
                    sample_rate: float = stream.sample_rate
                    while f.tell() < f.frames:
                        chunk = f.read(sample_rate)
                        stream.write(amp(chunk, sample_rate), sample_rate)

            except ValueError:
                print(f"No such input file: {input}")
                return 1

    except ValueError:
        print(f"No such audio device: {output}")
        print("Available devices:")
        print(*[f"- {d}" for d in AudioStream.output_device_names], sep="\n")

    return 0
