from .amps import AMPS, process, stream


def main() -> None:
    print("Main!")


def play() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Process or stream audio files through digital amps"
    )

    parser.add_argument("operation", type=str, choices=["process", "stream"])
    parser.add_argument("amp", type=str, help="amp to use", choices=AMPS.keys())
    parser.add_argument("cabinet", type=str, help="IR cabinet to use")
    parser.add_argument("input", type=str, help="audio to transform")
    parser.add_argument(
        "-o", "--output", type=str, required=False, help="output location"
    )

    args = parser.parse_args()

    try:
        amp = AMPS[args.amp](args.cabinet)
    except Exception:  # noqa: BLE001
        print(f"Cannot find cabinet: {args.cabinet}")
        return 1

    if args.operation == "stream":
        return stream(amp, args.input, args.output)

    if not args.output:
        print("Output is required for processing.")
        return 1

    return process(amp, args.input, args.output)
