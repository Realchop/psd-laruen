import argparse
import json
from datetime import datetime
from math import sqrt
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from .amps import AMPS, process, stream
from .data import MyDataset
from .models import MODELS, StreamingWaveNet, WaveNet


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "dry",
        type=str,
        help="Path to the raw guitar recording",
    )

    parser.add_argument(
        "wet",
        type=str,
        help="Path to the processed guitat recording",
    )

    parser.add_argument(
        "-s",
        "--seed",
        required=False,
        type=int,
        default=67,
        help="Random seed to use for training",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        required=False,
        type=int,
        default=5,
        help="Maximum number of epochs",
    )

    parser.add_argument(
        "-b",
        "--batch",
        required=False,
        type=int,
        default=8,
        help="Batch size",
    )

    parser.add_argument(
        "-w",
        "--workers",
        required=False,
        type=int,
        default=4,
        help="Workers per dataloader",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="WaveNet",
        choices=MODELS.keys(),
        help="Model to train",
    )

    parser.add_argument(
        "-c",
        "--config",
        required=False,
        type=str,
        help="Path to a .json file containg model hyperparameters",
    )

    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)

    dataset = MyDataset(args.dry, args.wet)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers
    )

    hyperparameters = {}
    if args.config is not None:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                hyperparameters = json.load(f)
        except FileNotFoundError:
            print(f"Provided config not found: {args.config}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON syntax at line {e.lineno}, col {e.colno}: {e.msg}")
        except UnicodeDecodeError as e:
            print(f"Encoding error reading file: {e}")
        except (TypeError, AttributeError) as e:
            print(f"Invalid file object passed to json.load: {e}")
        except OSError as e:
            print(f"I/O error reading file: {e}")

    model = MODELS[args.model](**hyperparameters)

    trainer = L.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def wavenet() -> None:
    parser = argparse.ArgumentParser(description="Run a WaveNet benchmark")

    parser.add_argument(
        "dry",
        type=str,
        help="Raw audio to sample from",
    )
    parser.add_argument("--max_samples", required=False, type=int, default=5)
    parser.add_argument("--checkpoint", required=False, type=str, default=None)

    args = parser.parse_args()

    max_id = args.max_samples
    dataset = MyDataset(args.dry, args.dry)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_dir = Path("lightning_logs/")
        checkpoint = max(
            checkpoint_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime
        )

    model = WaveNet.load_from_checkpoint(checkpoint)  # pyright: ignore[reportUnknownMemberType]
    model = StreamingWaveNet.from_wavenet(model)

    with torch.inference_mode():
        device = next(model.parameters()).device
        print("Model loaded, starting inference...")
        print(f"Model running on: {device}")

        sizes = (64, 128, 256, 512, 1024, 2048, 4096)

        stats: dict[int, list[float]] = {size: [] for size in sizes}
        best_time = None
        best_size = None
        for id, batch in enumerate(loader):
            x, _ = batch
            x = x.squeeze(0)

            print(f"\nSample #{id + 1}")

            for chunk_size in sizes:
                print("-------------------------")
                print(f"- Chunk size: {chunk_size}")
                started = datetime.now().timestamp()
                print(f"- Began inference at: {started}")
                _ = model.process_signal(x, chunk_size)
                ended = datetime.now().timestamp()
                print(f"- Ended inference at: {ended}")
                took = ended - started
                stats[chunk_size].append(took)
                print(f"- Took: {took}")
                if not best_time or took < best_time:
                    best_time = took
                    best_size = chunk_size

            if id == max_id - 1:
                break

        means: list[float] = []
        stds: list[float] = []
        for chunk_size, times in stats.items():
            print("-------------------------")
            n = len(times)
            s = sum(times)
            print(f"Chunk size: {chunk_size}")
            mean = s / n
            print(f"Mean: {mean}")
            means.append(mean)

            # Use uncorrected std in case of one sample
            if n == 1:
                n += 1
            std = sqrt(sum([(t - mean) ** 2 for t in times]) / (n - 1))
            stds.append(std)
            print(f"Std: {std}")

        print("-------------------------")
        print(f"Best time: {best_size} ({best_time}s)")
        best_mean_index = means.index(min(means))
        best_std_index = stds.index(min(stds))
        print(f"Best mean time: {sizes[best_mean_index]} ({means[best_mean_index]}s)")
        print(f"Most stable: {sizes[best_std_index]}")


def play() -> int:
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
