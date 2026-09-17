import argparse
import json
import sys
from datetime import datetime
from math import log10, sqrt
from pathlib import Path

import lightning as L
import torch
import torchaudio  # pyright: ignore[reportMissingTypeStubs]
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader, Dataset

from .data import (
    LEAD_IN,
    MIN_RMS,
    SAMPLE_RATE,
    SEGMENT_LENGTH,
    AmpDataset,
    contiguous_split,
)
from .inference import find_checkpoint, load_model, receptive_field, render
from .losses import AmpLoss
from .metrics import aliasing_to_signal_ratio
from .models import MODELS, StreamingWaveNet, WaveNet


def _datasets(
    args: argparse.Namespace,
) -> tuple[
    Dataset[tuple[torch.Tensor, torch.Tensor]],
    Dataset[tuple[torch.Tensor, torch.Tensor]],
]:
    """Prefer a held-out recording; fall back to a contiguous tail split."""
    train_dataset = AmpDataset(
        args.dry,
        args.wet,
        segment_length=args.segment,
        lead_in=args.lead_in,
        min_rms=args.min_rms,
    )

    if args.val_dry is None:
        print("No --val-dry given; holding out the tail of the training audio")
        return contiguous_split(train_dataset)

    val_dataset = AmpDataset(
        args.val_dry,
        args.val_wet,
        segment_length=args.segment,
        lead_in=args.lead_in,
        min_rms=args.min_rms,
    )
    return train_dataset, val_dataset


def _find_lr(
    trainer: L.Trainer,
    model: L.LightningModule,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Run a learning rate range test and adopt its suggestion."""
    tuner = Tuner(trainer)
    finder = tuner.lr_find(model, train_dataloaders=train_loader)

    if finder is None or (suggestion := finder.suggestion()) is None:
        print("No clear minimum in the range test; keeping the configured rate")
        return

    print(f"Suggested learning rate: {suggestion:.3e}")

    curve = Path(trainer.log_dir or ".") / "lr_find.json"
    curve.parent.mkdir(parents=True, exist_ok=True)
    with open(curve, "w", encoding="utf-8") as f:
        json.dump({"suggestion": suggestion, **finder.results}, f)
    print(f"Range test curve written to {curve}")


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
        "--val-dry",
        required=False,
        type=str,
        help="Path to a held-out raw recording (a different guitar, ideally)",
    )

    parser.add_argument(
        "--val-wet",
        required=False,
        type=str,
        help="Path to the processed version of --val-dry",
    )

    parser.add_argument(
        "--name",
        required=False,
        type=str,
        help="Run name; groups checkpoints and logs under lightning_logs/<name>",
    )

    parser.add_argument(
        "--segment",
        required=False,
        type=int,
        default=SEGMENT_LENGTH,
        help="Samples of target audio per training example",
    )

    parser.add_argument(
        "--grad-clip",
        required=False,
        type=float,
        default=1.0,
        help="Gradient norm clipping; 0 disables it",
    )

    parser.add_argument(
        "--patience",
        required=False,
        type=int,
        default=40,
        help="Stop after this many epochs without a val_esr improvement",
    )

    parser.add_argument(
        "--min-rms",
        required=False,
        type=float,
        default=MIN_RMS,
        help="Drop segments whose target is quieter than this RMS (0 keeps all)",
    )

    parser.add_argument(
        "--lead-in",
        required=False,
        type=int,
        default=LEAD_IN,
        help="Samples of warm-up history prepended to each example",
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

    parser.add_argument(
        "--learning-rate",
        required=False,
        type=float,
        help="Override the learning rate from --config; one point of the sweep",
    )

    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Range-test the learning rate before training and use the suggestion",
    )

    args = parser.parse_args()

    _ = L.seed_everything(args.seed, workers=True)

    if (args.val_dry is None) != (args.val_wet is None):
        print("--val-dry and --val-wet must be given together")
        return

    train_dataset, val_dataset = _datasets(args)

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

    if args.learning_rate is not None:
        hyperparameters["learning_rate"] = args.learning_rate
        print(f"Learning rate overridden from the command line: {args.learning_rate}")

    model = MODELS[args.model](**hyperparameters)

    checkpoint = ModelCheckpoint(
        monitor="val_esr",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="{epoch}-{val_esr:.5f}",
    )

    early_stop = EarlyStopping(
        monitor="val_esr", mode="min", patience=args.patience, min_delta=0.0
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint, early_stop],
        logger=TensorBoardLogger("lightning_logs", name=args.name or "default"),
        enable_progress_bar=sys.stdout.isatty(),
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
    )
    if args.find_lr:
        _find_lr(trainer, model, train_loader)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Epochs run: {trainer.current_epoch}")
    print(f"Best val_esr: {checkpoint.best_model_score}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")  # pyright: ignore[reportUnknownMemberType]


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
    dataset = AmpDataset(args.dry, args.dry, segment_length=3 * SAMPLE_RATE, lead_in=0)
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


def _evaluate_run(
    run: str,
    checkpoint: Path,
    guitar: str,
    target_root: str,
    batch: int,
    workers: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Metrics for one trained model on the held-out guitar."""
    amp = run.split("-")[0]
    target = Path(target_root) / f"CareerSG__{amp}.wav"
    if not target.exists():
        return None

    model = load_model(checkpoint)
    _ = model.to(device)
    criterion = AmpLoss().to(device)

    dataset = AmpDataset(guitar, str(target))
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)

    totals: dict[str, float] = {}
    seen = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predicted = model(x)[..., -y.shape[-1] :]
            losses = criterion(predicted, y)
            n = x.shape[0]
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value) * n
            seen += n

    metrics = {name: total / seen for name, total in totals.items()}

    align = int(getattr(model, "chunk_size", 1) or 1)

    def process(signal: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return model(signal.to(device)).cpu()

    asr, harmonic = aliasing_to_signal_ratio(process, align=align)
    metrics["asr"] = asr
    metrics["harmonic_energy"] = harmonic
    metrics["params"] = float(sum(p.numel() for p in model.parameters()))
    metrics["receptive_field"] = float(receptive_field(model))
    metrics["segments"] = float(len(dataset))
    return metrics


def evaluate() -> int:
    """Score every trained model on the held-out guitar, plus aliasing."""
    parser = argparse.ArgumentParser(
        description="Evaluate every trained model in lightning_logs/"
    )
    parser.add_argument("--logs", type=str, default="lightning_logs")
    parser.add_argument("--targets", type=str, default="data/targets")
    parser.add_argument("--val-dry", type=str, default="data/Carrer SG.wav")
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--csv", type=str, required=False, help="Also write a CSV here")
    parser.add_argument("--device", type=str, required=False)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")
    print(f"validation guitar: {args.val_dry}\n")

    runs = sorted(p.name for p in Path(args.logs).glob("*") if p.is_dir())
    rows: dict[str, dict[str, float]] = {}
    for run in runs:
        try:
            checkpoint = find_checkpoint(run, args.logs)
        except FileNotFoundError:
            print(f"{run:28s} no checkpoint yet -- skipped")
            continue
        result = _evaluate_run(
            run,
            checkpoint,
            args.val_dry,
            args.targets,
            args.batch,
            args.workers,
            device,
        )
        if result is None:
            print(f"{run:28s} no target audio -- skipped")
            continue
        rows[run] = result
        print(
            f"{run:28s} esr={10 * log10(result['esr']):7.2f} dB  "
            f"pre-emph={10 * log10(result['pre_emphasis']):7.2f} dB  "
            f"asr={10 * log10(result['asr'] + 1e-20):7.2f} dB"
        )

    if not rows:
        print("\nNothing to evaluate.")
        return 1

    print(
        f"\n{'run':28s} {'ESR dB':>8s} {'preESR dB':>10s} {'MRSTFT':>8s} "
        f"{'ASR dB':>8s} {'params':>8s} {'ctx ms':>7s}"
    )
    for run, m in sorted(rows.items(), key=lambda kv: kv[1]["esr"]):
        print(
            f"{run:28s} {10 * log10(m['esr']):8.2f} "
            f"{10 * log10(m['pre_emphasis']):10.2f} {m['stft']:8.4f} "
            f"{10 * log10(m['asr'] + 1e-20):8.2f} {int(m['params']):8d} "
            f"{m['receptive_field'] / SAMPLE_RATE * 1000:7.1f}"
        )

    if args.csv:
        columns = [
            "esr",
            "pre_emphasis",
            "stft",
            "dc",
            "asr",
            "params",
            "receptive_field",
            "segments",
        ]
        with open(args.csv, "w", encoding="utf-8") as f:
            _ = f.write("run," + ",".join(columns) + "\n")
            for run, m in sorted(rows.items()):
                _ = f.write(
                    run + "," + ",".join(f"{m.get(c, 0.0):.8g}" for c in columns) + "\n"
                )
        print(f"\nwrote {args.csv}")

    return 0


def demo() -> int:
    """Write dry / target / predicted audio side by side, so it can be heard."""
    parser = argparse.ArgumentParser(
        description="Render an excerpt through a trained model, for listening"
    )

    parser.add_argument("run", type=str, help="Run name under lightning_logs/")
    parser.add_argument("dry", type=str, help="Raw guitar recording")
    parser.add_argument(
        "-t", "--target", type=str, required=False, help="Processed reference audio"
    )
    parser.add_argument(
        "-o", "--out", type=str, default="demo", help="Directory for the wav files"
    )
    parser.add_argument(
        "--start", type=float, default=30.0, help="Excerpt start, in seconds"
    )
    parser.add_argument(
        "--seconds", type=float, default=15.0, help="Excerpt length, in seconds"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Override the checkpoint used"
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=0.95,
        help="Scale all three files by one common gain so the loudest hits this",
    )

    args = parser.parse_args()

    try:
        checkpoint = (
            Path(args.checkpoint) if args.checkpoint else find_checkpoint(args.run)
        )
    except FileNotFoundError as e:
        print(e)
        return 1

    model = load_model(checkpoint)
    field = receptive_field(model)
    print(f"run        : {args.run}")
    print(f"checkpoint : {checkpoint}")
    print(f"model      : {type(model).__name__}")
    print(f"context    : {field} samples = {field / SAMPLE_RATE * 1000:.1f} ms")

    dry, sample_rate = torchaudio.load(  # pyright: ignore[reportUnknownMemberType]
        args.dry,
        frame_offset=int(args.start * SAMPLE_RATE),
        num_frames=int(args.seconds * SAMPLE_RATE),
    )
    if dry.shape[-1] == 0:
        print(f"No audio at {args.start}s in {args.dry}")
        return 1

    predicted = render(model, dry)

    tracks = {"dry": dry, "pred": predicted}
    if args.target:
        target, _ = torchaudio.load(  # pyright: ignore[reportUnknownMemberType]
            args.target,
            frame_offset=int(args.start * SAMPLE_RATE),
            num_frames=int(args.seconds * SAMPLE_RATE),
        )
        tracks["target"] = target
        losses = AmpLoss()(predicted.unsqueeze(0), target.unsqueeze(0))
        print(
            f"\nesr on this excerpt : {losses['esr'].item():.6g} "
            f"({10 * log10(losses['esr'].item()):.1f} dB)"
        )
        for name in ("pre_emphasis", "stft", "dc"):
            print(f"{name:19s} : {losses[name].item():.6g}")

    loudest = max(float(t.abs().max()) for t in tracks.values())
    gain = args.peak / loudest if loudest > 0 else 1.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\ncommon gain applied : {gain:.4f}")
    for name, track in tracks.items():
        path = out / f"{args.run}__{name}.wav"
        torchaudio.save(str(path), track * gain, sample_rate)  # pyright: ignore[reportUnknownMemberType]
        peak = float(track.abs().max())
        print(f"  {path}  peak {peak:.3f}{'  (was clipping)' if peak > 1.0 else ''}")

    return 0


def play() -> int:
    from .amps import AMPS, process, stream

    parser = argparse.ArgumentParser(
        description="Process or stream audio files through digital amps"
    )

    parser.add_argument("operation", type=str, choices=["process", "stream"])
    parser.add_argument("amp", type=str, help="amp to use", choices=AMPS.keys())
    parser.add_argument(
        "cabinet",
        type=str,
        nargs="?",
        default="",
        help="IR cabinet to use (unused by the clean and preamp amps)",
    )
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
