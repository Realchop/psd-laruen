"""Assemble the evidence that the training runs actually happened.

Everything training writes -- `lightning_logs*/`, `logs*/`, `checkpoints/` --
is gitignored, so the repository itself records only that the grid *could* be
run, never that it *was*. This walks what the runs left on disk and pairs it
with SLURM's accounting database.

That pairing is the point. Files on disk are ours and we could have written
them by hand; the `sacct` rows are the cluster's, recorded by the scheduler at
the moment it allocated a GPU and again when the job exited. A run that shows
up in both -- same job id, same node, same wall-clock window -- is corroborated
by a party that was not us.

    .venv/bin/python scripts/run_manifest.py

Writes results/run_manifest.md, results/run_manifest.csv and
results/slurm_jobs.tsv, all small enough to commit.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"

# Lightning names its event files events.out.tfevents.<unix>.<host>.<pid>.<n>,
# which is the only place the originating host and process id survive.
EVENT_NAME = re.compile(
    r"events\.out\.tfevents\.(?P<unix>\d+)\.(?P<host>[^.]+)\.(?P<pid>\d+)\.(?P<idx>\d+)$"
)
# Slurm output is logs/<jobname>-<jobid>.out. The job name was the run tag
# before train.sbatch settled on a fixed --job-name=amp, so only the trailing
# number is dependable.
OUT_NAME = re.compile(r"^(?P<name>.+)-(?P<jobid>\d+)\.out$")
TAG_LINE = re.compile(r"\btag=(?P<tag>\S+)")
BEST_CKPT = re.compile(r"^Best checkpoint:\s*(?P<path>\S+)", re.MULTILINE)
BEST_ESR = re.compile(r"^Best val_esr:\s*(?P<esr>\S+)", re.MULTILINE)
EPOCHS_RUN = re.compile(r"^Epochs run:\s*(?P<n>\d+)", re.MULTILINE)
GPU_LINE = re.compile(r"^(?P<gpu>NVIDIA [^,]+),", re.MULTILINE)


def utc(unix: float) -> str:
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%d %H:%M:%S")


def sha256(path: Path) -> str:
    """Full digest, truncated for display but computed over the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SlurmJob:
    jobid: str
    name: str
    state: str
    start: str
    end: str
    elapsed: str
    node: str


@dataclass
class Run:
    root: str          # which lightning_logs* directory
    tag: str           # amp-architecture, e.g. metal-wavenet_46ms
    version: str       # version_0, version_1, ...
    host: str = ""
    pid: str = ""
    started: str = ""
    ended: str = ""
    minutes: float = 0.0
    epochs: int = 0
    best_val_esr: float | None = None
    last_val_esr: float | None = None
    scalar_points: int = 0
    learning_rate: float | None = None
    checkpoints: list[tuple[str, int, str]] = field(default_factory=list)
    jobids: list[str] = field(default_factory=list)
    gpu: str = ""
    match: str = ""      # how the job id was tied to this directory
    node: str = ""       # node SLURM says the job ran on
    host_ok: bool | None = None   # does that agree with the event file?

    @property
    def path(self) -> str:
        return f"{self.root}/{self.tag}/{self.version}"


def slurm_jobs(since: str, until: str) -> dict[str, list[SlurmJob]]:
    """Ask the scheduler what it remembers running for us in that window.

    Keyed by job id, but a job preempted and requeued keeps its id and gets a
    fresh allocation, so sacct emits one row per incarnation and all of them
    are kept. Collapsing them to the last row loses the window the run
    actually executed in.

    Failure here is not fatal: the on-disk half of the manifest still stands
    on its own, it just loses its independent witness.
    """
    fields = "JobID,JobName,State,Start,End,Elapsed,NodeList"
    try:
        out = subprocess.run(
            # -D is what makes a requeued job report every allocation it
            # held; without it sacct folds them into the final one, whose
            # window can postdate the run that actually wrote the logs.
            ["sacct", "-X", "-D", "--noheader", "--parsable2",
             f"--format={fields}", "-S", since, "-E", until],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  sacct unavailable ({error}); manifest will be disk-only")
        return {}

    jobs: dict[str, list[SlurmJob]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 7:
            jobs.setdefault(parts[0], []).append(SlurmJob(*parts))
    return jobs


def read_scalars(version_dir: Path, run: Run) -> None:
    accumulator = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]

    walls: list[float] = []
    for tag in tags:
        events = accumulator.Scalars(tag)
        run.scalar_points += len(events)
        walls.extend((events[0].wall_time, events[-1].wall_time))

    if walls:
        run.started, run.ended = utc(min(walls)), utc(max(walls))
        run.minutes = round((max(walls) - min(walls)) / 60, 1)

    if "val_esr" in tags:
        values = [event.value for event in accumulator.Scalars("val_esr")]
        run.best_val_esr, run.last_val_esr = min(values), values[-1]
    if "epoch" in tags:
        run.epochs = int(max(event.value for event in accumulator.Scalars("epoch"))) + 1


def collect_runs(roots: list[Path]) -> Iterator[Run]:
    for root in sorted(roots):
        for version_dir in sorted(root.glob("*/version_*")):
            if not version_dir.is_dir():
                continue
            run = Run(root=root.name, tag=version_dir.parent.name, version=version_dir.name)

            for event_file in sorted(version_dir.glob("events.out.tfevents.*")):
                match = EVENT_NAME.search(event_file.name)
                if match and not run.host:
                    run.host, run.pid = match["host"], match["pid"]

            read_scalars(version_dir, run)

            hparams = version_dir / "hparams.yaml"
            if hparams.exists():
                loaded = yaml.safe_load(hparams.read_text()) or {}
                run.learning_rate = loaded.get("learning_rate")

            for ckpt in sorted((version_dir / "checkpoints").glob("*.ckpt")):
                run.checkpoints.append((ckpt.name, ckpt.stat().st_size, sha256(ckpt)))

            yield run


def parse_slurm_time(stamp: str) -> float | None:
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=UTC
        ).timestamp()
    except ValueError:
        return None  # 'None', 'Unknown' -- the job never started


def covering(run: Run, incarnations: list[SlurmJob]) -> SlurmJob | None:
    """The allocation, if any, that this run's first event falls inside.

    The fallback for a job killed before it could print its checkpoint path.
    A tag names which cell of the grid a run belongs to but not which attempt,
    so the timestamps decide: a run whose logging began while that job held a
    GPU is that job's, since two attempts at the same cell never overlap.
    """
    if not run.started:
        return None
    began = datetime.strptime(run.started, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=UTC
    ).timestamp()
    for job in incarnations:
        start, end = parse_slurm_time(job.start), parse_slurm_time(job.end)
        if start is None:
            continue  # queued and cancelled; it never held a GPU
        if run.host and job.node and job.node != run.host:
            continue  # right window, wrong machine: not this allocation
        # Lightning opens its event file a little after srun hands over, and
        # the scheduler stamps the end after the process is gone.
        if start - 60 <= began <= (end + 60 if end is not None else start + 86400):
            return job
    return None


def attach_nodes(runs: list[Run], jobs: dict[str, list[SlurmJob]]) -> None:
    """Check each match against a fact neither side could fake alone.

    Lightning writes the hostname into its event filename; SLURM records the
    node it allocated. Nothing in the training code reads the accounting
    database, so the two agreeing on every run is the part of this manifest
    that is hard to have produced by accident.
    """
    for run in runs:
        incarnations = jobs.get(run.jobids[0], []) if run.jobids else []
        if not incarnations:
            continue
        job = covering(run, incarnations) or incarnations[0]
        run.node = job.node
        if run.host and job.node:
            run.host_ok = job.node == run.host


def link_slurm(runs: dict[str, Run], log_roots: list[Path],
               jobs: dict[str, list[SlurmJob]]) -> None:
    """Attach job ids using the checkpoint path each .out file printed.

    The log directories were renamed after the fact (lightning_logs ->
    lightning_logs_misaligned_grid and so on), so the path inside an old .out
    no longer resolves. A logs<suffix> directory pairs with the
    lightning_logs<suffix> of the same suffix, which is enough to repoint it.

    A job the scheduler killed never printed that path, so those fall back to
    matching on the run tag plus the job's own allocation window.
    """
    unresolved: list[tuple[str, str, str]] = []   # (suffix, tag, jobid)

    for log_root in sorted(log_roots):
        suffix = log_root.name[len("logs"):]
        for out_file in sorted(log_root.glob("*.out")):
            name_match = OUT_NAME.match(out_file.name)
            if not name_match:
                continue
            text = out_file.read_text(errors="replace")
            gpu_match = GPU_LINE.search(text)
            ckpt_match = BEST_CKPT.search(text)
            if not ckpt_match:
                tag_match = TAG_LINE.search(text)
                if tag_match:
                    unresolved.append(
                        (suffix, tag_match["tag"], name_match["jobid"])
                    )
                continue  # killed before it printed where it wrote

            parts = Path(ckpt_match["path"]).parts
            if len(parts) < 3:
                continue
            key = f"lightning_logs{suffix}/{parts[1]}/{parts[2]}"
            run = runs.get(key) or runs.get("/".join(parts[:3]))
            if run is None:
                continue

            run.jobids.append(name_match["jobid"])
            run.match = "checkpoint-path"
            if gpu_match and not run.gpu:
                run.gpu = gpu_match["gpu"]

    for suffix, tag, jobid in unresolved:
        incarnations = jobs.get(jobid)
        if not incarnations:
            continue
        for run in runs.values():
            if run.root != f"lightning_logs{suffix}" or run.tag != tag:
                continue
            if not run.jobids and covering(run, incarnations):
                run.jobids.append(jobid)
                run.match = "tag+window"
                break

    # Some .out files are simply gone. Before train.sbatch fixed the job name
    # to "amp", it submitted each cell under its own tag, so for those the
    # scheduler kept the name we need and the run can be matched without any
    # log file at all.
    for run in runs.values():
        if run.jobids:
            continue
        for jobid, incarnations in jobs.items():
            named = [job for job in incarnations if job.name == run.tag]
            if named and covering(run, named):
                run.jobids.append(jobid)
                run.match = "jobname+window"
                break


def write_csv(runs: list[Run], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run_dir", "tag", "version", "slurm_jobids", "node_host", "pid", "gpu",
            "started_utc", "ended_utc", "minutes", "epochs", "learning_rate",
            "best_val_esr", "last_val_esr", "scalar_points", "match_method",
            "slurm_node", "host_matches_slurm", "checkpoints",
            "checkpoint_bytes", "checkpoint_sha256",
        ])
        for run in runs:
            writer.writerow([
                run.path, run.tag, run.version, " ".join(run.jobids), run.host, run.pid,
                run.gpu, run.started, run.ended, run.minutes, run.epochs,
                run.learning_rate, run.best_val_esr, run.last_val_esr, run.scalar_points,
                run.match, run.node, run.host_ok, len(run.checkpoints),
                sum(size for _, size, _ in run.checkpoints),
                " ".join(digest[:16] for _, _, digest in run.checkpoints),
            ])


def write_markdown(runs: list[Run], jobs: dict[str, list[SlurmJob]],
                   path: Path) -> None:
    linked = [run for run in runs if run.jobids]
    corroborated = [run for run in linked if any(job in jobs for job in run.jobids)]
    total_ckpt = sum(size for run in runs for _, size, _ in run.checkpoints)
    span = [run.started for run in runs if run.started]

    lines = [
        "# Run manifest",
        "",
        (f"Generated {utc(datetime.now(UTC).timestamp())} UTC by "
         "`scripts/run_manifest.py`."),
        "",
        "Training artefacts are gitignored, so this file is the record that the runs",
        "happened. Each row pairs what training wrote on disk with the job id SLURM",
        "recorded independently; `sacct -j <jobid>` reproduces the scheduler's side",
        "for as long as the cluster keeps its accounting rows.",
        "",
        "## Summary",
        "",
        f"- Runs on disk: **{len(runs)}**",
        f"- Runs matched to a SLURM job id: **{len(linked)}**",
        f"- Job ids still present in `sacct`: **{len(corroborated)}**",
        (f"- Runs whose event-file hostname matches the node SLURM recorded: "
         f"**{sum(1 for run in runs if run.host_ok)}/"
         f"{sum(1 for run in runs if run.host_ok is not None)}**"),
        (f"- Checkpoints: **{sum(len(run.checkpoints) for run in runs)}** "
         f"({total_ckpt / 2**30:.1f} GiB), each SHA-256'd below"),
        f"- First run started: **{min(span) if span else 'n/a'} UTC**",
        f"- Last run started: **{max(span) if span else 'n/a'} UTC**",
        "",
        "## Runs",
        "",
        ("| run | slurm | matched by | node | started (UTC) | min | ep | lr | "
         "best val_esr | ckpts |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        job_cell = " ".join(
            f"`{jobid}`" + ("" if jobid in jobs else "†") for jobid in run.jobids
        ) or "—"
        node = run.node or run.host
        if run.host_ok is not None:
            node += " ✓" if run.host_ok else " ✗"
        esr = f"{run.best_val_esr:.3e}" if run.best_val_esr is not None else "—"
        lr = f"{run.learning_rate:g}" if run.learning_rate is not None else "—"
        lines.append(
            f"| `{run.path}` | {job_cell} | {run.match or '—'} | {node or '—'} | "
            f"{run.started or '—'} | {run.minutes:g} | {run.epochs} | {lr} | {esr} | "
            f"{len(run.checkpoints)} |"
        )

    lines += [
        "",
        "† job id parsed from a log file but no longer in the accounting database.",
        "",
        "`checkpoint-path` means the job printed the directory it wrote to, so the",
        "pairing is exact. `tag+window` means the scheduler killed the job before it",
        "could print that, and the run is tied to it by grid cell plus the fact that",
        "its logging began inside that job's allocation on that same node.",
        "`jobname+window` is the same match for a run whose log file is gone, using",
        "the name the scheduler itself stored. A ✓ on the node marks a run whose",
        "event filename names the machine SLURM says it allocated.",
        "",
        "## SLURM records",
        "",
        ("Verbatim `sacct` rows for the job ids above, also in "
         "`results/slurm_jobs.tsv`. A job that was preempted and requeued kept "
         "its id and appears once per allocation."),
        "",
        "| jobid | name | state | start | end | elapsed | node |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for run in runs:
        for jobid in run.jobids:
            if jobid in seen:
                continue
            seen.add(jobid)
            # A requeued job contributes a row per allocation it was given.
            for job in jobs.get(jobid, []):
                lines.append(
                    f"| {job.jobid} | {job.name} | {job.state} | {job.start} | "
                    f"{job.end} | {job.elapsed} | {job.node} |"
                )

    lines += ["", "## Checkpoint digests", "", "```"]
    for run in runs:
        for name, size, digest in run.checkpoints:
            lines.append(f"{digest}  {size:>12}  {run.path}/checkpoints/{name}")
    lines += ["```", ""]

    path.write_text("\n".join(lines))


def main() -> None:
    lightning_roots = [p for p in PROJECT.glob("lightning_logs*") if p.is_dir()]
    log_roots = [
        p for p in PROJECT.glob("logs*")
        if p.is_dir() and not p.name.startswith("lightning")
    ]
    print(f"lightning roots: {[p.name for p in lightning_roots]}")
    print(f"log roots:       {[p.name for p in log_roots]}")

    runs = list(collect_runs(lightning_roots))
    print(f"runs on disk:    {len(runs)}")

    stamps = [run.started for run in runs if run.started]
    since = (min(stamps)[:10] if stamps else "2026-01-01")
    jobs = slurm_jobs(since, "now")
    print(f"sacct rows since {since}: {len(jobs)}")

    by_path = {run.path: run for run in runs}
    link_slurm(by_path, log_roots, jobs)
    attach_nodes(runs, jobs)
    print(f"linked to a job:  {sum(1 for r in runs if r.jobids)}")

    RESULTS.mkdir(exist_ok=True)
    write_csv(runs, RESULTS / "run_manifest.csv")
    write_markdown(runs, jobs, RESULTS / "run_manifest.md")
    rows: list[str] = []
    emitted: set[str] = set()
    for run in runs:
        for jobid in run.jobids:
            if jobid in emitted:
                continue
            emitted.add(jobid)
            for j in jobs.get(jobid, []):
                rows.append(
                    f"{j.jobid}\t{j.name}\t{j.state}\t{j.start}\t"
                    f"{j.end}\t{j.elapsed}\t{j.node}"
                )
    (RESULTS / "slurm_jobs.tsv").write_text(
        "jobid\tname\tstate\tstart\tend\telapsed\tnode\n"
        + "".join(row + "\n" for row in rows)
    )
    print("wrote results/run_manifest.md, run_manifest.csv, slurm_jobs.tsv")


if __name__ == "__main__":
    main()
