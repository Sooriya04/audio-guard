"""
cli.py — Rich command-line interface for the Deepfake Audio Detector.

Commands
--------
  detect   — analyse one or more audio files
  batch    — analyse an entire directory
  evaluate — run full evaluation against a labelled dataset
  info     — display model information

Usage
-----
  python cli.py detect audio.wav --model checkpoints/best_model
  python cli.py detect audio.wav noisy.mp3 --verbose --json
  python cli.py batch /path/to/audio_folder --model checkpoints/best_model
  python cli.py evaluate --model checkpoints/best_model --csv data/processed/metadata.csv
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

import config as cfg

console = Console()

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ──────────────────────────────────────────────────────────
#  Shared options
# ──────────────────────────────────────────────────────────

def model_option(default=str(cfg.CHECKPOINTS / "best_model")):
    return click.option(
        "--model", "-m",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=default,
        show_default=True,
        help="Path to the trained model checkpoint directory.",
    )


def threshold_option():
    return click.option(
        "--threshold", "-t",
        type=float,
        default=cfg.CONFIDENCE_THRESHOLD,
        show_default=True,
        help="Confidence threshold below which the result is 'uncertain'.",
    )


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def _label_style(label: str) -> str:
    return {"real": "green", "fake": "red", "uncertain": "yellow"}.get(label, "white")


def _result_table(results: list, title: str = "Detection Results") -> Table:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("File",       style="cyan",  no_wrap=True, max_width=40)
    table.add_column("Label",      justify="center", width=12)
    table.add_column("Confidence", justify="right",  width=12)
    table.add_column("Real %",     justify="right",  width=10)
    table.add_column("Fake %",     justify="right",  width=10)
    table.add_column("Chunks",     justify="right",  width=8)

    for r in results:
        label  = r["label"]
        colour = _label_style(label)
        table.add_row(
            r.get("filename", "—"),
            f"[{colour}]{label.upper()}[/{colour}]",
            f"{r['confidence']:.2%}",
            f"{r['real_prob']:.2%}",
            f"{r['fake_prob']:.2%}",
            str(len(r.get("chunks", []))),
        )
    return table


def _load_detector(model_dir: str, threshold: float):
    from inference import DeepfakeDetector
    with console.status("[bold green]Loading model…"):
        det = DeepfakeDetector(model_dir=model_dir, threshold=threshold)
    return det


# ──────────────────────────────────────────────────────────
#  CLI group
# ──────────────────────────────────────────────────────────

@click.group()
@click.version_option("1.0.0", prog_name="deepfake-detector")
def cli():
    """
    \b
    ╔══════════════════════════════════════════╗
    ║   Deepfake Audio Detector  v1.0          ║
    ║   Powered by Wav2Vec2 + PyTorch          ║
    ╚══════════════════════════════════════════╝
    Detect AI-generated speech in audio files.
    """


# ──────────────────────────────────────────────────────────
#  detect
# ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("audio_files", nargs=-1, type=click.Path(exists=True))
@model_option()
@threshold_option()
@click.option("--verbose", "-v", is_flag=True,  help="Show per-chunk breakdown.")
@click.option("--json",    "-j", is_flag=True,  help="Print raw JSON output.")
@click.option("--device",       default=None,   help="cpu | cuda")
def detect(audio_files, model, threshold, verbose, json, device):
    """Analyse one or more AUDIO_FILES for deepfake content."""
    if not audio_files:
        console.print("[red]Error:[/red] Please provide at least one audio file.")
        sys.exit(1)

    det = _load_detector(model, threshold)
    if device:
        import torch; det.device = torch.device(device)

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analysing…", total=len(audio_files))
        for path in audio_files:
            result = det.predict(path)
            result["filename"] = Path(path).name
            all_results.append(result)
            progress.advance(task)

    if json:
        click.echo(globals()["json"].dumps(all_results, indent=2))
        return

    console.print()
    console.print(_result_table(all_results))

    if verbose:
        for r in all_results:
            if r.get("chunks"):
                console.print(f"\n[bold]{r['filename']}[/bold] — chunk breakdown:")
                for i, c in enumerate(r["chunks"], 1):
                    col = _label_style(c["label"])
                    console.print(
                        f"  Chunk {i:02d}: [{col}]{c['label']:<10}[/{col}]"
                        f"  conf={c['confidence']:.2%}  fake_p={c['fake_prob']:.4f}"
                    )

    # Summary banner
    labels = [r["label"] for r in all_results]
    fake_n = labels.count("fake")
    real_n = labels.count("real")
    unc_n  = labels.count("uncertain")
    console.print(
        Panel(
            f"[green]Real: {real_n}[/green]  [red]Fake: {fake_n}[/red]  "
            f"[yellow]Uncertain: {unc_n}[/yellow]",
            title="Summary",
        )
    )


# ──────────────────────────────────────────────────────────
#  batch
# ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("audio_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@model_option()
@threshold_option()
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Save results to JSON file.")
def batch(audio_dir, model, threshold, recursive, output):
    """Analyse all audio files in AUDIO_DIR."""
    audio_path = Path(audio_dir)
    glob_fn    = audio_path.rglob if recursive else audio_path.glob
    files      = [f for ext in AUDIO_EXTS for f in glob_fn(f"*{ext}")]

    if not files:
        console.print(f"[red]No audio files found in {audio_dir}[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(files)}[/bold] audio file(s) in [cyan]{audio_dir}[/cyan]")
    det = _load_detector(model, threshold)

    all_results = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn()) as p:
        task = p.add_task(f"Processing {len(files)} files…", total=len(files))
        for f in files:
            result = det.predict(f)
            result["filename"] = f.name
            result["path"]     = str(f)
            all_results.append(result)
            p.advance(task)

    console.print()
    console.print(_result_table(all_results, title=f"Batch Results — {audio_dir}"))

    if output:
        with open(output, "w") as fp:
            globals()["json"].dump(all_results, fp, indent=2)
        console.print(f"\n[green]Results saved → {output}[/green]")


# ──────────────────────────────────────────────────────────
#  evaluate
# ──────────────────────────────────────────────────────────

@cli.command()
@model_option()
@click.option("--csv",      type=click.Path(exists=True), default=None, help="Metadata CSV path.")
@click.option("--test-dir", type=click.Path(exists=True), default=None, help="Processed audio root dir.")
@click.option("--out-dir",  type=click.Path(), default=str(cfg.RESULTS_DIR), help="Output directory.")
@click.option("--tag",      type=str, default="cli_eval", help="Tag for output files.")
def evaluate(model, csv, test_dir, out_dir, tag):
    """Run full evaluation and generate report + plots."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor
    from dataset import DeepfakeAudioDataset, load_records_from_csv, load_records_from_dirs
    from evaluate import generate_report
    from model import load_model as _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with console.status("[bold green]Loading model…"):
        mdl = _load_model(model, device)
        fe  = Wav2Vec2FeatureExtractor.from_pretrained(model)

    if csv:
        records = load_records_from_csv(Path(csv))
    elif test_dir:
        td = Path(test_dir)
        records = load_records_from_dirs(td / "real", td / "fake")
    else:
        records = load_records_from_dirs(cfg.REAL_PROC, cfg.FAKE_PROC)

    console.print(f"[cyan]Evaluating on {len(records)} samples…[/cyan]")
    ds     = DeepfakeAudioDataset(records, fe, augment=False)
    loader = DataLoader(ds, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    with console.status("[bold green]Running evaluation…"):
        metrics = generate_report(mdl, loader, device, out_dir=Path(out_dir), tag=tag)

    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value",  justify="right", style="green")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), f"{v:.4f}")
    console.print(table)
    console.print(f"\n[green]Full report saved → {out_dir}[/green]")


# ──────────────────────────────────────────────────────────
#  info
# ──────────────────────────────────────────────────────────

@cli.command()
@model_option()
def info(model):
    """Display model and configuration information."""
    from model import load_model as _load_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with console.status("[bold green]Loading model…"):
        mdl = _load_model(model, device)

    n_total     = sum(p.numel() for p in mdl.parameters())
    n_trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    table = Table(title="Model Information", show_header=False)
    table.add_column("Key",   style="cyan", width=25)
    table.add_column("Value", style="white")

    rows = [
        ("Base model",         cfg.MODEL_NAME),
        ("Classes",            "0=real, 1=fake"),
        ("Sample rate",        f"{cfg.SAMPLE_RATE} Hz"),
        ("Chunk duration",     f"{cfg.CHUNK_DURATION} s"),
        ("Confidence threshold", str(cfg.CONFIDENCE_THRESHOLD)),
        ("Device",             str(device)),
        ("Total params",       f"{n_total/1e6:.2f}M"),
        ("Trainable params",   f"{n_trainable/1e6:.2f}M"),
        ("Model dir",          model),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
