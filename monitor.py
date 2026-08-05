import argparse
import os
import re
import sys
import time

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("rich library not installed. Install with: pip install rich")
    sys.exit(1)

from config.config import config as _raw_config


def _get_log_path() -> str:
    log_cfg = _raw_config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs/")
    log_file = log_cfg.get("log_file", "training.log")
    return os.path.join(log_dir, log_file)


def parse_metrics_line(line: str) -> dict | None:
    if not line.startswith("METRICS|"):
        return None
    parts = line.split("|")[1:]
    metrics = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                metrics[k] = float(v)
            except ValueError:
                metrics[k] = v
    return metrics


LEVEL_COLORS = {
    "ERROR": "bold red",
    "WARNING": "bold yellow",
    "INFO": "cyan",
    "DEBUG": "dim",
}


def _build_dashboard(log_lines: list[str], metrics: dict, log_file: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="logs", ratio=2),
        Layout(name="metrics", ratio=1),
    )

    header_text = Text(f"UniRes Training Monitor — {log_file}", style="bold white on blue")
    layout["header"].update(Panel(header_text))

    log_text = Text()
    displayed = log_lines[-50:]
    for line in displayed:
        match = re.match(r"^(\S+ \S+) \| (\S+)\s+\| (\S+)\s+\| (.*)", line)
        if match:
            ts, level, _module, msg = match.groups()
            color = LEVEL_COLORS.get(level.strip(), "")
            log_text.append(f"{ts} ", style="dim")
            log_text.append(f"{level:8s} ", style=color)
            log_text.append(f"{msg}\n")
        else:
            if len(log_text.plain) < 2000:
                log_text.append(f"{line}\n", style="dim")

    layout["logs"].update(
        Panel(log_text, title="Training Log", border_style="cyan")
    )

    if metrics:
        table = Table(expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", no_wrap=True)

        priority_order = ["phase", "epoch", "batch", "step", "loss", "accuracy", "miou", "oiou"]
        ordered = []
        for k in priority_order:
            if k in metrics:
                v = metrics[k]
                ordered.append((k, f"{v:.4f}" if isinstance(v, float) else str(v)))
        for k, v in sorted(metrics.items()):
            if k not in priority_order:
                ordered.append((k, f"{v:.4f}" if isinstance(v, float) else str(v)))

        for k, v in ordered:
            table.add_row(k, v)

        layout["metrics"].update(Panel(table, title="Live Metrics", border_style="green"))
    else:
        layout["metrics"].update(
            Panel("Waiting for metrics...", title="Live Metrics", border_style="yellow")
        )

    return layout


def monitor() -> None:
    parser = argparse.ArgumentParser(description="UniRes Real-Time Log Monitor")
    parser.add_argument("--log-file", default=None, help="Path to the training log file")
    args = parser.parse_args()

    log_file = args.log_file or _get_log_path()

    console = Console()
    console.print(f"[bold blue]Monitoring [underline]{log_file}[/underline][/bold blue]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    metrics: dict = {}
    log_lines: list[str] = []
    file_pos = 0

    with Live(
        _build_dashboard(log_lines, metrics, log_file),
        console=console,
        refresh_per_second=4,
        screen=True,
    ) as live:
        while True:
            try:
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        f.seek(file_pos)
                        new_lines = f.readlines()
                        if new_lines:
                            file_pos = f.tell()
                            for line in new_lines:
                                line = line.rstrip()
                                log_lines.append(line)
                                parsed = parse_metrics_line(line)
                                if parsed:
                                    metrics.update(parsed)
                else:
                    time.sleep(1)
                    continue

                live.update(_build_dashboard(log_lines, metrics, log_file))
                time.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(1)


if __name__ == "__main__":
    monitor()
