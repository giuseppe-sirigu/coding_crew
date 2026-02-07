#!/usr/bin/env python3
"""
Coding Swarm Dashboard – Real-time monitoring of GPU, LLM server, and MCP agents.

Usage:
    ./dashboard.py --port 8000 --backend vllm
    ./dashboard.py --port 11434 --backend ollama
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

LOG_PATH = "/tmp/swarm_mcp_server.log"
REFRESH_INTERVAL = 2

# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def get_gpu_stats() -> list[dict]:
    """Query nvidia-smi for per-GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            gpus.append(
                {
                    "name": parts[0],
                    "gpu_util": float(parts[1]),
                    "mem_used": float(parts[2]),
                    "mem_total": float(parts[3]),
                    "temp": float(parts[4]),
                    "power_draw": float(parts[5]),
                    "power_limit": float(parts[6]),
                }
            )
        return gpus
    except Exception:
        return []


def get_vllm_metrics(port: int) -> dict:
    """Parse the Prometheus /metrics endpoint exposed by vLLM."""
    try:
        url = f"http://localhost:{port}/metrics"
        with urllib.request.urlopen(url, timeout=2) as resp:
            text = resp.read().decode()
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].split("{")[0]
                try:
                    metrics[key] = float(parts[-1])
                except ValueError:
                    pass
        return metrics
    except Exception:
        return {}


def get_ollama_stats(port: int) -> list[dict]:
    """Query Ollama /api/ps for running models."""
    try:
        url = f"http://localhost:{port}/api/ps"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode())
        return data.get("models", [])
    except Exception:
        return []


def check_server_health(port: int) -> bool:
    """Quick connectivity check."""
    try:
        url = f"http://localhost:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def tail_log(path: str, n: int = 12) -> list[str]:
    """Return the last *n* lines of a log file."""
    try:
        with open(path) as f:
            return deque(f, maxlen=n)  # type: ignore[arg-type]
    except FileNotFoundError:
        return [f"(waiting for log file: {path})"]
    except Exception as e:
        return [f"(error reading log: {e})"]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _bar(pct: float, width: int = 20) -> Text:
    """Render a percentage as a coloured bar."""
    filled = int(pct / 100 * width)
    if pct >= 90:
        colour = "red"
    elif pct >= 70:
        colour = "yellow"
    else:
        colour = "green"
    bar = Text()
    bar.append("█" * filled, style=colour)
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {pct:5.1f}%")
    return bar


def render_gpu_panel(gpus: list[dict]) -> Panel:
    if not gpus:
        return Panel(
            Text("nvidia-smi not available", style="dim italic"),
            title="GPU",
            border_style="red",
        )

    table = Table(show_header=True, expand=True, padding=(0, 1))
    table.add_column("GPU", style="bold")
    table.add_column("Util")
    table.add_column("VRAM")
    table.add_column("Temp")
    table.add_column("Power")

    for g in gpus:
        mem_pct = g["mem_used"] / g["mem_total"] * 100 if g["mem_total"] else 0
        table.add_row(
            g["name"],
            _bar(g["gpu_util"]),
            _bar(mem_pct) + Text(f"  {g['mem_used']:.0f}/{g['mem_total']:.0f} MiB"),
            Text(f"{g['temp']:.0f}°C", style="red" if g["temp"] >= 80 else "green"),
            Text(f"{g['power_draw']:.0f}/{g['power_limit']:.0f} W"),
        )

    return Panel(table, title="GPU", border_style="green")


def render_vllm_panel(metrics: dict, port: int, healthy: bool) -> Panel:
    if not healthy:
        return Panel(
            Text(f"vLLM not responding on port {port}", style="dim italic"),
            title="vLLM Server",
            border_style="red",
        )

    table = Table(show_header=False, expand=True, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    prompt_tokens = metrics.get("vllm:prompt_tokens_total", 0)
    gen_tokens = metrics.get("vllm:generation_tokens_total", 0)
    running = metrics.get("vllm:num_requests_running", 0)
    waiting = metrics.get("vllm:num_requests_waiting", 0)
    avg_ttft = metrics.get("vllm:time_to_first_token_seconds_sum", 0)
    avg_ttft_count = metrics.get("vllm:time_to_first_token_seconds_count", 0)
    avg_tpot = metrics.get("vllm:time_per_output_token_seconds_sum", 0)
    avg_tpot_count = metrics.get("vllm:time_per_output_token_seconds_count", 0)

    table.add_row("Status", Text("● Online", style="bold green"))
    table.add_row("Prompt tokens (total)", f"{prompt_tokens:,.0f}")
    table.add_row("Generated tokens (total)", f"{gen_tokens:,.0f}")
    table.add_row("Requests running", f"{running:.0f}")
    table.add_row("Requests waiting", f"{waiting:.0f}")

    if avg_ttft_count > 0:
        table.add_row(
            "Avg time to first token",
            f"{avg_ttft / avg_ttft_count:.3f} s",
        )
    if avg_tpot_count > 0:
        tpot = avg_tpot / avg_tpot_count
        tok_per_sec = 1.0 / tpot if tpot > 0 else 0
        table.add_row(
            "Avg time per output token",
            f"{tpot:.3f} s",
        )
        table.add_row(
            "Token generation rate",
            Text(f"{tok_per_sec:.1f} tok/s", style="bold cyan"),
        )

    return Panel(table, title="vLLM Server", border_style="blue")


def render_ollama_panel(models: list[dict], port: int, healthy: bool) -> Panel:
    if not healthy:
        return Panel(
            Text(f"Ollama not responding on port {port}", style="dim italic"),
            title="Ollama Server",
            border_style="red",
        )

    if not models:
        return Panel(
            Text("● Online – no models loaded", style="yellow"),
            title="Ollama Server",
            border_style="blue",
        )

    table = Table(show_header=True, expand=True, padding=(0, 1))
    table.add_column("Model", style="bold")
    table.add_column("Size")
    table.add_column("VRAM")
    table.add_column("Until")

    for m in models:
        name = m.get("name", "?")
        size = m.get("size", 0)
        vram = m.get("size_vram", 0)
        expires = m.get("expires_at", "")
        table.add_row(
            name,
            f"{size / 1e9:.1f} GB" if size else "–",
            f"{vram / 1e9:.1f} GB" if vram else "–",
            expires[:19] if expires else "–",
        )

    return Panel(table, title="Ollama Server", border_style="blue")


def render_log_panel(lines: list[str]) -> Panel:
    text = Text()
    for line in lines:
        stripped = line.rstrip("\n")
        if "ERROR" in stripped:
            text.append(stripped + "\n", style="bold red")
        elif "WARNING" in stripped:
            text.append(stripped + "\n", style="yellow")
        else:
            text.append(stripped + "\n", style="dim")
    return Panel(text, title="MCP Server Log", border_style="magenta")


# ---------------------------------------------------------------------------
# Layout & main loop
# ---------------------------------------------------------------------------

def build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=2),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="gpu", ratio=3),
        Layout(name="llm", ratio=2),
    )
    return layout


def main():
    parser = argparse.ArgumentParser(description="Coding Swarm Dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SWARM_PORT", "8000")),
        help="LLM server port (default: 8000)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "ollama"],
        default=os.environ.get("SWARM_BACKEND", "ollama"),
        help="LLM backend (default: ollama)",
    )
    args = parser.parse_args()

    console = Console()
    layout = build_layout()

    with Live(layout, console=console, refresh_per_second=1, screen=True):
        while True:
            # GPU
            gpus = get_gpu_stats()
            layout["gpu"].update(render_gpu_panel(gpus))

            # LLM server
            healthy = check_server_health(args.port)
            if args.backend == "vllm":
                metrics = get_vllm_metrics(args.port) if healthy else {}
                layout["llm"].update(
                    render_vllm_panel(metrics, args.port, healthy)
                )
            else:
                models = get_ollama_stats(args.port) if healthy else []
                layout["llm"].update(
                    render_ollama_panel(models, args.port, healthy)
                )

            # MCP log
            lines = tail_log(LOG_PATH)
            layout["bottom"].update(render_log_panel(lines))

            time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
