from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bench_loader import TestRun

LAT = "Latency Distribution"


def build_comparison_table(runs: list[TestRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        s = r.summary
        lat = s.get(LAT, {})
        rows.append(
            {
                "Teste": r.label,
                "SGBD": s.get("DBMS Type"),
                "Versao": s.get("DBMS Version"),
                "Isolation": s.get("isolation") or r.config.get("isolation"),
                "Terminals": s.get("terminals") or r.config.get("terminals"),
                "Scale Factor": s.get("scalefactor") or r.config.get("scalefactor"),
                "Duracao (s)": round(s.get("Elapsed Time (nanoseconds)", 0) / 1e9, 1),
                "Requests medidos": s.get("Measured Requests"),
                "Throughput (req/s)": round(s.get("Throughput (requests/second)", 0), 2),
                "Goodput (req/s)": round(s.get("Goodput (requests/second)", 0), 2),
                "Latencia media (us)": lat.get("Average Latency (microseconds)"),
                "Latencia mediana (us)": lat.get("Median Latency (microseconds)"),
                "P90 (us)": lat.get("90th Percentile Latency (microseconds)"),
                "P95 (us)": lat.get("95th Percentile Latency (microseconds)"),
                "P99 (us)": lat.get("99th Percentile Latency (microseconds)"),
                "Latencia max (us)": lat.get("Maximum Latency (microseconds)"),
                "Latencia min (us)": lat.get("Minimum Latency (microseconds)"),
            }
        )
    return pd.DataFrame(rows)


def _save(fig, out_dir: Path, filename: str) -> Path:
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_throughput_over_time(runs: list[TestRun], out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for r in runs:
        if r.samples is None or r.samples.empty:
            continue
        ax.plot(r.samples["time_s"], r.samples["throughput"], label=r.label, linewidth=1.6)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput ao longo do tempo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "throughput_over_time.png")


def plot_latency_over_time(runs: list[TestRun], out_dir: Path, metric: str = "p95_latency_us") -> Path | None:
    label_map = {
        "avg_latency_us": "Latencia media",
        "median_latency_us": "Latencia mediana",
        "p95_latency_us": "Latencia P95",
        "p99_latency_us": "Latencia P99",
        "max_latency_us": "Latencia maxima",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for r in runs:
        if r.samples is None or r.samples.empty or metric not in r.samples.columns:
            continue
        ax.plot(r.samples["time_s"], r.samples[metric], label=r.label, linewidth=1.6)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(f"{label_map.get(metric, metric)} (us)")
    ax.set_title(f"{label_map.get(metric, metric)} ao longo do tempo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, f"{metric}_over_time.png")


def plot_summary_bars(table: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out = {}
    bar_specs = [
        ("Throughput (req/s)", "throughput_bar.png", "Throughput medio (req/s)"),
        ("Latencia media (us)", "avg_latency_bar.png", "Latencia media (us)"),
        ("P95 (us)", "p95_latency_bar.png", "Latencia P95 (us)"),
        ("P99 (us)", "p99_latency_bar.png", "Latencia P99 (us)"),
    ]
    for col, filename, title in bar_specs:
        if col not in table.columns or table[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(table["Teste"], table[col], color="#4C72B0")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(alpha=0.3, axis="y")
        out[filename] = _save(fig, out_dir, filename)
    return out


def generate_report(runs: list[TestRun], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(runs)
    table.to_csv(out_dir / "comparison_summary.csv", index=False)

    images: dict[str, Path] = {}
    tp_img = plot_throughput_over_time(runs, out_dir)
    if tp_img:
        images["Throughput ao longo do tempo"] = tp_img
    for metric in ("avg_latency_us", "p95_latency_us", "p99_latency_us"):
        img = plot_latency_over_time(runs, out_dir, metric)
        if img:
            images[img.stem.replace("_", " ")] = img
    images.update({k.replace(".png", "").replace("_", " "): v for k, v in plot_summary_bars(table, out_dir).items()})

    _write_html_report(runs, table, images, out_dir / "report.html")

    return out_dir


def _write_html_report(runs: list[TestRun], table: pd.DataFrame, images: dict[str, Path], out_path: Path) -> None:
    rows_html = table.to_html(index=False, na_rep="-", border=0, classes="tbl")

    warn_html = ""
    warns = [(r.label, w) for r in runs for w in r.warnings]
    if warns:
        items = "".join(f"<li><b>{html.escape(lbl)}</b>: {html.escape(w)}</li>" for lbl, w in warns)
        warn_html = f"<h2>Avisos</h2><ul>{items}</ul>"

    imgs_html = "".join(
        f'<div class="chart"><h3>{html.escape(title)}</h3>'
        f'<img src="{img.name}" alt="{html.escape(title)}"></div>'
        for title, img in images.items()
    )

    css = """
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; color:#1a1a1a; }
    h1 { margin-bottom: 4px; }
    .subtitle { color:#666; margin-top:0; }
    table.tbl { border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0 32px; }
    table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
    table.tbl th { background: #f4f6f8; text-align: right; }
    table.tbl td:first-child, table.tbl th:first-child { text-align: left; }
    .chart { margin-bottom: 32px; }
    .chart img { max-width: 100%; border: 1px solid #eee; }
    """

    doc = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatorio de comparacao de benchmarks</title>
<style>{css}</style>
</head>
<body>
<h1>Relatorio de comparacao de benchmarks</h1>
<p class="subtitle">{len(runs)} teste(s) analisado(s)</p>
<h2>Resumo comparativo</h2>
{rows_html}
<h2>Graficos</h2>
{imgs_html}
{warn_html}
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
