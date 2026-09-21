from __future__ import annotations

import html
import re
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


def _smoothing_window(n_points: int) -> int:
    """Escolhe o tamanho da janela de media movel com base na quantidade de
    amostras, para suavizar o ruido ponto-a-ponto sem esconder a tendencia.
    Ex.: 300 amostras (1 por segundo, 300s de teste) -> janela de 15."""
    window = max(3, round(n_points * 0.05))
    return min(window, 31)


def plot_throughput_over_time(runs: list[TestRun], out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    max_window = 3
    for r in runs:
        if r.samples is None or r.samples.empty:
            continue
        window = _smoothing_window(len(r.samples))
        max_window = max(max_window, window)
        smoothed = r.samples["throughput"].rolling(window, center=True, min_periods=1).mean()
        ax.plot(r.samples["time_s"], smoothed, label=r.label, linewidth=1.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title(f"Throughput ao longo do tempo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "throughput_over_time.png")


def plot_latency_over_time(runs: list[TestRun], out_dir: Path, metric: str = "avg_latency_us") -> Path | None:
    label_map = {
        "avg_latency_us": "Latencia media",
        "median_latency_us": "Latencia mediana",
        "p95_latency_us": "Latencia P95",
        "p99_latency_us": "Latencia P99",
        "max_latency_us": "Latencia maxima",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    max_window = 3
    for r in runs:
        if r.samples is None or r.samples.empty or metric not in r.samples.columns:
            continue
        window = _smoothing_window(len(r.samples))
        max_window = max(max_window, window)
        smoothed = r.samples[metric].rolling(window, center=True, min_periods=1).mean()
        ax.plot(r.samples["time_s"], smoothed, label=r.label, linewidth=1.8)
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


def _natural_key(s: str):
    """Ordena 'Q1', 'Q2', ..., 'Q10', 'Q22' na ordem numerica correta em vez
    de alfabetica (que colocaria Q10 antes de Q2)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", s)]


def build_per_query_table(runs: list[TestRun]) -> pd.DataFrame:
    """Uma linha por (query, teste), com media/min/max de latencia e
    throughput medio, a partir dos arquivos results_<Query>.csv de cada
    teste. O conjunto e a quantidade de queries dependem do tipo de
    benchmark (ex.: TPC-H vai ate Q22), entao aqui usamos o que cada teste
    realmente trouxe."""
    rows = []
    for r in runs:
        for txn_name, df in r.per_txn_results.items():
            if df is None or df.empty:
                continue
            # Quando nao ha nenhuma requisicao num intervalo, a ferramenta
            # grava valores-sentinela (ex.: latencia -0,001, tp_scaled
            # -1000,000) em vez de deixar em branco. Esses intervalos nao
            # sao amostras reais de latencia e distorcem o min/max/media,
            # entao sao descartados aqui.
            valid = df[df["throughput"] > 0]
            if valid.empty:
                continue
            rows.append(
                {
                    "Query": txn_name,
                    "Teste": r.label,
                    "Latencia media (ms)": valid["avg_latency_ms"].mean(),
                    "Latencia minima (ms)": valid["min_latency_ms"].min(),
                    "Latencia maxima (ms)": valid["max_latency_ms"].max(),
                    "Throughput medio (req/s)": valid["throughput"].mean(),
                }
            )
    cols = ["Query", "Teste", "Latencia media (ms)", "Latencia minima (ms)", "Latencia maxima (ms)", "Throughput medio (req/s)"]
    if not rows:
        return pd.DataFrame(columns=cols)
    table = pd.DataFrame(rows, columns=cols)
    table["_sort"] = table["Query"].map(lambda q: tuple(_natural_key(q)))
    table = table.sort_values(["_sort", "Teste"]).drop(columns="_sort").reset_index(drop=True)
    return table


def plot_per_query_avg_latency(per_query: pd.DataFrame, out_dir: Path) -> Path | None:
    """Grafico de barras agrupadas: uma barra por teste, para cada query,
    comparando a latencia media."""
    if per_query.empty:
        return None
    queries = sorted(per_query["Query"].unique(), key=_natural_key)
    tests = list(dict.fromkeys(per_query["Teste"]))  # mantem ordem de aparicao
    n_tests = max(len(tests), 1)
    width = 0.8 / n_tests

    fig, ax = plt.subplots(figsize=(max(10, len(queries) * 0.55 * n_tests * 0.5), 5.5))
    for i, test in enumerate(tests):
        sub = per_query[per_query["Teste"] == test].set_index("Query")["Latencia media (ms)"]
        values = [sub.get(q, float("nan")) for q in queries]
        offset = (i - (n_tests - 1) / 2) * width
        positions = [j + offset for j in range(len(queries))]
        ax.bar(positions, values, width=width, label=test)

    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels(queries, rotation=45, ha="right")
    ax.set_xlabel("Query")
    ax.set_ylabel("Latencia media (ms)")
    ax.set_title("Latencia media por consulta - comparacao entre testes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir, "per_query_avg_latency.png")



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
    lat_img = plot_latency_over_time(runs, out_dir, "avg_latency_us")
    if lat_img:
        images[lat_img.stem.replace("_", " ")] = lat_img

    per_query_table = build_per_query_table(runs)
    if not per_query_table.empty:
        per_query_table.to_csv(out_dir / "per_query_summary.csv", index=False)
        pq_img = plot_per_query_avg_latency(per_query_table, out_dir)
        if pq_img:
            images["Latencia media por consulta"] = pq_img

    _write_html_report(runs, table, images, per_query_table, out_dir / "report.html")

    return out_dir


def _build_query_cards_html(per_query: pd.DataFrame) -> str:
    """Monta uma caixinha por query com o minimo/maximo de latencia de cada
    teste, lado a lado, para comparacao rapida."""
    if per_query.empty:
        return ""

    queries = sorted(per_query["Query"].unique(), key=_natural_key)
    cards = []
    for q in queries:
        sub = per_query[per_query["Query"] == q]
        rows_html = "".join(
            f'<div class="query-row">'
            f'<span class="query-test-name">{html.escape(str(row["Teste"]))}</span>'
            f'<span>min: <b>{row["Latencia minima (ms)"]:.2f} ms</b></span>'
            f'<span>max: <b>{row["Latencia maxima (ms)"]:.2f} ms</b></span>'
            f'</div>'
            for _, row in sub.iterrows()
        )
        cards.append(
            f'<div class="query-card"><h4>{html.escape(q)}</h4>{rows_html}</div>'
        )
    return f'<div class="query-grid">{"".join(cards)}</div>'


def _write_html_report(
    runs: list[TestRun],
    table: pd.DataFrame,
    images: dict[str, Path],
    per_query_table: pd.DataFrame,
    out_path: Path,
) -> None:
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

    query_cards_html = _build_query_cards_html(per_query_table)
    query_section_html = (
        f"<h2>Minimo / maximo por consulta</h2>{query_cards_html}" if query_cards_html else ""
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
    .query-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin: 16px 0 32px; }
    .query-card { border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; background: #fafbfc; }
    .query-card h4 { margin: 0 0 6px; font-size: 14px; }
    .query-row { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; padding: 2px 0; border-top: 1px solid #eee; }
    .query-row:first-of-type { border-top: none; }
    .query-test-name { color: #555; max-width: 45%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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
{query_section_html}
{warn_html}
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
