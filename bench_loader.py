"""
Descobre subpastas de teste dentro de um diretorio raiz e carrega os
arquivos de cada uma delas (config.xml, summary.json, results*.csv,
samples.csv, raw.csv, metrics.json, params.json).

Cada subpasta deve conter arquivos com o padrao:
    <nome_do_teste>_<AAAA-MM-DD>_<HH-MM-SS><separador><sufixo>

onde o separador antes do sufixo pode ser "_" ou "." (as duas variantes
sao aceitas automaticamente, ex.:
    mytrace_2026-08-11_17-44-40_summary.json
    mytrace_2026-08-11_17-44-40.summary.json
)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from bench_parsers import (
    parse_config_xml,
    parse_json_kv,
    parse_raw_csv,
    parse_results_csv,
    parse_samples_csv,
    parse_summary_json,
)

# nome_<AAAA-MM-DD>_<HH-MM-SS>
_PREFIX_RE = re.compile(r"^(?P<name>.+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$")


@dataclass
class TestRun:
    label: str  # identificador legivel usado nos graficos/tabela
    test_name: str
    timestamp: str
    folder: Path
    prefix: str

    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    results: Optional[pd.DataFrame] = None
    samples: Optional[pd.DataFrame] = None
    raw: Optional[pd.DataFrame] = None
    per_txn_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _find_prefix(folder: Path) -> Optional[tuple[str, str]]:
    """Acha o prefixo comum dos arquivos e o separador usado (\"_\" ou \".\"),
    olhando para o arquivo *summary.json da pasta. Retorna (prefix, separador)
    ou None se nao encontrar."""
    for sep in ("_", "."):
        candidates = sorted(folder.glob(f"*{sep}summary.json"))
        if candidates:
            fname = candidates[0].name
            suffix = f"{sep}summary.json"
            return fname[: -len(suffix)], sep
    return None


def _split_prefix(prefix: str) -> tuple[str, str]:
    """Separa '<nome>_<data>_<hora>' em (nome, 'data hora')."""
    m = _PREFIX_RE.match(prefix)
    if m:
        return m.group("name"), f"{m.group('date')} {m.group('time')}"
    # fallback: usa o prefixo inteiro como nome, sem timestamp reconhecido
    return prefix, ""


def load_test_folder(folder: Path) -> Optional[TestRun]:
    found = _find_prefix(folder)
    if found is None:
        return None
    prefix, sep = found

    test_name, timestamp = _split_prefix(prefix)
    # O label exibido nas tabelas/graficos e o nome da pasta onde os arquivos
    # daquele teste estao (em vez de "<nome>_<timestamp>").
    label = folder.name
    run = TestRun(label=label, test_name=test_name, timestamp=timestamp, folder=folder, prefix=prefix)

    def p(suffix: str) -> Path:
        return folder / f"{prefix}{sep}{suffix}"

    # summary.json (obrigatorio para o teste ser considerado valido)
    summary_path = p("summary.json")
    try:
        run.summary = parse_summary_json(summary_path)
    except Exception as e:
        run.warnings.append(f"Falha ao ler summary.json: {e}")

    # config.xml
    config_path = p("config.xml")
    if config_path.exists():
        try:
            run.config = parse_config_xml(config_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler config.xml: {e}")

    # results.csv (agregado de todas as transacoes)
    results_path = p("results.csv")
    if results_path.exists():
        try:
            run.results = parse_results_csv(results_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler results.csv: {e}")

    # samples.csv
    samples_path = p("samples.csv")
    if samples_path.exists():
        try:
            run.samples = parse_samples_csv(samples_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler samples.csv: {e}")

    # raw.csv (pode ser grande)
    raw_path = p("raw.csv")
    if raw_path.exists():
        try:
            run.raw = parse_raw_csv(raw_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler raw.csv: {e}")

    # results<sep><TransactionName>.csv -> pode haver varios (multi-transacao)
    txn_prefix = f"{prefix}{sep}results{sep}"
    for f in sorted(folder.glob(f"{txn_prefix}*.csv")):
        # results.csv puro tambem bate no glob "*.csv" acima por causa do
        # sep=="." (prefix+".results." == prefix+".results" + "."); results.csv
        # sozinho nao tem o segundo separador+nome, entao filtramos por ele
        # ja ter sido tratado como results_path acima.
        if f == results_path:
            continue
        txn_name = f.name[len(txn_prefix):-len(".csv")]
        try:
            run.per_txn_results[txn_name] = parse_results_csv(f)
        except Exception as e:
            run.warnings.append(f"Falha ao ler results{sep}{txn_name}.csv: {e}")

    # metrics.json / params.json (snapshot do SGBD) - carregados mas nao
    # sao usados nos graficos por padrao
    metrics_path = p("metrics.json")
    if metrics_path.exists():
        try:
            run.metrics = parse_json_kv(metrics_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler metrics.json: {e}")

    params_path = p("params.json")
    if params_path.exists():
        try:
            run.params = parse_json_kv(params_path)
        except Exception as e:
            run.warnings.append(f"Falha ao ler params.json: {e}")

    return run


def discover_tests(root_dir: str | Path) -> list[TestRun]:
    """Percorre as subpastas de root_dir e carrega cada teste encontrado."""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} nao e um diretorio valido")

    runs: list[TestRun] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        run = load_test_folder(sub)
        if run is not None:
            runs.append(run)

    # tambem aceita chamar discover_tests apontando direto pra uma pasta de teste unica
    if not runs:
        run = load_test_folder(root)
        if run is not None:
            runs.append(run)

    return runs
