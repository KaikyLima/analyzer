from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Schemas: lista de (nome_da_coluna, tipo)
#   "int"   -> consome 1 token
#   "float" -> consome 2 tokens (parte inteira + parte decimal, juntos com ".")
#   "str"   -> consome 1 token (texto)
# ---------------------------------------------------------------------------

SCHEMA_RESULTS = [
    ("time_s", "int"),
    ("throughput", "float"),
    ("avg_latency_ms", "float"),
    ("min_latency_ms", "float"),
    ("p25_latency_ms", "float"),
    ("median_latency_ms", "float"),
    ("p75_latency_ms", "float"),
    ("p90_latency_ms", "float"),
    ("p95_latency_ms", "float"),
    ("p99_latency_ms", "float"),
    ("max_latency_ms", "float"),
    ("tp_scaled", "float"),
]

SCHEMA_SAMPLES = [
    ("time_s", "int"),
    ("requests", "int"),
    ("throughput", "float"),
    ("min_latency_us", "int"),
    ("p25_latency_us", "int"),
    ("median_latency_us", "int"),
    ("avg_latency_us", "int"),
    ("p75_latency_us", "int"),
    ("p90_latency_us", "int"),
    ("p95_latency_us", "int"),
    ("p99_latency_us", "int"),
    ("max_latency_us", "int"),
]

SCHEMA_RAW = [
    ("txn_type_index", "int"),
    ("txn_name", "str"),
    ("start_time_s", "float"),
    ("latency_us", "int"),
    ("worker_id", "int"),
    ("phase_id", "int"),
]


def _parse_row(tokens: list[str], schema: list[tuple[str, str]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    i = 0
    for name, kind in schema:
        if kind == "int":
            row[name] = int(tokens[i])
            i += 1
        elif kind == "float":
            row[name] = float(f"{tokens[i]}.{tokens[i + 1]}")
            i += 2
        else:  # str
            row[name] = tokens[i]
            i += 1
    if i != len(tokens):
        raise ValueError(
            f"Numero de tokens nao bate com o schema esperado "
            f"(consumidos {i}, recebidos {len(tokens)}): {tokens}"
        )
    return row


def parse_quirky_csv(path: str | Path, schema: list[tuple[str, str]]) -> pd.DataFrame:
    """Le um CSV com o problema de virgula-decimal e retorna um DataFrame limpo."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        lines = f.read().splitlines()
    if not lines:
        return pd.DataFrame(columns=[c for c, _ in schema])
    # primeira linha e o cabecalho -> ignorar (usamos nomes do schema)
    for line in lines[1:]:
        line = line.strip().strip("\ufeff")
        if not line:
            continue
        tokens = line.split(",")
        rows.append(_parse_row(tokens, schema))
    return pd.DataFrame(rows, columns=[c for c, _ in schema])


def parse_results_csv(path: str | Path) -> pd.DataFrame:
    return parse_quirky_csv(path, SCHEMA_RESULTS)


def parse_samples_csv(path: str | Path) -> pd.DataFrame:
    return parse_quirky_csv(path, SCHEMA_SAMPLES)


def parse_raw_csv(path: str | Path) -> pd.DataFrame:
    return parse_quirky_csv(path, SCHEMA_RAW)


def parse_summary_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_config_xml(path: str | Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    def _text(tag: str) -> str | None:
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    works = []
    for w in root.findall(".//works/work"):
        works.append(
            {
                "time": _text_of(w, "time"),
                "rate": _text_of(w, "rate"),
                "weights": _text_of(w, "weights"),
            }
        )

    txn_types = [
        t.findtext("name", "").strip()
        for t in root.findall(".//transactiontypes/transactiontype")
    ]

    return {
        "isolation": _text("isolation"),
        "batchsize": _text("batchsize"),
        "scalefactor": _text("scalefactor"),
        "terminals": _text("terminals"),
        "works": works,
        "transaction_types": [t for t in txn_types if t],
    }


def _text_of(el, tag: str) -> str | None:
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def parse_json_kv(path: str | Path) -> dict:
    """Para metrics.json / params.json: json simples chave->valor (strings)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
