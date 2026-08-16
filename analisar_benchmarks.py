from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_loader import discover_tests
from bench_report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa e compara resultados de benchmarks de banco de dados.")
    parser.add_argument("pasta_raiz", help="Pasta contendo as subpastas de cada teste")
    parser.add_argument(
        "--saida", "-o", default=None,
        help="Pasta onde salvar o relatorio, graficos e CSV (padrao: <pasta_raiz>/_relatorio)",
    )
    args = parser.parse_args()

    root = Path(args.pasta_raiz).expanduser().resolve()
    out_dir = Path(args.saida).expanduser().resolve() if args.saida else root / "_relatorio"

    print(f"Procurando testes em: {root}")
    runs = discover_tests(root)

    if not runs:
        print("Nenhum teste encontrado. Verifique se a pasta contem subpastas com os arquivos "
              "'*_summary.json', '*_results.csv', '*_samples.csv' etc.")
        sys.exit(1)

    print(f"{len(runs)} teste(s) encontrado(s):")
    for r in runs:
        flag = " (com avisos)" if r.warnings else ""
        print(f"  - {r.label}{flag}")
        for w in r.warnings:
            print(f"      ! {w}")

    result_dir = generate_report(runs, out_dir)
    print(f"\nRelatorio gerado em: {result_dir}")
    print(f"  - {result_dir / 'report.html'}  (abra no navegador)")
    print(f"  - {result_dir / 'comparison_summary.csv'}")
    print(f"  - graficos .png na mesma pasta")


if __name__ == "__main__":
    main()
