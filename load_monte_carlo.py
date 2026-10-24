"""
Skrypt do wyswietlania macierzy AHP z datasetu Monte Carlo.

Uzycie (z katalogu glownego projektu):
  python load_monte_carlo.py -n 4 -p test -s extreme -c 3
  python load_monte_carlo.py -n 3 4 5 -p train -s high -c 5 --weights
  python load_monte_carlo.py -n 4                          # domyslnie: train, all, 5 macierzy
  python load_monte_carlo.py -n 4 --info                   # statystyki datasetu
  python load_monte_carlo.py -n 3 4 --weights              # tylko wektory wag dla n=3 i n=4
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Mapowania krotkich nazw na pelne wartosci
# ---------------------------------------------------------------------------

PHASE_ALIASES: dict[str, str] = {
    "train":      "train",
    "tr":         "train",
    "validation": "validation",
    "valid":      "validation",
    "val":        "validation",
    "test":       "test",
    "te":         "test",
}

STRATUM_ALIASES: dict[str, str] = {
    "low":                 "low_inconsistency",
    "low_inconsistency":     "low_inconsistency",
    "moderate":             "moderate_inconsistency",
    "mod":                  "moderate_inconsistency",
    "moderate_inconsistency": "moderate_inconsistency",
    "high":                 "high_inconsistency",
    "high_inconsistency":  "high_inconsistency",
    "all":                  "all",
}

STRATA_ALL = ["low_inconsistency", "moderate_inconsistency", "high_inconsistency"]

STRATUM_LABELS: dict[str, str] = {
    "low_inconsistency":     "Low (CR <= 0.10)",
    "moderate_inconsistency": "Moderate (0.10 < CR <= 0.33)",
    "high_inconsistency":  "High (CR > 0.33)",
}

RANDOM_INDEX: dict[int, float] = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
}


# ---------------------------------------------------------------------------
# Obliczanie CR
# ---------------------------------------------------------------------------

def _compute_cr(matrix: np.ndarray) -> float:
    """Oblicza Consistency Ratio metoda EVM."""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    lambda_max = float(np.max(np.real(eigenvalues)))
    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX.get(n, 1.45)
    return ci / ri if ri > 0 else 0.0


# ---------------------------------------------------------------------------
# Formatowanie
# ---------------------------------------------------------------------------

def _format_fraction(val: float) -> str:
    """Zwraca wartosc jako ulamek Saaty'ego lub liczbe calkowita."""
    if val >= 1.0:
        return str(round(val))
    for denom in range(2, 10):
        if abs(val - 1.0 / denom) < 0.05:
            return f"1/{denom}"
    return f"{val:.3f}"


def _print_matrix(idx: int, matrix: np.ndarray, weights: np.ndarray) -> None:
    """Drukuje macierz AHP z wagami i CR."""
    n = matrix.shape[0]
    cr = _compute_cr(matrix)

    header = f"  Macierz #{idx + 1}  |  n={n}  |  CR = {cr:.4f}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    col_w = 7
    col_labels = "  " + " " * 5 + "".join(f"  C{j+1:>2} " for j in range(n))
    print(col_labels)

    for i in range(n):
        row_label = f"  C{i+1:>2} |"
        cells = "".join(f"{_format_fraction(matrix[i, j]):>{col_w}}" for j in range(n))
        print(row_label + cells)

    weights_str = "  ".join(f"C{j+1}={weights[j]:.4f}" for j in range(n))
    print(f"\n  Wagi: {weights_str}")
    print()


def _print_weights_only(idx: int, weights: np.ndarray, num_criteria: int) -> None:
    """Drukuje tylko wektor wag bez macierzy."""
    n = len(weights)
    bar_width = 30

    header = f"  Wagi #{idx + 1}  |  n={num_criteria}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for j in range(n):
        w = weights[j]
        bar = "#" * int(w * bar_width)
        print(f"  C{j+1:>2}: {w:.6f}  |{bar:<{bar_width}}|")
    print()


# ---------------------------------------------------------------------------
# Funkcja informacyjna
# ---------------------------------------------------------------------------

def _print_info(loader, criteria_list: list[int]) -> None:
    """Wyswietla statystyki datasetu dla kazdej z podanych liczb kryteriow."""
    for num_criteria in criteria_list:
        info = loader.dataset_info(num_criteria)
        total = 0
        print(f"\n  Statystyki  |  criteria = {num_criteria}\n")
        print(f"  {'Faza':<12}  {'Stratum':<26}  {'Macierzy':>10}")
        print("  " + "-" * 52)
        for phase_name, strata_dict in info.items():
            for stratum_name, count in strata_dict.items():
                label = STRATUM_LABELS.get(stratum_name, stratum_name)
                print(f"  {phase_name:<12}  {label:<26}  {count:>10}")
                total += count
        print("  " + "-" * 52)
        print(f"  {'RAZEM':<40}  {total:>10}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyswietla macierze AHP lub wektory wag z datasetu Monte Carlo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przyklady:\n"
            "  python load_monte_carlo.py -n 4 -p test -s extreme -c 3\n"
            "  python load_monte_carlo.py -n 3 4 5 -p train -s high -c 5 --weights\n"
            "  python load_monte_carlo.py -n 3 4 --weights\n"
            "  python load_monte_carlo.py -n 4 --info\n"
        ),
    )
    parser.add_argument(
        "-n", "--criteria",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="Liczba kryteriow (3-9). Mozna podac kilka, np. -n 3 4 5.",
    )
    parser.add_argument(
        "-p", "--phase",
        type=str,
        default="train",
        metavar="PHASE",
        help="Zbior danych: train | valid | test  (domyslnie: train).",
    )
    parser.add_argument(
        "-s", "--stratum",
        type=str,
        default="all",
        metavar="STRATUM",
        help="Poziom niespojnosci: high | moderate | extreme | all  (domyslnie: all).",
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=5,
        metavar="N",
        help="Liczba probek do wyswietlenia na kryterium (domyslnie: 5).",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        default=False,
        help="Wyswietl tylko wektory wag (bez macierzy).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        metavar="ALPHA",
        help=(
            "Filtruj wagi wedlug wartosci Dirichleta alfa uzytej przy generowaniu "
            "(np. --alpha 2.0 -> szuka weights_alpha2_00.npy)."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help=(
            "Filtruj macierze wedlug wartosci sigma uzytej przy generowaniu "
            "(np. --sigma 0.9 -> szuka matrices_sigma0_90.npy)."
        ),
    )
    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Wyswietl tylko statystyki datasetu.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Walidacja criteria
    invalid = [n for n in args.criteria if not (3 <= n <= 9)]
    if invalid:
        print(f"Blad: --criteria musi byc w zakresie 3-9, nieprawidlowe: {invalid}", file=sys.stderr)
        sys.exit(1)

    # Rozwiazanie aliasow fazy
    phase_key = PHASE_ALIASES.get(args.phase.lower())
    if phase_key is None:
        print(f"Blad: nieznana faza '{args.phase}'. Dopuszczalne: {list(PHASE_ALIASES)}", file=sys.stderr)
        sys.exit(1)

    # Rozwiazanie aliasow stratum
    stratum_key = STRATUM_ALIASES.get(args.stratum.lower())
    if stratum_key is None:
        print(f"Blad: nieznany stratum '{args.stratum}'. Dopuszczalne: {list(STRATUM_ALIASES)}", file=sys.stderr)
        sys.exit(1)

    from utils.phase import Phase
    from data_augmentation.monte_carlo_loader import MonteCarloLoader

    phase_enum = Phase(phase_key)
    loader = MonteCarloLoader()
    strata = STRATA_ALL if stratum_key == "all" else [stratum_key]

    # Tryb informacyjny
    if args.info:
        _print_info(loader, sorted(args.criteria))
        return

    # Tryb wyswietlania (macierze lub wagi)
    mode_label = "wagi" if args.weights else "macierze"

    for num_criteria in sorted(args.criteria):
        print()
        alpha_info = f"  alpha={args.alpha}" if args.alpha is not None else ""
        sigma_info  = f"  sigma={args.sigma}"  if args.sigma  is not None else ""
        print(f"  criteria={num_criteria}  phase={phase_key}  stratum={args.stratum}"
              f"  count={args.count}  tryb={mode_label}{alpha_info}{sigma_info}")
        print("  " + "=" * 60)
        print()

        try:
            matrices, weights = loader.load(
                phase=phase_enum,
                num_criteria=num_criteria,
                strata=strata,
                max_per_stratum=args.count,
                alpha=args.alpha,
                sigma=args.sigma,
            )
        except FileNotFoundError as exc:
            print(f"  Blad: {exc}\n", file=sys.stderr)
            continue

        limit = min(args.count, len(matrices))
        if limit < args.count:
            print(f"  Dostepnych probek: {len(matrices)} (zadan {args.count})\n")

        for i in range(limit):
            if args.weights:
                _print_weights_only(i, weights[i], num_criteria)
            else:
                _print_matrix(i, matrices[i], weights[i])


if __name__ == "__main__":
    main()
