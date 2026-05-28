"""Plot a CSV containing 'actual' and 'predicted' float columns as line graphs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Read ``csv_path`` and line-plot the ``actual`` and ``predicted`` columns.

    If ``output_path`` is given the figure is written there. If ``show`` is
    True (and not running headless) the figure is also displayed.
    """
    import matplotlib.pyplot as plt

    frame = pd.read_csv(csv_path)
    missing = {"actual", "predicted"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {sorted(missing)}"
        )

    actual = frame["actual"].astype(float)
    predicted = frame["predicted"].astype(float)
    diff = (
        frame["diff"].astype(float)
        if "diff" in frame.columns
        else predicted - actual
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual.index, actual.values, label="actual", linewidth=1.2)
    ax.plot(predicted.index, predicted.values, label="predicted", linewidth=1.2)
    ax.plot(
        diff.index,
        diff.values,
        label="diff (pred - actual)",
        linewidth=1.0,
        color="tab:red",
        alpha=0.7,
    )
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("sample index")
    ax.set_ylabel("value")
    ax.set_title(f"actual vs predicted ({Path(csv_path).name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote plot to %s", output_path)

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Plot 'actual' vs 'predicted' float columns from a CSV file as a "
            "line graph (e.g. the predictions CSV written by "
            "algoplayground.evaluate)."
        )
    )
    parser.add_argument("csv_path", help="Path to a CSV with 'actual' and 'predicted' columns.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="If set, save the figure to this path instead of (or as well as) showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive window (useful when only saving with --output).",
    )
    args = parser.parse_args()

    plot_actual_vs_predicted(
        args.csv_path,
        output_path=args.output,
        show=not args.no_show,
    )