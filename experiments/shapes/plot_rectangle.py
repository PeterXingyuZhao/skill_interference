"""Render the completed rectangle trajectory from published endpoint summaries."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.ticker import PercentFormatter


def read_b_endpoints(path: Path) -> list[dict[str, str]]:
    """Read the base and sixteen final B endpoints, rejecting incomplete data."""
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] in {"base", "B"}]
    if [int(row["round"]) for row in rows] != list(range(17)):
        raise ValueError("Expected base followed by B01 through B16")
    for row in rows:
        count, total = int(row["whole_word_mentions"]), int(row["draws"])
        if total != 400 or not 0 <= count <= total:
            raise ValueError("Invalid sample count")
        if abs(float(row["whole_word_mention_percent"]) - 100 * count / total) > 1e-9:
            raise ValueError("Mention percentage does not match counts")
        if not 0 <= float(row["target_prefix_probability_percent"]) <= 100:
            raise ValueError("Invalid prefix probability")
    return rows


def main() -> None:
    """Write the PNG beside the other published shape figures."""
    root = Path(__file__).resolve().parent
    rows = read_b_endpoints(root / "data/rectangle_all_final_endpoints.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x = [int(row["round"]) for row in rows]
    fields = ["target_prefix_probability_percent", "whole_word_mention_percent"]
    titles = ["Full-content prefix probability", "Whole-word mentions"]
    for ax, field, title in zip(axes, fields, titles, strict=True):
        y = [float(row[field]) for row in rows]
        ax.plot(x, y, "o-", color="#236fa6", linewidth=2)
        ax.set_title(title, loc="left", pad=15)
        ax.set_xticks(range(0, 17, 2), ["Base"] + [str(n) for n in range(2, 17, 2)])
        ax.set_xlabel("Completed B round")
        ax.set_ylim(-3, 103)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", alpha=0.2)
        ax.annotate(f"{y[-1]:.2f}%", (16, y[-1]), xytext=(-5, 15), textcoords="offset points", ha="right")
    fig.suptitle("Rectangle-only recursion · Qwen3-8B · complete through B16", fontsize=17)
    fig.text(0.07, 0.065, "20 neutral prompts × 20 samples. Mentions include negations and truncation; probability excludes EOS.", fontsize=10)
    fig.text(0.07, 0.025, "One seed. Replotted from preserved completion summaries; no new inference.", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.13, 0.99, 0.93))
    output = root / "figures/rectangle_B16.png"
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    logger.success("Wrote {}", output.name)


if __name__ == "__main__":
    main()
