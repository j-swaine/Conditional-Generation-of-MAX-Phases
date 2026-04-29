"""
Novel stable structure generation rates under conditional prompting.

Combines the per-condition-vector Fisher's exact test of Fig. 4 with the
paired bar chart used to visualise generation counts against the
non-conditional baseline. The sweep space covers 15 condition vectors:
MXene derivative dimension in {0.75, 0.85, 0.95} crossed with A-site
curvature dimension in {0.00, 0.10, 0.20, 0.30, 0.40}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------

@dataclass
class FisherResult:
    table: list
    odds_ratio: float
    p_value: float
    alternative: str
    notes: str


def fisher_exact_test(
    model_successes: int,
    model_trials: int,
    baseline_successes: int,
    baseline_trials: int,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> FisherResult:
    """Run a 2x2 Fisher's exact test for the null hypothesis that the
    conditioning regime does not alter the rate of novel stable structure
    generation relative to the non-conditional baseline."""
    m_fail = model_trials - model_successes
    b_fail = baseline_trials - baseline_successes
    table = [[model_successes, m_fail], [baseline_successes, b_fail]]
    odds_ratio, p_value = fisher_exact(table, alternative=alternative)
    return FisherResult(
        table=table,
        odds_ratio=odds_ratio,
        p_value=p_value,
        alternative=alternative,
        notes=f"Table: {table}",
    )


def extract_cv_tuple(cv) -> tuple[float, float] | None:
    """Normalise a condition vector to a two-element tuple rounded to 2 d.p."""
    if cv is None:
        return None
    try:
        if isinstance(cv, np.ndarray) and len(cv) >= 2:
            return (round(float(cv[0]), 2), round(float(cv[1]), 2))
        if isinstance(cv, (list, tuple)) and len(cv) >= 2:
            return (round(float(cv[0]), 2), round(float(cv[1]), 2))
    except (TypeError, ValueError):
        pass
    return None


def summarise_sweep(
    df_pkv_novel: pd.DataFrame,
    df_slider_novel: pd.DataFrame,
    df_baseline_novel: pd.DataFrame,
    *,
    cv_col: str = "_Condition Vector",
    total_generations: int = 19_800,
    n_condition_vectors: int = 15,
    specific_cvs: list[tuple[float, float]] | None = None,
    verbose: bool = True,
) -> dict:
    """Print the overall novel-stable generation rates and Fisher's exact
    test results for the condition vectors reported in the paper.

    Returns a dict containing aggregate rates and per-CV Fisher results."""
    if specific_cvs is None:
        specific_cvs = [(0.75, 0.40), (0.85, 0.40), (0.75, 0.30), (0.95, 0.40)]

    trials_per_cv = total_generations / n_condition_vectors

    pkv_total = len(df_pkv_novel)
    slider_total = len(df_slider_novel)
    baseline_total = len(df_baseline_novel)

    pkv_rate = pkv_total / total_generations * 100
    slider_rate = slider_total / total_generations * 100
    baseline_rate = baseline_total / total_generations * 100

    pkv_by_cv = df_pkv_novel[cv_col].apply(extract_cv_tuple).value_counts()
    slider_by_cv = df_slider_novel[cv_col].apply(extract_cv_tuple).value_counts()

    baseline_avg_per_cv = baseline_total / n_condition_vectors
    cv_trials = int(trials_per_cv)

    fisher_results: dict[tuple[float, float], dict[str, FisherResult]] = {}
    for target_cv in specific_cvs:
        pkv_s = int(pkv_by_cv.get(target_cv, 0))
        sld_s = int(slider_by_cv.get(target_cv, 0))
        bsl_s = int(baseline_avg_per_cv)

        fisher_results[target_cv] = {
            "pkv": fisher_exact_test(pkv_s, cv_trials, bsl_s, cv_trials, "greater"),
            "slider": fisher_exact_test(sld_s, cv_trials, bsl_s, cv_trials, "greater"),
        }

    if verbose:
        bar = "=" * 70
        print(bar)
        print("Overall sweep space: MXene in {0.75,0.85,0.95} x curvature in {0.0..0.40}")
        print(bar)
        print(f"PKV_prefix:   {pkv_total} novel stable structures, rate = {pkv_rate:.2f}%")
        print(f"PKV_residual: {slider_total} novel stable structures, rate = {slider_rate:.2f}%")
        print(f"Baseline:     {baseline_total} novel stable structures, rate = {baseline_rate:.2f}%")
        print(f"Baseline mean per condition vector: {baseline_avg_per_cv:.1f} (of {cv_trials} trials)")
        print()
        print(bar)
        print("Fisher's exact test (alternative = greater) for reported condition vectors")
        print(bar)
        for cv, res in fisher_results.items():
            pkv_r = res["pkv"]
            sld_r = res["slider"]
            print(f"\nCondition vector {cv}:")
            print(f"  PKV_prefix   OR={pkv_r.odds_ratio:.2f}  p={pkv_r.p_value:.4g}")
            print(f"  PKV_residual OR={sld_r.odds_ratio:.2f}  p={sld_r.p_value:.4g}")

    return {
        "totals": {
            "pkv": pkv_total,
            "slider": slider_total,
            "baseline": baseline_total,
        },
        "rates": {
            "pkv": pkv_rate,
            "slider": slider_rate,
            "baseline": baseline_rate,
        },
        "baseline_avg_per_cv": baseline_avg_per_cv,
        "fisher_results": fisher_results,
    }


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _cv_col(df: pd.DataFrame) -> str:
    for c in ("_Condition Vector", "Condition Vector", "condition_vector"):
        if c in df.columns:
            return c
    raise ValueError("No condition vector column found")


def _label_from_vec(vec) -> str:
    a, b = float(vec[0]), float(vec[1])
    return f"{a:.2f}, {b:.2f}"


def _summary_by_cv(df: pd.DataFrame) -> dict[str, int]:
    col = _cv_col(df)
    labels = df[col].map(_label_from_vec)
    return labels.value_counts().to_dict()


def plot_barchart_stability_sweep(
    df_pkv: pd.DataFrame,
    df_slider: pd.DataFrame,
    *,
    name_pkv: str = "PKV_prefix",
    name_slider: str = "PKV_residual",
    baseline: int | None = None,
    top_n: int | None = None,
    save_path: str | None = None,
    verbose: bool = False,
):
    """Paired bar chart of novel stable structure counts per condition
    vector, with an optional horizontal line marking the non-conditional
    baseline mean. Reproduces Fig. 4."""
    s_pkv = _summary_by_cv(df_pkv)
    s_sld = _summary_by_cv(df_slider)

    if verbose:
        print(f"[info] PKV stable total: {len(df_pkv)} | grouped sum: {sum(s_pkv.values())}")
        print(f"[info] Slider stable total: {len(df_slider)} | grouped sum: {sum(s_sld.values())}")
        print(f"[info] Baseline (mean per CV): {baseline}")

    labels = list(set(s_pkv) | set(s_sld))
    if top_n is not None and top_n > 0:
        labels = sorted(
            labels,
            key=lambda k: s_pkv.get(k, 0) + s_sld.get(k, 0),
            reverse=True,
        )[:top_n]

    def _key(lbl: str):
        a, b = (float(x) for x in lbl.split(", "))
        return (a, b)

    labels = sorted(labels, key=_key)
    v_pkv = [s_pkv.get(k, 0) for k in labels]
    v_sld = [s_sld.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, v_pkv, width, label=name_pkv, color="#4C78A8")
    ax.bar(x + width / 2, v_sld, width, label=name_slider, color="#F58518")
    if baseline is not None:
        ax.axhline(
            baseline, linestyle="dotted",
            label="Non-conditional baseline (mean per CV)", color="black",
        )

    ax.set_ylabel("Number of stable structures", fontsize=18)
    ax.set_xlabel("Condition vector", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=12)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}" if v.is_integer() else "")
    )
    ax.set_title("Novel stable structure count per condition vector", fontsize=20)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, transparent=True)
    return fig, ax