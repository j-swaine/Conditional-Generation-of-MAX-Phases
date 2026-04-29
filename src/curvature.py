"""
A-site well curvature calculation and visualisation.

Implements the finite-difference estimate of the second derivative of
E_hull with respect to A-site displacement used as a condition-vector
surrogate for A-site binding strength (see Methods IV.D and Eq. 5 of
the paper), and provides a paired boxplot for Fig. 3.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def calc_curvature(
    df: pd.DataFrame,
    *,
    delta: float = 0.01,
    e0_col: str = "ehull_mace_og",
    em_col: str = "ehull_mace_minus",
    ep_col: str = "ehull_mace_plus",
    out_col: str = "curvature",
    present_col: str = "curvature_present",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Central-difference curvature:

        E''(0) ≈ [E(+delta) + E(-delta) - 2 E(0)] / delta^2

    Non-finite results are imputed with the column mean across the finite
    entries. A boolean `present_col` records which rows had a finite raw
    estimate prior to imputation.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    out = df if inplace else df.copy()

    try:
        E0 = out[e0_col].astype(float)
        Em = out[em_col].astype(float)
        Ep = out[ep_col].astype(float)
    except KeyError as e:
        raise KeyError(f"Missing required energy column: {e}") from None

    curvature = (Ep + Em - 2.0 * E0) / (delta ** 2)
    present = np.isfinite(curvature)

    if present.any():
        mean_val = curvature[present].mean()
        curvature = curvature.fillna(mean_val)
        curvature = curvature.replace([np.inf, -np.inf], mean_val)

    out[out_col] = curvature
    out[present_col] = present
    return out


def _parse_cv(x):
    """Coerce a condition vector stored as list / tuple / ndarray / string
    into a plain Python list, or NaN on failure."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple, np.ndarray)):
                return list(v)
        except Exception:
            pass
    return np.nan


def plot_cv_boxplots_two_axes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    cv_col: str = "Condition Vector",
    curv_col: str = "curvature",
    varying_index: int = 1,
    round_to: int = 2,
    labels: tuple[str, str] = ("Set A", "Set B"),
    title: str | None = "Curvature by Condition Vector",
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 150,
    save_path: str | None = None,
):
    """
    Paired boxplots of measured curvature against the value of one element
    of the condition vector, for two dataframes side-by-side. Annotates
    each axis with its Pearson correlation.
    """
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        cvp = df[cv_col].apply(_parse_cv)
        m = cvp.apply(lambda v: isinstance(v, list) and len(v) > varying_index)
        x = cvp[m].apply(lambda v: float(v[varying_index])).round(round_to)
        y = df.loc[m, curv_col].astype(float)
        keep = x.notna() & y.notna()
        return pd.DataFrame({"cv": x[keep], "y": y[keep]})

    d1, d2 = _prep(df1), _prep(df2)
    if d1.empty and d2.empty:
        raise ValueError("No valid rows after parsing CV and curvature.")

    all_bins = sorted(set(d1["cv"]).union(set(d2["cv"])))
    data1 = [d1.loc[d1["cv"] == b, "y"].values for b in all_bins]
    data2 = [d2.loc[d2["cv"] == b, "y"].values for b in all_bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    for ax, data, colour, lbl, d in [
        (ax1, data1, "#4C78A8", labels[0], d1),
        (ax2, data2, "#F58518", labels[1], d2),
    ]:
        bp = ax.boxplot(
            data, positions=np.arange(len(all_bins)), widths=0.6,
            patch_artist=True, showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_title(lbl, fontsize=18)
        ax.set_xticks(np.arange(len(all_bins)))
        ax.set_xticklabels([f"{b:.2f}" for b in all_bins], rotation=45, fontsize=10)
        ax.set_xlabel("Curvature condition vector", fontsize=16)

        r, p = stats.pearsonr(d["cv"], d["y"])
        ax.text(
            0.02, 0.98, f"R={r:.2f}, p={p:.2g}",
            transform=ax.transAxes, ha="left", va="top",
        )

    ax1.set_ylabel("Raw curvature", fontsize=16)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, (ax1, ax2)