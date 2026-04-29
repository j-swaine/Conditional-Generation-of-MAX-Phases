"""
Split-diagonal periodic table heatmap for Fig. 5.

Each element cell is bisected by the main diagonal:
    upper-left triangle  -> stoichiometry-weighted element frequency
    lower-right triangle -> mean E_hull (eV/atom)

The public entry point is `make_heatmap`, which accepts the PKV and
Slider dataframes of novel stable structures produced by the workflow
described in Methods IV.A and returns a Bokeh figure.
"""

from __future__ import annotations

import datetime
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from bokeh.io.export import get_screenshot_as_png
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    Label,
    LinearColorMapper,
    PrintfTickFormatter,
    Range1d,
)
from bokeh.palettes import Cividis256, Inferno256, Magma256, Plasma256, Viridis256
from bokeh.plotting import figure, output_file, save, show


# ---------------------------------------------------------------------------
# MAX-phase element inventory
# ---------------------------------------------------------------------------

M_SITES = {"Ti", "V", "Cr", "Nb", "Sc", "Zr", "Ta", "Mo", "W", "Mn", "Hf"}
A_SITES = {"Al", "Si", "Ga", "Ge", "P", "S", "As", "Cd", "In", "Sn", "Tl", "Pb"}
X_SITES = {"C", "N", "B"}
VALID_MAX_ELEMENTS = M_SITES | A_SITES  # X-sites are excluded from the heatmap


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def _interpolate_palette(hex_stops: list[str], n: int = 256) -> list[str]:
    if n <= len(hex_stops):
        step = max(1, len(hex_stops) // n)
        return hex_stops[::step][:n]
    rgbs = [_hex_to_rgb(h) for h in hex_stops]
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(rgbs) - 1)
        lo = int(t)
        hi = min(lo + 1, len(rgbs) - 1)
        f = t - lo
        r = rgbs[lo][0] * (1 - f) + rgbs[hi][0] * f
        g = rgbs[lo][1] * (1 - f) + rgbs[hi][1] * f
        b = rgbs[lo][2] * (1 - f) + rgbs[hi][2] * f
        out.append(_rgb_to_hex(r, g, b))
    return out


PALETTE_OPTIONS: dict[str, list[str]] = {
    "plasma":    list(Plasma256),
    "plasma_r":  list(reversed(Plasma256)),
    "inferno":   list(Inferno256),
    "inferno_r": list(reversed(Inferno256)),
    "magma":     list(Magma256),
    "magma_r":   list(reversed(Magma256)),
    "cividis":   list(Cividis256),
    "cividis_r": list(reversed(Cividis256)),
    "viridis":   list(Viridis256),
    "viridis_r": list(reversed(Viridis256)),
    "warm_grey": _interpolate_palette(
        ["#f5f5f0", "#e8dcc8", "#d4b87a", "#c49a3c", "#a07828", "#7a5a18"], 256),
    "purple_yellow": _interpolate_palette(
        ["#1b0c41", "#4a1486", "#7e3f8e", "#b5648c", "#de9a71", "#fde725"], 256),
    "blue_teal": _interpolate_palette(
        ["#0d0887", "#3d049b", "#6a00a8", "#8f0da4", "#b83289", "#d8576b",
         "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"], 256),
}


# ---------------------------------------------------------------------------
# Periodic table layout
# ---------------------------------------------------------------------------

ELEMENT_POS = {
    "H":  (1, 1),  "He": (18, 1),
    "Li": (1, 2),  "Be": (2, 2),  "B":  (13, 2), "C":  (14, 2), "N":  (15, 2), "O":  (16, 2), "F":  (17, 2), "Ne": (18, 2),
    "Na": (1, 3),  "Mg": (2, 3),  "Al": (13, 3), "Si": (14, 3), "P":  (15, 3), "S":  (16, 3), "Cl": (17, 3), "Ar": (18, 3),
    "K":  (1, 4),  "Ca": (2, 4),  "Sc": (3, 4),  "Ti": (4, 4),  "V":  (5, 4),  "Cr": (6, 4),  "Mn": (7, 4),  "Fe": (8, 4),  "Co": (9, 4),  "Ni": (10, 4), "Cu": (11, 4), "Zn": (12, 4), "Ga": (13, 4), "Ge": (14, 4), "As": (15, 4), "Se": (16, 4), "Br": (17, 4), "Kr": (18, 4),
    "Rb": (1, 5),  "Sr": (2, 5),  "Y":  (3, 5),  "Zr": (4, 5),  "Nb": (5, 5),  "Mo": (6, 5),  "Tc": (7, 5),  "Ru": (8, 5),  "Rh": (9, 5),  "Pd": (10, 5), "Ag": (11, 5), "Cd": (12, 5), "In": (13, 5), "Sn": (14, 5), "Sb": (15, 5), "Te": (16, 5), "I":  (17, 5), "Xe": (18, 5),
    "Cs": (1, 6),  "Ba": (2, 6),  "Hf": (4, 6),  "Ta": (5, 6),  "W":  (6, 6),  "Re": (7, 6),  "Os": (8, 6),  "Ir": (9, 6),  "Pt": (10, 6), "Au": (11, 6), "Hg": (12, 6), "Tl": (13, 6), "Pb": (14, 6), "Bi": (15, 6), "Po": (16, 6), "At": (17, 6), "Rn": (18, 6),
    "Fr": (1, 7),  "Ra": (2, 7),  "Rf": (4, 7),  "Db": (5, 7),  "Sg": (6, 7),  "Bh": (7, 7),  "Hs": (8, 7),  "Mt": (9, 7),  "Ds": (10, 7), "Rg": (11, 7), "Cn": (12, 7), "Nh": (13, 7), "Fl": (14, 7), "Mc": (15, 7), "Lv": (16, 7), "Ts": (17, 7), "Og": (18, 7),
}
PERIOD_LABELS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for el, n in _FORMULA_RE.findall(formula):
        if not el:
            continue
        counts[el] = counts.get(el, 0) + (float(n) if n else 1.0)
    return counts


# ---------------------------------------------------------------------------
# Dataframe stacking and per-element statistics
# ---------------------------------------------------------------------------

def _stack_dataframes(*dfs_and_cols) -> pd.DataFrame:
    parts = []
    for df, ehull_col in dfs_and_cols:
        parts.append(pd.DataFrame({
            "formula": df["_reduced_formula"].values,
            "ehull":   df[ehull_col].values,
        }))
    return pd.concat(parts, ignore_index=True)


def compute_element_stats(
    df: pd.DataFrame,
    *,
    formula_col: str = "formula",
    ehull_col: str = "ehull",
    ehull_filter: float = 0.25,
    stable_freq_cutoff: float = 0.05,
    restrict_to_max_elements: bool = True,
):
    """Return per-element frequency, frequency under a low-E_hull cutoff,
    mean E_hull, raw E_hull lists, and composition counts."""
    allowed = VALID_MAX_ELEMENTS if restrict_to_max_elements else None

    freq: dict[str, float] = defaultdict(float)
    freq_below_cutoff: dict[str, float] = defaultdict(float)
    ehull_acc: dict[str, list] = defaultdict(list)
    compositions: dict[str, set] = defaultdict(set)

    for _, row in df.iterrows():
        formula = row[formula_col]
        ehull = row[ehull_col]
        elem_counts = parse_formula(formula)

        for el, count in elem_counts.items():
            if el in X_SITES:
                continue
            if allowed is not None and el not in allowed:
                continue

            freq[el] += count
            compositions[el].add(formula)

            if ehull < stable_freq_cutoff:
                freq_below_cutoff[el] += count
            if ehull < ehull_filter:
                ehull_acc[el].append(ehull)

    mean_ehull: dict[str, float] = {}
    for el in freq:
        vals = ehull_acc.get(el, [])
        mean_ehull[el] = float(np.mean(vals)) if vals else np.nan

    n_compositions = {el: len(formulas) for el, formulas in compositions.items()}

    return (
        dict(freq),
        dict(freq_below_cutoff),
        mean_ehull,
        dict(ehull_acc),
        dict(n_compositions),
    )


def print_element_analysis(
    freq: dict,
    freq_below_cutoff: dict,
    mean_ehull: dict,
    ehull_lists: dict,
    n_compositions: dict,
    *,
    metastability_threshold: float = 0.1,
    deep_stable_cutoff: float = 0.025,
) -> pd.DataFrame:
    """Tabular per-element summary for report writing."""
    records = []
    all_elements = set(freq) | set(mean_ehull) | set(freq_below_cutoff)
    for el in sorted(all_elements):
        f = freq.get(el, 0)
        f05 = freq_below_cutoff.get(el, 0)
        e_mean = mean_ehull.get(el, np.nan)
        vals = ehull_lists.get(el, [])
        n_comp = n_compositions.get(el, 0)

        if el in M_SITES:
            site = "M-site"
        elif el in A_SITES:
            site = "A-site"
        else:
            site = "other"

        e_median = float(np.median(vals)) if vals else np.nan
        e_std = float(np.std(vals)) if vals else np.nan
        pct_deep = (
            sum(1 for v in vals if v < deep_stable_cutoff) / len(vals) * 100
            if vals else 0.0
        )

        records.append({
            "element": el,
            "site": site,
            "freq": f,
            "freq_lt_005": f05,
            "n_comp": n_comp,
            "n_structs": len(vals),
            "mean_ehull": e_mean,
            "median_ehull": e_median,
            "std_ehull": e_std,
            "pct_deep": pct_deep,
        })

    df_stats = pd.DataFrame(records)

    m = df_stats[df_stats["site"] == "M-site"].sort_values("freq", ascending=False)
    a = df_stats[df_stats["site"] == "A-site"].sort_values("freq", ascending=False)

    bar = "=" * 110
    print("\n" + bar)
    print("Element analysis summary (stability-validated set)")
    print(bar)
    print("\nM-site elements:")
    print(m.to_string(index=False))
    print("\nA-site elements:")
    print(a.to_string(index=False))
    print("\nTop frequency for E_hull < 0.05 eV/atom:")
    ranked = df_stats.sort_values("freq_lt_005", ascending=False)[
        ["element", "site", "freq_lt_005", "freq", "mean_ehull"]
    ]
    print(ranked.to_string(index=False))
    print(bar + "\n")
    return df_stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _triangle_coords(cx: float, cy: float, h: float = 0.475):
    ul_xs = [cx - h, cx + h, cx - h]
    ul_ys = [cy - h, cy + h, cy + h]
    lr_xs = [cx + h, cx + h, cx - h]
    lr_ys = [cy + h, cy - h, cy - h]
    return (ul_xs, ul_ys), (lr_xs, lr_ys)


def _resolve_palette(palette_arg, fallback_name: str) -> list[str]:
    if palette_arg is None:
        return PALETTE_OPTIONS[fallback_name]
    if isinstance(palette_arg, str):
        return PALETTE_OPTIONS[palette_arg]
    return list(palette_arg)


def plot_split_heatmap(
    freq: dict,
    mean_ehull: dict,
    *,
    title: str = "Element frequency and mean energy above hull heatmap",
    width: int = 1100,
    height: int = 550,
    output_path: str | None = None,
    png_path: str | None = None,
    png_scale: int = 3,
    show_plot: bool = True,
    bg_color: str | None = None,
    empty_color: str = "#e8e8e3",
    freq_palette=None,
    ehull_palette=None,
    element_font_size: str = "13pt",
    period_font_size: str = "8pt",
    period_x_offset: float = 0.25,
    group_font_size: str = "10pt",
):
    """Render the split-diagonal periodic table using Bokeh."""
    pal_freq = _resolve_palette(freq_palette, "plasma")
    pal_ehull = _resolve_palette(ehull_palette, "cividis_r")

    freq_vals = [v for v in freq.values() if v > 0]
    ehull_vals = [v for v in mean_ehull.values() if not np.isnan(v)]

    freq_mapper = LinearColorMapper(
        palette=pal_freq,
        low=min(freq_vals) if freq_vals else 0,
        high=max(freq_vals) if freq_vals else 1,
        nan_color=empty_color,
    )
    ehull_mapper = LinearColorMapper(
        palette=pal_ehull,
        low=min(ehull_vals) if ehull_vals else 0,
        high=max(ehull_vals) if ehull_vals else 0.25,
        nan_color=empty_color,
    )

    ul_xs, ul_ys, ul_vals = [], [], []
    lr_xs, lr_ys, lr_vals = [], [], []
    empty_rx, empty_ry = [], []
    sym_x, sym_y, sym_text = [], [], []

    for sym, (gx, py) in ELEMENT_POS.items():
        cx, cy = float(gx), float(-py)
        f_val = freq.get(sym, 0)
        e_val = mean_ehull.get(sym, np.nan)
        has_data = (f_val > 0) or (not np.isnan(e_val))

        if has_data:
            (ux, uy), (lx, ly) = _triangle_coords(cx, cy)
            ul_xs.append(ux); ul_ys.append(uy)
            ul_vals.append(f_val if f_val > 0 else np.nan)
            lr_xs.append(lx); lr_ys.append(ly)
            lr_vals.append(e_val if not np.isnan(e_val) else np.nan)
        else:
            empty_rx.append(cx)
            empty_ry.append(cy)

        sym_x.append(cx); sym_y.append(cy); sym_text.append(sym)

    p = figure(
        width=width, height=height, title=title,
        x_range=Range1d(0.2, 18.8), y_range=Range1d(-7.8, 0.2),
        toolbar_location=None, tools="",
    )

    if bg_color is None:
        p.background_fill_color = None
        p.background_fill_alpha = 0
        p.border_fill_color = None
        p.border_fill_alpha = 0
    else:
        p.background_fill_color = bg_color
        p.border_fill_color = "white"

    p.title.text_font_size = "16pt"
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.visible = False

    for pnum, plabel in PERIOD_LABELS.items():
        p.add_layout(Label(
            x=period_x_offset, y=float(-pnum), text=plabel,
            text_font_size=period_font_size, text_align="center",
            text_baseline="middle", text_color="#888888",
        ))
    for g in range(1, 19):
        p.add_layout(Label(
            x=float(g), y=-7.65, text=str(g),
            text_font_size=group_font_size, text_align="center",
            text_baseline="middle", text_color="#888888",
        ))

    if empty_rx:
        p.rect(
            x="x", y="y", width=0.95, height=0.95,
            source=ColumnDataSource(dict(x=empty_rx, y=empty_ry)),
            fill_color=empty_color, line_color="#cccccc", line_width=0.4,
        )

    p.patches(
        xs="xs", ys="ys",
        source=ColumnDataSource(dict(xs=ul_xs, ys=ul_ys, val=ul_vals)),
        fill_color={"field": "val", "transform": freq_mapper},
        line_color="#999999", line_width=0.3, line_alpha=0.5,
    )
    p.patches(
        xs="xs", ys="ys",
        source=ColumnDataSource(dict(xs=lr_xs, ys=lr_ys, val=lr_vals)),
        fill_color={"field": "val", "transform": ehull_mapper},
        line_color="#999999", line_width=0.3, line_alpha=0.5,
    )

    text_src = ColumnDataSource(dict(x=sym_x, y=sym_y, text=sym_text))

    halo_src = ColumnDataSource(dict(
        x=sym_x, y=sym_y, text=sym_text,
        x1=[x - 0.015 for x in sym_x], y1=list(sym_y),
        x2=[x + 0.015 for x in sym_x], y2=list(sym_y),
        x3=list(sym_x), y3=[y - 0.015 for y in sym_y],
        x4=list(sym_x), y4=[y + 0.015 for y in sym_y],
    ))
    for xk, yk in [("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")]:
        p.text(
            x=xk, y=yk, text="text", source=halo_src,
            text_align="center", text_baseline="middle",
            text_font_size=element_font_size, text_font_style="bold",
            text_color="white", text_alpha=0.6,
        )
    p.text(
        x="x", y="y", text="text", source=text_src,
        text_align="center", text_baseline="middle",
        text_font_size=element_font_size, text_font_style="bold",
        text_color="#000000",
    )

    freq_cb = ColorBar(
        color_mapper=freq_mapper,
        ticker=BasicTicker(desired_num_ticks=6),
        formatter=PrintfTickFormatter(format="%.0f"),
        label_standoff=8, width=16, location=(0, 0),
        title="Element frequency",
        title_text_font_size="12pt", major_label_text_font_size="11pt",
        background_fill_alpha=0,
    )
    ehull_cb = ColorBar(
        color_mapper=ehull_mapper,
        ticker=BasicTicker(desired_num_ticks=6),
        formatter=PrintfTickFormatter(format="%.3f"),
        label_standoff=8, width=16, location=(0, 0),
        title="Mean energy above\nhull (eV/atom)",
        title_text_font_size="12pt", major_label_text_font_size="11pt",
        background_fill_alpha=0,
    )
    p.add_layout(freq_cb, "left")
    p.add_layout(ehull_cb, "right")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path:
        output_file(output_path, title=title)
        save(p)
        print(f"[info] HTML written to {output_path}")

    if png_path == "auto":
        png_path = str(Path.cwd() / f"ptab_heatmap_{ts}.png")
    if png_path:
        img = get_screenshot_as_png(p, scale_factor=png_scale)
        img.save(png_path, format="PNG", dpi=(600, 600))
        print(f"[info] PNG written to {png_path}")

    if show_plot:
        show(p)
    return p


def make_heatmap(
    df_pkv: pd.DataFrame,
    df_slider: pd.DataFrame,
    df_nc: pd.DataFrame | None = None,
    *,
    ehull_filter: float = 0.25,
    metastability_threshold: float = 0.1,
    restrict_to_max_elements: bool = True,
    print_analysis: bool = True,
    **plot_kwargs,
):
    """Top-level entry point: stack the supplied novel-stable dataframes,
    compute per-element statistics, optionally print a text summary, and
    render the heatmap.

    Expected columns:
        df_pkv:    "_reduced_formula", "ehull_mace_mp"
        df_slider: "_reduced_formula", "ehull_mace_og"
        df_nc:     "_reduced_formula", "ehull_mace_og"   (optional)
    """
    sources = [
        (df_pkv, "ehull_mace_mp"),
        (df_slider, "ehull_mace_og"),
    ]
    if df_nc is not None:
        sources.append((df_nc, "ehull_mace_og"))

    stacked = _stack_dataframes(*sources)
    print(f"[info] Stacked {len(stacked)} rows from {len(sources)} dataframes")

    freq, freq_below_cutoff, mean_ehull, ehull_lists, n_compositions = \
        compute_element_stats(
            stacked,
            formula_col="formula",
            ehull_col="ehull",
            ehull_filter=ehull_filter,
            stable_freq_cutoff=0.05,
            restrict_to_max_elements=restrict_to_max_elements,
        )

    if print_analysis:
        print_element_analysis(
            freq, freq_below_cutoff, mean_ehull, ehull_lists, n_compositions,
            metastability_threshold=metastability_threshold,
        )

    return plot_split_heatmap(freq, mean_ehull, **plot_kwargs)