"""
MAX phase formula validation.

Provides utilities to parse reduced chemical formulae and test whether
a composition matches the M_{n+1}AX_n stoichiometry used throughout the
paper. Downstream code uses `validate_max_phase_df` to annotate a
dataframe of generated structures with an `is_MAX` flag.
"""

from __future__ import annotations

import re
import pandas as pd

TRANSITION_METALS: set[str] = {
    "Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W",
    "Mn", "Tc", "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt",
    "Cu", "Ag", "Au", "Zn", "Hg",
}

# Groups 13-16, excluding C, N, O which are treated as anions / X-site.
A_GROUP: set[str] = {
    "B", "Al", "Ga", "In", "Tl",
    "Si", "Ge", "Sn", "Pb",
    "P", "As", "Sb", "Bi",
    "S", "Se", "Te", "Po",
    "Cd",
}


def _parse_reduced_formula(formula: str) -> dict:
    """Parse a reduced formula (e.g. 'Ti3AlC2') into a dict of element counts."""
    if not isinstance(formula, str) or not formula:
        return {}
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula.strip())
    comp: dict[str, float] = {}
    for el, num in tokens:
        cnt = 1.0 if num == "" else float(num)
        comp[el] = comp.get(el, 0.0) + cnt
    return comp


def _is_near_int(x: float, tol: float = 1e-6) -> bool:
    return abs(x - round(x)) < tol


def validate_max_formula(
    formula: str,
    allowed_ns: tuple[int, ...] = (1, 2, 3),
    include_boron_as_X: bool = False,
    require_integer: bool = True,
) -> tuple[bool, dict]:
    """Return `(is_max, info)` where `is_max` is True if the formula matches
    M_{n+1}AX_n, allowing up to two distinct M-site elements."""
    comp = _parse_reduced_formula(formula)
    info: dict = {"formula": formula, "reason": None}

    if not comp:
        info["reason"] = "Unparseable or empty formula"
        return False, info

    X_candidates = {"C", "N"}
    if include_boron_as_X:
        X_candidates.add("B")

    M_elems, A_elems, X_elems, others = [], [], [], []
    for el in comp:
        if el in X_candidates:
            X_elems.append(el)
        elif el in TRANSITION_METALS:
            M_elems.append(el)
        elif el in A_GROUP:
            A_elems.append(el)
        else:
            others.append(el)

    if others:
        info["reason"] = f"Contains non-MAX elements: {others}"
        return False, info
    if len(X_elems) != 1:
        info["reason"] = f"Must have exactly one X element, found {X_elems}"
        return False, info
    if len(A_elems) != 1:
        info["reason"] = f"Must have exactly one A element, found {A_elems}"
        return False, info
    if len(M_elems) < 1:
        info["reason"] = "No transition-metal M elements"
        return False, info
    if len(M_elems) > 2:
        info["reason"] = f"More than two distinct M elements: {M_elems}"
        return False, info

    A_el, X_el = A_elems[0], X_elems[0]
    A_count = comp.get(A_el, 0.0)
    X_count = comp.get(X_el, 0.0)
    M_total = sum(comp[el] for el in M_elems)

    if require_integer:
        for el, cnt in comp.items():
            if not _is_near_int(cnt):
                info["reason"] = f"Non-integer stoichiometry for {el}: {cnt}"
                return False, info
        A_count = int(round(A_count))
        X_count = int(round(X_count))
        M_total = int(round(M_total))

    if A_count != 1:
        info["reason"] = f"A count must be 1, got {A_count}"
        return False, info

    n = X_count
    if n not in allowed_ns:
        info["reason"] = f"n (= X count) must be in {allowed_ns}, got n={n}"
        return False, info
    if M_total != n + 1:
        info["reason"] = f"M total must equal n+1={n+1}, got {M_total}"
        return False, info

    info.update({
        "n": n,
        "M_total": M_total,
        "M_elems": tuple(M_elems),
        "A_elem": A_el,
        "X_elem": X_el,
    })
    return True, info


def validate_max_phase_df(
    df: pd.DataFrame,
    formula_col: str = "_reduced_formula",
    include_boron_as_X: bool = False,
    allowed_ns: tuple[int, ...] = (1, 2, 3),
    require_integer: bool = True,
) -> pd.DataFrame:
    """Add `is_MAX` (bool) and `MAX_info` (dict) columns to `df`."""
    out = df.copy()
    flags, infos = [], []
    for f in out[formula_col].astype(str):
        ok, info = validate_max_formula(
            f,
            allowed_ns=allowed_ns,
            include_boron_as_X=include_boron_as_X,
            require_integer=require_integer,
        )
        flags.append(ok)
        infos.append(info)
    out["is_MAX"] = flags
    out["MAX_info"] = infos
    return out