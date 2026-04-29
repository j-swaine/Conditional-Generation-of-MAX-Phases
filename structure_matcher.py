#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import re
import json

from functools import partial
from glob import glob

from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator, OrderDisorderElementComparator
from pymatgen.io.cif import CifParser

TRAIN_BUCKETS: Optional[Dict[frozenset, List[Tuple[int, str, "Structure"]]]] = None

def _init_buckets_global(buckets_in):
    global TRAIN_BUCKETS
    TRAIN_BUCKETS = buckets_in
    
def _vpa_bounds_for_key(key):
    bucket = TRAIN_BUCKETS.get(key, [])
    if not bucket:
        return None
    v = np.array([_vpa(s_tr) for (_, _, s_tr) in bucket], float)
    return float(v.min()), float(np.median(v)), float(v.max()), len(v)

_FORMULA_SUM_RE = re.compile(r'(?im)^\s*_chemical_formula_sum\s+(?P<val>.+)$')

def _fast_formula_from_cif_text(txt: str) -> Optional[str]:
    """
    Extract _chemical_formula_sum from CIF text (no full parse).
    Returns a canonical reduced formula string or None.
    """
    txt = _decode_cif(txt)
    if not txt:
        return None
    m = _FORMULA_SUM_RE.search(txt)
    if not m:
        return None
    raw = m.group("val").strip()
    if raw and raw[0] in {"'", '"'} and raw[-1:] == raw[0]:
        raw = raw[1:-1]
    try:
        return Composition(raw).reduced_formula
    except Exception:
        return None

def _clean_stale_shards(prefix: str, *, dry_run: bool = False) -> int:
    """
    Safely remove old shard files for a given prefix.
    Only deletes files exactly matching:
      <prefix>.part*.parquet and <prefix>.manifest.json
    Extra guards:
      - No wildcard chars allowed in 'prefix'
      - Files must live in the same directory as 'prefix'
      - Filenames must start with '<basename>.'
    """
    abs_prefix = os.path.abspath(prefix)
    shard_dir, base = os.path.split(abs_prefix)

    if not base:
        raise ValueError("Refusing to clean: 'shard_prefix' points to a directory.")
    if any(ch in base for ch in "*?[]"):
        raise ValueError("Refusing to clean: 'shard_prefix' contains wildcard characters.")

    patterns = [f"{abs_prefix}.part*.parquet", f"{abs_prefix}.manifest.json"]
    victims = []
    for pat in patterns:
        for p in glob(pat):
            p_abs = os.path.abspath(p)
            if os.path.dirname(p_abs) == shard_dir and os.path.basename(p_abs).startswith(base + "."):
                victims.append(p_abs)

    if not victims:
        print(f"[CKPT] No stale shards for prefix '{prefix}'.")
        return 0

    if dry_run:
        print("[CKPT] Would remove:")
        for v in victims:
            print("  ", v)
        return len(victims)

    for v in victims:
        os.remove(v)
    print(f"[CKPT] Removed {len(victims)} stale shard file(s) for prefix '{prefix}'.")
    return len(victims)
    
def _formula_to_element_set(formula: str) -> Optional[str]:
    """Return a sorted, comma-joined element set (e.g., 'B,Cr,Si') or None."""
    if not formula:
        return None
    try:
        el = sorted({el.symbol for el in Composition(formula).elements})
        return ",".join(el)
    except Exception:
        return None
    
def _add_reduced_formula_col(df: pd.DataFrame, cif_col: str, out_col: str = "reduced_formula") -> pd.DataFrame:
    """Add a reduced_formula column using a fast header parse; leaves df modified in place."""
    df[out_col] = df[cif_col].map(_fast_formula_from_cif_text)
    return df

def _add_element_set_col(df: pd.DataFrame, formula_col: str = "reduced_formula", out_col: str = "element_set") -> pd.DataFrame:
    """Add an element_set column derived from the (reduced) formula; leaves df modified in place."""
    df[out_col] = df[formula_col].map(_formula_to_element_set)
    return df

def prefilter(
    df_rel: pd.DataFrame,
    df_train: pd.DataFrame,
    col_rel: str,
    col_train: str,
    *,
    keep_element_set: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only training rows whose reduced_formula appears in the relaxed set.
    Adds a 'reduced_formula' column to both frames (and 'element_set' if requested).
    Returns (df_rel_filtered, df_train_filtered).
    """
    _add_reduced_formula_col(df_rel, col_rel, out_col="reduced_formula")
    _add_reduced_formula_col(df_train, col_train, out_col="reduced_formula")

    before_rel, before_tr = len(df_rel), len(df_train)
    df_rel = df_rel.dropna(subset=["reduced_formula"]).copy()
    df_train = df_train.dropna(subset=["reduced_formula"]).copy()

    if keep_element_set:
        _add_element_set_col(df_rel, "reduced_formula", "element_set")
        _add_element_set_col(df_train, "reduced_formula", "element_set")

    relaxed_formulas = set(df_rel["reduced_formula"].unique())
    df_train = df_train[df_train["reduced_formula"].isin(relaxed_formulas)].copy()

    if verbose:
        print(f"[PF] relaxed rows: {before_rel} to {len(df_rel)} after dropping NA formulas "
              f"(unique formulas={df_rel['reduced_formula'].nunique()}).")
        print(f"[PF] training rows: {before_tr} to {len(df_train)} after formula filter "
              f"(unique formulas kept={df_train['reduced_formula'].nunique()}).")

    return df_rel, df_train

def _match_one_relaxed(
    rel_item,
    tol_triples,
    attempt_supercell: bool,
    allow_subset: bool,
    vol_pad: float,
    max_matches_per_structure: int,
    debug: bool = False,
    debug_sample: int = 0,
    disorder_ok: bool = False
):
    i_rel, s_rel = rel_item
    el_key = frozenset(sp.symbol for sp in s_rel.composition.elements)
    elem_set_str = ",".join(sorted(el_key))

    if TRAIN_BUCKETS is None:
        raise RuntimeError("TRAIN_BUCKETS not initialized. (Did you set it in serial, or pass initializer in parallel?)")

    bucket = TRAIN_BUCKETS.get(el_key, [])
    vpa_rel = _vpa(s_rel)

    vpa_rel = _vpa(s_rel)
    if vol_pad is None or float(vol_pad) < 0:
        cand = list(bucket)
        lo = hi = None
    else:
        lo, hi = vpa_rel * (1 - vol_pad), vpa_rel * (1 + vol_pad)
        cand = [(j_tr, tr_id, s_tr) for (j_tr, tr_id, s_tr) in bucket
                if lo <= _vpa(s_tr) <= hi]
        
        if debug and len(cand) == 0:
            bnd = _vpa_bounds_for_key(el_key)
            if bnd is None:
                print(f"[DBG] rel#{i_rel}: element set {elem_set_str} has NO bucket in training.")
            else:
                bmin, bmed, bmax, bN = bnd
                if lo is None:
                    pad_txt = "no VPA prefilter"
                else:
                    pad_txt = f"vol_pad={vol_pad:.3f} -> window=[{lo:.3f}, {hi:.3f}]"
                print(f"[DBG] rel#{i_rel}: VPA_rel={vpa_rel:.3f}; bucket n={bN} VPA=[{bmin:.3f}, {bmed:.3f}, {bmax:.3f}] | {pad_txt} | candidates=0")
    
    # lo, hi = vpa_rel * (1 - vol_pad), vpa_rel * (1 + vol_pad)
    # cand = [(j_tr, tr_id, s_tr) for (j_tr, tr_id, s_tr) in bucket if lo <= _vpa(s_tr) <= hi]

    stats = {
        "relaxed_index": i_rel,
        "element_set": ",".join(sorted(el_key)),
        "bucket_size": len(bucket),
        "cands_after_vpa": len(cand),
        "pairs_checked": 0,
        "first_match_tol": None,
        "first_match_train_id": None,
        "rms_nearest_no_match": None,  
    }

    out_rows = []
    found = 0
    for (ltol, stol, angtol) in tol_triples:
        comp = OrderDisorderElementComparator() if disorder_ok else ElementComparator()
        matcher = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angtol,
            primitive_cell=False, scale=True,
            attempt_supercell=attempt_supercell, allow_subset=allow_subset,
            comparator=comp,
        )
        for (j_tr, tr_id, s_tr) in cand:
            stats["pairs_checked"] += 1
            try:
                if matcher.fit(s_rel, s_tr):
                    row = {
                        "relaxed_index": i_rel,
                        "training_index": j_tr,
                        "train_id": tr_id,
                        "relaxed_formula": s_rel.composition.reduced_formula,
                        "train_formula": s_tr.composition.reduced_formula,
                        "n_sites_relaxed": len(s_rel),
                        "n_sites_train": len(s_tr),
                        "ltol": ltol, "stol": stol, "angle_tol": angtol,
                        "attempt_supercell": attempt_supercell,
                        "allow_subset": allow_subset,
                        "element_set": stats["element_set"],
                    }
                    out_rows.append(row)
                    found += 1
                    if stats["first_match_tol"] is None:
                        stats["first_match_tol"] = f"l{ltol}/s{stol}/a{angtol}"
                        stats["first_match_train_id"] = tr_id
                    if found >= max_matches_per_structure:
                        return out_rows, stats
            except Exception:
                continue

    if debug and debug_sample > 0 and cand:
        ltol, stol, angtol = tol_triples[0]  
        matcher = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angtol,
            primitive_cell=False, scale=True,
            attempt_supercell=attempt_supercell, allow_subset=allow_subset,
            comparator=ElementComparator(),
        )
        rms_vals = []
        for (_, _, s_tr) in cand[:debug_sample]:
            try:
                rms, _, _ = matcher.get_rms_dist(s_rel, s_tr)
                if rms is not None:
                    rms_vals.append(rms)
            except Exception:
                pass
        if rms_vals:
            stats["rms_nearest_no_match"] = float(min(rms_vals))

    return out_rows, stats

def _vpa(s): 
    """Volume per atom: V / N."""
    return float(s.volume) / max(1, len(s))

def _decode_cif(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (bytes, bytearray, memoryview, np.bytes_)):
        b = bytes(val)
        if b[:2] == b"\x1f\x8b":  # gzip magic
            try:
                b = gzip.decompress(b)
            except Exception:
                pass
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("latin-1", errors="ignore")
    if isinstance(val, str):
        return val
    try:
        import pandas as _pd
        if _pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val)


def _structure_from_cif(
    val: Any,
    *,
    parser_site_tol: float = 1e-4,
    parser_occu_tol: float = 10.0,
    parser_check_occu: bool = False,
    _seen: list = [],  # for throttled debug
) -> Optional[Structure]:
    """
    Robust CIF to Structure with compatibility across pymatgen versions:
    - tries CifParser.from_string(...), then from_str(...), then CifParser(io=StringIO)
    - tries parse_structures(...), then get_structures()
    - falls back to Structure.from_str(...)
    """
    txt = _decode_cif(val)
    if not txt.strip():
        return None

    parser = None

    try:
        parser = CifParser.from_string(
            txt,
            site_tolerance=parser_site_tol,
            occupancy_tolerance=parser_occu_tol,
        )
    except TypeError:
        try:
            parser = CifParser.from_string(txt)
        except Exception:
            parser = None
    except AttributeError:
        parser = None

    if parser is None:
        try:
            parser = CifParser.from_str(txt)
        except Exception:
            parser = None

    if parser is None:
        try:
            parser = CifParser(io.StringIO(txt))
        except TypeError:
            try:
                parser = CifParser(io.StringIO(txt),
                                   site_tolerance=parser_site_tol,
                                   occupancy_tolerance=parser_occu_tol)
            except Exception:
                parser = None
        except Exception:
            parser = None

    if parser is not None:
        try:
            # new API
            try:
                structs = parser.parse_structures(
                    primitive=False,
                    symmetrized=False,
                    check_occu=parser_check_occu,
                )
            except TypeError:
                structs = parser.parse_structures(
                    primitive=False,
                    symmetrized=False,
                )
            if structs:
                return structs[0].get_sorted_structure()
        except Exception as e:
            if len(_seen) < 3:
                _seen.append(True)
                print(f"[PARSE-FAIL:parse_structures] {type(e).__name__}: {e}")

        try:
            structs = parser.get_structures()
            if structs:
                return structs[0].get_sorted_structure()
        except Exception as e2:
            if len(_seen) < 6:
                _seen.append(True)
                print(f"[PARSE-FAIL:get_structures] {type(e2).__name__}: {e2}")
    try:
        s = Structure.from_str(txt, fmt="cif")
        return s.get_sorted_structure()
    except Exception as e3:
        if len(_seen) < 9:
            _seen.append(True)
            head = repr(txt[:120])
            print(f"[PARSE-FAIL:Structure.from_str] {type(e3).__name__}: {e3} | head={head}")
        return None

def _standardize_structure(
    s: Structure,
    symprec: float = 0.1,
    angle_tol: float = 5.0,
    to_primitive: bool = True,
    conventional: bool = False,
) -> Structure:
    """Symmetry-refine and return a consistent representation."""
    try:
        s = s.copy()
        try:
            s.remove_oxidation_states()
        except Exception:
            pass
        sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tol)
        s_std = sga.get_conventional_standard_structure() if conventional else sga.get_refined_structure()
        if to_primitive:
            s_std = s_std.get_primitive_structure()
        s_std = s_std.get_sorted_structure()
        return s_std
    except Exception:
        return s

def _to_norm_cif(
    val,
    *,
    do_symmetrize: bool,
    symprec: float,
    sym_angle_tol: float,
    to_primitive: bool,
    conventional: bool,
):
    s = _structure_from_cif(val, parser_site_tol=1e-4, parser_occu_tol=10.0, parser_check_occu=False)
    if s is None:
        return None
    if do_symmetrize:
        s = _standardize_structure(s, symprec=symprec, angle_tol=sym_angle_tol,
                                   to_primitive=to_primitive, conventional=conventional)
    s = s.get_sorted_structure()
    return s.to(fmt="cif")

def _element_set_key(s: Structure) -> frozenset:
    """Bucket key: unordered set of elements (ignores counts)."""
    return frozenset(sp.symbol for sp in s.composition.elements)

def _summarize_buckets(train_records, top=15):
    """Print per-element-set counts and VPA stats."""
    by_key = {}
    for (j, tid, s) in train_records:
        k = _element_set_key(s)
        by_key.setdefault(k, []).append(_vpa(s))
    rows = []
    for k, vpas in by_key.items():
        v = np.array(vpas, float)
        rows.append({
            "element_set": ",".join(sorted(k)),
            "count": len(v),
            "vpa_min": float(v.min()) if len(v) else np.nan,
            "vpa_med": float(np.median(v)) if len(v) else np.nan,
            "vpa_max": float(v.max()) if len(v) else np.nan,
        })
    rows.sort(key=lambda r: r["count"], reverse=True)
    print("[DBG] Top element-set buckets by count:")
    for r in rows[:top]:
        print(f"  {r['element_set']:>15} | n={r['count']:>7} | VPA=[{r['vpa_min']:.3f}, {r['vpa_med']:.3f}, {r['vpa_max']:.3f}]")
    return by_key

def _count_parsing(df, colname):
    n = len(df)
    ok = 0
    for x in df[colname]:
        s = _structure_from_cif(x)
        if s is not None:
            ok += 1
    print(f"[DBG] Parse success for '{colname}': {ok}/{n} ({100.0*ok/n:.1f}%)")

def sweep_match_relaxed_vs_training(
    relaxed: pd.DataFrame,
    training: pd.DataFrame,
    *,
    relaxed_cif_col: str = "relaxed_cif",
    training_cif_col: str = "CIF",
    training_id_col: Optional[str] = None,
    symmetrize: bool = True,
    symprec: float = 0.1,
    sym_angle_tol: float = 5.0,
    to_primitive: bool = True,
    conventional: bool = False,
    ltol_list: Sequence[float] = (0.1, 0.2, 0.3),
    stol_list: Sequence[float] = (0.3, 0.5),
    angle_tol_list: Sequence[float] = (2.0, 5.0, 8.0),
    attempt_supercell: bool = True,
    allow_subset: bool = False,
    max_matches_per_structure: int = 1,
    carry_relaxed_cols: Sequence[str] = ("condition_vector", "Prompt", "ehull_mace", "ehull_mace_relaxed"),
    n_jobs: int = 1,
    vol_pad: float = 0.25,
    debug: bool = False,
    debug_sample: int = 0,
    log_stats: Optional[str] = None,
    disorder_ok: bool = False,
    chunk_size: int = 0,
    shard_prefix: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a DataFrame of matches with the first (tightest) tolerance triple that succeeds."""

    rel_base = relaxed.reset_index(drop=False)
    rel_base["relaxed_index"] = range(len(relaxed))
    rel_cols = [c for c in carry_relaxed_cols if c in rel_base.columns]
    relaxed_meta = rel_base[["relaxed_index"]+rel_cols].copy()
    relaxed_meta = relaxed_meta.rename(columns={c: f"_{c}" for c in rel_cols})

    shard_idx = 0
    def _flush(rows: List[Dict[str, Any]]):
        nonlocal shard_idx
        if not rows:
            return []

        # ensure shard_prefix has a directory
        shard_dir = os.path.join(os.path.dirname(shard_prefix), "temporary")
        os.makedirs(shard_dir, exist_ok=True)

        base_prefix = os.path.basename(shard_prefix)
        path = os.path.join(shard_dir, f"{base_prefix}.part{shard_idx:04d}.parquet")

        pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")
        print(f"[CKPT] wrote shard: {path} ({len(rows)} rows)")
        shard_idx += 1
        return []

    relaxed_structs: List[Tuple[int, Structure]] = []
    for i, val in enumerate(relaxed.get(relaxed_cif_col, [])):
        s = _structure_from_cif(val, parser_site_tol=1e-4, parser_occu_tol=10.0, parser_check_occu=False)
        if s is None:
            continue
        if symmetrize:
            s = _standardize_structure(s, symprec=symprec, angle_tol=sym_angle_tol,
                                       to_primitive=to_primitive, conventional=conventional)
        relaxed_structs.append((i, s))

    train_records: List[Tuple[int, str, Structure]] = []
    for j, row in training.iterrows():
        t = _structure_from_cif(row[training_cif_col], parser_site_tol=1e-4, parser_occu_tol=10.0, parser_check_occu=False)
        if t is None:
            continue
        if symmetrize:
            t = _standardize_structure(t, symprec=symprec, angle_tol=sym_angle_tol,
                                       to_primitive=to_primitive, conventional=conventional)
        tid = str(row[training_id_col]) if training_id_col and training_id_col in training.columns else str(j)
        train_records.append((j, tid, t))
        
    if debug:
        print(f"[DBG] relaxed parsed: {len(relaxed_structs)} / {len(relaxed.get(relaxed_cif_col, []))}")
        print(f"[DBG] training parsed: {len(train_records)} / {len(training)}")
        # If you want full counts independent of symmetrize choices:
        # _count_parsing(relaxed, relaxed_cif_col)
        # _count_parsing(training, training_cif_col)

    buckets: Dict[frozenset, List[Tuple[int, str, Structure]]] = {}
    for rec in train_records:
        buckets.setdefault(_element_set_key(rec[2]), []).append(rec)
    _vol_pad = float(vol_pad) 
    
    if debug:
        _ = _summarize_buckets(train_records, top=20)

    tol_triples = [(l, s, a) for l in ltol_list for s in stol_list for a in angle_tol_list]
    out_rows: List[Dict[str, Any]] = []
    stats_rows = []

    if n_jobs <= 1:
        
        global TRAIN_BUCKETS
        TRAIN_BUCKETS = buckets
        for rel_item in relaxed_structs:
            rows, stats = _match_one_relaxed(
                rel_item,
                tol_triples,
                attempt_supercell=attempt_supercell,
                allow_subset=allow_subset,
                vol_pad=_vol_pad,
                max_matches_per_structure=max_matches_per_structure,
                debug=debug,
                debug_sample=debug_sample,
                disorder_ok=disorder_ok
            )
            out_rows.extend(rows)
            stats_rows.append(stats)
            if chunk_size and out_rows:
                out_rows = _flush(out_rows)

    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_buckets_global,
            initargs=(buckets,),
        ) as ex:
            futs = [
                ex.submit(
                    _match_one_relaxed,
                    rel_item,
                    tol_triples,
                    attempt_supercell,
                    allow_subset,
                    vol_pad,  
                    max_matches_per_structure,
                    debug,
                    debug_sample,
                    disorder_ok
                )
                for rel_item in relaxed_structs
            ]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    rows,stats = fut.result()
                    out_rows.extend(rows)
                    stats_rows.append(stats)
                    if chunk_size and len(out_rows) >=chunk_size:
                        out_rows = _flush(out_rows)
                    if debug and (k % 50 == 0):
                        checked = sum(r["pairs_checked"] for r in stats_rows)
                        matches = sum(1 for r in stats_rows if r["first_match_tol"] is not None)
                        zero_cands = sum(1 for r in stats_rows if r["cands_after_vpa"]==0)
                        errs=sum(r.get("errors", 0) for r in stats_rows)
                        print(f"[DBG] processed={k}/{len(relaxed_structs)} | matches={matches} | zero_cands={zero_cands} | pairs_checked≈{checked} | errors={errs}")
                except Exception as e:
                    print(f"[WARN] worker failed: {e}")
                    
            if chunk_size and out_rows:
                out_rows = _flush(out_rows)

    EXPECTED_COLS = [
        "relaxed_index", "training_index",
        "ltol", "stol", "angle_tol",
        "primitive_cell", "attempt_supercell", "scale",
        "rms_disp", "matched"  
    ]

    matches = pd.DataFrame(out_rows)
    
    if chunk_size:
        manifest = {
            "prefix": shard_prefix,
            "num_shards":shard_idx,
            "schema_cols":list(matches.columns) if not matches.empty else [],
        }
    
        man_path = f"{shard_prefix}.manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f)
        print(f"[CKPT] wrote manifest to {man_path}")

    if debug and stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        print("[DBG] Stats summary:")
        print("  total relaxed processed:", len(stats_df))
        print("  with ≥1 candidate after VPA:", int((stats_df["cands_after_vpa"] > 0).sum()))
        print("  with first match:", int(stats_df["first_match_tol"].notna().sum()))
        print("  median candidates after VPA:", float(stats_df["cands_after_vpa"].median()))
        print("  median pairs checked:", float(stats_df["pairs_checked"].median()))
        if stats_df["first_match_tol"].notna().any():
            print("  first-match tol distribution:")
            print(stats_df["first_match_tol"].value_counts().head(10).to_string())
        if stats_df["rms_nearest_no_match"].notna().any():
            print("  min RMS (no-match) percentiles:",
                stats_df["rms_nearest_no_match"].dropna().quantile([0.1,0.5,0.9]).to_dict())
        if log_stats:
            stats_df.to_parquet(log_stats, index=False)
            print(f"[DBG] wrote per-structure stats to {log_stats}")
    
    # if matches.empty:
    #     print("[INFO] No matches found at any tolerance; returning empty table.")
    #     EXPECTED_COLS = [
    #         "relaxed_index", "training_index",
    #         "ltol", "stol", "angle_tol",
    #         "attempt_supercell", "allow_subset",
    #         "element_set",
    #     ]

    else:
        for col in ["relaxed_index", "training_index"]:
            if col not in matches.columns:
                matches[col] = pd.NA
                
        sort_keys = [c for c in ["relaxed_index", "training_index", "ltol", "stol", "angle_tol"]
                    if c in matches.columns]
        if sort_keys:
            matches = matches.sort_values(by=sort_keys).reset_index(drop=True)
        else:
            matches = matches.reset_index(drop=True)

    if chunk_size and out_rows:
        out_rows = _flush(out_rows)

    if chunk_size:
        shard_paths = sorted(glob(f"{shard_prefix}.part*.parquet"))
        if shard_paths:
            matches = pd.concat((pd.read_parquet(p) for p in shard_paths), ignore_index=True)
            print(f"[CKPT] merged {len(shard_paths)} shards to {len(matches)} rows")

            manifest = {
                "prefix": shard_prefix,
                "num_shards": len(shard_paths),
                "schema_cols": matches.columns.tolist(),
            }
            with open(f"{shard_prefix}.manifest.json", "w") as f:
                json.dump(manifest, f)
            print(f"[CKPT] wrote manifest to {shard_prefix}.manifest.json")
        else:
            matches = pd.DataFrame()
    else:
        matches = pd.DataFrame(out_rows)
    
    if matches.empty:
        print(f"[INFO] No matches found at any tolerance")
        matches = pd.DataFrame(columns=EXPECTED_COLS)

    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame(
        columns=["relaxed_index","element_set","bucket_size","cands_after_vpa",
                "pairs_checked","first_match_tol","first_match_train_id","rms_nearest_no_match"]
    )
    unmatched_df = stats_df[stats_df["first_match_tol"].isna()].copy() if not stats_df.empty else \
        pd.DataFrame(columns=["relaxed_index","element_set","bucket_size","cands_after_vpa",
                                     "pairs_checked","first_match_tol","first_match_train_id",
                                     "rms_nearest_no_match"])
    if not unmatched_df.empty:
        unmatched_df = unmatched_df.merge(relaxed_meta, on="relaxed_index", how="left")

    for col in ["relaxed_index", "training_index", "ltol", "stol", "angle_tol"]:
        if col not in matches.columns:
            matches[col] = pd.NA

    sort_keys = [c for c in ["relaxed_index", "training_index", "ltol", "stol", "angle_tol"]
                if c in matches.columns]
    if not matches.empty and sort_keys:
        matches = matches.sort_values(by=sort_keys).reset_index(drop=True)
    elif not matches.empty:
        matches = matches.reset_index(drop=True)

    return matches, unmatched_df

def comp_novel_vs_training(
    df_rel: pd.DataFrame,
    df_train: pd.DataFrame,
    col_rel: str,
    col_train: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return only relaxed rows whose reduced_formula does NOT occur in the training set.
    Assumes _add_reduced_formula_col is available.
    """
    _add_reduced_formula_col(df_rel, col_rel, out_col="reduced_formula")
    _add_reduced_formula_col(df_train, col_train, out_col="reduced_formula")

    before = len(df_rel)
    train_formulas = set(df_train["reduced_formula"].dropna().unique())
    novel_rel = df_rel[~df_rel["reduced_formula"].isin(train_formulas)].copy()

    if verbose:
        print(f"[COMP-NOVEL] relaxed {before} to {len(novel_rel)} "
              f"(novel formulas not in training: {novel_rel['reduced_formula'].nunique()}).")

    return novel_rel

def main():
    p = argparse.ArgumentParser(description="Match relaxed CIFs against a training set (both from parquet) using pymatgen StructureMatcher.")
    p.add_argument("--relaxed_parquet", required=True, help="Parquet with a column of CIF strings (default: relaxed_cif)")
    p.add_argument("--training_parquet", required=True, help="Parquet with training CIF strings")
    p.add_argument("--relaxed_cif_col", default="relaxed_cif", help="Column in relaxed_parquet containing CIF strings")
    p.add_argument("--training_cif_col", default="CIF", help="Column in training_parquet containing CIF strings")
    p.add_argument("--training_id_col", default=None, help="Optional ID column in training_parquet to report")
    p.add_argument("--out_parquet", required=True, help="Where to write matches parquet")

    p.add_argument("--symmetrize", action="store_true", help="Symmetry-refine/standardize structures (recommended)")
    p.add_argument("--no-symmetrize", dest="symmetrize", action="store_false")
    p.set_defaults(symmetrize=False)
    p.add_argument("--symprec", type=float, default=0.1)
    p.add_argument("--sym_angle_tol", type=float, default=5.0)
    p.add_argument("--primitive", dest="to_primitive", action="store_true")
    p.add_argument("--no-primitive", dest="to_primitive", action="store_false")
    p.set_defaults(to_primitive=False)
    p.add_argument("--conventional", action="store_true", help="Return conventional cells (default: refined + primitive)")

    p.add_argument("--ltol", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="Lattice length tolerances to sweep")
    p.add_argument("--stol", type=float, nargs="+", default=[0.3, 0.5], help="Site tolerance(s) to sweep")
    p.add_argument("--angle_tol", type=float, nargs="+", default=[2.0, 5.0, 8.0], help="Angle tolerance(s) to sweep (degrees)")
    p.add_argument("--attempt_supercell", action="store_true", help="Allow supercell/primitive relations")
    p.add_argument("--no_attempt_supercell", dest="attempt_supercell", action="store_false")
    p.set_defaults(attempt_supercell=True)
    p.add_argument("--allow_subset", action="store_true", help="Allow subset matching (defects, partial)")
    p.add_argument("--max_matches_per_structure", type=int, default=1)
    p.add_argument("--debug", action="store_true",
               help="Verbose diagnostics: count candidates, pairs checked, first match tol, etc.")
    p.add_argument("--debug_sample", type=int, default=0,
                help="If >0, compute/record the smallest RMS distance to the first N candidates when no match is found (slower).")
    p.add_argument("--log_stats", default=None,
                help="Optional path to write per-structure debug stats (parquet).")
    p.add_argument("--n_jobs", type=int, default=0,
                help="Number of worker processes for matching (0/1 = serial).")
    p.add_argument("--vol_pad", type=float, default=0.25, help="Volume padding for supercell matching.")
    p.add_argument("--disorder_ok", action="store_true",
               help="Use OrderDisorderElementComparator (more tolerant to site disorder).")
    p.add_argument("--prefilter", action="store_true", help="Apply a prefilter on the training set to drop any structures whose reduced formula is not in the relaxed set.")
    p.add_argument("--chunk_size", type=int, default=0, help="Chunk size for processing large datasets.")
    p.add_argument("--shard_prefix", default=None, help="If set, write output shards with this prefix instead of a single file.")
    p.add_argument("--unmatched_parquet", default="unmatched.parquet", help="If set, write unmatched structures to this parquet file.")
    p.add_argument("--clean_old_shards", default=False, action="store_true", help="If set, remove old shard files with the specified prefix.")
    p.add_argument("--normalize_cifs", action="store_true",
                help="Rewrite CIF strings from parsed structures for consistency.")
    p.add_argument("--norm_relaxed_col", default=None,
                help="Column to store normalized relaxed CIFs (default: overwrite input col).")
    p.add_argument("--norm_training_col", default=None,
                help="Column to store normalized training CIFs (default: overwrite input col).")
    p.add_argument("--comp_novel_parquet", default="comp_novel.parquet",
                help="If set, write compositionally novel structures to this parquet file.")

    args = p.parse_args()

    relaxed_df = pd.read_parquet(args.relaxed_parquet)
    train_df   = pd.read_parquet(args.training_parquet)
    
    if args.prefilter:
        relaxed_df, train_df = prefilter(relaxed_df, train_df, col_rel=args.relaxed_cif_col, col_train=args.training_cif_col, verbose=True)
        print(f"[INFO] relaxed kept: {len(relaxed_df)}"
              f" (unique formulas={relaxed_df['reduced_formula'].nunique()}).")
        print(f"[INFO] training kept: {len(train_df)}"
              f" (unique formulas={train_df['reduced_formula'].nunique()}).")

    if args.normalize_cifs:
        rel_out_col = args.relaxed_cif_col if not args.norm_relaxed_col else args.norm_relaxed_col
        trn_out_col = args.training_cif_col if not args.norm_training_col else args.norm_training_col

        relaxed_df[rel_out_col] = relaxed_df[args.relaxed_cif_col].map(
            lambda x: _to_norm_cif(
                x,
                do_symmetrize=args.symmetrize,
                symprec=args.symprec,
                sym_angle_tol=args.sym_angle_tol,
                to_primitive=args.to_primitive,
                conventional=args.conventional,
            )
        )
        train_df[trn_out_col] = train_df[args.training_cif_col].map(
            lambda x: _to_norm_cif(
                x,
                do_symmetrize=args.symmetrize,
                symprec=args.symprec,
                sym_angle_tol=args.sym_angle_tol,
                to_primitive=args.to_primitive,
                conventional=args.conventional,
            )
        )

        n_r_before = len(relaxed_df)
        n_t_before = len(train_df)
        relaxed_df = relaxed_df.dropna(subset=[rel_out_col]).copy()
        train_df   = train_df.dropna(subset=[trn_out_col]).copy()
        print(f"[NORM] relaxed {n_r_before} to {len(relaxed_df)}; training {n_t_before} to {len(train_df)}")

        args.relaxed_cif_col  = rel_out_col
        args.training_cif_col = trn_out_col

    if args.attempt_supercell and args.to_primitive:
        print("[WARN] --attempt_supercell and --primitive together reduce matches. "
            "Disabling primitive to allow supercell relations.")
        args.to_primitive = False
    if args.conventional and args.attempt_supercell:
        print("[WARN] --conventional with supercells is unusual; keeping conventional=False.")
        args.conventional = False

    for col, name in [(args.relaxed_cif_col, "relaxed"), (args.training_cif_col, "training")]:
        if col not in (relaxed_df.columns if name == "relaxed" else train_df.columns):
            raise SystemExit(f"[ERROR] {name} CIF column '{col}' not found.")

    print(f"[INFO] relaxed rows: {len(relaxed_df)} | training rows: {len(train_df)}")
    print(f"[INFO] sweeping ltol={args.ltol}, stol={args.stol}, angle_tol={args.angle_tol} | "
          f"symmetrize={args.symmetrize}, primitive={args.to_primitive}, conventional={args.conventional}, "
          f"attempt_supercell={args.attempt_supercell}, allow_subset={args.allow_subset}")

    if args.chunk_size and args.clean_old_shards:
        _clean_stale_shards(prefix=(args.shard_prefix or os.path.splitext(args.out_parquet)[0]))

    matches, unmatched = sweep_match_relaxed_vs_training(
        relaxed=relaxed_df,
        training=train_df,
        relaxed_cif_col=args.relaxed_cif_col,
        training_cif_col=args.training_cif_col,
        training_id_col=args.training_id_col,
        symmetrize=args.symmetrize,
        symprec=args.symprec,
        sym_angle_tol=args.sym_angle_tol,
        to_primitive=args.to_primitive,
        conventional=args.conventional,
        ltol_list=args.ltol,
        stol_list=args.stol,
        angle_tol_list=args.angle_tol,
        attempt_supercell=args.attempt_supercell,
        allow_subset=args.allow_subset,
        max_matches_per_structure=args.max_matches_per_structure,
        carry_relaxed_cols=relaxed_df.columns.tolist(),
        n_jobs=args.n_jobs,
        vol_pad=args.vol_pad,
        debug=args.debug,
        debug_sample=args.debug_sample,
        log_stats=args.log_stats,
        disorder_ok=args.disorder_ok, 
        chunk_size=args.chunk_size,
        shard_prefix = (args.shard_prefix or os.path.splitext(args.out_parquet)[0])
    )
    
    if args.comp_novel_parquet:
        comp_novel = comp_novel_vs_training(
            relaxed_df, train_df,
            col_rel=args.relaxed_cif_col,
            col_train=args.training_cif_col,
            verbose=True,
        )
        out = os.path.abspath(args.comp_novel_parquet)
        out_dir = os.path.dirname(out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        comp_novel.to_parquet(out, index=False)
        print(f"[OK] wrote compositionally novel relaxed structures: {out} (rows={len(comp_novel)})")

    out_dir = os.path.dirname(os.path.abspath(args.out_parquet))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    matches.to_parquet(args.out_parquet, index=False)
    print(f"[OK] wrote matches: {args.out_parquet}")
    print(matches.head(min(10, len(matches))))
    un_out = os.path.abspath(args.unmatched_parquet)
    un_dir = os.path.dirname(un_out)
    if un_dir and not os.path.isdir(un_dir):
        os.makedirs(un_dir, exist_ok=True)
    unmatched.to_parquet(un_out, index=False)
    print(f"[OK] wrote unmatched relaxed: {un_out} (rows={len(unmatched)})")


if __name__ == "__main__":
    main()