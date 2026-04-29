"""
Small script to generate compatible prompts for MAX and MAB phases. 
"""

from typing import List, Union
import argparse
from smact.screening import smact_validity
import pandas as pd 
from pymatgen.core import Composition
import re
from pymatgen.core import Composition
import numpy as np
import os


def generate_MAX_formulas(M: List[str] = ["Ti", "Nb", "Cr", "V"], A: List[str] = ["Al", "Si"], X: List[str] = ["C", "N"], max_n: int = 4) -> List[str]:
    """
    Generate all single‐ and double‐transition‐metal MAX formulas
    with M‐site multiplicity (n+1) and X‐site count n, for n=1..max_n.
    Omits the subscript "1" for readability.
    """
    formulas = []
    for n in range(1, max_n + 1):
        m_count = n + 1
        # for X-site, we only attach a number if >1
        x_sub = str(n) if n > 1 else "1"
        for a in A:
            for x in X:
                for i, m1 in enumerate(M):
                    for m2 in M[i:]:
                        if m1 == m2:
                            # Single-metal: e.g. "Ti2AlC" or "Ti3AlC2"
                            formula = f"{m1}{m_count}{a}1{x}{x_sub}"
                            formulas.append(formula)
                        else:
                            # Double-metal: split m_count into two positive ints
                            for m1_count in range(1, m_count):
                                m2_count = m_count - m1_count
                                # Only show the count if >1
                                m1_sub = f"{m1_count}" if m1_count > 1 else "1"
                                m2_sub = f"{m2_count}" if m2_count > 1 else "1"
                                mids = f"{m1}{m1_sub}{m2}{m2_sub}"
                                formula = f"{mids}{a}1{x}{x_sub}"
                                formulas.append(formula)

    valid_formulas = [formula for formula in formulas if smact_validity(formula)]

    return valid_formulas

def generate_MAB_formulas(M: List[str] = ["Ti", "Nb", "Cr", "V"], A: List[str] = ["Al", "Si"], B: str = "B") -> List[str]:
    """Generate other stoichiometries based on M, A, B elements.

    Args:
        M (str): Ti, V, Cr, or Nb
        A (str): Aluminum or Silicon.
        B (str): Boron
        max_n (int): The maximum number of atoms.
        
    Note:
        Listed allowed stoichiometries are as follows:
        M2AB2, M3AB4, M4AB6, M4AB4, M2A2B2

    Returns:
        list: A list of generated stoichiometries.
    """
    
    formulas = set()
    
    m_count = [2, 3, 4]
    for n in m_count:
        if n ==2:
            for elem_M in M:
                for elem_A in A:
                    formula_1 = f"{elem_M}{n}{elem_A}2B2"
                    formula_2 = f"{elem_M}{n}{elem_A}B2"
                    formulas.add(formula_1)
                    formulas.add(formula_2)
        elif n ==3:
            for elem_M in M:
                for elem_A in A:
                    formula_3 = f"{elem_M}{n}{elem_A}1B4"
                    formulas.add(formula_3)
        elif n==4:
            for elem_M in M:
                for elem_A in A:
                    formula_4 = f"{elem_M}{n}{elem_A}1B6"
                    formulas.add(formula_4)
                    formula_5 = f"{elem_M}{n}{elem_A}1B4"
                    formulas.add(formula_5)

    valid_formulas = [formula for formula in formulas if smact_validity(formula)]
    return valid_formulas

def scale_stoichiometry(
    formulas: List[str],
    factor: int,
    *,
    write_one: bool = True,
    snap_eps: float = 1e-8,
    enforce_integer: bool = False,
) -> List[str]:
    """
    Scale each chemical formula's stoichiometries by an integer 'factor'.

    - Preserves the element order as it appears in the original string.
    - Writes explicit counts (e.g. 'Ti1', 'C2'); set write_one=True to force '1'.
    - Snaps counts that are within 'snap_eps' of an integer to that integer.
    - If enforce_integer=True, raises ValueError if any scaled count is not ~integer.

    Parameters
    ----------
    formulas : list of str
        Input formula strings (e.g., 'Ti3AlC2', 'Cr2Ti1Al1N1', 'Fe0.5Ni0.5').
    factor : int
        Integer multiplier for all stoichiometric counts.
    write_one : bool
        If True, prints '1' explicitly (e.g., 'Al1'); if False, omits 1.
    snap_eps : float
        Tolerance to snap floats to nearest integer.
    enforce_integer : bool
        If True, error if any scaled count is not (within snap_eps of) an integer.

    Returns
    -------
    List[str]
        Scaled formulas in the same per-formula order.
    """
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError("factor must be a positive integer.")

    out: List[str] = []
    el_order_re = re.compile(r'([A-Z][a-z]?)')  # to capture original element order

    def _fmt_count(n):
        # n is int (after snapping) or float; format with/without '1' as requested
        if isinstance(n, int):
            return str(n) if (write_one or n != 1) else ""
        # float fallback (rare)
        txt = f"{n:g}"
        return txt if (write_one or abs(n - 1.0) > snap_eps) else ""

    for f in formulas:
        comp = Composition(f)
        d = comp.get_el_amt_dict()  # {'Ti': 3.0, 'Al': 1.0, 'C': 2.0} (floats)

        # scale counts
        scaled = {el: amt * factor for el, amt in d.items()}

        # snap near-integers
        snapped = {}
        for el, v in scaled.items():
            r = round(v)
            if abs(v - r) < snap_eps:
                snapped[el] = int(r)
            else:
                if enforce_integer:
                    raise ValueError(
                        f"Scaled count for {el} in '{f}' is non-integer ({v}); "
                        "set enforce_integer=False or adjust factor."
                    )
                snapped[el] = v  # keep float if not enforcing

        # rebuild string in the *original* element order
        seen = set()
        parts = []
        for el in el_order_re.findall(f):
            if el in snapped and el not in seen:
                parts.append(f"{el}{_fmt_count(snapped[el])}")
                seen.add(el)

        # append any elements not literally present in the string (rare, e.g., from parentheses)
        for el, v in snapped.items():
            if el not in seen:
                parts.append(f"{el}{_fmt_count(v)}")

        out.append("".join(parts))

    return out



def process_and_export(new_valid_formulas, file_name="new_valid_formulas.parquet",
                       condition_vector: List[float] = None, no_condition_vector: bool = False):
    import pandas as pd

    formulas = list(new_valid_formulas)
    df = pd.DataFrame({
        "Material ID": range(len(formulas)),
        "formula": formulas,
        "Prompt": [f"<bos>\ndata_[{f}]\n" for f in formulas],
        "Database": ["N/A"] * len(formulas),
    })

    if no_condition_vector or condition_vector is None:
        df["condition_vector"] = np.nan
    else:
        # normalize to list[float]
        if isinstance(condition_vector, (int, float)):
            cv = [float(condition_vector)]
        elif isinstance(condition_vector, (list, tuple, np.ndarray)):
            cv = [float(x) for x in condition_vector]
        else:
            raise TypeError(f"Unsupported type for condition_vector: {type(condition_vector)}")
        # broadcast the SAME list to every row
        df["condition_vector"] = [cv] * len(df)

    df.to_parquet(file_name, index=False, engine="pyarrow")
    return df

def _normalize_condition_vector(args):
    # --no-condition-vector wins
    if args.no_condition_vector:
        return None
    cv = args.condition_vector
    if cv is None:
        return None
    # validate in [0,1]
    # for x in cv:
    #     if not (0.0 <= x <= 1.0):
    #         raise ValueError(f"--condition_vector entries must be in [0,1], got {x}")
    # # optional: avoid exact 1.0 if needed by your model
    # cv = [0.999 if abs(x - 1.0) < 1e-12 else float(x) for x in cv]
    return cv

def main():
    parser = argparse.ArgumentParser(description="Generate MAX and MAB formulas")
    parser.add_argument("--max_n", type=int, default=4, help="Maximum n for MAX phase generation")
    parser.add_argument("--M", nargs="+", default=["Ti", "Nb", "Cr", "V"], help="List of M elements")
    parser.add_argument("--A", nargs="+", default=["Al", "Si"], help="List of A elements")
    parser.add_argument("--X", nargs="+", default=["C", "N"], help="List of X elements")
    parser.add_argument("--generate_MAX_formulas", action="store_true", help="Generate MAX formulas")
    parser.add_argument("--generate_MAB_formulas", action="store_true", help="Generate MAB formulas")
    parser.add_argument("--export_parquet", action="store_true", help="Export generated formulas to a parquet file")
    parser.set_defaults(export_parquet=True)
    parser.add_argument("--export_file_name", type=str, default="MAX_MAB_prompts.parquet", help="Name of the output prompt parquet file")
    parser.add_argument("--print_formulas", action="store_true", help="Print generated formulas to console")
    parser.add_argument("--condition_vector",nargs = "+", type=float, default=None, help="Condition vector for the prompts")
    parser.add_argument("--no-condition-vector", action="store_true", dest="no_condition_vector", help="Disable condition vector for the prompts")
    parser.add_argument("--scale_stoichiometry", action="store_true", help="Scales the stoichiometry for each compound in the valid formulas list")
    parser.add_argument("--factor", default=2, type=int, help="Scaling factor for stoichiometry scaling")
    args = parser.parse_args()

    cond_vec = _normalize_condition_vector(args)

    cond_val = _normalize_condition_vector(args)  

    if args.generate_MAX_formulas and not args.generate_MAB_formulas:
        max_formulas = generate_MAX_formulas(M=args.M, A=args.A, X=args.X, max_n=args.max_n)
        if args.scale_stoichiometry:
            max_formulas = scale_stoichiometry(max_formulas, factor=args.factor)
        if args.export_parquet:
            df_max = process_and_export(set(max_formulas), file_name=args.export_file_name, condition_vector=cond_val)
            print(f"MAX formulas exported to {args.export_file_name}")

    elif args.generate_MAB_formulas and not args.generate_MAX_formulas:
        mab_formulas = generate_MAB_formulas(M=args.M, A=args.A)
        if args.scale_stoichiometry:
            mab_formulas = scale_stoichiometry(mab_formulas, factor=args.factor)
        if args.export_parquet:
            df_mab = process_and_export(set(mab_formulas), file_name=args.export_file_name, condition_vector=cond_val)
            print(f"MAB formulas exported to {args.export_file_name}")
        else:
            print("MAB formulas generated but not exported")

    if args.generate_MAX_formulas and args.generate_MAB_formulas:
        max_formulas = generate_MAX_formulas(M=args.M, A=args.A, X=args.X, max_n=args.max_n)
        if args.scale_stoichiometry:
            max_formulas = scale_stoichiometry(max_formulas, factor=args.factor)
        mab_formulas = generate_MAB_formulas(M=args.M, A=args.A)
        if args.scale_stoichiometry:
            mab_formulas = scale_stoichiometry(mab_formulas, factor=args.factor)
        if args.export_parquet:
            full_formula_list = list(max_formulas) + list(mab_formulas)
            df_full = process_and_export(set(full_formula_list), file_name=args.export_file_name, condition_vector=cond_val)
            print(f"MAX and MAB formulas exported to {args.export_file_name}")
        else:
            print("MAX formulas generated but not exported")
            
    if not args.generate_MAX_formulas and not args.generate_MAB_formulas:
        raise ValueError("At least one of --generate_MAX_formulas or --generate_MAB_formulas must be specified.")

    if args.print_formulas:
        try: 
            print("Generated MAX formulas:")
            for formula in max_formulas:
                print(formula)
        except Exception as e:
            print("No MAX formulas generated. Please check the arguments.")

        try:
            print("\nGenerated MAB formulas:")
            for formula in mab_formulas:
                print(formula)
        except Exception as e:
            print("No MAB formulas generated. Please check the arguments.")

if __name__ == "__main__":
    main()