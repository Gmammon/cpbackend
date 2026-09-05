"""
Orthogonal Array Engine for Conjoint Analysis
Provides:
  1. Pre-defined Taguchi orthogonal arrays (equal + mixed level)
  2. Column collapsing for related level counts (e.g., 2×2-level → 4-level)
  3. D-optimal fractional factorial design with level balance constraints
"""

from __future__ import annotations
import numpy as np
from itertools import product as iter_product
from typing import Optional


# ── Pre-defined Orthogonal Arrays ──

EQUAL_ARRAYS = {
    "L4(2^3)": {"runs": 4, "col_levels": [2, 2, 2], "data": np.array([
        [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])},
    "L8(2^7)": {"runs": 8, "col_levels": [2]*7, "data": np.array([
        [0,0,0,0,0,0,0],[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[0,1,1,1,1,0,0],
        [1,0,1,0,1,0,1],[1,0,1,1,0,1,0],[1,1,0,0,1,1,0],[1,1,0,1,0,0,1]])},
    "L9(3^4)": {"runs": 9, "col_levels": [3]*4, "data": np.array([
        [0,0,0,0],[0,1,1,1],[0,2,2,2],[1,0,1,2],[1,1,2,0],[1,2,0,1],
        [2,0,2,1],[2,1,0,2],[2,2,1,0]])},
    "L12(2^11)": {"runs": 12, "col_levels": [2]*11, "data": np.array([
        [0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,1],[0,0,1,1,1,0,0,0,1,1,1],
        [0,1,0,1,1,0,1,1,0,0,1],[0,1,1,0,1,1,0,1,0,1,0],[0,1,1,1,0,1,1,0,1,0,0],
        [1,0,1,1,0,0,1,1,0,1,0],[1,0,1,0,1,1,1,0,0,0,1],[1,0,0,1,1,1,0,1,1,0,0],
        [1,1,1,0,0,0,0,1,1,0,1],[1,1,0,1,0,1,0,0,0,1,1],[1,1,0,0,1,0,1,0,1,1,0]])},
    "L16(2^15)": {"runs": 16, "col_levels": [2]*15, "data": np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
        [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
        [0,1,1,0,0,1,1,1,1,0,0,1,1,0,0],
        [0,1,1,1,1,0,0,0,0,1,1,1,1,0,0],
        [0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
        [1,0,1,1,0,1,0,0,1,0,1,1,0,1,0],
        [1,0,1,1,0,1,0,1,0,1,0,0,1,0,1],
        [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],
        [1,1,0,0,1,1,0,1,0,0,1,1,0,0,1],
        [1,1,0,1,0,0,1,0,1,1,0,1,0,0,1],
        [1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]])},
    "L16(4^5)": {"runs": 16, "col_levels": [4]*5, "data": np.array([
        [0,0,0,0,0],[0,1,1,1,1],[0,2,2,2,2],[0,3,3,3,3],
        [1,0,1,2,3],[1,1,0,3,2],[1,2,3,0,1],[1,3,2,1,0],
        [2,0,2,3,1],[2,1,3,2,0],[2,2,0,1,3],[2,3,1,0,2],
        [3,0,3,1,2],[3,1,2,0,3],[3,2,1,3,0],[3,3,0,2,1]])},
    "L25(5^6)": {"runs": 25, "col_levels": [5]*6, "data": np.array([
        [0,0,0,0,0,0],[0,1,1,1,1,1],[0,2,2,2,2,2],[0,3,3,3,3,3],[0,4,4,4,4,4],
        [1,0,1,2,3,4],[1,1,2,3,4,0],[1,2,3,4,0,1],[1,3,4,0,1,2],[1,4,0,1,2,3],
        [2,0,2,4,1,3],[2,1,3,0,2,4],[2,2,4,1,3,0],[2,3,0,2,4,1],[2,4,1,3,0,2],
        [3,0,3,1,4,2],[3,1,4,2,0,3],[3,2,0,3,1,4],[3,3,1,4,2,0],[3,4,2,0,3,1],
        [4,0,4,3,2,1],[4,1,0,4,3,2],[4,2,1,0,4,3],[4,3,2,1,0,4],[4,4,3,2,1,0]])},
    "L27(3^13)": {"runs": 27, "col_levels": [3]*13, "data": np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1],[0,0,0,0,2,2,2,2,2,2,2,2,2],
        [0,1,1,1,0,0,0,1,1,1,2,2,2],[0,1,1,1,1,1,1,2,2,2,0,0,0],
        [0,1,1,1,2,2,2,0,0,0,1,1,1],[0,2,2,2,0,0,0,2,2,2,1,1,1],
        [0,2,2,2,1,1,1,0,0,0,2,2,2],[0,2,2,2,2,2,2,1,1,1,0,0,0],
        [1,0,1,2,0,1,2,0,1,2,0,1,2],[1,0,1,2,1,2,0,1,2,0,1,2,0],
        [1,0,1,2,2,0,1,2,0,1,2,0,1],[1,1,2,0,0,1,2,1,2,0,2,0,1],
        [1,1,2,0,1,2,0,2,0,1,0,1,2],[1,1,2,0,2,0,1,0,1,2,1,2,0],
        [1,2,0,1,0,1,2,2,0,1,1,2,0],[1,2,0,1,1,2,0,0,1,2,2,0,1],
        [1,2,0,1,2,0,1,1,2,0,0,1,2],[2,0,2,1,0,2,1,0,2,1,0,2,1],
        [2,0,2,1,1,0,2,1,0,2,1,0,2],[2,0,2,1,2,1,0,2,1,0,2,1,0],
        [2,1,0,2,0,2,1,1,0,2,2,1,0],[2,1,0,2,1,0,2,2,1,0,0,2,1],
        [2,1,0,2,2,1,0,0,2,1,1,0,2],[2,2,1,0,0,2,1,2,1,0,1,0,2],
        [2,2,1,0,1,0,2,0,2,1,2,1,0],[2,2,1,0,2,1,0,1,0,2,0,2,1]])},
}

MIXED_ARRAYS = {
    "L18(2^1x3^7)": {"runs": 18, "col_levels": [2,3,3,3,3,3,3,3], "data": np.array([
        [0,0,0,0,0,0,0,0],[0,0,0,1,1,1,1,1],[0,0,0,2,2,2,2,2],
        [0,1,1,0,0,1,2,2],[0,1,1,1,1,2,0,0],[0,1,1,2,2,0,1,1],
        [1,0,1,0,1,2,2,0],[1,0,1,1,2,0,0,1],[1,0,1,2,0,1,1,2],
        [1,1,0,0,2,1,0,2],[1,1,0,1,0,2,1,0],[1,1,0,2,1,0,2,1],
        [0,0,1,0,2,2,1,1],[0,1,0,0,1,0,2,2],[1,0,0,1,0,1,1,2],
        [1,1,1,2,2,2,0,0],[0,0,1,2,1,0,0,2],[0,1,0,1,2,1,2,0]])},
}


def _try_match_columns(col_levels: list[int], attr_levels: list[int]) -> Optional[list[int]]:
    """Try to match attribute level counts to array columns (exact match)."""
    used = set()
    mapping = []
    for al in attr_levels:
        found = False
        for ci, cl in enumerate(col_levels):
            if ci not in used and cl == al:
                used.add(ci)
                mapping.append(ci)
                found = True
                break
        if not found:
            return None
    return mapping


def _try_collapsed_match(attr_levels: list[int]) -> Optional[tuple[str, np.ndarray]]:
    """Try to build a design by collapsing columns from 2-level arrays."""
    total_cols_needed = 0
    col_groups = []

    for al in attr_levels:
        k = 1
        while (1 << k) < al:
            k += 1
        total_cols_needed += k
        col_groups.append((k, al))

    best_arr = None
    best_name = None
    for name, arr_def in sorted(EQUAL_ARRAYS.items(), key=lambda x: x[1]["runs"]):
        if arr_def["col_levels"][0] == 2 and len(arr_def["col_levels"]) >= total_cols_needed:
            if best_arr is None or arr_def["runs"] < best_arr["runs"]:
                best_arr = arr_def
                best_name = name

    if best_arr is None:
        return None

    data = best_arr["data"]
    design = np.zeros((data.shape[0], len(attr_levels)), dtype=int)
    col_offset = 0
    for i, (k, al) in enumerate(col_groups):
        sub = data[:, col_offset:col_offset + k]
        powers = 2 ** np.arange(k - 1, -1, -1)
        vals = sub @ powers
        design[:, i] = vals % al
        col_offset += k

    return f"Collapsed {best_name}", design


def find_best_orthogonal_array(attr_levels: list[int]) -> Optional[dict]:
    """Find the best matching orthogonal array for the given attribute level counts."""
    n_attrs = len(attr_levels)
    full_size = int(np.prod(attr_levels))

    all_arrays = {**EQUAL_ARRAYS, **MIXED_ARRAYS}
    candidates = sorted(all_arrays.items(), key=lambda x: x[1]["runs"])

    for name, arr_def in candidates:
        if arr_def["runs"] > full_size:
            continue
        if len(arr_def["col_levels"]) < n_attrs:
            continue
        mapping = _try_match_columns(arr_def["col_levels"], attr_levels)
        if mapping is not None:
            design = arr_def["data"][:, mapping]
            return {
                "design_matrix": design.tolist(),
                "array_name": name,
                "is_fractional": False,
                "total_trials": design.shape[0],
            }

    return None


def generate_full_factorial(attr_levels: list[int]) -> np.ndarray:
    """Generate full factorial design matrix."""
    level_ranges = [list(range(l)) for l in attr_levels]
    return np.array(list(iter_product(*level_ranges)))


def _build_design_matrix(design: np.ndarray, attr_levels: list[int]) -> np.ndarray:
    """Build OLS design matrix with intercept + dummy coding."""
    n = design.shape[0]
    n_attrs = design.shape[1]

    cols = [np.ones(n)]
    for i in range(n_attrs):
        for j in range(1, attr_levels[i]):
            cols.append((design[:, i] == j).astype(float))

    return np.column_stack(cols)


def _d_efficiency(X: np.ndarray) -> float:
    """Compute D-efficiency = det(X'X)^(1/p) / n."""
    n, p = X.shape
    try:
        XtX = X.T @ X
        det_val = np.linalg.det(XtX)
        if det_val <= 0:
            return 0.0
        return det_val ** (1.0 / p) / n
    except np.linalg.LinAlgError:
        return 0.0


def generate_d_optimal(attr_levels: list[int], n_runs: int, max_iter: int = 500) -> dict:
    """Generate D-optimal design using coordinate exchange with level balance."""
    full = generate_full_factorial(attr_levels)
    n_full = full.shape[0]
    n_attrs = len(attr_levels)

    if n_runs >= n_full:
        return {
            "design_matrix": full.tolist(),
            "array_name": f"Full Factorial ({n_full} runs)",
            "is_fractional": False,
            "total_trials": n_full,
        }

    best_design = None
    best_eff = -1

    for restart in range(10):
        if restart == 0:
            indices = _balanced_init(attr_levels, n_runs, n_full)
        else:
            indices = np.random.choice(n_full, n_runs, replace=False)

        design = full[indices].copy()
        X = _build_design_matrix(design, attr_levels)
        current_eff = _d_efficiency(X)

        for iteration in range(max_iter):
            improved = False
            attr_order = np.random.permutation(n_attrs)

            for row_idx in range(n_runs):
                for attr_idx in attr_order:
                    current_val = design[row_idx, attr_idx]
                    best_val = current_val
                    best_row_eff = current_eff

                    for candidate_val in range(attr_levels[attr_idx]):
                        if candidate_val == current_val:
                            continue

                        old_val = design[row_idx, attr_idx]
                        design[row_idx, attr_idx] = candidate_val
                        X_new = _build_design_matrix(design, attr_levels)
                        new_eff = _d_efficiency(X_new)

                        if new_eff > best_row_eff:
                            best_row_eff = new_eff
                            best_val = candidate_val

                        design[row_idx, attr_idx] = old_val

                    if best_val != current_val:
                        design[row_idx, attr_idx] = best_val
                        X = _build_design_matrix(design, attr_levels)
                        current_eff = best_row_eff
                        improved = True

            if not improved:
                break

        if current_eff > best_eff:
            best_eff = current_eff
            best_design = design.copy()

    return {
        "design_matrix": best_design.tolist(),
        "array_name": f"D-optimal ({n_runs} runs, eff={best_eff:.4f})",
        "is_fractional": True,
        "total_trials": n_runs,
    }


def _balanced_init(attr_levels: list[int], n_runs: int, n_full: int) -> np.ndarray:
    """Create a balanced initial selection where each level appears ~equally."""
    full = generate_full_factorial(attr_levels)

    target_counts = {}
    for i, l in enumerate(attr_levels):
        base = n_runs // l
        remainder = n_runs % l
        target_counts[i] = [base + (1 if j < remainder else 0) for j in range(l)]

    selected = []
    current_counts = {i: [0] * l for i, l in enumerate(attr_levels)}
    available = list(range(n_full))

    for _ in range(n_runs):
        best_idx = -1
        best_score = -1

        for idx in available:
            profile = full[idx]
            score = 0
            for i, l in enumerate(attr_levels):
                level = profile[i]
                target = target_counts[i][level]
                current = current_counts[i][level]
                if current < target:
                    score += (target - current)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            best_idx = available[0]

        selected.append(best_idx)
        profile = full[best_idx]
        for i in range(len(attr_levels)):
            current_counts[i][profile[i]] += 1
        available.remove(best_idx)

    return np.array(selected)


def generate_design(attr_levels: list[int]) -> dict:
    """
    Main API: generate the best possible design for the given attributes.

    Args:
        attr_levels: list of level counts, e.g. [2, 3, 4]

    Returns:
        dict with design_matrix, array_name, is_fractional, total_trials
    """
    if not attr_levels or any(l < 2 for l in attr_levels):
        return {"error": "Each attribute must have at least 2 levels"}

    if len(attr_levels) > 8:
        return {"error": "Maximum 8 attributes supported"}

    if any(l > 5 for l in attr_levels):
        return {"error": "Maximum 5 levels per attribute supported"}

    full_size = int(np.prod(attr_levels))

    if full_size <= 20:
        full = generate_full_factorial(attr_levels)
        return {
            "design_matrix": full.tolist(),
            "array_name": f"Full Factorial ({full_size} runs)",
            "is_fractional": False,
            "total_trials": full_size,
        }

    result = find_best_orthogonal_array(attr_levels)
    if result:
        return result

    min_trials = sum(l - 1 for l in attr_levels) + 1
    n_runs = min(
        np.prod(attr_levels),
        max(min_trials * 2, 8)
    )

    return generate_d_optimal(attr_levels, n_runs)
