"""
MaxDiff 核心算法：BIBD 设计生成 + MLE 效用估计 + Count Model
从 maxdiff-backend 移植，去除 SQLAlchemy 依赖，改为纯函数接口。
"""

import math
import random
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# ============================================================
# BIBD 设计生成
# ============================================================

KNOWN_BIBD = {
    (7, 3): [[0,1,3],[1,2,4],[2,3,5],[3,4,6],[4,5,0],[5,6,1],[6,0,2]],
    (7, 4): [[0,1,2,4],[1,2,3,5],[2,3,4,6],[3,4,5,0],[4,5,6,1],[5,6,0,2],[6,0,1,3]],
    (9, 3): [[0,1,2],[0,3,6],[0,4,8],[0,5,7],[1,3,8],[1,4,7],[1,5,6],[2,3,7],[2,4,6],[2,5,8],[3,4,5],[6,7,8]],
    (9, 4): [[0,1,3,7],[0,2,5,6],[0,4,8,3],[1,2,4,8],[1,5,6,3],[2,6,7,4],[3,5,8,6],[4,6,7,1],[7,8,5,2]],
    (8, 4): [[0,1,2,3],[0,1,4,5],[0,2,4,6],[0,3,5,6],[1,2,5,6],[1,3,4,6],[2,3,4,5],[0,1,6,7],[0,2,5,7],[0,3,4,7],[1,2,3,7],[1,4,5,7],[2,4,6,7],[3,5,6,7]],
    (5, 3): [[0,1,2],[0,1,3],[0,2,4],[0,3,4],[1,2,3],[1,3,4],[2,3,4],[1,2,4],[0,1,4],[0,2,3]],
    (5, 4): [[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]],
    (6, 3): [[0,1,2],[0,3,4],[1,3,5],[2,4,5],[0,1,5],[0,2,4],[1,3,4],[2,3,5],[0,3,5],[1,2,4]],
}


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def try_cyclic_bibd(v: int, k: int) -> Optional[list]:
    if not is_prime(v) or v < k:
        return None

    rng = random.Random(42)
    for lambda_target in range(1, 6):
        denom = k * (k - 1)
        numer = v * (v - 1) * lambda_target
        if numer % denom != 0:
            continue
        b = numer // denom
        r = (v - 1) * lambda_target // (k - 1)
        if not float(r).is_integer():
            continue

        for _ in range(3000):
            base = sorted(rng.sample(range(v), k))
            blocks = []
            for shift in range(v):
                block = sorted((x + shift) % v for x in base)
                blocks.append(block)
            unique = []
            seen = set()
            for block in blocks:
                key = tuple(block)
                if key not in seen:
                    seen.add(key)
                    unique.append(block)
            if len(unique) < b:
                continue
            selected = unique[:b]
            pair_count = {}
            for block in selected:
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        key = (min(block[i], block[j]), max(block[i], block[j]))
                        pair_count[key] = pair_count.get(key, 0) + 1
            total_pairs = v * (v - 1) // 2
            if len(pair_count) != total_pairs:
                continue
            if all(c == lambda_target for c in pair_count.values()):
                return selected
    return None


def compute_design_cost(design: list, n: int, k: int) -> float:
    T = len(design)
    counts = [0] * n
    for task in design:
        for idx in task:
            counts[idx] += 1
    avg_count = (T * k) / n
    count_var = sum((c - avg_count) ** 2 for c in counts)

    pair_counts = {}
    for task in design:
        for i in range(len(task)):
            for j in range(i + 1, len(task)):
                key = (min(task[i], task[j]), max(task[i], task[j]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_vals = list(pair_counts.values())
    if pair_vals:
        avg_pair = sum(pair_vals) / len(pair_vals)
        pair_var = sum((p - avg_pair) ** 2 for p in pair_vals)
    else:
        pair_var = 0

    total_pairs = n * (n - 1) // 2
    target_pair = (T * k * (k - 1)) / (n * (n - 1))
    missing = total_pairs - len(pair_vals)
    pair_var += missing * max(target_pair + 1, 2) ** 2

    return count_var * 10 + pair_var


def optimize_design(design: list, n: int, k: int, max_iter: int, rng: random.Random) -> list:
    best = [t[:] for t in design]
    best_cost = compute_design_cost(best, n, k)

    for _ in range(max_iter):
        t1 = rng.randint(0, len(best) - 1)
        t2 = rng.randint(0, len(best) - 1)
        if t1 == t2:
            continue
        i1 = rng.randint(0, len(best[t1]) - 1)
        i2 = rng.randint(0, len(best[t2]) - 1)
        item1, item2 = best[t1][i1], best[t2][i2]
        if item1 == item2:
            continue
        if item2 in best[t1] or item1 in best[t2]:
            continue

        best[t1][i1], best[t2][i2] = item2, item1
        new_cost = compute_design_cost(best, n, k)
        if new_cost < best_cost:
            best_cost = new_cost
            if best_cost == 0:
                break
        else:
            best[t1][i1], best[t2][i2] = item1, item2

    return best


def get_design_metrics(design: list, n: int, k: int) -> dict:
    T = len(design)
    counts = [0] * n
    for task in design:
        for idx in task:
            counts[idx] += 1

    pair_counts = {}
    for task in design:
        for i in range(len(task)):
            for j in range(i + 1, len(task)):
                key = (min(task[i], task[j]), max(task[i], task[j]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_vals = list(pair_counts.values())
    total_pairs = n * (n - 1) // 2
    covered = len(pair_vals)
    pair_min = min(pair_vals) if pair_vals else 0
    pair_max = max(pair_vals) if pair_vals else 0
    max_deviation = pair_max - pair_min

    coverage = covered / total_pairs
    balance = pair_min / max(pair_max, 1)
    d_efficiency = round(coverage * balance * 100, 1)

    return {
        "task_count": T,
        "item_count": n,
        "appearance_min": min(counts),
        "appearance_max": max(counts),
        "covered_pairs": covered,
        "total_pairs": total_pairs,
        "pair_min": pair_min,
        "pair_max": pair_max,
        "max_pair_deviation": max_deviation,
        "d_efficiency": d_efficiency,
        "is_bibd": max_deviation <= 1 and min(counts) == max(counts),
    }


def generate_design(
    items: list, set_size: int, appearances: int, num_candidates: int = 500, seed: int = None
) -> dict:
    n = len(items)
    k = set_size
    r = appearances

    if seed is None:
        seed = random.randint(1, 2**31)
    rng = random.Random(seed)

    design = None
    method = ""

    # 方法 1：预计算 BIBD
    key = (n, k)
    if key in KNOWN_BIBD and KNOWN_BIBD[key] is not None:
        bibd = KNOWN_BIBD[key]
        bibd_r = len(bibd) * k // n
        if abs(bibd_r - r) <= 1:
            design = bibd
            method = f"预计算BIBD (r={bibd_r})"

    # 方法 2：循环 BIBD
    if design is None and is_prime(n):
        cyclic = try_cyclic_bibd(n, k)
        if cyclic is not None:
            design = cyclic
            method = "循环BIBD"

    # 方法 3：随机搜索 + 贪婪交换
    if design is None:
        T = math.ceil((n * r) / k)
        total_slots = T * k
        extra = total_slots - n * r

        best_design = None
        best_cost = float("inf")

        for _ in range(num_candidates):
            pool = []
            for i in range(n):
                pool.extend([i] * r)
            if extra > 0:
                idxs = list(range(n))
                rng.shuffle(idxs)
                pool.extend(idxs[:extra])

            rng.shuffle(pool)
            candidate = [pool[t * k : (t + 1) * k] for t in range(T)]

            valid = True
            for t in range(len(candidate)):
                seen = set()
                dupes = []
                for j in range(len(candidate[t])):
                    if candidate[t][j] in seen:
                        dupes.append(j)
                    else:
                        seen.add(candidate[t][j])
                for dj in dupes:
                    swapped = False
                    for t2 in range(len(candidate)):
                        if t2 == t:
                            continue
                        for j2 in range(len(candidate[t2])):
                            if candidate[t2][j2] not in seen and candidate[t][dj] not in set(candidate[t2]) - {candidate[t2][j2]}:
                                candidate[t][dj], candidate[t2][j2] = candidate[t2][j2], candidate[t][dj]
                                seen.add(candidate[t][dj])
                                swapped = True
                                break
                        if swapped:
                            break
                    if not swapped:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            has_dupes = any(len(set(t)) < len(t) for t in candidate)
            if has_dupes:
                continue

            cost = compute_design_cost(candidate, n, k)
            if cost < best_cost:
                best_cost = cost
                best_design = [t[:] for t in candidate]
            if best_cost == 0:
                break

        if best_design and best_cost > 0:
            best_design = optimize_design(best_design, n, k, 5000, rng)

        design = best_design
        method = f"随机搜索 ({num_candidates}候选)"

    if design is None:
        return None

    metrics = get_design_metrics(design, n, k)

    # 转换为选项名称 + 位置随机化
    tasks = []
    for task in design:
        names = [items[idx] for idx in task]
        task_rng = random.Random(seed + sum(hash(str(items[idx])) for idx in task))
        task_rng.shuffle(names)
        tasks.append(names)

    return {
        "tasks": tasks,
        "duplicate_pairs": [],
        "metrics": metrics,
        "seed": seed,
        "method": method,
    }


def insert_duplicate_tasks(tasks: list, num_dupes: int = None) -> tuple:
    if len(tasks) < 4:
        return tasks[:], []

    if num_dupes is None:
        num_dupes = 2 if len(tasks) >= 8 else 1

    T = len(tasks)
    start = int(T * 0.2)
    end = int(T * 0.8)

    used = set()
    dup_tasks = []
    for _ in range(num_dupes):
        idx = random.randint(start, end - 1)
        while idx in used:
            idx = random.randint(start, end - 1)
        used.add(idx)
        dup_tasks.append({"orig_idx": idx, "content": tasks[idx][:]})

    dup_tasks.sort(key=lambda x: -x["orig_idx"])
    result = tasks[:]
    duplicate_pairs = []
    for dt in dup_tasks:
        insert_at = min(dt["orig_idx"] + 2 + random.randint(0, 2), len(result))
        result.insert(insert_at, dt["content"])
        duplicate_pairs.append({"original": dt["orig_idx"], "duplicate": insert_at})

    duplicate_pairs.sort(key=lambda x: x["original"])
    return result, duplicate_pairs


# ============================================================
# MLE 效用估计
# ============================================================

def mle_estimate(responses: list, items: list) -> dict:
    n = len(items)
    item_idx = {item: i for i, item in enumerate(items)}

    tasks = []
    for r in responses:
        idxs = [item_idx[it] for it in r["items"] if it in item_idx]
        b = item_idx.get(r["best"])
        w = item_idx.get(r["worst"])
        if b is not None and w is not None and len(idxs) >= 2:
            tasks.append({"items": idxs, "best": b, "worst": w})

    if not tasks:
        return None

    N = len(tasks)

    def neg_log_likelihood(u):
        ll = 0.0
        for t in tasks:
            S = t["items"]
            b, w = t["best"], t["worst"]
            max_diff = max(u[i] - u[j] for i in S for j in S if i != j)
            if not np.isfinite(max_diff):
                max_diff = 0.0
            Z = sum(np.exp(u[i] - u[j] - max_diff) for i in S for j in S if i != j)
            if Z <= 0:
                return 1e10
            ll += u[b] - u[w] - np.log(Z) - max_diff
        return -ll

    def gradient(u):
        grad = np.zeros(n)
        for t in tasks:
            S = t["items"]
            b, w = t["best"], t["worst"]
            max_diff = max(u[i] - u[j] for i in S for j in S if i != j)
            if not np.isfinite(max_diff):
                max_diff = 0.0
            Z = sum(np.exp(u[i] - u[j] - max_diff) for i in S for j in S if i != j)
            if Z <= 0:
                continue
            for kk in S:
                sw = sum(np.exp(u[kk] - u[j] - max_diff) for j in S if j != kk)
                sl = sum(np.exp(u[j] - u[kk] - max_diff) for j in S if j != kk)
                g = (1.0 if kk == b else 0.0) - (1.0 if kk == w else 0.0) - (sw - sl) / Z
                grad[kk] += g
        return -grad

    x0 = np.zeros(n)
    constraints = {"type": "eq", "fun": lambda u: np.sum(u)}
    bounds = [(-50, 50)] * n

    result = minimize(
        neg_log_likelihood,
        x0,
        method="SLSQP",
        jac=gradient,
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    u_opt = result.x

    standard_errors = np.full(n, np.nan)
    try:
        from scipy.optimize import approx_fprime

        hess = np.zeros((n, n))
        eps = 1e-5
        for i in range(n):
            def gi(u):
                return gradient(u)[i]
            hess[i, :] = approx_fprime(u_opt, gi, eps)

        hess += np.eye(n) * 1e-4
        cov = np.linalg.inv(hess)
        standard_errors = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        pass

    max_u = np.max(u_opt)
    exp_u = np.exp(np.clip(u_opt - max_u, -50, 50))
    scores_arr = exp_u / exp_u.sum() * 100

    utilities = {items[i]: float(u_opt[i]) for i in range(n)}
    scores = {items[i]: float(scores_arr[i]) for i in range(n)}
    se_dict = {items[i]: float(standard_errors[i]) for i in range(n)}

    rlh = np.exp(result.fun / (-N))
    avg_k = np.mean([len(t["items"]) for t in tasks])
    random_rlh = 1.0 / (avg_k * (avg_k - 1))

    return {
        "utilities": utilities,
        "scores": scores,
        "standard_errors": se_dict,
        "log_likelihood": float(-result.fun),
        "rlh": float(rlh),
        "random_rlh": float(random_rlh),
        "rlh_ratio": float(rlh / random_rlh) if random_rlh > 0 else 0,
        "iterations": result.nit,
        "converged": bool(result.success),
    }


def count_model(responses: list, items: list) -> dict:
    best_count = {item: 0 for item in items}
    worst_count = {item: 0 for item in items}
    for r in responses:
        if r["best"] in best_count:
            best_count[r["best"]] += 1
        if r["worst"] in worst_count:
            worst_count[r["worst"]] += 1

    diff = {item: best_count[item] - worst_count[item] for item in items}
    vals = list(diff.values())
    min_v, max_v = min(vals), max(vals)
    rng = max_v - min_v or 1
    scores = {item: (diff[item] - min_v) / rng * 100 for item in items}

    return {
        "best_count": best_count,
        "worst_count": worst_count,
        "diff": diff,
        "scores": scores,
    }
