"""
Fast Polyhedral Adaptive Conjoint Analysis (ACA) Engine
=======================================================
Based on Toubia et al. (2003), Marketing Science 22(3).

Key design choices (vs. the earlier one-hot / effect-coded version):
- Baseline (dummy) coding: each attribute's first/lowest level is the
  baseline, fixed at utility 0; every non-baseline level is its own
  parameter measuring the utility GAIN over the baseline. All parameters
  are constrained to [0, ub] so gains are non-negative and a single
  attribute's contribution to a rating never exceeds ub.
- Optional monotone ordering constraints (u_j <= u_{j+1}) for attributes
  flagged as ordered (levels entered worst -> best).
- Cognitive-load cap: each question changes at most 3 attributes.
- Question budget = p (number of parameters); adaptive convergence only
  kicks in if a smaller min/max is configured.
- First question balances level coverage across respondents (low-frequency
  levels favoured), since the initial polyhedron is the full box and has no
  principal axis to guide it.
- Inconsistent responses: estimation falls back to minimax relaxation, but
  question design falls back to random (the relaxed center is distorted by
  delta and must not steer the next question).
- Session management (TTL + respondent_id), 0-rating axis switch,
  solver selection (OSQP for p > 30).
"""

import time
import uuid
import numpy as np
from numpy.linalg import pinv, eig
import cvxpy as cp
from scipy.optimize import linprog

# ── Session Management ──

SESSION_TTL = 1800  # 30 minutes

sessions: dict = {}


def total_params(attrs):
    """Number of utility parameters under baseline coding:
    each attribute contributes (levels - 1), the baseline level is not a parameter."""
    return sum(max(len(a['levels']) - 1, 0) for a in attrs)

import time
import uuid
import numpy as np
from numpy.linalg import pinv, eig
import cvxpy as cp
from scipy.optimize import linprog

# ── Session Management ──

SESSION_TTL = 1800  # 30 minutes

sessions: dict = {}


def start_session(respondent_id: str, attrs: list, ub: float = 100.0,
                  max_questions: int = None, min_questions: int = None,
                  convergence_threshold: float = 0.02,
                  consecutive_count: int = 3,
                  survey_id: str = None, level_counts: list = None) -> str:
    """Create a new ACA session, return session_id.

    Question budget defaults to p (the number of utility parameters under
    baseline coding). Both min and max default to p, so a survey runs exactly
    p questions unless explicitly overridden; adaptive convergence still
    applies when a smaller min/max is configured.
    """
    p = total_params(attrs)
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        'respondent_id': respondent_id,
        'survey_id': survey_id,
        'level_counts': level_counts,
        'X': np.empty((0, p)),
        'a': np.empty((1, 0)),
        'attrs': attrs,
        'ub': ub,
        'asked': [],
        'axes': [],
        'axis_index': 0,
        'last_active': time.time(),
        'round': 0,
        'est_history': [],
        'max_questions': max_questions if max_questions is not None else p,
        'min_questions': min_questions if min_questions is not None else p,
        'conv_threshold': convergence_threshold,
        'conv_count': consecutive_count,
        'state_stack': [],  # for undo support
    }
    return session_id


def get_session(session_id: str, respondent_id: str) -> dict:
    """Validate session exists, belongs to respondent, not expired."""
    sess = sessions.get(session_id)
    if not sess:
        raise ValueError("Session not found")
    if sess['respondent_id'] != respondent_id:
        raise ValueError("Session does not belong to this respondent")
    if time.time() - sess['last_active'] > SESSION_TTL:
        del sessions[session_id]
        raise ValueError("Session expired")
    sess['last_active'] = time.time()
    return sess


def cleanup_expired():
    """Remove expired sessions. Call periodically."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s['last_active'] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]


# ── Core Algorithm ──

def analytic_center(X, a, ub, attrs):
    """Analytic center: max Σ log(s_j), s.t. X@u=a, 0 ≤ u ≤ ub with margin s,
    and optional monotone ordering constraints on ordered attributes."""
    p = X.shape[1]
    u = cp.Variable(p)
    s = cp.Variable(p)
    constraints = [X @ u == a.flatten(), u <= ub - s, u >= s, s >= 1e-6]
    idx = 0
    for attr in attrs:
        if attr.get('ordered', False):
            k = len(attr['levels']) - 1
            for j in range(k - 1):
                constraints.append(u[idx + j] <= u[idx + j + 1])
        idx += len(attr['levels']) - 1
    obj = cp.Maximize(cp.sum(cp.log(s)))
    prob = cp.Problem(obj, constraints)
    solver = cp.OSQP if p > 30 else cp.CLARABEL
    prob.solve(solver=solver, verbose=False)
    if prob.status not in ['optimal', 'optimal_inaccurate'] or u.value is None:
        raise ValueError(f"Solver: {prob.status}")
    return u.value


def minimax_relax(X, a_flat, ub):
    """Minimax relaxation for inconsistent responses."""
    q, p = X.shape
    c = np.zeros(p + 1)
    c[-1] = 1.0
    A_ub = np.vstack([
        np.hstack([X, -np.ones((q, 1))]),
        np.hstack([-X, -np.ones((q, 1))]),
        np.hstack([np.eye(p), np.zeros((p, 1))]),
        np.hstack([-np.eye(p), np.zeros((p, 1))])
    ])
    b_ub = np.concatenate([a_flat, -a_flat, np.full(p, ub), np.zeros(p)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, ub)] * p + [(0, None)], method='highs')
    return (res.x[:p], res.x[p]) if res.success else (np.zeros(p), ub)


def _pinv(M, reg=1e-10):
    return pinv(M + reg * np.eye(M.shape[0]))


def principal_axes(X, u_center, ub):
    """Return all eigenvectors sorted by eigenvalue (smallest first = longest axis).

    Margins s are the distances from the center to the box bounds [0, ub],
    i.e. min(u, ub - u); the [0, ub] box keeps all u_j > 0.
    """
    p = len(u_center)
    s = np.maximum(np.minimum(u_center, ub - u_center), 1e-10)
    U2 = np.diag(1.0 / s ** 2)
    q = X.shape[0]
    P = X.T @ _pinv(X @ X.T) @ X if q < p else np.eye(p)
    M = U2 - P @ U2
    M = (M + M.T) / 2
    ev, evec = np.real(np.asarray(eig(M)[0])), np.real(np.asarray(eig(M)[1]))
    sorted_idx = np.argsort(ev)
    return [evec[:, i] for i in sorted_idx]


def aca_step(X, a, upper_bound, attrs):
    """Main ACA step: analytic center + principal axes."""
    X = np.atleast_2d(X)
    a_vec = np.atleast_2d(a).flatten()
    delta = 0.0
    try:
        u_center = analytic_center(X, a_vec, upper_bound, attrs)
    except Exception:
        u_relax, delta = minimax_relax(X, a_vec, upper_bound)
        a_relaxed = X @ u_relax
        try:
            u_center = analytic_center(X, a_relaxed, upper_bound, attrs)
        except Exception:
            u_center = u_relax
    axes = principal_axes(X, u_center, upper_bound)
    return {'est': u_center, 'axes': axes, 'delta': delta}


# ── Card Generation ──

def generate_card(attrs, lv):
    """Encode a card into a baseline-coded vector.

    Each attribute's baseline level (index 0) contributes nothing; a
    non-baseline level activates exactly one parameter (its gain over the
    baseline), so a card can never carry two levels of the same attribute.
    """
    p = total_params(attrs)
    card = np.zeros(p)
    idx = 0
    for i, attr in enumerate(attrs):
        if lv[i] > 0:
            card[idx + lv[i] - 1] = 1.0
        idx += len(attr['levels']) - 1
    return card


def _validate_card(attrs, card):
    """Defensive check: each attribute segment of a card has at most one
    active (non-baseline) level — never a card that is both 中画质 and 高画质."""
    idx = 0
    for attr in attrs:
        seg = card[idx:idx + len(attr['levels']) - 1]
        assert np.sum(seg) <= 1.0, f"mutual exclusion violated in '{attr['name']}'"
        idx += len(attr['levels']) - 1


def gen_card_pair(attrs, axis=None, exclude=None, level_counts=None, max_changes=3):
    """Generate an optimal card pair.

    - axis (post first round): pick extreme levels along the longest
      principal axis, but only for a subset of attributes so that at most
      `max_changes` attributes differ (cognitive-load constraint, paper p.10).
    - level_counts (first question): favour infrequently-shown levels.
    - exclude: avoid repeats of previously asked card pairs.
    At least 2 attributes must differ between the two cards.
    """
    n_a = len(attrs)
    offsets = []
    off = 0
    for attr in attrs:
        offsets.append(off)
        off += len(attr['levels']) - 1

    best, best_s = None, -1e20
    max_allowed = min(max_changes, n_a)
    for _ in range(300):
        # Pick which attributes differ (2..max_changes). Under an axis,
        # prefer attributes with the widest axis range so the pair stays
        # close to the longest axis while only changing a few attributes.
        if axis is not None and n_a > 1:
            ranges = []
            for i, attr in enumerate(attrs):
                k = len(attr['levels']) - 1
                ranges.append(float(np.ptp(axis[offsets[i]:offsets[i] + k])) if k > 0 else 0.0)
            rsum = sum(ranges)
            n_chg = np.random.randint(2, max_allowed + 1) if max_allowed >= 2 else n_a
            nz = sum(1 for r in ranges if r > 1e-12)
            # Some axis segments can be ~0 (params pinned near a bound); fall
            # back to uniform selection rather than erroring on a sparse p.
            prob = [r / rsum for r in ranges] if (rsum > 1e-12 and nz >= n_chg) else None
            chg = set(np.random.choice(n_a, n_chg, replace=False, p=prob))
        else:
            n_chg = np.random.randint(2, max_allowed + 1) if max_allowed >= 2 else n_a
            chg = set(np.random.choice(n_a, n_chg, replace=False))

        la, lb = [], []
        for i, attr in enumerate(attrs):
            k = len(attr['levels'])
            if i in chg:
                if axis is not None:
                    seg = axis[offsets[i]:offsets[i] + k - 1]
                    la.append(np.argmax(seg + np.random.randn(k - 1) * 0.5))
                    lb.append(np.argmin(seg + np.random.randn(k - 1) * 0.5))
                elif level_counts is not None:
                    counts = np.asarray(level_counts[i], dtype=float) if len(level_counts) > i else np.zeros(k)
                    p_ = 1.0 / (counts + 1.0)
                    p_ /= p_.sum()
                    la.append(np.random.choice(k, p=p_))
                    p2 = p_.copy()
                    p2[la[-1]] = 0
                    if p2.sum() > 0:
                        p2 /= p2.sum()
                        lb.append(np.random.choice(k, p=p2))
                    else:
                        lb.append(np.random.randint(k))
                else:
                    la.append(np.random.randint(k))
                    lb.append(np.random.randint(k))
            else:
                # Unchanged attributes show the same level on both cards.
                if axis is not None:
                    seg = axis[offsets[i]:offsets[i] + k - 1]
                    lv = np.argmax(seg + np.random.randn(k - 1) * 0.5)
                elif level_counts is not None:
                    counts = np.asarray(level_counts[i], dtype=float) if len(level_counts) > i else np.zeros(k)
                    p_ = 1.0 / (counts + 1.0)
                    p_ /= p_.sum()
                    lv = np.random.choice(k, p=p_)
                else:
                    lv = np.random.randint(k)
                la.append(lv)
                lb.append(lv)

        if sum(1 for i in range(n_a) if la[i] != lb[i]) < 2:
            continue
        card_a = generate_card(attrs, la)
        card_b = generate_card(attrs, lb)
        _validate_card(attrs, card_a)
        _validate_card(attrs, card_b)
        diff = card_a - card_b
        if exclude and any(np.linalg.norm(diff - ex) < 0.1 or np.linalg.norm(diff + ex) < 0.1 for ex in exclude):
            continue
        s = abs(diff @ axis) if axis is not None else np.random.rand()
        if s > best_s:
            best_s, best = s, (diff, la, lb)
    if best is None:
        la = [np.random.randint(len(a['levels'])) for a in attrs]
        lb = [np.random.randint(len(a['levels'])) for a in attrs]
        return generate_card(attrs, la) - generate_card(attrs, lb), la, lb
    return best


def generate_first_question(attrs, level_counts=None, max_changes=3):
    """First question for a respondent.

    The polyhedron is still the full box (no principal axis), so individual
    optimality is undefined; instead balance coverage across respondents by
    favouring levels that appeared infrequently in previous respondents'
    questions (paper p.10). Constrain changed attributes to [2, max_changes],
    then pick the pair with the highest low-frequency coverage.
    """
    n_a = len(attrs)
    weights = []
    for i, attr in enumerate(attrs):
        k = len(attr['levels'])
        counts = np.zeros(k)
        if level_counts is not None and i < len(level_counts):
            counts = np.asarray(level_counts[i], dtype=float)
        w = 1.0 / (counts + 1.0)
        w /= w.sum()
        weights.append(w)

    best, best_s = None, -1e20
    max_allowed = min(max_changes, n_a)
    for _ in range(300):
        n_chg = np.random.randint(2, max_allowed + 1) if max_allowed >= 2 else n_a
        chg = set(np.random.choice(n_a, n_chg, replace=False))
        la, lb = [], []
        for i, attr in enumerate(attrs):
            k = len(attr['levels'])
            if i in chg:
                la.append(np.random.choice(k, p=weights[i]))
                lb.append(np.random.choice(k, p=weights[i]))
                if lb[-1] == la[-1]:
                    p2 = weights[i].copy()
                    p2[la[-1]] = 0
                    if p2.sum() > 0:
                        p2 /= p2.sum()
                        lb[-1] = np.random.choice(k, p=p2)
                    else:
                        lb[-1] = np.random.randint(k)
            else:
                lv = np.random.choice(k, p=weights[i])
                la.append(lv)
                lb.append(lv)
        if sum(1 for i in range(n_a) if la[i] != lb[i]) < 2:
            continue
        # Coverage score: per attribute, credit the best (highest weight) level
        # shown on either card. Summing over attributes favours attribute-level
        # balance instead of double-counting one attribute's levels.
        score = sum(max(weights[i][la[i]], weights[i][lb[i]]) for i in range(n_a))
        if score > best_s:
            best_s, best = score, (la, lb)
    if best is None:
        la = [np.random.randint(len(a['levels'])) for a in attrs]
        lb = [np.random.randint(len(a['levels'])) for a in attrs]
        return generate_card(attrs, la) - generate_card(attrs, lb), la, lb
    la, lb = best
    return generate_card(attrs, la) - generate_card(attrs, lb), la, lb


def decode_card(attrs, lv):
    return {a['name']: a['levels'][l] for a, l in zip(attrs, lv)}


def format_est(est, attrs):
    """Format utility estimates as nested dict (baseline level fixed at 0).

    Each non-baseline level gets its gain-over-baseline part-worth; attribute
    importance is the range from the baseline (0) to the most preferred level.
    """
    utilities = {}
    importance_raw = {}
    idx = 0
    for attr in attrs:
        k = len(attr['levels'])
        seg = est[idx:idx + k - 1]
        utilities[attr['name']] = {attr['levels'][0]: 0.0}
        for j in range(1, k):
            utilities[attr['name']][attr['levels'][j]] = round(float(seg[j - 1]), 2)
        importance_raw[attr['name']] = float(np.max(seg)) if k - 1 > 0 else 0.0
        idx += k - 1

    total = sum(importance_raw.values())
    if total > 1e-10:
        importance = {n: round(v / total * 100, 1) for n, v in importance_raw.items()}
    else:
        importance = {n: 0.0 for n in importance_raw}

    return utilities, importance


# ── Session Operations ──

def start_aca_session(session_id: str):
    """Generate the first question for a session.

    The polyhedron is still the full box, so there is no principal axis to
    guide the first question. Instead, balance level coverage across
    respondents using low-frequency levels from previous respondents.
    """
    sess = sessions[session_id]
    attrs = sess['attrs']
    level_counts = sess.get('level_counts')

    diff, la, lb = generate_first_question(attrs, level_counts)
    sess['asked'].append(diff.copy())

    return {
        'round': 1,
        'card_a': decode_card(attrs, la),
        'card_b': decode_card(attrs, lb),
        'card_a_levels': [int(x) for x in la],
        'card_b_levels': [int(x) for x in lb],
    }


def record_answer(session_id: str, rating: float):
    """Record a rating and return next question + estimates."""
    sess = sessions[session_id]
    attrs = sess['attrs']
    p = total_params(attrs)

    # Save state snapshot for undo
    sess['state_stack'].append({
        'X': sess['X'].copy(),
        'a': sess['a'].copy(),
        'asked': [d.copy() for d in sess['asked']],
        'axes': [a.copy() for a in sess['axes']],
        'axis_index': sess['axis_index'],
        'round': sess['round'],
        'est_history': [e.copy() for e in sess['est_history']],
    })

    if rating == 0:
        sess['round'] += 1
        sess['axis_index'] = min(sess.get('axis_index', 0) + 1, len(sess['axes']) - 1) if sess['axes'] else 0

        if sess['axes']:
            axis = sess['axes'][sess['axis_index']]
        else:
            axis = None

        diff, la, lb = gen_card_pair(attrs, axis=axis, exclude=sess['asked'])
        sess['asked'].append(diff.copy())

        if sess['est_history']:
            est = sess['est_history'][-1]
            utilities, importance = format_est(est, attrs)
        else:
            utilities, importance = {}, {}

        return {
            'estimates': utilities,
            'importance': importance,
            'next_question': {
                'round': sess['round'] + 1,
                'card_a': decode_card(attrs, la),
                'card_b': decode_card(attrs, lb),
                'card_a_levels': [int(x) for x in la],
                'card_b_levels': [int(x) for x in lb],
            },
            'converged': False,
            'is_zero_rating_response': True,
            'delta': 0.0,
        }

    # Non-zero rating: add constraint and recompute
    last_diff = sess['asked'][-1] if sess['asked'] else np.zeros(p)

    sess['X'] = np.vstack([sess['X'], last_diff])
    sess['a'] = np.hstack([sess['a'], [[rating]]])
    sess['round'] += 1

    result = aca_step(sess['X'], sess['a'], sess['ub'], attrs)
    est = result['est']
    sess['axes'] = result['axes']
    sess['axis_index'] = 0
    sess['est_history'].append(est)

    # Check convergence
    converged = False
    min_q = sess.get('min_questions', 3)
    conv_threshold = sess.get('conv_threshold', 0.10)
    conv_count = sess.get('conv_count', 2)

    if len(sess['est_history']) >= min_q:
        n_conv = 0
        for i in range(len(sess['est_history']) - 1, 0, -1):
            prev = sess['est_history'][i - 1]
            curr = sess['est_history'][i]
            t = np.sum(np.abs(curr))
            if t > 0:
                change = np.sum(np.abs(curr - prev)) / t
                if change < conv_threshold:
                    n_conv += 1
                else:
                    break
        converged = n_conv >= conv_count

    # Max questions check
    max_q = sess.get('max_questions', p)
    if sess['round'] >= max_q:
        converged = True

    if converged:
        utilities, importance = format_est(est, attrs)
        return {
            'estimates': utilities,
            'importance': importance,
            'next_question': None,
            'converged': True,
            'is_zero_rating_response': False,
            'delta': float(result['delta']),
        }

    # Infeasible (relaxed) responses must not steer the next question: the
    # relaxed center is distorted by delta, so fall back to a random question
    # (paper: random fallback when the polyhedron is empty).
    if result['delta'] > 0:
        diff, la, lb = gen_card_pair(attrs, exclude=sess['asked'])
    else:
        axis = result['axes'][0]
        diff, la, lb = gen_card_pair(attrs, axis=axis, exclude=sess['asked'])
    sess['asked'].append(diff.copy())

    utilities, importance = format_est(est, attrs)

    return {
        'estimates': utilities,
        'importance': importance,
        'next_question': {
            'round': sess['round'] + 1,
            'card_a': decode_card(attrs, la),
            'card_b': decode_card(attrs, lb),
            'card_a_levels': [int(x) for x in la],
            'card_b_levels': [int(x) for x in lb],
        },
        'converged': False,
        'is_zero_rating_response': False,
        'delta': float(result['delta']),
    }


def undo_last_answer(session_id: str) -> dict:
    """Undo the last answer and return the restored state."""
    sess = sessions[session_id]
    if not sess['state_stack']:
        raise ValueError("No answers to undo")

    prev = sess['state_stack'].pop()
    sess['X'] = prev['X']
    sess['a'] = prev['a']
    sess['asked'] = prev['asked']
    sess['axes'] = prev['axes']
    sess['axis_index'] = prev['axis_index']
    sess['round'] = prev['round']
    sess['est_history'] = prev['est_history']

    # Re-generate the question for the restored round
    attrs = sess['attrs']
    if sess['axes']:
        axis = sess['axes'][sess['axis_index']]
    else:
        axis = None

    diff, la, lb = gen_card_pair(attrs, axis=axis, exclude=sess['asked'])
    sess['asked'].append(diff.copy())

    if sess['est_history']:
        est = sess['est_history'][-1]
        utilities, importance = format_est(est, attrs)
    else:
        utilities, importance = {}, {}

    return {
        'estimates': utilities,
        'importance': importance,
        'next_question': {
            'round': sess['round'] + 1,
            'card_a': decode_card(attrs, la),
            'card_b': decode_card(attrs, lb),
            'card_a_levels': [int(x) for x in la],
            'card_b_levels': [int(x) for x in lb],
        },
        'converged': False,
    }


def get_session_result(session_id: str):
    """Get final results for a session."""
    sess = sessions[session_id]
    attrs = sess['attrs']

    if sess['est_history']:
        est = sess['est_history'][-1]
    elif sess['X'].shape[0] > 0:
        result = aca_step(sess['X'], sess['a'], sess['ub'], attrs)
        est = result['est']
    else:
        est = np.zeros(total_params(attrs))

    utilities, importance = format_est(est, attrs)

    return {
        'utilities': utilities,
        'importance': importance,
        'total_rounds': sess['round'],
        'converged': len(sess['est_history']) >= sess.get('min_questions', 3),
    }


# ── Validation ──

def compute_choice_hit_rate(utilities: dict, tasks: list) -> dict:
    """
    Choice validation: predict the option with highest total utility,
    compare with actual respondent choice.
    Also computes Log-Likelihood using multinomial logit probabilities.

    utilities: {"brand": {"A": 15.2, "B": -15.2}, ...}
    tasks: [{"task_index": 0, "options": [{"id": "opt1", "levels": {"brand":"A", "price":"cheap"}}, ...],
              "respondent_choice": "opt1"}, ...]
    """
    import math

    results = []
    hits = 0
    total = 0
    log_likelihood = 0.0

    for task in tasks:
        options = task.get("options", [])
        actual = task.get("respondent_choice")

        best_id = None
        best_util = -float('inf')
        utils = []

        for opt in options:
            util = 0.0
            for attr_name, level_name in opt.get("levels", {}).items():
                attr_utils = utilities.get(attr_name, {})
                util += attr_utils.get(level_name, 0.0)
            opt["_computed_utility"] = round(util, 4)
            utils.append(util)
            if util > best_util:
                best_util = util
                best_id = opt["id"]

        # Multinomial logit probabilities (numerically stable)
        max_util = max(utils) if utils else 0.0
        exp_sum = sum(math.exp(u - max_util) for u in utils)
        probs = {}
        actual_prob = 0.0
        for opt, u in zip(options, utils):
            p = math.exp(u - max_util) / exp_sum if exp_sum > 0 else 1.0 / len(options)
            probs[opt["id"]] = round(p, 6)
            if opt["id"] == actual:
                actual_prob = p

        # Log-likelihood: log(P(chosen option))
        if actual_prob > 0:
            log_likelihood += math.log(actual_prob)

        hit = (best_id == actual)
        if hit:
            hits += 1
        total += 1

        results.append({
            "task_index": task.get("task_index", 0),
            "options": options,
            "predicted": best_id,
            "actual": actual,
            "hit": hit,
            "probabilities": probs,
            "actual_probability": round(actual_prob, 6),
        })

    hit_rate = hits / total if total > 0 else 0.0
    avg_ll = log_likelihood / total if total > 0 else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "log_likelihood": round(log_likelihood, 4),
        "avg_log_likelihood": round(avg_ll, 4),
        "hits": hits,
        "total": total,
        "per_task": results,
    }


def validate_mbc(utilities: dict, real_products: list,
                 ideal_config: dict, final_choice: str) -> dict:
    """
    MBC validation: per-attribute accuracy.
    For each attribute, check if ACA's predicted best level matches
    the level in the product the user actually chose.

    utilities: {"brand": {"A": 15.2, "B": -15.2}, ...}
    real_products: [{"id": "p1", "name": "Laptop A", "levels": {"brand":"A","price":"cheap"}}, ...]
    ideal_config: {"brand": "A", "price": "cheap", ...}  (computed from ACA)
    final_choice: "p3"
    """
    # Find the chosen product
    chosen_product = None
    for prod in real_products:
        if prod["id"] == final_choice:
            chosen_product = prod
            break

    if not chosen_product:
        return {"error": "Chosen product not found", "accuracy": 0.0}

    # Per-attribute: predicted best level vs chosen level
    per_attr = []
    matches = 0
    total = 0

    for attr_name, attr_utils in utilities.items():
        # ACA predicted best level = highest utility
        best_level = max(attr_utils, key=attr_utils.get) if attr_utils else None
        # Actual level in chosen product
        actual_level = chosen_product.get("levels", {}).get(attr_name)
        match = (best_level == actual_level)
        if match:
            matches += 1
        total += 1
        per_attr.append({
            "attribute": attr_name,
            "predicted_best": best_level,
            "actual_chosen": actual_level,
            "match": match,
        })

    accuracy = matches / total if total > 0 else 0.0

    # Also compute product utilities and predicted best product
    product_utils = {}
    for prod in real_products:
        util = 0.0
        for attr_name, level_name in prod.get("levels", {}).items():
            util += utilities.get(attr_name, {}).get(level_name, 0.0)
        product_utils[prod["id"]] = round(util, 4)

    predicted_id = max(product_utils, key=product_utils.get) if product_utils else None
    product_hit = (predicted_id == final_choice)

    # Ideal utility
    ideal_utility = sum(
        utilities.get(a, {}).get(l, 0.0) for a, l in ideal_config.items()
    )
    chosen_utility = product_utils.get(final_choice, 0.0)

    return {
        "accuracy": round(accuracy, 4),
        "matches": matches,
        "total_attrs": total,
        "per_attribute": per_attr,
        "product_hit": product_hit,
        "predicted_product": predicted_id,
        "actual_product": final_choice,
        "utility_gap": round(ideal_utility - chosen_utility, 4),
    }


def generate_holdout_tasks(attrs: list, n_tasks: int = 3, n_options: int = 3,
                           seed: int = None) -> list:
    """Generate random holdout choice tasks for validation (no duplicate options)."""
    rng = np.random.RandomState(seed)
    tasks = []

    for t in range(n_tasks):
        options = []
        seen = set()
        max_attempts = n_options * 50
        attempts = 0
        while len(options) < n_options and attempts < max_attempts:
            attempts += 1
            levels = {}
            for attr in attrs:
                idx = rng.randint(len(attr['levels']))
                levels[attr['name']] = attr['levels'][idx]
            key = tuple(sorted(levels.items()))
            if key in seen:
                continue
            seen.add(key)
            options.append({
                "id": f"opt{len(options) + 1}",
                "levels": levels,
            })
        tasks.append({
            "task_index": t,
            "options": options,
        })

    return tasks
