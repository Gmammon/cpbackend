"""Unit tests for the polyhedral ACA engine (baseline-coded version).

Run:  python test_aca_engine.py   (from the backend/ directory)
"""

import sys
import traceback

import numpy as np

from aca_engine import (
    total_params, generate_card, _validate_card, gen_card_pair,
    generate_first_question, aca_step, start_session, start_aca_session,
    record_answer, get_session_result, sessions,
)

attrs3 = [
    {'name': '品牌', 'levels': ['低', '中', '高'], 'ordered': True},
    {'name': '价格', 'levels': ['贵', '便宜'], 'ordered': False},
    {'name': '服务', 'levels': ['差', '一般', '好'], 'ordered': True},
]


def test_total_params():
    # p = sum(levels - 1); baselines are not parameters
    assert total_params(attrs3) == (3 - 1) + (2 - 1) + (3 - 1) == 5


def test_baseline_coding():
    attrs = [{'name': 'a', 'levels': ['L', 'M', 'H']}, {'name': 'b', 'levels': ['L', 'H']}]
    assert total_params(attrs) == 3
    # a=L (baseline -> nothing), b=H -> param index 2
    assert generate_card(attrs, [0, 1]).tolist() == [0.0, 0.0, 1.0]
    # a=H -> param index 1, b=L (baseline -> nothing)
    assert generate_card(attrs, [2, 0]).tolist() == [0.0, 1.0, 0.0]
    # fully-baseline card activates nothing
    assert generate_card(attrs, [0, 0]).tolist() == [0.0, 0.0, 0.0]


def test_card_pair_at_most_3_changes_and_exclusive():
    np.random.seed(7)
    attrs = [{'name': f'a{i}', 'levels': ['L', 'M', 'H']} for i in range(6)]
    for _ in range(50):
        axis = np.random.randn(total_params(attrs))
        _, la, lb = gen_card_pair(attrs, axis=axis)
        n_changed = sum(1 for i in range(len(attrs)) if la[i] != lb[i])
        assert 2 <= n_changed <= 3, f"changed={n_changed}"
        # no card may carry two levels of the same attribute
        _validate_card(attrs, generate_card(attrs, la))
        _validate_card(attrs, generate_card(attrs, lb))


def test_first_question_prefers_low_frequency():
    np.random.seed(3)
    attrs = [{'name': 'a', 'levels': ['L', 'M', 'H']}, {'name': 'b', 'levels': ['L', 'H']}]
    level_counts = [[100, 100, 0], [0, 50]]  # attr a level 'H' never shown before
    hits = 0
    for _ in range(20):
        _, la, lb = generate_first_question(attrs, level_counts)
        n_changed = sum(1 for i in range(2) if la[i] != lb[i])
        assert 2 <= n_changed <= 3
        if la[0] == 2 or lb[0] == 2:
            hits += 1
    assert hits >= 18, f"'H' only shown {hits}/20 times"


def test_session_budget_min_p_max_2p():
    np.random.seed(11)
    # True gains-over-baseline; small enough that |diff @ u| never exceeds 100,
    # so the constraint set stays feasible and the analytic center is used.
    u_true = np.array([20.0, 30.0, 15.0, 10.0, 25.0])
    sid = start_session('resp-1', attrs3, ub=100.0)
    q = start_aca_session(sid)
    assert q['round'] == 1

    rounds = 0
    converged = False
    while rounds < 20:
        diff = sessions[sid]['asked'][-1]
        score = float(diff @ u_true)
        rating = max(-100.0, min(100.0, score))
        res = record_answer(sid, rating)
        rounds += 1
        if res['converged']:
            converged = True
            break

    assert converged, "survey never converged within 20 rounds"
    # early_stop is OFF by default -> the survey always runs the full 2p = 10.
    assert rounds == 10, f"expected full 2p = 10 rounds, got {rounds}"

    result = get_session_result(sid)
    utils = result['utilities']
    # baselines fixed at 0
    assert utils['品牌']['低'] == 0.0
    assert utils['价格']['贵'] == 0.0
    assert utils['服务']['差'] == 0.0
    # all part-worths within [0, 100]
    for attr_u in utils.values():
        for v in attr_u.values():
            assert 0.0 <= v <= 100.0, f"out of range: {attr_u}"
    # ordering constraints hold for ordered attributes
    assert utils['品牌']['中'] <= utils['品牌']['高']
    assert utils['服务']['一般'] <= utils['服务']['好']


def test_early_stop_enabled_stops_between_p_and_2p():
    np.random.seed(13)
    u_true = np.array([20.0, 30.0, 15.0, 10.0, 25.0])
    sid = start_session('resp-es', attrs3, ub=100.0, early_stop=True)
    q = start_aca_session(sid)
    rounds = 0
    converged = False
    while rounds < 20:
        diff = sessions[sid]['asked'][-1]
        score = float(diff @ u_true)
        rating = max(-100.0, min(100.0, score))
        res = record_answer(sid, rating)
        rounds += 1
        if res['converged']:
            converged = True
            break
    assert converged
    # early_stop only fires after min=p=5 and before max=2p=10.
    assert 5 <= rounds <= 10, f"expected p..2p = 5..10, got {rounds}"


def test_inconsistent_responses_relax():
    attrs = [{'name': 'a', 'levels': ['L', 'H']}, {'name': 'b', 'levels': ['L', 'H']}]
    # First row says param0 = 50, its opposite says -param0 = 50 -> infeasible.
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    a = np.array([[50.0], [50.0]])
    res = aca_step(X, a, 100.0, attrs)
    assert res['delta'] > 0, "expected relaxation on contradictory constraints"
    assert np.all(res['est'] >= 0) and np.all(res['est'] <= 100)


def test_record_answer_infeasible_returns_random_next():
    np.random.seed(5)
    attrs = [
        {'name': 'a', 'levels': ['L', 'H']},
        {'name': 'b', 'levels': ['L', 'H']},
        {'name': 'c', 'levels': ['L', 'H']},
        {'name': 'd', 'levels': ['L', 'H']},
    ]
    sid = start_session('resp-2', attrs)  # p = 4 -> min = max = 4
    start_aca_session(sid)
    sess = sessions[sid]
    # Contradictory constraints: param0 = 50 and -param0 = 50. The engine
    # grows 'a' as a flat 1-row list (flattened in aca_step), so represent the
    # two ratings that way.
    sess['X'] = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    sess['a'] = np.array([[50.0, 50.0]])
    sess['round'] = 2
    res = record_answer(sid, 50.0)
    assert res['delta'] > 0, "infeasible answers must trigger relaxation"
    assert res['next_question'] is not None, "should still get a (random) next question"


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    _run_all()
