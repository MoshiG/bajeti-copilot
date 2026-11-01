import itertools
import random

import pytest

from bajeti.optimize import MAX_ITEMS, Intervention, rank_interventions


def _brute_force(items, ceiling):
    best = 0.0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            if sum(i.cost for i in combo) <= ceiling:
                best = max(best, sum(i.benefit for i in combo))
    return best


def test_beats_the_greedy_trap():
    # Best value-for-money first (A) leaves no room for B and C, which together are worth more.
    items = [Intervention("A", 6, 7), Intervention("B", 5, 5), Intervention("C", 5, 5)]
    plan = rank_interventions(items, 10)
    assert {i.name for i in plan.selected} == {"B", "C"}
    assert plan.total_benefit == 10 and plan.remaining == 0


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_on_random_instances(seed):
    rng = random.Random(seed)
    items = [Intervention(f"i{k}", rng.randint(1, 40), rng.randint(0, 50)) for k in range(12)]
    ceiling = rng.randint(20, 150)
    plan = rank_interventions(items, ceiling)
    assert plan.total_cost <= ceiling
    assert plan.total_benefit == pytest.approx(_brute_force(items, ceiling))


def test_mandatory_items_are_funded_first():
    items = [Intervention("Salaries", 8, 1, mandatory=True), Intervention("X", 5, 100),
             Intervention("Y", 2, 10)]
    plan = rank_interventions(items, 10)
    assert [i.name for i in plan.selected] == ["Salaries", "Y"]
    with pytest.raises(ValueError, match="mandatory"):
        rank_interventions([Intervention("Big", 20, 1, mandatory=True)], 10)


def test_explanations_show_funding_gaps():
    plan = rank_interventions([Intervention("Roads", 7, 9), Intervention("Clinics", 6, 8)], 10)
    md = plan.to_markdown()
    assert "| Roads | 7 | 9 |" in md and "funded" in md
    assert "Clinics" in md and "needs 3 more" in md      # 6 cost vs 3 remaining
    assert md.startswith("Budget ceiling: 10; allocated: 7 (70.0%); remaining: 3")


def test_zero_cost_and_zero_benefit_items():
    plan = rank_interventions([Intervention("Free", 0, 5), Intervention("Useless", 1, 0),
                               Intervention("Good", 3, 4)], 3)
    assert {i.name for i in plan.selected} == {"Free", "Good"}
    assert any("zero-benefit" in n for n in plan.notes)
    assert "adds no benefit" not in plan.to_markdown() or "Useless" in plan.to_markdown()


@pytest.mark.parametrize("items, ceiling, msg", [
    ([Intervention("A", 1, 1), Intervention("A", 2, 2)], 5, "unique"),
    ([Intervention("A", 1, 1)], -1, "non-negative"),
    ([Intervention(f"i{k}", 1, 1) for k in range(MAX_ITEMS + 1)], 5, "at most"),
])
def test_input_validation(items, ceiling, msg):
    with pytest.raises(ValueError, match=msg):
        rank_interventions(items, ceiling)


def test_invalid_intervention_rejected():
    with pytest.raises(ValueError):
        Intervention("A", -1, 2)
    with pytest.raises(ValueError):
        Intervention("", 1, 2)


def test_forty_items_solve_quickly():
    import time

    rng = random.Random(7)
    items = [Intervention(f"i{k}", rng.randint(10, 500), rng.randint(1, 900)) for k in range(40)]
    started = time.perf_counter()
    plan = rank_interventions(items, 3000)
    assert time.perf_counter() - started < 2.0
    assert plan.total_cost <= 3000
