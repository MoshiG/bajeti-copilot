"""Choose interventions under a budget ceiling: exact 0/1 knapsack by branch and bound.

Given interventions with a cost and a benefit score, pick the set that maximises total benefit
without exceeding the ceiling. Mandatory items are funded first. The solver is exact (it proves
optimality), dependency-free, and fast for the tens of items a planning question involves.

The result also explains itself: what was funded, what was not and why, how much budget is
left, and how much more money the best excluded item would need. That turns a number into
something a planner can argue with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_ITEMS = 60   # branch and bound is exact; beyond this, ask the user to pre-filter


@dataclass
class Intervention:
    name: str
    cost: float
    benefit: float
    mandatory: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("every intervention needs a name")
        if self.cost < 0 or self.benefit < 0:
            raise ValueError(f"{self.name}: cost and benefit must be non-negative")


@dataclass
class Plan:
    selected: list[Intervention]
    excluded: list[Intervention]
    budget_ceiling: float
    total_cost: float
    total_benefit: float
    notes: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return self.budget_ceiling - self.total_cost

    def to_markdown(self) -> str:
        def row(i: Intervention, why: str = "") -> str:
            ratio = f"{i.benefit / i.cost:,.3g}" if i.cost else "∞"
            return f"| {i.name} | {i.cost:,.0f} | {i.benefit:,.4g} | {ratio} | {why} |"

        used = self.total_cost / self.budget_ceiling * 100 if self.budget_ceiling else 0
        lines = [
            f"Budget ceiling: {self.budget_ceiling:,.0f}; allocated: {self.total_cost:,.0f} "
            f"({used:.1f}%); remaining: {self.remaining:,.0f}; "
            f"total benefit: {self.total_benefit:,.4g}.",
            "",
            "| Intervention | Cost | Benefit | Benefit per unit cost | Note |",
            "|---|---|---|---|---|",
        ]
        lines += [row(i, "mandatory" if i.mandatory else "funded") for i in self.selected]
        for i in self.excluded:
            gap = i.cost - self.remaining
            # In an optimal plan, an excluded item that fits the leftover budget must add
            # nothing; otherwise the solver would have funded it.
            lines.append(row(i, f"not funded; needs {gap:,.0f} more" if gap > 0
                             else "not funded; adds no benefit"))
        if self.notes:
            lines += [""] + [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _branch_and_bound(items: list[Intervention], capacity: float) -> list[int]:
    """Indices (into `items`) of the optimal subset. Items are pre-sorted by benefit/cost."""
    n = len(items)
    best_value, best_set = 0.0, []

    def bound(i: int, cap: float, value: float) -> float:
        # Fractional (LP) relaxation: an upper bound on what the remaining items can add.
        for item in items[i:]:
            if item.cost <= cap:
                cap -= item.cost
                value += item.benefit
            else:
                return value + (item.benefit * cap / item.cost if item.cost else item.benefit)
        return value

    def search(i: int, cap: float, value: float, chosen: list[int]) -> None:
        nonlocal best_value, best_set
        if value > best_value:
            best_value, best_set = value, chosen[:]
        if i == n or bound(i, cap, value) <= best_value:
            return
        item = items[i]
        if item.cost <= cap:
            chosen.append(i)
            search(i + 1, cap - item.cost, value + item.benefit, chosen)
            chosen.pop()
        search(i + 1, cap, value, chosen)

    search(0, capacity, 0.0, [])
    return best_set


def rank_interventions(interventions: list[Intervention], budget_ceiling: float) -> Plan:
    if budget_ceiling < 0:
        raise ValueError("budget_ceiling must be non-negative")
    if len(interventions) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} interventions; pre-filter the list first")
    names = [i.name for i in interventions]
    if len(names) != len(set(names)):
        raise ValueError("intervention names must be unique")

    mandatory = [i for i in interventions if i.mandatory]
    mandatory_cost = sum(i.cost for i in mandatory)
    if mandatory_cost > budget_ceiling:
        raise ValueError(f"mandatory items cost {mandatory_cost:,.0f}, more than the ceiling "
                         f"{budget_ceiling:,.0f}")

    optional = sorted((i for i in interventions if not i.mandatory),
                      key=lambda i: (-(i.benefit / i.cost) if i.cost else float("-inf"), i.name))
    picked = [optional[k] for k in _branch_and_bound(optional, budget_ceiling - mandatory_cost)]

    selected = mandatory + sorted(picked, key=lambda i: -i.benefit)
    chosen = {i.name for i in selected}
    excluded = [i for i in optional if i.name not in chosen]
    notes = []
    zero = [i.name for i in excluded if i.benefit == 0]
    if zero:
        notes.append(f"zero-benefit items never get funded: {', '.join(zero)}")
    return Plan(selected, excluded, budget_ceiling, sum(i.cost for i in selected),
                sum(i.benefit for i in selected), notes)
