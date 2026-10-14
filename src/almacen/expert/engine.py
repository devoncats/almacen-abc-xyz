"""Forward-chaining inference engine with an explanation trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from almacen.expert.facts import ProductFact
from almacen.expert.knowledge import Rule, matches


class InferenceError(RuntimeError):
    """Raised when inference cannot produce a policy."""


@dataclass
class Policy:
    review_type: Literal["continuous_sQ", "periodic_RS", "on_demand"]
    service_level: float
    review_weeks: int
    lot_weeks: float
    safety_factor_mult: float = 1.0
    max_cover_weeks: float | None = None
    provisional: bool = False
    manual_validation: bool = False
    discontinue_candidate: bool = False


@dataclass
class WorkingMemory:
    fact: ProductFact
    policy: Policy | None = None
    flags: set[str] = field(default_factory=set)


@dataclass
class Firing:
    rule_id: str
    description: str
    changes: dict[str, Any]


@dataclass
class Inference:
    item_nbr: int
    policy: Policy
    trace: list[Firing]
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_nbr": self.item_nbr,
            "policy": asdict(self.policy),
            "flags": self.flags,
            "fired_rules": [asdict(f) for f in self.trace],
        }


def _resolve(value: Any, fact: ProductFact) -> Any:
    """Values written as "$field" reference the product fact."""
    if isinstance(value, str) and value.startswith("$"):
        return getattr(fact, value[1:])
    return value


class InferenceEngine:
    def __init__(self, rules: list[Rule], max_cycles: int = 50) -> None:
        # conflict resolution: higher salience first, then file order
        self.rules = sorted(rules, key=lambda r: (-r.salience, r.order))
        self.max_cycles = max_cycles

    def _fire(self, rule: Rule, wm: WorkingMemory) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        assigned = {
            k: _resolve(v, wm.fact) for k, v in rule.then.get("set", {}).items()
        }

        if rule.stage == "segment":
            wm.policy = Policy(**assigned)
            changes.update(assigned)
        else:
            for key, value in assigned.items():
                setattr(wm.policy, key, value)
                changes[key] = value

        for key, factor in rule.then.get("scale", {}).items():
            new = getattr(wm.policy, key) * float(_resolve(factor, wm.fact))
            setattr(wm.policy, key, new)
            changes[key] = round(new, 4)

        for flag in rule.then.get("flag", []):
            wm.flags.add(flag)
            if hasattr(wm.policy, flag) and isinstance(getattr(wm.policy, flag), bool):
                setattr(wm.policy, flag, True)
            changes.setdefault("flags", []).append(flag)
        return changes

    def infer(self, fact: ProductFact) -> Inference:
        """Forward chaining: match -> conflict resolution (salience desc) -> fire.

        Refraction: each rule fires at most once per product. Stops at fixed point.
        Raises if no segment-stage rule fired.
        """
        wm = WorkingMemory(fact=fact)
        fired: set[str] = set()
        trace: list[Firing] = []

        for _ in range(self.max_cycles):
            agenda = [r for r in self.rules if r.id not in fired and matches(r, wm)]
            if not agenda:
                break
            rule = agenda[0]
            changes = self._fire(rule, wm)
            fired.add(rule.id)
            trace.append(
                Firing(rule_id=rule.id, description=rule.description, changes=changes)
            )
        else:
            raise InferenceError(
                f"item {fact.item_nbr}: no fixed point after {self.max_cycles} cycles"
            )

        if wm.policy is None:
            raise InferenceError(
                f"item {fact.item_nbr}: no segment rule matched segment '{fact.segment}'"
            )
        return Inference(
            item_nbr=fact.item_nbr,
            policy=wm.policy,
            trace=trace,
            flags=sorted(wm.flags),
        )

    def infer_all(self, facts: list[ProductFact]) -> list[Inference]:
        return [self.infer(f) for f in facts]
