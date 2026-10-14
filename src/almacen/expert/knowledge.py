"""Knowledge base: production rules loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from almacen.expert.facts import ProductFact

if TYPE_CHECKING:
    from almacen.expert.engine import WorkingMemory

Condition = dict[
    str, Any
]  # e.g. {"segment": "AX"} | {"perishable": True} | {"age_weeks": {"lt": 13}}

OPERATORS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a is not None and a < b,
    "le": lambda a, b: a is not None and a <= b,
    "gt": lambda a, b: a is not None and a > b,
    "ge": lambda a, b: a is not None and a >= b,
    "in": lambda a, b: a in b,
    "contains": lambda a, b: b in a,
    "not_contains": lambda a, b: b not in a,
}
FACT_FIELDS = {f.name for f in fields(ProductFact)}
POLICY_FIELDS = {
    "review_type",
    "service_level",
    "review_weeks",
    "lot_weeks",
    "safety_factor_mult",
    "max_cover_weeks",
    "provisional",
    "manual_validation",
    "discontinue_candidate",
}
REQUIRED_POLICY_FIELDS = {"review_type", "service_level", "review_weeks", "lot_weeks"}
REVIEW_TYPES = {"continuous_sQ", "periodic_RS", "on_demand"}
ACTIONS = {"set", "scale", "flag"}


class RuleError(ValueError):
    """Raised when the knowledge base file is invalid."""


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    salience: int
    stage: Literal["segment", "modifier"]
    when: list[Condition]  # AND over fact fields, `flags` and `policy.<field>`
    then: dict[str, Any]  # {"set": {...}, "scale": {...}, "flag": [...]}
    order: int = 0  # position in the file, used as a stable tie-breaker


def _check_field(rule_id: str, name: str) -> None:
    if name in FACT_FIELDS or name == "flags":
        return
    if name.startswith("policy.") and name.removeprefix("policy.") in POLICY_FIELDS:
        return
    raise RuleError(f"{rule_id}: unknown condition field '{name}'")


def _check_conditions(rule_id: str, when: Any) -> list[Condition]:
    if not isinstance(when, list) or not when:
        raise RuleError(f"{rule_id}: 'when' must be a non-empty list")
    for cond in when:
        if not isinstance(cond, dict) or not cond:
            raise RuleError(f"{rule_id}: each condition must be a non-empty mapping")
        for name, test in cond.items():
            _check_field(rule_id, name)
            if isinstance(test, dict):
                bad = set(test) - set(OPERATORS)
                if bad:
                    raise RuleError(f"{rule_id}: unknown operators {sorted(bad)}")
    return when


def _check_actions(rule_id: str, stage: str, then: Any) -> dict[str, Any]:
    if not isinstance(then, dict) or not then:
        raise RuleError(f"{rule_id}: 'then' must be a non-empty mapping")
    bad = set(then) - ACTIONS
    if bad:
        raise RuleError(f"{rule_id}: unknown actions {sorted(bad)}")
    for action in ("set", "scale"):
        unknown = set(then.get(action, {})) - POLICY_FIELDS
        if unknown:
            raise RuleError(
                f"{rule_id}: '{action}' targets unknown policy fields {sorted(unknown)}"
            )
    if stage == "segment":
        missing = REQUIRED_POLICY_FIELDS - set(then.get("set", {}))
        if missing:
            raise RuleError(f"{rule_id}: segment rules must set {sorted(missing)}")
    review = then.get("set", {}).get("review_type")
    if review is not None and review not in REVIEW_TYPES:
        raise RuleError(f"{rule_id}: review_type must be one of {sorted(REVIEW_TYPES)}")
    if not isinstance(then.get("flag", []), list):
        raise RuleError(f"{rule_id}: 'flag' must be a list")
    return then


def load_rules(path: Path) -> list[Rule]:
    """Validate unique ids, operators and known fields."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuleError(f"rules file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RuleError(f"invalid YAML in {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("rules")
    if not isinstance(raw, list) or not raw:
        raise RuleError(f"{path} must contain a non-empty list of rules")

    rules: list[Rule] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "id" not in item:
            raise RuleError(f"rule #{i + 1} must be a mapping with an 'id'")
        rid = str(item["id"])
        if rid in seen:
            raise RuleError(f"duplicate rule id '{rid}'")
        seen.add(rid)
        stage = item.get("stage")
        if stage not in ("segment", "modifier"):
            raise RuleError(f"{rid}: stage must be 'segment' or 'modifier'")
        rules.append(
            Rule(
                id=rid,
                description=str(item.get("description", "")),
                salience=int(item.get("salience", 0)),
                stage=stage,
                when=_check_conditions(rid, item.get("when")),
                then=_check_actions(rid, stage, item.get("then")),
                order=i,
            )
        )
    if not any(r.stage == "segment" for r in rules):
        raise RuleError("knowledge base has no segment rules")
    return rules


def _lookup(wm: WorkingMemory, name: str) -> Any:
    if name == "flags":
        return wm.flags
    if name.startswith("policy."):
        return (
            None
            if wm.policy is None
            else getattr(wm.policy, name.removeprefix("policy."))
        )
    return getattr(wm.fact, name)


def matches(rule: Rule, wm: WorkingMemory) -> bool:
    if rule.stage == "modifier" and wm.policy is None:
        return False
    for cond in rule.when:
        for name, test in cond.items():
            value = _lookup(wm, name)
            checks = test.items() if isinstance(test, dict) else [("eq", test)]
            if not all(OPERATORS[op](value, expected) for op, expected in checks):
                return False
    return True
