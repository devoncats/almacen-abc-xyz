import pytest
import yaml
from conftest import make_fact

from almacen.expert.engine import InferenceEngine, InferenceError
from almacen.expert.knowledge import RuleError, load_rules


@pytest.fixture
def engine(rules_path):
    return InferenceEngine(load_rules(rules_path))


@pytest.mark.parametrize(
    "segment,review",
    [
        ("AX", "continuous_sQ"),
        ("AY", "continuous_sQ"),
        ("AZ", "periodic_RS"),
        ("BX", "periodic_RS"),
        ("BY", "periodic_RS"),
        ("BZ", "periodic_RS"),
        ("CX", "periodic_RS"),
        ("CY", "periodic_RS"),
        ("CZ", "on_demand"),
    ],
)
def test_every_segment_gets_a_policy(engine, segment, review):
    inf = engine.infer(make_fact(abc=segment[0], xyz=segment[1], segment=segment))
    assert inf.trace[0].rule_id == f"R-{segment}"
    assert inf.policy.review_type == review


def test_az_requires_manual_validation(engine):
    inf = engine.infer(make_fact(abc="A", xyz="Z", segment="AZ"))
    assert inf.policy.manual_validation
    assert inf.policy.safety_factor_mult == pytest.approx(1.5)


def test_established_cz_is_discontinuation_candidate(engine):
    inf = engine.infer(make_fact(abc="C", xyz="Z", segment="CZ"))
    assert [f.rule_id for f in inf.trace] == ["R-CZ", "R-DISC"]
    assert inf.policy.discontinue_candidate


def test_new_cz_item_is_provisional_not_discontinued(engine):
    # chaining: R-NEW adds the 'provisional' flag, which blocks R-DISC
    inf = engine.infer(make_fact(abc="C", xyz="Z", segment="CZ", age_weeks=5))
    assert [f.rule_id for f in inf.trace] == ["R-CZ", "R-NEW"]
    assert inf.policy.provisional and not inf.policy.discontinue_candidate
    assert inf.policy.review_type == "periodic_RS"


def test_perishable_caps_cover_and_shortens_review(engine):
    inf = engine.infer(
        make_fact(abc="C", xyz="X", segment="CX", perishable=True, shelf_life_weeks=3)
    )
    assert [f.rule_id for f in inf.trace] == ["R-CX", "R-PER", "R-PER-REVIEW"]
    assert inf.policy.max_cover_weeks == 3
    assert inf.policy.review_weeks == 1


def test_weighted_item_scales_safety(engine):
    inf = engine.infer(make_fact(is_weighted=True))
    assert inf.policy.safety_factor_mult == pytest.approx(1.1)
    assert "weighted_item" in inf.flags


def test_no_matching_segment_rule_raises(rules_path):
    rules = [r for r in load_rules(rules_path) if r.id != "R-AX"]
    with pytest.raises(InferenceError):
        InferenceEngine(rules).infer(make_fact())


SEG_RULE = {
    "id": "R1",
    "stage": "segment",
    "salience": 1,
    "when": [{"segment": "AX"}],
    "then": {
        "set": {
            "review_type": "periodic_RS",
            "service_level": 0.9,
            "review_weeks": 1,
            "lot_weeks": 0,
        }
    },
}


@pytest.mark.parametrize(
    "bad",
    [
        [SEG_RULE, SEG_RULE],  # duplicate id
        [{**SEG_RULE, "when": [{"colour": "red"}]}],  # unknown field
        [{**SEG_RULE, "when": [{"cv": {"approx": 1}}]}],  # unknown operator
        [
            {**SEG_RULE, "then": {"set": {"review_type": "periodic_RS"}}}
        ],  # incomplete segment policy
        [{**SEG_RULE, "stage": "modifier"}],  # no segment rules
        [{**SEG_RULE, "then": {**SEG_RULE["then"], "delete": ["x"]}}],  # unknown action
    ],
)
def test_load_rules_rejects_invalid(tmp_path, bad):
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump({"rules": bad}))
    with pytest.raises(RuleError):
        load_rules(path)
