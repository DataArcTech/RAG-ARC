from core.knowledge_graph.schema import load_schema_from_yaml


def test_fin_kg_schema_file_covers_hk_insurance_brochure_predicates() -> None:
    schema = load_schema_from_yaml("./fin_kg_schema.yml")
    domain = schema.for_domain("finance_insurance")

    assert domain.normalize_predicate("Policy currency") == "HAS_CURRENCY"
    assert domain.normalize_predicate("Premium payment term") == "HAS_PREMIUM_PAYMENT_TERM"
    assert domain.normalize_predicate("Prevailing interest rate") == "HAS_INTEREST_RATE"
    assert domain.normalize_predicate("Premium refund") == "HAS_REFUND"
    assert domain.normalize_predicate("Cash value") == "HAS_CASH_VALUE"
    assert domain.normalize_predicate("Surrender benefit") == "HAS_BENEFIT"
    assert domain.normalize_predicate("Terminal bonus") == "HAS_BENEFIT"
    assert domain.normalize_predicate("Reversionary bonus") == "HAS_BENEFIT"
    assert domain.normalize_predicate("Policy loan") == "HAS_FEATURE"
    assert domain.normalize_predicate("Designated withdrawal") == "HAS_FEATURE"

