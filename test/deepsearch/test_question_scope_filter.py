from core.deepsearch.report.composer import _filter_evidences_by_question_scope


def test_filter_evidences_by_question_scope_prefers_discriminative_terms():
    question = "Compare AlphaOmegaPlan and BetaGammaPlan payment terms and death benefit structure."
    evidences = [
        {
            "chunk_id": "ev_alpha",
            "content": "AlphaOmegaPlan offers annual premium payment for 5 years.",
            "provenance": {"metadata": {"filename": "AlphaOmegaPlan.pdf"}},
        },
        {
            "chunk_id": "ev_beta",
            "content": "BetaGammaPlan supports single premium or 3-year payment period.",
            "provenance": {"metadata": {"filename": "BetaGammaPlan.pdf"}},
        },
        {
            "chunk_id": "ev_generic_1",
            "content": "Premium payment terms vary by product and currency.",
            "provenance": {"metadata": {"filename": "OtherPlan.pdf"}},
        },
        {
            "chunk_id": "ev_generic_2",
            "content": "Payment terms and premium mode are described in the policy schedule.",
            "provenance": {"metadata": {"filename": "AnotherPlan.pdf"}},
        },
        {
            "chunk_id": "ev_generic_3",
            "content": "Payment terms may be annual, semi-annual, quarterly, or monthly.",
            "provenance": {"metadata": {"filename": "YetAnotherPlan.pdf"}},
        },
    ]

    kept, diag = _filter_evidences_by_question_scope(question, evidences)

    assert diag.get("scope_terms_applied") is True
    assert {item["chunk_id"] for item in kept} == {"ev_alpha", "ev_beta"}


def test_filter_evidences_by_question_scope_refuses_anchorless_fallback():
    question = "Compare AlphaPlan3 and BetaPlan2 payment terms."
    evidences = [
        {
            "chunk_id": "ev_other",
            "content": "Some other plan mentions payment terms.",
            "provenance": {"metadata": {"filename": "OtherPlan.pdf"}},
        }
    ]

    kept, diag = _filter_evidences_by_question_scope(question, evidences)

    assert kept == []
    assert diag.get("anchor_miss") is True
