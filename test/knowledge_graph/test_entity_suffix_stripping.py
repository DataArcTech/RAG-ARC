from core.knowledge_graph.schema import schema_from_dict


def test_canonicalize_entity_name_strips_cjk_suffix_without_spaces() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "finance",
            "domains": {
                "finance": {
                    "entity_suffixes_to_strip": ["股份有限公司", "有限公司"],
                }
            },
        }
    )
    domain_schema = schema.for_domain("finance")
    assert domain_schema.canonicalize_entity_name("招商银行股份有限公司") == "招商银行"


def test_canonicalize_entity_name_strips_suffix_with_spaces() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {"default": {"entity_suffixes_to_strip": ["inc"]}},
        }
    )
    domain_schema = schema.for_domain("default")
    assert domain_schema.canonicalize_entity_name("Apex Inc") == "apex"

