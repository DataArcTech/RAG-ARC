# Semantic Unit Eval Complex (Fixture)

This file is a committed test fixture used by `test_semantic_unit_real_docs_e2e.py`.

Deployment Checklist
- [ ] Verify indexes
- [ ] Restart services
- [x] Run smoke tests

Plan Table
| Tier | Segment | Limit |
|---|---|---:|
| Basic | Personal | 10 |
| Pro | SMB | 100 |
| Enterprise | Large | 1000 |

```python
# symbol: get_user_name
def get_user_name(user_id: str) -> str:
    return f"user:{user_id}"
```

$$
E = mc^2
$$

> Quoted note: Customers may have custom SLAs.
