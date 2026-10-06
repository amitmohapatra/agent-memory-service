# Design records

Internal design documents, kept because code comments cite their sections. They are not user
documentation and they describe the design as it was proposed, not the current API: where they
disagree with the code or an ADR, the code and the ADR are right.

| Document | What it is | Current description |
|---|---|---|
| [TOOL_MEMORY.md](TOOL_MEMORY.md) | the design history of tool memory ("change 30"); `src/memory_service/modules/tools/*` cite its § numbers | [docs/api/tools.md](../docs/api/tools.md), [ADR 0018](../docs/adr/0018-tool-memory.md) |
| [PRODUCT_DECISIONS.md](PRODUCT_DECISIONS.md) | product-level choices and the evidence for each; ADR 0021 and `modules/authz/service.py` cite §4 | the [ADR index](../docs/adr/README.md) |

Dated measurement reports are in [`benchmark/reports/`](../benchmark/reports/README.md).
