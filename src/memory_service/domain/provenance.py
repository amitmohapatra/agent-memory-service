"""Model provenance policy: which model families may not run anywhere in the stack.

The rule is about origin, not language. Every script is supported and the multilingual
encoder is chosen for that; what is excluded is any Chinese-developed checkpoint or a
derivative of one, whether frozen as a default, fetched as a benchmark challenger, named by
an operator or discovered through the gateway. One pattern, consulted wherever a model is
named - ``ProviderInfo``, ``LLMSettings``, gateway discovery, the download catalogue and the
provenance test - so the rule cannot hold in one place and lapse in another.
"""

from __future__ import annotations

import re

#: Vendor names and family prefixes, matched case-insensitively as substrings of a model id,
#: repository name or gateway model name. Substrings on purpose: a distillation or fine-tune
#: that keeps the family in its name is a derivative and is caught with it.
EXCLUDED_MODEL_ORIGINS = re.compile(
    r"baai|bge|qwen|deepseek|alibaba|gte-|m3e|bce-|glm|thudm|zhipu|kimi|moonshot|minimax"
    r"|(?<![a-z])yi-|01-ai|baichuan|internlm|hunyuan|doubao|ernie|pangu|xverse|skywork"
    r"|telechat",
    re.IGNORECASE,
)


def permitted_model(name: str) -> bool:
    """True when nothing in ``name`` names an excluded family."""
    return EXCLUDED_MODEL_ORIGINS.search(name) is None


def require_permitted_model(name: str) -> str:
    """``name`` unchanged, or ``ValueError`` naming the policy that refused it."""
    if not permitted_model(name):
        raise ValueError(
            f"model {name!r} is from an excluded origin (memory_service.domain.provenance)"
        )
    return name
