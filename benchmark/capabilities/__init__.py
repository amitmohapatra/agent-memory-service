"""The capability eval suite: eleven memory capabilities, measured through the real service.

A measuring instrument, not a feature. Scenarios (``data/*.json``) write realistic,
time-stamped conversations through the SDK, then ask each question in many phrasings and score
what the service hands back - the pushed context (``context()``) and the agent's pull
(``search()``) - against checkable assertions: what must be there, what is stale and must not
be served as current, when the right answer is "I do not know". The scorer is generic: no
domain word lives in code, every domain specific lives in the data.

    make eval-capabilities MODE=nollm        # the deterministic read path
    make eval-capabilities MODE=llm          # the tenant's model policy (a deterministic sample)
    make eval-capabilities MODE=harness      # an agent wrapped by agent-harness answers (a sample)
    make eval-capabilities-gate              # any drop against the recorded baseline fails

``docs/eval/capabilities.md`` is the account of the dataset, the modes, the metrics and the gate.
"""
