# The LLM gateway is not deployed from this repository

Bifrost is an external service. It holds the provider keys, the prompts and the MCP
registry; this repository holds a client that knows a URL and a model name.

`config.example.json` is here as documentation of what the gateway needs to serve — it is
**not** read by anything in this stack, and no gateway is started by `docker compose up`.
Provider keys are resolved by the gateway from its own environment (`env.ANTHROPIC_API_KEY`,
`env.GEMINI_API_KEY`, …); they never enter this repository, this image, or `docker-compose.yml`.

## Pointing the service at a gateway

```
MEMORY__MODELS__LLM__ENABLED=true
MEMORY__MODELS__LLM__PROVIDER=bifrost
MEMORY__MODELS__LLM__BASE_URL=https://<your-gateway>/v1
MEMORY__MODELS__LLM__MODEL=<provider>/<model>        # e.g. gemini/gemini-3.6-flash
MEMORY__MODELS__LLM__API_KEY=<virtual key>           # issued by the gateway, never a provider key
MEMORY__MODELS__LLM__USES='["grounding_judge"]'      # which call sites may use it
```

With `enabled=false` the service runs complete and every LLM-backed step falls back to its
deterministic path. Generation is an enhancement here, never a dependency.

## Why no gateway service in `docker-compose.yml`

Running one from this compose file would put provider keys inside the application's own
deployment — the exact coupling the gateway exists to remove. It would also mean two places
own outbound model traffic, and the one that is easiest to start is the one that ends up in
production by accident.
