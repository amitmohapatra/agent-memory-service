# The LLM gateway is not deployed from this repository

Bifrost is an external service. It holds the provider keys, the prompts and the MCP
registry; this repository holds a client that knows a URL and a model name.

`config.example.json` is here as documentation of what the gateway needs to serve — it is
**not** read by anything in this stack, and no gateway is started by `docker compose up`.
Provider keys are resolved by the gateway from its own environment (`env.ANTHROPIC_API_KEY`,
`env.GEMINI_API_KEY`, …); they never enter this repository, this image, or `docker-compose.yml`.

## Pointing the service at a gateway

```
BIFROST_URL=https://<your-gateway>/v1
BIFROST_VIRTUAL_KEY=<virtual key>     # optional: the operator's key, issued by the gateway
```

Those are the only two settings. Which uses may call the model, and on which model, is each
tenant's policy (`PUT /v1/model-key/policy`: `uses`, `read_assist`, `models` per use); whose
key pays is the agent's (`PUT /v1/agents/model-key`), else the tenant's (`PUT
/v1/model-key`), else the operator's `BIFROST_VIRTUAL_KEY`. See `docs/USAGE.md`.

Without `BIFROST_URL` the service runs complete and every LLM-backed step falls back to its
deterministic path. Generation is an enhancement here, never a dependency.

## Why no gateway service in `docker-compose.yml`

Running one from this compose file would put provider keys inside the application's own
deployment — the exact coupling the gateway exists to remove. It would also mean two places
own outbound model traffic, and the one that is easiest to start is the one that ends up in
production by accident.
