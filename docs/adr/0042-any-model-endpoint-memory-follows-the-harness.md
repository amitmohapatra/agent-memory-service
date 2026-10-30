# ADR 0042: Any model endpoint, and memory follows the harness

Date: 2026-10-30. Status: accepted. Amends
[ADR 0023](0023-m3-lite-feedback-team-keys-pagination-webhooks.md)'s model keys (a registration
is now a whole endpoint, and the service's own endpoint follows it) and the gateway-only model
access of [chapter 8](../guide/08-models.md); the shared types are
[agent-contracts ADR 0009](https://github.com/amitmohapatra/agent-contracts/blob/main/docs/adr/0009-endpoint-and-model-profile.md).

## Context

The service reached a model only through a Bifrost gateway: `BIFROST_URL`, an operator virtual
key, and a virtual key per agent or tenant registered through the API. Three things were wrong
with that for a deployment that is not ours:

* a harness configured with OpenAI, OpenRouter, a LiteLLM proxy, Ollama, Vertex AI or Bedrock
  had to stand up a Bifrost gateway too, only so memory could call a model, and then configure
  the model twice;
* which model `auto` called was a regular expression over model names (`openai/gpt-4…`,
  `anthropic/claude-…`, `gemini/gemini-…-flash`): a bare `gpt-4o`, a local `llama3.1`, any
  alias a proxy defines was never chosen;
* how a request was shaped for a reasoning model (`max_completion_tokens`, no `temperature`)
  could only be guessed from a name, and was not.

## Decision

**One resolver, ordered sources** (`modules/llm/endpoints.py`, a chain like the harness's
`detect()`): a call's endpoint is (1) the one the acting agent registered, else its tenant's;
(2) the service's own (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_GATEWAY`, `LLM_AUTH`, the names the
harness reads; `MEMORY__MODELS__LLM__MODEL` and `__PROFILE__*`); (3) none, the `DisabledLLM`
path where every use keeps its native result. A revoked registration stops the chain: a revoked
agent is refused, never rerouted. Each answer carries its reason, and the model-key status
routes return it (`source`, `reason`) so a harness's `doctor` can say per agent which endpoint
memory uses and why.

**Configure the harness only, memory follows.** The registration routes keep their paths and
take a `trellis.contracts.Endpoint` (base URL, route and auth by name, model, key, profile; the
key a plain string on the wire). The harness registers its agent's endpoint at startup with
the profile LangChain already keeps for the model, so memory calls the model the agent was
configured with and shapes requests the way that model accepts. The type is vendored from
agent-contracts like bifrost-sdk (`make vendor`): one owner of the shape, not a copy kept in
step by hand.

**Probed before stored.** A registration lists the endpoint's models (`GET /models` with its
own route and auth, 5 s) and is refused with a 422 that says why when the service cannot reach
it. The likeliest failure is an address only the harness's machine resolves (its loopback, a
name in its own network); found at registration it is one clear error rather than every memory
call failing later. The whole endpoint is sealed with the existing credential cipher, so a
database dump shows neither the key nor where the agent's model is.

**Three strategies, each by name from one registry**, none as an if/else at a call site: the
route (`ROUTES`: `openai-compatible`, any endpoint; `bifrost`, which adds the gateway's per-call
headers: no content logging and the deny-all MCP scope), the auth (`bifrost_sdk.AUTH_STRATEGIES`:
`bearer`; `google`, a refreshed token for Vertex AI; `aws`, SigV4 for Bedrock), and the
model's profile (`ModelProfile`: a reasoning model, or one without a temperature, gets
`max_completion_tokens` and no `temperature`, shaped by the sdk; `max_output_tokens` caps the
budget; a model without `tool_calling` gets no `tool_choice`). One sdk client per endpoint, so
the circuit breaker is per endpoint and one agent's broken endpoint never trips another's.

**`auto` reads the listing, not the name.** It takes the cheapest model whose `/models` row
says it writes text (modalities, a `type`/`mode`, a price), any name accepted, with the row's
profile facts refining the request. A listing that says nothing is used only when it holds one
model; otherwise the error asks for an explicit model and names where to put it. The
provenance policy still applies to every candidate.

**The operator decides** with `MEMORY__MODELS__LLM__ACCEPT_AGENT_ENDPOINTS` (default `true`):
`false` stores registrations but never calls them, for an operator who pays for and chooses the
model. The old names (`BIFROST_URL`, `BIFROST_VIRTUAL_KEY`) are gone, with no alias.

## Consequences

* With nothing set, a model is reachable whenever an agent registers one, so the worker runs
  its model topology (enrichment as its own jobs, reflection registered). A deployment that
  must never call a model turns the switch off and sets no `LLM_BASE_URL`; the test suite runs
  that way unless a test names an endpoint.
* The service's own endpoint now serves every tenant and agent without a registration, with or
  without a key (a local server takes none). Before, a gateway URL without the operator's key
  served no one.
* Migration `0031_endpoint_credentials` deletes the stored bare keys (a key says nowhere to send
  it, and the envelope cannot be opened in a migration) and bumps those tenants' revisions; the
  harness registers its agent's endpoint again at its next start, and revocations are kept.
* `google` and `aws` need the `[google]` and `[aws]` extras; without them the probe refuses
  such an endpoint naming the extra.
* Proof: `tests/contract/test_llm_conformance.py` runs every registered route × auth × profile
  against a stub of each wire format (OpenAI chat, Bifrost's, the Anthropic- and
  Gemini-compatible ones) with a Google token refreshed once it expires and a SigV4 signature
  the stub recomputes from what it received; a strategy added without a stub fails it. The
  live-endpoint tests (`live_llm`) run only with `MEMORY_TEST_LIVE_LLM=1` and an endpoint in
  `LLM_*`.

## Not doing

LiteLLM as a library (a LiteLLM proxy is an endpoint like any other); fallback chains (a
gateway, or the harness's own middleware, owns failover); a provider SDK anywhere under `src/`.
