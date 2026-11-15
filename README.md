# Free-model aggregator

One zero-config server that serves the OpenAI Chat Completions API and the
Anthropic Messages API — JSON and streaming — and answers every request with a
real completion from a live free OpenRouter model, dialled through a gateway.

Built for the make.fun `free-model-aggregator-02` competition. Node 22,
standard library only, nothing to install.

## How it works

- **Two wires, one upstream.** `POST /v1/chat/completions` passes through to the
  gateway; `POST /v1/messages` is translated to Chat Completions upstream and
  translated back, including the Anthropic streaming event order, stop reasons,
  usage, and tool calls. `GET /v1/models` answers in the shape its auth header
  names (`Authorization: Bearer` → OpenAI, `x-api-key` → Anthropic).
- **Quality-ordered routing.** The brief lists the free models with the house's
  ratings and capabilities. For each request, the models meeting its
  requirements (tools, JSON mode, images — read off the request body) are ranked
  best-first, and the best is dialled first, every time.
- **Rate-limit-aware failover.** A refusal (4xx, 5xx, broken body, timeout, or
  an answer with no text at all) moves the request to the next-ranked
  candidate; the next request starts again from the best model, since a refusal
  is usually transient. Every refusal is reported on the response in the
  `x-makefun-upstream-failures` trail header. Streams are buffered whole, so a
  truncated stream fails over instead of reaching the client half-sent, and
  relayed streams are repaired (single finish chunk, usage, `[DONE]`
  terminator).
- **The catalog.** `node main.ts --list` prints the catalog document — the
  fourteen dialable free ids, exactly — byte for byte as the server registers
  it with the env.
- **Deterministic.** Every choice derives from the brief, the requests, and the
  gateway's answers. No wall clock is ever written into a request or response,
  and no unseeded randomness is used, so a replay reproduces the same export.

## Layout

- `main.ts` — the server: routes, routing, failover, register, work poll.
- `lib/brief.ts` — the brief parser, the catalog builder, candidate ranking.
- `lib/wires.ts` — wire translations, model listings, SSE handling.

## Running

In a trial the sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`MAKEFUN_TRIAL_SEED`, `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, and the trial's
stand-in keys, then runs `node main.ts`. The server picks a free port, registers
with the env, serves the prompts, polls `GET /work`, and exits 0 when done.

```sh
node main.ts --list   # print the catalog document and exit 0
```

No API key lives in this repository. The entrant's OpenRouter key is pushed to
the competition as a secret; the gateway swaps the trial's stand-in key for it,
and the real key never enters the sandbox.

## License

MIT — see `LICENSE`.
