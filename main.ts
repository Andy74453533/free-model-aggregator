// free-model-aggregator-02: one server, two wires, a real free model behind every answer.
//
// The trial runs this file as `node /bot/main.ts` with Node 22's standard library alone.
// It reads the brief, listens on a free port, registers with the env, and serves the six
// prompts the env sends, one at a time: each is answered by dialling a live free OpenRouter
// model through the grader's gateway, on the trial's stand-in key (the gateway swaps in the
// entrant's own pushed key; no real key ever enters the sandbox).
//
// ROUTING: for each prompt, the free models meeting its requirements (tools, JSON mode,
// images, read off the request body itself) are ranked by the brief's rating, best first,
// and dialled in that order. Quality is 60% of the score and pays the rating of the model
// that really answered over the best rating at the prompt's requirements, so the best
// model goes first, every time.
//
// FAILOVER: a gateway refusal (a 4xx, a 5xx, a broken body, a timeout, or an answer
// with no text at all) is never an answer. The prompt moves to the next-ranked
// candidate; the next prompt starts again from the best model, because a refusal is
// usually transient and a single 429 must not demote every later prompt. Every refusal
// is reported on the response in the x-makefun-upstream-failures trail, the only way
// the failover column can see a refusal the server recovered from. An answer with no
// text is treated as a refusal too: no scored check can credit it, so another model
// gets the chance, and the last well-formed answer is still relayed as a best effort
// when nothing better arrives.
//
// DETERMINISM: every choice derives from the brief, the requests, and the gateway's
// answers. No wall clock is ever written into a request or a response, and no unseeded
// randomness is used anywhere, so a replay leaves the same state export byte for byte.
// Timers exist only as watchdogs (a hung upstream call, the trial's wall), never as data.
//
// `node main.ts --list` prints the catalog document and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankedCandidates,
  readBrief,
  requirementsFromBody,
  type Brief,
  type CatalogDocument,
  type Json,
  type ModelRow,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  chatAnswerText,
  dialBody,
  isWellFormedChatAnswer,
  openAIAnswer,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToAnthropic,
  relayOpenAIStream,
  type ServedModel,
} from './lib/wires.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one buffered upstream stream this server keeps. */
const MAX_STREAM_BYTES = 4 * 1024 * 1024;
/** The most gateway attempts spent on one prompt. Six prompts stay under the trial's 60-call cap. */
const MAX_ATTEMPTS_PER_PROMPT = 8;
/** One gateway attempt's budget, in milliseconds: inside the env's 30 s per-prompt wait. */
const DIAL_TIMEOUT_MS = 25_000;
/** A prompt's whole budget, in milliseconds: attempts stop when it is spent. */
const PROMPT_BUDGET_MS = 28_000;
/** Last-resort exit when /work never says done: inside the trial's 240 s wall. */
const WALL_GUARD_MS = 225_000;
/** Register retries: the env may not be listening on the very first attempt. */
const REGISTER_ATTEMPTS = 25;
const REGISTER_RETRY_MS = 200;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's stand-in key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogDocument = buildCatalog(brief);
const served: ServedModel[] = freeModels(brief).map((model) => ({
  id: model.id,
  provider: model.provider,
  displayName: model.displayName,
}));

/** Gateway calls made this trial, counted for the log line only. */
let gatewayCalls = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route, whatever spelling of the base URL the trial sets. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** One refusal, as the trail header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header's value for a prompt's refusals: at most 8 rows, well under 8 KiB. */
function trailHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, 8).map((row) => ({
    status: row.status,
    code: row.code.slice(0, 64),
    model: row.model === null ? null : row.model.slice(0, 200),
  }));
  const text = JSON.stringify(rows);
  return text.length > 8192 ? null : text;
}

/** An upstream error body's code or type, as the gateway or provider spelled it. */
function errorCodeOf(text: string): string {
  if (text === '') return 'upstream_error';
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code;
      const type = value.error.type;
      if (typeof type === 'string' && type !== '') return type;
    }
  } catch {
    // Not JSON: fall through to the generic code.
  }
  return 'upstream_error';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One upstream response body as text, capped; null when the body runs past the cap. */
async function readCappedText(response: Response): Promise<string | null> {
  if (response.body === null) return '';
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const buffer = Buffer.from(value);
      size += buffer.length;
      if (size > MAX_STREAM_BYTES) return null;
      chunks.push(buffer);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly FailureRow[]): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendEvents(res: ServerResponse, bytes: string, failures: readonly FailureRow[]): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(Buffer.byteLength(bytes)),
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  res.end(bytes);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly FailureRow[]): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(served), []);
  else sendJson(res, 200, openAIModelList(served), []);
}

/** A dialled attempt's relayable content: the model, its answer, and a buffered stream if any. */
interface Relay {
  model: string;
  answer: Json;
  rawStream: string | null;
}

/** One dialled attempt's outcome: a usable answer, a well-formed but textless one, or a refusal. */
type Attempt =
  | { kind: 'answer'; relay: Relay }
  | { kind: 'empty'; relay: Relay }
  | { kind: 'refused'; failure: FailureRow };

/**
 * Dial one model once. A 2xx with a well-formed, text-bearing answer is usable; a 2xx
 * that is well-formed but carries no text at all is kept as a best effort (no scored
 * check can credit a textless answer, so the caller still fails over from it); anything
 * else — a refusal status, a broken body, a timeout — is a refusal. Streams are buffered
 * whole so a truncated or empty stream can fail over too, instead of reaching the
 * client half-sent.
 */
async function attempt(model: ModelRow, upstreamBody: Json, stream: boolean, deadlineMs: number): Promise<Attempt> {
  const remaining = deadlineMs - Date.now();
  if (remaining <= 0) {
    return { kind: 'refused', failure: { status: 504, code: 'prompt_budget_spent', model: model.id } };
  }
  gatewayCalls += 1;
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(stream ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify({ ...upstreamBody, model: model.id }),
      signal: AbortSignal.timeout(Math.min(DIAL_TIMEOUT_MS, remaining)),
    });
  } catch {
    return { kind: 'refused', failure: { status: 502, code: 'upstream_unreachable', model: model.id } };
  }
  const raw = await readCappedText(response);
  if (raw === null) {
    return { kind: 'refused', failure: { status: 502, code: 'upstream_body_too_large', model: model.id } };
  }
  if (!response.ok) {
    return { kind: 'refused', failure: { status: response.status, code: errorCodeOf(raw), model: model.id } };
  }
  if (stream) {
    const summary = streamSummary(raw);
    if (summary.chunks === 0) {
      return { kind: 'refused', failure: { status: 200, code: 'bad_upstream_body', model: model.id } };
    }
    const relay: Relay = { model: model.id, answer: {}, rawStream: raw };
    return summary.text === '' ? { kind: 'empty', relay } : { kind: 'answer', relay };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: 'refused', failure: { status: 200, code: 'bad_upstream_body', model: model.id } };
  }
  if (!isRecord(parsed) || !isWellFormedChatAnswer(parsed)) {
    return { kind: 'refused', failure: { status: 200, code: 'bad_upstream_body', model: model.id } };
  }
  const relay: Relay = { model: model.id, answer: parsed, rawStream: null };
  return chatAnswerText(parsed) === '' ? { kind: 'empty', relay } : { kind: 'answer', relay };
}

/** A buffered Chat Completions stream, summarized: its chunk count and the text it carried. */
function streamSummary(raw: string): { chunks: number; text: string } {
  let chunks = 0;
  let text = '';
  for (const line of raw.replace(/\r\n/g, '\n').split('\n')) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice('data:'.length).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
      chunks += 1;
      const choice = chunk.choices[0];
      if (isRecord(choice) && isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    } catch {
      // Not JSON: not a content chunk.
    }
  }
  return { chunks, text };
}

/**
 * Serve one prompt: dial ranked candidates in order until one answers with text. A
 * refused model is not retried for this prompt; the next prompt starts again from the
 * top of the ranking. When every attempt is spent, the last well-formed textless
 * answer is relayed as a best effort (its wire shape still counts for something); only
 * when nothing usable arrived at all is the prompt given up on.
 */
async function servePrompt(
  candidates: readonly ModelRow[],
  buildUpstream: (model: ModelRow) => Json,
  stream: boolean,
): Promise<{ relay: Relay | null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  let fallback: Relay | null = null;
  const deadlineMs = Date.now() + PROMPT_BUDGET_MS;
  const tries = Math.min(MAX_ATTEMPTS_PER_PROMPT, candidates.length);
  for (let i = 0; i < tries; i += 1) {
    const model = candidates[i];
    const outcome = await attempt(model, buildUpstream(model), stream, deadlineMs);
    if (outcome.kind === 'answer') return { relay: outcome.relay, failures };
    if (outcome.kind === 'empty') {
      failures.push({ status: 200, code: 'empty_upstream_answer', model: model.id });
      fallback = outcome.relay;
    } else {
      failures.push(outcome.failure);
    }
    log(`refused: ${model.id} (${failures[failures.length - 1].code}); failing over`);
  }
  return { relay: fallback, failures };
}

/** POST /v1/chat/completions: the client's body, dialled as-is, answered in the OpenAI wire. */
async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
    return;
  }
  const stream = body.stream === true;
  const candidates = rankedCandidates(freeModels(brief), requirementsFromBody(body));
  const served = await servePrompt(candidates, (model) => dialBody(body, model.id), stream);
  if (served.relay === null) {
    sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt', served.failures);
    return;
  }
  const { relay: won, failures } = served;
  if (stream && won.rawStream !== null) {
    sendEvents(res, relayOpenAIStream(won.rawStream, won.model).bytes, failures);
    return;
  }
  sendJson(res, 200, openAIAnswer(won.answer, won.model), failures);
}

/** POST /v1/messages: translated to Chat Completions upstream, translated back to Messages. */
async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
    return;
  }
  const stream = body.stream === true;
  const candidates = rankedCandidates(freeModels(brief), requirementsFromBody(body));
  const served = await servePrompt(candidates, (model) => anthropicToOpenAI(body, model.id), stream);
  if (served.relay === null) {
    sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt', served.failures);
    return;
  }
  const { relay: won, failures } = served;
  if (stream && won.rawStream !== null) {
    sendEvents(res, openAIStreamToAnthropic(won.rawStream, won.model).bytes, failures);
    return;
  }
  sendJson(res, 200, openAIToAnthropic(won.answer, won.model), failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleMessages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`, []);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the catalog, retrying while the env comes up. */
async function register(port: number): Promise<void> {
  let lastError = '';
  for (let i = 0; i < REGISTER_ATTEMPTS; i += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  throw new Error(`POST ${ENV_URL}/register failed after ${REGISTER_ATTEMPTS} attempts: ${lastError}`);
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          log(`done: ${gatewayCalls} gateway calls this trial`);
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. Wait for stdout to flush first.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer', []);
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${served.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  // Last resort: never be the process still running when the wall is spent.
  setTimeout(() => {
    log('wall guard: exiting');
    process.exit(0);
  }, WALL_GUARD_MS).unref();
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
