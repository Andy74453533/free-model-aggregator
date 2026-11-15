// The trial brief: the free model list, the prompt manifest, and the catalog document.
//
// A trial hands this server one file (MAKEFUN_PROMPT_FILE) holding prose with two JSON
// documents embedded in it. This module finds those documents by their `schema` member,
// never by parsing the prose, and turns them into the model table the router ranks and
// the catalog the trial scores.
//
// Everything here is a pure function of the brief's bytes: no clock, no randomness, so
// `--list` and the register body always agree, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

/** A JSON object, the only shape this module treats as a document or a row. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  /** The house's quality rating in milli-points; higher serves a prompt better. */
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
}

/** One prompt the env will send, as the brief's manifest lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as read. `error` names why a document could not be read, if any. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One provider of the catalog document `--list` prints and /register carries. */
export interface CatalogDocument {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The provider's public endpoint. Never the gateway's trial-local address: the catalog is
 *  part of the state export a replay must reproduce byte for byte, and the gateway's origin
 *  differs between a live pass and its replay. */
const PUBLIC_BASE_URL = 'https://openrouter.ai/api/v1';
const KEY_ENV = 'OPENROUTER_API_KEY';

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = matchingClose(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a JSON object on its own; keep scanning.
    }
  }
  return found;
}

/** The index of the `}` closing the object opened at `start`, skipping string contents. */
function matchingClose(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBool(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** The first document under this contest's prefix carrying one of `keys` as an array. */
function documentWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id` (or `model`); a bare upstream id on a row that
 * names the openrouter provider is spelled the way the gateway routes it. A row is free
 * unless it says otherwise; a paid row is never routed.
 */
function modelRowOf(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const raw = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (raw === '') return null;
  const named = asString(value.provider) !== '' ? asString(value.provider) : asString(value.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: asString(value.display_name) !== '' ? asString(value.display_name) : raw,
    ratingMilli: asFiniteNumber(value.rating_milli) ?? 0,
    contextLength: asFiniteNumber(value.context_length),
    tools: asBool(value.tools),
    vision: asBool(value.vision),
    json: asBool(value.json),
    free: value.free !== false,
  };
}

function promptRowOf(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: out,
  };
}

/** Read the brief at `path`. A missing or unreadable brief yields empty lists, never a crash. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no model list document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((row) => {
    const parsed = modelRowOf(row);
    return parsed === null ? [] : [parsed];
  });
  const promptsDoc = documentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no prompts document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((row) => {
    const parsed = promptRowOf(row);
    return parsed === null ? [] : [parsed];
  });
  return { models, prompts, error: null };
}

/** The brief's free models, in the brief's order. Only these may serve a prompt. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((model) => model.free);
}

/** Whether a model meets a prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  return true;
}

/**
 * The candidates for a prompt, best first: every free model meeting the requirements,
 * ordered by the brief's rating, highest first, ties keeping the brief's order. The
 * sort is stable and the order derives from the brief alone, so a live pass and its
 * replay rank identically. When nothing meets the requirements, every free model is a
 * candidate: a best-effort answer beats refusing the prompt outright.
 */
export function rankedCandidates(models: readonly ModelRow[], requires: Requirement): ModelRow[] {
  const meeting = models.filter((model) => meetsRequirements(requires, model));
  const pool = meeting.length > 0 ? meeting : [...models];
  return pool
    .map((model, index) => ({ model, index }))
    .sort((a, b) => b.model.ratingMilli - a.model.ratingMilli || a.index - b.index)
    .map((entry) => entry.model);
}

/** The requirements a request body itself declares: tools, JSON mode, and images. */
export function requirementsFromBody(body: Json): Requirement {
  const out: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') out.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) out.vision = true;
    }
  }
  return out;
}

/**
 * The catalog document: one provider, the brief's free models, each once, in the brief's
 * order. This is the document `--list` prints and /register carries, byte for byte, and
 * the same id set GET /v1/models answers with on both wires. An empty brief yields an
 * empty provider list: there is no brief to read and no route to probe.
 */
export function buildCatalog(brief: Brief): CatalogDocument {
  const seen = new Set<string>();
  const models: CatalogDocument['providers'][number]['models'] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: PUBLIC_BASE_URL,
        key_env: KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
