// The two client wires and the one upstream wire.
//
// The trial's client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or in
// server-sent events. The gateway speaks Chat Completions only, so every prompt goes
// upstream as a Chat Completions request and an Anthropic-shaped answer is translated back.
// This module holds the translations, the model listings, and the stream handling.
//
// The model field is the one truthful claim an answer makes: the gateway's provider names
// a free model without the `openrouter/` prefix, so every function here puts the id this
// entry actually dialled back on every answer and every streamed chunk.
//
// Everything here is a pure function of its arguments: no clock, no randomness, fixed
// listing timestamps, so the same request and the same gateway answer give the same bytes
// on a live pass and on its replay.

import { isRecord, type Json } from './brief.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** Fixed listing timestamps: a `created` that moved between runs would be nondeterministic. */
const LIST_CREATED = 1735689600;
const LIST_CREATED_AT = '2025-01-01T00:00:00Z';

/** GET /v1/models for an OpenAI-shaped request: OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((model) => ({ id: model.id, object: 'model', created: LIST_CREATED, owned_by: model.provider })),
  };
}

/** GET /v1/models for an Anthropic-shaped request (an x-api-key header): Anthropic's shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((model) => ({ type: 'model', id: model.id, display_name: model.displayName, created_at: LIST_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's Chat Completions shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's Messages shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The text of a string, or of a block array's text blocks joined; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** A rough token count for text, used only when a provider omits usage: ~4 chars a token. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function finiteNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One Anthropic content block as an OpenAI content part; null for blocks with no upstream form. */
function contentPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI finish_reason as the Anthropic stop_reason that means the same thing. */
export function anthropicStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as a Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url parts, tool_use blocks
 * become tool_calls, tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, sampling parameters, stop sequences and streaming pass
 * through. The output cap is always present: the gateway refuses a request without one.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = contentPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  out.max_tokens = typeof body.max_tokens === 'number' ? body.max_tokens : 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      if (raw.type !== undefined && raw.type !== 'custom' && raw.type !== 'function') return [];
      // Only plain function tools translate; provider-specific tool kinds are dropped.
      if (raw.type === 'custom') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A Chat Completions request body made safe to dial: the dialled model, an output cap the
 * gateway accepts (present, at most 8192), usage on streams, and no provider-side state
 * (the gateway refuses `store: true`). Everything else passes through unchanged.
 */
export function dialBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.store;
  const cap = (value: unknown): number | null =>
    typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.min(Math.floor(value), 8192) : null;
  const maxTokens = cap(body.max_tokens) ?? cap(body.max_completion_tokens);
  if (maxTokens === null) {
    out.max_tokens = 1024;
  } else {
    if (typeof body.max_tokens === 'number') out.max_tokens = maxTokens;
    if (typeof body.max_completion_tokens === 'number') out.max_completion_tokens = maxTokens;
  }
  if (body.stream === true) {
    const options = isRecord(body.stream_options) ? body.stream_options : {};
    out.stream_options = { ...options, include_usage: true };
  }
  return out;
}

/** The text a Chat Completions answer carries, across its choices. */
export function chatAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  let text = '';
  for (const choice of choices) {
    if (!isRecord(choice) || !isRecord(choice.message)) continue;
    const content = choice.message.content;
    text += typeof content === 'string' ? content : textOf(content);
  }
  return text;
}

/** Whether a Chat Completions answer carries tool calls in any choice. */
export function chatAnswerHasToolCalls(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  for (const choice of choices) {
    if (isRecord(choice) && isRecord(choice.message) && Array.isArray(choice.message.tool_calls) && choice.message.tool_calls.length > 0) {
      return true;
    }
  }
  return false;
}

/** Whether a Chat Completions answer is shaped well enough to relay: one object, choices, a message. */
export function isWellFormedChatAnswer(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  if (choices.length === 0) return false;
  const first = choices[0];
  return isRecord(first) && isRecord(first.message);
}

/**
 * Usage with the provider's counts where present, else derived from the answer's own
 * text: an output count of at least 1 whenever text was produced. Only used when the
 * provider omits usage; the gateway's answers normally carry it.
 */
function usageWithFallback(usage: unknown, text: string): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const record = isRecord(usage) ? usage : {};
  const prompt = finiteNumber(record.prompt_tokens);
  let completion = finiteNumber(record.completion_tokens);
  if (completion < 1 && text !== '') completion = estimateTokens(text);
  return { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion };
}

/**
 * A Chat Completions answer with `model` set to the id this entry dialled, and usage
 * filled in when the provider omitted it. The one truthful claim about the model.
 */
export function openAIAnswer(body: Json, model: string): Json {
  const usage = usageWithFallback(body.usage, chatAnswerText(body));
  return { ...body, model, usage };
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  calls.forEach((call, index) => {
    if (!isRecord(call) || !isRecord(call.function)) return;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsObject(call.function.arguments),
    });
  });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageWithFallback(body.usage, text);
  return {
    id: `msg_${typeof body.id === 'string' && body.id !== '' ? body.id : 'fallback'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event: an optional `event:` line, the JSON data line, the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (if any) and its data payload lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split raw SSE bytes into frames on blank lines, per the SSE grammar. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The JSON object of a frame's data payload, or null when it is [DONE] or not JSON. */
function frameJson(frame: SseFrame): Json | null {
  const payload = frame.data.trim();
  if (payload === '' || payload === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Whether a frame is the stream's [DONE] terminator. */
function isDone(frame: SseFrame): boolean {
  return frame.data.trim() === '[DONE]';
}

/**
 * A buffered Chat Completions stream, repaired and relayed: every chunk's `model` is the
 * id dialled, at most one chunk carries a finish_reason (the first), a finish chunk is
 * appended when the provider sent none, a usage chunk is appended when no chunk carried
 * usage, and the stream always ends with [DONE]. Repairs only fire when the provider's
 * stream is truncated or malformed; a healthy stream passes through chunk for chunk.
 * Returns the relay bytes and the text the stream carried.
 */
export function relayOpenAIStream(raw: string, model: string): { bytes: string; text: string; sawFinish: boolean } {
  const frames = parseSseFrames(raw);
  const chunks: Json[] = [];
  let text = '';
  let sawUsage = false;
  let sawFinish = false;
  let firstId: string | null = null;
  let firstCreated: number | null = null;
  for (const frame of frames) {
    if (isDone(frame)) continue;
    const chunk = frameJson(frame);
    if (chunk === null) continue;
    if (firstId === null && typeof chunk.id === 'string') firstId = chunk.id;
    if (firstCreated === null && typeof chunk.created === 'number') firstCreated = chunk.created;
    if (isRecord(chunk.usage)) sawUsage = true;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null) {
      const delta = isRecord(choice.delta) ? choice.delta : {};
      const hasFinish = typeof choice.finish_reason === 'string' && choice.finish_reason !== '';
      if (sawFinish) {
        // Content after the finish chunk breaks the stream's order: keep the chunk, drop
        // its content delta and any repeated finish_reason. Text already relayed stays.
        const { content: _dropped, ...restDelta } = delta;
        chunks.push({
          ...chunk,
          model,
          choices: [{ ...choice, delta: restDelta, finish_reason: hasFinish ? null : choice.finish_reason }],
        });
        continue;
      }
      if (typeof delta.content === 'string') text += delta.content;
      if (hasFinish) sawFinish = true;
    }
    chunks.push({ ...chunk, model });
  }
  const out: string[] = chunks.map((chunk) => sseEvent(null, chunk));
  if (!sawFinish) {
    out.push(
      sseEvent(null, {
        id: firstId ?? 'chatcmpl-buffered',
        object: 'chat.completion.chunk',
        created: firstCreated ?? 0,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      }),
    );
    sawFinish = true;
  }
  if (!sawUsage) {
    const completion = text !== '' ? estimateTokens(text) : 1;
    out.push(
      sseEvent(null, {
        id: firstId ?? 'chatcmpl-buffered',
        object: 'chat.completion.chunk',
        created: firstCreated ?? 0,
        model,
        choices: [],
        usage: { prompt_tokens: 0, completion_tokens: completion, total_tokens: completion },
      }),
    );
  }
  out.push('data: [DONE]\n\n');
  return { bytes: out.join(''), text, sawFinish };
}

/**
 * A buffered Chat Completions stream as the Anthropic event sequence: message_start first
 * (with the stream's real input count), one content block per text run and per tool call,
 * each opened, fed deltas and closed in order, then message_delta with the stop reason
 * and the real output count, then message_stop. The sequence is well-ordered by
 * construction whatever the provider sent. Returns the event bytes and the text carried.
 */
export function openAIStreamToAnthropic(raw: string, model: string): { bytes: string; text: string } {
  const frames = parseSseFrames(raw);
  const chunks: Json[] = [];
  for (const frame of frames) {
    if (isDone(frame)) continue;
    const chunk = frameJson(frame);
    if (chunk !== null) chunks.push(chunk);
  }
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let text = '';
  for (const chunk of chunks) {
    if (isRecord(chunk.usage)) {
      inputTokens = finiteNumber(chunk.usage.prompt_tokens);
      outputTokens = finiteNumber(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
      stopReason = anthropicStopReason(choice.finish_reason);
    }
  }
  if (outputTokens < 1 && text !== '') outputTokens = estimateTokens(text);
  if (outputTokens < 1) outputTokens = 1;

  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  );
  let textIndex = -1;
  let nextIndex = 0;
  const toolBlocks = new Map<number, number>();
  const closeText = (): void => {
    if (textIndex === -1) return;
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
    textIndex = -1;
  };
  const closeTools = (): void => {
    for (const index of [...toolBlocks.values()].sort((a, b) => a - b)) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolBlocks.clear();
  };
  for (const chunk of chunks) {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        events.push(
          sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }),
        );
      }
      events.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }),
      );
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolBlocks.size;
      let index = toolBlocks.get(upstreamIndex);
      if (index === undefined) {
        closeText();
        index = nextIndex;
        nextIndex += 1;
        toolBlocks.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        events.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        events.push(
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          }),
        );
      }
    }
  }
  closeText();
  closeTools();
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return { bytes: events.join(''), text };
}
