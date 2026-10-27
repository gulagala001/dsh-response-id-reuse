import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Context } from '@deepseek-ai/cordis';
import { createResponsesAuto } from '../src/engine.mjs';
import { apply, inject } from '../src/index.mjs';
import { acquireShared } from '../src/shared.mjs';
import { nativeReplayHistory } from '../src/replay.mjs';

const url = 'https://gateway.example/v1/responses';
const model = 'arbitrary-vision-model';
const meta = { provider: 'fixture', model, sessionId: 'session-a' };
const user = text => ({ role: 'user', content: [{ type: 'input_text', text }] });
const assistant = text => ({ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] });
const initial = [user('Read these images'), { role: 'user', content: [{ type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=' }] }];
const base = input => ({ model, input, store: false, stream: false, prompt_cache_key: 'session-a', reasoning: { effort: 'low' } });
const response = (id, output = [assistant('OK')], extra = {}) => ({ id, status: 'completed', store: true, output, ...extra });
const json = value => new Response(JSON.stringify(value), { headers: { 'content-type': 'application/json' } });
const sse = value => new Response(`data: ${JSON.stringify({ type: 'response.completed', response: value })}\n\n`, { headers: { 'content-type': 'text/event-stream' } });
async function fixture(handler, readback = value => value) {
  const calls = [], reads = [], stored = new Map();
  const send = async (input, init) => {
    const r = new Request(input, init);
    if (r.method === 'GET') {
      reads.push(r);
      const value = stored.get(decodeURIComponent(new URL(r.url).pathname.split('/').at(-1)));
      return value ? readback(structuredClone(value), r) : Response.json({}, { status: 404 });
    }
    const body = JSON.parse(await r.text());
    calls.push({ url: r.url, body, headers: r.headers });
    const result = handler ? await handler(body, calls.length, r) : json(response('r' + calls.length));
    const content = await result.clone().text();
    try {
      const value = result.headers.get('content-type')?.includes('text/event-stream')
        ? content.split('\n').filter(line => line.startsWith('data:')).map(line => JSON.parse(line.slice(5))).find(e => e.type === 'response.completed')?.response
        : JSON.parse(content);
      if (value?.id) stored.set(value.id, value);
    } catch {}
    return result;
  };
  const engine = await createResponsesAuto({ fetchImpl: send });
  const call = async (body, scope = meta, headers = {}, address = url) => {
    const result = await engine.fetchFor(scope)(address, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer fixture-a', ...headers }, body: JSON.stringify(body) });
    return result.headers.get('content-type')?.includes('text/event-stream') ? result.text() : result.json();
  };
  return { engine, calls, reads, call, send };
}

test('full initial request, multi-image delta, tool round and zero probe calls', async () => {
  const f = await fixture((body, n) => json(response('r' + n, n === 2
    ? [{ type: 'function_call', call_id: 'call-1', name: 'read', arguments: '{"path":"example.txt"}' }] : undefined)));
  const a = await f.call(base(initial));
  assert.equal(f.calls[0].body.store, true);
  const history = [...initial, ...a.output, user('Compare them')];
  const b = await f.call(base(history));
  assert.equal(f.calls[1].body.previous_response_id, 'r1');
  assert.deepEqual(f.calls[1].body.input, [user('Compare them')]);
  await f.call(base([...history, ...b.output, { type: 'function_call_output', call_id: 'call-1', output: 'contents' }]));
  assert.equal(f.calls[2].body.previous_response_id, 'r2');
  assert.equal(f.calls[2].body.input[0].type, 'function_call_output');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.calls.length, 3); assert.equal(f.engine.inspect().probeRequests, 0);
  await f.engine.close(); assert.equal(f.engine.inspect().activeChains, 0);
});

test('history changes, compaction, accounts, sessions and parameters cannot share IDs', async () => {
  const f = await fixture(); const a = await f.call(base(initial));
  const history = [...initial, ...a.output, user('Continue')];
  for (const [body, scope, headers] of [
    [base([user('Edited'), ...history.slice(1)]), meta, {}],
    [base([user('Compacted summary')]), meta, {}],
    [base(history), { ...meta, sessionId: 'branch' }, {}],
    [base(history), meta, { authorization: 'Bearer fixture-b' }],
    [{ ...base(history), reasoning: { effort: 'high' } }, meta, {}],
    [{ ...base(history), model: 'another-model' }, { ...meta, model: 'another-model' }, {}],
  ]) { await f.call(body, scope, headers); assert.equal(f.calls.at(-1).body.previous_response_id, undefined); }
});

test('no session and ZDR are exact pass-through, including Request objects', async () => {
  const f = await fixture();
  for (const [address, scope, headers] of [
    [url, { ...meta, sessionId: undefined }, {}],
    [url, meta, { 'x-cmd-zdr': '1' }],
  ]) {
    const body = { ...base(initial) }; if (!scope.sessionId) delete body.prompt_cache_key;
    const req = new Request(address, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
    await (await f.engine.fetchFor(scope)(req)).json();
    assert.deepEqual(f.calls.at(-1).body, body);
  }
  assert.equal(f.calls.length, 2);
});

test('new endpoints and model names use ordinary storage receipts without a provider allowlist', async () => {
  const f = await fixture();
  for (const [address, id] of [['https://new.example/custom/responses', 'new-model'], ['http://127.0.0.1:7777/v1/responses', 'local-model']]) {
    const scope = { ...meta, model: id }, one = { ...base(initial), model: id };
    const a = await f.call(one, scope, {}, address);
    await f.call({ ...one, input: [...initial, ...a.output, user('Next')] }, scope, {}, address);
    assert.equal(f.calls.at(-1).body.previous_response_id, a.id);
    assert.deepEqual(f.calls.at(-1).body.input, [user('Next')]);
  }
  assert.equal(f.calls.length, 4);
});

test('ID alone, omitted storage and explicitly disabled storage never authorize context omission', async () => {
  for (const storage of [false, undefined]) {
    const f = await fixture(() => json(response('opaque-id', undefined, { store: storage })));
    const a = await f.call(base(initial));
    const history = [...initial, ...a.output, user('Next')];
    await f.call(base(history));
    assert.equal(f.calls.length, 2);
    assert.deepEqual(f.calls[1].body, { ...base(history), store: true });
  }
});

test('an unconfirmed receipt does not prevent later ordinary requests from establishing a chain', async () => {
  for (const unconfirmed of [{ store: false }, { store: undefined }, { id: '' }, { output: undefined }]) {
    const f = await fixture((_body, n) => json(response('r' + n, undefined, n === 1 ? unconfirmed : {})));
    await f.call(base(initial));
    const history = [...initial, assistant('OK'), user('Next')];
    const b = await f.call(base(history));
    assert.deepEqual(f.calls[1].body, { ...base(history), store: true });
    await f.call(base([...history, ...b.output, user('Continue')]));
    assert.equal(f.calls[2].body.previous_response_id, 'r2');
    assert.equal(f.calls.length, 3); assert.equal(f.engine.inspect().probeRequests, 0);
  }
});

test('one unconfirmed response leaves other sessions confirmed chains usable', async () => {
  const f = await fixture((_body, n) => json(response('r' + n, undefined, n === 2 ? { store: false } : {})));
  const a = await f.call(base(initial));
  await f.call(base(initial), { ...meta, sessionId: 'session-b' });
  await f.call(base([...initial, ...a.output, user('Continue a')]));
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
});

test('an unstored text-only request in the same session cannot disable an established image chain', async () => {
  const f = await fixture((_body, n) => sse(response('r' + n, undefined, n === 2 ? { store: false } : {})));
  const headers = { session_id: meta.sessionId, 'x-client-request-id': meta.sessionId };
  await f.call({ ...base(initial), stream: true }, meta, headers);
  await f.call({ ...base([user('Summarize this text')]), max_output_tokens: 128, stream: true }, meta, headers);
  await f.call({ ...base([...initial, assistant('OK'), user('Recall the images')]), stream: true }, meta, headers);
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
  assert.deepEqual(f.calls[2].body.input, [user('Recall the images')]);
  assert.equal(f.calls.length, 3); assert.equal(f.engine.inspect().probeRequests, 0);
});

test('background histories with identical generation parameters preserve the main continuation', async () => {
  const f = await fixture((_body, n) => json(response('r' + n, undefined, n === 2 ? { store: false } : {})));
  const a = await f.call(base(initial));
  await f.call(base([user('Summarize unrelated background text')]));
  await f.call(base([...initial, ...a.output, user('Continue the main history')]));
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
});

test('a pending background request cannot reserve the main chain or replace it on late completion', async () => {
  let release;
  const f = await fixture((_body, n) => n === 2
    ? new Promise(resolve => { release = () => resolve(json(response('background'))); })
    : json(response('r' + n)));
  const a = await f.call(base(initial));
  const background = f.call(base([user('Unrelated background history')]));
  while (!release) await new Promise(resolve => setImmediate(resolve));
  const history = [...initial, ...a.output, user('Continue main')];
  const b = await f.call(base(history));
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
  release(); await background;
  await f.call(base([...history, ...b.output, user('Continue main again')]));
  assert.equal(f.calls[3].body.previous_response_id, 'r3');
});

test('an explicit storage rejection for text requests cannot disable image storage', async () => {
  const f = await fixture((body, n) => n === 1
    ? new Response(JSON.stringify({ error: { message: 'store is unsupported for this request' } }), { status: 400 })
    : json(response('r' + n)));
  await f.call(base([user('Text only')]));
  const a = await f.call(base(initial));
  assert.equal(f.calls[2].body.store, true);
  await f.call(base([...initial, ...a.output, user('Continue images')]));
  assert.equal(f.calls[3].body.previous_response_id, 'r3');
});

test('storage rejection restores the original string input and all generation controls', async () => {
  const f = await fixture(body => body.store ? new Response(JSON.stringify({ error: { message: 'store is not supported' } }), { status: 400 }) : json(response('plain')));
  const body = { ...base('A string input'), max_output_tokens: 12345, instructions: 'Preserve this instruction' };
  await f.call(body);
  assert.deepEqual(f.calls[1].body, body);
});

test('changing an assistant message phase sends the complete local history', async () => {
  const f = await fixture(() => json(response('phase-id', [{ ...assistant('OK'), phase: 'final_answer' }])));
  const a = await f.call(base(initial));
  await f.call(base([...initial, { ...a.output[0], phase: 'commentary' }, user('Next')]));
  assert.equal(f.calls[1].body.previous_response_id, undefined);
  assert.equal(f.calls[1].body.input.length, initial.length + 2);
});

test('changing a tool namespace cannot reuse the original tool history', async () => {
  const call = { type: 'function_call', call_id: 'call-1', name: 'read', namespace: 'files', arguments: '{}' };
  const f = await fixture(() => json(response('tool-id', [call])));
  await f.call(base(initial));
  await f.call(base([...initial, { ...call, namespace: 'network' }, { type: 'function_call_output', call_id: 'call-1', output: 'result' }]));
  assert.equal(f.calls[1].body.previous_response_id, undefined);
});

test('CMD uses its own policy without Bailian headers or optional cache key', async () => {
  const f = await fixture();
  const cmd = { provider: 'cmd', model: 'deepseek/deepseek-v4.1-flash', sessionId: 'cmd-session' };
  const address = 'https://api.commandcode.ai/provider/v1/responses';
  const a = await f.call({ ...base(initial), model: cmd.model }, cmd, {}, address);
  await f.call({ ...base([...initial, ...a.output, user('More')]), model: cmd.model }, cmd, {}, address);
  assert.equal(f.calls[1].body.previous_response_id, 'r1');
  for (const c of f.calls) {
    assert.equal(c.headers.get('x-dashscope-session-cache'), null);
    assert.equal(c.body.prompt_cache_key, undefined);
  }
});

test('explicit invalid ID falls back once and establishes the replacement chain in that same request', async () => {
  const f = await fixture((body, n) => n === 2 && body.previous_response_id
    ? new Response(JSON.stringify({ error: { message: 'previous_response_id not found', code: 'invalid_request_error' } }), { status: 400 })
    : json(response('r' + n)));
  const a = await f.call(base(initial)), history = [...initial, ...a.output, user('Continue')];
  const fresh = await f.call(base(history));
  assert.equal(f.calls.length, 3);
  assert.deepEqual(f.calls[2].body, { ...base(history), store: true });
  await f.call(base([...history, ...fresh.output, user('Continue fresh')]));
  assert.equal(f.calls.length, 4); assert.equal(f.calls[3].body.previous_response_id, 'r3');
});

test('unsupported store is reverted; unrelated provider errors are not retried', async () => {
  const f = await fixture(body => body.store
    ? new Response(JSON.stringify({ error: { message: 'store is unsupported' } }), { status: 400 }) : json(response('normal')));
  await f.call(base(initial)); assert.equal(f.calls.length, 2); assert.deepEqual(f.calls[1].body, base(initial));
  await f.call(base(initial)); assert.equal(f.calls.length, 3);
  const g = await fixture(() => new Response(JSON.stringify({ error: { message: 'rate limited' } }), { status: 429 }));
  await g.call(base(initial)); assert.equal(g.calls.length, 1);
  const h = await fixture(() => { throw Error('offline'); });
  await assert.rejects(h.call(base(initial)), /offline/); assert.equal(h.calls.length, 1);
});

test('Codex storage rejection falls back once with full history and is remembered', async () => {
  for (const error of [
    { detail: 'Store must be set to false' },
    { error: { message: 'Store must be set to false' } },
    { error: { detail: 'Store must be false' } },
    { error: { code: 'invalid_request_error' }, detail: 'Store must be set to false' },
  ]) {
    for (const established of [false, true]) {
      const f = await fixture((body, n) => established && n === 1 ? json(response('r1')) : body.store
        ? Response.json(error, { status: 400 }) : sse(response('normal', undefined, { store: false })));
      const scope = { ...meta, provider: 'codex' }, address = 'https://chatgpt.com/backend-api/codex/responses';
      let history = initial;
      if (established) {
        const previous = await f.call({ ...base(initial), instructions: 'Keep the full context' }, scope, {}, address);
        history = [...initial, ...previous.output, user('Continue')];
      }
      const body = { ...base(history), stream: true, instructions: 'Keep the full context' };
      const result = await f.call(body, scope, {}, address);
      assert.match(result, /response.completed/);
      assert.equal(f.calls.length, established ? 3 : 2);
      assert.equal(f.calls[established ? 1 : 0].body.previous_response_id, established ? 'r1' : undefined);
      assert.deepEqual(f.calls.at(-1).body, body);
      assert.equal(f.engine.inspect().fallbacks, 1);
      assert.equal(f.engine.inspect().activeChains, 0);
      await f.call(body, scope, {}, address);
      assert.equal(f.calls.length, established ? 4 : 3);
      assert.deepEqual(f.calls.at(-1).body, body);
      assert.equal(f.engine.inspect().fallbacks, 1);
      await f.engine.close();
    }
  }
});

test('detail errors unrelated to storage rejection and failed fallbacks are not retried again', async () => {
  for (const [status, detail, attempts] of [
    [400, 'Temperature is not supported', 1],
    [400, 'Store backend temporarily unavailable', 1],
    [429, 'Store must be set to false', 1],
    [400, 'Store must be set to false', 2],
  ]) {
    const f = await fixture(() => Response.json({ detail }, { status }));
    assert.deepEqual(await f.call(base(initial)), { detail });
    assert.equal(f.calls.length, attempts);
    await f.engine.close();
  }
});

test('partial streams retain the confirmed parent without adopting partial output', async () => {
  let turn = 0;
  const f = await fixture(() => ++turn === 1 ? sse(response('r1')) : new Response('data: {"type":"response.output_text.delta","delta":"partial"}\n\n', { headers: { 'content-type': 'text/event-stream' } }));
  await f.call({ ...base(initial), stream: true });
  const history = [...initial, assistant('OK'), user('Next')];
  const partial = await f.call({ ...base(history), stream: true });
  assert.match(partial, /partial/); assert.equal(f.calls[1].body.previous_response_id, 'r1');
  await f.call(base(history)); assert.equal(f.calls[2].body.previous_response_id, 'r1');
});

test('reasoning storage is read back automatically while native and opaque replay stay intact', async () => {
  for (const address of [url, 'https://other.example/v2/responses']) {
    for (const thought of [
      { id: 'thinking-1', type: 'reasoning', summary: [{ type: 'summary_text', text: 'Keep this analysis.' }] },
      { id: 'thinking-1', type: 'reasoning', summary: [], encrypted_content: 'opaque-native-reasoning' },
    ]) {
      const f = await fixture((_body, n) => json(response('r' + n, [thought, assistant('OK')])));
      const first = await f.call(base(initial), meta, {}, address);
      const history = [...initial, ...first.output, user('Next')];
      await f.call(base(history), { ...meta, replayHistory: { ids: [thought.id] } }, {}, address);
      assert.equal(f.calls[1].body.previous_response_id, first.id);
      assert.deepEqual(f.calls[1].body.input, [user('Next')]);
      const edited = [user('Edited prefix'), ...history.slice(1)];
      await f.call(base(edited), { ...meta, replayHistory: { ids: [thought.id] } }, {}, address);
      assert.deepEqual(f.calls.at(-1).body.input, edited);
      assert.equal(f.engine.inspect().probeRequests, 0);
      await f.engine.close();
    }
  }
});

const thought = { id: 'thinking-1', type: 'reasoning', summary: [], content: [{ type: 'reasoning_text', text: 'Synthetic earlier analysis.' }] };
const omittedThoughts = value => json({ ...value, output: value.output.filter(item => item.type !== 'reasoning') });
const nativeScope = { ...meta, replayHistory: { ids: [thought.id], sample: { id: 'r1', reasoningIds: [thought.id] } } };

test('stored-history projection keeps continuation, invalid-ID fallback and background snapshots equivalent', async () => {
  const f = await fixture((body, n) => n === 4 && body.previous_response_id
    ? Response.json({ error: { message: 'previous_response_id not found' } }, { status: 400 })
    : sse(response('r' + n, n === 1 ? [thought, assistant('OK')] : [assistant('Done')])) , omittedThoughts);
  const original = structuredClone(initial);
  await f.call({ ...base(initial), stream: true });
  const mainInput = [...initial, thought, assistant('OK'), user('Next')];
  await f.call(base(mainInput), nativeScope);
  // The snapshot predates the latest answer, so it needs the still-valid parent.
  await f.call(base([...mainInput, user('Prepare this snapshot')]), nativeScope);
  assert.equal(f.calls[1].body.previous_response_id, 'r1');
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
  await f.call(base([...mainInput, user('Try another branch')]), nativeScope);
  assert.equal(f.calls.length, 5);
  assert.equal(f.calls[4].body.previous_response_id, undefined);
  assert.deepEqual(f.calls[4].body.input, [...mainInput.filter(item => item !== thought), user('Try another branch')]);
  assert.deepEqual(initial, original);
  assert.equal(f.reads.length, 1);
  assert.equal(f.engine.inspect().probeRequests, 0);
});

test('cold resume and changed image history use stored representation without deleting unknown or encrypted reasoning', async () => {
  const f = await fixture((_body, n) => json(response('r' + n, [thought, assistant('OK')])), omittedThoughts);
  await f.call(base(initial)); await f.engine.close();
  const fresh = await createResponsesAuto({ fetchImpl: f.send });
  const unknown = { ...thought, id: 'not-native-history' };
  const encrypted = { id: 'opaque', type: 'reasoning', summary: [], encrypted_content: 'signed-state' };
  const input = [user('Image offloaded; original reference retained'), thought, unknown, encrypted, assistant('OK'), user('Continue')];
  const body = base(input), before = structuredClone(body);
  await (await fresh.fetchFor(nativeScope)(url, { method: 'POST', headers: { authorization: 'Bearer fixture-a', 'content-length': String(Buffer.byteLength(JSON.stringify(body))) }, body: JSON.stringify(body) })).text();
  assert.deepEqual(f.calls.at(-1).body.input, input.filter(item => item !== thought));
  assert.equal(f.calls.at(-1).headers.get('content-length'), null);
  assert.deepEqual(body, before);
  await fresh.close();
});

test('unavailable or mismatched readback cannot authorize reasoning removal', async () => {
  for (const readback of [() => Response.json({}, { status: 404 }), value => json({ ...value, id: 'wrong-response', output: [assistant('OK')] })]) {
    const f = await fixture((_body, n) => json(response('r' + n, [thought, assistant('OK')])), readback);
    await f.call(base(initial));
    const input = [user('Changed'), ...initial.slice(1), thought, assistant('OK'), user('Next')];
    await f.call(base(input), nativeScope);
    assert.deepEqual(f.calls.at(-1).body.input, input);
    assert.equal(f.reads.length, 1);
  }
});

test('native reasoning metadata is scoped to the actual model, API and successful assistant', () => {
  const message = { role: 'assistant', content: [{ type: 'reasoning', text: 'Analysis' }, { type: 'text', text: 'OK' }], source: { provider: meta.provider, model, replayState: {
    response: { kind: 'pi-ai', version: 2, api: 'openai-responses', provider: meta.provider, model, responseId: 'r1', stopReason: 'stop' },
    blocks: [{ type: 'reasoning', thinkingSignature: JSON.stringify(thought) }, { type: 'text' }],
  } } };
  assert.deepEqual(nativeReplayHistory({ ...meta, messages: [message] }), nativeScope.replayHistory);
  assert.deepEqual(nativeReplayHistory({ ...meta, model: 'other', messages: [message] }).ids, []);
  assert.deepEqual(nativeReplayHistory({ ...meta, messages: [{ ...message, role: 'user' }] }).ids, []);
  assert.deepEqual(nativeReplayHistory({ ...meta, messages: [{ ...message, content: [] }] }).ids, []);
  assert.deepEqual(nativeReplayHistory({ ...meta, messages: null }).ids, []);
});

test('transport failures preserve the last confirmed response for an unchanged retry', async () => {
  const f = await fixture((_body, n) => { if (n === 2) throw Error('connection lost'); return json(response('r' + n)); });
  const a = await f.call(base(initial)), history = [...initial, ...a.output, user('Continue')];
  await assert.rejects(f.call(base(history)), /connection lost/);
  await f.call(base(history));
  assert.equal(f.calls[1].body.previous_response_id, 'r1');
  assert.deepEqual(f.calls[2].body, f.calls[1].body);
});

test('changed assistant fields cannot be mistaken for an identical history prefix', async () => {
  const output = [{ ...assistant('OK'), name: 'original-speaker', tool_calls: [{ id: 'call-a' }] }];
  for (const patch of [{ name: 'other-speaker' }, { tool_calls: [{ id: 'call-b' }] }]) {
    const f = await fixture(() => json(response('r1', output)));
    await f.call(base(initial));
    await f.call(base([...initial, { ...output[0], ...patch }, user('Next')]));
    assert.equal(f.calls[1].body.previous_response_id, undefined);
  }
});

test('concurrent branches reuse their immutable parent and retain matching completed histories', async () => {
  let release;
  const f = await fixture((body, n) => n === 2 ? new Promise(resolve => { release = () => resolve(json(response('old'))); }) : json(response('r' + n)));
  const a = await f.call(base(initial)); const history = [...initial, ...a.output];
  const first = f.call(base([...history, user('branch-a')]));
  while (!release) await new Promise(resolve => setImmediate(resolve));
  const b = await f.call(base([...history, user('branch-b')]));
  assert.equal(f.calls[2].body.previous_response_id, 'r1');
  release(); await first;
  await f.call(base([...history, user('branch-b'), ...b.output, user('Continue b')]));
  assert.equal(f.calls.at(-1).body.previous_response_id, 'r3');
  await f.call(base([...history, user('branch-a'), assistant('OK'), user('Continue a')]));
  assert.equal(f.calls.at(-1).body.previous_response_id, 'old');
});

test('a newer runtime replaces an older shared owner and survives until the last owner leaves', async t => {
  const ctx = new Context(), events = [];
  ctx.reflect.provide('llm', {}); t.after(() => ctx.fiber.dispose());
  const owner = revision => ({ name: 'owner-' + revision, inject, apply: child => acquireShared(child, 'upgrade-test', {
    name: 'runtime-' + revision, runtimeRevision: revision,
    apply: runtime => runtime.effect(() => { events.push('start-' + revision); return () => events.push('stop-' + revision); }),
  }) });
  const first = ctx.plugin(owner(1)); await first;
  const state = [...ctx.root[Symbol.for('dsh-response-id-reuse.upgrade-test.v1')].values()][0];
  delete state.runtimeRevision; delete state.fiberRevision; // legacy owner state
  const second = ctx.plugin(owner(2)); await second;
  assert.deepEqual(events, ['start-1', 'stop-1', 'start-2']);
  await second.dispose(); assert.deepEqual(events, ['start-1', 'stop-1', 'start-2']);
  await first.dispose(); assert.deepEqual(events, ['start-1', 'stop-1', 'start-2', 'stop-2']);
});

test('actual Cordis middleware: shared owners, request scoping and uninstall restoration', async t => {
  const original = globalThis.fetch, f = await fixture();
  globalThis.fetch = f.send;
  const ctx = new Context(); ctx.reflect.provide('llm', {});
  t.after(async () => { await ctx.fiber.dispose(); globalThis.fetch = original; });
  const one = ctx.plugin({ name: 'owner-one', inject, apply }); await one;
  const installed = globalThis.fetch;
  const two = ctx.plugin({ name: 'owner-two', inject, apply }); await two;
  assert.equal(globalThis.fetch, installed);
  const invoke = async (input, sessionId = 'native') => {
    let result;
    const next = async function* () {
      const r = await globalThis.fetch(url, { method: 'POST', headers: { authorization: 'Bearer fixture-a' }, body: JSON.stringify(base(input)) });
      result = await r.json(); yield { type: 'finish', reason: { kind: 'stop' } };
    };
    for await (const chunk of ctx.waterfall(ctx.llm, 'llm/stream', { provider: 'fixture', model, sessionId }, next)) assert.equal(chunk.reason.kind, 'stop');
    return result;
  };
  const a = await invoke(initial);
  await invoke([...initial, ...a.output, user('Next')]);
  assert.equal(f.calls[1].body.previous_response_id, 'r1');
  await globalThis.fetch(url, { method: 'POST', body: JSON.stringify(base(initial)) });
  assert.equal(f.calls[2].body.store, false);
  await one.dispose(); assert.equal(globalThis.fetch, installed);
  await two.dispose(); assert.equal(globalThis.fetch, f.send);
  const reload = ctx.plugin({ name: 'owner-three', inject, apply }); await reload;
  await invoke([...initial, ...a.output, user('Next')]);
  assert.equal(f.calls.at(-1).body.previous_response_id, undefined);
  await reload.dispose();
});
