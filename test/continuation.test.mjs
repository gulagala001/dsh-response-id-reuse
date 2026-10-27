import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Context } from '@deepseek-ai/cordis';
import { createResponsesAuto } from '../src/engine.mjs';
import { apply, inject } from '../src/index.mjs';
import { acquireShared } from '../src/shared.mjs';

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
async function fixture(handler) {
  const calls = [];
  const send = async (input, init) => {
    const r = new Request(input, init), body = JSON.parse(await r.text());
    calls.push({ url: r.url, body, headers: r.headers });
    return handler ? handler(body, calls.length, r) : json(response('r' + calls.length));
  };
  const engine = await createResponsesAuto({ fetchImpl: send });
  const call = async (body, scope = meta, headers = {}, address = url) => {
    const result = await engine.fetchFor(scope)(address, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer fixture-a', ...headers }, body: JSON.stringify(body) });
    return result.headers.get('content-type')?.includes('text/event-stream') ? result.text() : result.json();
  };
  return { engine, calls, call, send };
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

test('partial streams do not advance continuation and keep their original output', async () => {
  let turn = 0;
  const f = await fixture(() => ++turn === 1 ? sse(response('r1')) : new Response('data: {"type":"response.output_text.delta","delta":"partial"}\n\n', { headers: { 'content-type': 'text/event-stream' } }));
  await f.call({ ...base(initial), stream: true });
  const history = [...initial, assistant('OK'), user('Next')];
  const partial = await f.call({ ...base(history), stream: true });
  assert.match(partial, /partial/); assert.equal(f.calls[1].body.previous_response_id, 'r1');
  await f.call(base(history)); assert.equal(f.calls[2].body.previous_response_id, undefined);
});

test('concurrent branches reserve an ID once and retain matching completed histories', async () => {
  let release;
  const f = await fixture((body, n) => n === 2 ? new Promise(resolve => { release = () => resolve(json(response('old'))); }) : json(response('r' + n)));
  const a = await f.call(base(initial)); const history = [...initial, ...a.output];
  const first = f.call(base([...history, user('branch-a')]));
  while (!release) await new Promise(resolve => setImmediate(resolve));
  const b = await f.call(base([...history, user('branch-b')]));
  assert.equal(f.calls[2].body.previous_response_id, undefined);
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
