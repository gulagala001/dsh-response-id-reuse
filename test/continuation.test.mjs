import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Context } from '@deepseek-ai/cordis';
import { createResponsesAuto } from '../src/engine.mjs';
import { apply, inject } from '../src/index.mjs';

const url = 'https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1/responses';
const model = 'deepseek-v4.1-flash-1017';
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
  assert.equal(f.calls[0].headers.get('x-dashscope-session-cache'), 'enable');
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
    [{ ...base(history), model: 'qwen3.8-flash' }, { ...meta, model: 'qwen3.8-flash' }, {}],
  ]) { await f.call(body, scope, headers); assert.equal(f.calls.at(-1).body.previous_response_id, undefined); }
});

test('unknown endpoints, no session and ZDR are exact pass-through, including Request objects', async () => {
  const f = await fixture();
  for (const [address, scope, headers] of [
    ['https://unverified.example/v1/responses', meta, {}],
    [url, { ...meta, sessionId: undefined }, {}],
    [url, meta, { 'x-cmd-zdr': '1' }],
  ]) {
    const body = { ...base(initial) }; if (!scope.sessionId) delete body.prompt_cache_key;
    const req = new Request(address, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
    await (await f.engine.fetchFor(scope)(req)).json();
    assert.deepEqual(f.calls.at(-1).body, body);
  }
  assert.equal(f.calls.length, 3);
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

test('explicit invalid ID is retried once with complete history and then remembered', async () => {
  const f = await fixture((body, n) => body.previous_response_id
    ? new Response(JSON.stringify({ error: { message: 'previous_response_id not found', code: 'invalid_request_error' } }), { status: 400 })
    : json(response('r' + n)));
  const a = await f.call(base(initial)), history = [...initial, ...a.output, user('Continue')];
  await f.call(base(history));
  assert.equal(f.calls.length, 3); assert.deepEqual(f.calls[2].body, base(history));
  await f.call(base(history)); assert.equal(f.calls.length, 4); assert.equal(f.calls[3].body.store, false);
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

test('partial streams do not advance continuation and keep their original output', async () => {
  let turn = 0;
  const f = await fixture(() => ++turn === 1 ? sse(response('r1')) : new Response('data: {"type":"response.output_text.delta","delta":"partial"}\n\n', { headers: { 'content-type': 'text/event-stream' } }));
  await f.call({ ...base(initial), stream: true });
  const history = [...initial, assistant('OK'), user('Next')];
  const partial = await f.call({ ...base(history), stream: true });
  assert.match(partial, /partial/); assert.equal(f.calls[1].body.previous_response_id, 'r1');
  await f.call(base(history)); assert.equal(f.calls[2].body.previous_response_id, undefined);
});

test('concurrent branches only commit the newest completed full-history request', async () => {
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
