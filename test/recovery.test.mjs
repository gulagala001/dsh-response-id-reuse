import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createResponsesAuto } from '../src/engine.mjs';

const model = 'synthetic-streaming-model';
const first = [{ role: 'user', content: 'synthetic input' }];
const answer = [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'OK' }] }];
const history = [...first, ...answer, { role: 'user', content: 'next' }];
const completed = id => ({ type: 'response.completed', response: { id, status: 'completed', store: true, output: answer } });
const event = (res, value) => res.write('data: ' + JSON.stringify(value) + '\n\n');
const reply = (res, id) => { res.writeHead(200, { 'content-type': 'text/event-stream' }); event(res, completed(id)); res.end(); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function local(t, handler, options = {}) {
  const calls = [];
  const server = createServer(async (req, res) => {
    let raw = ''; for await (const part of req) raw += part;
    const body = JSON.parse(raw); calls.push(body);
    if (calls.length === 1) reply(res, 'parent');
    else await handler(body, res, calls.length);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const engine = await createResponsesAuto({ recovery: { minWaitMs: 45, defaultWaitMs: 45, maxWaitMs: 45 }, ...options });
  t.after(async () => { await engine.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  const send = (input = history, extra = {}, scope = {}) => engine.fetchFor({ provider: 'fixture', model, sessionId: 'one', ...scope })(
    `http://127.0.0.1:${server.address().port}/responses`, {
      method: 'POST', headers: { authorization: 'Bearer synthetic' },
      body: JSON.stringify({ model, stream: true, input }), ...extra,
    });
  await (await send(first)).text();
  return { calls, engine, send };
}

for (const metadata of [false, true]) test(`stalled ${metadata ? 'SSE metadata' : 'response headers'} aborts and replays once without exposing abandoned state`, async t => {
  let closed = false;
  const f = await local(t, (body, res, n) => {
    if (!body.previous_response_id) return reply(res, 'recovered-' + n);
    res.on('close', () => { closed = true; });
    if (metadata) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      event(res, { type: 'response.created', response: { id: 'abandoned', status: 'in_progress' } });
      event(res, { type: 'response.output_item.added', item: { id: 'empty-message', type: 'message', content: [] } });
      res.write(': heartbeat\n\n');
    }
  });
  const start = performance.now();
  const text = await (await f.send()).text();
  assert.ok(performance.now() - start < 800);
  assert.match(text, /recovered/); assert.doesNotMatch(text, /abandoned/);
  assert.equal(f.calls.length, 3); assert.equal(f.calls[1].previous_response_id, 'parent');
  assert.equal(f.calls[2].previous_response_id, undefined); assert.deepEqual(f.calls[2].input, history);
  assert.equal(f.engine.inspect().last.fallback, 'first-output-timeout');
  await pause(5); assert.equal(closed, true);
  await (await f.send()).text();
  assert.equal(f.calls.length, 4); assert.equal(f.calls[3].previous_response_id, undefined);
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'cooling');
});

test('HTTP 5xx falls back on headers without waiting for a stalled error body', async t => {
  const f = await local(t, (body, res, n) => {
    if (!body.previous_response_id) return reply(res, 'full-' + n);
    res.writeHead(503, { 'content-type': 'application/json' }); res.flushHeaders(); res.write('{');
  });
  const start = performance.now();
  assert.match(await (await f.send()).text(), /full-/);
  assert.ok(performance.now() - start < 500);
  assert.equal(f.calls.length, 3); assert.equal(f.engine.inspect().last.fallback, 'server-error');
});

test('real first output disables the first-output timer; long thinking and partial failure are not replayed', async t => {
  const f = await local(t, (_body, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    event(res, { type: 'response.reasoning_text.delta', delta: 'Thinking normally' });
    const timer = setTimeout(() => { event(res, { type: 'response.failed', response: { id: 'partial', status: 'failed' } }); res.end(); }, 130);
    res.on('close', () => clearTimeout(timer));
  });
  const start = performance.now(), response = await f.send();
  const reader = response.body.getReader(), firstChunk = await reader.read();
  assert.match(new TextDecoder().decode(firstChunk.value), /Thinking normally/);
  assert.ok(performance.now() - start < 100);
  while (!(await reader.read()).done) {}
  assert.ok(performance.now() - start >= 100);
  assert.equal(f.calls.length, 2); assert.equal(f.engine.inspect().fallbacks, 0);
  assert.equal(f.engine.inspect().activeChains, 1);
});

test('a stalled full replay stops after the one allowed fallback', async t => {
  const f = await local(t, () => {});
  await assert.rejects(f.send(), { name: 'FirstOutputTimeout' });
  assert.equal(f.calls.length, 3); assert.equal(f.engine.inspect().fallbacks, 1);
  assert.equal(f.engine.inspect().activeChains, 1);
});

test('a failed replay with a stalled error body also finishes promptly', async t => {
  const f = await local(t, (_body, res) => {
    res.writeHead(500, { 'content-type': 'application/json' }); res.flushHeaders(); res.write('{');
  }, { recovery: { minWaitMs: 45, defaultWaitMs: 45, maxWaitMs: 45, errorBodyWaitMs: 40 } });
  const response = await f.send();
  assert.equal(response.status, 500);
  await assert.rejects(response.text(), /error body did not finish/);
  assert.equal(f.calls.length, 3);
});

test('caller cancellation during the first-output wait never triggers replay', async t => {
  const controller = new AbortController();
  const f = await local(t, (_body, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    event(res, { type: 'response.created', response: { id: 'waiting' } });
    controller.abort();
  });
  await assert.rejects(f.send(history, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(f.calls.length, 2); assert.equal(f.engine.inspect().fallbacks, 0);
});

test('cooldown admits one real continuation probe; full responses and concurrent calls cannot close the circuit', async t => {
  let clock = 1000, finishProbe, signalProbe;
  const probeStarted = new Promise(resolve => { signalProbe = resolve; });
  const f = await local(t, (body, res, n) => {
    if (n === 2) { res.writeHead(500); res.end(); return; }
    if (body.previous_response_id && !finishProbe) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      event(res, { type: 'response.output_text.delta', delta: 'probe output' });
      finishProbe = () => { event(res, completed('probe-result')); res.end(); }; signalProbe(); return;
    }
    reply(res, 'full-' + n);
  }, { now: () => clock });
  await (await f.send()).text();
  await (await f.send()).text();
  assert.equal(f.calls.at(-1).previous_response_id, undefined);
  clock += 60001;
  const probing = f.send().then(r => r.text()); await probeStarted;
  await (await f.send()).text();
  assert.equal(f.calls.at(-1).previous_response_id, undefined);
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'probing');
  finishProbe(); await probing;
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'healthy');
  await (await f.send()).text();
  assert.equal(f.calls.at(-1).previous_response_id, 'parent');
  assert.equal(f.engine.inspect().probeRequests, 0);
});

test('a late successful continuation cannot undo a newer failure cooldown', async t => {
  let finishOld, signalOld;
  const oldStarted = new Promise(resolve => { signalOld = resolve; });
  const f = await local(t, (body, res, n) => {
    if (n === 2) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      event(res, { type: 'response.output_text.delta', delta: 'already active' });
      finishOld = () => { event(res, completed('old-success')); res.end(); }; signalOld(); return;
    }
    if (body.previous_response_id) { res.writeHead(500); res.end(); }
    else reply(res, 'full-' + n);
  });
  const old = f.send().then(r => r.text()); await oldStarted;
  await (await f.send()).text();
  finishOld(); await old;
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'cooling');
  await (await f.send()).text();
  assert.equal(f.calls.at(-1).previous_response_id, undefined);
});

for (const terminal of ['response.failed', 'response.incomplete', 'error', 'eof', 'transport-error']) test(terminal + ' cools continuation without replaying delivered output', async t => {
  const f = await local(t, (body, res, n) => {
    if (!body.previous_response_id) return reply(res, 'full-' + n);
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    event(res, { type: 'response.output_text.delta', delta: 'already delivered' });
    if (terminal === 'transport-error') { const timer = setTimeout(() => res.destroy(), 10); res.on('close', () => clearTimeout(timer)); return; }
    if (terminal !== 'eof') event(res, { type: terminal, response: { status: 'failed' } });
    res.end();
  });
  const response = await f.send();
  if (terminal === 'transport-error') await assert.rejects(response.text()); else assert.match(await response.text(), /already delivered/);
  assert.equal(f.calls.length, 2); assert.equal(f.engine.inspect().fallbacks, 0);
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'cooling');
  await (await f.send()).text(); assert.equal(f.calls.at(-1).previous_response_id, undefined);
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'cooling', 'full success cannot clear a continuation failure');
});

test('a failed half-open stream renews cooldown and retains the confirmed parent', async t => {
  let clock = 1000;
  const f = await local(t, (body, res, n) => {
    if (n === 2) { res.writeHead(503); res.end(); return; }
    if (!body.previous_response_id) return reply(res, 'full-' + n);
    res.writeHead(200, { 'content-type': 'text/event-stream' }); event(res, { type: 'response.failed', response: { status: 'failed' } }); res.end();
  }, { now: () => clock });
  await (await f.send()).text(); clock += 60001;
  await (await f.send()).text();
  assert.equal(f.calls.at(-1).previous_response_id, 'parent');
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'cooling');
  assert.equal(f.engine.inspect().continuationHealth[0].retryAfterMs, 120000);
});

test('cancelling a consumed response body does not mark the route unhealthy', async t => {
  const f = await local(t, (_body, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' }); event(res, { type: 'response.output_text.delta', delta: 'partial' });
  });
  const response = await f.send(); await response.body.cancel();
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'healthy');
  assert.equal(f.engine.inspect().fallbacks, 0);
});

test('multi-line terminal events and EOF framing preserve successful continuation', async t => {
  const f = await local(t, (_body, res, n) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(JSON.stringify(completed('multiline-' + n), null, 2).split('\n').map(line => 'data: ' + line).join('\n'));
  });
  await (await f.send()).text();
  assert.equal(f.engine.inspect().continuationHealth[0].state, 'healthy');
  assert.equal(f.engine.inspect().last.committed, true);
});

test('probe leases expire without allowing a late old receipt to alter the new lease',async () => {
  const {ContinuationHealth} = await import('../src/recovery.mjs');
  let time=0; const h=new ContinuationHealth({now:()=>time,cooldownMs:10,maxWaitMs:50});
  const route=h.route('old',{});
  h.fail(h.enter(route));
  for(let i=0;i<520;i++) h.route('route-'+i,{});
  assert.equal(h.route('old',{}),route); assert.equal(h.enter(route),null);
  time=11;const abandoned=h.enter(route);
  time+=51;const next=h.enter(route);assert.ok(next); assert.equal(h.enter(route),null);
  h.fail(abandoned);h.success(abandoned);h.release(abandoned);
  assert.equal(route.probe,next);assert.equal(h.routes.size,512);
});
test('a full SSE response does not supply a continuation latency sample',async t=>{
  const f=await local(t,(_body,res)=>reply(res,'next'),{recovery:{minWaitMs:1000,defaultWaitMs:25000,maxWaitMs:60000}});
  assert.equal(f.engine.inspect().continuationHealth[0].firstOutputBudgetMs,25000);
  await (await f.send()).text();
  assert.ok(f.engine.inspect().continuationHealth[0].firstOutputBudgetMs<25000);
});
