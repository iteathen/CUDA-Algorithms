import test from 'node:test';
import assert from 'node:assert/strict';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { createCheckedExclusiveScanU32Plan, createSegmentOffsetsU32Plan,
  checkedScanU32Requirements, segmentOffsetsU32Requirements } from 'cuda-algorithms';
import { allocateBindings } from './checked-scan-fixtures.mjs';

for (const segments of [false, true]) test(`${segments ? 'segment offsets' : 'checked scan'} public bindings, aliasing, access, lifetime and cross-runtime rejection`, async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }); const other = await openCudaRuntimeForTesting({ compiler: true });
  const options = { inputCapacity: 65, groupCapacity: 33, blockSize: 64 };
  const req = segments ? segmentOffsetsU32Requirements(options) : checkedScanU32Requirements(options);
  let plan; let b; let foreign;
  const inputName = segments ? 'heads' : 'input'; const outputName = segments ? 'groupIds' : 'prefix';
  let backing; const views = [];
  try {
    plan = await (segments ? createSegmentOffsetsU32Plan : createCheckedExclusiveScanU32Plan)(runtime, options);
    b = await allocateBindings(runtime, req); foreign = await allocateBindings(other, req);
    assert.equal(plan.workspace.deviceBytes, req.workspaceBytes);
    await assert.rejects(plan.submit({ ...b.bindings, [outputName]: b.bindings[inputName] }), /access/);
    await assert.rejects(plan.submit({ ...b.bindings, upstreamStatus: b.bindings.status }), /same-range/);
    await assert.rejects(plan.submit({ ...b.bindings, [inputName]: foreign.bindings[inputName] }));
    backing = await runtime.allocateDevice({ byteLength: 66 * 4 });
    const read = await backing.view({ dtype: 'u32', elementCount: 65, access: 'read' }); views.push(read);
    const write = await backing.view({ dtype: 'u32', byteOffset: 4, elementCount: 65, access: 'write' }); views.push(write);
    await assert.rejects(plan.submit({ ...b.bindings, [inputName]: read, [outputName]: write }), /overlap/);
    const wrongType = await backing.view({ dtype: 'i32', elementCount: 65, access: 'read' }); views.push(wrongType);
    await assert.rejects(plan.submit({ ...b.bindings, [inputName]: wrongType }), /dtype/);
    const short = await backing.view({ dtype: 'u32', elementCount: 1, access: 'read' }); views.push(short);
    await assert.rejects(plan.submit({ ...b.bindings, [inputName]: short }), /element range/);
    await read.close();
    await assert.rejects(plan.submit({ ...b.bindings, [inputName]: read }));
    const submitting = plan.submit({ ...b.bindings, upstreamStatus: b.bindings.activeCount });
    await assert.rejects(plan.submit(b.bindings), /in-flight/); await assert.rejects(plan.close(), /in-flight/);
    const op = await submitting; assert.equal(op.kind, 'operation');
    assert.equal((await op.wait()).status, 'completed'); await op.close();
    await plan.close(); await plan.close(); await assert.rejects(plan.submit(b.bindings), /closed/);
  } finally {
    if (plan) await plan.close(); for (const v of views.reverse()) await v.close(); if (backing) await backing.close();
    if (b) await b.close(); if (foreign) await foreign.close(); assert.equal((await runtime.close()).graceful, true); assert.equal((await other.close()).graceful, true);
  }
});

test('bounds and workspace budgets reject before lower compilation', async () => {
  for (const options of [{ inputCapacity: 0 }, { inputCapacity: 262145 }, { inputCapacity: 2, blockSize: 96 },
    { inputCapacity: 65, maxWorkspaceBytes: 0 }, { inputCapacity: 65, maxWorkspaceBytes: NaN }]) {
    await assert.rejects(createCheckedExclusiveScanU32Plan({}, options), RangeError);
  }
  await assert.rejects(createSegmentOffsetsU32Plan({}, { inputCapacity: 65, groupCapacity: 66 }), RangeError);
  for (const n of [1, 65, 8192, 262144]) for (const blockSize of [64, 128, 256]) {
    const r = segmentOffsetsU32Requirements({ inputCapacity: n, groupCapacity: n, blockSize });
    assert(r.workspaceBytes < 8 * 1024 ** 2); assert(r.preparedNodeCount <= 32);
    assert.equal(segmentOffsetsU32Requirements({ inputCapacity: n, groupCapacity: n, blockSize, maxWorkspaceBytes: r.workspaceBytes }).workspaceBytes, r.workspaceBytes);
  }
});

test('partial construction releases allocated resources and preserves primary failure', async t => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }); let calls = 0;
  const allocate = runtime.allocateDevice.bind(runtime);
  t.mock.method(Object.getPrototypeOf(runtime), 'allocateDevice', async o => {
    if (++calls === 3) throw new Error('injected allocation failure'); return allocate(o);
  });
  try { await assert.rejects(createCheckedExclusiveScanU32Plan(runtime, { inputCapacity: 65 }), /injected allocation failure/); }
  finally { assert.equal((await runtime.close()).graceful, true); }
});

test('cleanup failure propagates consistently after releasing remaining resources', async t => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }); let first = true;
  const allocate = runtime.allocateDevice.bind(runtime);
  t.mock.method(Object.getPrototypeOf(runtime), 'allocateDevice', async o => {
    const memory = await allocate(o); if (!first) return memory; first = false;
    const originalClose = memory.close;
    t.mock.method(Object.getPrototypeOf(memory), 'close', async function() {
      const result = await originalClose.call(this);
      if (this === memory) throw new Error('injected cleanup failure'); return result;
    });
    return memory;
  });
  try {
    const plan = await createCheckedExclusiveScanU32Plan(runtime, { inputCapacity: 65 });
    await assert.rejects(plan.close(), AggregateError); await assert.rejects(plan.close(), AggregateError);
  } finally { assert.equal((await runtime.close()).graceful, true); }
});
