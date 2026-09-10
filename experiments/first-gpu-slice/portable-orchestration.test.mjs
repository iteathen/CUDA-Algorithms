import test from 'node:test';
import assert from 'node:assert/strict';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { closePreparedStableSelect, prepareStableSelectU32 } from './execution-plan.mjs';

async function allocateU32(runtime, count, access) {
  const memory = await runtime.allocateDevice({ byteLength: count * 4 });
  const view = await memory.view({ dtype: 'u32', elementCount: count, access });
  return { memory, view };
}

async function closePair(pair) {
  await pair.view.close();
  await pair.memory.close();
}

test('public CUDA-JS mock accepts one prepared reset -> scan -> select operation', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  let slice;
  let operation;
  const allocations = [];

  try {
    slice = await prepareStableSelectU32(runtime, { inputCapacity: 16, outputCapacity: 16, blockSize: 8 });

    assert.equal(slice.prepared.nodeCount, 3);
    assert.equal(slice.prepared.edgeCount, 2);
    assert.equal(slice.prepared.realization, 'semantic-single-stream');
    assert.deepEqual(
      slice.prepared.bindings.map((entry) => entry.name).sort(),
      ['activeCount', 'flags', 'outputCount', 'outputIndices', 'prefix', 'status'],
    );

    const flags = await allocateU32(runtime, 16, 'read');
    const prefix = await allocateU32(runtime, 16, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, 16, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(flags, prefix, activeCount, outputIndices, outputCount, status);

    operation = await slice.prepared.submit({
      bindings: {
        flags: flags.view,
        prefix: prefix.view,
        activeCount: activeCount.view,
        outputIndices: outputIndices.view,
        outputCount: outputCount.view,
        status: status.view,
      },
    });

    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.kind, 'prepared-batch');
    assert.equal(terminal.nodeCount, 3);
    assert.equal(terminal.edgeCount, 2);
  } finally {
    if (operation) await operation.close();
    if (slice) await closePreparedStableSelect(slice);
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closePair(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
    assert.equal(closed.restartRequired, false);
  }
});

test('prepared submission rejects forbidden same-buffer flags/prefix aliasing before execution', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  let slice;
  let returnedOperation;
  const allocations = [];
  let rejection = null;

  try {
    slice = await prepareStableSelectU32(runtime, { inputCapacity: 8, outputCapacity: 8, blockSize: 8 });
    const shared = await allocateU32(runtime, 8, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, 8, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(shared, activeCount, outputIndices, outputCount, status);

    try {
      returnedOperation = await slice.prepared.submit({
        bindings: {
          flags: shared.view,
          prefix: shared.view,
          activeCount: activeCount.view,
          outputIndices: outputIndices.view,
          outputCount: outputCount.view,
          status: status.view,
        },
      });
    } catch (error) {
      rejection = error;
    }

    if (returnedOperation) {
      await returnedOperation.wait();
      await returnedOperation.close();
      returnedOperation = null;
    }

    assert.ok(rejection, 'CUDA-JS prepared submission accepted a same-view read/write alias inside one algorithm node');
  } finally {
    if (returnedOperation) {
      await returnedOperation.wait();
      await returnedOperation.close();
    }
    if (slice) await closePreparedStableSelect(slice);
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closePair(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});

test('experimental plan rejects accidental zero/oversized physical launch bounds before CUDA work', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    await assert.rejects(() => prepareStableSelectU32(runtime, { inputCapacity: 0, outputCapacity: 1 }), /inputCapacity must be a positive safe integer/);
    await assert.rejects(() => prepareStableSelectU32(runtime, { inputCapacity: 1, outputCapacity: 0 }), /outputCapacity must be a positive safe integer/);
    await assert.rejects(() => prepareStableSelectU32(runtime, { inputCapacity: 1, outputCapacity: 1, blockSize: 1025 }), /blockSize exceeds CUDA architectural thread-block ceiling/);
  } finally {
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});
