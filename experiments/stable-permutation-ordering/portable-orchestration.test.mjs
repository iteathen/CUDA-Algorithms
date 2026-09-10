import test from 'node:test';
import assert from 'node:assert/strict';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { closePreparedTwoWordOrder, prepareTwoWordStableOrderU32 } from './execution-plan.mjs';

async function allocateU32(runtime, count, access) {
  const memory = await runtime.allocateDevice({ byteLength: count * 4 });
  const view = await memory.view({ dtype: 'u32', elementCount: count, access });
  return { memory, view };
}

async function closePair(pair) {
  await pair.view.close();
  await pair.memory.close();
}

test('public CUDA-JS mock accepts two stable key-order passes as one prepared DAG', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  let slice;
  let operation;
  const allocations = [];

  try {
    slice = await prepareTwoWordStableOrderU32(runtime, { recordCapacity: 32, indexCapacity: 16, blockSize: 8 });
    assert.equal(slice.prepared.nodeCount, 3);
    assert.equal(slice.prepared.edgeCount, 2);
    assert.equal(slice.prepared.realization, 'semantic-single-stream');
    assert.deepEqual(
      slice.prepared.bindings.map((entry) => entry.name).sort(),
      ['activeCount', 'highKeys', 'indicesA', 'indicesB', 'lowKeys', 'status'],
    );

    const lowKeys = await allocateU32(runtime, 32, 'read');
    const highKeys = await allocateU32(runtime, 32, 'read');
    const indicesA = await allocateU32(runtime, 16, 'read-write');
    const indicesB = await allocateU32(runtime, 16, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(lowKeys, highKeys, indicesA, indicesB, activeCount, status);

    operation = await slice.prepared.submit({
      bindings: {
        lowKeys: lowKeys.view,
        highKeys: highKeys.view,
        indicesA: indicesA.view,
        indicesB: indicesB.view,
        activeCount: activeCount.view,
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
    if (slice) await closePreparedTwoWordOrder(slice);
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closePair(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
    assert.equal(closed.restartRequired, false);
  }
});
