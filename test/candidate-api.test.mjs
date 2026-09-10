import test from 'node:test';
import assert from 'node:assert/strict';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import {
  createStableLexicographicOrderIndicesU32Plan,
  createStableSelectIndicesU32Plan,
} from '../src/index.mjs';

async function allocateU32(runtime, count, access = 'read-write') {
  const memory = await runtime.allocateDevice({ byteLength: count * 4 });
  const view = await memory.view({ dtype: 'u32', elementCount: count, access });
  return { memory, view };
}

async function closeAllocation(allocation) {
  await allocation.view.close();
  await allocation.memory.close();
}

test('candidate stable-select plan submits one nonblocking CUDA-JS operation', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const allocations = [];
  let plan;
  let operation;
  try {
    plan = await createStableSelectIndicesU32Plan(runtime, { inputCapacity: 16, outputCapacity: 16, blockSize: 8 });
    assert.equal(plan.kind, 'cuda-algorithms-plan');
    assert.equal(plan.family, 'stable-select-indices');
    assert.equal(plan.workspace.prefixElements, 16);

    const flags = await allocateU32(runtime, 16, 'read');
    const prefix = await allocateU32(runtime, 16, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, 16, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(flags, prefix, activeCount, outputIndices, outputCount, status);

    operation = await plan.submit({
      flags: flags.view,
      prefix: prefix.view,
      activeCount: activeCount.view,
      outputIndices: outputIndices.view,
      outputCount: outputCount.view,
      status: status.view,
    });
    assert.equal(operation.kind, 'operation');
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.kind, 'prepared-batch');
  } finally {
    if (operation) await operation.close();
    if (plan) await plan.close();
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});

test('candidate lexicographic ordering plan composes arbitrary bounded key-word passes', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const allocations = [];
  let plan;
  let operation;
  try {
    plan = await createStableLexicographicOrderIndicesU32Plan(runtime, {
      recordCapacity: 16,
      indexCapacity: 16,
      keyWordCount: 3,
      blockSize: 8,
    });
    assert.equal(plan.family, 'stable-lexicographic-order-indices');
    assert.equal(plan.resultBinding, 'indicesB');
    assert.equal(plan.realizationBounds.maxKeyWordCount, 31);

    const keyWords = [];
    for (let i = 0; i < 3; i += 1) {
      const allocation = await allocateU32(runtime, 16, 'read');
      allocations.push(allocation);
      keyWords.push(allocation.view);
    }
    const indicesA = await allocateU32(runtime, 16, 'read-write');
    const indicesB = await allocateU32(runtime, 16, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(indicesA, indicesB, activeCount, status);

    operation = await plan.submit({
      keyWords,
      indicesA: indicesA.view,
      indicesB: indicesB.view,
      activeCount: activeCount.view,
      status: status.view,
    });
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.nodeCount, 4);
  } finally {
    if (operation) await operation.close();
    if (plan) await plan.close();
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});

test('candidate realization bounds fail closed before compilation', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    await assert.rejects(
      () => createStableLexicographicOrderIndicesU32Plan(runtime, { recordCapacity: 1, indexCapacity: 1, keyWordCount: 32 }),
      /prepared-DAG realization bound of 31 words/,
    );
  } finally {
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});
