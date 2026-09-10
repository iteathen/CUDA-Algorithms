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

async function closeAll(runtime, plan, operation, allocations) {
  if (operation) await operation.close();
  if (plan) await plan.close();
  for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
  const closed = await runtime.close();
  assert.equal(closed.graceful, true);
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
    assert.equal(plan.aliasing.relationOwner, 'cuda-js:inspectDeviceViewRelation');

    const flags = await allocateU32(runtime, 16, 'read');
    const prefix = await allocateU32(runtime, 16, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, 16, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(flags, prefix, activeCount, outputIndices, outputCount, status);

    operation = await plan.submit({ flags: flags.view, prefix: prefix.view, activeCount: activeCount.view, outputIndices: outputIndices.view, outputCount: outputCount.view, status: status.view });
    assert.equal(operation.kind, 'operation');
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.kind, 'prepared-batch');
  } finally {
    await closeAll(runtime, plan, operation, allocations);
  }
});

test('candidate stable-select rejects same-range writes but permits read-read reuse', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const allocations = [];
  let plan;
  let operation;
  try {
    plan = await createStableSelectIndicesU32Plan(runtime, { inputCapacity: 8, outputCapacity: 8, blockSize: 8 });
    const sharedReadWrite = await allocateU32(runtime, 8, 'read-write');
    const sharedRead = await allocateU32(runtime, 8, 'read');
    const prefix = await allocateU32(runtime, 8, 'read-write');
    const outputIndices = await allocateU32(runtime, 8, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(sharedReadWrite, sharedRead, prefix, outputIndices, outputCount, status);

    await assert.rejects(
      () => plan.submit({ flags: sharedReadWrite.view, prefix: sharedReadWrite.view, activeCount: sharedRead.view, outputIndices: outputIndices.view, outputCount: outputCount.view, status: status.view }),
      /must be disjoint because at least one role writes; CUDA-JS reports same-range/,
    );

    operation = await plan.submit({ flags: sharedRead.view, prefix: prefix.view, activeCount: sharedRead.view, outputIndices: outputIndices.view, outputCount: outputCount.view, status: status.view });
    assert.equal((await operation.wait()).status, 'completed');
  } finally {
    await closeAll(runtime, plan, operation, allocations);
  }
});

test('candidate stable-select rejects overlapping sibling views through lower-owned range truth', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  let plan;
  const ordinary = [];
  let backing;
  let flags;
  let prefix;
  try {
    plan = await createStableSelectIndicesU32Plan(runtime, { inputCapacity: 8, outputCapacity: 8, blockSize: 8 });
    backing = await runtime.allocateDevice({ byteLength: 64 });
    flags = await backing.view({ dtype: 'u32', byteOffset: 0, elementCount: 8, access: 'read' });
    prefix = await backing.view({ dtype: 'u32', byteOffset: 16, elementCount: 8, access: 'read-write' });
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, 8, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    ordinary.push(activeCount, outputIndices, outputCount, status);

    await assert.rejects(
      () => plan.submit({ flags, prefix, activeCount: activeCount.view, outputIndices: outputIndices.view, outputCount: outputCount.view, status: status.view }),
      /must be disjoint because at least one role writes; CUDA-JS reports overlap/,
    );
  } finally {
    if (plan) await plan.close();
    for (let i = ordinary.length - 1; i >= 0; i -= 1) await closeAllocation(ordinary[i]);
    if (prefix) await prefix.close();
    if (flags) await flags.close();
    if (backing) await backing.close();
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
    plan = await createStableLexicographicOrderIndicesU32Plan(runtime, { recordCapacity: 16, indexCapacity: 16, keyWordCount: 3, blockSize: 8 });
    assert.equal(plan.family, 'stable-lexicographic-order-indices');
    assert.equal(plan.resultBinding, 'indicesB');
    assert.equal(plan.realizationBounds.maxKeyWordCount, 31);
    assert.equal(plan.aliasing.relationOwner, 'cuda-js:inspectDeviceViewRelation');

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

    operation = await plan.submit({ keyWords, indicesA: indicesA.view, indicesB: indicesB.view, activeCount: activeCount.view, status: status.view });
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.nodeCount, 4);
  } finally {
    await closeAll(runtime, plan, operation, allocations);
  }
});

test('candidate ordering rejects write-range conflicts but permits duplicate read-only key views', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const allocations = [];
  let plan;
  let operation;
  try {
    plan = await createStableLexicographicOrderIndicesU32Plan(runtime, { recordCapacity: 8, indexCapacity: 8, keyWordCount: 2, blockSize: 8 });
    const sharedReadWrite = await allocateU32(runtime, 8, 'read-write');
    const key = await allocateU32(runtime, 8, 'read');
    const indicesA = await allocateU32(runtime, 8, 'read-write');
    const indicesB = await allocateU32(runtime, 8, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(sharedReadWrite, key, indicesA, indicesB, activeCount, status);

    await assert.rejects(
      () => plan.submit({ keyWords: [sharedReadWrite.view, key.view], indicesA: sharedReadWrite.view, indicesB: indicesB.view, activeCount: activeCount.view, status: status.view }),
      /must be disjoint because at least one role writes; CUDA-JS reports same-range/,
    );
    await assert.rejects(
      () => plan.submit({ keyWords: [key.view, key.view], indicesA: indicesA.view, indicesB: indicesA.view, activeCount: activeCount.view, status: status.view }),
      /must be disjoint because at least one role writes; CUDA-JS reports same-range/,
    );

    operation = await plan.submit({ keyWords: [key.view, key.view], indicesA: indicesA.view, indicesB: indicesB.view, activeCount: activeCount.view, status: status.view });
    assert.equal((await operation.wait()).status, 'completed');
  } finally {
    await closeAll(runtime, plan, operation, allocations);
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
