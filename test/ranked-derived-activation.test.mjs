import test from 'node:test';
import assert from 'node:assert/strict';

import { compileDeviceLibrary } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { createRankedDerivedActivationU32Plan } from '../src/index.mjs';

async function allocateU32(runtime, count, access = 'read-write') {
  const memory = await runtime.allocateDevice({ byteLength: count * 4 });
  const view = await memory.view({ dtype: 'u32', elementCount: count, access });
  return { memory, view };
}

async function closeAllocation(allocation) {
  await allocation.view.close();
  await allocation.memory.close();
}

async function compileImplicitDagDerivation(runtime) {
  const compiled = await compileDeviceLibrary(runtime, {
    source: `
function deriveParent(sourceIndex, emissionLane) {
  if (sourceIndex === gpu.u32(0)) return gpu.u32(4294967295);
  if (emissionLane === gpu.u32(0)) return sourceIndex - gpu.u32(1);
  if (emissionLane === gpu.u32(1) && sourceIndex > gpu.u32(1)) return sourceIndex - gpu.u32(2);
  return gpu.u32(4294967295);
}
`,
    functions: [{
      name: 'deriveParent', kind: 'device', returns: 'u32',
      parameters: [
        { name: 'sourceIndex', type: 'u32' },
        { name: 'emissionLane', type: 'u32' },
      ],
    }],
    exports: ['deriveParent'],
    compile: { headerProfile: 'cuda-cccl' },
  });
  return compiled.library;
}

test('ranked derived activation composes an explicit typed consumer leaf library into one prepared device epoch', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const allocations = [];
  let plan;
  let operation;
  try {
    const library = await compileImplicitDagDerivation(runtime);
    plan = await createRankedDerivedActivationU32Plan(runtime, {
      itemCapacity: 16,
      inputCapacity: 8,
      outputCapacity: 16,
      maxEmissionsPerItem: 2,
      blockSize: 8,
      derivation: { library, name: 'deriveParent' },
    });

    assert.equal(plan.kind, 'cuda-algorithms-plan');
    assert.equal(plan.family, 'ranked-derived-activation');
    assert.equal(plan.derivation.librarySha256, library.sha256);
    assert.equal(plan.derivation.exportName, 'deriveParent');
    assert.equal(plan.realization.preparedNodeCount, 4);
    assert.equal(plan.realization.progressionOwner, 'device');
    assert.equal(plan.invalidTargetIndex, 0xffff_ffff);

    const activeIndices = await allocateU32(runtime, 8, 'read');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const ranks = await allocateU32(runtime, 16, 'read');
    const nextFlags = await allocateU32(runtime, 16, 'read-write');
    const prefix = await allocateU32(runtime, 16, 'read-write');
    const outputIndices = await allocateU32(runtime, 16, 'write');
    const nextCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(activeIndices, activeCount, ranks, nextFlags, prefix, outputIndices, nextCount, status);

    operation = await plan.submit({
      activeIndices: activeIndices.view,
      activeCount: activeCount.view,
      ranks: ranks.view,
      nextFlags: nextFlags.view,
      prefix: prefix.view,
      outputIndices: outputIndices.view,
      nextCount: nextCount.view,
      status: status.view,
    });
    assert.equal(operation.kind, 'operation');
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed');
    assert.equal(terminal.kind, 'prepared-batch');
    assert.equal(terminal.nodeCount, 4);
  } finally {
    if (operation) await operation.close();
    if (plan) await plan.close();
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});

test('ranked derived activation rejects consumer exports that do not match the finite index/lane contract', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    const wrong = await compileDeviceLibrary(runtime, {
      source: 'function wrong(sourceIndex) { return sourceIndex; }',
      functions: [{
        name: 'wrong', kind: 'device', returns: 'u32',
        parameters: [{ name: 'sourceIndex', type: 'u32' }],
      }],
      exports: ['wrong'],
      compile: { headerProfile: 'cuda-cccl' },
    });

    await assert.rejects(
      () => createRankedDerivedActivationU32Plan(runtime, {
        itemCapacity: 4,
        maxEmissionsPerItem: 1,
        derivation: { library: wrong.library, name: 'wrong' },
      }),
      /exact signature \(u32 sourceIndex, u32 emissionLane\) -> u32 targetIndex/,
    );
  } finally {
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
});
