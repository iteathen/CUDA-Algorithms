import assert from 'node:assert/strict';

import { compileDeviceLibrary, openCudaRuntime } from 'cuda-js';
import {
  RANKED_DERIVED_ACTIVATION_U32_STATUS as STATUS,
  createRankedDerivedActivationU32Plan,
} from '../../src/index.mjs';
import { runRankedDerivedActivationEpoch } from '../../reference/ranked-derived-activation.mjs';

const U32_BYTES = 4;

function encodeU32(values) {
  const bytes = new Uint8Array(values.length * U32_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`u32 value out of range at ${index}`);
    view.setUint32(index * U32_BYTES, value, true);
  }
  return bytes;
}

function decodeU32(bytes) {
  if (bytes.byteLength % U32_BYTES !== 0) throw new RangeError('u32 byte length is not aligned');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / U32_BYTES }, (_, index) => view.getUint32(index * U32_BYTES, true));
}

async function allocateU32(runtime, count, access = 'read-write') {
  const memory = await runtime.allocateDevice({ byteLength: count * U32_BYTES });
  const view = await memory.view({ dtype: 'u32', elementCount: count, access });
  return { memory, view, count };
}

async function writeU32(allocation, values) {
  if (values.length > allocation.count) throw new RangeError('fixture write exceeds allocation');
  await allocation.memory.write(encodeU32(values));
}

async function readU32(allocation, count = allocation.count) {
  if (!Number.isSafeInteger(count) || count < 0 || count > allocation.count) throw new RangeError('fixture read count is outside allocation');
  if (count === 0) return [];
  const result = await allocation.memory.read({ byteLength: count * U32_BYTES });
  return decodeU32(result.bytes);
}

async function closeAllocation(allocation) {
  await allocation.view.close();
  await allocation.memory.close();
}

async function compileDerivation(runtime, { name, source }) {
  const compiled = await compileDeviceLibrary(runtime, {
    source,
    functions: [{
      name,
      kind: 'device',
      returns: 'u32',
      parameters: [
        { name: 'sourceIndex', type: 'u32' },
        { name: 'emissionLane', type: 'u32' },
      ],
    }],
    exports: [name],
    compile: { headerProfile: 'cuda-cccl' },
  });
  return compiled.library;
}

const implicitDag = Object.freeze({
  name: 'implicit-dag',
  exportName: 'deriveParent',
  source: `
function deriveParent(sourceIndex, emissionLane) {
  if (sourceIndex === gpu.u32(0)) return gpu.u32(4294967295);
  if (emissionLane === gpu.u32(0)) return sourceIndex - gpu.u32(1);
  if (emissionLane === gpu.u32(1) && sourceIndex > gpu.u32(1)) return sourceIndex - gpu.u32(2);
  return gpu.u32(4294967295);
}
`,
  itemCapacity: 16,
  activeIndices: Object.freeze([7, 7, 6]),
  ranks: Object.freeze(Array.from({ length: 16 }, (_, index) => index)),
  maxEmissionsPerItem: 2,
  deriveTarget(sourceIndex, emissionLane) {
    if (sourceIndex === 0) return 0xffff_ffff;
    if (emissionLane === 0) return sourceIndex - 1;
    if (emissionLane === 1 && sourceIndex > 1) return sourceIndex - 2;
    return 0xffff_ffff;
  },
});

const stagedLineage = Object.freeze({
  name: 'staged-data-lineage',
  exportName: 'derivePriorStage',
  source: `
function derivePriorStage(sourceIndex, emissionLane) {
  const stageWidth = gpu.u32(16);
  const stage = sourceIndex / stageWidth;
  const local = sourceIndex % stageWidth;
  if (stage === gpu.u32(0)) return gpu.u32(4294967295);
  if (emissionLane === gpu.u32(0)) return sourceIndex - stageWidth;
  if (emissionLane === gpu.u32(1) && local > gpu.u32(0)) return sourceIndex - stageWidth - gpu.u32(1);
  if (emissionLane === gpu.u32(2) && local + gpu.u32(1) < stageWidth) return sourceIndex - stageWidth + gpu.u32(1);
  return gpu.u32(4294967295);
}
`,
  itemCapacity: 64,
  activeIndices: Object.freeze([50, 51, 50]),
  ranks: Object.freeze(Array.from({ length: 64 }, (_, index) => Math.floor(index / 16))),
  maxEmissionsPerItem: 3,
  deriveTarget(sourceIndex, emissionLane) {
    const stageWidth = 16;
    const stage = Math.floor(sourceIndex / stageWidth);
    const local = sourceIndex % stageWidth;
    if (stage === 0) return 0xffff_ffff;
    if (emissionLane === 0) return sourceIndex - stageWidth;
    if (emissionLane === 1 && local > 0) return sourceIndex - stageWidth - 1;
    if (emissionLane === 2 && local + 1 < stageWidth) return sourceIndex - stageWidth + 1;
    return 0xffff_ffff;
  },
});

async function runFixture(runtime, model, { outputCapacity = model.itemCapacity, ranks = model.ranks } = {}) {
  const library = await compileDerivation(runtime, { name: model.exportName, source: model.source });
  const plan = await createRankedDerivedActivationU32Plan(runtime, {
    itemCapacity: model.itemCapacity,
    inputCapacity: model.activeIndices.length,
    outputCapacity,
    maxEmissionsPerItem: model.maxEmissionsPerItem,
    blockSize: Math.min(64, model.itemCapacity),
    derivation: { library, name: model.exportName },
  });
  const allocations = [];
  let operation;

  try {
    const activeIndices = await allocateU32(runtime, model.activeIndices.length, 'read');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const rankAllocation = await allocateU32(runtime, model.itemCapacity, 'read');
    const nextFlags = await allocateU32(runtime, model.itemCapacity, 'read-write');
    const prefix = await allocateU32(runtime, model.itemCapacity, 'read-write');
    const outputIndices = await allocateU32(runtime, outputCapacity, 'write');
    const nextCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(activeIndices, activeCount, rankAllocation, nextFlags, prefix, outputIndices, nextCount, status);

    await writeU32(activeIndices, model.activeIndices);
    await writeU32(activeCount, [model.activeIndices.length]);
    await writeU32(rankAllocation, ranks);

    operation = await plan.submit({
      activeIndices: activeIndices.view,
      activeCount: activeCount.view,
      ranks: rankAllocation.view,
      nextFlags: nextFlags.view,
      prefix: prefix.view,
      outputIndices: outputIndices.view,
      nextCount: nextCount.view,
      status: status.view,
    });
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed', `${model.name}: CUDA operation did not complete`);

    const [semanticStatus] = await readU32(status, 1);
    const [requiredCount] = await readU32(nextCount, 1);
    const output = semanticStatus === STATUS.OK ? await readU32(outputIndices, requiredCount) : [];
    return {
      name: model.name,
      semanticStatus,
      requiredCount,
      output,
      derivationLibrarySha256: plan.derivation.librarySha256,
      preparedNodeCount: plan.realization.preparedNodeCount,
    };
  } finally {
    if (operation) await operation.close();
    await plan.close();
    for (let index = allocations.length - 1; index >= 0; index -= 1) await closeAllocation(allocations[index]);
  }
}

function expected(model, outputCapacity = model.itemCapacity) {
  return runRankedDerivedActivationEpoch({
    ranks: [...model.ranks],
    activeIndices: [...model.activeIndices],
    deriveTarget: model.deriveTarget,
    maxEmissionsPerItem: model.maxEmissionsPerItem,
    outputCapacity,
  });
}

let runtime;
const result = {
  schemaVersion: 1,
  kind: 'cuda-algorithms-native-ranked-derived-activation-qualification',
  consumers: [],
};

try {
  runtime = await openCudaRuntime({ compiler: true });

  for (const model of [implicitDag, stagedLineage]) {
    const actual = await runFixture(runtime, model);
    const reference = expected(model);
    assert.equal(actual.semanticStatus, STATUS.OK, `${model.name}: semantic status`);
    assert.equal(actual.requiredCount, reference.requiredCount, `${model.name}: required count`);
    assert.deepEqual(actual.output, reference.nextIndices, `${model.name}: activated indices`);
    result.consumers.push(actual);
  }

  const capacity = await runFixture(runtime, implicitDag, { outputCapacity: 2 });
  const capacityReference = expected(implicitDag, 2);
  assert.equal(capacityReference.status, 'capacity-yield');
  assert.equal(capacity.semanticStatus, STATUS.OUTPUT_CAPACITY_EXHAUSTED);
  assert.equal(capacity.requiredCount, capacityReference.requiredCount);
  result.capacity = capacity;

  const invalidRanks = [...implicitDag.ranks];
  invalidRanks[6] = invalidRanks[7];
  const rankViolation = await runFixture(runtime, implicitDag, { ranks: invalidRanks });
  assert.equal(rankViolation.semanticStatus, STATUS.RANK_DESCENT_VIOLATION);
  result.rankViolation = rankViolation;

  result.outcome = 'pass';
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  result.outcome = 'fail';
  result.firstFailure = {
    name: error?.name ?? 'Error',
    message: error?.message ?? String(error),
  };
  console.error(JSON.stringify(result, null, 2));
  throw error;
} finally {
  if (runtime) {
    const closed = await runtime.close();
    assert.equal(closed.graceful, true);
  }
}
