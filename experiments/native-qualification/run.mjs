import assert from 'node:assert/strict';
import { openCudaRuntime } from 'cuda-js';
import { scanUnsigned, selectIndices } from '../../reference/core-primitives.mjs';
import { stableLexicographicOrderIndices } from '../../reference/permutation-ordering.mjs';
import {
  createStableLexicographicOrderIndicesU32Plan,
  createStableSelectIndicesU32Plan,
  STABLE_ORDER_INDICES_U32_STATUS as ORDER_STATUS,
  STABLE_SELECT_INDICES_U32_STATUS as SELECT_STATUS,
} from '../../src/index.mjs';

const U32_BYTES = 4;

function encodeU32(values) {
  const bytes = new Uint8Array(values.length * U32_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  values.forEach((value, i) => {
    const n = typeof value === 'bigint' ? value : BigInt(value);
    if (n < 0n || n > 0xffffffffn) throw new RangeError(`u32 value out of range at ${i}`);
    view.setUint32(i * U32_BYTES, Number(n), true);
  });
  return bytes;
}

function decodeU32(bytes) {
  if (bytes.byteLength % U32_BYTES !== 0) throw new RangeError('u32 byte length is not aligned');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / U32_BYTES }, (_, i) => BigInt(view.getUint32(i * U32_BYTES, true)));
}

async function allocateU32(runtime, count, access = 'read-write') {
  if (!Number.isSafeInteger(count) || count < 1) throw new RangeError('native fixture allocations require positive safe-integer counts');
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

async function runSelectFixture(runtime, fixture) {
  const inputCapacity = fixture.flags.length;
  const outputCapacity = fixture.outputCapacity ?? inputCapacity;
  const plan = await createStableSelectIndicesU32Plan(runtime, {
    inputCapacity,
    outputCapacity,
    blockSize: Math.min(128, inputCapacity),
  });
  const allocations = [];
  let operation;

  try {
    const flags = await allocateU32(runtime, inputCapacity, 'read');
    const prefix = await allocateU32(runtime, inputCapacity, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const outputIndices = await allocateU32(runtime, outputCapacity, 'write');
    const outputCount = await allocateU32(runtime, 1, 'read-write');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(flags, prefix, activeCount, outputIndices, outputCount, status);

    await writeU32(flags, fixture.flags);
    await writeU32(prefix, new Array(inputCapacity).fill(0));
    await writeU32(activeCount, [fixture.active]);
    await writeU32(outputIndices, new Array(outputCapacity).fill(0));
    await writeU32(outputCount, [0]);
    await writeU32(status, [0]);

    operation = await plan.submit({
      flags: flags.view,
      prefix: prefix.view,
      activeCount: activeCount.view,
      outputIndices: outputIndices.view,
      outputCount: outputCount.view,
      status: status.view,
    });
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed', `${fixture.name}: CUDA operation did not complete`);

    const [statusValue] = await readU32(status, 1);
    const [countValue] = await readU32(outputCount, 1);
    assert.equal(statusValue, BigInt(fixture.expectedStatus), `${fixture.name}: semantic status`);

    if (fixture.expectedStatus === SELECT_STATUS.OK) {
      const active = Number(fixture.active);
      const referencePrefix = scanUnsigned(fixture.flags.slice(0, active).map(BigInt), {
        width: 32,
        mode: 'exclusive',
        op: 'add',
        init: 0n,
      });
      const referenceSelection = selectIndices(fixture.flags.slice(0, active), { indexWidth: 32 });
      assert.equal(countValue, referenceSelection.outputCount, `${fixture.name}: output count`);
      assert.deepEqual(await readU32(prefix, active), referencePrefix, `${fixture.name}: prefix`);
      assert.deepEqual(await readU32(outputIndices, Number(countValue)), referenceSelection.indices, `${fixture.name}: selected indices`);
    } else if (fixture.expectedStatus === SELECT_STATUS.OUTPUT_CAPACITY_EXHAUSTED) {
      const referenceSelection = selectIndices(fixture.flags.slice(0, Number(fixture.active)), { indexWidth: 32 });
      assert.equal(countValue, referenceSelection.outputCount, `${fixture.name}: required output count`);
    }

    return { name: fixture.name, status: Number(statusValue), outputCount: Number(countValue) };
  } finally {
    if (operation) await operation.close();
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
    await plan.close();
  }
}

async function runOrderingFixture(runtime, fixture) {
  const recordCapacity = fixture.keyWords[0].length;
  if (!fixture.keyWords.every((word) => word.length === recordCapacity)) throw new RangeError(`${fixture.name}: key-word capacities differ`);
  const indexCapacity = fixture.indices.length;
  const plan = await createStableLexicographicOrderIndicesU32Plan(runtime, {
    recordCapacity,
    indexCapacity,
    keyWordCount: fixture.keyWords.length,
    blockSize: Math.min(128, indexCapacity),
  });
  const allocations = [];
  let operation;

  try {
    const keyAllocations = [];
    for (const word of fixture.keyWords) {
      const allocation = await allocateU32(runtime, recordCapacity, 'read');
      allocations.push(allocation);
      keyAllocations.push(allocation);
      await writeU32(allocation, word);
    }
    const indicesA = await allocateU32(runtime, indexCapacity, 'read-write');
    const indicesB = await allocateU32(runtime, indexCapacity, 'read-write');
    const activeCount = await allocateU32(runtime, 1, 'read');
    const status = await allocateU32(runtime, 1, 'read-write');
    allocations.push(indicesA, indicesB, activeCount, status);

    await writeU32(indicesA, fixture.indices);
    await writeU32(indicesB, new Array(indexCapacity).fill(0));
    await writeU32(activeCount, [fixture.active]);
    await writeU32(status, [0]);

    operation = await plan.submit({
      keyWords: keyAllocations.map((allocation) => allocation.view),
      indicesA: indicesA.view,
      indicesB: indicesB.view,
      activeCount: activeCount.view,
      status: status.view,
    });
    const terminal = await operation.wait();
    assert.equal(terminal.status, 'completed', `${fixture.name}: CUDA operation did not complete`);

    const [statusValue] = await readU32(status, 1);
    assert.equal(statusValue, BigInt(fixture.expectedStatus), `${fixture.name}: semantic status`);

    if (fixture.expectedStatus === ORDER_STATUS.OK) {
      const active = Number(fixture.active);
      const inputOrder = fixture.indices.slice(0, active).map(BigInt);
      const expected = stableLexicographicOrderIndices(
        fixture.keyWords.map((word) => word.map(BigInt)),
        inputOrder,
        { wordWidth: 32, indexWidth: 32 },
      );
      const resultAllocation = plan.resultBinding === 'indicesA' ? indicesA : indicesB;
      assert.deepEqual(await readU32(resultAllocation, active), expected, `${fixture.name}: final ordered indices`);
    }

    return { name: fixture.name, status: Number(statusValue), keyWordCount: fixture.keyWords.length, resultBinding: plan.resultBinding };
  } finally {
    if (operation) await operation.close();
    for (let i = allocations.length - 1; i >= 0; i -= 1) await closeAllocation(allocations[i]);
    await plan.close();
  }
}

const selectFixtures = [
  { name: 'select-sparse', flags: [1, 0, 1, 1, 0, 0, 1, 0], active: 8, expectedStatus: SELECT_STATUS.OK },
  { name: 'select-active-prefix', flags: [1, 0, 1, 0, 1, 1, 1, 1], active: 5, expectedStatus: SELECT_STATUS.OK },
  { name: 'select-empty-active', flags: [1, 1, 1, 1], active: 0, expectedStatus: SELECT_STATUS.OK },
  { name: 'select-output-pressure', flags: [1, 1, 1, 0], active: 4, outputCapacity: 2, expectedStatus: SELECT_STATUS.OUTPUT_CAPACITY_EXHAUSTED },
  { name: 'select-invalid-flag', flags: [1, 2, 0, 1], active: 4, expectedStatus: SELECT_STATUS.INVALID_FLAG },
  { name: 'select-invalid-extent', flags: [1, 0, 1, 0], active: 5, expectedStatus: SELECT_STATUS.INVALID_EXTENT },
];

const orderingFixtures = [
  {
    name: 'order-two-word-stable',
    keyWords: [[2, 1, 2, 1, 2, 1], [5, 1, 4, 7, 4, 6]],
    indices: [2, 5, 0, 4, 1, 3],
    active: 6,
    expectedStatus: ORDER_STATUS.OK,
  },
  {
    name: 'order-three-word-stable',
    keyWords: [[1, 0, 1, 0, 1, 0], [2, 2, 1, 1, 2, 1], [9, 3, 4, 5, 4, 5]],
    indices: [5, 2, 0, 4, 1, 3],
    active: 6,
    expectedStatus: ORDER_STATUS.OK,
  },
  {
    name: 'order-active-prefix',
    keyWords: [[1, 1, 0, 2, 9, 9], [3, 3, 7, 1, 0, 0]],
    indices: [3, 0, 2, 1, 5, 4],
    active: 4,
    expectedStatus: ORDER_STATUS.OK,
  },
  {
    name: 'order-duplicate-index-values',
    keyWords: [[1, 0, 1, 0], [2, 9, 2, 1]],
    indices: [2, 2, 3, 1],
    active: 4,
    expectedStatus: ORDER_STATUS.OK,
  },
  {
    name: 'order-empty-active',
    keyWords: [[1, 0, 1, 0], [2, 9, 2, 1]],
    indices: [3, 2, 1, 0],
    active: 0,
    expectedStatus: ORDER_STATUS.OK,
  },
  {
    name: 'order-invalid-index',
    keyWords: [[1, 0, 1, 0], [2, 9, 2, 1]],
    indices: [0, 4, 1, 2],
    active: 4,
    expectedStatus: ORDER_STATUS.INVALID_INDEX,
  },
];

let runtime;
const results = {
  schemaVersion: 1,
  kind: 'cuda-algorithms-native-first-profile-qualification',
  maintainedSurface: true,
  select: [],
  ordering: [],
};

try {
  runtime = await openCudaRuntime({ compiler: true });
  for (const fixture of selectFixtures) results.select.push(await runSelectFixture(runtime, fixture));
  for (const fixture of orderingFixtures) results.ordering.push(await runOrderingFixture(runtime, fixture));
  results.outcome = 'pass';
  console.log(JSON.stringify(results, null, 2));
} catch (error) {
  results.outcome = 'fail';
  results.firstFailure = {
    name: error?.name ?? 'Error',
    message: error?.message ?? String(error),
    code: error?.code ?? null,
    category: error?.category ?? null,
  };
  console.error(JSON.stringify(results, null, 2));
  process.exitCode = 1;
} finally {
  if (runtime) {
    try {
      const closed = await runtime.close();
      if (!closed.graceful || closed.restartRequired) {
        console.error(JSON.stringify({ kind: 'runtime-close', graceful: closed.graceful, restartRequired: closed.restartRequired }, null, 2));
        process.exitCode = 1;
      }
    } catch (error) {
      console.error(JSON.stringify({ kind: 'runtime-close-error', message: error?.message ?? String(error) }, null, 2));
      process.exitCode = 1;
    }
  }
}
