import { compileDeviceProgram } from 'cuda-js';
import { stableOrderIndicesU32DeviceProgram, STABLE_ORDER_INDICES_U32_STATUS } from '../device/stable-order-indices-u32-program.mjs';
import { binding, blockSize, closeResources, kernelByName, positiveSafeInteger, PREPARED_KERNEL_NODE_CEILING, rejectSameViewWriteConflicts, requireU32View, U32_BYTES, u32Bytes } from '../internal/common.mjs';

export const STABLE_LEXICOGRAPHIC_ORDER_INDICES_U32_CONTRACT = 'CUDA-Algorithms-stable-lexicographic-order-indices-u32-candidate-v0';

export async function createStableLexicographicOrderIndicesU32Plan(runtime, options = {}) {
  const recordCapacity = positiveSafeInteger(options.recordCapacity, 'recordCapacity');
  const indexCapacity = positiveSafeInteger(options.indexCapacity, 'indexCapacity');
  const keyWordCount = positiveSafeInteger(options.keyWordCount, 'keyWordCount');
  const threads = blockSize(options.blockSize);
  if (keyWordCount + 1 > PREPARED_KERNEL_NODE_CEILING) {
    throw new RangeError(`keyWordCount exceeds the current CUDA-JS prepared-DAG realization bound of ${PREPARED_KERNEL_NODE_CEILING - 1} words`);
  }

  const recordBytes = u32Bytes(recordCapacity, 'record');
  const indexBytes = u32Bytes(indexCapacity, 'index');
  const grid = { x: Math.ceil(indexCapacity / threads), y: 1, z: 1 };
  const block = { x: threads, y: 1, z: 1 };

  const compiled = await compileDeviceProgram(runtime, stableOrderIndicesU32DeviceProgram);
  const artifact = compiled.compiler.artifact;
  if (artifact.format !== 'ptx' && artifact.format !== 'cubin') throw new Error(`unexpected executable artifact format: ${artifact.format}`);

  const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes });
  const resetKernel = kernelByName(compiled, 'resetOrderingStatus');
  const orderKernel = kernelByName(compiled, 'stableOrderIndicesByKeyU32');
  const reset = await module.getFunction({ name: resetKernel.functionName, parameters: resetKernel.parameters });
  const order = await module.getFunction({ name: orderKernel.functionName, parameters: orderKernel.parameters });

  const nodes = [{
    id: 'reset', function: reset,
    grid: { x: 1, y: 1, z: 1 }, block: { x: 1, y: 1, z: 1 },
    arguments: [binding('status')],
    accesses: [{ argumentIndex: 0, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' }],
  }];

  let inputBinding = 'indicesA';
  let outputBinding = 'indicesB';
  let previous = 'reset';
  for (let word = keyWordCount - 1; word >= 0; word -= 1) {
    const id = `order-word-${word}`;
    nodes.push({
      id,
      after: [previous],
      function: order,
      grid,
      block,
      arguments: [
        binding(`keyWord${word}`), binding(inputBinding), binding(outputBinding), binding('activeCount'),
        recordCapacity, indexCapacity, indexCapacity, binding('status'),
      ],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: recordBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: indexBytes, mode: 'read' },
        { argumentIndex: 2, byteOffset: 0, byteLength: indexBytes, mode: 'write' },
        { argumentIndex: 3, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
        { argumentIndex: 7, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
      ],
    });
    previous = id;
    [inputBinding, outputBinding] = [outputBinding, inputBinding];
  }

  const resultBinding = inputBinding;
  const prepared = await runtime.prepareOperationDag({ nodes });
  let closed = false;
  const owned = [module, reset, order, prepared];

  return Object.freeze({
    kind: 'cuda-algorithms-plan',
    contract: STABLE_LEXICOGRAPHIC_ORDER_INDICES_U32_CONTRACT,
    family: 'stable-lexicographic-order-indices',
    keyDtype: 'u32',
    indexDtype: 'u32',
    recordCapacity,
    indexCapacity,
    keyWordCount,
    blockSize: threads,
    resultBinding,
    status: STABLE_ORDER_INDICES_U32_STATUS,
    realizationBounds: Object.freeze({ maxKeyWordCount: PREPARED_KERNEL_NODE_CEILING - 1, reason: 'current CUDA-JS prepared kernel DAG node ceiling' }),
    aliasing: Object.freeze({ exactWriteConflictRejected: true, readReadSameViewAllowed: true, overlappingSiblingViews: 'requires CUDA-JS #260 before acceptance' }),
    async submit(bindings) {
      if (closed) throw new Error('stable ordering plan is closed');
      if (!Array.isArray(bindings?.keyWords) || bindings.keyWords.length !== keyWordCount) {
        throw new RangeError(`keyWords must contain exactly ${keyWordCount} u32 device views ordered most-significant to least-significant`);
      }
      const normalized = {};
      const roles = [];
      for (let word = 0; word < keyWordCount; word += 1) {
        const view = requireU32View(bindings.keyWords[word], recordCapacity, `keyWords[${word}]`, 'read');
        normalized[`keyWord${word}`] = view;
        roles.push({ label: `keyWords[${word}]`, view, access: 'read' });
      }
      normalized.indicesA = requireU32View(bindings.indicesA, indexCapacity, 'indicesA', 'read-write');
      normalized.indicesB = requireU32View(bindings.indicesB, indexCapacity, 'indicesB', 'read-write');
      normalized.activeCount = requireU32View(bindings.activeCount, 1, 'activeCount', 'read');
      normalized.status = requireU32View(bindings.status, 1, 'status', 'read-write');
      roles.push(
        { label: 'indicesA', view: normalized.indicesA, access: 'write' },
        { label: 'indicesB', view: normalized.indicesB, access: 'write' },
        { label: 'activeCount', view: normalized.activeCount, access: 'read' },
        { label: 'status', view: normalized.status, access: 'write' },
      );
      rejectSameViewWriteConflicts(roles);
      return prepared.submit({ bindings: normalized });
    },
    async close() {
      if (closed) return;
      closed = true;
      await closeResources(owned, 'stable ordering plan');
    },
  });
}
