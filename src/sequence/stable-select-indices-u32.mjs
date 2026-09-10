import { compileDeviceProgram } from 'cuda-js';
import { stableSelectIndicesU32DeviceProgram, STABLE_SELECT_INDICES_U32_STATUS } from '../device/stable-select-indices-u32-program.mjs';
import { binding, blockSize, closeResources, kernelByName, positiveSafeInteger, requireU32View, U32_BYTES, u32Bytes } from '../internal/common.mjs';

export const STABLE_SELECT_INDICES_U32_CONTRACT = 'CUDA-Algorithms-stable-select-indices-u32-candidate-v0';

export async function createStableSelectIndicesU32Plan(runtime, options = {}) {
  const inputCapacity = positiveSafeInteger(options.inputCapacity, 'inputCapacity');
  const outputCapacity = positiveSafeInteger(options.outputCapacity, 'outputCapacity');
  const threads = blockSize(options.blockSize);
  const inputBytes = u32Bytes(inputCapacity, 'input');
  const outputBytes = u32Bytes(outputCapacity, 'output');
  const gridX = Math.ceil(inputCapacity / threads);

  const compiled = await compileDeviceProgram(runtime, stableSelectIndicesU32DeviceProgram);
  const artifact = compiled.compiler.artifact;
  if (artifact.format !== 'ptx' && artifact.format !== 'cubin') throw new Error(`unexpected executable artifact format: ${artifact.format}`);

  const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes });
  const resetKernel = kernelByName(compiled, 'resetControl');
  const scanKernel = kernelByName(compiled, 'exclusiveFlagScanU32');
  const selectKernel = kernelByName(compiled, 'selectIndicesFromScanU32');
  const reset = await module.getFunction({ name: resetKernel.functionName, parameters: resetKernel.parameters });
  const scan = await module.getFunction({ name: scanKernel.functionName, parameters: scanKernel.parameters });
  const select = await module.getFunction({ name: selectKernel.functionName, parameters: selectKernel.parameters });

  const prepared = await runtime.prepareOperationDag({ nodes: [
    {
      id: 'reset', function: reset,
      grid: { x: 1, y: 1, z: 1 }, block: { x: 1, y: 1, z: 1 },
      arguments: [binding('status'), binding('outputCount')],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
        { argumentIndex: 1, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
      ],
    },
    {
      id: 'scan', after: ['reset'], function: scan,
      grid: { x: gridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [binding('flags'), binding('prefix'), binding('activeCount'), inputCapacity, binding('status')],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: inputBytes, mode: 'write' },
        { argumentIndex: 2, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
        { argumentIndex: 4, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
      ],
    },
    {
      id: 'select', after: ['scan'], function: select,
      grid: { x: gridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [
        binding('flags'), binding('prefix'), binding('outputIndices'), binding('outputCount'), binding('activeCount'),
        inputCapacity, outputCapacity, binding('status'),
      ],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
        { argumentIndex: 2, byteOffset: 0, byteLength: outputBytes, mode: 'write' },
        { argumentIndex: 3, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
        { argumentIndex: 4, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
        { argumentIndex: 7, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
      ],
    },
  ] });

  let closed = false;
  const owned = [module, reset, scan, select, prepared];

  return Object.freeze({
    kind: 'cuda-algorithms-plan',
    contract: STABLE_SELECT_INDICES_U32_CONTRACT,
    family: 'stable-select-indices',
    dtype: 'u32',
    inputCapacity,
    outputCapacity,
    blockSize: threads,
    status: STABLE_SELECT_INDICES_U32_STATUS,
    workspace: Object.freeze({ prefixElements: inputCapacity, prefixBytes: inputBytes, controlU32Elements: 2 }),
    async submit(bindings) {
      if (closed) throw new Error('stable select plan is closed');
      requireU32View(bindings?.flags, inputCapacity, 'flags', 'read');
      requireU32View(bindings?.prefix, inputCapacity, 'prefix', 'write');
      requireU32View(bindings?.activeCount, 1, 'activeCount', 'read');
      requireU32View(bindings?.outputIndices, outputCapacity, 'outputIndices', 'write');
      requireU32View(bindings?.outputCount, 1, 'outputCount', 'write');
      requireU32View(bindings?.status, 1, 'status', 'write');
      return prepared.submit({ bindings: {
        flags: bindings.flags,
        prefix: bindings.prefix,
        activeCount: bindings.activeCount,
        outputIndices: bindings.outputIndices,
        outputCount: bindings.outputCount,
        status: bindings.status,
      } });
    },
    async close() {
      if (closed) return;
      closed = true;
      await closeResources(owned, 'stable select plan');
    },
  });
}
