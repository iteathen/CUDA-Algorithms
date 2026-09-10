import { compileDeviceProgram } from 'cuda-js';

import {
  RANKED_DERIVED_ACTIVATION_U32_STATUS,
  RANKED_DERIVED_INVALID_U32,
  rankedDerivedActivationU32DeviceProgram,
} from '../device/ranked-derived-activation-u32-program.mjs';
import {
  U32_BYTES,
  binding,
  blockSize,
  closeResources,
  kernelByName,
  positiveSafeInteger,
  rejectViewWriteConflicts,
  requireU32View,
  u32Bytes,
} from '../internal/common.mjs';

export const RANKED_DERIVED_ACTIVATION_U32_CONTRACT = 'CUDA-Algorithms-ranked-derived-activation-u32-working-v0';
export { RANKED_DERIVED_ACTIVATION_U32_STATUS, RANKED_DERIVED_INVALID_U32 };

const U32_MAX = 0xffff_ffff;

function positiveU32(value, label) {
  const normalized = positiveSafeInteger(value, label);
  if (normalized > U32_MAX) throw new RangeError(`${label} exceeds u32 range`);
  return normalized;
}

function normalizeDerivation(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('derivation must be an object containing library and name');
  }
  if (typeof value.name !== 'string' || value.name.length === 0) {
    throw new TypeError('derivation.name must select one Device-JS library export');
  }
  const exports = value.library?.exports;
  if (!Array.isArray(exports)) throw new TypeError('derivation.library must be a compiled public Device-JS library record');
  const selected = exports.find((entry) => entry?.name === value.name);
  if (!selected) throw new RangeError('derivation.name does not select a declared library export');
  const parameterTypes = Array.isArray(selected.parameters) ? selected.parameters.map((entry) => entry?.type) : [];
  if (parameterTypes.length !== 2 || parameterTypes[0] !== 'u32' || parameterTypes[1] !== 'u32' || selected.returns !== 'u32') {
    throw new TypeError('ranked derivation export must have exact signature (u32 sourceIndex, u32 emissionLane) -> u32 targetIndex');
  }
  return Object.freeze({ library: value.library, name: value.name, selected });
}

export async function createRankedDerivedActivationU32Plan(runtime, options = {}) {
  const itemCapacity = positiveU32(options.itemCapacity, 'itemCapacity');
  const inputCapacity = positiveU32(options.inputCapacity ?? itemCapacity, 'inputCapacity');
  const outputCapacity = positiveU32(options.outputCapacity ?? itemCapacity, 'outputCapacity');
  const maxEmissionsPerItem = positiveU32(options.maxEmissionsPerItem, 'maxEmissionsPerItem');
  if (outputCapacity > itemCapacity) throw new RangeError('outputCapacity cannot exceed itemCapacity');
  const threads = blockSize(options.blockSize);
  const derivation = normalizeDerivation(options.derivation);

  const itemBytes = u32Bytes(itemCapacity, 'item universe');
  const inputBytes = u32Bytes(inputCapacity, 'input workset');
  const outputBytes = u32Bytes(outputCapacity, 'output workset');
  const itemGridX = Math.ceil(itemCapacity / threads);
  const inputGridX = Math.ceil(inputCapacity / threads);

  const compiled = await compileDeviceProgram(runtime, rankedDerivedActivationU32DeviceProgram(derivation));
  const artifact = compiled.linker?.artifact ?? compiled.compiler?.artifact;
  if (!artifact || (artifact.format !== 'ptx' && artifact.format !== 'cubin')) {
    throw new Error(`unexpected executable artifact format: ${artifact?.format ?? 'missing'}`);
  }

  const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes });
  const resetKernel = kernelByName(compiled, 'resetRankedDerivedActivation');
  const deriveKernel = kernelByName(compiled, 'deriveAndActivateRankedU32');
  const scanKernel = kernelByName(compiled, 'exclusiveActivationScanU32');
  const compactKernel = kernelByName(compiled, 'compactActivatedIndicesU32');
  const reset = await module.getFunction({ name: resetKernel.functionName, parameters: resetKernel.parameters });
  const derive = await module.getFunction({ name: deriveKernel.functionName, parameters: deriveKernel.parameters });
  const scan = await module.getFunction({ name: scanKernel.functionName, parameters: scanKernel.parameters });
  const compact = await module.getFunction({ name: compactKernel.functionName, parameters: compactKernel.parameters });

  const prepared = await runtime.prepareOperationDag({ nodes: [
    {
      id: 'reset', function: reset,
      grid: { x: itemGridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [binding('nextFlags'), binding('status'), binding('nextCount'), itemCapacity],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: itemBytes, mode: 'write' },
        { argumentIndex: 1, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
        { argumentIndex: 2, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
      ],
    },
    {
      id: 'derive', after: ['reset'], function: derive,
      grid: { x: inputGridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [
        binding('activeIndices'), binding('activeCount'), binding('ranks'), binding('nextFlags'),
        inputCapacity, itemCapacity, maxEmissionsPerItem, binding('status'),
      ],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
        { argumentIndex: 2, byteOffset: 0, byteLength: itemBytes, mode: 'read' },
        { argumentIndex: 3, byteOffset: 0, byteLength: itemBytes, mode: 'read-write' },
        { argumentIndex: 7, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
      ],
    },
    {
      id: 'scan', after: ['derive'], function: scan,
      grid: { x: itemGridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [binding('nextFlags'), binding('prefix'), itemCapacity, binding('status')],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: itemBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: itemBytes, mode: 'write' },
        { argumentIndex: 3, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
      ],
    },
    {
      id: 'compact', after: ['scan'], function: compact,
      grid: { x: itemGridX, y: 1, z: 1 }, block: { x: threads, y: 1, z: 1 },
      arguments: [
        binding('nextFlags'), binding('prefix'), binding('outputIndices'), binding('nextCount'),
        itemCapacity, outputCapacity, binding('status'),
      ],
      accesses: [
        { argumentIndex: 0, byteOffset: 0, byteLength: itemBytes, mode: 'read' },
        { argumentIndex: 1, byteOffset: 0, byteLength: itemBytes, mode: 'read' },
        { argumentIndex: 2, byteOffset: 0, byteLength: outputBytes, mode: 'write' },
        { argumentIndex: 3, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
        { argumentIndex: 6, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
      ],
    },
  ] });

  let closed = false;
  const owned = [module, reset, derive, scan, compact, prepared];

  return Object.freeze({
    kind: 'cuda-algorithms-plan',
    contract: RANKED_DERIVED_ACTIVATION_U32_CONTRACT,
    family: 'ranked-derived-activation',
    dtype: 'u32',
    itemCapacity,
    inputCapacity,
    outputCapacity,
    maxEmissionsPerItem,
    invalidTargetIndex: RANKED_DERIVED_INVALID_U32,
    blockSize: threads,
    status: RANKED_DERIVED_ACTIVATION_U32_STATUS,
    derivation: Object.freeze({
      librarySha256: derivation.library.sha256,
      exportName: derivation.name,
      parameters: Object.freeze(['u32', 'u32']),
      returns: 'u32',
    }),
    workspace: Object.freeze({
      activationFlagElements: itemCapacity,
      prefixElements: itemCapacity,
      controlU32Elements: 2,
    }),
    realization: Object.freeze({
      preparedNodeCount: 4,
      fullUniverseCompaction: true,
      progressionOwner: 'device',
    }),
    aliasing: Object.freeze({
      writeRolesRequireDisjointRanges: true,
      overlappingReadOnlyRangesAllowed: true,
      relationOwner: 'cuda-js:inspectDeviceViewRelation',
    }),
    async submit(bindings) {
      if (closed) throw new Error('ranked derived activation plan is closed');
      const normalized = {
        activeIndices: requireU32View(bindings?.activeIndices, inputCapacity, 'activeIndices', 'read'),
        activeCount: requireU32View(bindings?.activeCount, 1, 'activeCount', 'read'),
        ranks: requireU32View(bindings?.ranks, itemCapacity, 'ranks', 'read'),
        nextFlags: requireU32View(bindings?.nextFlags, itemCapacity, 'nextFlags', 'read-write'),
        prefix: requireU32View(bindings?.prefix, itemCapacity, 'prefix', 'read-write'),
        outputIndices: requireU32View(bindings?.outputIndices, outputCapacity, 'outputIndices', 'write'),
        nextCount: requireU32View(bindings?.nextCount, 1, 'nextCount', 'write'),
        status: requireU32View(bindings?.status, 1, 'status', 'read-write'),
      };
      rejectViewWriteConflicts([
        { label: 'activeIndices', view: normalized.activeIndices, access: 'read' },
        { label: 'activeCount', view: normalized.activeCount, access: 'read' },
        { label: 'ranks', view: normalized.ranks, access: 'read' },
        { label: 'nextFlags', view: normalized.nextFlags, access: 'write' },
        { label: 'prefix', view: normalized.prefix, access: 'write' },
        { label: 'outputIndices', view: normalized.outputIndices, access: 'write' },
        { label: 'nextCount', view: normalized.nextCount, access: 'write' },
        { label: 'status', view: normalized.status, access: 'write' },
      ]);
      return prepared.submit({ bindings: normalized });
    },
    async close() {
      if (closed) return;
      closed = true;
      await closeResources(owned, 'ranked derived activation plan');
    },
  });
}
