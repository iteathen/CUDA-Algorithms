export const RANKED_DERIVED_ACTIVATION_U32_STATUS = Object.freeze({
  OK: 0,
  INVALID_ACTIVE_EXTENT: 1,
  SOURCE_INDEX_OUT_OF_RANGE: 2,
  TARGET_INDEX_OUT_OF_RANGE: 3,
  RANK_DESCENT_VIOLATION: 4,
  OUTPUT_CAPACITY_EXHAUSTED: 5,
});

export const RANKED_DERIVED_INVALID_U32 = 0xffff_ffff;

const source = `
function resetRankedDerivedActivation(nextFlags, status, nextCount, itemCapacity) {
  const i = gpu.thread.globalX();
  if (i < itemCapacity) nextFlags[i] = gpu.u32(0);
  if (i === gpu.u32(0)) {
    status[gpu.u32(0)] = gpu.u32(0);
    nextCount[gpu.u32(0)] = gpu.u32(0);
  }
}

function deriveAndActivateRankedU32(activeIndices, activeCount, ranks, nextFlags, inputCapacity, itemCapacity, maxEmissionsPerItem, status) {
  const i = gpu.thread.globalX();
  const active = activeCount[gpu.u32(0)];

  if (active > inputCapacity) {
    if (i === gpu.u32(0)) gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(1));
    return;
  }
  if (i >= active) return;

  const sourceIndex = activeIndices[i];
  if (sourceIndex >= itemCapacity) {
    gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(2));
    return;
  }

  const sourceRank = ranks[sourceIndex];
  let lane = gpu.u32(0);
  while (lane < maxEmissionsPerItem) {
    const targetIndex = deriveTargetIndex(sourceIndex, lane);
    if (targetIndex !== gpu.u32(4294967295)) {
      if (targetIndex >= itemCapacity) {
        gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(3));
        return;
      }
      if (ranks[targetIndex] >= sourceRank) {
        gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(4));
        return;
      }
      gpu.atomic.cas(nextFlags, targetIndex, gpu.u32(0), gpu.u32(1));
    }
    lane++;
  }
}

function exclusiveActivationScanU32(nextFlags, prefix, itemCapacity, status) {
  const i = gpu.thread.globalX();
  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) return;
  if (i >= itemCapacity) return;

  let sum = gpu.u32(0);
  let j = gpu.u32(0);
  while (j < i) {
    sum = sum + nextFlags[j];
    j++;
  }
  prefix[i] = sum;
}

function compactActivatedIndicesU32(nextFlags, prefix, outputIndices, nextCount, itemCapacity, outputCapacity, status) {
  const i = gpu.thread.globalX();
  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) return;

  let selectedCount = gpu.u32(0);
  if (itemCapacity > gpu.u32(0)) {
    const last = itemCapacity - gpu.u32(1);
    selectedCount = prefix[last] + nextFlags[last];
  }

  if (i === gpu.u32(0)) {
    nextCount[gpu.u32(0)] = selectedCount;
    if (selectedCount > outputCapacity) gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(5));
  }

  if (selectedCount > outputCapacity || i >= itemCapacity) return;
  if (nextFlags[i] === gpu.u32(1)) outputIndices[prefix[i]] = i;
}
`;

const functions = Object.freeze([
  Object.freeze({
    name: 'resetRankedDerivedActivation', kind: 'kernel', returns: 'void',
    parameters: Object.freeze([
      Object.freeze({ name: 'nextFlags', type: 'ptr<u32>' }),
      Object.freeze({ name: 'status', type: 'ptr<u32>' }),
      Object.freeze({ name: 'nextCount', type: 'ptr<u32>' }),
      Object.freeze({ name: 'itemCapacity', type: 'u32' }),
    ]),
  }),
  Object.freeze({
    name: 'deriveAndActivateRankedU32', kind: 'kernel', returns: 'void',
    parameters: Object.freeze([
      Object.freeze({ name: 'activeIndices', type: 'ptr<u32>' }),
      Object.freeze({ name: 'activeCount', type: 'ptr<u32>' }),
      Object.freeze({ name: 'ranks', type: 'ptr<u32>' }),
      Object.freeze({ name: 'nextFlags', type: 'ptr<u32>' }),
      Object.freeze({ name: 'inputCapacity', type: 'u32' }),
      Object.freeze({ name: 'itemCapacity', type: 'u32' }),
      Object.freeze({ name: 'maxEmissionsPerItem', type: 'u32' }),
      Object.freeze({ name: 'status', type: 'ptr<u32>' }),
    ]),
  }),
  Object.freeze({
    name: 'exclusiveActivationScanU32', kind: 'kernel', returns: 'void',
    parameters: Object.freeze([
      Object.freeze({ name: 'nextFlags', type: 'ptr<u32>' }),
      Object.freeze({ name: 'prefix', type: 'ptr<u32>' }),
      Object.freeze({ name: 'itemCapacity', type: 'u32' }),
      Object.freeze({ name: 'status', type: 'ptr<u32>' }),
    ]),
  }),
  Object.freeze({
    name: 'compactActivatedIndicesU32', kind: 'kernel', returns: 'void',
    parameters: Object.freeze([
      Object.freeze({ name: 'nextFlags', type: 'ptr<u32>' }),
      Object.freeze({ name: 'prefix', type: 'ptr<u32>' }),
      Object.freeze({ name: 'outputIndices', type: 'ptr<u32>' }),
      Object.freeze({ name: 'nextCount', type: 'ptr<u32>' }),
      Object.freeze({ name: 'itemCapacity', type: 'u32' }),
      Object.freeze({ name: 'outputCapacity', type: 'u32' }),
      Object.freeze({ name: 'status', type: 'ptr<u32>' }),
    ]),
  }),
]);

export function rankedDerivedActivationU32DeviceProgram(derivation) {
  return Object.freeze({
    source,
    compile: Object.freeze({ headerProfile: 'cuda-cccl' }),
    imports: Object.freeze([
      Object.freeze({ library: derivation.library, name: derivation.name, as: 'deriveTargetIndex' }),
    ]),
    functions,
  });
}
