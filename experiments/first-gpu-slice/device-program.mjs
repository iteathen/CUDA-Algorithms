export const STATUS = Object.freeze({
  OK: 0,
  INVALID_EXTENT: 1,
  OUTPUT_CAPACITY_EXHAUSTED: 2,
  INVALID_FLAG: 3,
});

export const source = `
function resetControl(status, outputCount) {
  const i = gpu.thread.globalX();
  if (i === gpu.u32(0)) {
    status[gpu.u32(0)] = gpu.u32(0);
    outputCount[gpu.u32(0)] = gpu.u32(0);
  }
}

function exclusiveFlagScanU32(flags, prefix, activeCount, inputCapacity, status) {
  const i = gpu.thread.globalX();
  const active = activeCount[gpu.u32(0)];

  if (active > inputCapacity) {
    if (i === gpu.u32(0)) {
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(1));
    }
    return;
  }

  if (i >= active) {
    return;
  }

  const ownFlag = flags[i];
  if (ownFlag > gpu.u32(1)) {
    gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(3));
  }

  let sum = gpu.u32(0);
  let j = gpu.u32(0);
  while (j < i) {
    sum = sum + flags[j];
    j++;
  }
  prefix[i] = sum;
}

function selectIndicesFromScanU32(flags, prefix, outputIndices, outputCount, activeCount, inputCapacity, outputCapacity, status) {
  const i = gpu.thread.globalX();
  const active = activeCount[gpu.u32(0)];

  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) {
    return;
  }

  if (active > inputCapacity) {
    if (i === gpu.u32(0)) {
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(1));
    }
    return;
  }

  let selectedCount = gpu.u32(0);
  if (active > gpu.u32(0)) {
    const last = active - gpu.u32(1);
    selectedCount = prefix[last] + flags[last];
  }

  if (i === gpu.u32(0)) {
    outputCount[gpu.u32(0)] = selectedCount;
    if (selectedCount > outputCapacity) {
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(2));
    }
  }

  if (selectedCount > outputCapacity || i >= active) {
    return;
  }

  if (flags[i] === gpu.u32(1)) {
    outputIndices[prefix[i]] = i;
  }
}
`;

export const deviceProgramRequest = Object.freeze({
  source,
  compile: Object.freeze({ headerProfile: 'cuda-cccl' }),
  functions: Object.freeze([
    Object.freeze({
      name: 'resetControl',
      kind: 'kernel',
      parameters: Object.freeze([
        Object.freeze({ name: 'status', type: 'ptr<u32>' }),
        Object.freeze({ name: 'outputCount', type: 'ptr<u32>' }),
      ]),
      returns: 'void',
    }),
    Object.freeze({
      name: 'exclusiveFlagScanU32',
      kind: 'kernel',
      parameters: Object.freeze([
        Object.freeze({ name: 'flags', type: 'ptr<u32>' }),
        Object.freeze({ name: 'prefix', type: 'ptr<u32>' }),
        Object.freeze({ name: 'activeCount', type: 'ptr<u32>' }),
        Object.freeze({ name: 'inputCapacity', type: 'u32' }),
        Object.freeze({ name: 'status', type: 'ptr<u32>' }),
      ]),
      returns: 'void',
    }),
    Object.freeze({
      name: 'selectIndicesFromScanU32',
      kind: 'kernel',
      parameters: Object.freeze([
        Object.freeze({ name: 'flags', type: 'ptr<u32>' }),
        Object.freeze({ name: 'prefix', type: 'ptr<u32>' }),
        Object.freeze({ name: 'outputIndices', type: 'ptr<u32>' }),
        Object.freeze({ name: 'outputCount', type: 'ptr<u32>' }),
        Object.freeze({ name: 'activeCount', type: 'ptr<u32>' }),
        Object.freeze({ name: 'inputCapacity', type: 'u32' }),
        Object.freeze({ name: 'outputCapacity', type: 'u32' }),
        Object.freeze({ name: 'status', type: 'ptr<u32>' }),
      ]),
      returns: 'void',
    }),
  ]),
});
