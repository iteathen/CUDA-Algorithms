export const STATUS = Object.freeze({
  OK: 0,
  INVALID_EXTENT: 1,
  OUTPUT_CAPACITY_EXHAUSTED: 2,
  INVALID_INDEX: 4,
});

export const source = `
function resetOrderingStatus(status) {
  const i = gpu.thread.globalX();
  if (i === gpu.u32(0)) {
    status[gpu.u32(0)] = gpu.u32(0);
  }
}

function stableOrderIndicesByKeyU32(keys, indicesIn, indicesOut, activeCount, keyCapacity, inputCapacity, outputCapacity, status) {
  const i = gpu.thread.globalX();
  const active = activeCount[gpu.u32(0)];

  if (status[gpu.u32(0)] !== gpu.u32(0)) {
    return;
  }

  if (active > inputCapacity) {
    if (i === gpu.u32(0)) {
      status[gpu.u32(0)] = gpu.u32(1);
    }
    return;
  }

  if (active > outputCapacity) {
    if (i === gpu.u32(0)) {
      status[gpu.u32(0)] = gpu.u32(2);
    }
    return;
  }

  if (i >= active) {
    return;
  }

  const ownRecord = indicesIn[i];
  if (ownRecord >= keyCapacity) {
    gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(4));
    return;
  }
  const ownKey = keys[ownRecord];

  let rank = gpu.u32(0);
  let j = gpu.u32(0);
  while (j < active) {
    const otherRecord = indicesIn[j];
    if (otherRecord >= keyCapacity) {
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(4));
      return;
    }
    const otherKey = keys[otherRecord];
    if (otherKey < ownKey || (otherKey === ownKey && j < i)) {
      rank++;
    }
    j++;
  }

  indicesOut[rank] = ownRecord;
}
`;

export const deviceProgramRequest = Object.freeze({
  source,
  functions: Object.freeze([
    Object.freeze({
      name: 'resetOrderingStatus',
      kind: 'kernel',
      parameters: Object.freeze([
        Object.freeze({ name: 'status', type: 'ptr<u32>' }),
      ]),
      returns: 'void',
    }),
    Object.freeze({
      name: 'stableOrderIndicesByKeyU32',
      kind: 'kernel',
      parameters: Object.freeze([
        Object.freeze({ name: 'keys', type: 'ptr<u32>' }),
        Object.freeze({ name: 'indicesIn', type: 'ptr<u32>' }),
        Object.freeze({ name: 'indicesOut', type: 'ptr<u32>' }),
        Object.freeze({ name: 'activeCount', type: 'ptr<u32>' }),
        Object.freeze({ name: 'keyCapacity', type: 'u32' }),
        Object.freeze({ name: 'inputCapacity', type: 'u32' }),
        Object.freeze({ name: 'outputCapacity', type: 'u32' }),
        Object.freeze({ name: 'status', type: 'ptr<u32>' }),
      ]),
      returns: 'void',
    }),
  ]),
});
