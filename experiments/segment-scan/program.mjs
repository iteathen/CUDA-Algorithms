const ptr = name => ({ name, type: 'ptr<u32>' });
const scalar = name => ({ name, type: 'u32' });
const kernel = (name, pointers, scalars = []) => ({ name, kind: 'kernel', returns: 'void', parameters: [...pointers.map(ptr), ...scalars.map(scalar)] });

export const program = {
  compile: { headerProfile: 'cuda-cccl' },
  source: `
function resetSegments(status, groupCount, requiredGroups, totalLength) {
  if (gpu.thread.globalX() === gpu.u32(0)) {
    status[gpu.u32(0)] = gpu.u32(0); groupCount[gpu.u32(0)] = gpu.u32(0);
    requiredGroups[gpu.u32(0)] = gpu.u32(0); totalLength[gpu.u32(0)] = gpu.u32(0);
  }
}
function scanTiles(input, scratchA, scratchB, prefix, sums, activeCount, status,
  capacity, divisor, validateHeads) {
  const i = gpu.thread.globalX(); const lane = gpu.thread.x();
  const width = gpu.blockDim.x(); const base = i - lane;
  const original = activeCount[gpu.u32(0)];
  if (original > capacity) {
    if (i === gpu.u32(0)) gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(1));
    return;
  }
  let n = original / divisor;
  if (original % divisor !== gpu.u32(0)) n++;
  let own = gpu.u32(0);
  if (i < n) own = input[i];
  if (validateHeads !== gpu.u32(0) && i < n) {
    if (own > gpu.u32(1) || (i === gpu.u32(0) && own !== gpu.u32(1))) {
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(2));
    }
  }
  scratchA[i] = own;
  gpu.barrier.block();
  let offset = gpu.u32(1); let flip = gpu.u32(0);
  while (offset < width) {
    let a = gpu.u32(0); let b = gpu.u32(0);
    if (flip === gpu.u32(0)) {
      a = scratchA[i]; if (lane >= offset) b = scratchA[i - offset];
    } else {
      a = scratchB[i]; if (lane >= offset) b = scratchB[i - offset];
    }
    let sum = a + b;
    if (a > gpu.u32(4294967295) - b) {
      sum = gpu.u32(4294967295);
      gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(3));
    }
    if (flip === gpu.u32(0)) scratchB[i] = sum; else scratchA[i] = sum;
    gpu.barrier.block();
    flip = gpu.u32(1) - flip; offset = offset * gpu.u32(2);
  }
  let inclusive = scratchA[i]; if (flip !== gpu.u32(0)) inclusive = scratchB[i];
  prefix[i] = inclusive - own;
  if (lane === width - gpu.u32(1)) sums[base / width] = inclusive;
}
function addCarries(prefix, parentPrefix, activeCount, status, capacity, divisor, width) {
  const i = gpu.thread.globalX(); const original = activeCount[gpu.u32(0)];
  if (original > capacity) return;
  let n = original / divisor; if (original % divisor !== gpu.u32(0)) n++;
  if (i >= n) return;
  const a = prefix[i]; const b = parentPrefix[i / width];
  if (a > gpu.u32(4294967295) - b) {
    gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(3)); return;
  }
  prefix[i] = a + b;
}
function emitSegments(heads, lengths, prefix, headTotal, ids, representatives, compactLengths,
  activeCount, status, groupCount, requiredGroups, capacity, groupCapacity) {
  const i = gpu.thread.globalX();
  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) return;
  const n = activeCount[gpu.u32(0)]; if (n > capacity) return;
  const groups = headTotal[gpu.u32(0)];
  if (i === gpu.u32(0)) requiredGroups[gpu.u32(0)] = groups;
  if (groups > groupCapacity) {
    if (i === gpu.u32(0)) gpu.atomic.cas(status, gpu.u32(0), gpu.u32(0), gpu.u32(4));
    return;
  }
  if (i === gpu.u32(0)) groupCount[gpu.u32(0)] = groups;
  if (i >= n) return;
  ids[i] = prefix[i] + heads[i] - gpu.u32(1);
  if (heads[i] === gpu.u32(1)) {
    representatives[prefix[i]] = i; compactLengths[prefix[i]] = lengths[i];
  }
}
function finishSegments(lengthTotal, status, totalLength) {
  if (gpu.thread.globalX() === gpu.u32(0) && gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) === gpu.u32(0)) {
    totalLength[gpu.u32(0)] = lengthTotal[gpu.u32(0)];
  }
}
`,
  functions: [
    kernel('resetSegments', ['status', 'groupCount', 'requiredGroups', 'totalLength']),
    kernel('scanTiles', ['input', 'scratchA', 'scratchB', 'prefix', 'sums', 'activeCount', 'status'], ['capacity', 'divisor', 'validateHeads']),
    kernel('addCarries', ['prefix', 'parentPrefix', 'activeCount', 'status'], ['capacity', 'divisor', 'width']),
    kernel('emitSegments', ['heads', 'lengths', 'prefix', 'headTotal', 'ids', 'representatives', 'compactLengths', 'activeCount', 'status', 'groupCount', 'requiredGroups'], ['capacity', 'groupCapacity']),
    kernel('finishSegments', ['lengthTotal', 'status', 'totalLength']),
  ],
};
