const ptr = name => ({ name, type: 'ptr<u32>' });
const scalar = name => ({ name, type: 'u32' });
const kernel = (name, pointers, scalars = []) => ({ name, kind: 'kernel', returns: 'void', parameters: [...pointers.map(ptr), ...scalars.map(scalar)] });

export const CHECKED_SCAN_U32_STATUS = Object.freeze({ OK: 0, INVALID_EXTENT: 1, INVALID_HEAD: 2,
  SUM_OVERFLOW: 3, GROUP_CAPACITY_EXHAUSTED: 4, UPSTREAM_FAILED: 5 });
export const checkedScanU32Program = {
  compile: { headerProfile: 'cuda-cccl' },
  source: `
function prepareCheckedScan(status, upstreamStatus, activeCount, safeCount, nextCount, publishedCount, requiredGroups, publishedTotal, capacity) {
  if (gpu.thread.globalX() === gpu.u32(0)) {
    status[gpu.u32(0)] = gpu.u32(0); safeCount[gpu.u32(0)] = gpu.u32(0);
    nextCount[gpu.u32(0)] = gpu.u32(0); publishedCount[gpu.u32(0)] = gpu.u32(0);
    requiredGroups[gpu.u32(0)] = gpu.u32(0); publishedTotal[gpu.u32(0)] = gpu.u32(0);
    if (upstreamStatus[gpu.u32(0)] !== gpu.u32(0)) { status[gpu.u32(0)] = gpu.u32(5); return; }
    const n = activeCount[gpu.u32(0)];
    if (n > capacity) { status[gpu.u32(0)] = gpu.u32(1); return; }
    safeCount[gpu.u32(0)] = n;
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
function finishCheckedScan(localPrefix, localTotal, safeCount, status, prefix, total, outputCount) {
  const i = gpu.thread.globalX();
  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) return;
  const n = safeCount[gpu.u32(0)];
  if (i === gpu.u32(0)) { outputCount[gpu.u32(0)] = n; total[gpu.u32(0)] = localTotal[gpu.u32(0)]; }
  if (i < n) prefix[i] = localPrefix[i];
}
function finishSegmentOffsets(lengthTotal, status, nextCount, selectedLengths, lengthPrefix,
  compactLengths, offsets, groupCount, totalLength) {
  const i = gpu.thread.globalX();
  if (gpu.atomic.loadRelaxedDevice(status, gpu.u32(0)) !== gpu.u32(0)) return;
  const n = nextCount[gpu.u32(0)];
  if (i === gpu.u32(0)) { groupCount[gpu.u32(0)] = n; totalLength[gpu.u32(0)] = lengthTotal[gpu.u32(0)]; }
  if (i < n) { compactLengths[i] = selectedLengths[i]; offsets[i] = lengthPrefix[i]; }
}
`,
  functions: [
    kernel('prepareCheckedScan', ['status', 'upstreamStatus', 'activeCount', 'safeCount', 'nextCount', 'publishedCount', 'requiredGroups', 'publishedTotal'], ['capacity']),
    kernel('scanTiles', ['input', 'scratchA', 'scratchB', 'prefix', 'sums', 'activeCount', 'status'], ['capacity', 'divisor', 'validateHeads']),
    kernel('addCarries', ['prefix', 'parentPrefix', 'activeCount', 'status'], ['capacity', 'divisor', 'width']),
    kernel('emitSegments', ['heads', 'lengths', 'prefix', 'headTotal', 'ids', 'representatives', 'compactLengths', 'activeCount', 'status', 'groupCount', 'requiredGroups'], ['capacity', 'groupCapacity']),
    kernel('finishCheckedScan', ['localPrefix', 'localTotal', 'safeCount', 'status', 'prefix', 'total', 'outputCount']),
    kernel('finishSegmentOffsets', ['lengthTotal', 'status', 'nextCount', 'selectedLengths', 'lengthPrefix', 'compactLengths', 'offsets', 'groupCount', 'totalLength']),
  ],
};
