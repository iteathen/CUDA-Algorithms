import { compileDeviceProgram } from 'cuda-js';
import { checkedScanU32Program, CHECKED_SCAN_U32_STATUS } from '../device/checked-scan-u32-program.mjs';
import { closeResources, requireU32View, rejectViewWriteConflicts } from './common.mjs';

const role = (elements, access) => Object.freeze({ elements, access });
export function checkedShape(options = {}, segments = false) {
  const { inputCapacity: capacity, blockSize = 128, maxWorkspaceBytes } = options;
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 262144) throw new RangeError('inputCapacity must be 1..262144');
  if (![64, 128, 256].includes(blockSize)) throw new RangeError('blockSize must be 64, 128 or 256');
  const groupCapacity = segments ? options.groupCapacity : 0;
  if (segments && (!Number.isInteger(groupCapacity) || groupCapacity < 0 || groupCapacity > capacity)) throw new RangeError('groupCapacity must be 0..inputCapacity');
  const bindings = { activeCount: role(1, 'read'), upstreamStatus: role(1, 'read'), status: role(1, 'read-write') };
  if (segments) Object.assign(bindings, { heads: role(capacity, 'read'), lengths: role(capacity, 'read'),
    groupIds: role(capacity, 'write'), representatives: role(Math.max(1, groupCapacity), 'write'),
    compactLengths: role(Math.max(1, groupCapacity), 'write'), offsets: role(Math.max(1, groupCapacity), 'write'),
    groupCount: role(1, 'write'), requiredGroups: role(1, 'write'), totalLength: role(1, 'write') });
  else Object.assign(bindings, { input: role(capacity, 'read'), prefix: role(capacity, 'write'), total: role(1, 'write'), outputCount: role(1, 'write') });
  const workspaceSizes = { safeCount: 1, nextCount: 1 };
  if (segments) workspaceSizes.selectedLengths = Math.max(1, groupCapacity); else workspaceSizes.unusedRequired = 1;
  const levels = []; let n = capacity; let divisor = 1;
  do {
    const blocks = Math.ceil(n / blockSize); const padded = blocks * blockSize;
    levels.push(Object.freeze({ n, blocks, padded, divisor })); n = blocks; divisor *= blockSize;
  } while (n > 1);
  for (const phase of segments ? ['head', 'length'] : ['value']) levels.forEach((l, i) => {
    for (const part of ['a', 'b', 'prefix']) workspaceSizes[`${phase}${i}${part}`] = l.padded;
    workspaceSizes[`${phase}${i}sums`] = l.blocks;
  });
  const workspaceBytes = Object.values(workspaceSizes).reduce((a, b) => a + b, 0) * 4;
  const preparedNodeCount = segments ? 3 + 2 * (2 * levels.length - 1) : 2 + 2 * levels.length - 1;
  if (preparedNodeCount > 32) throw new RangeError('prepared node ceiling exceeded');
  if (maxWorkspaceBytes !== undefined && (!Number.isSafeInteger(maxWorkspaceBytes) || maxWorkspaceBytes < workspaceBytes)) throw new RangeError('insufficient or invalid maxWorkspaceBytes');
  return { capacity, blockSize, groupCapacity, levels, workspaceSizes,
    requirements: Object.freeze({ inputCapacity: capacity, ...(segments ? { groupCapacity } : {}), blockSize,
      workspaceBytes, preparedNodeCount, bindings: Object.freeze(bindings) }) };
}

export async function createCheckedPlan(runtime, options, segments, contract) {
  const s = checkedShape(options, segments); const resources = []; const internal = {};
  let closed = false; let submitting = false; let lastOperation; let closing;
  try {
    const compiled = await compileDeviceProgram(runtime, checkedScanU32Program);
    const artifact = compiled.linker?.artifact ?? compiled.compiler?.artifact;
    if (!artifact || !['ptx', 'cubin'].includes(artifact.format)) throw new Error('missing checked-scan executable');
    const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes }); resources.push(module);
    const functions = {};
    for (const k of compiled.deviceProgram.kernels) {
      functions[k.name] = await module.getFunction({ name: k.functionName, parameters: k.parameters }); resources.push(functions[k.name]);
    }
    for (const [name, count] of Object.entries(s.workspaceSizes)) {
      const memory = await runtime.allocateDevice({ byteLength: count * 4 }); resources.push(memory);
      const view = await memory.view({ dtype: 'u32', elementCount: count, access: 'read-write' }); resources.push(view);
      internal[`w_${name}`] = view;
    }
    const nodes = [];
    function node(name, roles, scalars, gridX) {
      const descriptor = checkedScanU32Program.functions.find(f => f.name === name);
      const id = `stage${nodes.length}`;
      nodes.push({ id, ...(nodes.length ? { after: [nodes.at(-1).id] } : {}), function: functions[name],
        grid: { x: gridX, y: 1, z: 1 }, block: { x: s.blockSize, y: 1, z: 1 },
        arguments: descriptor.parameters.map(p => p.type.startsWith('ptr') ? { binding: roles[p.name] } : scalars[p.name]),
        accesses: descriptor.parameters.flatMap((p, argumentIndex) => {
          if (!p.type.startsWith('ptr')) return [];
          const bindingName = roles[p.name]; const privateRole = bindingName.startsWith('w_');
          const spec = privateRole ? role(s.workspaceSizes[bindingName.slice(2)], 'read-write') : s.requirements.bindings[bindingName];
          return [{ argumentIndex, byteOffset: 0, byteLength: spec.elements * 4, mode: spec.access }];
        }),
      });
    }
    node('prepareCheckedScan', { status: 'status', upstreamStatus: 'upstreamStatus', activeCount: 'activeCount',
      safeCount: 'w_safeCount', nextCount: 'w_nextCount', publishedCount: segments ? 'groupCount' : 'outputCount',
      requiredGroups: segments ? 'requiredGroups' : 'w_unusedRequired', publishedTotal: segments ? 'totalLength' : 'total' }, { capacity: s.capacity }, 1);
    function scan(phase, input, extent, heads) {
      s.levels.forEach((l, i) => node('scanTiles', { input: i ? `w_${phase}${i - 1}sums` : input,
        scratchA: `w_${phase}${i}a`, scratchB: `w_${phase}${i}b`, prefix: `w_${phase}${i}prefix`, sums: `w_${phase}${i}sums`,
        activeCount: extent, status: 'status' }, { capacity: s.capacity, divisor: l.divisor, validateHeads: i === 0 && heads ? 1 : 0 }, l.blocks));
      for (let i = s.levels.length - 2; i >= 0; i--) {
        const l = s.levels[i];
        node('addCarries', { prefix: `w_${phase}${i}prefix`, parentPrefix: `w_${phase}${i + 1}prefix`, activeCount: extent, status: 'status' },
          { capacity: s.capacity, divisor: l.divisor, width: s.blockSize }, l.blocks);
      }
      return `w_${phase}${s.levels.length - 1}sums`;
    }
    if (segments) {
      const headTotal = scan('head', 'heads', 'w_safeCount', true);
      node('emitSegments', { heads: 'heads', lengths: 'lengths', prefix: 'w_head0prefix', headTotal,
        ids: 'groupIds', representatives: 'representatives', compactLengths: 'w_selectedLengths', activeCount: 'w_safeCount',
        status: 'status', groupCount: 'w_nextCount', requiredGroups: 'requiredGroups' },
        { capacity: s.capacity, groupCapacity: s.groupCapacity }, s.levels[0].blocks);
      const lengthTotal = scan('length', 'w_selectedLengths', 'w_nextCount', false);
      node('finishSegmentOffsets', { lengthTotal, status: 'status', nextCount: 'w_nextCount', selectedLengths: 'w_selectedLengths',
        lengthPrefix: 'w_length0prefix', compactLengths: 'compactLengths', offsets: 'offsets', groupCount: 'groupCount', totalLength: 'totalLength' }, {}, s.levels[0].blocks);
    } else {
      const localTotal = scan('value', 'input', 'w_safeCount', false);
      node('finishCheckedScan', { localPrefix: 'w_value0prefix', localTotal, safeCount: 'w_safeCount', status: 'status',
        prefix: 'prefix', total: 'total', outputCount: 'outputCount' }, {}, s.levels[0].blocks);
    }
    if (nodes.length !== s.requirements.preparedNodeCount) throw new Error('checked-scan node accounting mismatch');
    const dag = await runtime.prepareOperationDag({ nodes }); resources.push(dag);
    function requireIdle() {
      if (submitting || ['pending', 'orphaned'].includes(lastOperation?.state)) throw new Error('checked-scan plan has an in-flight or orphaned operation');
    }
    return Object.freeze({ kind: 'cuda-algorithms-plan', contract, family: segments ? 'segment-offsets' : 'checked-exclusive-scan',
      dtype: 'u32', ...s.requirements, status: CHECKED_SCAN_U32_STATUS,
      workspace: Object.freeze({ deviceBytes: s.requirements.workspaceBytes, owner: 'plan' }),
      aliasing: Object.freeze({ writeRolesRequireDisjointRanges: true, overlappingReadOnlyRangesAllowed: true, relationOwner: 'cuda-js:inspectDeviceViewRelation' }),
      async submit(bindings, { after } = {}) {
        if (closed) throw new Error('checked-scan plan is closed'); requireIdle();
        const normalized = Object.fromEntries(Object.entries(s.requirements.bindings).map(([name, spec]) =>
          [name, requireU32View(bindings?.[name], spec.elements, name, spec.access)]));
        rejectViewWriteConflicts(Object.entries(normalized).map(([label, view]) => ({ label, view, access: s.requirements.bindings[label].access })));
        submitting = true;
        try { lastOperation = await dag.submit({ bindings: { ...internal, ...normalized }, ...(after ? { after } : {}) }); return lastOperation; }
        finally { submitting = false; }
      },
      close() {
        if (closing) return closing;
        try { requireIdle(); } catch (e) { return Promise.reject(e); }
        closed = true; closing = closeResources(resources, 'checked-scan plan'); return closing;
      },
    });
  } catch (error) {
    try { await closeResources(resources, 'checked-scan construction'); }
    catch (cleanup) { throw new AggregateError([error, cleanup], 'checked-scan construction and cleanup failed'); }
    throw error;
  }
}
