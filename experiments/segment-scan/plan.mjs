import { compileDeviceProgram } from 'cuda-js';
import { program } from './program.mjs';

export function shape(capacity, blockSize = 128) {
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 262144) throw new RangeError('capacity outside experiment envelope');
  if (![64, 128, 256].includes(blockSize)) throw new RangeError('unsupported block size');
  const sizes = { heads: capacity, lengths: capacity, activeCount: 1, status: 1,
    ids: capacity, representatives: capacity, compactLengths: capacity, groupCount: 1, requiredGroups: 1, totalLength: 1 };
  const levels = []; let n = capacity; let divisor = 1;
  do {
    const blocks = Math.ceil(n / blockSize); const padded = blocks * blockSize;
    levels.push({ n, blocks, padded, divisor }); n = blocks; divisor *= blockSize;
  } while (n > 1);
  for (const phase of ['head', 'length']) levels.forEach((l, i) => {
    for (const role of ['a', 'b', 'prefix']) sizes[`${phase}${i}${role}`] = l.padded;
    sizes[`${phase}${i}sums`] = l.blocks;
  });
  const nodeCount = 3 + 2 * (2 * levels.length - 1);
  if (nodeCount > 32) throw new RangeError('DAG limit');
  const deviceBytes = Object.values(sizes).reduce((a, b) => a + b, 0) * 4;
  return { capacity, blockSize, sizes, levels, nodeCount, deviceBytes, upperBoundBytes: deviceBytes + 256 * 1024 ** 2 };
}

async function closeAll(resources) {
  const errors = [];
  for (const r of [...resources].reverse()) try { await r.close(); } catch (e) { errors.push(e); }
  if (errors.length) throw new AggregateError(errors, 'segment scan cleanup');
}

// Experiment-owned buffers; not an exported CUDA-Algorithms production plan.
export async function createExperiment(runtime, capacity, blockSize = 128, groupCapacity = capacity) {
  const s = shape(capacity, blockSize);
  if (!Number.isInteger(groupCapacity) || groupCapacity < 0 || groupCapacity > capacity) throw new RangeError('invalid group capacity');
  const owned = []; const buffers = {}; const started = performance.now(); let closed = false; let active = false;
  try {
    const compiled = await compileDeviceProgram(runtime, program);
    const artifact = compiled.linker?.artifact ?? compiled.compiler.artifact;
    const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes }); owned.push(module);
    const functions = {};
    for (const k of compiled.deviceProgram.kernels) {
      functions[k.name] = await module.getFunction({ name: k.functionName, parameters: k.parameters }); owned.push(functions[k.name]);
    }
    for (const [name, count] of Object.entries(s.sizes)) {
      const memory = await runtime.allocateDevice({ byteLength: count * 4 }); owned.push(memory);
      const view = await memory.view({ dtype: 'u32', elementCount: count, access: 'read-write' }); owned.push(view);
      buffers[name] = { memory, view };
    }
    const nodes = [];
    function node(kernel, bindings, scalars, grid) {
      const f = program.functions.find(f => f.name === kernel);
      const id = `stage${nodes.length}`;
      nodes.push({ id, ...(nodes.length ? { after: [nodes.at(-1).id] } : {}), function: functions[kernel],
        grid: { x: grid, y: 1, z: 1 }, block: { x: blockSize, y: 1, z: 1 },
        arguments: f.parameters.map(p => p.type.startsWith('ptr') ? { binding: bindings[p.name] } : scalars[p.name]),
        accesses: f.parameters.flatMap((p, argumentIndex) => p.type.startsWith('ptr') ? [{ argumentIndex,
          byteOffset: 0, byteLength: s.sizes[bindings[p.name]] * 4, mode: 'read-write' }] : []),
      });
    }
    const controls = { status: 'status', groupCount: 'groupCount', requiredGroups: 'requiredGroups', totalLength: 'totalLength' };
    node('resetSegments', controls, {}, 1);
    function scan(phase, input, extent, flags) {
      s.levels.forEach((l, i) => node('scanTiles', { input: i ? `${phase}${i - 1}sums` : input,
        scratchA: `${phase}${i}a`, scratchB: `${phase}${i}b`, prefix: `${phase}${i}prefix`, sums: `${phase}${i}sums`,
        activeCount: extent, status: 'status' }, { capacity, divisor: l.divisor, validateHeads: i === 0 && flags ? 1 : 0 }, l.blocks));
      for (let i = s.levels.length - 2; i >= 0; i--) {
        const l = s.levels[i];
        node('addCarries', { prefix: `${phase}${i}prefix`, parentPrefix: `${phase}${i + 1}prefix`, activeCount: extent, status: 'status' },
          { capacity, divisor: l.divisor, width: blockSize }, l.blocks);
      }
      return `${phase}${s.levels.length - 1}sums`;
    }
    const headTotal = scan('head', 'heads', 'activeCount', true);
    node('emitSegments', { ...controls, heads: 'heads', lengths: 'lengths', prefix: 'head0prefix', headTotal,
      ids: 'ids', representatives: 'representatives', compactLengths: 'compactLengths', activeCount: 'activeCount' },
      { capacity, groupCapacity }, s.levels[0].blocks);
    const lengthTotal = scan('length', 'compactLengths', 'groupCount', false);
    node('finishSegments', { lengthTotal, status: 'status', totalLength: 'totalLength' }, {}, 1);
    if (nodes.length !== s.nodeCount) throw new Error('node accounting mismatch');
    const dag = await runtime.prepareOperationDag({ nodes }); owned.push(dag);
    const bindings = Object.fromEntries(Object.entries(buffers).map(([n, b]) => [n, b.view]));
    const read = async (name, n = 1) => {
      if (!n) return new Uint32Array();
      const { bytes } = await buffers[name].memory.read({ byteLength: n * 4 });
      return new Uint32Array(bytes.buffer, bytes.byteOffset, n);
    };
    return {
      shape: s, setupMs: performance.now() - started,
      async run(heads, lengths, count = heads.length, numerical = true) {
        if (closed || active) throw new Error('experiment closed or active');
        if (!(heads instanceof Uint32Array) || !(lengths instanceof Uint32Array) || heads.length > capacity || lengths.length > capacity) throw new RangeError('input extent');
        if (!Number.isInteger(count) || count < 0 || count > 0xffffffff) throw new RangeError('invalid count encoding');
        active = true; let operation;
        try {
          const inputHeads = new Uint32Array(capacity).fill(2); inputHeads.set(heads);
          const inputLengths = new Uint32Array(capacity).fill(0xffffffff); inputLengths.set(lengths);
          const start = performance.now();
          for (const [name, data] of [['heads', inputHeads], ['lengths', inputLengths], ['activeCount', new Uint32Array([count])]]) {
            await buffers[name].memory.write(new Uint8Array(data.buffer));
          }
          const uploaded = performance.now();
          operation = await dag.submit({ bindings }); const terminal = await operation.wait();
          if (terminal.status !== 'completed') throw new Error(`runtime ${terminal.status}`);
          const waited = performance.now(); await operation.close(); operation = null;
          if (!numerical) return { portable: true, submitWaitMs: waited - uploaded };
          const status = (await read('status'))[0]; const requiredGroups = (await read('requiredGroups'))[0];
          const groups = (await read('groupCount'))[0]; const totalLength = (await read('totalLength'))[0];
          if (groups > capacity) throw new Error('invalid group count');
          const output = status ? {} : {
            ids: await read('ids', count), representatives: await read('representatives', groups),
            lengths: await read('compactLengths', groups), offsets: await read('length0prefix', groups),
          };
          return { status, requiredGroups, groups, totalLength, ...output,
            uploadMs: uploaded - start, submitWaitMs: waited - uploaded, readbackMs: performance.now() - waited };
        } finally { try { if (operation) await operation.close(); } finally { active = false; } }
      },
      async close() { if (closed) return; if (active) throw new Error('active experiment'); closed = true; await closeAll(owned); },
    };
  } catch (e) { try { await closeAll(owned); } catch (cleanup) { throw new AggregateError([e, cleanup], 'construction failed'); } throw e; }
}
