import assert from 'node:assert/strict';
import { createSegmentOffsetsU32Plan, segmentOffsetsU32Requirements } from 'cuda-algorithms';

export async function allocateBindings(runtime, requirements) {
  const allocations = {}; const owned = [];
  try {
    for (const [name, spec] of Object.entries(requirements.bindings)) {
      const memory = await runtime.allocateDevice({ byteLength: spec.elements * 4 }); owned.push(memory);
      const view = await memory.view({ dtype: 'u32', elementCount: spec.elements, access: spec.access }); owned.push(view);
      allocations[name] = { memory, view, count: spec.elements };
    }
  } catch (e) { for (const r of owned.reverse()) await r.close(); throw e; }
  return {
    allocations, bindings: Object.fromEntries(Object.entries(allocations).map(([n, a]) => [n, a.view])),
    async write(name, values) {
      const a = allocations[name]; const data = new Uint32Array(a.count); data.set(values);
      await a.memory.write(new Uint8Array(data.buffer));
    },
    async read(name, n = 1) {
      if (!n) return new Uint32Array();
      const { bytes } = await allocations[name].memory.read({ byteLength: n * 4 });
      return new Uint32Array(bytes.buffer, bytes.byteOffset, n);
    },
    async close() { const errors = []; for (const r of [...owned].reverse()) try { await r.close(); } catch (e) { errors.push(e); }
      if (errors.length) throw new AggregateError(errors, 'fixture cleanup'); },
  };
}

// Qualification-only adapter preserving the original 72-fixture oracle surface.
export async function createPublicSegmentFixture(runtime, capacity, blockSize = 128, groupCapacity = capacity) {
  const options = { inputCapacity: capacity, blockSize, groupCapacity }; const r = segmentOffsetsU32Requirements(options);
  const start = performance.now(); const plan = await createSegmentOffsetsU32Plan(runtime, options); let b;
  try { b = await allocateBindings(runtime, r); } catch (e) { await plan.close(); throw e; }
  return {
    setupMs: performance.now() - start,
    shape: { capacity, blockSize, deviceBytes: r.workspaceBytes + Object.values(r.bindings).reduce((v, s) => v + s.elements * 4, 0), nodeCount: r.preparedNodeCount },
    async run(heads, lengths, count = heads.length, numerical = true) {
      const start = performance.now(); let operation;
      try {
        // Poison inactive tails so a hidden capacity scan cannot pass silently.
        const h = new Uint32Array(capacity).fill(2); h.set(heads);
        const l = new Uint32Array(capacity).fill(0xffffffff); l.set(lengths);
        await b.write('heads', h); await b.write('lengths', l); await b.write('activeCount', [count]); await b.write('upstreamStatus', [0]);
        const uploaded = performance.now(); operation = await plan.submit(b.bindings);
        assert.equal((await operation.wait()).status, 'completed'); const waited = performance.now();
        await operation.close(); operation = null;
        if (!numerical) return { submitWaitMs: waited - uploaded };
        const status = (await b.read('status'))[0]; const groups = (await b.read('groupCount'))[0];
        const requiredGroups = (await b.read('requiredGroups'))[0]; const totalLength = (await b.read('totalLength'))[0];
        assert(groups <= groupCapacity);
        if (status) { assert.equal(groups, 0); assert.equal(totalLength, 0); }
        const payload = status ? {} : { ids: await b.read('groupIds', count), representatives: await b.read('representatives', groups),
          lengths: await b.read('compactLengths', groups), offsets: await b.read('offsets', groups) };
        return { status, groups, requiredGroups, totalLength, ...payload, uploadMs: uploaded - start,
          submitWaitMs: waited - uploaded, readbackMs: performance.now() - waited };
      } finally { if (operation) await operation.close(); }
    },
    async close() { try { await plan.close(); } finally { await b.close(); } },
  };
}
