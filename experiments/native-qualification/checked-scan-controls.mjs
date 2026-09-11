import assert from 'node:assert/strict';
import { createCheckedExclusiveScanU32Plan, checkedScanU32Requirements, createSegmentOffsetsU32Plan,
  segmentOffsetsU32Requirements, createStableSelectIndicesU32Plan } from 'cuda-algorithms';
import { allocateBindings } from '../../test/checked-scan-fixtures.mjs';

export async function qualifyScanControls(runtime, numerical) {
  const controls = []; const benchmark = [];
  for (const blockSize of [64, 128, 256]) {
    const options = { inputCapacity: 513, blockSize };
    const plan = await createCheckedExclusiveScanU32Plan(runtime, options); let b;
    try {
      b = await allocateBindings(runtime, checkedScanU32Requirements(options));
      const cases = [0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513].map(n =>
        ({ name: `scan-extent-${n}`, values: Array.from({ length: n }, (_, i) => i % 17), count: n, status: 0 }));
      const cross = Array(513).fill(0); cross[0] = 0xffffffff; cross[512] = 1;
      cases.push({ name: 'scan-max-u32', values: [0xfffffffe, 1, 0], count: 3, status: 0 },
        { name: 'scan-overflow', values: [0xffffffff, 1], count: 2, status: 3 },
        { name: 'scan-cross-tile-overflow', values: cross, count: 513, status: 3 },
        { name: 'scan-invalid-extent', values: [1], count: 514, status: 1 },
        { name: 'scan-upstream-failure-precedes-invalid-extent', values: [1], count: 0xffffffff, upstream: 9, status: 5 },
        { name: 'scan-recovery', values: [3, 0, 7], count: 3, status: 0 });
      for (const c of cases) {
        const values = new Uint32Array(513).fill(0xffffffff); values.set(c.values);
        await b.write('input', values); await b.write('activeCount', [c.count]); await b.write('upstreamStatus', [c.upstream ?? 0]);
        const op = await plan.submit(b.bindings);
        try {
          assert.equal((await op.wait()).status, 'completed');
          if (numerical) {
            assert.equal((await b.read('status'))[0], c.status, c.name);
            assert.equal((await b.read('upstreamStatus'))[0], c.upstream ?? 0);
            if (c.status) { assert.equal((await b.read('outputCount'))[0], 0); assert.equal((await b.read('total'))[0], 0); }
            else {
              let total = 0n; const prefix = c.values.map(v => { const p = Number(total); total += BigInt(v); return p; });
              assert.deepEqual(Array.from(await b.read('prefix', c.count)), prefix);
              assert.equal((await b.read('total'))[0], Number(total)); assert.equal((await b.read('outputCount'))[0], c.count);
            }
          }
          controls.push({ name: c.name, blockSize, nativeVerified: numerical });
        } finally { await op.close(); }
      }
    } finally { await plan.close(); if (b) await b.close(); }
  }

  // Two plans, one explicit operation edge, zero intermediate reads/waits.
  const options = { inputCapacity: 8 };
  const first = await createCheckedExclusiveScanU32Plan(runtime, options);
  let second; let a; let b; let totalView;
  try {
    second = await createCheckedExclusiveScanU32Plan(runtime, options);
    a = await allocateBindings(runtime, checkedScanU32Requirements(options));
    b = await allocateBindings(runtime, checkedScanU32Requirements(options));
    totalView = await a.allocations.total.memory.view({ dtype: 'u32', elementCount: 1, access: 'read-write' });
    for (const fail of [false, true, false]) {
      await a.write('input', fail ? [0xffffffff, 1] : [2, 0, 3]); await a.write('activeCount', [fail ? 2 : 3]); await a.write('upstreamStatus', [0]);
      await b.write('input', [10, 20, 30, 40, 50, 60, 70, 80]);
      let op1; let op2;
      try {
        op1 = await first.submit({ ...a.bindings, total: totalView });
        op2 = await second.submit({ ...b.bindings, activeCount: totalView, upstreamStatus: a.bindings.status }, { after: op1 });
        assert.equal((await op2.wait()).status, 'completed'); assert.equal((await op1.wait()).status, 'completed');
        if (numerical) {
          assert.equal((await b.read('status'))[0], fail ? 5 : 0);
          assert.equal((await b.read('outputCount'))[0], fail ? 0 : 5);
          assert.equal((await b.read('total'))[0], fail ? 0 : 150);
          if (!fail) assert.deepEqual(Array.from(await b.read('prefix', 5)), [0, 10, 30, 60, 100]);
        }
        controls.push({ name: fail ? 'device-chained-upstream-failure' : 'device-chained-count', nativeVerified: numerical });
      } finally {
        if (op2) { await op2.wait(); await op2.close(); }
        if (op1) { await op1.wait(); await op1.close(); }
      }
    }
  } finally { if (second) await second.close(); await first.close(); if (totalView) await totalView.close(); if (b) await b.close(); if (a) await a.close(); }

  // Same representative set; new plan also computes IDs, checked lengths and offsets.
  const n = 8192; const heads = Array.from({ length: n }, (_, i) => i % 7 === 0 ? 1 : 0);
  const expected = heads.flatMap((h, i) => h ? [i] : []);
  const req = segmentOffsetsU32Requirements({ inputCapacity: n, groupCapacity: n });
  const modern = await createSegmentOffsetsU32Plan(runtime, { inputCapacity: n, groupCapacity: n });
  let legacy; let modernBuffers; let legacyBuffers;
  try {
    legacy = await createStableSelectIndicesU32Plan(runtime, { inputCapacity: n, outputCapacity: n });
    modernBuffers = await allocateBindings(runtime, req);
    legacyBuffers = await allocateBindings(runtime, { bindings: Object.fromEntries([
      ['flags', n, 'read'], ['prefix', n, 'read-write'], ['activeCount', 1, 'read'],
      ['outputIndices', n, 'write'], ['outputCount', 1, 'write'], ['status', 1, 'read-write'],
    ].map(([name, elements, access]) => [name, { elements, access }])) });
    await modernBuffers.write('heads', heads); await modernBuffers.write('lengths', heads.map((_, i) => i % 31));
    await modernBuffers.write('activeCount', [n]); await modernBuffers.write('upstreamStatus', [0]);
    await legacyBuffers.write('flags', heads); await legacyBuffers.write('activeCount', [n]);
    for (let pass = 0; pass < (numerical ? 4 : 1); pass++) for (const name of pass % 2 ? ['modern', 'legacy'] : ['legacy', 'modern']) {
      const plan = name === 'modern' ? modern : legacy; const buffers = name === 'modern' ? modernBuffers : legacyBuffers;
      const start = performance.now(); const op = await plan.submit(buffers.bindings);
      try {
        assert.equal((await op.wait()).status, 'completed'); const submitWaitMs = performance.now() - start;
        if (numerical) {
          assert.equal((await buffers.read('status'))[0], 0);
          assert.equal((await buffers.read(name === 'modern' ? 'groupCount' : 'outputCount'))[0], expected.length);
          assert.deepEqual(Array.from(await buffers.read(name === 'modern' ? 'representatives' : 'outputIndices', expected.length)), expected);
        }
        benchmark.push({ name, n, pass, warmup: pass === 0, submitWaitMs, nativeVerified: numerical });
      } finally { await op.close(); }
    }
    for (const fail of [true, false]) {
      await modernBuffers.write('upstreamStatus', [fail ? 9 : 0]);
      await modernBuffers.write('activeCount', [fail ? 0xffffffff : n]);
      const op = await modern.submit(modernBuffers.bindings);
      try {
        assert.equal((await op.wait()).status, 'completed');
        if (numerical) {
          assert.equal((await modernBuffers.read('status'))[0], fail ? 5 : 0);
          assert.equal((await modernBuffers.read('groupCount'))[0], fail ? 0 : expected.length);
          if (fail) { assert.equal((await modernBuffers.read('requiredGroups'))[0], 0); assert.equal((await modernBuffers.read('totalLength'))[0], 0); }
        }
        controls.push({ name: fail ? 'segment-upstream-failure' : 'segment-upstream-recovery', nativeVerified: numerical });
      } finally { await op.close(); }
    }
  } finally { if (legacy) await legacy.close(); await modern.close(); if (legacyBuffers) await legacyBuffers.close(); if (modernBuffers) await modernBuffers.close(); }
  return { controls, benchmark, interpretation: 'Legacy performs stable selection only; modern additionally computes group IDs, compact lengths and checked offsets. Submit/wait includes orchestration; inputs stay resident.' };
}
