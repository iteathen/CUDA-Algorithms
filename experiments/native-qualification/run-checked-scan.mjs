import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { openCudaRuntime } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { segmentOffsetsU32Requirements } from 'cuda-algorithms';
import { createPublicSegmentFixture as createExperiment } from '../../test/checked-scan-fixtures.mjs';
import { reference } from '../segment-scan/reference.mjs';
import { qualifyScanControls } from './checked-scan-controls.mjs';

const mode = process.argv[2] ?? 'portable'; assert(['native', 'portable'].includes(mode));
const native = mode === 'native';
const maxRequirement = segmentOffsetsU32Requirements({ inputCapacity: 262144, groupCapacity: 262144 });
const deviceBytes = maxRequirement.workspaceBytes + Object.values(maxRequirement.bindings).reduce((n, r) => n + r.elements * 4, 0);
const maximum = { deviceBytes, upperBoundBytes: deviceBytes + 256 * 1024 ** 2 };
const options = { compiler: true, driver: { execution: { maxPendingGpuOperations: 2 }, memory: { maxDeviceBytes: maximum.upperBoundBytes,
  maxAllocationBytes: 64 * 1024 ** 2, maxTransferBytes: 64 * 1024 ** 2 } } };
const runtime = native ? await openCudaRuntime(options) : await openCudaRuntimeForTesting(options);
const results = []; const samples = []; const setups = []; const errors = []; let activePlan; let scanControls;
const check = (out, expected) => {
  if (!native) return;
  assert.equal(out.status, expected.status);
  if (expected.requiredGroups !== undefined) assert.equal(out.requiredGroups, expected.requiredGroups);
  if (expected.status) return;
  for (const k of ['groups', 'totalLength']) assert.equal(out[k], expected[k], k);
  for (const k of ['ids', 'representatives', 'lengths', 'offsets']) assert.deepEqual(Array.from(out[k]), expected[k], k);
};
async function use(capacity, blockSize, groupCapacity, action) {
  activePlan = await createExperiment(runtime, capacity, blockSize, groupCapacity);
  setups.push({ capacity, blockSize, setupMs: activePlan.setupMs, deviceBytes: activePlan.shape.deviceBytes, nodes: activePlan.shape.nodeCount });
  try { await action(activePlan); } finally { await activePlan.close(); activePlan = null; }
}
async function run(plan, name, heads, lengths, count = heads.length, groupCapacity = plan.shape.capacity, timed = false) {
  const expected = reference(heads, lengths, count, groupCapacity);
  const h = Uint32Array.from(heads); const l = Uint32Array.from(lengths);
  for (let pass = 0; pass < (timed && native ? 4 : 1); pass++) {
    const out = await plan.run(h, l, count, native); check(out, expected);
    if (timed) samples.push({ name, capacity: plan.shape.capacity, blockSize: plan.shape.blockSize,
      pass, warmup: pass === 0, uploadMs: out.uploadMs ?? null, submitWaitMs: out.submitWaitMs, readbackMs: out.readbackMs ?? null });
  }
  results.push({ name, count, blockSize: plan.shape.blockSize, expectedStatus: expected.status,
    groups: expected.groups ?? null, totalLength: expected.totalLength ?? null, nativeVerified: native });
}
try {
  for (const block of [64, 128, 256]) await use(513, block, 513, async plan => {
    for (const n of [0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513]) {
      const heads = Array.from({ length: n }, (_, i) => i % 3 === 0 ? 1 : 0);
      const lengths = Array.from({ length: n }, (_, i) => i % 19);
      await run(plan, `extent-${n}`, heads, lengths);
    }
    await run(plan, 'all-equal', [1, 0, 0, 0], [19, 0xffffffff, 0xffffffff, 0xffffffff]);
    await run(plan, 'all-distinct-and-zero-length', [1, 1, 1, 1], [0, 7, 0, 2]);
    await run(plan, 'invalid-first-head', [0, 1], [1, 2]);
    await run(plan, 'invalid-flag', [1, 2], [1, 2]);
    await run(plan, 'invalid-extent', Array(513).fill(1), Array(513).fill(1), 514);
    await run(plan, 'exact-u32-limit', [1, 1], [0xfffffffe, 1]);
    await run(plan, 'u32-overflow', [1, 1], [0xffffffff, 1]);
    const heads = Array(513).fill(1); const lengths = Array(513).fill(0); lengths[0] = 0xffffffff; lengths[512] = 1;
    await run(plan, 'cross-tile-overflow', heads, lengths);
    await run(plan, 'valid-reuse-after-errors', [1, 0, 1], [2, 0xffffffff, 9]);
  });
  await use(8, 128, 1, async plan => {
    await run(plan, 'group-capacity', [1, 1, 1], [1, 1, 1], 3, 1);
    await run(plan, 'capacity-recovery', [1, 0, 0], [5, 0, 0], 3, 1);
  });
  await use(8, 128, 0, async plan => {
    await run(plan, 'zero-group-capacity-empty', [], [], 0, 0);
    await run(plan, 'zero-group-capacity-nonempty', [1], [1], 1, 0);
  });
  const fixture = JSON.parse(readFileSync(new URL('../segment-scan/fixtures/oqs-cut-five.json', import.meta.url)));
  await use(128, 128, 128, async plan => {
    const expected = reference(fixture.heads, fixture.lengths);
    assert.equal(expected.groups, fixture.expectedGroups); assert.equal(expected.totalLength, fixture.expectedTotalLength);
    await run(plan, fixture.name, fixture.heads, fixture.lengths, fixture.heads.length, 128, true);
    // Separate consumer supplies null-normalized row groups; irrelevant payloads do not affect heads.
    const rows = [null, null, 'alpha', 'alpha', 'beta', 'gamma', 'gamma'];
    const heads = rows.map((v, i) => i === 0 || v !== rows[i - 1] ? 1 : 0);
    await run(plan, 'nullable-row-groups', heads, [0, 999, 5, 123, 4, 5, 456]);
  });
  for (const n of [8192, 65536, 262144]) await use(n, 128, n, async plan => {
    const heads = Array.from({ length: n }, (_, i) => i % 7 === 0 ? 1 : 0);
    const lengths = Array.from({ length: n }, (_, i) => i % 31);
    await run(plan, `scaling-${n}`, heads, lengths, n, n, true);
    // All heads exercise length-scan hierarchy over the complete input extent too.
    await run(plan, `all-distinct-${n}`, Array(n).fill(1), lengths);
    console.error(`[segment-scan] ${n} complete`);
  });
  scanControls = await qualifyScanControls(runtime, native);
} catch (e) { errors.push(e); }
finally {
  if (activePlan) try { await activePlan.close(); } catch (e) { errors.push(e); }
  try { assert.equal((await runtime.close()).graceful, true); } catch (e) { errors.push(e); }
}
if (errors.length) throw new AggregateError(errors, 'segment scan experiment failed');
console.log(JSON.stringify({ outcome: native ? 'native-checked-scan-api-pass' : 'portable-checked-scan-api-submit-pass',
  nativeVerified: native, maintainedCandidateApi: true, gpuEqualityOrOrdering: false, payloadCopy: false,
  maximumCapacity: 262144, maximumDeviceBytes: maximum.deviceBytes, upperBoundBytes: maximum.upperBoundBytes,
  results, samples, setups, scanControls, cleanup: 'graceful' }, null, 2));
