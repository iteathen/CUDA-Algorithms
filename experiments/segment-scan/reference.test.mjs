import test from 'node:test';
import assert from 'node:assert/strict';
import { reference } from './reference.mjs';
import { shape } from './plan.mjs';

test('segment reference preserves null/empty groups and checked u32 length truth', () => {
  assert.deepEqual(reference([1, 0, 1, 1], [0, 999, 3, 2]), {
    status: 0, groups: 3, requiredGroups: 3, totalLength: 5, ids: [0, 0, 1, 2],
    representatives: [0, 2, 3], lengths: [0, 3, 2], offsets: [0, 0, 3],
  });
  assert.equal(reference([], []).totalLength, 0);
  assert.equal(reference([1, 1], [0xffffffff, 0]).totalLength, 0xffffffff);
  assert.equal(reference([1, 1], [0xffffffff, 1]).status, 3);
  assert.deepEqual(reference([1, 1], [1, 1], 2, 1), { status: 4, requiredGroups: 2 });
});

test('all declared shapes fit the DAG and exact disjoint allocation accounting', () => {
  for (const n of [1, 63, 64, 65, 128, 513, 8192, 65536, 262144]) for (const b of [64, 128, 256]) {
    const s = shape(n, b);
    assert.equal(s.deviceBytes, Object.values(s.sizes).reduce((a, b) => a + b) * 4);
    assert(s.nodeCount <= 32); assert(s.deviceBytes < 16 * 1024 ** 2);
    assert(s.levels.every(l => l.padded >= l.n && l.padded % b === 0));
  }
  assert.throws(() => shape(262145)); assert.throws(() => shape(1, 96));
});
