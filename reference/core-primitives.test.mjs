import test from 'node:test';
import assert from 'node:assert/strict';
import {
  gather,
  independentLexicographicIndexOrder,
  reduceByKey,
  reduceUnsigned,
  runLengthEncode,
  scanUnsigned,
  selectIndices,
  stableLexicographicIndexOrder,
  stableRadixSortPairs,
  validateDeviceExtent,
} from './core-primitives.mjs';

const B = (...values) => values.map(BigInt);

test('exclusive and inclusive u32 scan preserve exact order', () => {
  assert.deepEqual(scanUnsigned(B(1, 2, 3), { width: 32, mode: 'exclusive', init: 0n }), B(0, 1, 3));
  assert.deepEqual(scanUnsigned(B(1, 2, 3), { width: 32, mode: 'inclusive' }), B(1, 3, 6));
});

test('u32 addition semantics wrap explicitly rather than using JS bitwise coercion', () => {
  assert.deepEqual(scanUnsigned([0xffffffffn, 1n], { width: 32, mode: 'inclusive' }), [0xffffffffn, 0n]);
  assert.equal(reduceUnsigned([0xffffffffn, 2n], { width: 32, op: 'add', init: 0n }), 1n);
});

test('u64 reference path stays exact beyond Number safe integer range', () => {
  const a = 1n << 60n;
  const b = (1n << 59n) + 7n;
  assert.deepEqual(scanUnsigned([a, b], { width: 64, mode: 'inclusive' }), [a, a + b]);
});

test('active extent processes only the active prefix', () => {
  assert.deepEqual(scanUnsigned(B(1, 2, 100, 200), { width: 32, mode: 'inclusive', active: 2, capacity: 4 }), B(1, 3));
  assert.throws(() => scanUnsigned(B(1, 2), { width: 32, active: 3, capacity: 2 }), /active exceeds capacity/);
});

test('selectIndices is stable and reports capacity without truncation', () => {
  const selected = selectIndices([1, 0, 1, 1, 0], { indexWidth: 32 });
  assert.equal(selected.status, 'ok');
  assert.deepEqual(selected.indices, B(0, 2, 3));
  assert.equal(selected.outputCount, 3n);

  const exhausted = selectIndices([1, 1, 1], { outputCapacity: 2 });
  assert.equal(exhausted.status, 'capacity-exhausted');
  assert.deepEqual(exhausted.indices, []);
  assert.equal(exhausted.outputCount, 3n);
});

test('selectIndices rejects non-boolean flag values', () => {
  assert.throws(() => selectIndices([0, 2, 1]), /must be 0 or 1/);
});

test('gather preserves index order and allows duplicate sources', () => {
  assert.deepEqual(gather(['a', 'b', 'c'], B(2, 0, 2)), ['c', 'a', 'c']);
  assert.throws(() => gather(['a'], [1n]), /outside input length/);
});

test('radix sort pairs is stable across equal keys', () => {
  const result = stableRadixSortPairs(B(5, 2, 5, 2, 5), B(10, 11, 12, 13, 14), { width: 32 });
  assert.deepEqual(result.keys, B(2, 2, 5, 5, 5));
  assert.deepEqual(result.indices, B(11, 13, 10, 12, 14));
});

test('radix sort honors selected bit range without disturbing stable ties', () => {
  const keys = [0b1001n, 0b0011n, 0b1010n, 0b0000n];
  const result = stableRadixSortPairs(keys, B(0, 1, 2, 3), { width: 32, beginBit: 0, endBit: 2 });
  assert.deepEqual(result.indices, B(3, 0, 2, 1));
});

test('runLengthEncode only groups adjacent equal keys', () => {
  const rle = runLengthEncode(B(1, 1, 2, 1, 1, 1), { width: 32 });
  assert.equal(rle.status, 'ok');
  assert.deepEqual(rle.uniqueKeys, B(1, 2, 1));
  assert.deepEqual(rle.runLengths, B(2, 1, 3));
  assert.equal(rle.outputRunCount, 3n);
});

test('runLengthEncode reports output capacity without partial semantic output', () => {
  const rle = runLengthEncode(B(1, 2, 3), { outputCapacity: 2 });
  assert.equal(rle.status, 'capacity-exhausted');
  assert.deepEqual(rle.uniqueKeys, []);
  assert.deepEqual(rle.runLengths, []);
  assert.equal(rle.outputRunCount, 3n);
});

test('reduceByKey reduces adjacent runs only', () => {
  const reduced = reduceByKey(B(1, 1, 2, 2, 2, 1), B(3, 4, 5, 6, 7, 8), { op: 'add' });
  assert.equal(reduced.status, 'ok');
  assert.deepEqual(reduced.uniqueKeys, B(1, 2, 1));
  assert.deepEqual(reduced.aggregates, B(7, 18, 8));
});

test('stable repeated word passes equal independent wide-key lexicographic oracle', () => {
  const words = [
    B(2, 1, 2, 1, 2, 1),
    B(0, 9, 0, 8, 0, 8),
    B(5, 1, 4, 7, 4, 6),
  ];
  assert.deepEqual(stableLexicographicIndexOrder(words), independentLexicographicIndexOrder(words));
});

test('wide-key composition survives many deterministic duplicate-heavy fixtures', () => {
  let state = 0x12345678n;
  const next = () => {
    state = (1664525n * state + 1013904223n) & 0xffffffffn;
    return state;
  };
  for (let fixture = 0; fixture < 100; fixture += 1) {
    const rows = 1 + Number(next() % 64n);
    const cols = 1 + Number(next() % 6n);
    const words = Array.from({ length: cols }, () =>
      Array.from({ length: rows }, () => next() % 17n));
    assert.deepEqual(
      stableLexicographicIndexOrder(words),
      independentLexicographicIndexOrder(words),
      `fixture ${fixture}`,
    );
  }
});

test('device extent model rejects count beyond declared capacity', () => {
  assert.deepEqual(validateDeviceExtent(4n, 4n, 32), { status: 'ok', active: 4n });
  assert.deepEqual(validateDeviceExtent(5n, 4n, 32), { status: 'invalid-extent', active: null });
});
