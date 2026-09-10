import test from 'node:test';
import assert from 'node:assert/strict';

import {
  RANKED_DERIVED_INVALID_U32,
  runRankedDerivedActivationEpoch,
} from './ranked-derived-activation.mjs';

const INVALID = RANKED_DERIVED_INVALID_U32;

test('ranked derived activation is duplicate-idempotent and shard invariant for an implicit dependency DAG', () => {
  const ranks = [0, 1, 1, 2, 2, 3, 3];
  const edges = new Map([
    [6, [4, 3, 4]],
    [5, [4, 2]],
    [4, [2, 1]],
    [3, [2, 1]],
  ]);
  const deriveTarget = (source, lane) => edges.get(source)?.[lane] ?? INVALID;

  const expected = runRankedDerivedActivationEpoch({
    ranks,
    activeIndices: [6, 5, 6],
    deriveTarget,
    maxEmissionsPerItem: 3,
    shardSize: 1,
  });

  assert.equal(expected.status, 'rank-complete');
  assert.deepEqual(expected.nextIndices, [2, 3, 4]);

  for (const shardSize of [2, 3, 16]) {
    const actual = runRankedDerivedActivationEpoch({
      ranks,
      activeIndices: [6, 5, 6],
      deriveTarget,
      maxEmissionsPerItem: 3,
      shardSize,
    });
    assert.deepEqual(actual.nextIndices, expected.nextIndices);
    assert.equal(actual.requiredCount, expected.requiredCount);
  }
});

test('the same reference serves an unrelated staged data-lineage consumer', () => {
  const ranks = [0, 0, 1, 1, 2, 2, 3];
  const parents = [
    [], [],
    [0], [1],
    [2, 3], [3],
    [4, 5],
  ];
  const deriveTarget = (source, lane) => parents[source]?.[lane] ?? INVALID;

  const actual = runRankedDerivedActivationEpoch({
    ranks,
    activeIndices: [6],
    deriveTarget,
    maxEmissionsPerItem: 2,
  });

  assert.equal(actual.status, 'rank-complete');
  assert.deepEqual(actual.nextIndices, [4, 5]);
});

test('rank descent, target bounds and output capacity fail truthfully', () => {
  assert.throws(
    () => runRankedDerivedActivationEpoch({
      ranks: [0, 1],
      activeIndices: [1],
      maxEmissionsPerItem: 1,
      deriveTarget: () => 1,
    }),
    /rank descent violated/,
  );

  assert.throws(
    () => runRankedDerivedActivationEpoch({
      ranks: [0, 1],
      activeIndices: [1],
      maxEmissionsPerItem: 1,
      deriveTarget: () => 2,
    }),
    /derived target index exceeds item universe/,
  );

  const capacity = runRankedDerivedActivationEpoch({
    ranks: [0, 0, 1],
    activeIndices: [2],
    maxEmissionsPerItem: 2,
    outputCapacity: 1,
    deriveTarget: (_source, lane) => lane,
  });
  assert.equal(capacity.status, 'capacity-yield');
  assert.equal(capacity.requiredCount, 2);
  assert.deepEqual(capacity.nextIndices, []);
});
