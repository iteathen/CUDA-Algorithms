import test from 'node:test';
import assert from 'node:assert/strict';
import { runRankedIndexClosure } from './ranked-index-closure.mjs';

function fixtureRunner({ ranks, edges, seeds, maxEmissionsPerItem, shardSize }) {
  return runRankedIndexClosure({
    itemCapacity: ranks.length,
    maxRank: Math.max(...ranks),
    seeds,
    rankOf: (index) => ranks[index],
    derive: (index) => edges[index],
    maxEmissionsPerItem,
    shardSize,
  });
}

const dependencyRanks = [0, 0, 1, 1, 2, 2, 3, 3, 4];
const dependencyEdges = [
  [], [],
  [0, 1], [1],
  [2, 3], [3],
  [4, 5], [5, 4],
  [6, 7],
];

test('ranked closure resolves a nested dependency index universe exactly', () => {
  const result = fixtureRunner({
    ranks: dependencyRanks,
    edges: dependencyEdges,
    seeds: [8],
    maxEmissionsPerItem: 2,
    shardSize: 2,
  });
  assert.deepEqual(result.activeIndices, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(result.activatedCount, 9);
  assert.equal(result.processedCount, 9);
  assert.ok(result.duplicateActivations > 0);
});

test('ranked closure is invariant to physical shard size', () => {
  const canonical = fixtureRunner({
    ranks: dependencyRanks,
    edges: dependencyEdges,
    seeds: [8],
    maxEmissionsPerItem: 2,
    shardSize: 1,
  });
  for (const shardSize of [2, 3, 4, 9, 64]) {
    const candidate = fixtureRunner({
      ranks: dependencyRanks,
      edges: dependencyEdges,
      seeds: [8],
      maxEmissionsPerItem: 2,
      shardSize,
    });
    assert.deepEqual(candidate.activeIndices, canonical.activeIndices, `shardSize=${shardSize}`);
    assert.deepEqual(candidate.activeByRank, canonical.activeByRank, `by-rank shardSize=${shardSize}`);
    assert.equal(candidate.derivedEdgeCount, canonical.derivedEdgeCount);
  }
});

test('the same ranked algebra handles recursive data-row lineage without importing row semantics', () => {
  // Consumer interpretation: indices are normalized-row IDs at descending ETL stages.
  const ranks = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3];
  const edges = [
    [], [], [],
    [0], [0, 2], [1, 2],
    [3, 4], [4, 5],
    [6], [7, 6],
  ];
  const result = fixtureRunner({ ranks, edges, seeds: [9], maxEmissionsPerItem: 2, shardSize: 3 });
  assert.deepEqual(result.activeIndices, [0, 1, 2, 3, 4, 5, 6, 7, 9]);
  assert.equal(result.activeIndices.includes(8), false);
});

test('duplicate seeds are idempotent index activation, not duplicate semantic work', () => {
  const result = fixtureRunner({
    ranks: dependencyRanks,
    edges: dependencyEdges,
    seeds: [8, 8, 8],
    maxEmissionsPerItem: 2,
    shardSize: 1,
  });
  assert.deepEqual(result.activeIndices, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(result.processedCount, 9);
  assert.ok(result.duplicateActivations >= 2);
});

test('rank monotonicity violations fail closed even when the target was already active', () => {
  assert.throws(
    () => runRankedIndexClosure({
      itemCapacity: 2,
      maxRank: 1,
      seeds: [1, 0],
      rankOf: (index) => [0, 1][index],
      derive: (index) => index === 1 ? [1] : [],
      maxEmissionsPerItem: 1,
      shardSize: 1,
    }),
    /rank monotonicity violation/,
  );
});

test('fanout and target bounds fail closed', () => {
  assert.throws(
    () => runRankedIndexClosure({
      itemCapacity: 3,
      maxRank: 1,
      seeds: [2],
      rankOf: (index) => [0, 0, 1][index],
      derive: () => [0, 1],
      maxEmissionsPerItem: 1,
      shardSize: 1,
    }),
    /exceeding maxEmissionsPerItem/,
  );

  assert.throws(
    () => runRankedIndexClosure({
      itemCapacity: 2,
      maxRank: 1,
      seeds: [1],
      rankOf: (index) => [0, 1][index],
      derive: () => [2],
      maxEmissionsPerItem: 1,
      shardSize: 1,
    }),
    /outside itemCapacity/,
  );
});

test('rank-zero-only closure supports a zero emission bound', () => {
  const result = runRankedIndexClosure({
    itemCapacity: 3,
    maxRank: 0,
    seeds: [0, 2],
    rankOf: () => 0,
    derive: () => [],
    maxEmissionsPerItem: 0,
    shardSize: 8,
  });
  assert.deepEqual(result.activeIndices, [0, 2]);
  assert.equal(result.processedCount, 2);
  assert.equal(result.derivedEdgeCount, 0);
});
