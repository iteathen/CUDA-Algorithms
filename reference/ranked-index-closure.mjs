function nonNegativeSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${label} must be a non-negative safe integer`);
  return value;
}

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function requireIndex(value, itemCapacity, label) {
  const index = nonNegativeSafeInteger(value, label);
  if (index >= itemCapacity) throw new RangeError(`${label} ${index} is outside itemCapacity ${itemCapacity}`);
  return index;
}

function requireRank(value, maxRank, label) {
  const rank = nonNegativeSafeInteger(value, label);
  if (rank > maxRank) throw new RangeError(`${label} ${rank} exceeds maxRank ${maxRank}`);
  return rank;
}

/**
 * Exact reference semantics for bounded ranked closure over a consumer-owned
 * index universe. CUDA-Algorithms owns progression; callbacks retain consumer
 * meaning for rank and dependency derivation.
 */
export function runRankedIndexClosure({
  itemCapacity,
  maxRank,
  seeds,
  rankOf,
  derive,
  maxEmissionsPerItem,
  shardSize = itemCapacity,
}) {
  const capacity = positiveSafeInteger(itemCapacity, 'itemCapacity');
  const maximumRank = nonNegativeSafeInteger(maxRank, 'maxRank');
  const emissionBound = nonNegativeSafeInteger(maxEmissionsPerItem, 'maxEmissionsPerItem');
  const physicalShardSize = positiveSafeInteger(shardSize, 'shardSize');
  if (!Array.isArray(seeds)) throw new TypeError('seeds must be an array of item indices');
  if (typeof rankOf !== 'function') throw new TypeError('rankOf must be a function');
  if (typeof derive !== 'function') throw new TypeError('derive must be a function');

  const active = new Uint8Array(capacity);
  const queuedByRank = Array.from({ length: maximumRank + 1 }, () => []);
  const rankCache = new Array(capacity);
  let activatedCount = 0;
  let duplicateActivations = 0;
  let derivedEdgeCount = 0;
  let processedCount = 0;
  let shardCount = 0;

  const rankFor = (index) => {
    if (rankCache[index] !== undefined) return rankCache[index];
    const rank = requireRank(rankOf(index), maximumRank, `rankOf(${index})`);
    rankCache[index] = rank;
    return rank;
  };

  const activate = (index, sourceRank = null) => {
    const item = requireIndex(index, capacity, 'derived item');
    const rank = rankFor(item);
    if (sourceRank !== null && rank >= sourceRank) {
      throw new RangeError(`rank monotonicity violation: ${item} has rank ${rank}, source rank is ${sourceRank}`);
    }
    if (active[item] !== 0) {
      duplicateActivations += 1;
      return false;
    }
    active[item] = 1;
    queuedByRank[rank].push(item);
    activatedCount += 1;
    return true;
  };

  for (let i = 0; i < seeds.length; i += 1) {
    activate(requireIndex(seeds[i], capacity, `seeds[${i}]`));
  }

  for (let rank = maximumRank; rank >= 0; rank -= 1) {
    const items = queuedByRank[rank];
    for (let start = 0; start < items.length; start += physicalShardSize) {
      const end = Math.min(items.length, start + physicalShardSize);
      shardCount += 1;
      for (let position = start; position < end; position += 1) {
        const item = items[position];
        const outputs = derive(item);
        if (!Array.isArray(outputs)) throw new TypeError(`derive(${item}) must return an array`);
        if (outputs.length > emissionBound) {
          throw new RangeError(`derive(${item}) emitted ${outputs.length}, exceeding maxEmissionsPerItem ${emissionBound}`);
        }
        processedCount += 1;
        derivedEdgeCount += outputs.length;
        for (let lane = 0; lane < outputs.length; lane += 1) {
          activate(requireIndex(outputs[lane], capacity, `derive(${item})[${lane}]`), rank);
        }
      }
    }
  }

  const activeIndices = [];
  const activeByRank = Array.from({ length: maximumRank + 1 }, () => []);
  for (let index = 0; index < capacity; index += 1) {
    if (active[index] === 0) continue;
    activeIndices.push(index);
    activeByRank[rankFor(index)].push(index);
  }

  return Object.freeze({
    itemCapacity: capacity,
    maxRank: maximumRank,
    maxEmissionsPerItem: emissionBound,
    shardSize: physicalShardSize,
    activeIndices: Object.freeze(activeIndices),
    activeByRank: Object.freeze(activeByRank.map((items) => Object.freeze(items))),
    activatedCount,
    processedCount,
    derivedEdgeCount,
    duplicateActivations,
    shardCount,
  });
}
