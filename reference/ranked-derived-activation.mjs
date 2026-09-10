export const RANKED_DERIVED_INVALID_U32 = 0xffff_ffff;

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function u32(value, label) {
  if (!Number.isInteger(value) || value < 0 || value > RANKED_DERIVED_INVALID_U32) {
    throw new RangeError(`${label} must be a u32 integer`);
  }
  return value;
}

/**
 * Exact JavaScript reference for one generic ranked derivation/activation epoch.
 *
 * The consumer owns the meaning of each item index and the deriveTarget function.
 * CUDA-Algorithms owns only bounded emission lanes, strict rank validation,
 * duplicate-idempotent activation, and deterministic next-workset ordering.
 */
export function runRankedDerivedActivationEpoch({
  ranks,
  activeIndices,
  deriveTarget,
  maxEmissionsPerItem,
  outputCapacity = ranks?.length,
  shardSize = activeIndices?.length || 1,
}) {
  if (!Array.isArray(ranks) || ranks.length < 1 || ranks.length >= RANKED_DERIVED_INVALID_U32) {
    throw new RangeError('ranks must be a nonempty u32-addressable array');
  }
  if (!Array.isArray(activeIndices)) throw new TypeError('activeIndices must be an array');
  if (typeof deriveTarget !== 'function') throw new TypeError('deriveTarget must be a function');

  const emissions = positiveSafeInteger(maxEmissionsPerItem, 'maxEmissionsPerItem');
  const capacity = positiveSafeInteger(outputCapacity, 'outputCapacity');
  const shard = positiveSafeInteger(shardSize, 'shardSize');
  if (capacity > ranks.length) throw new RangeError('outputCapacity cannot exceed item universe');

  for (let index = 0; index < ranks.length; index += 1) u32(ranks[index], `ranks[${index}]`);

  const activated = new Set();
  let processed = 0;
  for (let start = 0; start < activeIndices.length; start += shard) {
    const end = Math.min(start + shard, activeIndices.length);
    for (let offset = start; offset < end; offset += 1) {
      const source = u32(activeIndices[offset], `activeIndices[${offset}]`);
      if (source >= ranks.length) throw new RangeError('active source index exceeds item universe');
      const sourceRank = ranks[source];
      processed += 1;

      for (let lane = 0; lane < emissions; lane += 1) {
        const target = u32(deriveTarget(source, lane), `deriveTarget(${source}, ${lane})`);
        if (target === RANKED_DERIVED_INVALID_U32) continue;
        if (target >= ranks.length) throw new RangeError('derived target index exceeds item universe');
        if (ranks[target] >= sourceRank) {
          throw new RangeError(`rank descent violated: source ${source}@${sourceRank} -> target ${target}@${ranks[target]}`);
        }
        activated.add(target);
      }
    }
  }

  const nextIndices = [...activated].sort((left, right) => left - right);
  if (nextIndices.length > capacity) {
    return Object.freeze({
      status: 'capacity-yield',
      processed,
      requiredCount: nextIndices.length,
      nextIndices: Object.freeze([]),
    });
  }

  return Object.freeze({
    status: 'rank-complete',
    processed,
    requiredCount: nextIndices.length,
    nextIndices: Object.freeze(nextIndices),
  });
}
