export function reference(heads, lengths, count = heads.length, groupCapacity = heads.length) {
  if (count > heads.length) return { status: 1 };
  if (heads.slice(0, count).some(h => h !== 0 && h !== 1) || (count && heads[0] !== 1)) return { status: 2 };
  const ids = []; const representatives = []; const compactLengths = []; const offsets = [];
  let total = 0n;
  for (let i = 0; i < count; i++) {
    if (heads[i]) {
      representatives.push(i); compactLengths.push(lengths[i]); offsets.push(Number(total)); total += BigInt(lengths[i]);
    }
    ids.push(representatives.length - 1);
  }
  const groups = representatives.length;
  if (groups > groupCapacity) return { status: 4, requiredGroups: groups };
  if (total > 0xffffffffn) return { status: 3 };
  return { status: 0, groups, requiredGroups: groups, totalLength: Number(total), ids, representatives, lengths: compactLengths, offsets };
}
