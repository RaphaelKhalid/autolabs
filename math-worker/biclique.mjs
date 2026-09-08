export function extractExactBicliques(candidates, differences, options = {}) {
  const consumeCheck = options.consumeCheck ?? (() => true);
  const stopReason = options.stopReason ?? (() => null);
  const resultLimit = options.resultLimit ?? 20;

  function subsetGroups(rowCount, numberCount, target) {
    const groups = new Map();
    const found = [];
    let complete = true;
    let truncatedBy = null;

    function visitSubsets(indices, start, picked, candidate) {
      if (picked.length === rowCount) {
        if (!consumeCheck()) return false;
        const key = picked.join(',');
        let group = groups.get(key);
        if (!group) {
          group = { indices: [...picked], candidates: [], reported: false };
          groups.set(key, group);
        }
        if (group.candidates.length < numberCount) group.candidates.push(candidate);
        if (group.candidates.length === numberCount && !group.reported) {
          group.reported = true;
          const selectedDifferences = group.indices.map((index) => differences[index].toString());
          const witnesses = group.candidates.flatMap((item) =>
            (item.witnesses ?? []).filter((witness) => selectedDifferences.includes(witness.difference)));
          found.push({
            target,
            numbers: group.candidates.map((item) => item.number),
            differences: selectedDifferences,
            exactCells: numberCount * rowCount,
            witnesses,
          });
        }
        return true;
      }

      const remaining = rowCount - picked.length;
      for (let index = start; index <= indices.length - remaining; index += 1) {
        picked.push(indices[index]);
        if (!visitSubsets(indices, index + 1, picked, candidate)) return false;
        picked.pop();
      }
      return true;
    }

    for (const candidate of candidates) {
      if (found.length >= resultLimit) {
        complete = false;
        truncatedBy = 'result_limit';
        break;
      }
      const indices = candidate.supportMask.flatMap((supported, index) => supported ? [index] : []);
      if (indices.length < rowCount) continue;
      if (!visitSubsets(indices, 0, [], candidate)) {
        complete = false;
        truncatedBy = stopReason();
        break;
      }
    }

    return { target, rowCount, numberCount, matches: found, complete, truncatedBy };
  }

  const searches = [
    subsetGroups(5, 5, '5x5'),
    subsetGroups(4, 6, '6x4'),
    subsetGroups(5, 4, '4x5'),
  ];
  return {
    searches,
    matches: searches.flatMap((search) => search.matches),
    complete: searches.every((search) => search.complete),
    truncatedBy: searches.find((search) => !search.complete)?.truncatedBy ?? null,
  };
}
