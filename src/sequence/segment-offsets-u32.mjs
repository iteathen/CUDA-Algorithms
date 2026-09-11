import { checkedShape, createCheckedPlan } from '../internal/checked-scan-plan.mjs';
export const SEGMENT_OFFSETS_U32_CONTRACT = 'CUDA-Algorithms-segment-offsets-u32-candidate-v0';
export function segmentOffsetsU32Requirements(options) { return checkedShape(options, true).requirements; }
export function createSegmentOffsetsU32Plan(runtime, options) {
  return createCheckedPlan(runtime, options, true, SEGMENT_OFFSETS_U32_CONTRACT);
}
