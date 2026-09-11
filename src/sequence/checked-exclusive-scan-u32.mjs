import { checkedShape, createCheckedPlan } from '../internal/checked-scan-plan.mjs';
export const CHECKED_EXCLUSIVE_SCAN_U32_CONTRACT = 'CUDA-Algorithms-checked-exclusive-scan-u32-candidate-v0';
export function checkedScanU32Requirements(options) { return checkedShape(options, false).requirements; }
export function createCheckedExclusiveScanU32Plan(runtime, options) {
  return createCheckedPlan(runtime, options, false, CHECKED_EXCLUSIVE_SCAN_U32_CONTRACT);
}
