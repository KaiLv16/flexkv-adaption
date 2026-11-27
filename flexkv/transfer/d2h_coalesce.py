# SPDX-License-Identifier: Apache-2.0
"""Transport-agnostic worker-side coalescing for compatible D2H transfer ops.

Upper-layer store interfaces (``store_kv``, ``store_kv_with_result``, or
``store_kv_batch_with_result``) only determine whether D2H tasks are created one
by one or in groups. This module operates purely on ``WorkerTransferOp`` batches
already queued at a worker: consecutive, uncompressed, destination-disjoint D2H
operations targeting the same pool are combined into one underlying transfer
call while keeping per-op completion reporting, metrics, and error/quarantine
semantics intact.
"""

import os
from typing import Any, Callable, List, Optional, Set, Tuple

from flexkv.common.transfer import PoolId, TransferType
from flexkv.transfer.worker_op import WorkerTransferOp


def d2h_coalesce_enabled() -> bool:
    return os.environ.get("FLEXKV_D2H_COALESCE", "1").strip() != "0"


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    return value if value > 0 else default


class CoalescedD2HGroup:
    """Lazy execution group shared by 2+ compatible D2H ``WorkerTransferOp``s.

    The first member reaching ``execute()`` runs the concatenated transfer once.
    Subsequent members observe the shared outcome (returning ``True`` on success
    or re-raising the captured normal ``Exception`` on failure). Fatal
    ``UnsafeWorkerTransfer`` exceptions are never swallowed: they propagate
    immediately from the first member to ``TransferWorkerBase.run()``'s
    quarantine handler.
    """

    __slots__ = (
        "pool_id",
        "ops",
        "src_tensors",
        "dst_tensors",
        "_executed",
        "_error",
    )

    def __init__(
        self,
        pool_id: PoolId,
        entries: List[Tuple[WorkerTransferOp, Any, Any]],
    ) -> None:
        self.pool_id = pool_id
        self.ops = [entry[0] for entry in entries]
        self.src_tensors = [entry[1] for entry in entries]
        self.dst_tensors = [entry[2] for entry in entries]
        self._executed = False
        self._error: Optional[Exception] = None

    def execute(
        self,
        run_combined: Callable[[PoolId, Any, Any, List[WorkerTransferOp]], None],
        op: WorkerTransferOp,
    ) -> bool:
        if not self._executed:
            import torch

            src_all = torch.cat(self.src_tensors, dim=0)
            dst_all = torch.cat(self.dst_tensors, dim=0)
            if hasattr(src_all, "pin_memory"):
                src_all = src_all.pin_memory()
                dst_all = dst_all.pin_memory()
            try:
                run_combined(self.pool_id, src_all, dst_all, self.ops)
            except Exception as exc:
                # Note: UnsafeWorkerTransfer subclasses Exception, so if raised
                # here it propagates out immediately and halts the worker loop
                # before subsequent ops are visited. For ordinary Exceptions,
                # record and re-raise so every grouped op fails consistently.
                self._executed = True
                self._error = exc
                raise
            self._executed = True
            return True

        if self._error is not None:
            raise self._error
        return True


def _extract_int_list(tensor_or_seq: Any) -> List[int]:
    if hasattr(tensor_or_seq, "tolist"):
        vals = tensor_or_seq.tolist()
        return vals if isinstance(vals, list) else [int(vals)]
    return [int(v) for v in tensor_or_seq]


def attach_d2h_coalesce_groups(
    batch_ops: List[Any],
    can_coalesce: Callable[[WorkerTransferOp], bool],
    resolve_blocks: Callable[[WorkerTransferOp], Tuple[Any, Any]],
    max_ops: Optional[int] = None,
    max_blocks: Optional[int] = None,
) -> List[CoalescedD2HGroup]:
    """Partition contiguous compatible D2H ops into ``CoalescedD2HGroup``s.

    Ordering invariants:
    - Non-D2H ops (H2D, LAYERWISE) and control dicts act as hard fences and are
      never reordered across.
    - Ops within a coalesced group must target the same ``PoolId`` and have
      strictly disjoint ``dst_block_ids`` (overlapping GPU ``src_block_ids`` are
      safe because D2H only reads GPU memory).
    """
    if len(batch_ops) < 2 or not d2h_coalesce_enabled():
        return []

    eff_max_ops = max_ops if max_ops is not None else _env_positive_int("FLEXKV_D2H_COALESCE_MAX_OPS", 64)
    eff_max_blocks = (
        max_blocks if max_blocks is not None else _env_positive_int("FLEXKV_D2H_COALESCE_MAX_BLOCKS", 8192)
    )
    if eff_max_ops < 2:
        return []

    groups: List[CoalescedD2HGroup] = []
    pending: List[Tuple[WorkerTransferOp, Any, Any]] = []
    pending_pool: Optional[PoolId] = None
    pending_dst: Set[int] = set()
    pending_blocks = 0

    def flush() -> None:
        nonlocal pending, pending_pool, pending_dst, pending_blocks
        if len(pending) >= 2 and pending_pool is not None:
            group = CoalescedD2HGroup(pending_pool, pending)
            for entry in pending:
                entry[0]._d2h_coalesce_group = group
            groups.append(group)
        pending = []
        pending_pool = None
        pending_dst = set()
        pending_blocks = 0

    for item in batch_ops:
        if not isinstance(item, WorkerTransferOp) or item.transfer_type != TransferType.D2H:
            flush()
            continue
        if item.valid_block_num <= 0 or not can_coalesce(item):
            flush()
            continue

        src_ids, dst_ids = resolve_blocks(item)
        dst_list = _extract_int_list(dst_ids)
        num_blocks = len(dst_list)
        if num_blocks == 0 or num_blocks > eff_max_blocks:
            flush()
            continue
        dst_set = set(dst_list)
        if len(dst_set) != num_blocks:
            # Duplicate destination blocks inside one op: keep standalone.
            flush()
            continue

        pool_id = getattr(item, "pool_id", PoolId.FULL_KV)
        if (
            pending
            and (
                pool_id != pending_pool
                or len(pending) >= eff_max_ops
                or pending_blocks + num_blocks > eff_max_blocks
                or not pending_dst.isdisjoint(dst_set)
            )
        ):
            flush()

        pending.append((item, src_ids, dst_ids))
        pending_pool = pool_id
        pending_dst.update(dst_set)
        pending_blocks += num_blocks

    flush()
    return groups
