# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from flexkv.common.transfer import PoolId, TransferOp, TransferType
from flexkv.transfer.d2h_coalesce import attach_d2h_coalesce_groups
from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp, WorkerTransferOp
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker
from flexkv.transfer.workers.handle import WorkerHandle
from flexkv.transfer.workers.runtime import _append_received_payload


def _make_worker_op(
    op_id: int,
    src_ids,
    dst_ids,
    transfer_type: TransferType = TransferType.D2H,
    pool_id: PoolId = PoolId.FULL_KV,
) -> WorkerTransferOp:
    src_arr = np.asarray(src_ids, dtype=np.int64)
    dst_arr = np.asarray(dst_ids, dtype=np.int64)
    op = TransferOp(
        graph_id=op_id,
        transfer_type=transfer_type,
        src_block_ids=src_arr,
        dst_block_ids=dst_arr,
        pool_id=pool_id,
    )
    op.op_id = op_id
    return WorkerTransferOp(op)


def _resolve_tensors(op: WorkerTransferOp):
    return (
        torch.as_tensor(op.src_block_ids, dtype=torch.int64),
        torch.as_tensor(op.dst_block_ids, dtype=torch.int64),
    )


def test_merges_compatible_disjoint_dst_even_with_overlapping_src():
    op1 = _make_worker_op(1, [1, 2], [10, 11])
    op2 = _make_worker_op(2, [2, 3], [12, 13])
    op3 = _make_worker_op(3, [4], [14])
    groups = attach_d2h_coalesce_groups(
        [op1, op2, op3],
        can_coalesce=lambda _: True,
        resolve_blocks=_resolve_tensors,
    )
    assert len(groups) == 1
    assert op1._d2h_coalesce_group is groups[0]
    assert op2._d2h_coalesce_group is groups[0]
    assert op3._d2h_coalesce_group is groups[0]

    calls = []

    def run_combined(pool_id, src_all, dst_all, ops):
        calls.append((pool_id, src_all.tolist(), dst_all.tolist(), [o.transfer_op_id for o in ops]))

    assert groups[0].execute(run_combined, op1) is True
    assert groups[0].execute(run_combined, op2) is True
    assert groups[0].execute(run_combined, op3) is True
    assert calls == [(PoolId.FULL_KV, [1, 2, 2, 3, 4], [10, 11, 12, 13, 14], [1, 2, 3])]


def test_splits_on_overlapping_dst_and_non_d2h_fences():
    op1 = _make_worker_op(1, [1, 2], [10, 11])
    op2 = _make_worker_op(2, [3, 4], [11, 12])  # overlaps dst 11
    op3 = _make_worker_op(3, [5, 6], [13, 14])
    ctrl = {"type": "control", "cmd": "barrier"}
    op4 = _make_worker_op(4, [7], [15])
    h2d = _make_worker_op(5, [15], [8], transfer_type=TransferType.H2D)
    op5 = _make_worker_op(6, [9], [16])
    op6 = _make_worker_op(7, [10], [17], pool_id=PoolId.SWA)
    op7 = _make_worker_op(8, [11], [18], pool_id=PoolId.SWA)

    groups = attach_d2h_coalesce_groups(
        [op1, op2, op3, ctrl, op4, h2d, op5, op6, op7],
        can_coalesce=lambda _: True,
        resolve_blocks=_resolve_tensors,
    )
    assert len(groups) == 2
    assert getattr(op1, "_d2h_coalesce_group", None) is None
    assert op2._d2h_coalesce_group is groups[0]
    assert op3._d2h_coalesce_group is groups[0]
    assert getattr(op4, "_d2h_coalesce_group", None) is None
    assert getattr(op5, "_d2h_coalesce_group", None) is None
    assert op6._d2h_coalesce_group is groups[1]
    assert op7._d2h_coalesce_group is groups[1]
    assert groups[1].pool_id == PoolId.SWA


def test_error_and_fatal_exception_propagation():
    op1 = _make_worker_op(1, [1], [10])
    op2 = _make_worker_op(2, [2], [11])
    groups = attach_d2h_coalesce_groups(
        [op1, op2], can_coalesce=lambda _: True, resolve_blocks=_resolve_tensors
    )
    assert len(groups) == 1

    def fail_runtime(*_args):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        groups[0].execute(fail_runtime, op1)
    with pytest.raises(RuntimeError, match="boom"):
        groups[0].execute(fail_runtime, op2)

    class _FatalTransferError(Exception):
        pass

    op3 = _make_worker_op(3, [3], [12])
    op4 = _make_worker_op(4, [4], [13])
    q_groups = attach_d2h_coalesce_groups(
        [op3, op4], can_coalesce=lambda _: True, resolve_blocks=_resolve_tensors
    )

    def fail_fatal(*_args):
        raise _FatalTransferError("dma uncertain")

    with pytest.raises(_FatalTransferError, match="dma uncertain"):
        q_groups[0].execute(fail_fatal, op3)


def test_env_disable_and_max_limits_and_edge_cases():
    ops = [_make_worker_op(i + 1, [i], [100 + i]) for i in range(5)]
    with mock.patch.dict(os.environ, {"FLEXKV_D2H_COALESCE": "0"}):
        assert attach_d2h_coalesce_groups(ops, lambda _: True, _resolve_tensors) == []

    groups = attach_d2h_coalesce_groups(
        ops, lambda _: True, _resolve_tensors, max_ops=2, max_blocks=10
    )
    assert len(groups) == 2
    assert [o.transfer_op_id for o in groups[0].ops] == [1, 2]
    assert [o.transfer_op_id for o in groups[1].ops] == [3, 4]
    assert getattr(ops[4], "_d2h_coalesce_group", None) is None

    # max_blocks boundary splits a group even when max_ops allows more
    wide_ops = [_make_worker_op(i + 1, [i * 3, i * 3 + 1, i * 3 + 2], [20 + i * 3, 21 + i * 3, 22 + i * 3]) for i in range(3)]
    block_groups = attach_d2h_coalesce_groups(
        wide_ops, lambda _: True, _resolve_tensors, max_ops=10, max_blocks=6
    )
    assert len(block_groups) == 1
    assert [o.transfer_op_id for o in block_groups[0].ops] == [1, 2]
    assert getattr(wide_ops[2], "_d2h_coalesce_group", None) is None

    # Duplicate dst inside a single op and empty valid_block_num act as fences
    a = _make_worker_op(1, [1], [10])
    dup = _make_worker_op(2, [2, 3], [11, 11])
    empty = _make_worker_op(3, [], [])
    b = _make_worker_op(4, [4], [12])
    c = _make_worker_op(5, [5], [13])
    fenced = attach_d2h_coalesce_groups(
        [a, dup, b, empty, c], lambda _: True, _resolve_tensors
    )
    assert fenced == []


def test_atomic_batch_pipe_framing_and_append_received_payload():
    recv_conn, send_conn = mp.Pipe(duplex=False)
    try:
        ops = [_make_worker_op(i + 1, [i], [200 + i]) for i in range(4)]
        send_conn.send(ops)
        payload = recv_conn.recv()
        assert isinstance(payload, list)
        assert len(payload) == 4
        groups = attach_d2h_coalesce_groups(payload, lambda _: True, _resolve_tensors)
        assert len(groups) == 1
        assert [o.transfer_op_id for o in groups[0].ops] == [1, 2, 3, 4]
    finally:
        recv_conn.close()
        send_conn.close()

    batch = []
    assert _append_received_payload(batch, None, 10) is True
    assert batch == []

    ctrl = {"type": "control", "cmd": "ping"}
    assert _append_received_payload(batch, ctrl, 20) is False
    assert batch == [ctrl]

    single = _make_worker_op(10, [1], [2])
    assert _append_received_payload(batch, single, 30) is False
    assert single._received_ns == 30

    op_a = _make_worker_op(11, [3], [4])
    assert _append_received_payload(batch, [op_a, None], 40) is True
    assert op_a._received_ns == 40
    assert batch == [ctrl, single, op_a]


def test_worker_begin_transfer_batch_and_launch_coalesces_real_worker():
    w = object.__new__(GPUCPUTransferWorker)
    w._use_region_batch = lambda: True
    w._backend = None
    pool = SimpleNamespace(pool_id=PoolId.FULL_KV, chunk_layer_bytes=64, bytes_per_block=128)
    w._pools = {PoolId.FULL_KV: pool}
    w._pool_for = lambda _op: pool
    w._is_per_group = lambda _p: True
    w._compressor = object()
    w.kv_dim = 1
    w._op_buffer_pinned = True
    w.get_transfer_block_ids = _resolve_tensors
    w._transfer_impl = mock.MagicMock()
    w._log_transfer_performance = mock.MagicMock()

    op1 = _make_worker_op(1, [1, 2], [10, 11])
    op2 = _make_worker_op(2, [3], [12])
    h2d = _make_worker_op(3, [10], [1], transfer_type=TransferType.H2D)
    w._begin_transfer_batch([op1, op2, h2d])

    assert w.launch_transfer(op1) is True
    assert w.launch_transfer(op2) is True
    assert w._transfer_impl.call_count == 1
    src_arg, dst_arg, ttype = w._transfer_impl.call_args.args
    assert src_arg.tolist() == [1, 2, 3]
    assert dst_arg.tolist() == [10, 11, 12]
    assert ttype == TransferType.D2H
    assert w._transfer_impl.call_args.kwargs == {"pool": pool}
    assert w._log_transfer_performance.call_count == 2
    assert w._log_transfer_performance.call_args_list[0].args[1] == 128
    assert w._log_transfer_performance.call_args_list[1].args[1] == 64

    # Layerwise op cannot coalesce
    lw_op = object.__new__(WorkerLayerwiseTransferOp)
    lw_op.transfer_type = TransferType.LAYERWISE
    assert w._can_coalesce_d2h_op(lw_op) is False

    with mock.patch.dict(os.environ, {"FLEXKV_D2H_COALESCE": "0"}):
        assert w._batch_drain_timeout_s(op1) == 0.0
    with mock.patch.dict(os.environ, {"FLEXKV_D2H_COALESCE": "1", "FLEXKV_D2H_COALESCE_WINDOW_US": "250"}):
        assert w._batch_drain_timeout_s(op1) == pytest.approx(0.00025)
    assert w._batch_drain_timeout_s(h2d) == 0.0


def test_transfer_engine_dispatch_flushes_multi_op_batch_and_handle_remote_fallback():
    from flexkv.transfer.transfer_engine import TransferEngine

    eng = object.__new__(TransferEngine)
    eng._pending_worker_batches = None
    local_worker = mock.MagicMock(spec=WorkerHandle)
    local_worker._remote_sender = None
    remote_worker = mock.MagicMock(spec=WorkerHandle)
    remote_worker._remote_sender = object()

    ops = [_make_worker_op(i + 1, [i], [10 + i]) for i in range(3)]
    for i, op in enumerate(ops):
        op._op = TransferOp(
            graph_id=1, transfer_type=TransferType.D2H,
            src_block_ids=op.src_block_ids, dst_block_ids=op.dst_block_ids,
        )
        op._op.op_id = i + 1
    raw_ops = [o._op for o in ops]

    eng._note_ops_completed = mock.MagicMock()
    eng._take_unparked_skip_io = mock.MagicMock(return_value=[])
    eng.scheduler = mock.MagicMock()
    eng.scheduler.schedule.side_effect = [([], raw_ops), ([], [])]
    eng.completed_queue = mock.MagicMock()

    def _assign(op):
        if op.op_id <= 2:
            eng._submit_worker_op(local_worker, op)
        else:
            eng._submit_worker_op(remote_worker, op)
        return None

    eng._assign_or_complete_op = _assign
    eng._dispatch_scheduled_ops([])

    local_worker.submit_transfer_batch.assert_called_once_with(raw_ops[:2])
    local_worker.submit_transfer.assert_not_called()
    # Remote worker has _remote_sender != None, so must bypass batching
    remote_worker.submit_transfer.assert_called_once_with(raw_ops[2])
    remote_worker.submit_transfer_batch.assert_not_called()

    # WorkerHandle.submit_transfer_batch sends list when >1 local ops and falls back when remote
    conn = mock.MagicMock()
    wh = WorkerHandle.__new__(WorkerHandle)
    wh.worker_id = 0
    wh.transfer_conn = conn
    wh._remote_sender = None
    wh.submit_transfer_batch(raw_ops[:2])
    conn.send.assert_called_once()
    sent_payload = conn.send.call_args.args[0]
    assert isinstance(sent_payload, list) and len(sent_payload) == 2

    wh._remote_sender = object()
    wh.submit_transfer = mock.MagicMock()
    wh.submit_transfer_batch(raw_ops[:2])
    assert wh.submit_transfer.call_count == 2


def test_kvtask_reset_cache_drains_finishing_task_and_rejects_active():
    from flexkv.kvtask import KVTaskEngine

    eng = object.__new__(KVTaskEngine)
    finishing = SimpleNamespace(
        task_end_op_finished=True,
        _done=False,
        is_completed=lambda: finishing._done,
    )
    eng.tasks = {1: finishing}
    calls = {"n": 0}

    def _update(timeout=0.0):
        calls["n"] += 1
        if calls["n"] >= 2:
            finishing._done = True

    eng._update_tasks = _update
    eng.prefetch_jobs = {}
    eng.prefetch_tasks = {}
    eng.graph_to_task = {}
    eng.uncompleted_ops = {}
    eng.uncompleted_op_results = {}
    eng.uncompleted_graphs = {}
    eng.cache_engine = mock.MagicMock()
    eng.reset_cache()
    assert eng.tasks == {}
    eng.cache_engine.reset.assert_called_once()

    stuck = SimpleNamespace(
        task_end_op_finished=False,
        is_completed=lambda: False,
    )
    eng.tasks = {2: stuck}
    eng._update_tasks = mock.MagicMock()
    with pytest.raises(RuntimeError, match="1 task\\(s\\) active"):
        eng.reset_cache()
