"""CPU protocol tests using the real Linux eventfd counter (no GPU DMA)."""

import os
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flexkv.integration.sglang.comm import FlexKVLayerDoneCounter
from flexkv.integration.sglang.connector import FlexKVConnector

pytestmark = pytest.mark.unit


@pytest.fixture
def connector():
    c = FlexKVConnector.__new__(FlexKVConnector)
    c.enable_layerwise = True
    c.page_size = 1
    c.layer_done_counter = FlexKVLayerDoneCounter(3)
    c.layer_done_counter.wait_timeout_s = 0.1
    c._sync_ctx = SimpleNamespace(
        is_sync_leader=True, needs_sync=False, world_rank=0,
        should_send_slot_mapping_to_remote=False,
    )
    c.kv_manager = MagicMock()
    c.kv_manager.launch.return_value = [900]
    c._swa_kv_pool = None
    c._pending_lookups = {}
    c._pending_lookup_contexts = {}
    c._prepared_layerwise_loads = {}
    c._layerwise_batch_launch_error = None
    c._launched_load_tids = []
    c._launched_load_contexts = {}
    c._prefetch_contexts = {}
    c._ongoing_prefetches = {}
    c._inflight_loads = {}
    c._completed_layerwise = []
    c._inflight_stores = {}
    c._log_cache_op = MagicMock()
    yield c
    for event in c.layer_done_counter.events:
        event.close()


def prepare(c, rid, tid, size=4, defer=True):
    c._pending_lookups[rid] = tid
    return c.start_load_kv_layerwise(
        rid, torch.arange(tid * 10, tid * 10 + size), defer_launch=defer
    )


def consume(c, producer):
    event = c.layer_done_counter.events[producer]
    for layer, fd in enumerate(event.load_event_fds):
        os.eventfd_write(fd, 1)
        c.layer_done_counter.wait_until(layer)
    assert c.layer_done_counter.consumer_index == -1


@pytest.mark.parametrize("local_ready", [False, True])
@pytest.mark.parametrize("needs_sync", [False, True])
def test_allocation_readiness_votes_before_restore(connector, local_ready, needs_sync):
    connector._sync_ctx.needs_sync = needs_sync
    connector._sync_ctx.all_reduce_min = MagicMock(return_value=0)
    assert connector.sync_load_ready(local_ready) is (False if needs_sync else local_ready)
    if needs_sync:
        connector._sync_ctx.all_reduce_min.assert_called_once_with(int(local_ready))
    else:
        connector._sync_ctx.all_reduce_min.assert_not_called()


def test_admitted_batch_launches_once_and_consumes_one_signal_per_layer(connector):
    c = connector
    for round_id in range(5):  # rotate through the counter ring more than once
        tids = [round_id * 10 + i for i in range(1, 4)]
        rids = [f"r{tid}" for tid in tids]
        pids = [prepare(c, rid, tid, size=i + 2)[1]
                for i, (rid, tid) in enumerate(zip(rids, tids))]
        assert len(set(pids)) == 1
        assert c.kv_manager.launch.call_count == round_id
        assert c.layer_done_counter.events[pids[0]].wait_remaining == [3] * 3
        c.flush_layerwise_loads(["gpu-hit", *rids, "uncached"])
        call = c.kv_manager.launch.call_args.kwargs
        assert call["task_ids"] == tids
        assert [m.numel() for m in call["slot_mappings"]] == [2, 3, 4]
        assert call["as_batch"] and call["layerwise_transfer"]
        assert call["counter_id"] == pids[0]
        assert c.layer_done_counter.events[pids[0]].wait_remaining == [1] * 3
        assert not c._prepared_layerwise_loads
        consume(c, pids[0])
    assert c.kv_manager.launch.call_count == 5


def test_empty_batch_has_no_rpc(connector):
    connector.flush_layerwise_loads(["gpu-hit"])
    connector.kv_manager.launch.assert_not_called()


def test_legacy_immediate_api_keeps_per_request_signals(connector):
    c = connector
    _, producer = prepare(c, "a", 1, defer=False)
    prepare(c, "b", 2, defer=False)
    assert c.kv_manager.launch.call_count == 2
    assert c.layer_done_counter.events[producer].wait_remaining == [2] * 3
    c.flush_layerwise_loads(["a", "b"])
    for layer, fd in enumerate(c.layer_done_counter.events[producer].load_event_fds):
        os.eventfd_write(fd, 2)
        c.layer_done_counter.wait_until(layer)


def test_cancel_prepared_member_retains_other_destinations(connector):
    c = connector
    _, producer = prepare(c, "a", 1)
    prepare(c, "b", 2)
    c.release_pending("a")
    c.kv_manager.cancel.assert_called_once_with([1])
    assert c.layer_done_counter.events[producer].wait_remaining == [1] * 3
    c.flush_layerwise_loads(["b"])
    assert c.kv_manager.launch.call_args.kwargs["task_ids"] == [2]
    consume(c, producer)


def test_reset_cancels_unsubmitted_members_and_reuses_counter(connector):
    c = connector
    prepare(c, "a", 1)
    prepare(c, "b", 2)
    c.reset()
    assert c.kv_manager.cancel.call_count == 2
    c.kv_manager.launch.assert_not_called()
    assert not c._prepared_layerwise_loads
    _, producer = prepare(c, "c", 3)
    c.flush_layerwise_loads(["c"])
    consume(c, producer)


@pytest.mark.parametrize("rids", [["a"], ["a", "a", "b"], ["b", "a"]])
def test_invalid_admission_never_submits_dma(connector, rids):
    c = connector
    prepare(c, "a", 1)
    prepare(c, "b", 2)
    with pytest.raises(RuntimeError):
        c.flush_layerwise_loads(rids)
    c.kv_manager.launch.assert_not_called()
    assert len(c._prepared_layerwise_loads) == 2
    c.reset()


def test_other_rank_rejection_retains_prepared_batch(connector):
    c = connector
    prepare(c, "a", 1)
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter = MagicMock(side_effect=lambda payload, **kw: payload)
    c._sync_ctx.all_reduce_min = MagicMock(return_value=0)
    with pytest.raises(RuntimeError, match="Another rank"):
        c.flush_layerwise_loads(["a"])
    c.kv_manager.launch.assert_not_called()
    assert len(c._prepared_layerwise_loads) == 1


def test_launch_transport_failure_cannot_free_potential_dma_targets(connector):
    c = connector
    prepare(c, "a", 1)
    c.kv_manager.launch.side_effect = OSError("reply lost")
    with pytest.raises(RuntimeError, match="reply lost"):
        c.flush_layerwise_loads(["a"])
    for action in [lambda: c.release_pending("a"), c.reset,
                   lambda: c.flush_layerwise_loads(["a"])]:
        with pytest.raises(RuntimeError):
            action()
    assert "a" in c._prepared_layerwise_loads
    c.kv_manager.cancel.assert_not_called()


@pytest.mark.parametrize("mismatch", [None, "batch", "slots"])
def test_four_rank_protocol_agrees_before_single_leader_launch(mismatch):
    # Synchronous collectives are simulated; eventfd state is real on each rank.
    fixtures = [connector.__wrapped__() for _ in range(4)]
    ranks = [next(f) for f in fixtures]
    barrier = threading.Barrier(4, timeout=5)
    payload = [None]
    votes = [None] * 4
    def context(rank):
        def scatter(value, **kwargs):
            if rank == 0:
                payload[0] = value
            barrier.wait()
            result = copy.deepcopy(payload[0])
            barrier.wait()
            return result
        def reduce_min(value):
            votes[rank] = value
            barrier.wait()
            result = min(votes)
            barrier.wait()
            return result
        return SimpleNamespace(
            is_sync_leader=rank == 0, needs_sync=True, world_rank=rank,
            should_send_slot_mapping_to_remote=False,
            scatter=scatter, all_reduce_min=reduce_min,
        )
    for i, c in enumerate(ranks):
        c._sync_ctx = context(i)
        c._cache_layer_split = True
    def run(rank):
        c = ranks[rank]
        if mismatch == "slots":
            c._pending_lookups["a"] = 1
            start = 100 if rank == 2 else 0
            assert c.start_load_kv_layerwise(
                "a", torch.arange(start, start + 4), defer_launch=True
            ) == (0, -1)
            assert not c._prepared_layerwise_loads
            return
        for rid, tid in [("a", 1), ("b", 2), ("c", 3)]:
            _, producer = prepare(c, rid, tid)
        rids = ["a", "b"] if mismatch and rank == 2 else ["a", "b", "c"]
        if mismatch:
            with pytest.raises(RuntimeError):
                c.flush_layerwise_loads(rids)
        else:
            c.flush_layerwise_loads(rids)
            consume(c, producer)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, range(4)))
        assert ranks[0].kv_manager.launch.call_count == (0 if mismatch else 1)
        if mismatch == "slots":
            ranks[0].kv_manager.cancel.assert_called_once_with([1])
        for c in ranks[1:]:
            c.kv_manager.launch.assert_not_called()
    finally:
        for f in fixtures:
            next(f, None)


def test_cp_sync_restore_with_layerwise_worker(connector):
    c = connector
    c.model_config = SimpleNamespace(num_layers=3)
    c._sync_ctx.all_reduce_min = MagicMock(side_effect=lambda value: value)
    c._cache_layer_split = True
    c._invalidate_layer_split_cache = MagicMock()
    c._pending_lookups["r"] = 71

    def launch(**kwargs):
        assert kwargs["layerwise_transfer"] and kwargs["task_ids"] == [71]
        event = c.layer_done_counter.events[kwargs["counter_id"]]
        for fd in event.load_event_fds:
            os.eventfd_write(fd, 1)
        return [900]

    c.kv_manager.launch.side_effect = launch
    assert c.retrieve_kv("r", torch.arange(4)) == 4
    c._invalidate_layer_split_cache.assert_called_once()
    assert c.layer_done_counter.consumer_index == -1
    assert c._launched_load_tids == [900]


def test_cp_immediate_launch_failure_retains_counter(connector):
    c = connector
    c._cache_layer_split = True
    c._invalidate_layer_split_cache = MagicMock()
    c._sync_ctx.all_reduce_min = MagicMock(side_effect=lambda value: value)
    c.kv_manager.launch.side_effect = OSError("reply lost")
    with pytest.raises(OSError, match="reply lost"):
        prepare(c, "a", 1, defer=False)
    c._sync_ctx.all_reduce_min.assert_called_with(0)
    assert c.layer_done_counter.consumer_index >= 0
    with pytest.raises(RuntimeError):
        c.reset()


def test_cp_whole_restore_retains_task_until_terminal(connector):
    c = connector
    c.enable_layerwise = False
    c._cache_layer_split = True
    c._invalidate_layer_split_cache = MagicMock()
    c._sync_ctx.all_reduce_min = MagicMock(side_effect=lambda value: value)
    c._sync_ctx.cp_agrees = MagicMock(return_value=True)
    c._pending_lookups["r"] = 71
    c.kv_manager.wait.return_value = {}
    with pytest.raises(RuntimeError, match="uncertain"):
        c.retrieve_kv("r", torch.arange(4))
    assert c.kv_manager.launch.call_args.kwargs["task_ids"] == [71]
    assert not c.kv_manager.launch.call_args.kwargs["as_batch"]
    c.kv_manager.wait.assert_called_with([71], timeout=30.0, completely=True)
    assert c._launched_load_tids == [71]
    c._sync_ctx.all_reduce_min.assert_called_with(-1)

def test_cp_whole_restore_rejects_slots_before_launch(connector):
    connector.enable_layerwise = False
    connector._cache_layer_split = True
    connector._sync_ctx.cp_agrees = MagicMock(return_value=False)
    connector._sync_ctx.all_reduce_min = MagicMock(side_effect=lambda value: value)
    connector._pending_lookups["r"] = 71
    with pytest.raises(RuntimeError, match="disagree on H2D slots"):
        connector.retrieve_kv("r", torch.arange(4))
    connector.kv_manager.cancel.assert_called_once_with([71])
    connector.kv_manager.launch.assert_not_called()


def test_cp_sync_restore_leaves_existing_prepared_batch_intact(connector):
    c = connector
    c._cache_layer_split = True
    c._invalidate_layer_split_cache = MagicMock()
    c._sync_ctx.all_reduce_min = MagicMock(side_effect=lambda value: value)
    prepare(c, "first", 1)
    c._invalidate_layer_split_cache.assert_not_called()
    c._pending_lookups["second"] = 2
    with pytest.raises(RuntimeError, match="active layerwise batch"):
        c.retrieve_kv("second", torch.arange(4))
    assert list(c._prepared_layerwise_loads) == ["first"]
    assert c._pending_lookups["second"] == 2
    c.kv_manager.launch.assert_not_called()



def test_restore_hook_alone_does_not_enable_cache_layer_split(connector):
    connector.enable_layerwise = False
    connector._invalidate_layer_split_cache = MagicMock()
    connector._pending_lookups["r"] = 71
    connector.kv_manager.wait.return_value = {71: SimpleNamespace(status="success")}

    assert connector.retrieve_kv("r", torch.arange(4)) == 4
    connector._invalidate_layer_split_cache.assert_called_once()
    assert connector.kv_manager.launch.call_args.kwargs["as_batch"]
    assert not connector.kv_manager.wait.call_args.kwargs["completely"]
    assert connector._launched_load_tids == []
