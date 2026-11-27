from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from flexkv.common.request import KVResponse, KVResponseStatus, PutMatchDisposition
from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector, StoreSubmissionUncertainError


class _CoordinatedConnector(FlexKVConnector):
    _force_coordinated: bool = False

    @property
    def _coordinated_restore(self) -> bool:
        return self._force_coordinated


def _follower_connector(payload):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector.kv_manager = None
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        should_send_slot_mapping_to_remote=False,
        scatter=MagicMock(return_value=payload),
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    return connector


def test_async_store_slot_mapping_support_is_topology_driven():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._sync_ctx = SimpleNamespace(is_pp_active=False)
    connector._swa_kv_pool = None

    assert connector.supports_async_store_slot_mapping is True

    connector._sync_ctx.is_pp_active = True
    assert connector.supports_async_store_slot_mapping is False

    connector._sync_ctx.is_pp_active = False
    connector._swa_kv_pool = object()
    assert connector.supports_async_store_slot_mapping is False


def test_store_start_tracks_task_on_nonleader_rank():
    payload = {
        "rid": "request",
        "task_id": 17,
        "active": True,
        "slot_count": 4,
        "unmatched_count": 2,
        "unmatched_mask": [],
        "error": "",
    }
    connector = _follower_connector(payload)

    task_id = connector.store_kv("request", range(1, 5), torch.arange(4, dtype=torch.int64))

    assert task_id == 17
    assert connector._inflight_stores == {"request": 17}
    connector._sync_ctx.scatter.assert_called_once_with(
        {
            "rid": "request",
            "task_id": -1,
            "active": False,
            "already_cached": False,
            "reads_gpu": False,
            "slot_count": 4,
            "unmatched_count": 0,
            "unmatched_mask": [],
            "error": "",
        },
        channel=FlexKVScatterChannel.STORE_START,
    )


def test_store_leader_accepts_pinned_sideband_cpu_mapping():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._swa_kv_pool = None
    connector._profile_store_stages = False
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = (
        17,
        np.asarray([True, False, True, False], dtype=np.bool_),
        PutMatchDisposition.READS_GPU,
    )
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
        should_send_slot_mapping_to_remote=False,
        is_cross_node_pp=False,
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    cpu_mapping = torch.tensor([4, 5, 8, 9], dtype=torch.int64)

    assert connector.store_kv("request", np.arange(1, 5, dtype=np.int64), cpu_mapping) == 17
    launch = connector.kv_manager.launch.call_args.kwargs
    assert launch["slot_mappings"][0].tolist() == [4, 8]
    assert launch["swa_slot_mappings"] == [None]


def test_swa_only_gpu_store_binds_the_tail_even_with_an_empty_full_mask():
    connector = _follower_connector(None)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.scatter.side_effect = lambda value, **_: value
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = (
        17, np.zeros(4, dtype=np.bool_), PutMatchDisposition.READS_GPU
    )
    connector._build_swa_slot_mapping = MagicMock(return_value=torch.arange(1, 5))
    gpu_slots = torch.arange(4)

    result = connector.store_kv_with_result("request", range(4), gpu_slots)
    assert result.task_id == 17 and result.reads_gpu
    assert connector._build_swa_slot_mapping.call_args.args[0] is gpu_slots
    launch = connector.kv_manager.launch.call_args.kwargs
    assert launch["slot_mappings"][0].numel() == 0
    assert launch["swa_slot_mappings"][0].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("disposition, task_id", [
    (PutMatchDisposition.ALREADY_CACHED, -1),
    (PutMatchDisposition.SKIPPED, -1),
    (PutMatchDisposition.HOST_ONLY, 17),
])
def test_zero_d2h_store_result_is_consistent_across_ranks(disposition, task_id):
    leader = _follower_connector(None)
    leader._sync_ctx.is_sync_leader = True
    leader._sync_ctx.scatter.side_effect = lambda value, **_: value
    leader.kv_manager = MagicMock()
    leader.kv_manager.put_match.return_value = (
        task_id, np.zeros(4, dtype=np.bool_), disposition
    )
    gpu_slots = torch.arange(4)

    result = leader.store_kv_with_result("request", range(4), gpu_slots)
    payload = leader._sync_ctx.scatter.call_args.args[0]
    follower = _follower_connector(payload)
    follower._sync_ctx.should_send_slot_mapping_to_remote = True
    follower._send_slot_mapping_to_remote = MagicMock()
    assert follower.store_kv_with_result("request", range(4), gpu_slots) == result

    assert result.task_id == task_id
    assert result.already_cached == (disposition == PutMatchDisposition.ALREADY_CACHED)
    assert not result.reads_gpu
    if task_id >= 0:
        assert leader.kv_manager.launch.call_args.kwargs["slot_mappings"][0].size == 0
        assert leader._inflight_stores == follower._inflight_stores == {"request": task_id}
        assert follower._send_slot_mapping_to_remote.call_args.args[1].numel() == 0
    else:
        leader.kv_manager.launch.assert_not_called()
        follower._send_slot_mapping_to_remote.assert_not_called()
        assert not leader._inflight_stores and not follower._inflight_stores


def test_lookup_accepts_sglang_array_token_ids():
    connector = _follower_connector({"task_id": -1, "hit": 0})
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    token_ids = array("q", [1, 2, 3, 4])

    assert connector.lookup_kv(
        token_ids,
        torch.ones(len(token_ids), dtype=torch.bool),
        rid="request",
    ) == (-1, 0)
    connector._log_cache_op.assert_called_once()


def test_prefetch_start_accepts_legacy_manager_result_with_planned_tokens():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.prefetch_async.return_value = (23, 256)
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {}
    connector._prefetch_contexts = {}
    connector._prefetch_planned_tokens = {}
    connector._prefetch_loaded_tokens = {}
    context = SimpleNamespace(task_id=-1)
    connector._new_op_context = MagicMock(return_value=context)
    connector._log_cache_op = MagicMock()

    task_id = connector.prefetch_async("request", [1, 2, 3])

    assert task_id == 23
    assert context.task_id == 23
    assert connector._ongoing_prefetches == {"request": 23}
    assert connector._prefetch_contexts == {"request": context}
    assert connector._prefetch_planned_tokens == {"request": 256}


def test_prefetch_completion_reports_loaded_tokens_once():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector.kv_manager = MagicMock()
    connector.kv_manager.try_wait.return_value = {
        23: SimpleNamespace(
            status=KVResponseStatus.SUCCESS,
            return_mask=np.ones(256, dtype=np.bool_),
        )
    }
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {"request": 23}
    connector._prefetch_contexts = {"request": SimpleNamespace(task_id=23)}
    connector._prefetch_planned_tokens = {"request": 0}
    connector._prefetch_loaded_tokens = {}
    connector._new_op_context = MagicMock()
    connector._pop_context = MagicMock(return_value=SimpleNamespace(task_id=23))
    connector._log_cache_op = MagicMock()

    assert connector.check_prefetch_progress("request") is True
    assert connector.pop_prefetch_loaded_tokens("request") == 256
    assert connector.pop_prefetch_loaded_tokens("request") == 0


def test_prefetch_completion_reports_partial_remote_prefix():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector.kv_manager = MagicMock()
    connector.kv_manager.try_wait.return_value = {
        23: SimpleNamespace(
            status=KVResponseStatus.SUCCESS,
            return_mask=np.asarray([True] * 192 + [False] * 64),
        )
    }
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {"request": 23}
    connector._prefetch_contexts = {"request": SimpleNamespace(task_id=23)}
    connector._prefetch_planned_tokens = {"request": 0}
    connector._prefetch_loaded_tokens = {}
    connector._new_op_context = MagicMock()
    connector._pop_context = MagicMock(return_value=SimpleNamespace(task_id=23))
    connector._log_cache_op = MagicMock()

    assert connector.check_prefetch_progress("request") is True
    assert connector.pop_prefetch_loaded_tokens("request") == 192


def test_duplicate_lookup_cancels_replaced_held_task():
    """A repeated SGLang match for one rid must not orphan the first lock."""
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.get_match.side_effect = [
        (41, np.ones(4, dtype=np.bool_)),
        (42, np.ones(4, dtype=np.bool_)),
    ]
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    contexts = [
        SimpleNamespace(task_id=-1, operation="lookup"),
        SimpleNamespace(task_id=-1, operation="lookup"),
    ]
    connector._new_op_context = MagicMock(side_effect=contexts)
    connector._log_cache_op = MagicMock()

    mask = torch.ones(4, dtype=torch.bool)
    assert connector.lookup_kv([1, 2, 3, 4], mask, rid="same") == (41, 4)
    assert connector.lookup_kv([1, 2, 3, 4], mask, rid="same") == (42, 4)

    connector.kv_manager.cancel.assert_called_once_with([41])
    assert connector._pending_lookups == {"same": 42}
    assert connector._pending_lookup_contexts == {"same": contexts[1]}
    assert contexts[0].operation == "load"
    assert any(
        call.args[1:3] == ("complete", "cancelled") and call.kwargs.get("reason") == "lookup_replaced"
        for call in connector._log_cache_op.call_args_list
    )

    connector.release_pending("same")
    assert connector.kv_manager.cancel.call_args_list == [
        (([41],), {}),
        (([42],), {}),
    ]
    assert connector._pending_lookups == {}
    assert connector._pending_lookup_contexts == {}


def test_store_completion_clears_nonleader_tracking():
    connector = _follower_connector(["request"])
    connector._sync_ctx.is_sync_leader = False
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": object()}

    assert connector.check_completed_stores() == ["request"]
    assert connector._inflight_stores == {}
    assert connector._inflight_store_contexts == {}
    connector._sync_ctx.scatter.assert_called_once_with([], channel=FlexKVScatterChannel.STORE_COMPLETION)


def test_store_ready_uses_leader_payload_and_dedicated_channel():
    connector = _follower_connector(["request"])

    assert connector.sync_ready_store_rids(["ignored-on-follower"]) == ["request"]
    connector._sync_ctx.scatter.assert_called_once_with([], channel=FlexKVScatterChannel.STORE_READY)


def test_store_completion_waits_for_fully_drained_graph():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {
        "request": SimpleNamespace(task_id=17)
    }
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=17)
    )
    connector._log_cache_op = MagicMock()
    connector.kv_manager.wait.side_effect = [
        {17: KVResponse(KVResponseStatus.TIMEOUT, 17, None)},
        {17: KVResponse(KVResponseStatus.SUCCESS, 17, None)},
    ]

    assert connector.check_completed_stores() == []
    assert connector._inflight_stores == {"request": 17}

    assert connector.check_completed_stores() == ["request"]
    assert connector._inflight_stores == {}
    assert connector.kv_manager.wait.call_args_list == [
        (([17],), {"timeout": 0.0, "completely": True}),
        (([17],), {"timeout": 0.0, "completely": True}),
    ]


def test_wait_store_keeps_tracking_until_fully_drained():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {
        "request": SimpleNamespace(task_id=17)
    }
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=17)
    )
    connector._log_cache_op = MagicMock()
    connector.kv_manager.wait.side_effect = [
        {17: KVResponse(KVResponseStatus.TIMEOUT, 17, None)},
        {17: KVResponse(KVResponseStatus.SUCCESS, 17, None)},
    ]

    assert not connector.wait_store("request", timeout=5.0)
    assert connector._inflight_stores == {"request": 17}

    assert connector.wait_store("request", timeout=5.0)
    assert connector._inflight_stores == {}
    assert connector.kv_manager.wait.call_args_list == [
        (([17],), {"timeout": 5.0, "completely": True}),
        (([17],), {"timeout": 5.0, "completely": True}),
    ]


def test_store_reset_waits_for_leader_drain_on_follower():
    connector = _follower_connector(None)
    connector._sync_ctx.scatter.side_effect = [
        {"ok": True, "error": ""},
        {"ok": True, "error": ""},
    ]
    connector._launched_load_tids = []
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._prefetch_contexts = {}
    connector._ongoing_prefetches = {}
    connector._inflight_loads = {}
    connector._completed_layerwise = {}
    connector._launched_load_contexts = {}
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": object()}
    connector.layer_done_counter = None

    connector.reset()

    assert connector._sync_ctx.scatter.call_args_list[1].kwargs == {
        "channel": FlexKVScatterChannel.STORE_RESET,
        "blocking": True,
    }
    assert connector._inflight_stores == {}


def test_namespace_capability_is_explicit_for_all_active_paths():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    assert connector.supports_cache_namespace is True
    # A chunked-prefetch implementation needs its own full-chain opt-in.
    connector._chunked_prefetch = True
    connector._chunked_namespace_supported = False
    assert connector.supports_cache_namespace is False
    connector._chunked_namespace_supported = True
    assert connector.supports_cache_namespace is True


def test_namespace_is_forwarded_to_lookup_store_and_prefetch():
    connector = _follower_connector(None)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.needs_sync = False
    connector._sync_ctx.is_pp_active = False
    connector._swa_kv_pool = None
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._prefetch_enabled = True
    connector._ongoing_prefetches = {}
    connector._prefetch_contexts = {}
    connector._prefetch_planned_tokens = {}
    connector.kv_manager = MagicMock()
    connector.kv_manager.get_match.return_value = (-1, np.zeros(4, dtype=bool))
    connector.kv_manager.put_match.return_value = (
        -1, np.zeros(4, dtype=bool), PutMatchDisposition.SKIPPED
    )
    connector.kv_manager.prefetch_async.return_value = (23, 4)
    namespace = ['["sglang-cache-v1","adapter","salt"]']
    connector.lookup_kv([1, 2, 3, 4], torch.ones(4, dtype=torch.bool), rid="r", namespace=namespace)
    connector.store_kv("r", [1, 2, 3, 4], torch.arange(4), namespace=namespace)
    assert connector.prefetch_async("r", [1, 2, 3, 4], namespace=namespace) == 23
    for method in (connector.kv_manager.get_match, connector.kv_manager.put_match, connector.kv_manager.prefetch_async):
        assert method.call_args.kwargs["namespace"] == namespace
        assert method.call_args.kwargs["token_ids"].tolist() == [1, 2, 3, 4]


def test_coordinated_store_marks_uncertainty_after_launch_attempt():
    connector = FlexKVConnector.__new__(_CoordinatedConnector)
    connector._force_coordinated = True
    connector.page_size = 1
    connector._swa_kv_pool = None
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=True,
        should_send_slot_mapping_to_remote=False,
        is_cross_node_pp=False,
        scatter=MagicMock(side_effect=lambda value, **_: value),
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = (
        17, np.ones(4, dtype=np.bool_), PutMatchDisposition.READS_GPU
    )
    connector.kv_manager.launch.side_effect = OSError()

    with pytest.raises(StoreSubmissionUncertainError, match="OSError"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert connector._inflight_stores == {"request": 17}


def test_coordinated_store_notification_failure_without_launch_has_no_task():
    connector = FlexKVConnector.__new__(_CoordinatedConnector)
    connector._force_coordinated = True
    connector.page_size = 1
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=True,
        should_send_slot_mapping_to_remote=False,
        scatter=MagicMock(side_effect=OSError("start notification failed")),
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = None

    with pytest.raises(OSError, match="start notification failed"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert not connector._inflight_stores
    connector.kv_manager.launch.assert_not_called()


def test_coordinated_retrieve_propagates_peer_failure():
    connector = FlexKVConnector.__new__(_CoordinatedConnector)
    connector._force_coordinated = True
    connector.enable_layerwise = False
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        should_send_slot_mapping_to_remote=False,
        all_reduce_min=MagicMock(return_value=-1),
    )
    connector._pop_context = MagicMock(return_value=SimpleNamespace(task_id=-1))

    with pytest.raises(RuntimeError, match="KV restore failed on a peer"):
        connector.retrieve_kv("r", torch.arange(4))


def test_validate_kv_pool_support_rejects_unsupported_layouts_and_k_only():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._is_dsv4 = False
    connector._swa_kv_pool = None
    connector._deduplicate_indexer_group = False

    base_args = lambda **kw: SimpleNamespace(**kw)
    ok_pool = SimpleNamespace(kv_buffer=[torch.zeros((4, 1, 8))])
    connector._validate_kv_pool_support(ok_pool, ok_pool, base_args())

    cases = [
        (ok_pool, base_args(enable_kv_cache_sharding=True), "page-interleaved"),
        (ok_pool, base_args(enable_hisparse=True), "HiSparse"),
        (ok_pool, base_args(dcp_size=2), "Decode Context Parallel"),
        (ok_pool, base_args(enable_unified_memory=True), "unified-memory"),
        (SimpleNamespace(k_scale_buffer=[torch.ones(1)]), base_args(), "external quantization scale"),
        (SimpleNamespace(enable_int8_kv_cache=True, int8_kv_cache_offline_load=False), base_args(), "external quantization scale"),
        (SimpleNamespace(mamba_pool=object()), base_args(), "Mamba"),
        (SimpleNamespace(sparse_layer_id_mapping={0: 0}), base_args(), "MiniMaxSparseKVPool"),
    ]
    for pool, args, msg in cases:
        with pytest.raises(ValueError, match=msg):
            connector._validate_kv_pool_support(pool, pool, args)

    # Offline INT8 is permitted
    connector._validate_kv_pool_support(
        SimpleNamespace(enable_int8_kv_cache=True, int8_kv_cache_offline_load=True),
        SimpleNamespace(enable_int8_kv_cache=True, int8_kv_cache_offline_load=True),
        base_args(),
    )

    # K-only without v_buffer is rejected explicitly
    with pytest.raises(ValueError, match="K-only"):
        connector._resolve_kv_buffers(SimpleNamespace(k_buffer=[torch.zeros(4)], v_buffer=None))


@pytest.mark.parametrize("tokens, slots, agrees", [(4, 4, False), (3, 3, True), (4, 3, False)])
def test_cp_store_votes_before_short_or_invalid_input_returns(tokens, slots, agrees):
    connector = _follower_connector(None)
    connector.page_size = 4
    connector._cache_layer_split = True
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.cp_agrees = MagicMock(return_value=agrees)
    connector.kv_manager = MagicMock()

    assert connector.store_kv("request", range(tokens), torch.arange(slots)) == -1

    connector._sync_ctx.cp_agrees.assert_called_once()
    if tokens != slots:
        connector._sync_ctx.cp_agrees.assert_called_with(None)
    connector.kv_manager.put_match.assert_not_called()
    connector.kv_manager.launch.assert_not_called()


def test_cp_store_marks_uncertainty_after_launch_attempt():
    connector = _follower_connector(None)
    connector._cache_layer_split = True
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.cp_agrees = MagicMock(return_value=True)
    connector._sync_ctx.scatter.side_effect = lambda value, **_: value
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = (
        17, np.ones(4, dtype=np.bool_), PutMatchDisposition.READS_GPU
    )
    connector.kv_manager.launch.side_effect = OSError()

    with pytest.raises(StoreSubmissionUncertainError, match="OSError"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert connector._inflight_stores == {"request": 17}


def test_draft_store_marks_uncertainty_after_launch_attempt():
    connector = _follower_connector(None)
    connector._draft_regions = (object(),)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.scatter.side_effect = lambda value, **_: value
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = (
        17, np.ones(4, dtype=np.bool_), PutMatchDisposition.READS_GPU
    )
    connector.kv_manager.launch.side_effect = OSError()

    with pytest.raises(StoreSubmissionUncertainError, match="OSError"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert connector._inflight_stores == {"request": 17}


def test_cp_store_notification_failure_without_leader_launch_has_no_inflight_task():
    connector = _follower_connector(None)
    connector._cache_layer_split = True
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.cp_agrees = MagicMock(return_value=True)
    connector._sync_ctx.scatter.side_effect = OSError("start notification failed")
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = None

    with pytest.raises(OSError, match="start notification failed"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert not connector._inflight_stores
    connector.kv_manager.launch.assert_not_called()


def test_draft_store_notification_failure_without_launch_has_no_task():
    connector = _follower_connector(None)
    connector._draft_regions = (object(),)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.scatter.side_effect = OSError("start notification failed")
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match.return_value = None

    with pytest.raises(OSError, match="start notification failed"):
        connector.store_kv("request", range(4), torch.arange(4))
    assert not connector._inflight_stores
    connector.kv_manager.launch.assert_not_called()


def _make_batch_leader_connector(page_size: int = 64):
    connector = _follower_connector(None)
    connector.page_size = page_size
    connector._cache_layer_split = True
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.cp_agrees = MagicMock(return_value=True)
    connector._sync_ctx.barrier = MagicMock()
    connector._sync_ctx.scatter.side_effect = lambda value, **_: value
    connector._swa_kv_pool = None
    connector._cache_namespace = lambda ns: ns
    connector._store_profile_scope = lambda _name: nullcontext()
    connector._slot_signature = lambda slots: slots.cpu().numpy().tobytes()
    connector._to_cpu_int64 = lambda t: t.cpu().to(torch.int64)
    connector._build_swa_slot_mapping = lambda _t: None
    connector.tp_client = None
    connector.kv_manager = MagicMock()
    return connector


def test_store_kv_batch_coalesces_small_requests_into_single_launch_and_scatter():
    connector = _make_batch_leader_connector(page_size=64)
    tids = iter([101, 102, 103, 104])
    connector.kv_manager.put_match.side_effect = lambda **kw: (
        next(tids),
        np.ones(len(kw["token_ids"]), dtype=np.bool_),
        PutMatchDisposition.READS_GPU,
    )
    items = [(f"r{i}", range(128), torch.arange(128)) for i in range(4)]

    results = connector.store_kv_batch_with_result(items)

    assert [r.task_id for r in results] == [101, 102, 103, 104]
    assert connector.kv_manager.launch.call_count == 1
    assert connector.kv_manager.launch.call_args.kwargs["task_ids"] == [101, 102, 103, 104]
    connector._sync_ctx.scatter.assert_called_once()
    assert connector._inflight_stores == {"r0": 101, "r1": 102, "r2": 103, "r3": 104}


def test_store_kv_batch_flushes_large_requests_incrementally_with_single_scatter():
    connector = _make_batch_leader_connector(page_size=64)
    events = []
    tids = iter([201, 202, 203, 204])

    def _put_match(**kw):
        tid = next(tids)
        events.append(("put_match", tid))
        return tid, np.ones(len(kw["token_ids"]), dtype=np.bool_), PutMatchDisposition.READS_GPU

    def _launch(**kw):
        events.append(("launch", list(kw["task_ids"])))
        return list(kw["task_ids"])

    connector.kv_manager.put_match.side_effect = _put_match
    connector.kv_manager.launch.side_effect = _launch
    items = [(f"r{i}", range(4096), torch.arange(4096)) for i in range(4)]

    results = connector.store_kv_batch_with_result(items)

    assert [r.task_id for r in results] == [201, 202, 203, 204]
    assert events == [
        ("put_match", 201),
        ("launch", [201]),
        ("put_match", 202),
        ("launch", [202]),
        ("put_match", 203),
        ("launch", [203]),
        ("put_match", 204),
        ("launch", [204]),
    ]
    connector._sync_ctx.scatter.assert_called_once()


def test_store_kv_batch_stops_after_partial_launch_failure():
    connector = _make_batch_leader_connector(page_size=64)
    tids = iter([301, 302, 303])
    connector.kv_manager.put_match.side_effect = lambda **kw: (
        next(tids),
        np.ones(len(kw["token_ids"]), dtype=np.bool_),
        PutMatchDisposition.READS_GPU,
    )
    connector.kv_manager.launch.side_effect = [[301], OSError("launch boom")]
    items = [(f"r{i}", range(4096), torch.arange(4096)) for i in range(3)]

    with pytest.raises(StoreSubmissionUncertainError, match="launch boom"):
        connector.store_kv_batch_with_result(items)

    assert connector.kv_manager.put_match.call_count == 2
    assert connector.kv_manager.launch.call_count == 2
    assert connector._inflight_stores == {"r0": 301, "r1": 302}
    connector._sync_ctx.scatter.assert_called_once()


def test_store_kv_batch_edge_cases_empty_single_mixed_cp_abort_and_follower():
    connector = _make_batch_leader_connector(page_size=64)
    assert connector.store_kv_batch_with_result([]) == []

    # Single-item batch delegates directly to store_kv_with_result
    connector.store_kv_with_result = MagicMock(return_value=MagicMock(task_id=99))
    single = connector.store_kv_batch_with_result([("r0", range(64), torch.arange(64), "req0", ["ns"])])
    assert len(single) == 1 and single[0].task_id == 99
    connector.store_kv_with_result.assert_called_once()
    del connector.store_kv_with_result

    # Mixed cached, partial-block short item, and active item
    connector.kv_manager.put_match.side_effect = [
        (-1, np.zeros(64, dtype=np.bool_), PutMatchDisposition.ALREADY_CACHED),
        (401, np.ones(64, dtype=np.bool_), PutMatchDisposition.READS_GPU),
    ]
    mixed = connector.store_kv_batch_with_result([
        ("r-cached", range(64), torch.arange(64)),
        ("r-short", range(32), torch.arange(32)),
        ("r-active", range(64), torch.arange(64)),
    ])
    assert mixed[0].already_cached and mixed[0].task_id == -1
    assert mixed[1].task_id == -1 and not mixed[1].already_cached
    assert mixed[2].task_id == 401 and mixed[2].reads_gpu
    assert connector.kv_manager.launch.call_args.kwargs["task_ids"] == [401]

    # CP disagreement aborts the whole batch cleanly
    connector = _make_batch_leader_connector(page_size=64)
    connector._sync_ctx.cp_agrees.return_value = False
    aborted = connector.store_kv_batch_with_result([
        ("a", range(64), torch.arange(64)),
        ("b", range(64), torch.arange(64)),
    ])
    assert all(r.task_id == -1 for r in aborted)
    connector.kv_manager.put_match.assert_not_called()

    # Non-split mode rejects mismatched lengths loudly
    non_split = _make_batch_leader_connector(page_size=64)
    non_split._cache_layer_split = False
    with pytest.raises(ValueError, match="has 64 entries"):
        non_split.store_kv_batch_with_result([
            ("a", range(64), torch.arange(32)),
            ("b", range(64), torch.arange(64)),
        ])

    # Follower receives scattered batch results and raises StoreSubmissionUncertainError on failure
    follower = _follower_connector({
        "items": [
            {"rid": "f0", "task_id": 501, "active": True, "already_cached": False, "reads_gpu": True, "slot_count": 64},
            {"rid": "f1", "task_id": -1, "active": False, "already_cached": True, "reads_gpu": False, "slot_count": 64},
        ]
    })
    follower.page_size = 64
    follower._cache_layer_split = False
    f_res = follower.store_kv_batch_with_result([
        ("f0", range(64), torch.arange(64)),
        ("f1", range(64), torch.arange(64)),
    ])
    assert [(r.task_id, r.reads_gpu, r.already_cached) for r in f_res] == [
        (501, True, False), (-1, False, True)
    ]
    assert follower._inflight_stores == {"f0": 501}

    follower._sync_ctx.scatter.side_effect = OSError("scatter lost")
    with pytest.raises(StoreSubmissionUncertainError, match="scatter lost"):
        follower.store_kv_batch_with_result([
            ("f2", range(64), torch.arange(64)),
            ("f3", range(64), torch.arange(64)),
        ])
