"""Owner-only registration and request planning on the real transfer paths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.integration.sglang.layer_split import resolve_layer_split


OWNERS = (0, 0, 0, 1, 1)
ACTIVE = (0, 2, 4)


def pool_for(rank, page_major):
    def row(shape, owned):
        return torch.zeros(shape if owned else (0, *shape[1:]), dtype=torch.uint8)

    kv_shape = (3, 1, 4, 8) if page_major else (12, 1, 8)
    scratch = row((3, 32), True)
    pool = SimpleNamespace(
        layer_num=5, start_layer=10, layer_shard_rank=rank, layer_shard_size=2,
        layer_shard_enabled=True, _get_layer_owner_rank=lambda i: OWNERS[i - 10],
        skip_topk_layers=[i not in ACTIVE for i in range(5)],
        kv_buffer=[row(kv_shape, owner == rank) for owner in OWNERS],
        remote_kv_buffer=row(kv_shape, True),
        index_k_with_scale_buffer=[
            None if page_major and i not in ACTIVE else row((3, 32), i in ACTIVE and rank == owner)
            for i, owner in enumerate(OWNERS)
        ],
    )
    if page_major:
        pool.remote_index_k_with_scale_buffer = scratch
    else:
        pool.index_key_cache = SimpleNamespace(remote_buffer=scratch)
    return pool


@pytest.mark.parametrize("page_major", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_split_registration(rank, page_major):
    pool = pool_for(rank, page_major)
    holders, kv, active, index = resolve_layer_split(pool, 4, rank, 2, 5, None)
    assert holders == OWNERS and active == ACTIVE
    for layer, buf in enumerate(kv):
        assert buf.shape == (12, 1, 8)
        expected = pool.kv_buffer[layer] if holders[layer] == rank else pool.remote_kv_buffer
        assert buf.data_ptr() == expected.data_ptr()
    for layer, buf in zip(active, index):
        assert buf.shape == (3, 32)
        if holders[layer] == rank:
            assert buf is pool.index_k_with_scale_buffer[layer]
        else:
            assert not pool.index_k_with_scale_buffer[layer].numel()
            assert buf is (pool.remote_index_k_with_scale_buffer if page_major else pool.index_key_cache.remote_buffer)


@pytest.mark.parametrize("fault", ["owner_empty", "nonowner_full", "overlap", "mask", "plan", "index_pages"])
def test_bad_registration_fails_before_export(fault):
    pool = pool_for(1, False)
    plan = None
    if fault == "owner_empty":
        pool.index_k_with_scale_buffer[4] = pool.index_k_with_scale_buffer[0]
    elif fault == "nonowner_full":
        pool.kv_buffer[0] = pool.remote_kv_buffer
    elif fault == "overlap":
        pool.kv_buffer[3] = pool.kv_buffer[4]
    elif fault == "mask":
        pool.skip_topk_layers.pop()
    elif fault == "plan":
        plan = SimpleNamespace(cp_rank=1, cp_size=2, num_layers=5, holder_of=lambda i: 0)
    else:
        pool.index_key_cache.remote_buffer = torch.empty((2, 32), dtype=torch.uint8)
    with pytest.raises(ValueError):
        resolve_layer_split(pool, 4, 1, 2, 5, plan)


def test_worker_routes_compact_layers_and_preserves_swa_policy():
    from flexkv.c_ext import RankShareMode
    from flexkv.common.config import LayerGroupSpec
    from flexkv.common.pool import PoolId
    from flexkv.common.transfer import TransferType
    from flexkv.transfer.completion import CompletionContract
    from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool

    w = object.__new__(GPUCPUTransferWorker)
    w._cp_holders = OWNERS
    w.num_gpus = 2
    w._num_original_layers = 5
    w.region_batch = Mock(num_regions=3)
    w._compressor = SimpleNamespace(needs_gpu_cpu_thread_group=False)
    w._rank_share_mode = RankShareMode.SHARDED
    w._use_async_launch = w.use_ce_transfer_d2h = w.use_ce_transfer_h2d = False
    w.transfer_num_cta_d2h = w.transfer_num_cta_h2d = 4
    w._completion = CompletionContract.PER_LAYER
    w._request_cache = {}
    w.layer_groups = [LayerGroupSpec(5, 1, 8, list(range(5))), LayerGroupSpec(3, 1, 32, list(ACTIVE))]
    full = _Pool(PoolId.FULL_KV, "full", region_indices=[0, 1])
    full.layer_members = [[(0, i)] for i in range(5)]
    for local, layer in enumerate(ACTIVE):
        full.layer_members[layer].append((1, local))
    swa = _Pool(PoolId.SWA, "swa", region_indices=[2], layer_members=[[(0, i)] for i in range(5)])
    w._pools = {PoolId.FULL_KV: full, PoolId.SWA: swa}
    w._init_cp_routes(full)
    w._layer_milestones = w._build_layer_milestones()
    for direction in (TransferType.D2H, TransferType.H2D):
        w._transfer_impl(torch.arange(2), torch.arange(2), direction)
        reqs, _ = w.region_batch.submit.call_args.args
        assert [(r.region_index, r.layer_id, r.layer_granularity, r.designated_rank)
                for r in reqs] == [(0, 0, 3, 0), (0, 3, 2, 1), (1, 0, 2, 0), (1, 2, 1, 1)]
        assert all(r.rank_share_mode == RankShareMode.CP_OWNER_ONLY for r in reqs)
    for has_swa in (False, True):
        compiled = w._compiled_requests(has_swa)
        assert not compiled.empty_layers
        swa_count = 0
        for req in compiled.requests:
            if req.region_index == 2:
                swa_count += 1
                assert req.rank_share_mode == RankShareMode.SHARDED
            else:
                assert req.rank_share_mode == RankShareMode.CP_OWNER_ONLY
                assert req.designated_rank == OWNERS[req.milestone_layer]
        assert swa_count == (5 if has_swa else 0)


def test_sparse_cp_cache_namespace_preserves_user_keys():
    from flexkv.integration.sglang.connector import FlexKVConnector

    connector = object.__new__(FlexKVConnector)
    keys = ["tenant", "adapter"]
    assert connector._cache_namespace(keys) is keys
    connector._cache_layout_key = "layout.a"
    original = connector._cache_namespace(keys)
    assert original != keys
    assert connector._cache_namespace(["tenant\x00adapter"]) != original
    assert connector._cache_namespace([]) == connector._cache_namespace(None)
    connector._cache_layout_key = "layout.b"
    assert connector._cache_namespace(keys) != original


def test_single_node_sparse_layout_is_scoped_to_model_revision(monkeypatch):
    from flexkv.common.config import CacheConfig, ModelConfig, RankInfo
    from flexkv.common.storage import KVCacheLayoutType
    from flexkv.integration.sglang import connector

    pool = pool_for(0, False)
    pool.invalidate_layer_split_cache = lambda: None
    model = ModelConfig(num_layers=5, cp_size=2, kv_dim=1)
    model.freeze()
    c = object.__new__(connector.FlexKVConnector)
    c.page_size, c._kvcache, c._is_dsv4, c._swa_kv_pool = 4, pool, False, None
    c.cache_config = CacheConfig()
    c.flexkv_config = SimpleNamespace(model_config=model)
    group = SimpleNamespace(size=lambda: 2)
    c._sync_ctx = SimpleNamespace(needs_sync=True, attn_cp_cpu_group=group)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "region_batch", True)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object",
        lambda peers, signature, group: peers.__setitem__(slice(None), [signature] * 2),
    )

    keys = []
    for model_identity in (("/weights/a", "v1"), ("/weights/a", "v2"), ("/weights/b", "v1")):
        c.model_config = model
        c.rank_info = RankInfo(model_config=model, cp_rank=0, pp_start_layer=0, pp_end_layer=5)
        c._resolve_cache_layer_split(pool, None, model_identity=model_identity)
        keys.append(c._cache_layout_key)
    assert len(set(keys)) == 3


def test_cp8_78_layer_host_geometry():
    from flexkv.integration.sglang.connector import FlexKVConnector, _IndexerBufferGroup

    count, page = 78, 128
    holders = tuple(r for r in range(8) for _ in range(10 if r < 6 else 9))
    active = tuple(range(0, 61, 3))  # 21 layers, including a rank with no local indexer
    rank = 7
    kv_shape, idx_shape = (page, 1, 656), (1, page * 132)
    kv = [torch.empty(kv_shape if r == rank else (0, 1, 656), dtype=torch.int8) for r in holders]
    ix = [torch.empty((0, idx_shape[1]), dtype=torch.uint8) if i in active else None for i in range(count)]
    pool = SimpleNamespace(layer_num=count, start_layer=0, layer_shard_rank=rank,
                           layer_shard_size=8, layer_shard_enabled=True,
                           _get_layer_owner_rank=holders.__getitem__,
                           skip_topk_layers=[i not in active for i in range(count)],
                           kv_buffer=kv, remote_kv_buffer=torch.empty(kv_shape, dtype=torch.int8),
                           index_k_with_scale_buffer=ix,
                           remote_index_k_with_scale_buffer=torch.empty(idx_shape, dtype=torch.uint8))
    owners, kv, ids, idx = resolve_layer_split(pool, page, rank, 8, count, None)
    c = object.__new__(FlexKVConnector)
    c.page_size = page
    c.rank_info = SimpleNamespace(num_layers_per_pp_stage=count)
    specs = c._build_indexer_layer_group_specs(kv, _IndexerBufferGroup(tuple(idx), ids, count))
    assert owners == holders
    assert all(b is pool.remote_index_k_with_scale_buffer for b in idx)
    assert sum(g.num_layers * g.head_size * g.dtype.itemsize * (page // g.compress_ratio)
               for g in specs) == page * (78 * 656 + 21 * 132)


def test_plain_mla_split_without_indexer(monkeypatch):
    from flexkv.common.config import CacheConfig, LayerGroupSpec, ModelConfig, RankInfo
    from flexkv.common.storage import KVCacheLayoutType
    from flexkv.integration.sglang import connector

    rank = 1
    kv_shape = (12, 1, 8)
    pool = SimpleNamespace(
        layer_num=5,
        start_layer=10,
        layer_shard_rank=rank,
        layer_shard_size=2,
        layer_shard_enabled=True,
        _get_layer_owner_rank=lambda i: OWNERS[i - 10],
        kv_buffer=[torch.zeros(kv_shape if o == rank else (0, 1, 8), dtype=torch.bfloat16) for o in OWNERS],
        remote_kv_buffer=torch.zeros(kv_shape, dtype=torch.bfloat16),
        invalidate_layer_split_cache=lambda: None,
    )
    holders, kv, active, index = resolve_layer_split(pool, 4, rank, 2, 5, None)
    assert holders == OWNERS and active == () and index is None
    assert len(kv) == 5

    model = ModelConfig(num_layers=5, cp_size=2, kv_dim=1)
    model.freeze()
    c = object.__new__(connector.FlexKVConnector)
    c.page_size, c._kvcache, c._is_dsv4, c._swa_kv_pool = 4, pool, False, None
    c.cache_config = CacheConfig()
    c.flexkv_config = SimpleNamespace(model_config=model)
    c.model_config = model
    c.rank_info = RankInfo(model_config=model, cp_rank=rank, pp_start_layer=0, pp_end_layer=5)
    c._sync_ctx = SimpleNamespace(needs_sync=True, attn_cp_cpu_group=SimpleNamespace(size=lambda: 2))
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "region_batch", True)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda peers, signature, group: peers.__setitem__(slice(None), [signature] * 2),
    )

    out_kv, out_indexer = c._resolve_cache_layer_split(pool, None, model_identity=("m", "v1"))
    assert len(out_kv) == 5 and out_indexer is None
    assert c.sparse_indexer is False
    assert c._cache_layout_key is None
    assert c.model_config.layer_groups == [
        LayerGroupSpec(5, 1, 8, [0, 1, 2, 3, 4], compress_ratio=1, dtype=torch.bfloat16)
    ]

    c.tp_client = Mock()
    c._label = "[plain-mla-split]"
    c._register_standard_to_server(out_kv, out_indexer)
    call_kwargs = c.tp_client.register_to_server.call_args.kwargs
    assert call_kwargs["layer_groups"] == c.model_config.layer_groups
    assert len(call_kwargs["gpu_layouts"]) == 1
    assert call_kwargs["handles_per_group"] == [out_kv]


def test_turboquant_format_tag_and_dtype_resolution(monkeypatch):
    from flexkv.common.config import CacheConfig, ModelConfig, RankInfo
    from flexkv.common.storage import KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.integration.sglang import connector

    cfg = FlexKVConfig()
    cfg._resolve_dtype("turboquant_4bit_nc", torch.bfloat16)
    assert cfg.model_config.dtype == torch.uint8

    pool = pool_for(0, False)
    pool.skip_topk_layers = [False] * 5  # dense indexer, single node
    pool.index_k_with_scale_buffer = [
        torch.zeros((3, 32) if o == 0 else (0, 32), dtype=torch.uint8)
        for o in OWNERS
    ]
    pool.invalidate_layer_split_cache = lambda: None
    model = ModelConfig(num_layers=5, cp_size=2, kv_dim=1)
    model.freeze()
    c = object.__new__(connector.FlexKVConnector)
    c.page_size, c._kvcache, c._is_dsv4, c._swa_kv_pool = 4, pool, False, None
    c.cache_config = CacheConfig()
    c.flexkv_config = SimpleNamespace(model_config=model)
    c._sync_ctx = SimpleNamespace(needs_sync=True, attn_cp_cpu_group=SimpleNamespace(size=lambda: 2))
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "region_batch", True)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda peers, signature, group: peers.__setitem__(slice(None), [signature] * 2),
    )

    tag_keys = []
    for tag in ("tq4-bf16-scale-v3", "tq4-bf16-scale-v4"):
        pool.storage_format_tag = tag
        c.model_config = model
        c.rank_info = RankInfo(model_config=model, cp_rank=0, pp_start_layer=0, pp_end_layer=5)
        c._resolve_cache_layer_split(pool, None, model_identity=("tq-model", "rev1"))
        tag_keys.append(c._cache_layout_key)
    assert all(k is not None for k in tag_keys)
    assert tag_keys[0] != tag_keys[1]


@pytest.mark.parametrize(
    ("args_extra", "pool_extra", "pattern"),
    [
        ({"enable_kv_cache_sharding": True}, {}, "sharding"),
        ({}, {"placement": object()}, "sharding"),
        ({"enable_hisparse": True}, {}, "HiSparse"),
        ({}, {"full_to_hisparse_device_index_mapping": torch.zeros(4, dtype=torch.int64)}, "HiSparse"),
        ({"dcp_size": 2}, {}, "Decode Context Parallel"),
        ({"attn_dcp_size": 2}, {}, "Decode Context Parallel"),
        ({"enable_unified_memory": True}, {}, "unified-memory"),
        ({}, {"_arena": object()}, "unified-memory"),
        ({}, {"requires_physical_write_loc": True}, "unified-memory"),
        ({}, {"kv_scale_buffer": [torch.ones(1)]}, "scale"),
        ({}, {"k_scale_buffer": [torch.ones(1)]}, "scale"),
        ({}, {"enable_int8_kv_cache": True}, "scale"),
        ({}, {"mamba_pool": object()}, "Mamba"),
        ({}, {"sparse_layer_id_mapping": {0: 0}}, "MiniMaxSparse"),
        ({}, {"swa_kv_pool": object()}, "hybrid SWA"),
    ],
)
def test_reject_unsupported_sharded_pools(monkeypatch, args_extra, pool_extra, pattern):
    from flexkv.common.config import CacheConfig, ModelConfig, RankInfo
    from flexkv.integration.sglang import connector

    model = ModelConfig(num_layers=2, kv_dim=1, num_kv_heads=1, head_size=576)
    model.freeze()
    rank_info = RankInfo(model_config=model, pp_start_layer=0, pp_end_layer=2, local_rank=0)
    fake_cfg = SimpleNamespace(
        model_config=model,
        cache_config=CacheConfig(tokens_per_block=4),
        post_init_from_sglang_config=lambda **kwargs: rank_info,
    )
    monkeypatch.setattr(connector.FlexKVConfig, "from_env", classmethod(lambda cls: fake_cfg))
    monkeypatch.setattr(
        connector,
        "FlexKVComm",
        lambda **kwargs: SimpleNamespace(needs_sync=False, all_reduce_min=lambda x: x),
    )
    sgl_model = SimpleNamespace(num_hidden_layers=2, qk_nope_head_dim=512, qk_rope_head_dim=64)
    server_args = SimpleNamespace(tp_size=1, pp_size=1, nnodes=1, node_rank=0, **args_extra)
    kvcache = SimpleNamespace(kv_buffer=[torch.zeros((8, 1, 576), dtype=torch.bfloat16) for _ in range(2)], **pool_extra)
    with pytest.raises(ValueError, match=pattern):
        connector.FlexKVConnector(
            sgl_model_config=sgl_model,
            server_args=server_args,
            page_size=4,
            kvcache=kvcache,
            tp_rank=0,
            dp_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
        )


def test_supported_pool_attrs_pass_validation_and_reject_k_only():
    from flexkv.integration.sglang.connector import FlexKVConnector

    c = object.__new__(FlexKVConnector)
    c._is_dsv4 = False
    c._swa_kv_pool = None
    offline_int8_pool = SimpleNamespace(
        enable_int8_kv_cache=True,
        int8_kv_cache_offline_load=True,
        layer_router=object(),
    )
    c._validate_kv_pool_support(offline_int8_pool, offline_int8_pool, SimpleNamespace())

    k_only_pool = SimpleNamespace(k_buffer=[torch.zeros((4, 1, 8))], v_buffer=None)
    with pytest.raises(ValueError, match="K-only"):
        c._resolve_kv_buffers(k_only_pool)
