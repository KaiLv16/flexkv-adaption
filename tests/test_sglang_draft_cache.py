"""Standalone speculative registration, cache identity and draft readiness without layer split."""
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from flexkv.common.config import CacheConfig, ModelConfig, RankInfo
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.integration.sglang.draft_cache import build_draft_registration


def mla(count=2, width=8, pages=3):
    return SimpleNamespace(
        layer_num=count, page_size=4,
        kv_buffer=[torch.zeros((pages * 4, 1, width), dtype=torch.int8) for _ in range(count)],
    )


def mha(layers=2):
    def rows(width, dtype):
        return [torch.zeros((12, 2, width), dtype=dtype) for _ in range(layers)]
    return SimpleNamespace(layer_num=layers, page_size=4, k_buffer=rows(8, torch.float16),
                           v_buffer=rows(16, torch.uint8))


def connector(target, draft, args, key_format="token"):
    c = object.__new__(FlexKVConnector)
    m = ModelConfig(num_layers=target.layer_num)
    c.model_config = m
    c.flexkv_config = SimpleNamespace(model_config=m)
    c.cache_config = CacheConfig()
    c.rank_info = RankInfo(model_config=m, pp_start_layer=0, pp_end_layer=m.num_layers)
    c._sync_ctx = SimpleNamespace(attn_tp_cpu_group=None)
    c._is_dsv4, c._swa_kv_pool, c._kvcache, c.page_size = False, None, target, 4
    c._resolve_draft_registration(target, draft, args, None, None, key_format)
    return c


def test_multirunner_store_identity_and_layerwise_fence():
    a, b = mla(), mha()
    args = SimpleNamespace(model_path="target", speculative_algorithm="other",
                           speculative_draft_model_revision="r1")
    c = connector(a, (b, mla(1)), args)
    regions = c._draft_regions
    assert [r.name for r in regions] == [
        "target.kv_buffer", "draft.0.k_buffer.0", "draft.0.k_buffer.1",
        "draft.0.v_buffer.0", "draft.0.v_buffer.1", "draft.1.kv_buffer.0",
    ]
    assert all(r.spec.layer_indices == [0] for r in regions[1:])
    assert regions[1].spec.dtype == torch.float16 and regions[3].spec.head_size == 16
    assert regions[1].buffers[0] is b.k_buffer[0]
    c.tp_client = Mock()
    c._register_to_server(regions[0].buffers, None)
    assert c.tp_client.register_to_server.call_args.kwargs["handles_per_group"][3] == [b.v_buffer[0]]

    a.register_layer_transfer_counter = Mock()
    b.register_layer_transfer_counter = Mock()
    c.layer_done_counter = Mock()
    c.register_layer_transfer_counter(a)
    b.register_layer_transfer_counter.call_args.args[0].wait_until(37)
    c.layer_done_counter.wait_until.assert_called_once_with(0)
    a.register_layer_transfer_counter.assert_called_once_with(c.layer_done_counter)

    key = c._cache_namespace(["tenant"])
    assert key == connector(mla(), (mha(), mla(1)), args)._cache_namespace(["tenant"])
    assert key != connector(a, (b, mla(1)), args, "bigram")._cache_namespace(["tenant"])
    args.speculative_draft_model_revision = "r2"
    assert key != connector(a, (b, mla(1)), args)._cache_namespace(["tenant"])


def test_separate_tp_rank_chunks_and_independent_target_kv():
    m = ModelConfig(num_layers=2, tp_size=2)
    rs = build_draft_registration(mha(), (mla(),), m, 4)
    assert [r.name for r in rs[:2]] == ["target.k_buffer", "target.v_buffer"]
    assert all(r.layout.num_kv_heads == 4 for r in rs)
    assert [r.spec.num_kv_heads for r in rs] == [2, 2, 1, 1]


@pytest.mark.parametrize("fault", ["overlap", "target_overlap", "page_count", "scale", "index", "stride", "no_pool"])
def test_reject_incomplete_storage(fault):
    t, d = mla(), mla()
    drafts = [d]
    if fault == "overlap":
        d.kv_buffer[1] = t.kv_buffer[0]
    elif fault == "target_overlap":
        t.kv_buffer[1] = t.kv_buffer[0]
    elif fault == "page_count":
        drafts = [mla(pages=2)]
    elif fault == "scale":
        d.k_scale_buffer = [torch.ones(1)]
    elif fault == "index":
        d.index_k_with_scale_buffer = [None, None]
    elif fault == "stride":
        d.kv_buffer[0] = torch.zeros((12, 1, 16))[:, :, ::2]
    else:
        drafts = []
    with pytest.raises(ValueError):
        build_draft_registration(t, drafts, ModelConfig(num_layers=2), 4)


def test_draft_cache_layout_key_shared_across_dp_ranks():
    target = mla()
    draft = mla(1)
    namespaces = []
    layout_keys = []
    for dp_size, dp_rank in ((2, 0), (2, 1), (4, 3)):
        c = object.__new__(FlexKVConnector)
        m = ModelConfig(num_layers=target.layer_num, dp_size=dp_size)
        c.model_config = m
        c.flexkv_config = SimpleNamespace(model_config=m)
        c.cache_config = CacheConfig()
        c.rank_info = RankInfo(model_config=m, dp_rank=dp_rank, pp_start_layer=0, pp_end_layer=m.num_layers)
        c._sync_ctx = SimpleNamespace(attn_tp_cpu_group=None)
        c._is_dsv4, c._swa_kv_pool, c._kvcache, c.page_size = False, None, target, 4
        args = SimpleNamespace(model_path="target", speculative_algorithm="EAGLE", dp_size=dp_size)
        c._resolve_draft_registration(target, (draft,), args, None, None, "token")
        layout_keys.append(c._cache_layout_key)
        namespaces.append(c._cache_namespace(["tenant"]))
    assert len(set(layout_keys)) == 1
    assert len({tuple(ns) for ns in namespaces}) == 1


def test_draft_cache_layout_key_tracks_storage_format_tag():
    args = SimpleNamespace(model_path="target", speculative_algorithm="EAGLE")
    keys = []
    for tag in ("tq4-bf16-scale-v3", "tq4-bf16-scale-v4"):
        target = mla()
        target.storage_format_tag = tag
        keys.append(connector(target, (mla(1),), args)._cache_layout_key)
    assert keys[0] != keys[1]


def test_draft_allows_plain_pool_router_and_rejects_routed_arena():
    # XingLiu KVCache defines @cached_property layer_router -> PlainPoolRouter on all pools;
    # only RoutedKVCache / arena-backed / sparse pools should be rejected.
    target = mla()
    target.layer_router = object()
    draft = mla(1)
    draft.layer_router = object()
    rs = build_draft_registration(target, (draft,), ModelConfig(num_layers=2), 4)
    assert [r.name for r in rs] == ["target.kv_buffer", "draft.0.kv_buffer.0"]

    for attr in ("_arena", "requires_physical_write_loc", "sparse_layer_id_mapping"):
        bad = mla()
        setattr(bad, attr, object() if attr != "requires_physical_write_loc" else True)
        with pytest.raises(ValueError, match="full-attention"):
            build_draft_registration(bad, (mla(1),), ModelConfig(num_layers=2), 4)


def test_cp_draft_requires_resolved_layer_split():
    target = mla()
    draft = mla(1)
    with pytest.raises(ValueError, match="physically layer-split"):
        build_draft_registration(target, (draft,), ModelConfig(num_layers=2, cp_size=2), 4)


def test_standalone_draft_with_sparse_indexer_and_4d_paged_kv():
    target = mla()
    target.skip_topk_layers = [True, False]
    target.index_k_with_scale_buffer = [None, torch.zeros((3, 32), dtype=torch.uint8)]
    draft = SimpleNamespace(
        layer_num=1,
        page_size=4,
        kv_buffer=[torch.zeros((3, 1, 4, 8), dtype=torch.uint8)],
        skip_topk_layers=[False],
        index_k_with_scale_buffer=[torch.zeros((3, 32), dtype=torch.uint8)],
    )
    rs = build_draft_registration(target, [draft], ModelConfig(num_layers=2), 4)
    assert [r.name for r in rs] == [
        "target.kv_buffer", "target.index", "draft.0.kv_buffer.0", "draft.0.index.0",
    ]
    assert rs[1].layout.tokens_per_block == 1
    assert rs[2].spec.head_size == 8
    assert rs[3].layout.tokens_per_block == 1


def test_standalone_draft_coordinated_retrieve_layerwise_and_reset():
    c = object.__new__(FlexKVConnector)
    c._draft_regions = (object(),)
    c._coordinated_restore
    assert c._coordinated_restore is True
    c.enable_layerwise = False
    c._pending_lookups = {}
    c._pending_lookup_contexts = {}
    c._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        all_reduce_min=MagicMock(return_value=-1),
    )
    with pytest.raises(RuntimeError, match="KV restore failed on a peer"):
        c.retrieve_kv("r1", torch.arange(4))

    c._layerwise_batch_launch_error = "unresolved"
    with pytest.raises(RuntimeError, match="unresolved"):
        c.reset()


def test_draft_kvcache_requires_env_opt_in(monkeypatch):
    monkeypatch.setenv("FLEXKV_CACHE_DRAFT_KV", "0")
    with pytest.raises(ValueError, match="FLEXKV_CACHE_DRAFT_KV=1"):
        FlexKVConnector(
            sgl_model_config=SimpleNamespace(),
            server_args=SimpleNamespace(),
            page_size=4,
            kvcache=SimpleNamespace(),
            tp_rank=0,
            dp_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            draft_kvcache=mla(1),
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_cp_rank_private_draft_kv_and_sparse_index(rank):
    from flexkv.integration.sglang.layer_split import resolve_layer_split
    from test_sglang_cache_layer_split import pool_for

    target = pool_for(rank, True)
    split = resolve_layer_split(target, 4, rank, 2, 5, None)
    draft = mla()
    draft.skip_topk_layers = [True, False]
    draft.index_k_with_scale_buffer = [None, torch.zeros((3, 32), dtype=torch.uint8)]
    model = ModelConfig(num_layers=5, cp_size=2)
    rs = build_draft_registration(target, [draft], model, 4, split)
    assert [r.spec.transfer_owner_rank for r in rs[2:]] == [0, 1] * 3
    assert all(r.spec.layer_indices == [0] for r in rs[2:])
    assert rs[-1].buffers[0] is draft.index_k_with_scale_buffer[1]
    assert rs[-1].layout.tokens_per_block == 1
    assert sum(r.spec.num_layers * r.layout.get_chunk_size() * r.spec.dtype.itemsize
               for r in rs) == 5 * 32 + 3 * 32 + 2 * (2 * 32 + 32)


def test_cp_block_offsets_roundtrip_distinct_draft_replicas():
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.sglang.layer_split import resolve_layer_split
    from flexkv.transfer.template import compile_host_regions
    from test_sglang_cache_layer_split import pool_for

    model = ModelConfig(num_layers=5, cp_size=2)
    ranks = []
    for rank in range(2):
        t = pool_for(rank, True)
        split = resolve_layer_split(t, 4, rank, 2, 5, None)
        d = mla(1)
        d.kv_buffer[0].fill_(19 + rank)
        ranks.append(build_draft_registration(t, (d,), model, 4, split))
    groups = [r.spec for r in ranks[0]]
    host_layout = KVCacheLayout(type=KVCacheLayoutType.BLOCKFIRST, num_layer=5, num_block=3,
                                tokens_per_block=4, num_head=1, head_size=8, kv_dim=1,
                                layer_groups=groups)
    offsets = compile_host_regions(groups, host_layout, 1, torch.uint8)
    block = torch.zeros(host_layout.get_block_stride(), dtype=torch.uint8)
    for ordinal in (2, 3):
        owner = groups[ordinal].transfer_owner_rank
        src = ranks[owner][ordinal].buffers[0].view(torch.uint8).reshape(3, -1)
        offset = offsets[ordinal]
        saved = block[offset.base_offset:offset.base_offset + offset.chunk_bytes]
        saved.copy_(src[2])
        src[0].zero_()
    for ordinal in (2, 3):
        owner = groups[ordinal].transfer_owner_rank
        dst = ranks[owner][ordinal].buffers[0].view(torch.uint8).reshape(3, -1)
        offset = offsets[ordinal]
        dst[0].copy_(block[offset.base_offset:offset.base_offset + offset.chunk_bytes])
        assert torch.all(dst[0] == 19 + owner)


def test_plain_mla_layer_split_with_draft():
    from flexkv.integration.sglang.layer_split import resolve_layer_split

    kv_shape = (12, 1, 8)
    owners = (0, 0, 1, 1, 1)
    for rank in (0, 1):
        target = SimpleNamespace(
            layer_num=5,
            start_layer=0,
            layer_shard_rank=rank,
            layer_shard_size=2,
            layer_shard_enabled=True,
            _get_layer_owner_rank=lambda i: owners[i],
            kv_buffer=[torch.zeros(kv_shape if o == rank else (0, 1, 8), dtype=torch.uint8) for o in owners],
            remote_kv_buffer=torch.zeros(kv_shape, dtype=torch.uint8),
        )
        split = resolve_layer_split(target, 4, rank, 2, 5, None)
        assert split[2] == () and split[3] is None
        draft = mla(1)
        rs = build_draft_registration(target, [draft], ModelConfig(num_layers=5, cp_size=2), 4, split)
        assert [r.name for r in rs] == ["target.kv_buffer", "draft.0.kv_buffer.0", "draft.0.kv_buffer.0"]
        assert [r.spec.transfer_owner_rank for r in rs] == [None, 0, 1]


def test_worker_routes_cp_fixed_owner_draft_group_and_preserves_swa_policy():
    from flexkv.c_ext import RankShareMode
    from flexkv.common.config import LayerGroupSpec
    from flexkv.common.pool import PoolId
    from flexkv.common.transfer import TransferType
    from flexkv.transfer.completion import CompletionContract
    from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool
    from test_sglang_cache_layer_split import ACTIVE, OWNERS

    w = object.__new__(GPUCPUTransferWorker)
    w._cp_holders = OWNERS
    w.num_gpus = 2
    w._num_original_layers = 5
    w.region_batch = Mock(num_regions=4)
    w._compressor = SimpleNamespace(needs_gpu_cpu_thread_group=False)
    w._rank_share_mode = RankShareMode.SHARDED
    w._use_async_launch = w.use_ce_transfer_d2h = w.use_ce_transfer_h2d = False
    w.transfer_num_cta_d2h = w.transfer_num_cta_h2d = 4
    w._completion = CompletionContract.PER_LAYER
    w._request_cache = {}
    w.layer_groups = [
        LayerGroupSpec(5, 1, 8, list(range(5))),
        LayerGroupSpec(3, 1, 32, list(ACTIVE)),
        LayerGroupSpec(1, 1, 8, [0], transfer_owner_rank=1),
    ]
    full = _Pool(PoolId.FULL_KV, "full", region_indices=[0, 1, 2])
    full.layer_members = [[(0, i)] for i in range(5)]
    for local, layer in enumerate(ACTIVE):
        full.layer_members[layer].append((1, local))
    full.layer_members[0].append((2, 0))
    swa = _Pool(PoolId.SWA, "swa", region_indices=[3], layer_members=[[(0, i)] for i in range(5)])
    w._pools = {PoolId.FULL_KV: full, PoolId.SWA: swa}
    w._init_cp_routes(full)
    w._layer_milestones = w._build_layer_milestones()
    for direction in (TransferType.D2H, TransferType.H2D):
        w._transfer_impl(torch.arange(2), torch.arange(2), direction)
        reqs, _ = w.region_batch.submit.call_args.args
        assert [(r.region_index, r.layer_id, r.layer_granularity, r.designated_rank)
                for r in reqs] == [(0, 0, 3, 0), (0, 3, 2, 1), (1, 0, 2, 0), (1, 2, 1, 1), (2, 0, 1, 1)]
        assert all(r.rank_share_mode == RankShareMode.CP_OWNER_ONLY for r in reqs)
    for has_swa in (False, True):
        compiled = w._compiled_requests(has_swa)
        assert not compiled.empty_layers
        swa_count = 0
        for req in compiled.requests:
            if req.region_index == 3:
                swa_count += 1
                assert req.rank_share_mode == RankShareMode.SHARDED
            else:
                assert req.rank_share_mode == RankShareMode.CP_OWNER_ONLY
                assert req.designated_rank == (1 if req.region_index == 2 else OWNERS[req.milestone_layer])
        assert swa_count == (5 if has_swa else 0)


def test_multinode_cp_localize_and_packed_page_bytes_with_draft():
    from flexkv.common.cache_layer_split import localize, slice_registration
    from flexkv.common.config import LayerGroupSpec, block_size_in_bytes_for_cache
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

    owners = (0, 0, 1, 2, 3)
    groups = [LayerGroupSpec(5, 1, 8, list(range(5)), dtype=torch.uint8),
              LayerGroupSpec(2, 1, 16, [0, 2], dtype=torch.uint8, compress_ratio=4)]
    groups.extend(LayerGroupSpec(1, 1, 8, [0], dtype=torch.uint8, transfer_owner_rank=r)
                  for r in range(4))
    model = ModelConfig(num_layers=5, cp_size=4, nnodes=2, kv_dim=1,
                        h2d_layer_holders=owners, layer_groups=groups, layer_split_nodes=2)
    sizes = []
    for node in range(2):
        local, selection = localize(model, list(range(node * 2, node * 2 + 2)))
        assert local.h2d_layer_holders == ((0, 0, 1, -1, -1) if node == 0 else (-1, -1, -1, 0, 1))
        assert [g.transfer_owner_rank for g in local.layer_groups[-2:]] == [0, 1]
        assert len(local.layer_groups) == (4 if node == 0 else 3)
        layouts = [KVCacheLayout(KVCacheLayoutType.LAYERFIRST, g.num_layers, 3, 4, 1, 8) for g in groups]
        rows = [g.layer_indices for g in groups]
        blocks, slices = slice_registration(rows, layouts, selection)
        assert blocks == [g.layer_indices for g in local.layer_groups]
        assert [l.num_layer for l in slices] == [len(b) for b in blocks]
        sizes.append(block_size_in_bytes_for_cache(local, CacheConfig(tokens_per_block=4)))
    assert sizes == [192, 128]
    assert block_size_in_bytes_for_cache(model, CacheConfig(tokens_per_block=4)) == max(sizes)

    for nodes, expected in ((1, [3855616]), (2, [2175744, 2083328])):
        cp = 8 * nodes
        cp_owners = tuple(rank for rank in range(cp)
                          for _ in range(78 // cp + (rank < 78 % cp)))
        active = list(range(0, 40, 4)) + [39] + list(range(40, 78, 4))
        cp_groups = [
            LayerGroupSpec(78, 1, 656, list(range(78)), dtype=torch.uint8),
            LayerGroupSpec(21, 1, 8448, active, dtype=torch.uint8, compress_ratio=64),
        ]
        for width, ratio in [(656, 1), (8448, 64)]:
            cp_groups.extend(LayerGroupSpec(1, 1, width, [0], dtype=torch.uint8,
                                            compress_ratio=ratio, transfer_owner_rank=r)
                             for r in range(cp))
        cp_model = ModelConfig(num_layers=78, cp_size=cp, nnodes=nodes, kv_dim=1,
                               h2d_layer_holders=cp_owners, layer_groups=cp_groups,
                               layer_split_nodes=nodes)
        cfg = CacheConfig(tokens_per_block=64)
        per_node = [
            block_size_in_bytes_for_cache(localize(cp_model, list(range(n * 8, (n + 1) * 8)))[0], cfg)
            for n in range(nodes)
        ]
        assert per_node == expected
        assert block_size_in_bytes_for_cache(cp_model, cfg) == max(per_node)


@pytest.mark.parametrize("notify_mode", [None, "polling", "hostfunc"])
@pytest.mark.parametrize("use_ce", [False, True], ids=["sm", "ce"])
def test_cp_draft_owner_roundtrip_gpu(notify_mode, use_ce):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA GPUs")
    from dataclasses import replace
    from eventfd_probe import Fds
    from flexkv.c_ext import RankShareMode
    from test_region_batch_equivalence import RegionSpec, block_ids, build_region_batch, make_requests, sync_all

    owners = (0, 0, 1)
    layers = len(owners)
    blocks, width = 4, 128
    host = torch.zeros((blocks, layers, width), dtype=torch.uint8, pin_memory=True)
    scratch = [torch.full((blocks, width), 0xFE, dtype=torch.uint8, device=f"cuda:{r}")
               for r in range(2)]
    gpu = [[torch.full_like(scratch[r], i + 1) if owner == r else scratch[r]
            for i, owner in enumerate(owners)] for r in range(2)]
    spec = RegionSpec(
        name="split", cpu_ptr=host.data_ptr(), cpu_kv_stride=width,
        cpu_layer_stride=width, cpu_block_stride=layers * width, cpu_tp_stride=0,
        gpu_block_ptrs_flat=[buf.data_ptr() for rank in gpu for buf in rank],
        num_tensors_per_gpu=layers, gpu_kv_strides=[0, 0],
        gpu_block_strides=[width] * 2, gpu_layer_strides=[width * blocks] * 2,
        gpu_chunk_sizes=[width] * 2, num_layers=layers,
    )
    draft = [torch.full_like(scratch[r], 0xA0 + r) for r in range(2)]
    draft_host = torch.zeros((blocks, 2, width), dtype=torch.uint8, pin_memory=True)
    replicas = [
        replace(spec, name=f"draft-rank-{r}", num_layers=1, num_tensors_per_gpu=1,
                cpu_ptr=draft_host.data_ptr() + r * width, cpu_block_stride=2 * width,
                gpu_block_ptrs_flat=[buf.data_ptr() for buf in draft])
        for r in range(2)
    ]
    group = build_region_batch([spec, *replicas], [0, 1], is_blockfirst=True)
    ids = block_ids(blocks)

    def requests(h2d):
        result = []
        for layer, owner in enumerate(owners):
            req, = make_requests(1, ids, ids, h2d, layer_id=layer, layer_granularity=1,
                                 use_ce_transfer=use_ce, share_mode=RankShareMode.CP_OWNER_ONLY,
                                 designated_rank=owner)
            req.milestone_layer = layer
            result.append(req)
        for owner in range(2):
            req, = make_requests(3, ids, ids, h2d, region_indices=[owner + 1],
                                 use_ce_transfer=use_ce, share_mode=RankShareMode.CP_OWNER_ONLY,
                                 designated_rank=owner)
            req.milestone_layer = 0
            result.append(req)
        return result

    sync_all(2)
    group.submit(requests(False), True)
    for layer, owner in enumerate(owners):
        assert torch.all(host[:, layer] == layer + 1)
        gpu[owner][layer].zero_()
    for rank, buf in enumerate(draft):
        assert torch.all(draft_host[:, rank] == 0xA0 + rank)
        buf.zero_()
    sync_all(2)
    if notify_mode is None:
        group.submit(requests(True), True)
    else:
        with Fds(1, 2, layers) as fds:
            group.set_layer_eventfds(fds.tensor(), 2, layers, notify_mode)
            group.submit_layerwise(requests(True), [], 0)
            ok, error = group.wait_layer_completion(30)
            assert ok, error
            group.wait_all_streams()
            for layer in range(layers):
                assert fds.units(layer) == [1, 1]
    for layer, owner in enumerate(owners):
        assert torch.all(gpu[owner][layer] == layer + 1)
    for buf in scratch:
        assert torch.all(buf == 0xFE)
    for rank, buf in enumerate(draft):
        assert torch.all(buf == 0xA0 + rank)


def test_cp_draft_connector_registration_and_counter_wiring(monkeypatch):
    from unittest.mock import patch
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    from flexkv.common.storage import KVCacheLayoutType
    from flexkv.integration.sglang.layer_split import resolve_layer_split
    from test_sglang_cache_layer_split import pool_for

    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "region_batch", True)
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "kv_shared_across_ranks_mode", "first_only")

    target = pool_for(0, True)
    target.invalidate_layer_split_cache = Mock()
    draft = mla(1)
    c = object.__new__(FlexKVConnector)
    m = ModelConfig(num_layers=5, cp_size=2, tp_size=1, kv_dim=1, num_kv_heads=1)
    c.model_config = m
    c.flexkv_config = SimpleNamespace(model_config=m)
    c.cache_config = CacheConfig()
    c.rank_info = RankInfo(model_config=m, pp_start_layer=0, pp_end_layer=5)
    c._sync_ctx = SimpleNamespace(
        attn_cp_cpu_group=SimpleNamespace(size=lambda: 2),
        attn_tp_cpu_group=None,
        needs_sync=True,
    )
    c._is_dsv4, c._swa_kv_pool, c._kvcache, c.page_size = False, None, target, 4
    c._cache_layer_split = False
    c._draft_pools = ()
    c._draft_regions = ()

    args = SimpleNamespace(model_path="target", speculative_algorithm="EAGLE")
    fake_rank_split = resolve_layer_split(target, 4, 0, 2, 5, None)
    assert len(fake_rank_split[1]) == 5

    with patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.all_gather_object", side_effect=lambda out, obj, group=None: out.__setitem__(0, obj) or out.__setitem__(1, obj)):
        kv, indexer = c._resolve_cache_layer_split(target, None, validate_groups=False, model_identity=("target", None))
        split = (
            c.model_config.h2d_layer_holders,
            kv,
            indexer.layer_indices if indexer is not None else (),
            indexer.buffers if indexer is not None else None,
        )
        first_buffers, out_idx = c._resolve_draft_registration(target, (draft,), args, split, None, "bigram")

    assert out_idx is None
    assert first_buffers is c._draft_regions[0].buffers
    assert c._cache_layer_split is True
    assert c._coordinated_restore is True
    assert len(c._draft_regions) == 4  # target.kv_buffer, target.index, draft.0.0 rank 0, draft.0.0 rank 1

    target.register_layer_transfer_counter = Mock()
    draft.register_layer_transfer_counter = Mock()
    c.layer_done_counter = Mock()
    wrapper = SimpleNamespace(full_kv_pool=target, register_layer_transfer_counter=Mock())
    c.register_layer_transfer_counter(wrapper)
    wrapper.register_layer_transfer_counter.assert_called_once_with(c.layer_done_counter)
    target.register_layer_transfer_counter.assert_called_once_with(c.layer_done_counter)
    draft.register_layer_transfer_counter.call_args.args[0].wait_until(2)
    c.layer_done_counter.wait_until.assert_called_once_with(0)

