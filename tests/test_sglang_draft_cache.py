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
    c._resolve_draft_registration(target, draft, args, None, key_format)
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
        c._resolve_draft_registration(target, (draft,), args, None, "token")
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


def test_standalone_draft_rejects_cp_and_layer_split():
    target = mla()
    draft = mla(1)
    with pytest.raises(ValueError, match="cache layer split"):
        build_draft_registration(target, (draft,), ModelConfig(num_layers=2, cp_size=2), 4)

    target.layer_shard_enabled = True
    with pytest.raises(ValueError, match="cache layer split"):
        build_draft_registration(target, (draft,), ModelConfig(num_layers=2), 4)


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
