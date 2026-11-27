import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.common.config import CacheConfig, LayerGroupSpec, ModelConfig, block_size_in_bytes_for_cache
from flexkv.common.cache_layer_split import localize, slice_registration
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.external.mooncake_store_keys import PoolKind, build_key


def test_packed_cp_nodes_skip_empty_index_group():
    owners = (0, 0, 1, 2, 3)
    groups = [LayerGroupSpec(5, 1, 8, list(range(5)), dtype=torch.uint8),
              LayerGroupSpec(2, 1, 16, [0, 2], dtype=torch.uint8, compress_ratio=4)]
    model = ModelConfig(num_layers=5, cp_size=4, nnodes=2, kv_dim=1,
                        h2d_layer_holders=owners, layer_groups=groups, layer_split_nodes=2)
    sizes = []
    for node in range(2):
        local, selection = localize(model, list(range(node * 2, node * 2 + 2)))
        assert local.h2d_layer_holders == ((0, 0, 1, -1, -1) if node == 0 else (-1, -1, -1, 0, 1))
        assert len(local.layer_groups) == (2 if node == 0 else 1)
        layouts = [KVCacheLayout(KVCacheLayoutType.LAYERFIRST, g.num_layers, 3, 4, 1, 8) for g in groups]
        rows = [g.layer_indices for g in groups]
        blocks, slices = slice_registration(rows, layouts, selection)
        assert blocks == [g.layer_indices for g in local.layer_groups]
        assert [l.num_layer for l in slices] == [len(b) for b in blocks]
        sizes.append(block_size_in_bytes_for_cache(local, CacheConfig(tokens_per_block=4)))
    assert sizes == [128, 64]
    assert block_size_in_bytes_for_cache(model, CacheConfig(tokens_per_block=4)) == max(sizes)
    with pytest.raises(ValueError):
        localize(model, [1, 2])


def test_partial_l3_fragments_terminate_prefix_in_one_rpc():
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine
    engine = object.__new__(MooncakeStoreCacheEngine)
    engine.cp_nodes = 2
    engine.mooncake_store_client = Mock()
    engine.mooncake_store_client.batch_exists_impl.return_value = [1, 1, 1, -1, 1, 1]
    assert engine._joint_match_length(["a", "b", "c"]) == (1, 1, 0)
    engine.mooncake_store_client.batch_exists_impl.assert_called_once_with(
        [build_key(h, PoolKind.KV, cp_nodes=2, cp_node=r) for h in "abc" for r in range(2)])


@pytest.mark.parametrize("nodes,width", [(2, 8), (3, 2), (4, 1)])
def test_each_node_round_trips_only_owned_bytes(nodes, width):
    from flexkv.common.transfer import TransferType
    from flexkv.transfer.backends import MooncakeStoreBackend

    layers = nodes * width + 1
    owners = tuple(range(nodes * width)) + (0,)
    model = ModelConfig(num_layers=layers, cp_size=nodes * width, nnodes=nodes,
                        layer_split_nodes=nodes, h2d_layer_holders=owners,
                        layer_groups=[LayerGroupSpec(layers, 1, 1, list(range(layers)))])
    model.freeze()
    ports = [p for node in range(1, nodes) for p in model.remote_ports_for_node(node)]
    assert len(set(ports)) == 3 * (nodes - 1)

    objects = {}

    class Store:
        def batch_put(self, keys, pointers, sizes):
            for key, ptr, size in zip(keys, pointers, sizes, strict=True):
                objects[key] = ctypes.string_at(ptr, size)
            return [True] * len(keys)

        def batch_get(self, keys, pointers, sizes):
            for key, ptr, size in zip(keys, pointers, sizes, strict=True):
                assert len(objects[key]) == size
                ctypes.memmove(ptr, objects[key], size)
            return [True] * len(keys)

    restored = [None] * layers
    for node in range(nodes):
        local, _ = localize(model, list(range(node * width, (node + 1) * width)))
        indices = local.layer_groups[0].layer_indices
        backend = MooncakeStoreBackend(CacheConfig(layer_split_node=node, layer_split_nodes=nodes), PoolKind.KV)
        backend.block_size_bytes = len(indices)
        backend._cpu_buffer = torch.zeros((2, len(indices)), dtype=torch.uint8)
        backend._cpu_buffer[0] = torch.tensor(indices)
        backend.mooncake_client = Store()
        op = SimpleNamespace(valid_block_num=1, transfer_op_id=1, transfer_graph_id=1,
                             transfer_type=TransferType.H2REMOTE, mooncake_store_block_hashes=["prefix"])
        assert backend.transfer_blocks(None, op, [0], [0]) == ((True,), len(indices))
        op.transfer_type = TransferType.REMOTE2H
        assert backend.transfer_blocks(None, op, [0], [1]) == ((True,), len(indices))
        for layer, value in zip(indices, backend._cpu_buffer[1].tolist(), strict=True):
            restored[layer] = value
    assert restored == list(range(layers))
    assert len(objects) == nodes and sum(map(len, objects.values())) == layers


@pytest.mark.parametrize("nodes,expected", [
    (1, [3452160]), (2, [1772288, 1679872]),
])
def test_packed_page_bytes(nodes, expected):
    cp = 8 * nodes
    owners = tuple(rank for rank in range(cp)
                   for _ in range(78 // cp + (rank < 78 % cp)))
    active = list(range(0, 40, 4)) + [39] + list(range(40, 78, 4))
    groups = [
        LayerGroupSpec(78, 1, 656, list(range(78)), dtype=torch.uint8),
        LayerGroupSpec(21, 1, 8448, active, dtype=torch.uint8, compress_ratio=64),
    ]
    model = ModelConfig(num_layers=78, cp_size=cp, nnodes=nodes, kv_dim=1,
                        h2d_layer_holders=owners, layer_groups=groups, layer_split_nodes=nodes)
    cfg = CacheConfig(tokens_per_block=64)
    for node, size in enumerate(expected):
        local, _ = localize(model, list(range(node * 8, node * 8 + 8)))
        assert block_size_in_bytes_for_cache(local, cfg) == size
    assert block_size_in_bytes_for_cache(model, cfg) == max(expected)


def test_multinode_plain_mla_localize_and_slice():
    owners = (0, 0, 1, 2, 3)
    groups = [LayerGroupSpec(5, 1, 8, list(range(5)), dtype=torch.bfloat16)]
    model = ModelConfig(
        num_layers=5,
        cp_size=4,
        nnodes=2,
        kv_dim=1,
        h2d_layer_holders=owners,
        layer_groups=groups,
        layer_split_nodes=2,
    )
    for node in range(2):
        local, selection = localize(model, list(range(node * 2, node * 2 + 2)))
        layouts = [KVCacheLayout(KVCacheLayoutType.LAYERFIRST, 5, 3, 4, 1, 8)]
        rows = [list(range(5))]
        blocks, slices = slice_registration(rows, layouts, selection)
        expected_rows = [0, 1, 2] if node == 0 else [3, 4]
        assert blocks == [expected_rows]
        assert [s.num_layer for s in slices] == [len(expected_rows)]
        assert local.layer_groups[0].layer_indices == expected_rows


def test_multinode_cp_rejects_chunked_prefetch(monkeypatch):
    from flexkv.integration.sglang import connector

    model = ModelConfig(num_layers=4, cp_size=4, nnodes=2, kv_dim=1, num_kv_heads=1)
    model.freeze()
    c = object.__new__(connector.FlexKVConnector)
    c.page_size = 4
    c.model_config = model
    c.flexkv_config = SimpleNamespace(model_config=model)
    c.cache_config = CacheConfig(enable_cpu=True)
    c.rank_info = SimpleNamespace(effective_tp_rank=0, node_rank=0)
    c._is_dsv4 = False
    c._swa_kv_pool = None
    c._kvcache = None
    c._chunked_prefetch = True
    group = SimpleNamespace(size=lambda: 4)
    c._sync_ctx = SimpleNamespace(needs_sync=True, attn_cp_cpu_group=group)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "region_batch", True)
    monkeypatch.setattr(connector.GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda peers, signature, group: peers.__setitem__(slice(None), [signature] * 4),
    )

    with pytest.raises(ValueError, match="Cross-node physical CP"):
        c._resolve_cache_layer_split(SimpleNamespace(), None)


def test_eventfd_handshake_uses_node_local_rank_without_layer_split(monkeypatch):
    import socket as std_socket
    import struct
    from flexkv.integration.sglang import connector

    sent = []

    class FakeSock:
        def connect(self, addr):
            pass

        def sendall(self, data):
            sent.append(data)

        def settimeout(self, timeout):
            pass

        def recv(self, n):
            return b"\x01"

        def close(self):
            pass

    model = ModelConfig(num_layers=2, tp_size=16, nnodes=2, kv_dim=1, layer_split_nodes=1)
    model.freeze()
    c = object.__new__(connector.FlexKVConnector)
    c.model_config = model
    c.rank_info = SimpleNamespace(effective_tp_rank=9, num_layers_per_pp_stage=2)
    c.layer_done_counter = SimpleNamespace(
        num_counters=1,
        events=[SimpleNamespace(load_event_fds=[11, 12])],
    )
    c._layerwise_socket = "/tmp/fake.sock"
    c._layerwise_eventfd_connect_max_retries = 1
    c._label = "[test]"
    monkeypatch.setattr(std_socket, "socket", lambda *args: FakeSock())
    monkeypatch.setattr(connector, "send_fds", lambda sock, fds, data: None)
    monkeypatch.setattr(connector.time, "sleep", lambda s: None)

    c._send_eventfds_to_worker(retry_interval=0.0)
    tp_rank_sent, tp_per_node, num_layers, num_counters = struct.unpack("iiii", sent[0])
    assert (tp_rank_sent, tp_per_node, num_layers, num_counters) == (1, 8, 2, 1)
