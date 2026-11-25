"""Standalone FlexKV transfer benchmark for a specific model's KV layout.

    python tests/gdr/benchmark_local_kv_transfer.py --plan
    python tests/gdr/benchmark_local_kv_transfer.py --output /path/to/new/result_dir

Each experiment starts real FlexKV Connectors, a TransferManager and workers.
A synthetic KV pool and local Gloo groups replace the model and its runtime.
See EXPERIMENTS below. Full verification is outside the performance samples.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import select
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class Experiment:
    name: str
    gpu_count: int
    runtime: str
    layer_split: tuple[bool, ...]
    backends: tuple[str, ...]
    h2d_tokens: tuple[int, ...]
    h2d_requests: tuple[int, ...]
    d2h_tokens: tuple[int, ...]
    d2h_requests: tuple[int, ...]
    host_cache_gib: int
    qp_count: int
    warmups: int
    rounds: int
    scattered_slots: bool = True
    batch_preflight: bool = False
    worker_trace: bool = False
    read_depth: int = 0
    readbacks: int = 0


# Uncomment entire experiment blocks to run them. Each direction independently
# sweeps its token lengths × requests per round. `mapped` selects the CE-only
# runtime and opts the benchmark into experimental mapped H2D.
EXPERIMENTS = (
    Experiment(
        name="cp8-mapped",
        gpu_count=8, runtime="mapped", layer_split=(False, True),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        warmups=2, rounds=5,
    ),
    # Experiment(
    #     name="cp2-native-smoke",
    #     gpu_count=2, runtime="native", layer_split=(False, True),
    #     backends=("ce", "gdr"), qp_count=1, host_cache_gib=1,
    #     h2d_tokens=(128, 512), h2d_requests=(1, 3),
    #     d2h_tokens=(256, 1024), d2h_requests=(1, 3),
    #     warmups=1, rounds=2,
    # ),
    # Experiment(
    #     name="cp8-large-batches",
    #     gpu_count=8, runtime="mapped", layer_split=(True,),
    #     backends=("ce", "gdr"), qp_count=4, host_cache_gib=32,
    #     h2d_tokens=(8192, 32768), h2d_requests=(2, 8),
    #     d2h_tokens=(4096, 16384), d2h_requests=(2, 8),
    #     warmups=2, rounds=10,
    # ),
)

# Default KV layout for a specific model. Supply its config.json (absolute
# or relative to the repo root) for the exact indexer mask and dimensions.
MODEL_CONFIG_JSON = None
PAGE_SIZE = 128
NUM_LAYERS = 78
KV_LORA_RANK = 512
ROPE_DIM = 64
INDEX_HEAD_DIM = 128
GDR_NICS = None  # None inherits FLEXKV_GDR_NICS. Or set a tuple of NIC[:port].
SEED = 1907
TIMEOUT_S = 240
POLL_INTERVAL_S = 0.001
VERIFY_CHUNK_PAGES = 16

ROOT = Path(__file__).resolve().parents[2]
COUNTERS_MARKER = "per-QP payload WRs): "
PROGRESS_MARKER = "[KV-TRANSFER-BENCH]"


@dataclass(frozen=True)
class Geometry:
    layers: int
    lora: int
    rope: int
    index_dim: int
    index_layers: tuple[int, ...]

    @property
    def kv_width(self):
        return self.lora + self.lora // 128 * 4 + self.rope * 2

    @property
    def index_width(self):
        return self.index_dim + self.index_dim // 128 * 4

    @property
    def page_bytes(self):
        return PAGE_SIZE * (self.layers * self.kv_width + len(self.index_layers) * self.index_width)


def geometry():
    cfg = {}
    if MODEL_CONFIG_JSON:
        cfg = json.loads((ROOT / MODEL_CONFIG_JSON).read_text())
        cfg = cfg.get("text_config", cfg)
    layers = int(cfg.get("num_hidden_layers", NUM_LAYERS))
    kinds = cfg.get("indexer_types")
    if kinds is None:
        active = tuple(i for i in range(layers) if i == 0 or (i - 1) % 4 == 0)
    else:
        if len(kinds) != layers or set(kinds) - {"full", "shared"} or kinds[0] != "full":
            raise ValueError("Unexpected model indexer_types")
        active = tuple(i for i, kind in enumerate(kinds) if kind == "full")
    g = Geometry(layers, int(cfg.get("kv_lora_rank", KV_LORA_RANK)),
                 int(cfg.get("qk_rope_head_dim", ROPE_DIM)),
                 int(cfg.get("index_head_dim", INDEX_HEAD_DIM)), active)
    if (g.layers <= 0 or g.lora <= 0 or g.lora % 128 or g.rope <= 0
            or g.index_dim <= 0 or g.index_dim % 128 or not active
            or PAGE_SIZE * g.kv_width % 8 or PAGE_SIZE * g.index_width % 8):
        raise ValueError(f"Invalid model KV geometry: {g}")
    return g


def scenarios(exp):
    for direction, lengths, counts in (("h2d", exp.h2d_tokens, exp.h2d_requests),
                                       ("d2h", exp.d2h_tokens, exp.d2h_requests)):
        yield from ((direction, tokens, n) for tokens, n in itertools.product(lengths, counts))


def validate(exp, g):
    if not 1 <= exp.qp_count <= 16 or exp.runtime not in {"native", "mapped"}:
        raise ValueError(f"Invalid QP count or runtime in {exp.name}")
    if not exp.backends or set(exp.backends) - {"ce", "gdr"} or not exp.layer_split:
        raise ValueError(f"Invalid backend/split matrix in {exp.name}")
    if exp.read_depth < 0 or not 0 <= exp.readbacks <= 16:
        raise ValueError(f"Invalid READ/read-back limits in {exp.name}")
    if (exp.gpu_count < 1 or exp.gpu_count > g.layers
            or exp.gpu_count == 1 and True in exp.layer_split):
        raise ValueError(f"Invalid CP size in {exp.name}")
    if (PAGE_SIZE <= 0 or exp.warmups < 0 or exp.rounds < 1 or exp.host_cache_gib <= 0
            or not all((exp.h2d_tokens, exp.h2d_requests, exp.d2h_tokens, exp.d2h_requests))):
        raise ValueError(f"Invalid size or empty matrix in {exp.name}")
    matrix = list(scenarios(exp))
    if any(tokens <= 0 or tokens % PAGE_SIZE or count <= 0 for _, tokens, count in matrix):
        raise ValueError(f"{exp.name}: positive lengths must be multiples of PAGE_SIZE={PAGE_SIZE}")
    pages = max(tokens // PAGE_SIZE * count for _, tokens, count in matrix)
    if (pages + 2) * g.page_bytes > exp.host_cache_gib * 1024**3:
        raise ValueError(f"{exp.name}: Host cache cannot fit the largest round ({pages} pages)")
    return pages


def owners(layers, ranks):
    each, extra = divmod(layers, ranks)
    return tuple(rank for rank in range(ranks) for _ in range(each + (rank < extra)))


def payload_by_rank(g, split, ranks, direction, pages):
    if direction == "h2d" and not split:
        return [pages * g.page_bytes] * ranks
    kv_owners = owners(g.layers, ranks)
    index_owners = ([kv_owners[layer] for layer in g.index_layers] if split
                    else owners(len(g.index_layers), ranks))
    return [pages * PAGE_SIZE * (kv_owners.count(rank) * g.kv_width
                                + index_owners.count(rank) * g.index_width)
            for rank in range(ranks)]


class SyntheticPool:
    """Synthetic GPU pool for the configured model layout."""

    def __init__(self, torch, rank, ranks, split, g, pages):
        self.torch = torch
        self.rank, self.geometry, self.split = rank, g, split
        self.layer_num, self.start_layer = g.layers, 0
        self.layer_shard_enabled = split
        self.layer_shard_rank, self.layer_shard_size = rank, ranks
        self.holders = owners(g.layers, ranks)
        self.skip_topk_layers = [i not in g.index_layers for i in range(g.layers)]
        self._kv_shape = (pages * PAGE_SIZE, 1, g.kv_width)
        self._index_shape = (pages, PAGE_SIZE * g.index_width)
        self.kv_buffer = [self._alloc(self._kv_shape, torch.int8, i) for i in range(g.layers)]
        self.index_k_with_scale_buffer = [
            self._alloc(self._index_shape, torch.uint8, i) if i in g.index_layers else None
            for i in range(g.layers)
        ]
        self.remote_kv_buffer = self._alloc(self._kv_shape, torch.int8) if split else None
        self.remote_index_k_with_scale_buffer = self._alloc(self._index_shape, torch.uint8) if split else None
        self.scratch_invalidations = 0

    def _alloc(self, shape, dtype, layer=None):
        if self.split and layer is not None and self.holders[layer] != self.rank:
            shape = (0, *shape[1:])
        return self.torch.empty(shape, device=f"cuda:{self.rank}", dtype=dtype)

    def _get_layer_owner_rank(self, layer):
        return self.holders[layer]

    def invalidate_layer_split_cache(self):
        # The synthetic pool has no active broadcast cache or tickets.
        self.scratch_invalidations += 1

    def rows(self, layer):
        kv = self.kv_buffer[layer]
        if kv.numel():
            yield "mla", kv.view(-1, PAGE_SIZE * self.geometry.kv_width)
        index = self.index_k_with_scale_buffer[layer]
        if index is not None and index.numel():
            yield "indexer", index

    def scratch_rows(self):
        if self.split:
            yield self.remote_kv_buffer.view(-1, PAGE_SIZE * self.geometry.kv_width)
            yield self.remote_index_k_with_scale_buffer

    def bytes(self):
        return sum(t.nbytes for l in range(self.layer_num) for _, t in self.rows(l)) + sum(
            t.nbytes for t in self.scratch_rows())


@dataclass
class Slots:
    rid: str
    tokens: object
    source: object
    dest: object
    source_slots: object
    dest_slots: object
    mask: object


def round_slots(torch, np, exp, tokens, count, epoch, arena_pages):
    rng = np.random.default_rng(SEED + epoch)
    pages_per_req = tokens // PAGE_SIZE
    n = count * pages_per_req
    source = np.arange(1, 1 + arena_pages, dtype=np.int64)
    dest = source + arena_pages
    if exp.scattered_slots:
        rng.shuffle(source)
        rng.shuffle(dest)
    result = []
    offsets = np.arange(PAGE_SIZE, dtype=np.int64)
    for i in range(count):
        section = slice(i * pages_per_req, (i + 1) * pages_per_req)
        src, dst = source[section], dest[section]
        source_slots = torch.from_numpy((src[:, None] * PAGE_SIZE + offsets).reshape(-1))
        dest_slots = torch.from_numpy((dst[:, None] * PAGE_SIZE + offsets).reshape(-1))
        # Every request has a distinct first token, even for identical lengths.
        ids = np.arange(tokens, dtype=np.int64) + (i + 1) * (tokens + 1)
        result.append(Slots(f"r{epoch}-{i}", ids, src, dst, source_slots, dest_slots,
                            torch.ones(tokens, dtype=torch.bool)))
    assert n <= arena_pages
    return result


def pattern(np, width, count, start, layer, name, generation, dtype):
    # Distinguish page order, layer, region, byte offset and overwritten generation.
    rows = np.arange(start, start + count, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    return ((rows * 31 + (rows // 127) * (cols % 37 + 1)
             + cols * 17 + (cols >> 7) + layer * 13 + generation * 23
             + (71 if name == "indexer" else 0)) % 127).astype(dtype, copy=False)


def touch_pages(torch, np, pool, page_ids, generation, *, prepare=False, poison=False, layer=None):
    layers = range(pool.layer_num) if layer is None else (layer,)
    tensors = [(layer, name, buf) for layer in layers for name, buf in pool.rows(layer)]
    if poison:
        tensors.extend((-1, "scratch", buf) for buf in pool.scratch_rows())
    for lid, name, buf in tensors:
        for begin in range(0, len(page_ids), VERIFY_CHUNK_PAGES):
            pages = page_ids[begin:begin + VERIFY_CHUNK_PAGES]
            ids = torch.as_tensor(pages, device=buf.device)
            if poison:
                buf.index_fill_(0, ids, 127)
                continue
            dtype = np.int8 if buf.dtype == torch.int8 else np.uint8
            expected = pattern(np, buf.shape[1], len(pages), begin, lid, name, generation, dtype)
            if prepare:
                buf.index_copy_(0, ids, torch.from_numpy(expected).to(buf.device))
            else:
                observed = buf.index_select(0, ids).cpu().numpy()
                if not np.array_equal(observed, expected):
                    page_row, col = np.argwhere(observed != expected)[0]
                    raise AssertionError(
                        f"rank={pool.rank} layer={lid} region={name} page={pages[page_row]} "
                        f"byte={col} found={observed[page_row, col]} expected={expected[page_row, col]}")


def check_untouched(torch, pool, pages):
    for layer in range(pool.layer_num):
        for name, rows in pool.rows(layer):
            if not bool((rows[0] == 127).all().item()):
                raise AssertionError(f"rank={pool.rank} layer={layer} {name} touched guard page")
    for rows in pool.scratch_rows():
        ix = torch.as_tensor(pages, device=rows.device)
        for chunk in ix.split(VERIFY_CHUNK_PAGES):
            if not bool((rows.index_select(0, chunk) == 127).all().item()):
                raise AssertionError(f"rank={pool.rank} transfer touched receive scratch")


class WaitObserver:
    """Capture terminal statuses from the real Connector's drained responses."""

    def __init__(self, wait):
        self.wait = wait
        self.done = {}

    def __call__(self, *args, **kwargs):
        result = self.wait(*args, **kwargs)
        now = time.perf_counter_ns()
        for tid, response in (result or {}).items():
            status = response.status.value
            if status in {"success", "failed", "cancelled", "not_found"}:
                self.done.setdefault(int(tid), {"status": status, "observed_ns": now})
        return result

    def take(self, task_ids):
        rows = [self.done.pop(tid, None) for tid in task_ids]
        if any(r is None or r["status"] != "success" for r in rows):
            raise RuntimeError(f"FlexKV tasks did not finish successfully: {list(zip(task_ids, rows))}")
        return rows


def store(connector, requests, clock, observer):
    start = clock()
    submitted, completed, pending = [], {}, {r.rid for r in requests}
    for request in requests:
        before = clock()
        result = connector.store_kv_with_result(request.rid, request.tokens, request.source_slots)
        if result.task_id < 0 or not result.reads_gpu or result.already_cached:
            raise RuntimeError(f"{request.rid}: expected an actual D2H store, got {result}")
        submitted.append({"rid": request.rid, "task_id": int(result.task_id),
                          "begin_ns": before, "submitted_ns": clock()})
    submit_end = clock()
    polls = 0
    deadline = time.monotonic() + TIMEOUT_S
    while pending:
        polls += 1
        finished = connector.check_completed_stores()
        now = clock()
        for rid in finished:
            if rid not in pending:
                raise RuntimeError(f"Unexpected D2H completion: {rid}")
            completed[rid] = now
            pending.remove(rid)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"D2H did not finish: {sorted(pending)}")
            time.sleep(POLL_INTERVAL_S)
    for request in submitted:
        request["completed_ns"] = completed[request["rid"]]
    if observer is not None:
        for request, result in zip(submitted, observer.take([x["task_id"] for x in submitted])):
            request.update(result)
    return {"begin_ns": start, "submit_end_ns": submit_end, "end_ns": max(completed.values()),
            "polls": polls, "requests": submitted}


def restore(connector, requests, clock, observer, exp, verify_layer=None):
    start = clock()
    lookups = []
    keys = [r.rid for r in requests]
    for request in requests:
        began = clock()
        task_id, hit = connector.lookup_kv(request.tokens, request.mask, rid=request.rid)
        if task_id < 0 or hit != len(request.tokens):
            raise RuntimeError(f"{request.rid}: expected a complete Host hit, got {hit}")
        lookups.append({"rid": request.rid, "task_id": int(task_id), "hit_tokens": int(hit),
                        "begin_ns": began, "end_ns": clock()})
    preparation_start = clock()
    if exp.batch_preflight:
        connector.prepare_layerwise_loads([(r.rid, r.dest_slots) for r in requests])
    else:
        for request in requests:
            loaded, _ = connector.start_load_kv_layerwise(request.rid, request.dest_slots, defer_launch=True)
            if loaded != len(request.tokens):
                raise RuntimeError(f"{request.rid}: only {loaded} tokens prepared")
    launch_start = clock()
    connector.flush_layerwise_loads(keys)
    launch_end = clock()
    counter = connector.layer_done_counter
    producer = counter.consumer_index
    if producer < 0:
        raise RuntimeError("No layerwise consumer armed")
    event = counter.events[producer]
    if event.pending_transfers() != 1:
        raise RuntimeError("Merged restore must expect one eventfd signal per layer and rank")
    layers = []
    for layer in range(counter.num_layers):
        waiting = clock()
        counter.wait_until(layer)
        layers.append({"layer": layer, "wait_begin_ns": waiting, "ready_ns": clock()})
        if verify_layer is not None:
            verify_layer(layer)
    graph = None
    if observer is not None:
        task_ids = list(connector._launched_load_tids)
        if len(task_ids) != 1:
            raise RuntimeError(f"Expected one newly merged H2D task, got {task_ids}")
        deadline = time.monotonic() + TIMEOUT_S
        while connector._launched_load_tids:
            connector.drain_launched_loads()
            if connector._launched_load_tids:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"H2D graph did not drain: {task_ids}")
                time.sleep(POLL_INTERVAL_S)
        graph = {"task_id": int(task_ids[0]), **observer.take(task_ids)[0]}
    return {"begin_ns": start, "prepare_begin_ns": preparation_start,
            "launch_begin_ns": launch_start, "launch_end_ns": launch_end,
            "end_ns": layers[-1]["ready_ns"], "counter_id": producer, "layers": layers,
            "lookups": lookups, "graph": graph}, event


def require_empty_eventfds(event):
    poller = select.poll()
    for fd in event.load_event_fds:
        poller.register(fd, select.POLLIN)
    if poller.poll(0):
        raise RuntimeError("Worker emitted an extra layer notification after the graph drained")


class VerificationError(RuntimeError):
    """The round drained but the data or eventfd contract failed validation."""


def capture_round(dist, rank, ranks, output, g, split, direction, tokens, count,
                  phase, iteration, parts):
    rows = [None] * ranks if rank == 0 else None
    dist.gather_object(parts, rows, dst=0)
    if rank != 0:
        return
    start = min(row["begin_ns"] for row in rows)
    end = max(row["end_ns"] for row in rows)
    seconds = (end - start) / 1e9
    by_rank = payload_by_rank(g, split, ranks, direction, tokens // PAGE_SIZE * count)
    data = {"direction": direction, "tokens_per_request": tokens, "requests": count,
            "phase": phase, "iteration": iteration, "total_tokens": tokens * count,
            "logical_payload_bytes": g.page_bytes * (tokens // PAGE_SIZE * count),
            "payload_bytes_by_rank": by_rank, "payload_bytes": sum(by_rank),
            "begin_ns": start, "end_ns": end, "elapsed_ms": seconds * 1000,
            "payload_gib_s": sum(by_rank) / 1024**3 / seconds,
            "tokens_per_s": tokens * count / seconds, "ranks": rows}
    if direction == "h2d":
        layer_ms = [(max(r["layers"][layer]["ready_ns"] for r in rows) - start) / 1e6
                    for layer in range(g.layers)]
        data.update(all_ranks_layer_ready_ms=layer_ms, first_layer_ms=layer_ms[0],
                    last_layer_ms=layer_ms[-1],
                    graph_observed_ms=(rows[0]["graph"]["observed_ns"] - start) / 1e6,
                    lookup_ms=(max(r["prepare_begin_ns"] for r in rows) - start) / 1e6,
                    prepare_ms=(rows[0]["launch_begin_ns"] - rows[0]["prepare_begin_ns"]) / 1e6,
                    launch_ms=(rows[0]["launch_end_ns"] - rows[0]["launch_begin_ns"]) / 1e6)
    else:
        data.update(submit_window_ms=(max(r["submit_end_ns"] for r in rows) - start) / 1e6,
                    request_latency_ms=[(request["completed_ns"] - request["begin_ns"]) / 1e6
                                        for request in rows[0]["requests"]])
    output.write(json.dumps(data) + "\n")
    output.flush()
    if phase == "measure":
        print(f"{PROGRESS_MARKER} {direction} tokens={tokens} requests={count} round={iteration + 1} "
              f"time={data['elapsed_ms']:.3f} ms payload={data['payload_gib_s']:.3f} GiB/s", flush=True)


def run_rank(rank, exp, split, backend, g, arena_pages, rendezvous, arm_dir):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import torch.distributed as dist
    from flexkv import c_ext
    from flexkv.integration.sglang import connector as adapter
    from flexkv.integration.sglang.comm import FlexKVComm

    if backend == "gdr" and getattr(c_ext, "gdr_uses_mapped_memory", None) != (exp.runtime == "mapped"):
        raise RuntimeError(f"Rebuild the local c_ext for GDR runtime={exp.runtime}")
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    ranks = exp.gpu_count
    clock = time.perf_counter_ns
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=ranks, timeout=timedelta(seconds=TIMEOUT_S + 60))
    gpu_pages = 2 * arena_pages + 1
    pool = SyntheticPool(torch, rank, ranks, split, g, gpu_pages)
    sgl_model = SimpleNamespace(num_hidden_layers=g.layers, kv_lora_rank=g.lora,
                                qk_nope_head_dim=128, qk_rope_head_dim=g.rope,
                                index_head_dim=g.index_dim, dtype=torch.int8)
    args = SimpleNamespace(tp_size=ranks, pp_size=1, dp_size=1, nnodes=1, node_rank=0,
                           attn_cp_size=ranks, enable_dp_attention=False, kv_cache_dtype=None,
                           enable_nsa_cache_layer_split=split, model_path="synthetic-target-model")
    # Supply the actual Gloo world group at the framework hook so FlexKVComm
    # neither imports nor calls SGLang. All collectives below run FlexKV code.
    original_comm = adapter.FlexKVComm
    adapter.FlexKVComm = lambda **kw: FlexKVComm(**kw, world_group=dist.group.WORLD)
    try:
        connector = adapter.FlexKVConnector(
            sgl_model_config=sgl_model, server_args=args, page_size=PAGE_SIZE,
            kvcache=pool, tp_rank=rank, dp_rank=0, pp_rank=0, attn_cp_rank=rank,
            attn_cp_group=dist.group.WORLD)
    finally:
        adapter.FlexKVComm = original_comm

    manager = connector.kv_manager
    if connector._cache_layer_split != split:
        raise RuntimeError("Connector did not use the requested physical layer split")
    if connector.cache_config.num_cpu_blocks < arena_pages + 1:
        raise RuntimeError("Resolved Host cache is too small for the largest round")
    owner = getattr(connector.tp_client, "_local_gdr_owner", None)
    if backend == "gdr" and owner is None:
        raise RuntimeError("Connector did not establish a local GDR owner")
    observer = WaitObserver(manager.wait) if manager is not None else None
    if observer is not None:
        manager.wait = observer
    connector.layer_done_counter.wait_timeout_s = TIMEOUT_S

    device_info = torch.cuda.get_device_properties(rank)
    meta = {"rank": rank, "device": device_info.name, "gpu_total_bytes": device_info.total_memory,
            "gpu_pool_bytes": pool.bytes(), "gpu_pages": gpu_pages,
            "host_pages": connector.cache_config.num_cpu_blocks,
            "layer_split": connector._cache_layer_split,
            "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
            "extension_path": str(c_ext.__file__),
            "extension_commit": getattr(c_ext, "__git_commit__", None),
            "gdr_port": owner.endpoint.nic if owner is not None else None,
            "gdr_read_depth": owner.endpoint.read_depth if owner is not None else None,
            "owned_layers": [layer for layer in range(g.layers) if pool.holders[layer] == rank]
                            if split else list(range(g.layers))}
    metadata = [None] * ranks if rank == 0 else None
    dist.gather_object(meta, metadata, dst=0)
    if rank == 0:
        (arm_dir / "ranks.json").write_text(json.dumps(metadata, indent=2))

    def reset_host():
        if manager is not None:
            manager.reset()
        dist.barrier()

    def run_store(requests, tokens, count, phase, iteration):
        dist.barrier()
        row = store(connector, requests, clock, observer)
        capture_round(dist, rank, ranks, output, g, split, "d2h", tokens, count, phase, iteration, row)

    def run_restore(requests, tokens, count, phase, iteration, generation, verify):
        pages = np.concatenate([r.dest for r in requests])
        touch_pages(torch, np, pool, np.r_[0, pages], generation, poison=True)
        torch.cuda.synchronize(rank)
        dist.barrier()
        errors = []
        checker = None
        if verify:
            def checker(layer):
                if not errors:
                    try:
                        touch_pages(torch, np, pool, pages, generation, layer=layer)
                    except AssertionError as exc:
                        # Consume the remaining signals and drain the graph
                        # before propagating a content mismatch across ranks.
                        errors.append(str(exc))
        row, event = restore(connector, requests, clock, observer, exp, checker)
        # The collective happens after the leader observes graph completion.
        # That establishes quiescence before checking for extra tokens.
        capture_round(dist, rank, ranks, output, g, split, "h2d", tokens, count, phase, iteration, row)
        dist.barrier()
        try:
            require_empty_eventfds(event)
            if verify and not errors:
                check_untouched(torch, pool, pages)
        except (RuntimeError, AssertionError) as exc:
            errors.append(str(exc))
        outcomes = [None] * ranks
        dist.all_gather_object(outcomes, errors)
        if any(outcomes):
            raise VerificationError(f"{phase}: layerwise verification failed: {outcomes}")

    def shutdown_drained():
        dist.barrier()
        # Stop both Worker directions before followers can drop their GPU MRs.
        if rank == 0:
            connector.shutdown()
        dist.barrier()
        if rank != 0:
            connector.shutdown()

    # Only the leader writes round records. Keep completed rows on disk if a
    # later case fails. The resident Host/GPU pools are reused for each arm.
    output = (arm_dir / "rounds.jsonl").open("w") if rank == 0 else None
    generation = 0
    try:
        for direction, tokens, count in scenarios(exp):
            if direction == "h2d":
                reset_host()
                generation += 1
                requests = round_slots(torch, np, exp, tokens, count, generation, arena_pages)
                touch_pages(torch, np, pool, np.concatenate([r.source for r in requests]),
                            generation, prepare=True)
                torch.cuda.synchronize(rank)
                run_store(requests, tokens, count, "seed", 0)
            phases = [("verify", 0)] + [("warmup", i) for i in range(exp.warmups)] + [
                ("measure", i) for i in range(exp.rounds)]
            for phase, iteration in phases:
                if direction == "h2d":
                    run_restore(requests, tokens, count, phase, iteration, generation, phase == "verify")
                    continue
                # Fresh Host keys and the same GPU arena size for every D2H sample.
                reset_host()
                generation += 1
                requests = round_slots(torch, np, exp, tokens, count, generation, arena_pages)
                touch_pages(torch, np, pool, np.concatenate([r.source for r in requests]),
                            generation, prepare=True)
                torch.cuda.synchronize(rank)
                run_store(requests, tokens, count, phase, iteration)
                if phase == "verify":
                    run_restore(requests, tokens, count, "verify_d2h", iteration, generation, True)
        reset_host()
        shutdown_drained()
    except VerificationError:
        shutdown_drained()
        raise
    finally:
        if output is not None:
            output.close()
        dist.destroy_process_group()


def arm_env(exp, backend, scratch):
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ENABLE_FLEXKV": "1", "FLEXKV_CONFIG_PATH": str(scratch / "config.json"),
        "FLEXKV_SERVER_CLIENT_MODE": "0", "FLEXKV_SERVER_LAUNCH_MODE": "embedded",
        "FLEXKV_SERVER_RECV_PORT": f"ipc://{scratch}/main",
        "FLEXKV_LAYERWISE_EVENTFD_SOCKET": str(scratch / "layers"),
        "FLEXKV_ENABLE_LAYERWISE_TRANSFER": "1", "FLEXKV_REGION_BATCH": "1",
        "FLEXKV_ENABLE_COLLECTIVE_SYNC": "1", "FLEXKV_DEDUP_INDEXER_GROUP": "1",
        "FLEXKV_CPU_LAYOUT": "BLOCKFIRST", "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE": "layer_parallel",
        "FLEXKV_ENABLE_MPS": "0", "FLEXKV_ENABLE_RADIXSHMEM": "0", "FLEXKV_CACHE_DRAFT_KV": "0",
        "FLEXKV_ENABLE_GDS": "0", "FLEXKV_ENABLE_NVCOMP": "0",
        "FLEXKV_USE_CE_TRANSFER_H2D": "1", "FLEXKV_USE_CE_TRANSFER_D2H": "1",
        "FLEXKV_ENABLE_CE_MEMCPY2D": "0", "FLEXKV_LAYERWISE_NOTIFY_MODE": "polling",
        "FLEXKV_ENABLE_CE_ONLY": str(int(exp.runtime == "mapped")),
        "FLEXKV_LOCAL_GDR": "both" if backend == "gdr" else "off",
        "FLEXKV_GDR_EXPERIMENTAL_MAPPED_H2D": str(int(exp.runtime == "mapped")),
        "FLEXKV_GDR_QP_COUNT": str(exp.qp_count), "FLEXKV_GDR_LOG_COUNTERS": "1",
        "FLEXKV_GDR_READ_DEPTH": str(exp.read_depth), "FLEXKV_GDR_READBACKS": str(exp.readbacks),
        "FLEXKV_LOG_LEVEL": "INFO", "FLEXKV_TRANSFER_TRACE": str(int(exp.worker_trace)),
    })
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    if GDR_NICS is not None:
        env["FLEXKV_GDR_NICS"] = ",".join(GDR_NICS)
    # These allocator settings are needed by the real cross-process IPC export.
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if "expandable_segments:true" in env.get(key, "").replace(" ", "").lower():
            raise ValueError(f"Disable expandable_segments in {key} for this CUDA IPC benchmark")
    return env


def read_route_counters(lines, backend, exp, rounds):
    groups = [ast.literal_eval(line.split(COUNTERS_MARKER, 1)[1].strip())
              for line in lines if COUNTERS_MARKER in line]
    if backend == "ce":
        if groups:
            raise RuntimeError("Unexpected GDR counters in CE arm")
        return None
    if len(groups) != 2 or any(len(group) != exp.gpu_count for group in groups):
        raise RuntimeError("Expected one complete GDR counter group per direction at shutdown")
    result = {}
    for index, direction in enumerate(("h2d", "d2h")):
        candidates = [group for group in groups if sum(row[index] for row in group) > 0]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot identify GDR counters for {direction}")
        group = candidates[0]
        if any(len(row) != 6 + exp.qp_count for row in group):
            raise RuntimeError(f"Unexpected QP counter shape for {direction}")
        measured = [row[index] for row in group]
        expected = [sum(row["payload_bytes_by_rank"][rank] for row in rounds
                        if row["direction"] == direction) for rank in range(exp.gpu_count)]
        if measured != expected:
            raise RuntimeError(f"{direction}: GDR rank payload bytes {measured} != expected {expected}")
        result[direction] = [dict(rank=rank, h2d_bytes=row[0], d2h_bytes=row[1], payload_wrs=row[2],
                                  posts=row[3], cqes=row[4], readbacks=row[5], qp_payload_wrs=row[6:])
                             for rank, row in enumerate(group)]
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {"min": min(values), "mean": statistics.fmean(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values)}


def summarize(rounds):
    buckets = {}
    for row in rounds:
        if row["phase"] == "measure":
            key = (row["direction"], row["tokens_per_request"], row["requests"])
            buckets.setdefault(key, []).append(row)
    summaries = []
    for (direction, tokens, count), rows in buckets.items():
        keys = ["elapsed_ms", "payload_gib_s", "tokens_per_s"]
        extras = (["first_layer_ms", "last_layer_ms", "graph_observed_ms", "lookup_ms", "prepare_ms", "launch_ms"]
                  if direction == "h2d" else ["submit_window_ms"])
        totals = sum(r["elapsed_ms"] for r in rows) / 1000
        summary = {"direction": direction, "tokens_per_request": tokens, "requests": count,
                   "rounds": len(rows), "payload_bytes_per_round": rows[0]["payload_bytes"],
                   "weighted_payload_gib_s": sum(r["payload_bytes"] for r in rows) / 1024**3 / totals,
                   "metrics": {key: distribution([r[key] for r in rows]) for key in keys + extras}}
        if direction == "h2d":
            summary["layer_ready_ms"] = [distribution(list(times)) for times in
                                         zip(*(r["all_ranks_layer_ready_ms"] for r in rows))]
        else:
            summary["request_latency_ms"] = distribution([ms for r in rows for ms in r["request_latency_ms"]])
        summaries.append(summary)
    return summaries


def run_arm(exp, split, backend, g, arm_dir):
    arm_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="kv-transfer-", dir="/tmp") as directory:
        scratch = Path(directory)
        (scratch / "config.json").write_text(json.dumps({
            "cpu_cache_gb": exp.host_cache_gib,
            "ssd_cache_gb": 0,
        }))
        env = arm_env(exp, backend, scratch)
        command = [sys.executable, str(Path(__file__).resolve()), "--run-arm", exp.name,
                   backend, str(int(split)), str(arm_dir), str(scratch / "gloo")]
        # Worker stdout/stderr are drained throughout the run and preserved.
        log_path = arm_dir / "runner.log"
        with log_path.open("w") as log, subprocess.Popen(command, env=env, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line)
                if line.startswith(PROGRESS_MARKER):
                    print(line.rstrip(), flush=True)
            if proc.wait() != 0:
                raise RuntimeError(f"{exp.name}/{backend}/split={split} failed. See {log_path}")
    rounds = [json.loads(line) for line in (arm_dir / "rounds.jsonl").read_text().splitlines()]
    measured = [(r["direction"], r["tokens_per_request"], r["requests"])
                for r in rounds if r["phase"] == "measure"]
    if sorted(measured) != sorted(list(scenarios(exp)) * exp.rounds):
        raise RuntimeError("Measured scenario matrix is incomplete")
    lines = log_path.read_text(errors="replace").splitlines()
    return {"experiment": exp.name, "backend": backend, "layer_split": split,
            "configuration": asdict(exp), "geometry": asdict(g),
            "rank_metadata": json.loads((arm_dir / "ranks.json").read_text()),
            "round_file": str(arm_dir / "rounds.jsonl"), "log_file": str(log_path),
            "payload_basis": "per-rank GDR counters reconciled with all rounds" if backend == "gdr"
                             else "region geometry and confirmed logical transfer lengths",
            "gdr_counters": read_route_counters(lines, backend, exp, rounds),
            "worker_trace": [line for line in lines if "[XFER] op=" in line],
            "summary": summarize(rounds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="validate and print the matrix without GPU imports")
    parser.add_argument("--output", type=Path, default=Path(time.strftime("kv-transfer-%Y%m%d-%H%M%S")))
    parser.add_argument("--run-arm", nargs=5, help=argparse.SUPPRESS)
    args = parser.parse_args()
    g = geometry()
    names = {exp.name: exp for exp in EXPERIMENTS}
    if not names or len(names) != len(EXPERIMENTS):
        raise ValueError("Configure at least one experiment with a unique name")
    arenas = {name: validate(exp, g) for name, exp in names.items()}
    if args.run_arm:
        name, backend, split, arm_dir, rendezvous = args.run_arm
        exp = names[name]
        import torch
        if torch.cuda.device_count() < exp.gpu_count:
            raise RuntimeError(f"{exp.name} requires {exp.gpu_count} visible GPUs")
        torch.multiprocessing.spawn(run_rank, nprocs=exp.gpu_count,
                                    args=(exp, split == "1", backend, g, arenas[name], rendezvous, Path(arm_dir)))
        return
    if args.plan:
        print(json.dumps({"page_tokens": PAGE_SIZE, "geometry": asdict(g),
                          "logical_bytes_per_page": g.page_bytes,
                          "experiments": [{**asdict(exp), "arena_pages_per_direction": arenas[exp.name],
                                           "cases": list(scenarios(exp))} for exp in EXPERIMENTS]}, indent=2))
        return
    if sys.platform != "linux":
        parser.error("This benchmark requires Linux, GPU CUDA IPC and real RDMA for the GDR arm")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"page_tokens": PAGE_SIZE, "seed": SEED, "host_poll_interval_s": POLL_INTERVAL_S,
              "time_scope": "first Connector entry to observed data readiness or drained D2H completion",
              "sample_scope": "local Host L2, synthetic target KV and indexer, no model compute",
              "arms": []}
    report_path = output / "report.json"
    for exp in EXPERIMENTS:
        for split, backend in itertools.product(exp.layer_split, exp.backends):
            print(f"{PROGRESS_MARKER} starting {exp.name} backend={backend} split={int(split)}", flush=True)
            row = run_arm(exp, split, backend, g, output / f"{exp.name}-{backend}-split{int(split)}")
            report["arms"].append(row)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{PROGRESS_MARKER} report {report_path}", flush=True)


if __name__ == "__main__":
    main()
