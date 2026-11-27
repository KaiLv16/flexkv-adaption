"""Standalone FlexKV transfer benchmark for a specific model's KV layout.

    python tests/gdr/benchmark_local_kv_transfer.py --plan
    python tests/gdr/benchmark_local_kv_transfer.py --output /path/to/new/result_dir

Each experiment starts real FlexKV Connectors, a TransferManager and workers.
A synthetic KV pool and local Gloo groups replace the model and its runtime.
See EXPERIMENTS below. Byte verification uses CPU readback outside the
performance samples. See test_local_gdr_visibility.py for kernel visibility.
"""

from __future__ import annotations

import argparse
import ast
import faulthandler
import gc
import itertools
import json
import os
import select
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
import re
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
    batch_lookup: bool = False
    batch_store: bool = False
    layerwise: bool = True
    bidirectional: bool = False
    cp_size: int | None = None
    worker_trace: bool = False
    gdr_profile: bool = False
    inline_control_bytes: int = 0
    compute_overlap: bool = False
    read_depth: int = 0
    readbacks: int = 0
    h2d_qp_count: int = 0
    d2h_qp_count: int = 0
    max_sge: int = 0
    wr_api: int = 1


# Full GDR & CE benchmark matrix. `runtime="auto"` matches the compiled c_ext
# (`mapped` or `native`), while explicit `mapped`/`native` arms skip cleanly if
# they do not match the current build.
EXPERIMENTS = (
    Experiment(
        name="cp8-mapped",
        gpu_count=8, runtime="mapped", layer_split=(False, True),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp2-native-smoke",
        gpu_count=2, runtime="native", layer_split=(False, True),
        backends=("ce", "gdr"), qp_count=1, host_cache_gib=1,
        h2d_tokens=(128, 512), h2d_requests=(1, 3),
        d2h_tokens=(256, 1024), d2h_requests=(1, 3),
        warmups=1, rounds=2,
    ),
    Experiment(
        name="cp8-large-batches",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=32,
        h2d_tokens=(8192, 32768), h2d_requests=(2, 8),
        d2h_tokens=(4096, 16384), d2h_requests=(2, 8),
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-compute-overlap",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        compute_overlap=True, warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-contiguous",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        scattered_slots=False, warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-ctrl-batch",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("gdr",), qp_count=4, host_cache_gib=4,
        h2d_tokens=(128, 1024, 4096), h2d_requests=(1, 4, 8),
        d2h_tokens=(128, 1024), d2h_requests=(1, 4),
        batch_preflight=True, batch_lookup=True, batch_store=True,
        inline_control_bytes=1024,
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-d2h-batch",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("gdr",), qp_count=8, host_cache_gib=32,
        h2d_tokens=(4096, 32768), h2d_requests=(4, 8),
        d2h_tokens=(4096, 16384), d2h_requests=(4, 8),
        scattered_slots=False, batch_preflight=True, batch_lookup=True,
        batch_store=True, inline_control_bytes=1024,
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-whole-sync",
        gpu_count=8, runtime="auto", layer_split=(False, True),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024,), d2h_requests=(1,),
        layerwise=False, warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-bidir",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=16,
        h2d_tokens=(4096,), h2d_requests=(4,),
        d2h_tokens=(4096,), d2h_requests=(4,),
        bidirectional=True, warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-qp1",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("gdr",), qp_count=1, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-qp8",
        gpu_count=8, runtime="auto", layer_split=(True,),
        backends=("gdr",), qp_count=8, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        warmups=2, rounds=5,
    ),
    Experiment(
        name="cp8-readback-tune",
        gpu_count=8, runtime="mapped", layer_split=(True,),
        backends=("gdr",), qp_count=4, host_cache_gib=4,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024,), d2h_requests=(1,),
        read_depth=0, readbacks=16, warmups=2, rounds=5,
    ),
    Experiment(
        name="tp8-nocp",
        gpu_count=8, runtime="auto", layer_split=(False,),
        backends=("ce", "gdr"), qp_count=4, host_cache_gib=16,
        h2d_tokens=(1024, 4096), h2d_requests=(1, 4),
        d2h_tokens=(1024, 4096), d2h_requests=(1, 4),
        cp_size=1, warmups=2, rounds=5,
    ),
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
PROFILE_MARKER = "D2H[same]): "
OWNER_PROFILE_RE = re.compile(
    r"Local GDR owner profile \(device=(\d+), R_calls=(\d+), R_sync_ms=([\d.]+), "
    r"R_total_ms=([\d.]+), V_calls=(\d+), V_sync_ms=([\d.]+), V_total_ms=([\d.]+)\)"
)
XFER_TRACE_RE = re.compile(
    r"\[XFER\] op=(-?\d+) type=(\S+) nb=(\d+) bytes=(\d+) h2d=(\d+) kv_dim=(\d+) "
    r"ipc=([\d.]+)ms wait=([\d.]+)ms xfer=([\d.]+)ms e2e=([\d.]+)ms bw=([\d.]+)MB/s "
    r"worker=(\d+) graph=(\d+) backlog=(\d+)"
)
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
    if exp.bidirectional:
        yield from (("bidir", tokens, n) for tokens, n in itertools.product(exp.h2d_tokens, exp.h2d_requests))
        return
    for direction, lengths, counts in (("h2d", exp.h2d_tokens, exp.h2d_requests),
                                       ("d2h", exp.d2h_tokens, exp.d2h_requests)):
        yield from ((direction, tokens, n) for tokens, n in itertools.product(lengths, counts))


def validate(exp, g):
    if not 1 <= exp.qp_count <= 16 or exp.runtime not in {"auto", "native", "mapped"}:
        raise ValueError(f"Invalid QP count or runtime in {exp.name}")
    if not (0 <= exp.h2d_qp_count <= exp.qp_count and 0 <= exp.d2h_qp_count <= exp.qp_count):
        raise ValueError(f"Invalid directional QP counts in {exp.name}")
    if not 0 <= exp.max_sge <= 16 or exp.wr_api not in (0, 1):
        raise ValueError(f"Invalid max_sge or wr_api in {exp.name}")
    if not exp.backends or set(exp.backends) - {"ce", "gdr"} or not exp.layer_split:
        raise ValueError(f"Invalid backend/split matrix in {exp.name}")
    if exp.read_depth < 0 or not 0 <= exp.readbacks <= 16:
        raise ValueError(f"Invalid READ/read-back limits in {exp.name}")
    if exp.inline_control_bytes not in (0, 256, 512, 1024, 4096):
        raise ValueError(f"Invalid inline_control_bytes in {exp.name}")
    effective_cp = exp.gpu_count if exp.cp_size is None else exp.cp_size
    if (exp.gpu_count < 1 or exp.gpu_count > g.layers
            or effective_cp < 1 or exp.gpu_count % effective_cp != 0
            or (effective_cp == 1 and True in exp.layer_split)):
        raise ValueError(f"Invalid TP/CP size in {exp.name}")
    effective_tp = exp.gpu_count // effective_cp
    if exp.bidirectional and not exp.layerwise:
        raise ValueError(f"{exp.name}: bidirectional requires layerwise=True")
    if (PAGE_SIZE <= 0 or exp.warmups < 0 or exp.rounds < 1 or exp.host_cache_gib <= 0
            or not all((exp.h2d_tokens, exp.h2d_requests, exp.d2h_tokens, exp.d2h_requests))):
        raise ValueError(f"Invalid size or empty matrix in {exp.name}")
    if exp.bidirectional and (exp.h2d_tokens != exp.d2h_tokens or exp.h2d_requests != exp.d2h_requests):
        raise ValueError(f"{exp.name}: bidirectional requires identical H2D/D2H token and request grids")
    matrix = list(scenarios(exp))
    if any(tokens <= 0 or tokens % PAGE_SIZE or count <= 0 for _, tokens, count in matrix):
        raise ValueError(f"{exp.name}: positive lengths must be multiples of PAGE_SIZE={PAGE_SIZE}")
    pages = max(tokens // PAGE_SIZE * count for _, tokens, count in matrix)
    if exp.bidirectional:
        pages *= 2
    if (pages + 2) * g.page_bytes * effective_tp > exp.host_cache_gib * 1024**3:
        raise ValueError(f"{exp.name}: Host cache cannot fit the largest round ({pages} pages, tp={effective_tp})")
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


def round_slots(torch, np, exp, tokens, count, epoch, arena_pages, *,
                source_offset=0, dest_offset=None, window_pages=None, token_salt=0, rid_prefix="r"):
    rng = np.random.default_rng(SEED + epoch)
    pages_per_req = tokens // PAGE_SIZE
    n = count * pages_per_req
    span = arena_pages if window_pages is None else window_pages
    if dest_offset is None:
        dest_offset = arena_pages
    source = np.arange(1 + source_offset, 1 + source_offset + span, dtype=np.int64)
    dest = np.arange(1 + dest_offset, 1 + dest_offset + span, dtype=np.int64)
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
        ids = token_salt + np.arange(tokens, dtype=np.int64) + (i + 1) * (tokens + 1)
        result.append(Slots(f"{rid_prefix}{epoch}-{i}", ids, src, dst, source_slots, dest_slots,
                            torch.ones(tokens, dtype=torch.bool)))
    assert n <= span
    return result


def page_runs(np, page_ids):
    # Sort physical pages for contiguous copies while keeping their positions
    # in the original request order for pattern generation.
    order = np.argsort(page_ids)
    physical = page_ids[order]
    boundaries = np.flatnonzero(np.diff(physical) != 1) + 1
    for run in np.split(order, boundaries):
        for begin in range(0, len(run), VERIFY_CHUNK_PAGES):
            positions = run[begin:begin + VERIFY_CHUNK_PAGES]
            yield int(page_ids[positions[0]]), positions


def pattern(np, width, positions, layer, name, generation, dtype):
    # Distinguish page order, layer, region, byte offset and overwritten generation.
    rows = positions[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    return ((rows * 31 + (rows // 127) * (cols % 37 + 1)
             + cols * 17 + (cols >> 7) + layer * 13 + generation * 23
             + (71 if name == "indexer" else 0)) % 127).astype(dtype, copy=False)


def touch_pages(torch, np, pool, page_ids, generation, *, prepare=False, poison=False, layer=None):
    runs = list(page_runs(np, page_ids))
    layers = range(pool.layer_num) if layer is None else (layer,)
    tensors = [(layer, name, buf) for layer in layers for name, buf in pool.rows(layer)]
    if poison:
        tensors.extend((-1, "scratch", buf) for buf in pool.scratch_rows())
    for lid, name, buf in tensors:
        dtype = np.int8 if buf.dtype == torch.int8 else np.uint8
        for first, positions in runs:
            rows = buf[first:first + len(positions)]
            if poison:
                rows.copy_(torch.from_numpy(np.full(rows.shape, 127, dtype=dtype)))
                continue
            expected = pattern(np, buf.shape[1], positions, lid, name, generation, dtype)
            if prepare:
                rows.copy_(torch.from_numpy(expected))
            else:
                observed = rows.cpu().numpy()
                if not np.array_equal(observed, expected):
                    page_row, col = np.argwhere(observed != expected)[0]
                    raise AssertionError(
                        f"rank={pool.rank} layer={lid} region={name} page={first + page_row} "
                        f"byte={col} found={observed[page_row, col]} expected={expected[page_row, col]}")


def check_untouched(np, pool, pages):
    for layer in range(pool.layer_num):
        for name, rows in pool.rows(layer):
            if not np.all(rows[0].cpu().numpy() == 127):
                raise AssertionError(f"rank={pool.rank} layer={layer} {name} touched guard page")
    runs = list(page_runs(np, pages))
    for rows in pool.scratch_rows():
        for first, positions in runs:
            if not np.all(rows[first:first + len(positions)].cpu().numpy() == 127):
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


def store(connector, requests, clock, observer, exp):
    start = clock()
    submitted, completed, pending = [], {}, {r.rid for r in requests}
    if exp.batch_store and hasattr(connector, "store_kv_batch_with_result"):
        before = clock()
        batch_results = connector.store_kv_batch_with_result(
            [(r.rid, r.tokens, r.source_slots) for r in requests]
        )
        submitted_ns = clock()
        for request, result in zip(requests, batch_results):
            if result.task_id < 0 or not result.reads_gpu or result.already_cached:
                raise RuntimeError(f"{request.rid}: expected an actual D2H store, got {result}")
            submitted.append({"rid": request.rid, "task_id": int(result.task_id),
                              "begin_ns": before, "submitted_ns": submitted_ns})
    else:
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


def _lookup_requests(connector, requests, clock, exp):
    lookups = []
    if exp.batch_lookup:
        began = clock()
        batch_hits = connector.lookup_kv_batch([
            (r.rid, lambda toks=r.tokens: toks, 0, len(r.tokens), r.rid, None)
            for r in requests
        ])
        ended = clock()
        for request, (task_id, hit) in zip(requests, batch_hits):
            if task_id < 0 or hit != len(request.tokens):
                raise RuntimeError(f"{request.rid}: expected a complete Host hit, got {hit}")
            lookups.append({"rid": request.rid, "task_id": int(task_id), "hit_tokens": int(hit),
                            "begin_ns": began, "end_ns": ended})
        return lookups
    for request in requests:
        began = clock()
        task_id, hit = connector.lookup_kv(request.tokens, request.mask, rid=request.rid)
        if task_id < 0 or hit != len(request.tokens):
            raise RuntimeError(f"{request.rid}: expected a complete Host hit, got {hit}")
        lookups.append({"rid": request.rid, "task_id": int(task_id), "hit_tokens": int(hit),
                        "begin_ns": began, "end_ns": clock()})
    return lookups


def restore(connector, requests, clock, observer, exp, verify_layer=None):
    start = clock()
    lookups = _lookup_requests(connector, requests, clock, exp)
    keys = [r.rid for r in requests]
    if not exp.layerwise:
        launch_start = clock()
        for request in requests:
            loaded = connector.retrieve_kv(request.rid, request.dest_slots)
            if loaded != len(request.tokens):
                raise RuntimeError(f"{request.rid}: only {loaded} tokens restored")
        end_ns = clock()
        if observer is not None:
            observer.take([item["task_id"] for item in lookups])
        return {"begin_ns": start, "prepare_begin_ns": launch_start,
                "launch_begin_ns": launch_start, "launch_end_ns": end_ns,
                "end_ns": end_ns, "counter_id": -1, "layers": [],
                "lookups": lookups, "graph": None}, None
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


def bidir_transfer(connector, h2d_requests, d2h_requests, clock, observer, exp):
    start = clock()
    submitted, completed, pending = [], {}, {r.rid for r in d2h_requests}
    if exp.batch_store and hasattr(connector, "store_kv_batch_with_result"):
        before = clock()
        batch_results = connector.store_kv_batch_with_result(
            [(r.rid, r.tokens, r.source_slots) for r in d2h_requests]
        )
        submitted_ns = clock()
        for request, result in zip(d2h_requests, batch_results):
            if result.task_id < 0 or not result.reads_gpu or result.already_cached:
                raise RuntimeError(f"{request.rid}: expected an actual D2H store, got {result}")
            submitted.append({"rid": request.rid, "task_id": int(result.task_id),
                              "begin_ns": before, "submitted_ns": submitted_ns})
    else:
        for request in d2h_requests:
            before = clock()
            result = connector.store_kv_with_result(request.rid, request.tokens, request.source_slots)
            if result.task_id < 0 or not result.reads_gpu or result.already_cached:
                raise RuntimeError(f"{request.rid}: expected an actual D2H store, got {result}")
            submitted.append({"rid": request.rid, "task_id": int(result.task_id),
                              "begin_ns": before, "submitted_ns": clock()})
    d2h_submit_end = clock()

    h2d_begin = clock()
    lookups = _lookup_requests(connector, h2d_requests, clock, exp)
    h2d_keys = [r.rid for r in h2d_requests]
    preparation_start = clock()
    if exp.batch_preflight:
        connector.prepare_layerwise_loads([(r.rid, r.dest_slots) for r in h2d_requests])
    else:
        for request in h2d_requests:
            loaded, _ = connector.start_load_kv_layerwise(request.rid, request.dest_slots, defer_launch=True)
            if loaded != len(request.tokens):
                raise RuntimeError(f"{request.rid}: only {loaded} tokens prepared")
    launch_start = clock()
    connector.flush_layerwise_loads(h2d_keys)
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
    h2d_end = layers[-1]["ready_ns"]

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
                raise TimeoutError(f"Bidir D2H did not finish: {sorted(pending)}")
            time.sleep(POLL_INTERVAL_S)
    for request in submitted:
        request["completed_ns"] = completed[request["rid"]]
    d2h_end = max(completed.values())
    end_ns = max(h2d_end, d2h_end)

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
        for request, result in zip(submitted, observer.take([x["task_id"] for x in submitted])):
            request.update(result)
    return {
        "begin_ns": start, "end_ns": end_ns,
        "h2d_begin_ns": h2d_begin, "h2d_end_ns": h2d_end,
        "prepare_begin_ns": preparation_start,
        "launch_begin_ns": launch_start, "launch_end_ns": launch_end,
        "counter_id": producer, "layers": layers, "lookups": lookups, "graph": graph,
        "d2h_begin_ns": submitted[0]["begin_ns"], "submit_end_ns": d2h_submit_end,
        "d2h_end_ns": d2h_end, "polls": polls, "requests": submitted,
    }, event


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
    pages = tokens // PAGE_SIZE * count
    if direction == "bidir":
        h2d_by_rank = payload_by_rank(g, split, ranks, "h2d", pages)
        d2h_by_rank = payload_by_rank(g, split, ranks, "d2h", pages)
        by_rank = [a + b for a, b in zip(h2d_by_rank, d2h_by_rank)]
        logical_bytes = 2 * g.page_bytes * pages
        total_tokens = 2 * tokens * count
    else:
        by_rank = payload_by_rank(g, split, ranks, direction, pages)
        logical_bytes = g.page_bytes * pages
        total_tokens = tokens * count
    data = {"direction": direction, "tokens_per_request": tokens, "requests": count,
            "phase": phase, "iteration": iteration, "total_tokens": total_tokens,
            "logical_payload_bytes": logical_bytes,
            "payload_bytes_by_rank": by_rank, "payload_bytes": sum(by_rank),
            "begin_ns": start, "end_ns": end, "elapsed_ms": seconds * 1000,
            "payload_gib_s": sum(by_rank) / 1024**3 / seconds,
            "tokens_per_s": total_tokens / seconds, "ranks": rows}
    if direction == "h2d":
        if rows[0]["layers"]:
            layer_ms = [(max(r["layers"][layer]["ready_ns"] for r in rows) - start) / 1e6
                        for layer in range(g.layers)]
            data.update(all_ranks_layer_ready_ms=layer_ms, first_layer_ms=layer_ms[0],
                        last_layer_ms=layer_ms[-1],
                        graph_observed_ms=(rows[0]["graph"]["observed_ns"] - start) / 1e6,
                        lookup_ms=(max(r["prepare_begin_ns"] for r in rows) - start) / 1e6,
                        prepare_ms=(rows[0]["launch_begin_ns"] - rows[0]["prepare_begin_ns"]) / 1e6,
                        launch_ms=(rows[0]["launch_end_ns"] - rows[0]["launch_begin_ns"]) / 1e6)
        else:
            data.update(lookup_ms=(max(r["prepare_begin_ns"] for r in rows) - start) / 1e6,
                        retrieve_ms=(end - min(r["launch_begin_ns"] for r in rows)) / 1e6)
    elif direction == "bidir":
        layer_ms = [(max(r["layers"][layer]["ready_ns"] for r in rows) - start) / 1e6
                    for layer in range(g.layers)]
        h2d_start = min(r["h2d_begin_ns"] for r in rows)
        h2d_finish = max(r["h2d_end_ns"] for r in rows)
        d2h_start = min(r["d2h_begin_ns"] for r in rows)
        d2h_finish = max(r["d2h_end_ns"] for r in rows)
        data.update(
            h2d_payload_bytes_by_rank=h2d_by_rank, d2h_payload_bytes_by_rank=d2h_by_rank,
            h2d_payload_bytes=sum(h2d_by_rank), d2h_payload_bytes=sum(d2h_by_rank),
            h2d_elapsed_ms=(h2d_finish - h2d_start) / 1e6,
            d2h_elapsed_ms=(d2h_finish - d2h_start) / 1e6,
            all_ranks_layer_ready_ms=layer_ms, first_layer_ms=layer_ms[0],
            last_layer_ms=layer_ms[-1],
            graph_observed_ms=(rows[0]["graph"]["observed_ns"] - start) / 1e6,
            lookup_ms=(max(r["prepare_begin_ns"] for r in rows) - h2d_start) / 1e6,
            prepare_ms=(rows[0]["launch_begin_ns"] - rows[0]["prepare_begin_ns"]) / 1e6,
            launch_ms=(rows[0]["launch_end_ns"] - rows[0]["launch_begin_ns"]) / 1e6,
            submit_window_ms=(max(r["submit_end_ns"] for r in rows) - d2h_start) / 1e6,
        )
    else:
        data.update(submit_window_ms=(max(r["submit_end_ns"] for r in rows) - start) / 1e6,
                    request_latency_ms=[(request["completed_ns"] - request["begin_ns"]) / 1e6
                                        for request in rows[0]["requests"]])
    output.write(json.dumps(data) + "\n")
    output.flush()
    if phase == "measure":
        print(f"{PROGRESS_MARKER} {direction} tokens={tokens} requests={count} round={iteration + 1} "
              f"time={data['elapsed_ms']:.3f} ms payload={data['payload_gib_s']:.3f} GiB/s", flush=True)


def _start_compute_overlap(torch, rank, enabled):
    if not enabled:
        return None, None
    import threading

    stop_event = threading.Event()
    stream = torch.cuda.Stream(device=rank)
    a = torch.randn((2048, 2048), device=f"cuda:{rank}", dtype=torch.float16)
    b = torch.randn((2048, 2048), device=f"cuda:{rank}", dtype=torch.float16)

    def worker():
        with torch.cuda.stream(stream):
            while not stop_event.is_set():
                torch.matmul(a, b)
                if not stop_event.is_set():
                    stream.synchronize()
            stream.synchronize()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return (stop_event, stream, a, b), thread


def _stop_compute_overlap(state, thread):
    if state is None:
        return
    stop_event, stream, a, b = state
    stop_event.set()
    thread.join()
    stream.synchronize()
    del a, b


def run_rank(rank, exp, split, backend, g, arena_pages, rendezvous, arm_dir):
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import torch.distributed as dist
    from flexkv import c_ext
    from flexkv.integration.sglang import connector as adapter
    from flexkv.integration.sglang.comm import FlexKVComm

    resolved_runtime = ("mapped" if getattr(c_ext, "gdr_uses_mapped_memory", False) else "native")
    effective_runtime = resolved_runtime if exp.runtime == "auto" else exp.runtime
    if backend == "gdr" and getattr(c_ext, "gdr_uses_mapped_memory", None) != (effective_runtime == "mapped"):
        raise RuntimeError(f"Rebuild the local c_ext for GDR runtime={effective_runtime}")
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    ranks = exp.gpu_count
    effective_cp = ranks if exp.cp_size is None else exp.cp_size
    clock = time.perf_counter_ns
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=ranks, timeout=timedelta(seconds=TIMEOUT_S + 60))
    gpu_pages = 2 * arena_pages + 1
    pool = SyntheticPool(torch, rank, ranks, split, g, gpu_pages)
    sgl_model = SimpleNamespace(num_hidden_layers=g.layers, kv_lora_rank=g.lora,
                                qk_nope_head_dim=128, qk_rope_head_dim=g.rope,
                                index_head_dim=g.index_dim, dtype=torch.int8)
    args = SimpleNamespace(tp_size=ranks, pp_size=1, dp_size=1, nnodes=1, node_rank=0,
                           attn_cp_size=effective_cp, enable_dp_attention=False, kv_cache_dtype=None,
                           enable_nsa_cache_layer_split=split, model_path="synthetic-target-model")
    # Supply the actual Gloo world group at the framework hook so FlexKVComm
    # neither imports nor calls SGLang. All collectives below run FlexKV code.
    original_comm = adapter.FlexKVComm
    adapter.FlexKVComm = lambda **kw: FlexKVComm(**kw, world_group=dist.group.WORLD)
    try:
        comm_kwargs = (
            {"attn_cp_rank": rank, "attn_cp_group": dist.group.WORLD}
            if effective_cp == ranks
            else {"attn_cp_rank": 0, "attn_tp_group": dist.group.WORLD, "attn_cp_group": None}
        )
        connector = adapter.FlexKVConnector(
            sgl_model_config=sgl_model, server_args=args, page_size=PAGE_SIZE,
            kvcache=pool, tp_rank=rank, dp_rank=0, pp_rank=0, **comm_kwargs)
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
    if connector.layer_done_counter is not None:
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
        stop_compute, compute_thread = _start_compute_overlap(torch, rank, exp.compute_overlap)
        try:
            row = store(connector, requests, clock, observer, exp)
        finally:
            _stop_compute_overlap(stop_compute, compute_thread)
        capture_round(dist, rank, ranks, output, g, split, "d2h", tokens, count, phase, iteration, row)

    def run_restore(requests, tokens, count, phase, iteration, generation, verify):
        pages = np.concatenate([r.dest for r in requests])
        touch_pages(torch, np, pool, np.r_[0, pages], generation, poison=True)
        torch.cuda.synchronize(rank)
        dist.barrier()
        errors = []
        checker = None
        if verify and exp.layerwise:
            def checker(layer):
                if not errors:
                    try:
                        touch_pages(torch, np, pool, pages, generation, layer=layer)
                    except AssertionError as exc:
                        # Consume the remaining signals and drain the graph
                        # before propagating a content mismatch across ranks.
                        errors.append(str(exc))
        stop_compute, compute_thread = _start_compute_overlap(torch, rank, exp.compute_overlap)
        try:
            row, event = restore(connector, requests, clock, observer, exp, checker)
        finally:
            _stop_compute_overlap(stop_compute, compute_thread)
        # The collective happens after the leader observes graph completion.
        # That establishes quiescence before checking for extra tokens.
        capture_round(dist, rank, ranks, output, g, split, "h2d", tokens, count, phase, iteration, row)
        dist.barrier()
        try:
            if event is not None:
                require_empty_eventfds(event)
            if verify and not exp.layerwise:
                touch_pages(torch, np, pool, pages, generation)
            if verify and not errors:
                check_untouched(np, pool, pages)
        except (RuntimeError, AssertionError) as exc:
            errors.append(str(exc))
        outcomes = [None] * ranks
        dist.all_gather_object(outcomes, errors)
        if any(outcomes):
            raise VerificationError(f"{phase}: layerwise verification failed: {outcomes}")

    def run_bidir(h2d_reqs, d2h_reqs, tokens, count, phase, iteration, h2d_gen, d2h_gen, verify):
        h2d_dest_pages = np.concatenate([r.dest for r in h2d_reqs])
        d2h_src_pages = np.concatenate([r.source for r in d2h_reqs])
        touch_pages(torch, np, pool, d2h_src_pages, d2h_gen, prepare=True)
        touch_pages(torch, np, pool, np.r_[0, h2d_dest_pages], h2d_gen, poison=True)
        torch.cuda.synchronize(rank)
        dist.barrier()
        stop_compute, compute_thread = _start_compute_overlap(torch, rank, exp.compute_overlap)
        try:
            row, event = bidir_transfer(connector, h2d_reqs, d2h_reqs, clock, observer, exp)
        finally:
            _stop_compute_overlap(stop_compute, compute_thread)
        capture_round(dist, rank, ranks, output, g, split, "bidir", tokens, count, phase, iteration, row)
        dist.barrier()
        errors = []
        try:
            if event is not None:
                require_empty_eventfds(event)
            if verify:
                touch_pages(torch, np, pool, h2d_dest_pages, h2d_gen)
                check_untouched(np, pool, h2d_dest_pages)
        except (RuntimeError, AssertionError) as exc:
            errors.append(str(exc))
        outcomes = [None] * ranks
        dist.all_gather_object(outcomes, errors)
        if any(outcomes):
            raise VerificationError(f"{phase}: bidirectional verification failed: {outcomes}")

    def shutdown_drained():
        nonlocal connector, manager, observer, owner, pool
        dist.barrier()
        # Join the IPC importers before any rank retires its exported buffers.
        if manager is not None:
            manager.shutdown()
            connector.kv_manager = None
        dist.barrier()
        connector.shutdown()
        if owner is not None:
            deadline = time.monotonic() + TIMEOUT_S
            while True:
                owner.check_healthy()
                with owner.lock:
                    if owner.endpoint is None:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"rank={rank}: GDR owner did not retire")
                time.sleep(POLL_INTERVAL_S)

        # Collect the exported storage while the CUDA runtime is still live.
        print(f"{PROGRESS_MARKER} rank={rank} collecting CUDA IPC allocations", flush=True)
        torch.cuda.synchronize(rank)
        del connector, manager, observer, owner, pool
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        print(f"{PROGRESS_MARKER} rank={rank} CUDA IPC cleanup complete", flush=True)
        dist.barrier()

    # Only the leader writes round records. Keep completed rows on disk if a
    # later case fails. The resident Host/GPU pools are reused for each arm.
    output = (arm_dir / "rounds.jsonl").open("w") if rank == 0 else None
    generation = 0
    last_step = "init"
    try:
        for direction, tokens, count in scenarios(exp):
            if direction == "bidir":
                last_step = f"bidir:{tokens}x{count}:reset_host"
                reset_host()
                half_pages = arena_pages // 2
                generation += 1
                h2d_gen = generation
                h2d_reqs = round_slots(
                    torch, np, exp, tokens, count, h2d_gen, arena_pages,
                    source_offset=0, dest_offset=arena_pages,
                    window_pages=half_pages, token_salt=0, rid_prefix="h",
                )
                last_step = f"bidir:{tokens}x{count}:seed_touch"
                touch_pages(torch, np, pool, np.concatenate([r.source for r in h2d_reqs]),
                            h2d_gen, prepare=True)
                torch.cuda.synchronize(rank)
                last_step = f"bidir:{tokens}x{count}:seed_store"
                run_store(h2d_reqs, tokens, count, "seed", 0)
                phases = [("verify", 0)] + [("warmup", i) for i in range(exp.warmups)] + [
                    ("measure", i) for i in range(exp.rounds)]
                for phase, iteration in phases:
                    generation += 1
                    d2h_gen = generation
                    d2h_reqs = round_slots(
                        torch, np, exp, tokens, count, d2h_gen, arena_pages,
                        source_offset=half_pages, dest_offset=arena_pages + half_pages,
                        window_pages=half_pages, token_salt=d2h_gen * 1_000_000_000, rid_prefix="d",
                    )
                    last_step = f"bidir:{tokens}x{count}:{phase}:{iteration}"
                    run_bidir(h2d_reqs, d2h_reqs, tokens, count, phase, iteration,
                              h2d_gen, d2h_gen, phase == "verify")
                    if phase == "verify":
                        last_step = f"bidir:{tokens}x{count}:verify_d2h"
                        run_restore(d2h_reqs, tokens, count, "verify_d2h", iteration, d2h_gen, True)
                continue
            if direction == "h2d":
                last_step = f"h2d:{tokens}x{count}:reset_host"
                reset_host()
                generation += 1
                requests = round_slots(torch, np, exp, tokens, count, generation, arena_pages)
                last_step = f"h2d:{tokens}x{count}:seed_touch"
                touch_pages(torch, np, pool, np.concatenate([r.source for r in requests]),
                            generation, prepare=True)
                torch.cuda.synchronize(rank)
                last_step = f"h2d:{tokens}x{count}:seed_store"
                run_store(requests, tokens, count, "seed", 0)
            phases = [("verify", 0)] + [("warmup", i) for i in range(exp.warmups)] + [
                ("measure", i) for i in range(exp.rounds)]
            for phase, iteration in phases:
                if direction == "h2d":
                    last_step = f"h2d:{tokens}x{count}:{phase}:{iteration}"
                    run_restore(requests, tokens, count, phase, iteration, generation, phase == "verify")
                    continue
                # Fresh Host keys and the same GPU arena size for every D2H sample.
                last_step = f"d2h:{tokens}x{count}:{phase}:{iteration}:reset_host"
                reset_host()
                generation += 1
                requests = round_slots(torch, np, exp, tokens, count, generation, arena_pages)
                last_step = f"d2h:{tokens}x{count}:{phase}:{iteration}:touch"
                touch_pages(torch, np, pool, np.concatenate([r.source for r in requests]),
                            generation, prepare=True)
                torch.cuda.synchronize(rank)
                last_step = f"d2h:{tokens}x{count}:{phase}:{iteration}:store"
                run_store(requests, tokens, count, phase, iteration)
                if phase == "verify":
                    last_step = f"d2h:{tokens}x{count}:{phase}:{iteration}:verify_restore"
                    run_restore(requests, tokens, count, "verify_d2h", iteration, generation, True)
        last_step = "final_reset_host"
        reset_host()
        last_step = "shutdown_drained"
        shutdown_drained()
    except VerificationError:
        shutdown_drained()
        raise
    except BaseException as exc:
        print(f"{PROGRESS_MARKER} rank={rank} failed at step={last_step}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        raise
    finally:
        if output is not None:
            output.close()
        dist.destroy_process_group()


def arm_env(exp, backend, scratch, runtime=None):
    effective_runtime = exp.runtime if runtime is None else runtime
    trace_on = bool(exp.worker_trace or exp.gdr_profile)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ENABLE_FLEXKV": "1", "FLEXKV_CONFIG_PATH": str(scratch / "config.json"),
        "FLEXKV_SERVER_CLIENT_MODE": "0", "FLEXKV_SERVER_LAUNCH_MODE": "embedded",
        "FLEXKV_SERVER_RECV_PORT": f"ipc://{scratch}/main",
        "FLEXKV_LAYERWISE_EVENTFD_SOCKET": str(scratch / "layers"),
        "FLEXKV_ENABLE_LAYERWISE_TRANSFER": str(int(exp.layerwise)), "FLEXKV_REGION_BATCH": "1",
        "FLEXKV_ENABLE_COLLECTIVE_SYNC": "1", "FLEXKV_DEDUP_INDEXER_GROUP": "1",
        "FLEXKV_CPU_LAYOUT": "BLOCKFIRST", "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE": "layer_parallel",
        "FLEXKV_ENABLE_MPS": "0", "FLEXKV_ENABLE_RADIXSHMEM": "0", "FLEXKV_CACHE_DRAFT_KV": "0",
        "FLEXKV_ENABLE_GDS": "0", "FLEXKV_ENABLE_NVCOMP": "0",
        "FLEXKV_USE_CE_TRANSFER_H2D": "1", "FLEXKV_USE_CE_TRANSFER_D2H": "1",
        "FLEXKV_ENABLE_CE_MEMCPY2D": "0", "FLEXKV_LAYERWISE_NOTIFY_MODE": "polling",
        "FLEXKV_ENABLE_CE_ONLY": str(int(effective_runtime == "mapped")),
        "FLEXKV_LOCAL_GDR": "both" if backend == "gdr" else "off",
        "FLEXKV_GDR_EXPERIMENTAL_MAPPED_H2D": str(int(effective_runtime == "mapped")),
        "FLEXKV_GDR_QP_COUNT": str(exp.qp_count), "FLEXKV_GDR_LOG_COUNTERS": "1",
        "FLEXKV_GDR_READ_DEPTH": str(exp.read_depth), "FLEXKV_GDR_READBACKS": str(exp.readbacks),
        "FLEXKV_GDR_WR_API": str(exp.wr_api),
        "FLEXKV_LOG_LEVEL": "INFO" if trace_on else "WARNING",
        "FLEXKV_TRANSFER_TRACE": str(int(trace_on)),
    })
    if exp.h2d_qp_count > 0:
        env["FLEXKV_GDR_H2D_QP_COUNT"] = str(exp.h2d_qp_count)
    if exp.d2h_qp_count > 0:
        env["FLEXKV_GDR_D2H_QP_COUNT"] = str(exp.d2h_qp_count)
    if exp.max_sge > 0:
        env["FLEXKV_GDR_MAX_SGE"] = str(exp.max_sge)
    if exp.inline_control_bytes > 0:
        env["FLEXKV_CONTROL_INLINE_BYTES"] = str(exp.inline_control_bytes)
    if exp.gdr_profile:
        env["FLEXKV_GDR_PROFILE"] = "1"
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
        expected = [
            sum(
                row["payload_bytes_by_rank"][rank] if row["direction"] == direction
                else row[f"{direction}_payload_bytes_by_rank"][rank]
                for row in rounds
                if row["direction"] in (direction, "bidir")
            )
            for rank in range(exp.gpu_count)
        ]
        if measured != expected:
            raise RuntimeError(f"{direction}: GDR rank payload bytes {measured} != expected {expected}")
        result[direction] = [dict(rank=rank, h2d_bytes=row[0], d2h_bytes=row[1], payload_wrs=row[2],
                                  posts=row[3], cqes=row[4], readbacks=row[5], qp_payload_wrs=row[6:])
                             for rank, row in enumerate(group)]
    return result


def _decode_dir_profile(vals):
    return {
        "calls": vals[0],
        "dispatch_ms": vals[1] / 1e6,
        "post_ms": vals[2] / 1e6,
        "poll_ms": vals[3] / 1e6,
        "owner_wait_ms": vals[4] / 1e6,
        "sleep_ms": vals[5] / 1e6,
        "sleep_count": vals[6],
        "notify_ms": vals[7] / 1e6,
        "first_milestone_ms": vals[8] / 1e6,
        "route_total_ms": vals[9] / 1e6,
    }


def read_route_profiles(lines, backend, exp):
    if backend == "ce" or not exp.gdr_profile:
        return None
    groups = [ast.literal_eval(line.split(PROFILE_MARKER, 1)[1].strip())
              for line in lines if PROFILE_MARKER in line]
    if not groups:
        return None
    result = {}
    for offset, direction in ((0, "h2d"), (10, "d2h")):
        candidates = [g for g in groups if sum(r[offset] for r in g) > 0]
        if not candidates:
            continue
        group = candidates[0]
        result[direction] = [{"rank": rank, **_decode_dir_profile(row[offset:offset + 10])}
                             for rank, row in enumerate(group)]
    return result or None


def read_owner_profiles(lines):
    rows = []
    for line in lines:
        m = OWNER_PROFILE_RE.search(line)
        if m:
            rows.append({
                "device": int(m.group(1)),
                "r_calls": int(m.group(2)),
                "r_sync_ms": float(m.group(3)),
                "r_total_ms": float(m.group(4)),
                "v_calls": int(m.group(5)),
                "v_sync_ms": float(m.group(6)),
                "v_total_ms": float(m.group(7)),
            })
    return rows or None


def summarize_worker_trace(lines):
    buckets = {"h2d": [], "d2h": []}
    for line in lines:
        m = XFER_TRACE_RE.search(line)
        if not m:
            continue
        key = "h2d" if m.group(5) == "1" else "d2h"
        buckets[key].append({
            "type": m.group(2),
            "blocks": int(m.group(3)),
            "bytes": int(m.group(4)),
            "ipc_ms": float(m.group(7)),
            "wait_ms": float(m.group(8)),
            "xfer_ms": float(m.group(9)),
            "e2e_ms": float(m.group(10)),
            "bw_mb_s": float(m.group(11)),
        })
    out = {}
    for key, entries in buckets.items():
        if not entries:
            continue
        out[key] = {
            "count": len(entries),
            **{field: distribution([e[field] for e in entries])
               for field in ("ipc_ms", "wait_ms", "xfer_ms", "e2e_ms", "bw_mb_s")},
        }
    return out or None


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
        if direction == "h2d":
            if "first_layer_ms" in rows[0]:
                extras = ["first_layer_ms", "last_layer_ms", "graph_observed_ms", "lookup_ms", "prepare_ms", "launch_ms"]
            else:
                extras = ["lookup_ms", "retrieve_ms"]
        elif direction == "bidir":
            extras = ["h2d_elapsed_ms", "d2h_elapsed_ms", "first_layer_ms", "last_layer_ms",
                      "graph_observed_ms", "lookup_ms", "prepare_ms", "launch_ms", "submit_window_ms"]
        else:
            extras = ["submit_window_ms"]
        totals = sum(r["elapsed_ms"] for r in rows) / 1000
        summary = {"direction": direction, "tokens_per_request": tokens, "requests": count,
                   "rounds": len(rows), "payload_bytes_per_round": rows[0]["payload_bytes"],
                   "weighted_payload_gib_s": sum(r["payload_bytes"] for r in rows) / 1024**3 / totals,
                   "metrics": {key: distribution([r[key] for r in rows]) for key in keys + extras}}
        if "first_layer_ms" in rows[0]:
            summary["metrics"]["l0_after_launch_ms"] = distribution([
                r["first_layer_ms"] - (r["lookup_ms"] + r["prepare_ms"] + r["launch_ms"])
                for r in rows
            ])
            summary["metrics"]["l1_to_last_ms"] = distribution([
                r["last_layer_ms"] - r["first_layer_ms"] for r in rows
            ])
            if direction == "h2d":
                summary["metrics"]["drain_obs_ms"] = distribution([
                    r["graph_observed_ms"] - r["last_layer_ms"] for r in rows
                ])
            if rows[0].get("ranks") and all("layers" in rk and rk["layers"] for rk in rows[0]["ranks"]):
                ranks = len(rows[0]["ranks"])
                per_rank_last = []
                for idx in range(ranks):
                    vals = [(r["ranks"][idx]["layers"][-1]["ready_ns"] - r["begin_ns"]) / 1e6 for r in rows]
                    per_rank_last.append(statistics.fmean(vals))
                summary["per_rank_mean_ms"] = per_rank_last
            summary["layer_ready_ms"] = [distribution(list(times)) for times in
                                         zip(*(r["all_ranks_layer_ready_ms"] for r in rows))]
        if direction == "d2h":
            summary["request_latency_ms"] = distribution([ms for r in rows for ms in r["request_latency_ms"]])
        summaries.append(summary)
    return summaries


def detect_gdr_runtime():
    probe = subprocess.run(
        [sys.executable, "-c",
         "from flexkv import c_ext; print('mapped' if getattr(c_ext, 'gdr_uses_mapped_memory', False) else 'native')"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return None
    out = probe.stdout.strip().splitlines()
    return out[-1] if out else None


def run_arm(exp, split, backend, g, arm_dir, runtime, extra_cli=()):
    arm_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="kv-transfer-", dir="/tmp") as directory:
        scratch = Path(directory)
        (scratch / "config.json").write_text(json.dumps({
            "cpu_cache_gb": exp.host_cache_gib,
            "ssd_cache_gb": 0,
        }))
        env = arm_env(exp, backend, scratch, runtime=runtime)
        command = [sys.executable, str(Path(__file__).resolve()), *extra_cli,
                   "--run-arm", exp.name, backend, str(int(split)),
                   str(arm_dir), str(scratch / "gloo")]
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
            "runtime": runtime,
            "configuration": asdict(exp), "geometry": asdict(g),
            "rank_metadata": json.loads((arm_dir / "ranks.json").read_text()),
            "round_file": str(arm_dir / "rounds.jsonl"), "log_file": str(log_path),
            "payload_basis": "per-rank GDR counters reconciled with all rounds" if backend == "gdr"
                             else "region geometry and confirmed logical transfer lengths",
            "gdr_counters": read_route_counters(lines, backend, exp, rounds),
            "gdr_profile": read_route_profiles(lines, backend, exp),
            "gdr_owner_profile": read_owner_profiles(lines),
            "worker_trace": [line for line in lines if "[XFER] op=" in line],
            "worker_trace_summary": summarize_worker_trace(lines),
            "summary": summarize(rounds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="validate and print the matrix without GPU imports")
    parser.add_argument("--output", type=Path, default=Path(time.strftime("kv-transfer-%Y%m%d-%H%M%S")))
    parser.add_argument("--experiments", nargs="+", help="optional subset of experiment names to run")
    parser.add_argument("--backends", nargs="+", choices=("ce", "gdr"), default=None,
                        help="optional backend filter (e.g. --backends gdr)")
    parser.add_argument("--layer-split", nargs="+", type=int, choices=(0, 1), default=None,
                        help="optional layer_split filter (e.g. --layer-split 0 1)")
    parser.add_argument("--qp-count", type=int, default=None,
                        help="override qp_count (1..16) for all selected experiments")
    parser.add_argument("--h2d-qp-count", type=int, default=None,
                        help="override active H2D QP count (0=auto min(qp_count,4), or 1..qp_count)")
    parser.add_argument("--d2h-qp-count", type=int, default=None,
                        help="override active D2H QP count (0=auto qp_count, or 1..qp_count)")
    parser.add_argument("--max-sge", type=int, default=None,
                        help="override max SGEs per RDMA WR (0=auto up to 16, 1=disable Multi-SGE)")
    parser.add_argument("--read-depth", type=int, default=None,
                        help="override read_depth (>=0) for all selected experiments")
    parser.add_argument("--gdr-profile", action="store_true", help="enable C++ GDR phase profile and [XFER] trace")
    parser.add_argument("--worker-trace", action="store_true", help="enable FLEXKV_TRANSFER_TRACE=1")
    parser.add_argument("--batch-preflight", action="store_true", help="override batch_preflight=True for all arms")
    parser.add_argument("--batch-store", action="store_true",
                        help="override batch_store=True (use store_kv_batch_with_result) for all arms")
    parser.add_argument("--no-wr-api", action="store_true",
                        help="disable ibv_qp_ex extended WR API (fall back to ibv_post_send)")
    parser.add_argument("--inline-control-bytes", type=int, choices=(0, 256, 512, 1024, 4096),
                        default=None, help="override FLEXKV_CONTROL_INLINE_BYTES for all arms")
    parser.add_argument("--compute-overlap", action="store_true", help="override compute_overlap=True for all arms")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort immediately on first failed arm instead of recording failed_arms and continuing")
    parser.add_argument("--continue-on-error", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-arm", nargs=5, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.qp_count is not None and not 1 <= args.qp_count <= 16:
        raise ValueError("--qp-count must be between 1 and 16")
    if args.h2d_qp_count is not None and not 0 <= args.h2d_qp_count <= 16:
        raise ValueError("--h2d-qp-count must be between 0 and 16")
    if args.d2h_qp_count is not None and not 0 <= args.d2h_qp_count <= 16:
        raise ValueError("--d2h-qp-count must be between 0 and 16")
    if args.max_sge is not None and not 0 <= args.max_sge <= 16:
        raise ValueError("--max-sge must be between 0 and 16")
    if args.read_depth is not None and args.read_depth < 0:
        raise ValueError("--read-depth must be non-negative")
    g = geometry()
    wanted_backends = set(args.backends) if args.backends is not None else None
    wanted_splits = {bool(v) for v in args.layer_split} if args.layer_split is not None else None
    selected = []
    for exp in EXPERIMENTS:
        overrides = {}
        if args.gdr_profile:
            overrides["gdr_profile"] = True
            overrides["worker_trace"] = True
        elif args.worker_trace:
            overrides["worker_trace"] = True
        if args.batch_preflight:
            overrides["batch_preflight"] = True
        if args.batch_store:
            overrides["batch_store"] = True
        if args.no_wr_api:
            overrides["wr_api"] = 0
        if args.inline_control_bytes is not None:
            overrides["inline_control_bytes"] = args.inline_control_bytes
        if args.compute_overlap:
            overrides["compute_overlap"] = True
        if args.qp_count is not None:
            overrides["qp_count"] = args.qp_count
        if args.h2d_qp_count is not None:
            overrides["h2d_qp_count"] = args.h2d_qp_count
        if args.d2h_qp_count is not None:
            overrides["d2h_qp_count"] = args.d2h_qp_count
        if args.max_sge is not None:
            overrides["max_sge"] = args.max_sge
        if args.read_depth is not None:
            overrides["read_depth"] = args.read_depth
        if wanted_backends is not None:
            filtered_backends = tuple(b for b in exp.backends if b in wanted_backends)
            if not filtered_backends:
                continue
            overrides["backends"] = filtered_backends
        if wanted_splits is not None:
            filtered_splits = tuple(s for s in exp.layer_split if s in wanted_splits)
            if not filtered_splits:
                continue
            overrides["layer_split"] = filtered_splits
        selected.append(replace(exp, **overrides) if overrides else exp)
    if args.experiments is not None:
        wanted = set(args.experiments)
        known = {exp.name for exp in EXPERIMENTS}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown experiment names: {unknown}")
        selected = [exp for exp in selected if exp.name in wanted]
    names = {exp.name: exp for exp in selected}
    if not names or len(names) != len(selected):
        raise ValueError("No matching experiment/backend/layer-split arms selected")
    extra_cli = []
    if args.gdr_profile:
        extra_cli.append("--gdr-profile")
    elif args.worker_trace:
        extra_cli.append("--worker-trace")
    if args.batch_preflight:
        extra_cli.append("--batch-preflight")
    if args.batch_store:
        extra_cli.append("--batch-store")
    if args.no_wr_api:
        extra_cli.append("--no-wr-api")
    if args.inline_control_bytes is not None:
        extra_cli.extend(["--inline-control-bytes", str(args.inline_control_bytes)])
    if args.compute_overlap:
        extra_cli.append("--compute-overlap")
    if args.qp_count is not None:
        extra_cli.extend(["--qp-count", str(args.qp_count)])
    if args.h2d_qp_count is not None:
        extra_cli.extend(["--h2d-qp-count", str(args.h2d_qp_count)])
    if args.d2h_qp_count is not None:
        extra_cli.extend(["--d2h-qp-count", str(args.d2h_qp_count)])
    if args.max_sge is not None:
        extra_cli.extend(["--max-sge", str(args.max_sge)])
    if args.read_depth is not None:
        extra_cli.extend(["--read-depth", str(args.read_depth)])
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
                                           "cases": list(scenarios(exp))} for exp in selected]}, indent=2))
        return
    if sys.platform != "linux":
        parser.error("This benchmark requires Linux, GPU CUDA IPC and real RDMA for the GDR arm")
    built_runtime = detect_gdr_runtime()
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"page_tokens": PAGE_SIZE, "seed": SEED, "host_poll_interval_s": POLL_INTERVAL_S,
              "built_gdr_runtime": built_runtime,
              "time_scope": "first Connector entry to observed data readiness or drained D2H completion",
              "sample_scope": "local Host L2, synthetic target KV and indexer, no model compute (unless compute_overlap=True)",
              "arms": [], "skipped_arms": [], "failed_arms": []}
    report_path = output / "report.json"
    for exp in selected:
        effective_runtime = (built_runtime or "mapped") if exp.runtime == "auto" else exp.runtime
        for split, backend in itertools.product(exp.layer_split, exp.backends):
            if built_runtime is not None and built_runtime != effective_runtime:
                msg = f"requires runtime={effective_runtime}, built c_ext is {built_runtime}"
                print(f"{PROGRESS_MARKER} skipping {exp.name} backend={backend} split={int(split)} ({msg})",
                      flush=True)
                report["skipped_arms"].append({"experiment": exp.name, "backend": backend,
                                               "layer_split": split, "reason": msg})
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                continue
            print(f"{PROGRESS_MARKER} starting {exp.name} backend={backend} split={int(split)} "
                  f"runtime={effective_runtime}", flush=True)
            arm_out = output / f"{exp.name}-{backend}-split{int(split)}"
            try:
                row = run_arm(exp, split, backend, g, arm_out, effective_runtime, extra_cli=extra_cli)
            except Exception as exc:
                if args.fail_fast and not args.continue_on_error:
                    raise
                print(f"{PROGRESS_MARKER} failed {exp.name} backend={backend} split={int(split)}: {exc}",
                      flush=True)
                report["failed_arms"].append({"experiment": exp.name, "backend": backend,
                                              "layer_split": split, "runtime": effective_runtime,
                                              "error": str(exc), "log_file": str(arm_out / "runner.log")})
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                continue
            report["arms"].append(row)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{PROGRESS_MARKER} report {report_path} "
          f"(passed={len(report['arms'])}, skipped={len(report['skipped_arms'])}, "
          f"failed={len(report['failed_arms'])})", flush=True)


if __name__ == "__main__":
    main()
