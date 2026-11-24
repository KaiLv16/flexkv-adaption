"""Bounded IO on independent Mooncake Stores sharing a TransferEngine."""

import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from flexkv.common.debug import flexkv_logger
from flexkv.external.mooncake_store_utils import (
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreClient,
    _split_host_port,
)


def _start_engine(config):
    from mooncake.engine import TransferEngine

    host, port = _split_host_port(config.local_hostname)
    host = host or "127.0.0.1"
    session_host = f"[{host}]" if ":" in host else host
    metadata = config.metadata_server or "P2PHANDSHAKE"
    if metadata == "P2PHANDSHAKE":
        engine = TransferEngine()
        status = engine.initialize(host, metadata, config.protocol, config.device_name)
        if status == 0:
            return engine, f"{session_host}:{engine.get_rpc_port()}"
    else:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        for attempt in range(5):
            # For HTTP/etcd the engine and Store must publish the same session.
            if attempt == 0 and port:
                selected_port = int(port)
            else:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.bind(("", 0))
                    selected_port = sock.getsockname()[1]
            session = f"{session_host}:{selected_port}"
            engine = TransferEngine()
            status = engine.initialize(session, metadata, config.protocol, config.device_name)
            if status == 0:
                return engine, session
    raise RuntimeError(f"Mooncake parallel TransferEngine initialization failed: {status}")


class ParallelMooncakeStoreClient(MooncakeStoreClient):
    def __init__(self, config):
        self._lanes = int(os.getenv("FLEXKV_MOONCAKE_IO_WORKERS", "8"))
        self._min_pages = int(os.getenv("FLEXKV_MOONCAKE_IO_MIN_PAGES", "8"))
        self._io_budget = int(os.getenv("FLEXKV_MOONCAKE_IO_MAX_INFLIGHT_BYTES", str(512 * 2**20)))
        if not 1 <= self._lanes <= 8 or min(self._min_pages, self._io_budget) <= 0:
            raise ValueError("Invalid Mooncake parallel IO lane, page or byte limit")
        self._io_lock = threading.Lock()
        # Setup, health and warmup retain the base client's contract.
        super().__init__(config)
        super().shutdown_get_workers()
        self._get_closed = False
        self._io_executor = ThreadPoolExecutor(
            max_workers=self._lanes, thread_name_prefix="flexkv-store-lane"
        )

    def setup(self):
        from mooncake.store import MooncakeDistributedStore

        if self._is_setup:
            return
        cfg = self._config
        self._engine, session = _start_engine(cfg)
        raw_engine = self._engine.get_engine()
        segment = cfg.global_segment_size // self._lanes
        if segment >= 4096:
            segment = segment // 4096 * 4096
        self._stores = []
        for lane in range(self._lanes):
            store = MooncakeDistributedStore()
            code = store.setup(
                session, cfg.metadata_server or "P2PHANDSHAKE", segment, DEFAULT_LOCAL_BUFFER_SIZE,
                cfg.protocol, cfg.device_name, cfg.master_addr, raw_engine,
            )
            if code != 0:
                raise RuntimeError(f"Mooncake Store lane {lane} setup failed: {code}")
            self._stores.append(store)
        self._store = self._stores[0]
        self._is_setup = True
        flexkv_logger.info(
            "[Mooncake] parallel Store lanes=%d session=%s device=%s",
            self._lanes, session, cfg.device_name,
        )

    def register_buffer(self, ptr, size=0):
        status = self._engine.register_memory(int(ptr), size)
        if status != 0:
            raise RuntimeError(f"Mooncake shared buffer registration failed: {status}")

    def unregister_buffer(self, ptr):
        status = self._engine.unregister_memory(int(ptr))
        if status != 0:
            raise RuntimeError(f"Mooncake shared buffer unregistration failed: {status}")

    def shutdown_get_workers(self):
        super().shutdown_get_workers()
        with self._io_lock:
            if self._io_executor is not None:
                self._io_executor.shutdown(wait=True)
                self._io_executor = None

    def zero_copy_get_impl(self, keys, pointers, sizes):
        return self._parallel_io("batch_get_into", keys, pointers, sizes, self._config.max_get_batch_bytes)

    def zero_copy_put_impl(self, keys, pointers, sizes):
        return self._parallel_io("batch_put_from", keys, pointers, sizes, self._config.max_put_batch_bytes)

    def _parallel_io(self, method, keys, pointers, sizes, sdk_limit):
        self._validate_buffers(keys, pointers, sizes)
        costs = [((size + 4095) // 4096) * 4096 + 8192 for size in sizes]
        if any(cost > self._io_budget for cost in costs):
            raise ValueError("A Mooncake object exceeds the padded parallel IO budget")
        if sdk_limit and any(size > sdk_limit for size in sizes):
            raise ValueError("A Mooncake object exceeds the SDK batch limit")
        if method == "batch_get_into":
            regions = sorted(zip(pointers, sizes))
            if any(ptr + size > next_ptr for (ptr, size), (next_ptr, _) in zip(regions, regions[1:])):
                raise ValueError("Overlapping Mooncake GET destinations")
        ordered = method == "batch_put_from" and len(set(keys)) != len(keys)
        results = []
        with self._io_lock:
            if self._io_executor is None:
                raise RuntimeError("Mooncake parallel IO is closed")
            start = 0
            window_bytes = 0
            for end, cost in enumerate(costs):
                if window_bytes + cost > self._io_budget:
                    results.extend(self._run_window(method, keys, pointers, sizes, sdk_limit, start, end, ordered))
                    start, window_bytes = end, 0
                window_bytes += cost
            if start < len(keys):
                results.extend(self._run_window(method, keys, pointers, sizes, sdk_limit, start, len(keys), ordered))
        return results

    def _run_window(self, method, keys, pointers, sizes, sdk_limit, start, end, ordered):
        count = end - start
        lanes = 1 if ordered else min(self._lanes, max(1, count // self._min_pages))

        def run(lane, lo, hi):
            operation = getattr(self._stores[lane], method)
            return self._bounded_io(operation, keys[lo:hi], pointers[lo:hi], sizes[lo:hi], sdk_limit)

        if lanes == 1:
            return run(0, start, end)
        futures = []
        try:
            for lane in range(lanes):
                lo = start + count * lane // lanes
                hi = start + count * (lane + 1) // lanes
                futures.append(self._io_executor.submit(run, lane, lo, hi))
        finally:
            # Even when one lane fails, no caller may recycle the Host pages
            # while another lane still owns the source or destination.
            wait(futures)
        return [result for future in futures for result in future.result()]
