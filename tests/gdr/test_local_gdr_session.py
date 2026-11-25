"""Regressions for local GDR Owner leases and fault propagation."""

import importlib.util
import os
import socket
import sys
import threading
import types
import unittest
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[2] / "flexkv" / "transfer" / "local_gdr.py"
compiled = (source.with_name("local_gdr" + suffix) for suffix in EXTENSION_SUFFIXES)
module_path = next((path for path in compiled if path.is_file()), source)
spec = importlib.util.spec_from_file_location(
    "local_gdr", module_path,
)
local_gdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_gdr)


class LocalGdrLeaseTest(unittest.TestCase):
    def setUp(self):
        self.native = types.SimpleNamespace(gdr_producer_ready=Mock(), gdr_device_visible=Mock())
        self.package = types.ModuleType("flexkv")
        self.package.c_ext = self.native
        self.patch = patch.dict(sys.modules, {"flexkv": self.package})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def start_lease(self, qp_count=1, senders=None, accept=None):
        owner = local_gdr.Owner.__new__(local_gdr.Owner)
        owner._log = Mock()
        owner.device = 3
        owner.qp_count = qp_count
        owner.endpoint = types.SimpleNamespace(read_depth=4, accept=Mock(side_effect=accept or (lambda qpn: qpn + 16)),
                                               release=Mock())
        owner.fingerprint = "local"
        owner.lock = threading.Lock()
        owner.sessions = set()
        owner.confirmed_sockets = set()
        owner.retiring = owner.uncertain = False
        caller, owner_fd = socket.socketpair()
        caller.settimeout(1)
        thread = threading.Thread(target=owner._session, args=(owner_fd,))
        owner.sessions.add(thread)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(caller.close)
        with caller.makefile("rwb") as stream:
            local_gdr._write(stream, dict(
                protocol=local_gdr._PROTOCOL, endpoint="local", depth=4,
                qpns=[7] if senders is None else senders))
            if senders is None:
                self.assertEqual(local_gdr._read(stream)["qpns"], [23])
            elif accept is None:
                self.assertEqual(local_gdr._read(stream)["qpns"], [qpn + 16 for qpn in senders])
        return owner, caller, thread

    def test_producer_visibility_and_explicit_drain(self):
        owner, caller, thread = self.start_lease()
        caller.sendall(b"CRVVD")
        with caller.makefile("rb") as stream:
            self.assertEqual(stream.read(5), b"AAAAA")
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.native.gdr_producer_ready.assert_called_once_with(3)
        self.assertEqual(self.native.gdr_device_visible.call_count, 2)
        owner.endpoint.release.assert_called_once_with(23)
        self.assertFalse(owner.uncertain)

    def test_confirmed_worker_loss_keeps_target_registered(self):
        owner, caller, thread = self.start_lease()
        caller.sendall(b"C")
        self.assertEqual(caller.recv(1), b"A")
        caller.close()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(owner.uncertain)
        owner.endpoint.release.assert_not_called()

    def test_unconfirmed_handshake_releases_unused_target(self):
        owner, caller, thread = self.start_lease()
        caller.close()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(owner.uncertain)
        owner.endpoint.release.assert_called_once_with(23)

    def test_four_queues_drain_in_one_lease(self):
        owner, caller, thread = self.start_lease(4, [4, 7, 5, 11])
        caller.sendall(b"CD")
        with caller.makefile("rwb") as stream:
            self.assertEqual(stream.read(2), b"AA")
        thread.join(1)
        self.assertFalse(owner.uncertain)
        self.assertEqual([call.args[0] for call in owner.endpoint.release.call_args_list],
                         [27, 21, 23, 20])

    def test_partial_handshake_releases_created_targets(self):
        def accept(qpn):
            if qpn == 5:
                raise RuntimeError("test QP registration failure")
            return qpn + 16

        owner, caller, thread = self.start_lease(4, [4, 7, 5, 11], accept)
        self.assertEqual(caller.recv(1), b"")
        thread.join(1)
        self.assertFalse(owner.uncertain)
        self.assertEqual([call.args[0] for call in owner.endpoint.release.call_args_list],
                         [20, 23])

    def test_peer_loss_detected_while_owner_waits_for_gpu(self):
        waiting = threading.Event()
        ready = threading.Event()

        def wait_for_gpu(_):
            waiting.set()
            ready.wait(2)

        self.native.gdr_producer_ready.side_effect = wait_for_gpu
        owner, caller, thread = self.start_lease()
        self.addCleanup(ready.set)
        caller.sendall(b"CR")
        self.assertEqual(caller.recv(1), b"A")
        self.assertTrue(waiting.wait(1))
        owner.check_healthy()
        caller.close()
        with self.assertRaises(local_gdr.LocalGdrQuarantined):
            owner.check_healthy()
        self.assertTrue(thread.is_alive())
        ready.set()
        thread.join(1)
        owner.endpoint.release.assert_not_called()

    def test_mapped_h2d_requires_explicit_opt_in(self):
        self.native.gdr_uses_mapped_memory = True
        self.native.GdrOwnerEndpoint = Mock()
        debug = types.SimpleNamespace(flexkv_logger=Mock())
        tensor = types.SimpleNamespace(is_cuda=True, device=types.SimpleNamespace(index=0))
        with patch.dict(os.environ, {"FLEXKV_LOCAL_GDR": "h2d"}, clear=True), patch.dict(
            sys.modules, {"flexkv.common.debug": debug}
        ):
            with self.assertRaisesRegex(RuntimeError, "FLEXKV_GDR_EXPERIMENTAL_MAPPED_H2D"):
                local_gdr.Owner([tensor], 0)
        self.native.GdrOwnerEndpoint.assert_not_called()

    def test_sender_teardown_precedes_owner_ack(self):
        order = []

        class Lease:
            def sendall(self, message):
                order.append(message)

            def recv(self, _):
                return b"A"

            def close(self):
                order.append("fd closed")

        group = types.SimpleNamespace(close_gdr=lambda: order.append("QP closed"))
        local_gdr.close_routes(group, [Lease()])
        self.assertEqual(order, ["QP closed", b"D", "fd closed"])


@unittest.skipUnless(sys.platform == "linux" and importlib.util.find_spec("torch"),
                     "requires a FlexKV runtime on Linux")
class LocalGdrIncidentTest(unittest.TestCase):
    def setUp(self):
        try:
            from flexkv import c_ext  # noqa: F401
        except ImportError:
            self.skipTest("requires a built FlexKV extension")
        from flexkv.transfer_manager import TransferManagerInterProcessHandle

        self.handle = TransferManagerInterProcessHandle(
            types.SimpleNamespace(tp_size=1, dp_size=1), types.SimpleNamespace(), "/tmp/unused"
        )
        self.addCleanup(self.handle.shutdown)

    def test_worker_can_spawn_with_parent_health_lock(self):
        # The incomplete config makes the child fail before allocating any KV.
        self.handle._start_process()
        self.assertTrue(self.handle.start_event.wait(timeout=15))
        self.assertTrue(self.handle.error_parent_conn.poll(timeout=15))
        self.assertIn("instance_num", self.handle.error_parent_conn.recv())
        self.handle.process.join(timeout=5)
        self.assertEqual(self.handle.process.exitcode, 1)

    def test_pending_dma_reports_session_fault_across_process_pipe(self):
        from flexkv.transfer.local_gdr import LocalGdrQuarantined

        self.handle.error_child_conn.send(LocalGdrQuarantined("uncertain RDMA drain"))
        with self.assertRaisesRegex(LocalGdrQuarantined, "uncertain RDMA drain"):
            self.handle.wait(timeout=0)
        with self.assertRaisesRegex(LocalGdrQuarantined, "uncertain RDMA drain"):
            self.handle.submit_batch([])


if __name__ == "__main__":
    unittest.main()
