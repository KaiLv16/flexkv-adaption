"""CPU-only regressions for the local GDR Owner lease protocol."""

import importlib.util
import os
import socket
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "local_gdr",
    Path(__file__).resolve().parents[1] / "flexkv" / "transfer" / "local_gdr.py",
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

    def start_lease(self):
        owner = local_gdr.Owner.__new__(local_gdr.Owner)
        owner._log = Mock()
        owner.device = 3
        owner.endpoint = types.SimpleNamespace(read_depth=4, accept=Mock(return_value=8),
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
                protocol=local_gdr._PROTOCOL, endpoint="local", depth=4, qpn=7))
            self.assertEqual(local_gdr._read(stream)["qpn"], 8)
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
        owner.endpoint.release.assert_called_once_with(8)
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
        owner.endpoint.release.assert_called_once_with(8)

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


if __name__ == "__main__":
    unittest.main()
