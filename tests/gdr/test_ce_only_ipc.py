"""CPU-only regression tests for CUDA IPC in the CE-only profile."""

import ctypes
import importlib
import os
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call, patch


class Device:
    def __init__(self, kind, index):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return (self.type, self.index) == (other.type, other.index)


class Tensor:
    is_cuda = True

    def __init__(self, device):
        self.device = device


class CeOnlyIpcTest(unittest.TestCase):
    def setUp(self):
        devices = ["first", "second"]
        self.devices = devices
        torch = types.ModuleType("torch")
        torch.Tensor, torch.device, torch.dtype = Tensor, Device, type
        torch.cuda = types.SimpleNamespace(
            get_device_properties=lambda index: types.SimpleNamespace(
                uuid=devices[index.index if isinstance(index, Device) else index]
            ),
            device_count=lambda: len(devices),
            device=Mock(return_value=nullcontext()),
        )
        self.torch = torch
        self.reductions = types.ModuleType("torch.multiprocessing.reductions")
        torch_mp = types.ModuleType("torch.multiprocessing")
        torch_mp.reductions = self.reductions
        package = types.ModuleType("flexkv")
        package.__path__ = [str(Path(__file__).resolve().parents[2] / "flexkv")]
        debug = types.ModuleType("flexkv.common.debug")
        debug.flexkv_logger = Mock()
        modules = {
            "flexkv": package, "torch": torch, "torch.multiprocessing": torch_mp,
            "torch.multiprocessing.reductions": self.reductions,
            "flexkv.common.debug": debug, "zmq": types.ModuleType("zmq"),
        }
        with patch.dict(sys.modules, modules), patch.dict(
            os.environ, {"FLEXKV_ENABLE_CE_ONLY": "1"}
        ), patch.object(ctypes, "CDLL", return_value=Mock()) as cdll:
            for name in ("common", "common.ce_only", "common.ipc_uuid", "common.memory_handle"):
                sys.modules.pop(f"flexkv.{name}", None)
            self.handles = importlib.import_module("flexkv.common.memory_handle")
        cdll.assert_any_call("libcudart.so")
        self.assertFalse(any(call.args[0].startswith("libcuda.so") for call in cdll.call_args_list))

    def test_rebuild_uses_worker_ordinal(self):
        def rebuild(storage_device):
            return Tensor(Device("cuda", storage_device))

        self.reductions.reduce_tensor = lambda tensor: (rebuild, (tensor.device.index,))
        with patch.dict(os.environ, {"FLEXKV_ENABLE_CE_ONLY": "1"}):
            handle = self.handles.TensorSharedHandle(Tensor(Device("cuda", 0)), device_id=4)
        self.devices.reverse()
        self.assertEqual(handle.get_tensor().device, Device("cuda", 1))
        self.torch.cuda.device.assert_called_once_with(Device("cuda", 1))
        self.assertEqual(handle.rebuild_args, (0,))

    def test_rebuild_rejects_copy_for_gdr(self):
        handle = self.handles.TensorSharedHandle.__new__(self.handles.TensorSharedHandle)
        handle.use_direct_ipc = False
        handle.local_gdr_view = ("owner",)
        handle.device = Device("cuda", 0)
        handle.rebuild_func = lambda: Tensor(Device("cuda", 1))
        handle.rebuild_args = ()
        with self.assertRaisesRegex(RuntimeError, "CUDA IPC rebuilt on"):
            handle.get_tensor()

    def test_host_workers_export_independent_refcounts(self):
        counters = {1: 1, 2: 1}

        def rebuild(storage_device, ticket):
            counters[ticket] -= 1
            return Tensor(Device("cuda", storage_device))

        self.reductions.reduce_tensor = Mock(side_effect=[(rebuild, (0, 1)), (rebuild, (0, 2))])
        tensor = Tensor(Device("cuda", 0))
        with patch.dict(os.environ, {"FLEXKV_ENABLE_CE_ONLY": "1"}):
            handle = self.handles.TensorSharedHandle.for_host_workers(tensor, 0)
        self.reductions.reduce_tensor.assert_has_calls([call(tensor), call(tensor)])
        self.assertEqual(handle.get_tensor().device, handle.h2d_peer.get_tensor().device)
        self.assertEqual(counters, {1: 0, 2: 0})


if __name__ == "__main__":
    unittest.main()
