"""Resolve torch CUDA IPC storage to the local ordinal of its owner device."""

import inspect

import torch


def ipc_device_uuid(device) -> str:
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if uuid is None or not str(uuid).strip():
        raise RuntimeError("CUDA device UUID is unavailable for tensor IPC")
    return str(uuid)


def ipc_device_from_uuid(uuid: str) -> torch.device:
    for index in range(torch.cuda.device_count()):
        if ipc_device_uuid(index) == uuid:
            return torch.device("cuda", index)
    raise RuntimeError(f"CUDA IPC owner device UUID is not visible: {uuid}")


def ipc_import_device(handle) -> torch.device:
    uuid = getattr(handle, "ipc_device_uuid", None)
    return ipc_device_from_uuid(uuid) if uuid is not None else handle.device


def ipc_device_arg(rebuild, args, uuid):
    parameters = list(inspect.signature(rebuild).parameters)
    has_storage_device = "storage_device" in parameters
    index = parameters.index("storage_device") if has_storage_device else 6
    if index < len(args):
        if args[index] == uuid:
            return None
        if has_storage_device and isinstance(args[index], int):
            return index
    raise RuntimeError("Cannot resolve the torch CUDA IPC storage device argument")
