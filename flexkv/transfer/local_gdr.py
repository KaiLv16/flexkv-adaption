"""Control plane for opt-in same-host GDR. Socket messages never carry KV."""

import json
import os
import socket
import tempfile
import threading

_PROTOCOL = "flexkv-local-gdr-1"
_TIMEOUT = 30
_owners = set()


def enabled(direction=None):
    mode = os.environ.get("FLEXKV_LOCAL_GDR", "off").strip().lower()
    if mode not in {"off", "h2d", "d2h", "both"}:
        raise ValueError("FLEXKV_LOCAL_GDR must be off, h2d, d2h, or both")
    return mode != "off" if direction is None else mode in {direction, "both"}


def _read(stream):
    raw = stream.readline(65537)
    if not raw or len(raw) > 65536 or not raw.endswith(b"\n"):
        raise RuntimeError("Invalid GDR handshake")
    return json.loads(raw)


def _write(stream, message):
    stream.write(json.dumps(message).encode() + b"\n")
    stream.flush()


class Owner:
    def __init__(self, tensors, local_rank):
        from flexkv import c_ext
        from flexkv.common.debug import flexkv_logger

        self._log = flexkv_logger
        self.tensors = list(tensors)
        self.device = self.tensors[0].device.index
        self.d2h_enabled = enabled("d2h")
        if any(not tensor.is_cuda or tensor.device.index != self.device for tensor in self.tensors):
            raise ValueError("A GDR owner requires views on exactly one local device")
        names = os.environ.get("FLEXKV_GDR_NICS", "").strip()
        nic = ""
        if names:
            choices = [name.strip() for name in names.split(",")]
            if not 0 <= local_rank < len(choices) or not choices[local_rank]:
                raise ValueError("FLEXKV_GDR_NICS must cover every local rank")
            nic = choices[local_rank]
        depth = int(os.environ.get("FLEXKV_GDR_READ_DEPTH", "0"))
        if not hasattr(c_ext, "GdrOwnerEndpoint"):
            raise RuntimeError("Build c_ext with FLEXKV_ENABLE_GDR=1")
        self.endpoint = c_ext.GdrOwnerEndpoint(self.device, nic, depth)
        if enabled("h2d"):
            c_ext.gdr_device_visible(self.device)
        # Cache the negotiated endpoint once so a link fault during registration
        # cannot leave half of the handles advertising a different endpoint.
        self.fingerprint = self.endpoint.endpoint
        self.views = [self.endpoint.export_tensor(tensor) for tensor in self.tensors]
        self.directory = tempfile.TemporaryDirectory(prefix="flexkv-gdr-", dir="/tmp")
        self.path = os.path.join(self.directory.name, "owner")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.lock = threading.Lock()
        self.sessions = set()
        self.retiring = False
        self.uncertain = False
        try:
            self.listener.bind(self.path)
            self.listener.listen()
            self.listener.settimeout(0.5)
            self.thread = threading.Thread(target=self._listen, daemon=True, name="flexkv-gdr-owner")
            _owners.add(self)
            self.thread.start()
        except BaseException:
            self.listener.close()
            self.directory.cleanup()
            _owners.discard(self)
            raise

    def metadata(self, index):
        remote, length, key = self.views[index]
        return (self.path, self.endpoint.nic, self.fingerprint,
                self.endpoint.read_depth, remote, length, key)

    def _listen(self):
        while not self.retiring:
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self.retiring:
                    raise
                break
            thread = threading.Thread(target=self._session, args=(conn,),
                                      daemon=True, name="flexkv-gdr-lease")
            with self.lock:
                if self.retiring or self.uncertain:
                    conn.close()
                    continue
                self.sessions.add(thread)
            try:
                thread.start()
            except BaseException:
                with self.lock:
                    self.sessions.remove(thread)
                conn.close()
                raise

    def _session(self, conn):
        from flexkv import c_ext

        qpn = None
        posted_possible = False
        try:
            with conn, conn.makefile("rwb") as stream:
                conn.settimeout(_TIMEOUT)
                request = _read(stream)
                sender = request.get("qpn")
                if (request.get("protocol") != _PROTOCOL or
                        request.get("endpoint") != self.fingerprint or
                        request.get("depth") != self.endpoint.read_depth or
                        type(sender) is not int or not 0 < sender <= 0xFFFFFF):
                    raise RuntimeError("GDR endpoint handshake mismatch")
                with self.lock:
                    if self.retiring or self.uncertain:
                        raise RuntimeError("GDR owner no longer accepting leases")
                    qpn = self.endpoint.accept(sender)
                _write(stream, {"protocol": _PROTOCOL, "qpn": qpn})
                if stream.read(1) != b"C":
                    return
                # The sender can post only after acknowledging connection and
                # receiving its lease confirmation. Treat lost confirmation as unsafe.
                posted_possible = True
                stream.write(b"A")
                stream.flush()
                conn.settimeout(None)
                while True:
                    command = stream.read(1)
                    if command == b"R":
                        c_ext.gdr_producer_ready(self.device)
                    elif command == b"V":
                        c_ext.gdr_device_visible(self.device)
                    elif command == b"D":
                        with self.lock:
                            self.endpoint.release(qpn)
                            qpn = None
                        stream.write(b"A")
                        stream.flush()
                        return
                    else:
                        return
                    stream.write(b"A")
                    stream.flush()
        except Exception as exc:
            self._log.error("Local GDR owner session failed: %s", exc)
        finally:
            with self.lock:
                if qpn is not None:
                    if posted_possible:
                        # A dropped connection does not revoke DMA. Retain the
                        # original allocation and its target until isolation.
                        self.uncertain = True
                        self._log.error("Local GDR Owner lease lost; allocation retained")
                    else:
                        try:
                            self.endpoint.release(qpn)
                        except Exception:
                            self.uncertain = True
                self.sessions.discard(threading.current_thread())

    def retire(self):
        with self.lock:
            if self.retiring:
                return
            self.retiring = True
        threading.Thread(target=self._retire, daemon=True,
                         name="flexkv-gdr-retire").start()

    def _retire(self):
        self.thread.join()
        self.listener.close()
        with self.lock:
            sessions = list(self.sessions)
        for session in sessions:
            session.join()
        with self.lock:
            if self.uncertain:
                return
            self.endpoint = None
            self.tensors.clear()
        self.directory.cleanup()
        _owners.discard(self)


def export_views(client, pairs):
    pairs = list(pairs)
    if not pairs:
        return
    owner = Owner((tensor for tensor, _ in pairs), client.intra_client_id)
    client._local_gdr_owner = owner
    for index, (_, handle) in enumerate(pairs):
        handle.local_gdr_view = owner.metadata(index)


def import_views(handles, tensors):
    for handle, tensor in zip(handles, tensors, strict=True):
        tensor._flexkv_gdr_view = getattr(handle, "local_gdr_view", None)


def attach_routes(group, devices, tensors, host_tensors):
    from flexkv import c_ext
    from flexkv.common.debug import flexkv_logger

    leases = []
    confirmed = []
    readbacks = int(os.environ.get("FLEXKV_GDR_READBACKS", "1"))
    if not 1 <= readbacks <= 16:
        raise ValueError("FLEXKV_GDR_READBACKS must be 1..16")
    try:
        for rank, device in enumerate(devices):
            views = {}
            identity = None
            for tensor in tensors:
                if tensor.device.index != device:
                    continue
                meta = getattr(tensor, "_flexkv_gdr_view", None)
                if meta is None:
                    raise RuntimeError(f"GDR: GPU {device} has a view without an Owner")
                path, nic, fingerprint, depth, remote, length, key = meta
                if length != tensor.numel() * tensor.element_size():
                    raise ValueError("GDR: device view size changed across import")
                if identity is None:
                    identity = (path, nic, fingerprint, depth)
                elif identity != (path, nic, fingerprint, depth):
                    raise ValueError("GDR: inconsistent Owner identity")
                views[(tensor.data_ptr(), length, remote, key)] = None
            if identity is None:
                raise RuntimeError(f"GDR: GPU {device} has no Owner views")
            path, nic, fingerprint, depth = identity
            sender = c_ext.GdrSender(device, nic, fingerprint, depth, readbacks)
            for host in host_tensors:
                sender.register_host(host)
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            leases.append(conn)
            conn.settimeout(_TIMEOUT)
            conn.connect(path)
            with conn.makefile("rwb") as stream:
                _write(stream, {"protocol": _PROTOCOL, "endpoint": fingerprint,
                                "depth": depth, "qpn": sender.qpn})
                reply = _read(stream)
                if reply.get("protocol") != _PROTOCOL:
                    raise RuntimeError("GDR: Owner protocol mismatch")
                sender.connect(reply["qpn"], list(views))
            group.install_gdr_route(rank, sender, conn.fileno())
            conn.sendall(b"C")
            if conn.recv(1) != b"A":
                raise RuntimeError("GDR: Owner did not confirm the lease")
            confirmed.append(conn)
            flexkv_logger.info("Local GDR route ready: rank=%s device=%s nic=%s depth=%s",
                               rank, device, nic, depth)
    except BaseException:
        try:
            close_routes(group, confirmed)
        finally:
            for conn in leases:
                conn.close()
        raise
    return leases


def close_routes(group, leases):
    """Destroy the drained sender QPs before acknowledging Owner retirement."""
    group.close_gdr()
    error = None
    for conn in leases:
        try:
            conn.sendall(b"D")
            if conn.recv(1) != b"A":
                raise RuntimeError("GDR: missing Owner drain acknowledgement")
        except Exception as exc:
            if error is None:
                error = exc
        finally:
            conn.close()
    if error is not None:
        raise error


def quarantine(leases):
    # SHUT_RDWR also wakes the owner for descriptors duplicated into c_ext.
    for conn in leases:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
