"""Control plane for opt-in same-host GDR. Socket messages never carry KV."""

import json
import os
import select
import socket
import tempfile
import threading

_PROTOCOL = "flexkv-local-gdr-2"
_TIMEOUT = 30
_owners = set()
_PEER_CLOSED = select.POLLHUP | select.POLLERR | getattr(select, "POLLRDHUP", 0)


class LocalGdrQuarantined(RuntimeError):
    """The transfer session lost its DMA drain guarantee."""


def enabled(direction=None):
    mode = os.environ.get("FLEXKV_LOCAL_GDR", "off").strip().lower()
    if mode not in {"off", "h2d", "d2h", "both"}:
        raise ValueError("FLEXKV_LOCAL_GDR must be off, h2d, d2h, or both")
    return mode != "off" if direction is None else mode in {direction, "both"}


def _qp_count():
    count = int(os.environ.get("FLEXKV_GDR_QP_COUNT", "1"))
    if not 1 <= count <= 16:
        raise ValueError("FLEXKV_GDR_QP_COUNT must be 1..16")
    return count


def _valid_qpns(values, count):
    return (isinstance(values, list) and len(values) == count
            and all(type(qpn) is int and 0 < qpn <= 0xFFFFFF for qpn in values)
            and len(set(values)) == count)


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
        if (enabled("h2d") and c_ext.gdr_uses_mapped_memory
                and os.environ.get("FLEXKV_GDR_EXPERIMENTAL_MAPPED_H2D") != "1"):
            raise RuntimeError("Mapped GDR H2D requires FLEXKV_GDR_EXPERIMENTAL_MAPPED_H2D=1")
        self.qp_count = _qp_count()
        self.endpoint = c_ext.GdrOwnerEndpoint(self.device, nic, depth, self.qp_count)
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
        self.confirmed_sockets = set()
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

        targets = []
        posted_possible = False
        try:
            with conn, conn.makefile("rwb") as stream:
                conn.settimeout(_TIMEOUT)
                request = _read(stream)
                senders = request.get("qpns")
                if (request.get("protocol") != _PROTOCOL or
                        request.get("endpoint") != self.fingerprint or
                        request.get("depth") != self.endpoint.read_depth or
                        not _valid_qpns(senders, self.qp_count)):
                    raise RuntimeError("GDR endpoint handshake mismatch")
                with self.lock:
                    if self.retiring or self.uncertain:
                        raise RuntimeError("GDR owner no longer accepting leases")
                    for sender in senders:
                        targets.append(self.endpoint.accept(sender))
                _write(stream, {"protocol": _PROTOCOL, "qpns": targets})
                if stream.read(1) != b"C":
                    return
                # The sender can post only after acknowledging connection and
                # receiving its lease confirmation. Treat lost confirmation as unsafe.
                posted_possible = True
                with self.lock:
                    self.confirmed_sockets.add(conn)
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
                            while targets:
                                self.endpoint.release(targets[len(targets) - 1])
                                targets.pop()
                            self.confirmed_sockets.discard(conn)
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
                self.confirmed_sockets.discard(conn)
                if targets:
                    if posted_possible:
                        # A dropped connection does not revoke DMA. Retain the
                        # original allocation and its target until isolation.
                        self.uncertain = True
                        self._log.error("Local GDR Owner lease lost; allocation retained")
                    else:
                        for target in targets:
                            try:
                                self.endpoint.release(target)
                            except Exception:
                                self.uncertain = True
                self.sessions.discard(threading.current_thread())

    def check_healthy(self):
        """Observe a lost sender even while its Owner command is in a CUDA call."""
        with self.lock:
            if not self.uncertain and self.confirmed_sockets:
                poller = select.poll()
                for conn in self.confirmed_sockets:
                    fd = conn.fileno()
                    if fd < 0:
                        self.uncertain = True
                        break
                    poller.register(fd, _PEER_CLOSED)
                if poller.poll(0):
                    self.uncertain = True
            if self.uncertain:
                raise LocalGdrQuarantined("Local GDR Owner lost a confirmed Worker lease")

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
    qp_count = _qp_count()
    readbacks = int(os.environ.get("FLEXKV_GDR_READBACKS", "0"))
    if not 0 <= readbacks <= 16:
        raise ValueError("FLEXKV_GDR_READBACKS must be 0 (auto) or 1..16")
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
            sender = c_ext.GdrSender(device, nic, fingerprint, depth, readbacks, qp_count)
            for host in host_tensors:
                sender.register_host(host)
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            leases.append(conn)
            conn.settimeout(_TIMEOUT)
            conn.connect(path)
            with conn.makefile("rwb") as stream:
                _write(stream, {"protocol": _PROTOCOL, "endpoint": fingerprint,
                                "depth": depth, "qpns": sender.qpns})
                reply = _read(stream)
                targets = reply.get("qpns")
                if reply.get("protocol") != _PROTOCOL or not _valid_qpns(targets, qp_count):
                    raise RuntimeError("GDR: Owner QP handshake mismatch")
                sender.connect(targets, list(views))
            group.install_gdr_route(rank, sender, conn.fileno())
            conn.sendall(b"C")
            if conn.recv(1) != b"A":
                raise RuntimeError("GDR: Owner did not confirm the lease")
            confirmed.append(conn)
            flexkv_logger.info("Local GDR route ready: rank=%s device=%s nic=%s qps=%s read_depth=%s",
                               rank, device, nic, qp_count, depth)
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
