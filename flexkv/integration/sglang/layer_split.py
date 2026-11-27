"""Physical DSA KV/indexer registration, shared by the two SGLang adapters."""

from flexkv.integration.sglang.indexer import active_index_layers


def _registered_buffers(buffers, scratch, layers, holders, rank):
    if len(buffers) != len(holders) or scratch is None or not scratch.is_contiguous() or not scratch.numel():
        raise ValueError("CP registration requires aligned layers and a contiguous receive buffer")
    # Scratch makes a valid IPC handle even on a rank owning no active layers.
    # Only owners may ever dereference these pointers in a transfer.
    spans = [(scratch.data_ptr(), scratch.data_ptr() + scratch.nbytes)]
    registered = []
    for layer in layers:
        buf = buffers[layer]
        owned = holders[layer] == rank
        shape = scratch.shape if owned else (0, *scratch.shape[1:])
        if (buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != scratch.dtype
                or buf.device != scratch.device or not buf.is_contiguous()):
            raise ValueError(f"CP buffer disagrees with owner or geometry at layer {layer}")
        if owned:
            spans.append((buf.data_ptr(), buf.data_ptr() + buf.nbytes))
        registered.append(buf if owned else scratch)
    spans.sort()
    if any(end > begin for (_, end), (begin, _) in zip(spans, spans[1:])):
        raise ValueError("CP owner buffers and receive scratch must not overlap")
    return registered


def resolve_layer_split(pool, page_size, rank, size, count, plan):
    """Return a rank-independent owner map and IPC-ready rank-local tensors."""
    if (not getattr(pool, "layer_shard_enabled", False) or pool.layer_num != count
            or pool.layer_shard_size != size or pool.layer_shard_rank != rank):
        raise ValueError("Physical CP pool and FlexKV transfer topology disagree")
    holders = tuple(pool._get_layer_owner_rank(pool.start_layer + i) for i in range(count))
    if any(type(owner) is not int or not 0 <= owner < size for owner in holders):
        raise ValueError("CP pool returned an invalid layer owner")
    if plan is not None and (plan.num_layers != count or plan.cp_rank != rank or
                             plan.cp_size != size or holders != tuple(plan.holder_of(i) for i in range(count))):
        raise ValueError("SGLang broadcast plan and physical CP owners disagree")
    kv = _registered_buffers(pool.kv_buffer, pool.remote_kv_buffer, range(count), holders, rank)
    sample = kv[0]
    if sample.ndim == 4 and sample.shape[1:3] == (1, page_size):
        # The page-major MLA buffer has one head. A view preserves its byte image
        # and exposes the ordinary token-major registration contract.
        kv = [buf.view(-1, 1, sample.shape[3]) for buf in kv]
        sample = kv[0]
    if sample.ndim != 3 or sample.shape[1] != 1 or sample.shape[0] % page_size:
        raise ValueError("CP KV requires page-aligned single-head MLA buffers")
    if sample.shape[2] * sample.element_size() * page_size % 8:
        raise ValueError("CP KV page bytes must be 8-byte aligned")

    raw_index = getattr(pool, "index_k_with_scale_buffer", None)
    if raw_index is None:
        return holders, kv, (), None

    active = active_index_layers(getattr(pool, "skip_topk_layers", None), count)
    scratch = getattr(pool, "remote_index_k_with_scale_buffer", None)
    if scratch is None:
        scratch = pool.index_key_cache.remote_buffer
    if (scratch.ndim != 2 or scratch.shape[0] != sample.shape[0] // page_size
            or scratch.shape[1] * scratch.element_size() % 8):
        raise ValueError("CP indexer requires one aligned row per KV page")
    if not active:
        raise ValueError("DSA pool has no active indexer layers")
    index = _registered_buffers(raw_index, scratch, active, holders, rank)
    return holders, kv, active, index
