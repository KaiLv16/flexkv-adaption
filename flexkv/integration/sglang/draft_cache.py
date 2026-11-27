"""Describe slot-aligned speculative KV as ordinary FlexKV block members."""

from dataclasses import dataclass

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang.indexer import active_index_layers


@dataclass
class CacheRegion:
    name: str
    spec: LayerGroupSpec
    layout: KVCacheLayout
    buffers: list


class DraftLayerCounter:
    """Every draft layer belongs to target milestone zero."""

    def __init__(self, counter):
        self.counter = counter

    def wait_until(self, layer):
        self.counter.wait_until(0)


def _check_pool(pool):
    if (
        getattr(pool, "swa_kv_pool", None) is not None
        or getattr(pool, "_arena", None) is not None
        or getattr(pool, "requires_physical_write_loc", False)
        or getattr(pool, "sparse_layer_id_mapping", None) is not None
    ):
        raise ValueError("Speculative KV requires full-attention pools in the target slot space")
    if any(getattr(pool, attr, None) is not None for attr in
           ("kv_scale_buffer", "k_scale_buffer", "v_scale_buffer", "mamba_pool")):
        raise ValueError("Speculative KV cannot omit external scale or recurrent state")
    if getattr(pool, "enable_int8_kv_cache", False) and not getattr(pool, "int8_kv_cache_offline_load", False):
        raise ValueError("Speculative KV requires inline or constant quantization scales")


def _fields(pool, page):
    _check_pool(pool)
    count = pool.layer_num
    names = ("kv_buffer",) if getattr(pool, "kv_buffer", None) is not None else ("k_buffer", "v_buffer")
    for name in names:
        values = getattr(pool, name, None)
        if values is None or len(values) != count:
            raise ValueError(f"Incomplete {name} in speculative KV pool")
        normalized = []
        for value in values:
            if value is None or not value.is_contiguous():
                raise ValueError(f"Expected contiguous storage in {name}")
            if value.ndim == 4 and value.shape[1:3] == (1, page):
                value = value.view(-1, 1, value.shape[3])
            if value.ndim != 3:
                raise ValueError(f"Expected token-major [tokens, heads, width] in {name}")
            normalized.append(value)
        yield name, tuple(range(count)), normalized, page
    index = getattr(pool, "index_k_with_scale_buffer", None)
    if index is not None:
        if len(index) != count:
            raise ValueError("Indexer layer count differs from KV")
        ids = active_index_layers(getattr(pool, "skip_topk_layers", None), count)
        if ids:
            buffers = [index[i] for i in ids]
            if any(t is None or t.ndim != 2 for t in buffers):
                raise ValueError("Expected one page-packed indexer row per page")
            yield "index", ids, buffers, 1


def build_draft_registration(target, drafts, model, page, split=None):
    """Build one atomic Host block schema. All draft regions complete at target layer zero."""
    _check_pool(target)
    if model.pp_size != 1 or (split is None and model.effective_tp_size_per_node != model.effective_tp_size):
        raise ValueError("Speculative cache requires the complete transfer group on one node and PP=1")
    if target.layer_num != model.num_layers or not drafts:
        raise ValueError("Speculative cache needs the full target stage and at least one draft pool")
    if split is None and model.cp_size > 1:
        raise ValueError("Speculative CP cache requires a physically layer-split target")
    if split is None:
        fields = list(_fields(target, page))
    else:
        _, kv, ids, index = split
        fields = [("kv_buffer", tuple(range(model.num_layers)), kv, page)]
        if index is not None:
            fields.append(("index", ids, index, 1))
    first = fields[0][2][0]
    pages = first.shape[0] // page
    if pages == 0 or first.shape[0] % page:
        raise ValueError("Target KV is not page aligned")
    # With TP, each rank has a private slice of the packed host block. CP
    # replicas instead get distinct explicitly owned regions in a shared block.
    storage_heads = first.shape[1] * model.tp_size
    result = []

    def region(name, ids, tensors, tokens, owner=None):
        sample = tensors[0]
        if sample is None or not sample.numel() or sample.ndim not in (2, 3):
            raise ValueError(f"Empty or unsupported buffer in {name}")
        heads = sample.shape[1] if sample.ndim == 3 else 1
        width = sample.shape[sample.ndim - 1]
        if (sample.shape[0] != pages * tokens or sample.device != first.device
                or heads * width * tokens * sample.element_size() % 8):
            raise ValueError(f"Invalid page geometry or device in {name}")
        for tensor in tensors:
            if (tensor is None or tensor.shape != sample.shape or tensor.dtype != sample.dtype
                    or tensor.device != first.device or not tensor.is_contiguous()
                    or tensor.data_ptr() % 8):
                raise ValueError(f"Inconsistent or unaligned buffers in {name}")
        spec = LayerGroupSpec(len(ids), heads, width, list(ids), dtype=sample.dtype,
                              compress_ratio=page // tokens, transfer_owner_rank=owner)
        layout = KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=len(ids),
                               num_block=pages, tokens_per_block=tokens, num_head=heads,
                               head_size=width, kv_dim=1, num_kv_heads=storage_heads)
        result.append(CacheRegion(name, spec, layout, tensors))

    for name, ids, tensors, tokens in fields:
        region("target." + name, ids, tensors, tokens)
    spans = [(t.data_ptr(), t.data_ptr() + t.nbytes) for r in result for t in r.buffers]
    if split is not None:
        # resolve_layer_split already validated intentional shared broadcast scratch.
        spans = list(set(spans))
    draft_spans = []
    for runner, pool in enumerate(drafts):
        if getattr(pool, "layer_shard_enabled", False) or getattr(pool, "page_size", page) != page:
            raise ValueError("Draft KV must use full layers and the same page size as the target")
        if pool.layer_num < 1:
            raise ValueError("Empty draft KV pool")
        for name, ids, tensors, tokens in _fields(pool, page):
            for layer, tensor in zip(ids, tensors):
                # A CP rank's draft bytes can differ from every other rank's.
                # Fence *all* replicas on the first target milestone.
                owners = range(model.cp_size) if split is not None else (None,)
                for owner in owners:
                    region(f"draft.{runner}.{name}.{layer}", [0], [tensor], tokens, owner)
                draft_spans.append((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes))
    ordered = sorted((*spans, *draft_spans))
    if any(end > start for (_, end), (start, _) in zip(ordered, ordered[1:])):
        raise ValueError("Speculative KV cache regions overlap")
    return result
