"""Project global CP registrations onto a node's packed host block."""
from dataclasses import replace


def node_groups(model, node):
    """Return packed groups and their row positions in the global registration."""
    width = model.cp_size // model.nnodes
    first = node * width
    groups, selections = [], []
    for ordinal, group in enumerate(model.layer_groups):
        indices = [
            i for i, layer in enumerate(group.layer_indices)
            if first <= model.h2d_layer_holders[layer] < first + width
        ]
        if indices:
            groups.append(replace(group, num_layers=len(indices),
                                  layer_indices=[group.layer_indices[i] for i in indices]))
            selections.append((ordinal, indices))
    if not groups:
        raise ValueError(f"Physical CP node {node} has no cache data")
    return groups, selections


def localize(model, ranks):
    width = model.effective_tp_size_per_node
    if not ranks or ranks[0] < 0:
        raise ValueError(f"Invalid global CP registration ranks: {ranks}")
    node = ranks[0] // width
    if node >= model.nnodes or ranks != list(range(node * width, (node + 1) * width)):
        raise ValueError(f"Unexpected CP registration ranks on one node: {ranks}")
    groups, selections = node_groups(model, node)
    owners = tuple(r - node * width if r // width == node else -1 for r in model.h2d_layer_holders)
    local = replace(model, layer_split_node=node, layer_groups=groups, h2d_layer_holders=owners)
    local.freeze()
    return local, selections


def slice_registration(blocks, layouts, selections):
    local_blocks, local_layouts = [], []
    if blocks is None or layouts is None or len(blocks) != len(layouts):
        raise ValueError("Physical CP requires grouped GPU handles and layouts")
    for ordinal, indices in selections:
        if ordinal >= len(layouts) or len(blocks[ordinal]) != layouts[ordinal].num_layer:
            raise ValueError("Physical CP registration has inconsistent group rows")
        if not indices or max(indices) >= layouts[ordinal].num_layer:
            raise ValueError("Physical CP group selection exceeds registered rows")
        local_blocks.append([blocks[ordinal][i] for i in indices])
        local_layouts.append(replace(layouts[ordinal], num_layer=len(indices), _kv_shape=None))
    return local_blocks, local_layouts
