"""Runtime settings for the host-only transfer build."""

import os


def ce_only_enabled() -> bool:
    return os.environ.get("FLEXKV_ENABLE_CE_ONLY", "0") == "1"


def apply_ce_only_settings() -> None:
    if not ce_only_enabled():
        return
    defaults = {
        "FLEXKV_USE_CE_TRANSFER_H2D": "1",
        "FLEXKV_USE_CE_TRANSFER_D2H": "1",
        "FLEXKV_ENABLE_CE_MEMCPY2D": "0",
        "FLEXKV_LAYERWISE_NOTIFY_MODE": "polling",
        "FLEXKV_ENABLE_MPS": "0",
    }
    for key, expected in defaults.items():
        if os.environ.get(key, expected) != expected:
            raise ValueError(f"FLEXKV_ENABLE_CE_ONLY=1 requires {key}={expected}")
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
