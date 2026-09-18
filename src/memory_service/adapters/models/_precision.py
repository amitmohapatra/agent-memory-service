"""Model precision on CPU.

Several published checkpoints declare ``float16`` or ``bfloat16`` in their ``config.json`` —
right for the GPU they were trained on, wrong here. The CPU has no native half-precision
arithmetic, so torch emulates it: slower than ``float32``, and measurably less accurate.
With the checkpoint's own dtype, ``granite-embedding-small-english-r2`` returns vectors whose
norm is 1.0041 rather than 1.0 (cosine similarity assumes unit length),
``DeBERTa-v3-mnli-fever-anli`` returns probabilities summing to 1.00012, and numpy refuses
``bfloat16`` outright — which is what breaks late chunking.

So on CPU we load in ``float32``: faster *and* more correct. On an accelerator the
checkpoint's own dtype is left alone, because there half precision is the point.
"""

from __future__ import annotations


def cpu_dtype_kwargs(device: str | None = "cpu") -> dict[str, str]:
    """``from_pretrained`` keyword arguments pinning CPU inference to float32."""
    if device and device.strip().lower() not in ("cpu", ""):
        return {}
    return {"dtype": "float32"}
