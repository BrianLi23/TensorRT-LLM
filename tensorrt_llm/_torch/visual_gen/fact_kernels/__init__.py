# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACT agent-authored fused kernels (visual_gen). Importing this package
registers the custom ops (``torch.ops.trtllm.fact_*``)."""

from . import fact_cute_qkv_norm_rope_pack  # noqa: F401

__all__ = ["fact_cute_qkv_norm_rope_pack"]
