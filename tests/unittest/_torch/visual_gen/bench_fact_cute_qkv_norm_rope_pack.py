# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACT kernel-contract case file for qwen_image_cute_qkv_norm_rope_pack_v1.

eager_fn = the production chain being replaced (post-qkv_merge): sequence-cat
+ torch.ops.trtllm.fused_dit_qk_norm_rope in place. fused_fn = the CuteDSL op.
Production Qwen-Image shapes: H=24, D=128, S_txt=51, S_img=4096, bf16.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.environ["FACT_ROOT"])
from backends.core.kernel_contract import Case  # noqa: E402
from tensorrt_llm._torch.visual_gen import fact_kernels  # noqa: E402,F401

NUM_HEADS, HEAD_DIM = 24, 128
HD = NUM_HEADS * HEAD_DIM
S_TXT, S_IMG = 51, 4096
S_TOTAL = S_TXT + S_IMG
EPS = 1e-6


def _make_inputs():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    mk = lambda s: torch.randn(1, s, 3 * HD, device=dev, dtype=dt)  # noqa: E731
    theta = torch.randn(S_TOTAL, HEAD_DIM // 2, device=dev, dtype=torch.float32)
    cos = torch.cos(theta).repeat_interleave(2, dim=-1).contiguous()
    sin = torch.sin(theta).repeat_interleave(2, dim=-1).contiguous()
    w = lambda: torch.randn(HEAD_DIM, device=dev, dtype=dt) * 0.1 + 1.0  # noqa: E731
    return (mk(S_TXT), mk(S_IMG), cos, sin, w(), w(), w(), w())


def _eager(txt_qkv, img_qkv, cos, sin, nqw, nkw, naqw, nakw):
    qkv = torch.cat([txt_qkv, img_qkv], dim=1).contiguous()
    B, S, D = qkv.shape
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv.view(B * S, D),
        NUM_HEADS, NUM_HEADS, NUM_HEADS, HEAD_DIM, EPS,
        nqw, nkw, naqw, nakw, cos, sin, S_TXT, True, S,
    )
    return qkv


def _fused(txt_qkv, img_qkv, cos, sin, nqw, nkw, naqw, nakw):
    return torch.ops.trtllm.fact_cute_qkv_norm_rope_pack(
        txt_qkv, img_qkv, cos, sin, nqw, nkw, naqw, nakw, HEAD_DIM, EPS,
    )


_BYTES = (2 * S_TOTAL * HD + S_TOTAL * 3 * HD) * 2  # read 2 packed inputs + write joint

CASES = [
    Case(
        label="fact_cute_qkv_norm_rope_pack[qwen_image,txt51+img4096,bf16]",
        make_inputs=_make_inputs,
        eager_fn=_eager,
        fused_fn=_fused,
        tol=2e-2,
        bytes_moved=_BYTES,
    ),
]

if __name__ == "__main__":
    from backends.core.kernel_contract import run_cases
    run_cases(CASES, source=__file__)
