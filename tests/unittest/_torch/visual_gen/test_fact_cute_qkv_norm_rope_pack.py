# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity + CUDA-graph test for the FACT qwen_image CuteDSL QKV-norm-rope-pack
superset kernel, vs the pure-torch _fact_cute_qkv_norm_rope_pack_ref."""
import pytest
import torch

from tensorrt_llm._torch.visual_gen.fact_kernels.fact_cute_qkv_norm_rope_pack import (
    _fact_cute_qkv_norm_rope_pack_ref,
    fact_cute_qkv_norm_rope_pack,
)

_H, _D = 24, 128
_HD = _H * _D
_EPS = 1e-6


def _inputs(s_txt, s_img, device):
    torch.manual_seed(0)
    dt = torch.bfloat16
    mk = lambda s: torch.randn(1, s, 3 * _HD, device=device, dtype=dt)  # noqa: E731
    th = torch.randn(s_txt + s_img, _D // 2, device=device, dtype=torch.float32)
    cos = torch.cos(th).repeat_interleave(2, dim=-1).contiguous()
    sin = torch.sin(th).repeat_interleave(2, dim=-1).contiguous()
    w = lambda: torch.randn(_D, device=device, dtype=dt) * 0.1 + 1.0  # noqa: E731
    return (mk(s_txt), mk(s_img), cos, sin, w(), w(), w(), w())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qwen_image_fact_cute_qkv_norm_rope_pack_parity():
    dev = "cuda"
    args = _inputs(51, 4096, dev)
    ref = _fact_cute_qkv_norm_rope_pack_ref(*args, _D, _EPS)
    out = fact_cute_qkv_norm_rope_pack(*args, _D, _EPS)
    assert out.shape == ref.shape == (1, 51 + 4096, 3 * _HD)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qwen_image_fact_cute_qkv_norm_rope_pack_cudagraph():
    dev = "cuda"
    args = _inputs(51, 4096, dev)
    fact_cute_qkv_norm_rope_pack(*args, _D, _EPS)  # warm (compile)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = fact_cute_qkv_norm_rope_pack(*args, _D, _EPS)
    g.replay()
    torch.cuda.synchronize()
