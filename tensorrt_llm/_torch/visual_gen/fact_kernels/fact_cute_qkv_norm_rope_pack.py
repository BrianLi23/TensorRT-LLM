# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACT fused CuteDSL superset: sequence-cat + per-head QK-norm + interleaved
RoPE + [q|k|v] packing for Qwen-Image joint attention (stacks on qkv_merge).

Production (post-qkv_merge) builds the joint packed QKV with a sequence
``torch.cat([txt_qkv, img_qkv])`` (inductor ``triton_poi_fused_cat_view``) then
runs the compiled ``torch.ops.trtllm.fused_dit_qk_norm_rope`` in place. This
kernel does both in ONE CuteDSL launch: reads the two packed inputs once,
applies per-head RMSNorm (fp32) + interleaved complex RoPE to q/k, copies v, and
writes the joint ``[txt|img] x [q|k|v]`` buffer directly — eliminating the cat's
write and the fused-op's re-read of the wide packed buffer.

Mapping: one warp per (token, head); head_dim=128 = 32 lanes x vec(4). RMS
sum-of-squares reduced across the warp in fp32; RoPE in fp32; single bf16 round
on store. v is a bf16 passthrough. `_fact_cute_qkv_norm_rope_pack_ref` is the
pure-torch allclose anchor. Authored/validated via the kernel-cute-writing
skill (verify_kernel.py PASS @ 2e-2; CUDA-graph capturable).
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

VEC = 4
HEAD_DIM = 128
HDV = HEAD_DIM // VEC   # 32 lanes per head == warp size
WARPS_PER_BLOCK = 8
THREADS = WARPS_PER_BLOCK * 32

torch2cute = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


class _CudaGraphDLPack:
    """dlpack wrapper forcing stream=-1 so from_dlpack does no per-call stream
    sync (required for CUDA-graph capture)."""

    def __init__(self, t):
        self._t = t

    def __dlpack__(self, stream=None):
        return self._t.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._t.__dlpack_device__()


def _process(mSrc, mOut, mCos, mSin, wq, wk,
             b, srow, t, h, lane,
             num_heads: cutlass.Constexpr, eps: cutlass.Constexpr):
    HDvec = num_heads * HDV
    cv_q = h * HDV + lane
    cv_k = HDvec + h * HDV + lane
    cv_v = 2 * HDvec + h * HDV + lane

    inv_n = cutlass.Float32(1.0) / cutlass.Float32(HEAD_DIM)

    cvec = mCos[(t, lane, None)].load().to(cutlass.Float32)
    svec = mSin[(t, lane, None)].load().to(cutlass.Float32)
    c0 = cvec[0]
    s0 = svec[0]
    c1 = cvec[2]
    s1 = svec[2]

    # ---- q ----
    qv = mSrc[(b, srow, cv_q, None)].load().to(cutlass.Float32)
    wqv = wq[(lane, None)].load().to(cutlass.Float32)
    ss = qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2] + qv[3] * qv[3]
    ss = cute.arch.warp_reduction_sum(ss)
    inv = cute.math.rsqrt(ss * inv_n + cutlass.Float32(eps))
    xn0 = qv[0] * inv * wqv[0]
    xn1 = qv[1] * inv * wqv[1]
    xn2 = qv[2] * inv * wqv[2]
    xn3 = qv[3] * inv * wqv[3]
    fq = cute.make_fragment(VEC, cutlass.BFloat16)
    fq[0] = (xn0 * c0 - xn1 * s0).to(cutlass.BFloat16)
    fq[1] = (xn1 * c0 + xn0 * s0).to(cutlass.BFloat16)
    fq[2] = (xn2 * c1 - xn3 * s1).to(cutlass.BFloat16)
    fq[3] = (xn3 * c1 + xn2 * s1).to(cutlass.BFloat16)
    cute.autovec_copy(fq, mOut[(b, t, cv_q, None)])

    # ---- k ----
    kv = mSrc[(b, srow, cv_k, None)].load().to(cutlass.Float32)
    wkv = wk[(lane, None)].load().to(cutlass.Float32)
    ssk = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2] + kv[3] * kv[3]
    ssk = cute.arch.warp_reduction_sum(ssk)
    invk = cute.math.rsqrt(ssk * inv_n + cutlass.Float32(eps))
    kn0 = kv[0] * invk * wkv[0]
    kn1 = kv[1] * invk * wkv[1]
    kn2 = kv[2] * invk * wkv[2]
    kn3 = kv[3] * invk * wkv[3]
    fk = cute.make_fragment(VEC, cutlass.BFloat16)
    fk[0] = (kn0 * c0 - kn1 * s0).to(cutlass.BFloat16)
    fk[1] = (kn1 * c0 + kn0 * s0).to(cutlass.BFloat16)
    fk[2] = (kn2 * c1 - kn3 * s1).to(cutlass.BFloat16)
    fk[3] = (kn3 * c1 + kn2 * s1).to(cutlass.BFloat16)
    cute.autovec_copy(fk, mOut[(b, t, cv_k, None)])

    # ---- v : passthrough ----
    fv = cute.make_fragment(VEC, cutlass.BFloat16)
    cute.autovec_copy(mSrc[(b, srow, cv_v, None)], fv)
    cute.autovec_copy(fv, mOut[(b, t, cv_v, None)])


@cute.kernel
def _fused_kernel(
    mTxt: cute.Tensor, mImg: cute.Tensor, mOut: cute.Tensor,
    mCos: cute.Tensor, mSin: cute.Tensor,
    m_nqw: cute.Tensor, m_nkw: cute.Tensor,
    m_naqw: cute.Tensor, m_nakw: cute.Tensor,
    B: cutlass.Constexpr, S_total: cutlass.Constexpr, S_txt: cutlass.Constexpr,
    num_heads: cutlass.Constexpr, eps: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    warp_in_block = tidx // 32
    lane = tidx % 32
    unit = bidx * WARPS_PER_BLOCK + warp_in_block

    U = B * S_total * num_heads
    if cutlass.dynamic_expr(unit < U):
        grow = unit // num_heads
        h = unit % num_heads
        b = grow // S_total
        t = grow % S_total

        if cutlass.dynamic_expr(t < S_txt):
            _process(mTxt, mOut, mCos, mSin, m_naqw, m_nakw,
                     b, t, t, h, lane, num_heads, eps)
        else:
            _process(mImg, mOut, mCos, mSin, m_nqw, m_nkw,
                     b, t - S_txt, t, h, lane, num_heads, eps)


@cute.jit
def _fused_host(
    mTxt: cute.Tensor, mImg: cute.Tensor, mOut: cute.Tensor,
    mCos: cute.Tensor, mSin: cute.Tensor,
    m_nqw: cute.Tensor, m_nkw: cute.Tensor,
    m_naqw: cute.Tensor, m_nakw: cute.Tensor,
    B: cutlass.Constexpr, S_total: cutlass.Constexpr, S_txt: cutlass.Constexpr,
    num_heads: cutlass.Constexpr, eps: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    U = B * S_total * num_heads
    num_blocks = (U + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
    _fused_kernel(
        mTxt, mImg, mOut, mCos, mSin, m_nqw, m_nkw, m_naqw, m_nakw,
        B, S_total, S_txt, num_heads, eps,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


# One compiled kernel per (shapes, dtype). Output is allocated fresh per call
# (caching-allocator gives static addresses under CUDA-graph capture), so the
# op stays functional for torch.compile — only the compiled kernel is cached.
_compile_cache = {}


def _get_compiled(txt_qkv, img_qkv, num_heads, S_total, S_txt, eps):
    B = txt_qkv.shape[0]
    key = (B, S_txt, img_qkv.shape[1], num_heads, txt_qkv.dtype)
    hit = _compile_cache.get(key)
    if hit is not None:
        return hit
    D3 = txt_qkv.shape[2]
    D3v = D3 // VEC

    def rv(shape, dtype):
        return from_dlpack(torch.empty(shape, dtype=dtype, device="cuda"),
                           assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        _fused_host,
        rv((B, S_txt, D3v, VEC), txt_qkv.dtype),
        rv((B, img_qkv.shape[1], D3v, VEC), img_qkv.dtype),
        rv((B, S_total, D3v, VEC), txt_qkv.dtype),
        rv((S_total, HDV, VEC), torch.float32),
        rv((S_total, HDV, VEC), torch.float32),
        rv((HDV, VEC), txt_qkv.dtype), rv((HDV, VEC), txt_qkv.dtype),
        rv((HDV, VEC), txt_qkv.dtype), rv((HDV, VEC), txt_qkv.dtype),
        B, S_total, S_txt, num_heads, eps, stream,
    )
    _compile_cache[key] = compiled
    return compiled


@torch.library.custom_op("trtllm::fact_cute_qkv_norm_rope_pack", mutates_args=())
def fact_cute_qkv_norm_rope_pack(
    txt_qkv: torch.Tensor, img_qkv: torch.Tensor,
    freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
    norm_q_weight: torch.Tensor, norm_k_weight: torch.Tensor,
    norm_added_q_weight: torch.Tensor, norm_added_k_weight: torch.Tensor,
    head_dim: int, eps: float,
) -> torch.Tensor:
    """Joint packed QKV [B, S_txt+S_img, 3*HD] ([txt|img] rows, [q|k|v] cols)
    with per-head QK-norm + interleaved RoPE applied to q/k. head_dim must be
    128 (kernel is specialized)."""
    B, S_txt, D3 = txt_qkv.shape
    S_img = img_qkv.shape[1]
    S_total = S_txt + S_img
    num_heads = (D3 // 3) // head_dim
    compiled = _get_compiled(txt_qkv, img_qkv, num_heads, S_total, S_txt, eps)

    D3v = D3 // VEC
    out = torch.empty((B, S_total, D3), dtype=txt_qkv.dtype, device=txt_qkv.device)

    def w(t, shape):
        # detach: in-model inputs (norm weights) are nn.Parameters with
        # requires_grad=True, which from_dlpack refuses to export. detach shares
        # storage (no copy). Inference tensors are contiguous so .view is safe.
        return from_dlpack(_CudaGraphDLPack(t.detach().view(shape)), assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        w(txt_qkv, (B, S_txt, D3v, VEC)),
        w(img_qkv, (B, S_img, D3v, VEC)),
        w(out, (B, S_total, D3v, VEC)),
        w(freqs_cos, (S_total, HDV, VEC)),
        w(freqs_sin, (S_total, HDV, VEC)),
        w(norm_q_weight, (HDV, VEC)), w(norm_k_weight, (HDV, VEC)),
        w(norm_added_q_weight, (HDV, VEC)), w(norm_added_k_weight, (HDV, VEC)),
        stream,
    )
    return out


@fact_cute_qkv_norm_rope_pack.register_fake
def _(txt_qkv, img_qkv, freqs_cos, freqs_sin,
      norm_q_weight, norm_k_weight, norm_added_q_weight, norm_added_k_weight,
      head_dim, eps):
    B, S_txt, D3 = txt_qkv.shape
    return txt_qkv.new_empty((B, S_txt + img_qkv.shape[1], D3))


def _fact_cute_qkv_norm_rope_pack_ref(
    txt_qkv, img_qkv, freqs_cos, freqs_sin,
    norm_q_weight, norm_k_weight, norm_added_q_weight, norm_added_k_weight,
    head_dim, eps,
) -> torch.Tensor:
    """Pure-torch reference (unfused per-head norm+rope, fp32, single bf16 round;
    joint [txt|img] x [q|k|v] packing). allclose anchor for the registered test."""
    B, S_txt, D3 = txt_qkv.shape
    HD = D3 // 3
    H = HD // head_dim

    def nr(x, wt, c, s):
        x = x.unflatten(-1, (H, head_dim)).float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * wt.float()
        xe, xo = x[..., 0::2], x[..., 1::2]
        ce = c[..., 0::2].float()[None, :, None, :]
        se = s[..., 0::2].float()[None, :, None, :]
        o = torch.empty_like(x)
        o[..., 0::2] = xe * ce - xo * se
        o[..., 1::2] = xo * ce + xe * se
        return o.flatten(2)

    def proc(qkv, c, s, wq, wk):
        q, k, v = qkv[..., :HD], qkv[..., HD:2 * HD], qkv[..., 2 * HD:]
        return torch.cat([nr(q, wq, c, s).to(v.dtype), nr(k, wk, c, s).to(v.dtype), v], dim=-1)

    t = proc(txt_qkv, freqs_cos[:S_txt], freqs_sin[:S_txt], norm_added_q_weight, norm_added_k_weight)
    i = proc(img_qkv, freqs_cos[S_txt:], freqs_sin[S_txt:], norm_q_weight, norm_k_weight)
    return torch.cat([t, i], dim=1)
