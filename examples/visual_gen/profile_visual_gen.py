#!/usr/bin/env python3
"""Generic single-inference Torch-profiling runner for VisualGen models.

Same VisualGen API as the per-model example scripts, but model-agnostic
(--model) and with a --num_inference_steps knob so the torch.profiler trace
stays tractable. Torch profiling is enabled worker-side via env vars:

    TLLM_TORCH_PROFILE_VISUAL_GEN=generate   (full inference) | denoise
    TLLM_TORCH_PROFILE_VISUAL_GEN_OUT=/path/trace.json.gz
"""

import argparse
import os

from tensorrt_llm import VisualGen, VisualGenArgs


def main():
    p = argparse.ArgumentParser(description="VisualGen Torch-profile runner")
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--visual_gen_args", dest="visual_gen_args", default=None)
    p.add_argument("--prompt", default="A cat sitting on a windowsill, cinematic lighting, highly detailed")
    p.add_argument("--negative_prompt", default=None,
                   help="Negative prompt (enables true CFG for models like Qwen-Image; "
                        "match serving traffic, e.g. ' ').")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Guidance/CFG scale. Default = model default.")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=None, help="Output height (px). Default = model default.")
    p.add_argument("--width", type=int, default=None, help="Output width (px). Default = model default.")
    p.add_argument("--num_warmup", type=int, default=1,
                   help="Identical unprofiled-in-effect requests sent before the measured one. "
                        "CUDA graph keys include the prompt's token length, which the server-side "
                        "warmup (fixed 'warmup' prompt) cannot match, so the first request with a "
                        "given prompt pays torch.compile shakeout + graph capture. Each request "
                        "overwrites the trace file, so the final trace is the last (clean) request "
                        "— mirroring the client-side warmup of the serving benchmarks.")
    p.add_argument("--output_path", default="profile_output.png")
    args = p.parse_args()

    print(f"[profile] model={args.model}", flush=True)
    print(f"[profile] TLLM_TORCH_PROFILE_VISUAL_GEN={os.environ.get('TLLM_TORCH_PROFILE_VISUAL_GEN')!r}", flush=True)
    print(f"[profile] TLLM_TORCH_PROFILE_VISUAL_GEN_OUT={os.environ.get('TLLM_TORCH_PROFILE_VISUAL_GEN_OUT')!r}", flush=True)

    extra = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    print("[profile] Building VisualGen (load weights + warmup)...", flush=True)
    vg = VisualGen(model=args.model, args=extra)

    params = vg.default_params
    params.num_images_per_prompt = 1
    params.num_inference_steps = args.num_inference_steps
    if args.height is not None:
        params.height = args.height
    if args.width is not None:
        params.width = args.width
    if args.negative_prompt is not None:
        params.negative_prompt = args.negative_prompt
    if args.guidance_scale is not None:
        params.guidance_scale = args.guidance_scale
    print(f"[profile] steps={params.num_inference_steps} "
          f"h={getattr(params,'height',None)} w={getattr(params,'width',None)} "
          f"neg={args.negative_prompt!r} cfg={getattr(params,'guidance_scale',None)}", flush=True)

    for i in range(args.num_warmup):
        print(f"[profile] Client warmup request {i + 1}/{args.num_warmup} "
              "(same params; trace will be overwritten by the measured request)...", flush=True)
        vg.generate(inputs=args.prompt, params=params)

    print("[profile] Running single inference (profiled region)...", flush=True)
    out = vg.generate(inputs=args.prompt, params=params)
    print(f"[profile] Saved: {out.save(args.output_path)}", flush=True)
    print("[profile] DONE", flush=True)


if __name__ == "__main__":
    main()
