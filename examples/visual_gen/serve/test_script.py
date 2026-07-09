#!/usr/bin/env python
"""Test script for image generation endpoints.

Tests:
- POST /v1/images/generations - Generate images from text

Examples:
  # FLUX.2 (default)
  python send_request.py

  # FLUX.1
  python send_request.py --model flux1

  # Custom server and prompt
  python send_request.py --base-url http://your-server:8000/v1 --prompt "A sunset"
"""

import argparse
import asyncio
import base64
import sys
import time

import openai


async def send_one_request(
    client: openai.AsyncOpenAI,
    request_id: int,
    model: str,
    prompt: str,
    n: int,
    size: str,
    quality: str,
    response_format: str,
    output_file: str,
):
    """Send a single image generation request and return timing info."""
    start_time = time.perf_counter()
    response = await client.images.generate(
        model=model,
        prompt=prompt,
        n=n,
        size=size,
        quality=quality,
        response_format=response_format,
        extra_body={"num_inference_steps": 20},
    )
    elapsed = time.perf_counter() - start_time

    for i, image in enumerate(response.data):
        if response_format == "b64_json":
            image_data = base64.b64decode(image.b64_json)
            stem = output_file.rsplit(".", 1)[0]
            output = f"{stem}_req{request_id}_{i}.png"
            with open(output, "wb") as f:
                f.write(image_data)

    print(f"   Request {request_id:>2d} completed in {elapsed:.3f}s")
    return request_id, elapsed


async def test_image_generation(
    base_url: str = "http://localhost:8000/v1",
    model: str = "flux2",
    prompt: str = "A lovely cat lying on a sofa",
    n: int = 1,
    size: str = "1024x1024",
    quality: str = "standard",
    response_format: str = "b64_json",
    output_file: str = "output_generation.png",
    num_requests: int = 10,
):
    """Test image generation endpoint with parallel async requests."""
    print("=" * 80)
    print("Testing Image Generation API (POST /v1/images/generations)")
    print("=" * 80)

    client = openai.AsyncOpenAI(base_url=base_url, api_key="tensorrt_llm")

    print(f"\n   Model: {model}")
    print(f"   Prompt: {prompt}")
    print(f"   Size: {size}")
    print(f"   Quality: {quality}")
    print(f"   Parallel requests: {num_requests}")

    try:
        e2e_start = time.perf_counter()

        tasks = [
            send_one_request(
                client, i, model, prompt, n, size, quality,
                response_format, output_file,
            )
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

        e2e_elapsed = time.perf_counter() - e2e_start

        results = sorted(results, key=lambda x: x[0])
        latencies = [r[1] for r in results]

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"   Total requests:      {num_requests}")
        print(f"   End-to-end time:     {e2e_elapsed:.3f}s")
        print(f"   Avg latency/request: {sum(latencies) / len(latencies):.3f}s")
        print(f"   Min latency:         {min(latencies):.3f}s")
        print(f"   Max latency:         {max(latencies):.3f}s")
        print(f"   Throughput:          {num_requests / e2e_elapsed:.2f} req/s")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test image generation API (FLUX.1 / FLUX.2)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux2",
        help="Model name (e.g., flux1, flux2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A lovely cat lying on a sofa",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Image size in WxH format (e.g., 512x512, 1024x1024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_generation.png",
        help="Output image file path",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of parallel requests to send",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OpenAI SDK - Image Generation Tests")
    print("=" * 80)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print()

    success = asyncio.run(
        test_image_generation(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            size=args.size,
            output_file=args.output,
            num_requests=args.num_requests,
        )
    )

    sys.exit(0 if success else 1)