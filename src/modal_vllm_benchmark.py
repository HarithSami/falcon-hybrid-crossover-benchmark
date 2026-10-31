"""
Modal Benchmark for Open-Source vLLM Energy, Roofline MFU, and Hybrid MBU Profiler
==================================================================================
Runs `src/vllm_energy_mfu_profiler.py` on isolated Modal L4 GPUs comparing:
  1. vLLM Engine (CUDA Graphs + PagedAttention / Hybrid Mamba-Attention scheduler)
  2. Hugging Face Eager PyTorch Engine (Baseline)
across TII's 7B Triad (`Falcon3-7B-Instruct`, `Falcon3-Mamba-7B-Instruct`, `Falcon-H1-7B-Instruct`)
to quantify how CUDA Graphs eliminate sequential layer-launch overhead and boost
Model FLOPs Utilization (MFU %), Model Bandwidth Utilization (MBU %), and Joules/token.

Usage:
  python src/run_modal_cli.py run src/modal_vllm_benchmark.py
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "tii-falcon-vllm-energy-mfu"
VOLUME_NAME = "tii-falcon-cache"
CACHE_DIR = "/cache"
HF_CACHE_DIR = f"{CACHE_DIR}/hf"
RESULTS_DIR = f"{CACHE_DIR}/results"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
local_hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "vllm>=0.8.5",
        "nvidia-ml-py>=12.560.30",
        "huggingface_hub>=0.27.0",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_LOG_DIR": "/tmp/hf_xet_logs",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .run_commands(
        "find /usr/local/lib/python3.11/site-packages/flashinfer -name '*.py' -exec sed -i 's/--compress-mode=size//g' {} +"
    )
    .add_local_file(
        "src/vllm_energy_mfu_profiler.py",
        remote_path="/root/vllm_energy_mfu_profiler.py",
    )
)

app = modal.App(APP_NAME, image=vllm_image, secrets=[hf_secret, local_hf_secret])

SEVEN_B_TRIAD = [
    ("Pure Transformer (7B)", "tiiuae/Falcon3-7B-Instruct", 14220.4),
    ("Pure SSM / Mamba (7B)", "tiiuae/Falcon3-Mamba-7B-Instruct", 13896.7),
    ("Parallel Hybrid (7B)", "tiiuae/Falcon-H1-7B-Instruct", 14477.3),
]


@app.function(
    gpu="L4",
    cpu=4.0,
    memory=32768,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def run_vllm_roofline_profile(
    arch_label: str,
    model_id: str,
    weights_vram_mb: float,
    context_lengths: list[int] = [512, 2048, 8192],
    decode_tokens: int = 128,
) -> list[dict[str, Any]]:
    import gc
    import sys

    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm_energy_mfu_profiler import (
        VLLMEnergyMFUProfiler,
        build_architecture_profile,
    )

    try:
        volume.reload()
    except Exception:
        pass

    profiler = VLLMEnergyMFUProfiler(device_index=0, poll_interval_s=0.02)
    idle_w = profiler.calibrate_idle_floor(duration_s=2.0)
    print(f"[{arch_label} | vLLM] GPU: {profiler.gpu_name} | Clean Idle Floor: {idle_w:.2f} W")

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    weights_bytes = int(weights_vram_mb * (1024**2))
    arch_profile = build_architecture_profile(
        model_id=model_id,
        hf_config=hf_config,
        weights_vram_bytes=weights_bytes,
        precision="bf16",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"[{arch_label} | vLLM] Initializing vLLM engine for '{model_id}' (CUDA Graphs enabled)...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(context_lengths) + decode_tokens + 64,
        max_num_seqs=8,
        max_num_batched_tokens=max(context_lengths) + decode_tokens + 64,
        gpu_memory_utilization=0.92,
        enforce_eager=False,  # Capture CUDA Graphs
    )

    # Warmup pass
    warmup_params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    _ = llm.generate(["Hello world " * 16], warmup_params, use_tqdm=False)
    torch.cuda.synchronize()

    records: list[dict[str, Any]] = []
    for ctx_len in context_lengths:
        # Build distinct exact token-length prompts for Step 1 (TTFT) and Step 2 (Decode)
        # so vLLM v1 Automatic Prefix Caching (APC) does not skip prefill in Step 2.
        synth_ids_ttft = [100 + (i % 1000) for i in range(ctx_len)]
        synth_ids_decode = [1200 + (i % 1000) for i in range(ctx_len)]

        # Step 1: Measure 1-token prefill (TTFT)
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)
        torch.cuda.synchronize()
        t_p0 = time.perf_counter()
        _ = llm.generate([{"prompt_token_ids": synth_ids_ttft}], prefill_params, use_tqdm=False)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_p0) * 1000.0

        # Step 2: Measure full prefill + decode under NVML Energy & MFU/MBU Profiler
        gen_params = SamplingParams(
            temperature=0.0,
            max_tokens=decode_tokens,
            min_tokens=decode_tokens,
            ignore_eos=True,
        )
        torch.cuda.synchronize()
        profiler.start()
        _ = llm.generate([{"prompt_token_ids": synth_ids_decode}], gen_params, use_tqdm=False)
        torch.cuda.synchronize()
        raw_stats = profiler.stop_raw()

        # Isolate pure decode window by subtracting TTFT
        decode_only_s = max(raw_stats["duration_s"] - (ttft_ms / 1000.0), 1e-4)
        decode_stats = {
            "duration_s": decode_only_s,
            "mean_watts": raw_stats["mean_watts"],
            "peak_watts": raw_stats["peak_watts"],
            "total_joules": raw_stats["mean_watts"] * decode_only_s,
        }

        report = profiler.compute_report(
            raw_stats=decode_stats,
            arch=arch_profile,
            weights_vram_bytes=weights_bytes,
            engine="vllm_cudagraph",
            precision="bf16",
            context_length=ctx_len,
            decode_tokens=decode_tokens - 1,
            batch_size=1,
        )
        row = report.to_dict()
        row["arch_label"] = arch_label
        row["ttft_ms"] = round(ttft_ms, 2)
        print(
            f"  [vLLM OK] ctx={ctx_len:5d} | TTFT={row['ttft_ms']:6.1f}ms | "
            f"Decode={row['decode_tok_per_s']:6.1f} tok/s | "
            f"Energy={row['joules_per_token_total']:.4f} J/tok | "
            f"MFU={row['mfu_pct']:5.2f}% | MBU={row['mbu_pct']:5.1f}% ({row['achieved_hbm_bw_gbps']:.1f} GB/s)"
        )
        records.append(row)

    profiler.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return records


@app.local_entrypoint()
def main():
    from vllm_energy_mfu_profiler import (
        ArchitectureProfile,
        GPUSpec,
        VLLMEnergyMFUProfiler,
    )

    print("=" * 78)
    print("TII FALCON 7B | vLLM CUDA-Graph Energy, Roofline MFU & Hybrid MBU Benchmark")
    print("=" * 78)

    # First, retroactively compute Roofline MFU (%) and Hybrid MBU (%) for our existing
    # Hugging Face 7B crossover & quant-ablation records so we can compare HF vs. vLLM!
    crossover_path = Path("results/benchmark_crossover.json")
    hf_enriched: list[dict[str, Any]] = []
    if crossover_path.exists():
        with open(crossover_path, "r", encoding="utf-8") as f:
            hf_rows = json.load(f)

        l4_spec = GPUSpec("NVIDIA L4", peak_bf16_tflops=121.0, peak_hbm_bw_gbps=300.0, tdp_watts=72.0)
        print("\n[Stage 1/2] Enriching existing 7B Crossover dataset with Roofline MFU% & MBU%...")
        print(
            "| Architecture | Context | Engine | Decode (tok/s) | Energy (J/tok) | Achieved BW (GB/s) | MBU (%) | MFU (%) |"
        )
        print(
            "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |"
        )
        for r in hf_rows:
            if r.get("status") != "ok":
                continue
            w_bytes = int(r["weights_vram_mb"] * (1024**2))
            kv_bytes = int(r["kv_or_state_growth_mb"] * (1024**2))
            tps = r["decode_tok_per_s"]
            # Total HBM bytes streamed per generated token ≈ static weights + active KV/SSM state
            total_bytes_per_tok = w_bytes + kv_bytes
            bw_gbps = (total_bytes_per_tok * tps) / 1e9
            mbu_pct = (bw_gbps / l4_spec.peak_hbm_bw_gbps) * 100.0
            est_params = w_bytes / 2.0
            flops_per_tok = 2.0 * est_params
            tflops = (flops_per_tok * tps) / 1e12
            mfu_pct = (tflops / l4_spec.peak_bf16_tflops) * 100.0

            r["achieved_hbm_bw_gbps"] = round(bw_gbps, 2)
            r["mbu_pct"] = round(mbu_pct, 2)
            r["achieved_tflops"] = round(tflops, 2)
            r["mfu_pct"] = round(mfu_pct, 2)
            hf_enriched.append(r)
            print(
                f"| {r['arch_label']} | {r['context_length']} | `hf_eager` | "
                f"{tps:.1f} | **{r['joules_per_token_total']:.4f}** | "
                f"{bw_gbps:.1f} GB/s | **{mbu_pct:.1f}%** | {mfu_pct:.2f}% |"
            )

    print("\n[Stage 2/2] Running vLLM CUDA-Graph Roofline & Energy Benchmarks on Modal L4...")
    vllm_results: list[dict[str, Any]] = []
    for arch_label, model_id, w_mb in SEVEN_B_TRIAD:
        try:
            rows = run_vllm_roofline_profile.remote(
                arch_label=arch_label,
                model_id=model_id,
                weights_vram_mb=w_mb,
                context_lengths=[512, 2048, 8192],
                decode_tokens=128,
            )
            vllm_results.extend(rows)
        except Exception as e:
            print(f"  [Warning] vLLM run for {arch_label} ({model_id}) encountered: {e}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_vllm_mfu.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"hf_roofline": hf_enriched, "vllm_cudagraph": vllm_results}, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Saved Roofline MFU/MBU & vLLM Telemetry Report to: {out_file}")
    print("=" * 78)
