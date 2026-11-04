"""
Modal Benchmark for Open-Source vLLM Energy, Roofline MFU, and Hybrid MBU Profiler
==================================================================================
Runs `src/vllm_energy_mfu_profiler.py` on isolated Modal L4 GPUs (24GB GDDR6, 300 GB/s)
with `num_trials = 5` (mean ± std), `enable_prefix_caching=False`, exact package & model
revision logging, and raw 50 Hz NVML power trace export.

Compares:
  1. vLLM Engine (CUDA Graphs + PagedAttention / Hybrid Mamba-Attention scheduler)
  2. Hugging Face Eager PyTorch Engine (Baseline)
across TII's 7B Triad (`Falcon3-7B-Instruct`, `Falcon3-Mamba-7B-Instruct`, `Falcon-H1-7B-Instruct`).

Usage:
  python src/run_modal_cli.py run src/modal_vllm_benchmark.py
"""

import json
import math
import os
import statistics
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


def _mean_std(vals: list[float], ndigits: int = 4) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return round(m, ndigits), round(s, ndigits)


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
    num_trials: int = 5,
) -> dict[str, Any]:
    import gc
    import sys

    sys.path.insert(0, "/root")
    import torch
    import transformers
    import vllm
    from huggingface_hub import HfApi
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
    idle_w = profiler.calibrate_idle_floor(duration_s=2.5)
    print(
        f"[{arch_label} | vLLM] GPU: {profiler.gpu_name} ({profiler.gpu_spec.memory_type}) | "
        f"Driver: {profiler.driver_version} | Pre-Load Idle Floor: {idle_w:.2f} W"
    )

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_revision = getattr(hf_config, "_commit_hash", None) or "main"
    try:
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(model_id)
        if info.sha:
            model_revision = info.sha
    except Exception:
        pass

    weights_bytes = int(weights_vram_mb * (1024**2))
    arch_profile = build_architecture_profile(
        model_id=model_id,
        hf_config=hf_config,
        weights_vram_bytes=weights_bytes,
        precision="bf16",
    )

    env_metadata = {
        "model_revision": model_revision,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda or "12.4",
        "nvidia_driver_version": profiler.driver_version,
        "gpu_name": profiler.gpu_name,
        "memory_type": profiler.gpu_spec.memory_type,
        "peak_mem_bw_gbps": profiler.gpu_spec.peak_mem_bw_gbps,
        "peak_bf16_tflops": profiler.gpu_spec.peak_bf16_tflops,
    }

    print(
        f"[{arch_label} | vLLM] Initializing vLLM engine for '{model_id}' "
        f"(rev={model_revision[:8]}, CUDA Graphs=True, PrefixCaching=False)..."
    )
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": max(context_lengths) + decode_tokens + 64,
        "max_num_seqs": 8,
        "max_num_batched_tokens": max(context_lengths) + decode_tokens + 64,
        "gpu_memory_utilization": 0.92,
        "enforce_eager": False,
        "enable_prefix_caching": False,
    }
    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("enable_prefix_caching", None)
        llm = LLM(**llm_kwargs)

    # Warmup pass
    warmup_params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    _ = llm.generate([{"prompt_token_ids": [100 + i for i in range(128)]}], warmup_params, use_tqdm=False)
    torch.cuda.synchronize()

    # Also measure post-load idle power floor (when CUDA context & weights reside on GPU)
    profiler.start()
    time.sleep(1.5)
    post_load_idle_stats = profiler.stop_raw()
    post_load_idle_w = round(post_load_idle_stats["mean_watts"], 2)

    records: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []

    for ctx_len in context_lengths:
        ttft_trials: list[float] = []
        tps_trials: list[float] = []
        tpot_trials: list[float] = []
        j_total_trials: list[float] = []
        j_dyn_trials: list[float] = []
        bw_trials: list[float] = []
        mbu_trials: list[float] = []
        mfu_trials: list[float] = []
        tflops_trials: list[float] = []
        mean_w_trials: list[float] = []
        peak_w_trials: list[float] = []
        dur_trials: list[float] = []

        for trial in range(num_trials):
            # Distinct non-repeating token sequences for every trial & phase so prefix cache can never match
            base_ttft = 200 + (trial * 1500)
            base_dec = 10000 + (trial * 1500)
            synth_ids_ttft = [base_ttft + ((i * 7) % 8000) for i in range(ctx_len)]
            synth_ids_decode = [base_dec + ((i * 13) % 8000) for i in range(ctx_len)]

            # Step 1: Measure 1-token prefill (TTFT)
            prefill_params = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)
            torch.cuda.synchronize()
            t_p0 = time.perf_counter()
            _ = llm.generate([{"prompt_token_ids": synth_ids_ttft}], prefill_params, use_tqdm=False)
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_p0) * 1000.0
            ttft_trials.append(ttft_ms)

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

            # Isolate pure decode window by subtracting Step 1 cold TTFT
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

            tps_trials.append(report.decode_tok_per_s)
            tpot_trials.append(report.tpot_ms)
            j_total_trials.append(report.joules_per_token_total)
            j_dyn_trials.append(report.joules_per_token_dynamic)
            bw_trials.append(report.achieved_mem_bw_gbps)
            mbu_trials.append(report.mbu_pct)
            mfu_trials.append(report.mfu_pct)
            tflops_trials.append(report.achieved_tflops)
            mean_w_trials.append(report.mean_watts)
            peak_w_trials.append(report.peak_watts)
            dur_trials.append(report.duration_s)

            if trial == 0:
                raw_traces.append(
                    {
                        "arch_label": arch_label,
                        "model_id": model_id,
                        "engine": "vllm_cudagraph",
                        "context_length": ctx_len,
                        "poll_interval_s": raw_stats["poll_interval_s"],
                        "pre_load_idle_watts": round(idle_w, 2),
                        "post_load_idle_watts": post_load_idle_w,
                        "samples_rel_time_s": raw_stats["samples_rel_time_s"],
                        "samples_watts": raw_stats["samples_watts"],
                    }
                )

        ttft_m, ttft_s = _mean_std(ttft_trials, 2)
        tps_m, tps_s = _mean_std(tps_trials, 2)
        tpot_m, tpot_s = _mean_std(tpot_trials, 2)
        jt_m, jt_s = _mean_std(j_total_trials, 4)
        jd_m, jd_s = _mean_std(j_dyn_trials, 4)
        bw_m, bw_s = _mean_std(bw_trials, 2)
        mbu_m, mbu_s = _mean_std(mbu_trials, 2)
        mfu_m, mfu_s = _mean_std(mfu_trials, 2)
        tflops_m, _ = _mean_std(tflops_trials, 2)
        mw_m, mw_s = _mean_std(mean_w_trials, 2)
        dur_m, _ = _mean_std(dur_trials, 4)

        row = {
            "arch_label": arch_label,
            "model_id": model_id,
            "model_revision": model_revision,
            "engine": "vllm_cudagraph",
            "precision": "bf16",
            "gpu_name": profiler.gpu_name,
            "memory_type": profiler.gpu_spec.memory_type,
            "context_length": ctx_len,
            "decode_tokens": decode_tokens - 1,
            "batch_size": 1,
            "num_trials": num_trials,
            "duration_s": dur_m,
            "ttft_ms": ttft_m,
            "ttft_std_ms": ttft_s,
            "decode_tok_per_s": tps_m,
            "decode_tok_per_s_std": tps_s,
            "tpot_ms": tpot_m,
            "tpot_std_ms": tpot_s,
            "idle_watts": round(idle_w, 2),
            "post_load_idle_watts": post_load_idle_w,
            "mean_watts": mw_m,
            "mean_watts_std": mw_s,
            "peak_watts": round(max(peak_w_trials), 2),
            "total_joules": round(mw_m * dur_m, 3),
            "joules_per_token_total": jt_m,
            "joules_per_token_total_std": jt_s,
            "joules_per_token_dynamic": jd_m,
            "joules_per_token_dynamic_std": jd_s,
            "achieved_tflops": tflops_m,
            "mfu_pct": mfu_m,
            "mfu_pct_std": mfu_s,
            "achieved_mem_bw_gbps": bw_m,
            "achieved_hbm_bw_gbps": bw_m,
            "achieved_mem_bw_gbps_std": bw_s,
            "mbu_pct": mbu_m,
            "mbu_pct_std": mbu_s,
            "analytical_persistent_state_mb": round(arch_profile.analytical_persistent_state_mb(ctx_len), 1),
            "environment": env_metadata,
        }
        print(
            f"  [vLLM OK | n={num_trials}] ctx={ctx_len:5d} | "
            f"TTFT={ttft_m:6.1f}±{ttft_s:4.1f}ms | "
            f"Decode={tps_m:5.2f}±{tps_s:4.2f} tok/s | "
            f"Energy={jt_m:.4f}±{jt_s:.4f} J/tok | "
            f"MFU={mfu_m:4.2f}% | MBU={mbu_m:5.1f}±{mbu_s:3.1f}% ({bw_m:.1f} GB/s)"
        )
        records.append(row)

    profiler.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {"records": records, "raw_traces": raw_traces}


@app.local_entrypoint()
def main():
    from vllm_energy_mfu_profiler import ArchitectureProfile, GPUSpec

    print("=" * 82)
    print("TII FALCON 7B | vLLM CUDA-Graph Energy, Roofline MFU & Hybrid MBU (5-Trial Suite)")
    print("=" * 82)

    print("\n[Stage 1/2] Launching parallel 5-trial vLLM CUDA-Graph benchmarks across 3 L4 GPUs...")
    starmap_args = [
        (arch_label, model_id, w_mb, [512, 2048, 8192], 128, 5)
        for arch_label, model_id, w_mb in SEVEN_B_TRIAD
    ]
    vllm_outputs = list(run_vllm_roofline_profile.starmap(starmap_args))

    vllm_results: list[dict[str, Any]] = []
    all_raw_traces: list[dict[str, Any]] = []
    for out in vllm_outputs:
        vllm_results.extend(out["records"])
        all_raw_traces.extend(out["raw_traces"])

    # Stage 2: Enrich existing HF eager crossover dataset using exact analytical KV + SSM state bytes
    crossover_path = Path("results/benchmark_crossover.json")
    hf_enriched: list[dict[str, Any]] = []
    if crossover_path.exists():
        with open(crossover_path, "r", encoding="utf-8") as f:
            hf_rows = json.load(f)

        l4_spec = GPUSpec("NVIDIA L4", memory_type="GDDR6", peak_bf16_tflops=121.0, peak_mem_bw_gbps=300.0, tdp_watts=72.0)
        # Analytical profiles for the 7B triad
        analytical_profiles = {
            "tiiuae/Falcon3-7B-Instruct": ArchitectureProfile(
                model_id="tiiuae/Falcon3-7B-Instruct",
                arch_type="transformer",
                num_params=7_450_000_000,
                bytes_per_param=2.0,
                num_layers=28,
                num_kv_heads=4,
                head_dim=256,
                ssm_state_bytes=0,
            ),
            "tiiuae/Falcon3-Mamba-7B-Instruct": ArchitectureProfile(
                model_id="tiiuae/Falcon3-Mamba-7B-Instruct",
                arch_type="mamba",
                num_params=7_280_000_000,
                bytes_per_param=2.0,
                num_layers=64,
                num_kv_heads=0,
                head_dim=0,
                ssm_state_bytes=64 * 8192 * (4 + 16) * 2,
            ),
            "tiiuae/Falcon-H1-7B-Instruct": ArchitectureProfile(
                model_id="tiiuae/Falcon-H1-7B-Instruct",
                arch_type="hybrid",
                num_params=7_590_000_000,
                bytes_per_param=2.0,
                num_layers=44,
                num_kv_heads=4,
                head_dim=128,
                ssm_state_bytes=44 * 64 * 64 * (4 + 128) * 2,
            ),
        }

        print("\n[Stage 2/2] Computing analytical GDDR6 Roofline MFU% & MBU% for HF Eager records...")
        for r in hf_rows:
            if r.get("status") != "ok":
                continue
            mid = r["model_id"]
            prof = analytical_profiles[mid]
            w_bytes = int(r["weights_vram_mb"] * (1024**2))
            avg_seq_len = r["context_length"] + (r.get("decode_tokens", 128) // 2)
            kv_bytes = prof.analytical_kv_cache_bytes(avg_seq_len, batch_size=1)
            ssm_bytes = 2 * prof.ssm_state_bytes
            total_bytes_per_tok = w_bytes + kv_bytes + ssm_bytes

            tps = r["decode_tok_per_s"]
            tps_std = r.get("decode_tok_per_s_std", 0.0)
            bw_gbps = (total_bytes_per_tok * tps) / 1e9
            bw_std = (total_bytes_per_tok * tps_std) / 1e9
            mbu_pct = (bw_gbps / l4_spec.peak_mem_bw_gbps) * 100.0
            mbu_std = (bw_std / l4_spec.peak_mem_bw_gbps) * 100.0

            attn_flops = 4.0 * prof.num_layers * prof.num_kv_heads * prof.head_dim * avg_seq_len
            flops_per_tok = (2.0 * prof.num_params) + attn_flops
            tflops = (flops_per_tok * tps) / 1e12
            mfu_pct = (tflops / l4_spec.peak_bf16_tflops) * 100.0

            r["memory_type"] = "GDDR6"
            r["achieved_mem_bw_gbps"] = round(bw_gbps, 2)
            r["achieved_hbm_bw_gbps"] = round(bw_gbps, 2)
            r["achieved_mem_bw_gbps_std"] = round(bw_std, 2)
            r["mbu_pct"] = round(mbu_pct, 2)
            r["mbu_pct_std"] = round(mbu_std, 2)
            r["achieved_tflops"] = round(tflops, 2)
            r["mfu_pct"] = round(mfu_pct, 2)
            r["analytical_persistent_state_mb"] = round(prof.analytical_persistent_state_mb(r["context_length"]), 1)
            hf_enriched.append(r)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_vllm_mfu.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"hf_roofline": hf_enriched, "vllm_cudagraph": vllm_results}, f, indent=2)

    traces_file = out_dir / "raw_50hz_vllm_traces.json"
    with open(traces_file, "w", encoding="utf-8") as f:
        json.dump(all_raw_traces, f, indent=2)

    print("\n" + "=" * 82)
    print(f"Saved 5-Trial Roofline MFU/MBU Report to: {out_file}")
    print(f"Saved Raw 50 Hz NVML Power Traces to:     {traces_file}")
    print("=" * 82)
