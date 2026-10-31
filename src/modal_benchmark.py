"""
TII 3-Architecture Crossover & Hybrid Quantization Benchmark on Modal
=====================================================================
Benchmarks TII's three sequence modeling paradigms on isolated datacenter GPUs:
  1. Pure Transformer : tiiuae/Falcon3-7B-Instruct (or Falcon3-1B-Instruct for smoke tests)
  2. Pure SSM (Mamba) : tiiuae/Falcon3-Mamba-7B-Instruct
  3. Parallel Hybrid  : tiiuae/Falcon-H1-7B-Instruct (or Falcon-H1-1.5B-Deep-Instruct)

Supported precision modes:
  - "bf16"             : Native BFloat16 baseline
  - "int4_uniform"     : Uniform 4-bit NormalFloat (NF4) quantization across all linear layers
  - "int4_mixed_ssm"   : Branch-Aware Hybrid Quantization (keeps Mamba SSM selective state
                         projections in BF16 while quantizing Attention & MLP blocks to INT4)

Usage (from local terminal, using 0% local GPU):
  # 1. Smoke test (1.5B hybrid + 1B transformer, ~2 mins, costs ~$0.03):
  python -m modal run src/modal_benchmark.py --mode smoke

  # 2. Full 7B 3-Architecture BF16 Crossover Sweep (Week 1):
  python -m modal run src/modal_benchmark.py --mode crossover

  # 3. Hybrid Quantization Degradation vs. Mixed-Precision Fix (Week 2-3):
  python -m modal run src/modal_benchmark.py --mode quant-ablation
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# 1. Modal App, Persistent Volume & Container Image
# ---------------------------------------------------------------------------
APP_NAME = "tii-falcon-crossover-benchmark"
VOLUME_NAME = "tii-falcon-cache"
CACHE_DIR = "/cache"
HF_CACHE_DIR = f"{CACHE_DIR}/hf"
RESULTS_DIR = f"{CACHE_DIR}/results"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Forward HF_TOKEN from both named Modal Secret ("huggingface-secret") and local Windows environment
hf_secret = modal.Secret.from_name("huggingface-secret")
local_hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])

# Clean Linux CUDA 12.4 devel container image with PyTorch 2.5.1, fused Mamba-2 / SSM
# CUDA kernels (causal-conv1d, mamba-ssm), FlashAttention-2, bitsandbytes, and NVML telemetry.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "ninja-build", "clang")
    .pip_install(
        "torch==2.5.1",
        "packaging",
        "wheel",
        "setuptools",
        "einops",
    )
    .pip_install(
        "torch==2.5.1",
        "causal-conv1d>=1.5.0",
        "mamba-ssm>=2.2.4",
        "flash-attn>=2.7.0",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51.0,<5.0.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "safetensors",
        "huggingface_hub>=0.27.0,<1.0.0",
        "nvidia-ml-py>=12.560.30",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_LOG_DIR": "/tmp/hf_xet_logs",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

app = modal.App(APP_NAME, image=image, secrets=[hf_secret, local_hf_secret])

# ---------------------------------------------------------------------------
# 2. Model Presets & Evaluation-Quality Calibration Corpus
# ---------------------------------------------------------------------------
SMOKE_MODELS = [
    ("Transformer (1B)", "tiiuae/Falcon3-1B-Instruct"),
    ("Hybrid Mamba-Attn (1.5B)", "tiiuae/Falcon-H1-1.5B-Deep-Instruct"),
]

SEVEN_B_TRIAD = [
    ("Pure Transformer (7B)", "tiiuae/Falcon3-7B-Instruct"),
    ("Pure SSM / Mamba (7B)", "tiiuae/Falcon3-Mamba-7B-Instruct"),
    ("Parallel Hybrid (7B)", "tiiuae/Falcon-H1-7B-Instruct"),
]

# Deterministic multi-domain evaluation corpus (Long-Context Technical + Mathematical
# Reasoning + Arabic/UAE Cultural Knowledge) for computing Perplexity (PPL), Logit KL
# Divergence, Top-1 Token Agreement, and Reasoning/Arabic Task Accuracy across precisions.
EVAL_QUALITY_CORPUS = [
    (
        "The Technology Innovation Institute (TII) in Abu Dhabi developed the Falcon-H1 architecture "
        "by combining Mamba-2 selective state-space models (SSMs) and multi-query Transformer attention "
        "heads in parallel within each mixer block. In a standard autoregressive Transformer, key-value "
        "(KV) cache memory scales linearly with sequence length L, whereas a recurrent state-space model "
        "maintains a fixed-size hidden state h_t = A_t h_{t-1} + B_t x_t. However, when low-bit uniform "
        "INT4 quantization is applied to the selective state projection matrices (in_proj, x_proj, dt_proj), "
        "quantization noise compounds multiplicatively across recurrent steps, whereas keeping the SSM "
        "branch in BF16 preserves state fidelity while compressing Attention and MLP weights."
    ),
    (
        "Consider an IEEE P1959 datacenter energy telemetry benchmark evaluating inference efficiency on "
        "an NVIDIA L4 Ada Lovelace GPU with a 72W TDP envelope. Let P(t) denote instantaneous board power "
        "sampled via NVML hardware energy counters and P_idle denote the pre-load baseline power floor. "
        "The total energy per generated token is E_tok = (E_end - E_start) / N_tokens, while the dynamic "
        "marginal energy per token subtracts P_idle * T_decode. As context length scales from 512 to 32768 "
        "tokens, quadratic prefill compute increases Time-To-First-Token (TTFT), while memory-bandwidth "
        "bottlenecks during KV cache reads increase Time-Per-Output-Token (TPOT)."
    ),
    (
        "يعد معهد الابتكار التكنولوجي في أبوظبي مركزاً عالمياً رائداً للبحث العلمي التطبيقي وتطوير نماذج "
        "الذكاء الاصطناعي اللغوية الكبيرة مثل سلسلة نماذج فالكون. تجمع البنية الهجينة الحديثة بين كفاءة "
        "نماذج الفضاء الحالة الانتقائية وآليات الانتباه المتوازية لمعالجة النصوص العربية والإنجليزية الطويلة "
        "بدقة عالية واستهلاك منخفض للطاقة والذاكرة الحاسوبية."
    ),
]

EVAL_TASKS = [
    {
        "name": "math_gsm1",
        "prompt": "Question: A datacenter cluster runs 8 GPUs. Each GPU consumes 60 Watts during inference and generates 40 tokens per second. How many total tokens does the entire 8-GPU cluster generate in 10 seconds? Answer with the exact number only.\nAnswer:",
        "expected_keywords": ["3200", "3,200"],
    },
    {
        "name": "math_algebra",
        "prompt": "Question: Solve for x: 3x + 15 = 42. Answer with the number only.\nAnswer:",
        "expected_keywords": ["9"],
    },
    {
        "name": "reasoning_logic",
        "prompt": "Question: Model A uses less memory than Model B. Model B uses less memory than Model C. Which model uses the least memory? Answer with 'Model A', 'Model B', or 'Model C'.\nAnswer:",
        "expected_keywords": ["Model A"],
    },
    {
        "name": "uae_arabic_knowledge",
        "prompt": "سؤال: ما هي عاصمة دولة الإمارات العربية المتحدة التي يقع فيها معهد الابتكار التكنولوجي (TII)؟ أجب باسم المدينة فقط.\nالإجابة:",
        "expected_keywords": ["أبوظبي", "أبو ظبي", "Abu Dhabi"],
    },
    {
        "name": "arch_recall",
        "prompt": "Context: Code-key for Pure Transformer is ALPHA-701. Code-key for Pure Mamba SSM is BETA-802. Code-key for Parallel Hybrid Falcon-H1 is GAMMA-903.\nQuestion: What is the exact code-key for Parallel Hybrid Falcon-H1?\nAnswer:",
        "expected_keywords": ["GAMMA-903", "903"],
    },
]


# ---------------------------------------------------------------------------
# 3. CPU-Only Model Pre-Downloader ($0 GPU Spend)
# ---------------------------------------------------------------------------
@app.function(
    cpu=4.0,
    memory=16384,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def download_model_weights(model_id: str) -> str:
    """Downloads model weights to the persistent Modal Volume using CPU only."""
    from huggingface_hub import snapshot_download

    print(f"[CPU Downloader] Ensuring '{model_id}' is cached in {HF_CACHE_DIR}...")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=HF_CACHE_DIR,
        ignore_patterns=["*.gguf", "*.bin", "original/*"],
    )
    volume.commit()
    elapsed = time.time() - t0
    print(f"[CPU Downloader] Ready: '{model_id}' ({elapsed:.1f}s) -> {local_dir}")
    return model_id


# ---------------------------------------------------------------------------
# 4. Isolated Datacenter GPU Benchmark Runner
# ---------------------------------------------------------------------------
@app.function(
    gpu="L4",  # 24GB Ada Lovelace (native BF16, clean NVML hardware energy counters)
    cpu=4.0,
    memory=32768,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def run_gpu_benchmark(
    arch_label: str,
    model_id: str,
    precision: str = "bf16",
    context_lengths: list[int] = [512, 2048, 4096, 8192],
    decode_tokens: int = 128,
    num_trials: int = 2,
) -> list[dict[str, Any]]:
    """
    Runs TTFT, Decode Throughput, VRAM Growth, and IEEE-style NVML Energy (Joules/token)
    benchmarks on an isolated Modal L4 GPU.
    """
    import gc
    import pynvml
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        volume.reload()
    except RuntimeError as e:
        if "open files" not in str(e):
            raise
        print(f"[{arch_label}] Note: Volume already active in reused container ({e}), continuing...")

    # Initialize NVML for isolated datacenter GPU telemetry
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")

    # Background power sampler + hardware cumulative energy counter
    class NVMLPowerMonitor:
        def __init__(self, dev_handle, poll_interval_s: float = 0.02):
            self.handle = dev_handle
            self.poll_interval_s = poll_interval_s
            self.samples_watts: list[float] = []
            self._stop = threading.Event()
            self._thread = None
            self.e_start_mj = None
            self.e_end_mj = None
            self.t_start = 0.0
            self.t_end = 0.0

        def _poll(self):
            while not self._stop.is_set():
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.samples_watts.append(mw / 1000.0)
                except Exception:
                    pass
                time.sleep(self.poll_interval_s)

        def start(self):
            self.samples_watts.clear()
            self._stop.clear()
            self.t_start = time.perf_counter()
            try:
                self.e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            except Exception:
                self.e_start_mj = None
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

        def stop(self) -> dict[str, float]:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self.t_end = time.perf_counter()
            try:
                self.e_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            except Exception:
                self.e_end_mj = None

            duration_s = max(self.t_end - self.t_start, 1e-6)
            mean_w = (
                sum(self.samples_watts) / len(self.samples_watts)
                if self.samples_watts
                else 0.0
            )
            peak_w = max(self.samples_watts) if self.samples_watts else 0.0

            # Prefer hardware accumulator (millijoules) if supported, fallback to trapezoidal integration
            if (
                self.e_start_mj is not None
                and self.e_end_mj is not None
                and self.e_end_mj > self.e_start_mj
            ):
                total_joules = (self.e_end_mj - self.e_start_mj) / 1000.0
            else:
                total_joules = mean_w * duration_s

            return {
                "duration_s": duration_s,
                "mean_watts": mean_w,
                "peak_watts": peak_w,
                "total_joules": total_joules,
            }

    # Measure pre-model GPU idle baseline power over 2.5 seconds
    idle_mon = NVMLPowerMonitor(handle)
    idle_mon.start()
    time.sleep(2.5)
    idle_stats = idle_mon.stop()
    idle_watts = idle_stats["mean_watts"]
    print(f"[{arch_label}] GPU: {gpu_name} | Clean Idle Power Floor: {idle_watts:.2f} W")

    # Configure precision / quantization
    quant_config = None
    if precision == "int4_uniform":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "int4_mixed_ssm":
        # Branch-Aware Hybrid Quantization:
        # Keep Mamba SSM selective state & recurrence projections in BF16 while
        # quantizing Attention (q_proj, k_proj, v_proj, o_proj) & MLP projections to INT4.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "mamba",
                "in_proj",
                "x_proj",
                "dt_proj",
                "conv1d",
                "out_proj",
                "lm_head",
            ],
        )

    print(f"[{arch_label}] Loading '{model_id}' in mode '{precision}'...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    torch.cuda.synchronize()
    weights_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"[{arch_label}] Loaded! Static Model Footprint: {weights_vram_mb:.1f} MB")

    # Warmup pass
    warmup_ids = torch.randint(100, 2000, (1, 128), device="cuda")
    with torch.no_grad():
        _ = model.generate(
            warmup_ids,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Evaluation-Quality & Quantization Delta Suite
    #   1. Corpus Perplexity (PPL) & Negative Log-Likelihood (NLL)
    #   2. Token-Level Logit KL Divergence & Top-1 Agreement vs. BF16 Baseline
    #   3. Deterministic Reasoning + Arabic/UAE Task Accuracy (%)
    # -----------------------------------------------------------------------
    print(f"[{arch_label}] Running Evaluation-Quality & Fidelity Suite ({precision})...")
    import math
    import torch.nn.functional as F

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model_tag = model_id.replace("/", "__")
    bf16_ref_path = f"{RESULTS_DIR}/{safe_model_tag}__bf16_ref_logits.pt"

    total_nll = 0.0
    total_eval_tokens = 0
    current_logits_list: list[torch.Tensor] = []

    with torch.no_grad():
        for passage in EVAL_QUALITY_CORPUS:
            enc = tokenizer(
                passage, return_tensors="pt", truncation=True, max_length=1024
            ).to("cuda")
            ids = enc["input_ids"]
            seq_len = ids.shape[1]
            if seq_len < 2:
                continue
            out_eval = model(ids, labels=ids)
            n_tok = seq_len - 1
            total_nll += float(out_eval.loss.item()) * n_tok
            total_eval_tokens += n_tok
            # Store FP32 CPU logits for exact KL / Top-1 comparison across precisions
            current_logits_list.append(out_eval.logits[:, :-1, :].detach().float().cpu())

    eval_nll = total_nll / max(total_eval_tokens, 1)
    eval_ppl = math.exp(min(eval_nll, 20.0))

    # Save or compare against BF16 reference logits
    if precision == "bf16":
        torch.save(
            {"eval_ppl": eval_ppl, "logits": current_logits_list},
            bf16_ref_path,
        )
        volume.commit()
        ppl_delta_pct = 0.0
        kl_div_vs_bf16 = 0.0
        top1_agree_pct = 100.0
    elif os.path.exists(bf16_ref_path):
        ref_data = torch.load(bf16_ref_path, map_location="cpu", weights_only=False)
        ref_ppl = float(ref_data["eval_ppl"])
        ref_logits_list = ref_data["logits"]
        ppl_delta_pct = ((eval_ppl - ref_ppl) / max(ref_ppl, 1e-6)) * 100.0

        kl_sum = 0.0
        agree_tokens = 0
        compared_tokens = 0
        for ref_l, cur_l in zip(ref_logits_list, current_logits_list):
            p_ref = F.softmax(ref_l, dim=-1)
            log_p_ref = F.log_softmax(ref_l, dim=-1)
            log_q_cur = F.log_softmax(cur_l, dim=-1)
            # Token-wise KL(P_bf16 || Q_quant)
            kl_tok = (p_ref * (log_p_ref - log_q_cur)).sum(dim=-1)
            kl_sum += float(kl_tok.sum().item())
            agree_tokens += int(
                (ref_l.argmax(dim=-1) == cur_l.argmax(dim=-1)).sum().item()
            )
            compared_tokens += int(ref_l.shape[1])

        kl_div_vs_bf16 = kl_sum / max(compared_tokens, 1)
        top1_agree_pct = (agree_tokens / max(compared_tokens, 1)) * 100.0
    else:
        ppl_delta_pct = 0.0
        kl_div_vs_bf16 = -1.0
        top1_agree_pct = -1.0

    # Deterministic Reasoning + Arabic/UAE Task Accuracy
    correct_tasks = 0
    with torch.no_grad():
        for task in EVAL_TASKS:
            t_enc = tokenizer(task["prompt"], return_tensors="pt").to("cuda")
            gen_ids = model.generate(
                **t_enc,
                max_new_tokens=24,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = gen_ids[0, t_enc["input_ids"].shape[1] :]
            ans_text = tokenizer.decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            if any(
                kw.lower() in ans_text.lower() for kw in task["expected_keywords"]
            ):
                correct_tasks += 1

    task_accuracy_pct = (correct_tasks / len(EVAL_TASKS)) * 100.0
    print(
        f"[{arch_label}] Eval Quality ({precision}): PPL={eval_ppl:.3f} "
        f"(ΔPPL={ppl_delta_pct:+.2f}%) | KL_vs_BF16={kl_div_vs_bf16:.4f} | "
        f"Top1_Agree={top1_agree_pct:.1f}% | TaskAcc={task_accuracy_pct:.1f}%"
    )

    results: list[dict[str, Any]] = []

    for ctx_len in context_lengths:
        print(f"  -> Sweeping context_length={ctx_len}, decode_tokens={decode_tokens}...")
        gc.collect()
        torch.cuda.empty_cache()

        # Construct synthetic prompt of exact token length ctx_len
        input_ids = torch.randint(100, 5000, (1, ctx_len), device="cuda")
        attention_mask = torch.ones_like(input_ids, device="cuda")

        ttft_ms_list = []
        prefill_joules_list = []
        decode_tps_list = []
        tpot_ms_list = []
        total_joules_per_tok_list = []
        dynamic_joules_per_tok_list = []
        mean_power_list = []
        peak_power_list = []
        peak_vram_mb = 0.0
        cache_growth_mb = 0.0

        try:
            for trial in range(num_trials):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base_mem_mb = torch.cuda.memory_allocated() / (1024**2)

                # --- Step A: Measure Prefill (TTFT) ---
                prefill_mon = NVMLPowerMonitor(handle)
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                prefill_mon.start()
                ev_start.record()
                with torch.no_grad():
                    _ = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        min_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                ev_end.record()
                torch.cuda.synchronize()
                p_stats = prefill_mon.stop()

                ttft_ms = ev_start.elapsed_time(ev_end)
                ttft_ms_list.append(ttft_ms)
                prefill_joules_list.append(p_stats["total_joules"])

                # --- Step B: Measure Full Prefill + Autoregressive Decode ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                gen_mon = NVMLPowerMonitor(handle)
                ev_gen_start = torch.cuda.Event(enable_timing=True)
                ev_gen_end = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                gen_mon.start()
                ev_gen_start.record()
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=decode_tokens,
                        min_new_tokens=decode_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                ev_gen_end.record()
                torch.cuda.synchronize()
                g_stats = gen_mon.stop()

                total_gen_ms = ev_gen_start.elapsed_time(ev_gen_end)
                actual_new_tokens = max(out.shape[1] - ctx_len, 1)

                # Isolate pure decode time & energy by subtracting the single-token prefill pass
                decode_only_ms = max(total_gen_ms - ttft_ms, 1.0)
                decode_only_s = decode_only_ms / 1000.0
                decode_tps = (actual_new_tokens - 1) / decode_only_s
                tpot_ms = decode_only_ms / max(actual_new_tokens - 1, 1)

                # Energy calculations (Total vs. Dynamic above idle floor)
                decode_joules_total = max(
                    g_stats["total_joules"] - p_stats["total_joules"],
                    g_stats["mean_watts"] * decode_only_s,
                )
                decode_joules_dynamic = max(
                    (g_stats["mean_watts"] - idle_watts) * decode_only_s, 0.01
                )

                j_per_tok_total = decode_joules_total / actual_new_tokens
                j_per_tok_dynamic = decode_joules_dynamic / actual_new_tokens

                decode_tps_list.append(decode_tps)
                tpot_ms_list.append(tpot_ms)
                total_joules_per_tok_list.append(j_per_tok_total)
                dynamic_joules_per_tok_list.append(j_per_tok_dynamic)
                mean_power_list.append(g_stats["mean_watts"])
                peak_power_list.append(g_stats["peak_watts"])

                peak_vram_mb = max(
                    peak_vram_mb, torch.cuda.max_memory_allocated() / (1024**2)
                )
                cache_growth_mb = max(peak_vram_mb - weights_vram_mb, 0.0)

            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "arch_label": arch_label,
                "model_id": model_id,
                "precision": precision,
                "gpu_name": gpu_name,
                "idle_watts": round(idle_watts, 2),
                "weights_vram_mb": round(weights_vram_mb, 1),
                "context_length": ctx_len,
                "decode_tokens": decode_tokens,
                "ttft_ms": round(sum(ttft_ms_list) / len(ttft_ms_list), 2),
                "decode_tok_per_s": round(
                    sum(decode_tps_list) / len(decode_tps_list), 2
                ),
                "tpot_ms": round(sum(tpot_ms_list) / len(tpot_ms_list), 2),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "kv_or_state_growth_mb": round(cache_growth_mb, 1),
                "mean_watts": round(sum(mean_power_list) / len(mean_power_list), 2),
                "peak_watts": round(max(peak_power_list), 2),
                "joules_per_token_total": round(
                    sum(total_joules_per_tok_list) / len(total_joules_per_tok_list), 4
                ),
                "joules_per_token_dynamic": round(
                    sum(dynamic_joules_per_tok_list)
                    / len(dynamic_joules_per_tok_list),
                    4,
                ),
                "eval_ppl": round(eval_ppl, 4),
                "ppl_delta_pct": round(ppl_delta_pct, 2),
                "kl_div_vs_bf16": round(kl_div_vs_bf16, 5),
                "top1_agree_pct": round(top1_agree_pct, 2),
                "task_accuracy_pct": round(task_accuracy_pct, 1),
                "status": "ok",
            }
            print(
                f"     [OK] ctx={ctx_len:5d} | TTFT={row['ttft_ms']:7.1f}ms | "
                f"Decode={row['decode_tok_per_s']:6.1f} tok/s | "
                f"PeakVRAM={row['peak_vram_mb']:7.1f}MB (+{row['kv_or_state_growth_mb']:.1f}MB) | "
                f"Energy={row['joules_per_token_total']:.3f} J/tok | "
                f"PPL={row['eval_ppl']:.2f} ({row['ppl_delta_pct']:+.1f}%)"
            )
            results.append(row)

        except torch.cuda.OutOfMemoryError:
            print(f"     [OOM] {arch_label} hit CUDA OOM at context_length={ctx_len}")
            results.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "arch_label": arch_label,
                    "model_id": model_id,
                    "precision": precision,
                    "gpu_name": gpu_name,
                    "context_length": ctx_len,
                    "status": "oom",
                }
            )
            torch.cuda.empty_cache()
            break

    # Persist results inside the Modal Volume as well
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = model_id.replace("/", "__") + f"__{precision}.json"
    with open(f"{RESULTS_DIR}/{safe_name}", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    pynvml.nvmlShutdown()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# 5. Local CLI Entrypoint (Orchestrates CPU Download -> GPU Benchmarks)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    mode: str = "smoke",
    context_lengths: str = "",
    decode_tokens: int = 128,
):
    """
    Modes:
      - smoke          : Fast verification on 1B/1.5B models at [512, 2048] (~$0.03)
      - crossover      : Week 1 Full 7B 3-Architecture BF16 Crossover Sweep
      - quant-ablation : Week 2-3 Hybrid Quantization Degradation vs. Mixed-Precision Fix
    """
    if mode == "smoke":
        models = SMOKE_MODELS
        ctx_list = [int(x) for x in context_lengths.split(",")] if context_lengths else [512, 2048]
        precisions = ["bf16", "int4_uniform", "int4_mixed_ssm"]
    elif mode == "crossover":
        models = SEVEN_B_TRIAD
        ctx_list = (
            [int(x) for x in context_lengths.split(",")]
            if context_lengths
            else [512, 2048, 4096, 8192, 16384]
        )
        precisions = ["bf16"]
    elif mode == "quant-ablation":
        models = SEVEN_B_TRIAD
        ctx_list = (
            [int(x) for x in context_lengths.split(",")]
            if context_lengths
            else [512, 2048, 8192]
        )
        precisions = ["bf16", "int4_uniform", "int4_mixed_ssm"]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: smoke, crossover, quant-ablation")

    print("=" * 72)
    print(f"TII FALCON BENCHMARK | Mode: {mode} | Contexts: {ctx_list}")
    print("=" * 72)

    # Step 1: Pre-download all required models on CPU-only workers in parallel ($0 GPU spend)
    print("\n[Stage 1/2] Pre-caching model weights into Modal Volume (CPU workers)...")
    model_ids = [m[1] for m in models]
    list(download_model_weights.map(model_ids))

    # Step 2: Execute isolated GPU benchmarks
    print("\n[Stage 2/2] Running isolated GPU telemetry benchmarks on Modal L4...")
    all_results: list[dict[str, Any]] = []

    # In quant-ablation mode, reuse existing BF16 records from benchmark_crossover.json if available
    existing_bf16: dict[str, list[dict[str, Any]]] = {}
    crossover_file = Path("results/benchmark_crossover.json")
    if mode == "quant-ablation" and crossover_file.exists():
        try:
            with open(crossover_file, "r", encoding="utf-8") as cf:
                for r in json.load(cf):
                    if r.get("precision") == "bf16" and r.get("context_length") in ctx_list:
                        existing_bf16.setdefault(r["model_id"], []).append(r)
        except Exception:
            pass

    for arch_label, model_id in models:
        for prec in precisions:
            # Skip int4_mixed_ssm on pure Transformer (only relevant for Mamba/Hybrid, though safe)
            if prec == "int4_mixed_ssm" and "Transformer" in arch_label and "Hybrid" not in arch_label:
                continue
            if prec == "bf16" and model_id in existing_bf16 and len(existing_bf16[model_id]) == len(ctx_list):
                print(f"[{arch_label}] Reusing cached BF16 baseline ({len(ctx_list)} contexts) from benchmark_crossover.json...")
                all_results.extend(existing_bf16[model_id])
                continue
            rows = run_gpu_benchmark.remote(
                arch_label=arch_label,
                model_id=model_id,
                precision=prec,
                context_lengths=ctx_list,
                decode_tokens=decode_tokens,
            )
            all_results.extend(rows)

    # Save locally to results/
    local_out_dir = Path("results")
    local_out_dir.mkdir(exist_ok=True)
    out_file = local_out_dir / f"benchmark_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Done! Saved {len(all_results)} benchmark records to: {out_file}")
    print("=" * 72)
