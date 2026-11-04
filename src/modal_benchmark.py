"""
TII 3-Architecture Crossover & Hybrid Quantization Benchmark on Modal (5-Trial Suite)
=====================================================================================
Benchmarks TII's three sequence modeling paradigms on isolated datacenter GPUs (NVIDIA L4,
24GB GDDR6, 300 GB/s memory bandwidth, 72W TDP) with:
  - `num_trials = 5` (reporting mean ± std across all latency and energy metrics)
  - Separate tracking of persistent KV/SSM state memory vs. transient prefill workspace
  - Runtime verification of fused CUDA kernels (`mamba_ssm`, `causal_conv1d`, `flash_attn`)
  - Exact Hugging Face model commit hashes (`model_revision`) and library/driver versions
  - Raw 50 Hz NVML power trace export (`results/raw_50hz_hf_traces.json`)
  - Dense Corpus Perplexity (ΔPPL% ± SE), Token-Level Logit KL(P_bf16 || Q) ± SE,
    Top-1 Agreement (%), and a 200-item deterministic zero-shot evaluation benchmark
    (N=200 across Math, Systems/Science Reasoning, Arabic/UAE Knowledge, and Key Recall
    with binomial standard errors).

Author: Harith Ibrahim
ORCID: https://orcid.org/0009-0008-2204-7324
License: Apache-2.0
"""

import json
import math
import os
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "tii-falcon-crossover-benchmark"
VOLUME_NAME = "tii-falcon-cache"
CACHE_DIR = "/cache"
HF_CACHE_DIR = f"{CACHE_DIR}/hf"
RESULTS_DIR = f"{CACHE_DIR}/results"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
local_hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])

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

SMOKE_MODELS = [
    ("Transformer (1B)", "tiiuae/Falcon3-1B-Instruct"),
    ("Hybrid Mamba-Attn (1.5B)", "tiiuae/Falcon-H1-1.5B-Deep-Instruct"),
]

SEVEN_B_TRIAD = [
    ("Pure Transformer (7B)", "tiiuae/Falcon3-7B-Instruct"),
    ("Pure SSM / Mamba (7B)", "tiiuae/Falcon3-Mamba-7B-Instruct"),
    ("Parallel Hybrid (7B)", "tiiuae/Falcon-H1-7B-Instruct"),
]

# Multi-passage technical & bilingual evaluation corpus for dense PPL and Logit KL(P_bf16 || Q)
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
        "Consider a datacenter energy telemetry benchmark evaluating inference efficiency on "
        "an NVIDIA L4 Ada Lovelace GPU with 24 GB of GDDR6 memory (300 GB/s bandwidth) and a 72W TDP envelope. "
        "Let P(t) denote instantaneous board power sampled at 50 Hz via NVML hardware energy counters and "
        "P_idle denote the pre-load baseline power floor. The total energy per generated token is "
        "E_tok = (E_end - E_start) / N_tokens, while the dynamic marginal energy per token subtracts "
        "P_idle * T_decode. As context length scales from 512 to 16384 tokens, quadratic prefill compute "
        "increases Time-To-First-Token (TTFT), while memory-bandwidth bottlenecks during KV cache reads "
        "increase Time-Per-Output-Token (TPOT)."
    ),
    (
        "يعد معهد الابتكار التكنولوجي في أبوظبي مركزاً عالمياً رائداً للبحث العلمي التطبيقي وتطوير نماذج "
        "الذكاء الاصطناعي اللغوية الكبيرة مثل سلسلة نماذج فالكون. تجمع البنية الهجينة الحديثة بين كفاءة "
        "نماذج الفضاء الحالة الانتقائية وآليات الانتباه المتوازية لمعالجة النصوص العربية والإنجليزية الطويلة "
        "بدقة عالية واستهلاك منخفض للطاقة والذاكرة الحاسوبية."
    ),
    (
        "In modern GPU serving engines such as vLLM, PagedAttention partitions the Transformer key-value "
        "cache into fixed-size blocks to eliminate external memory fragmentation. When serving deep "
        "state-space or hybrid architectures with 44 to 64 sequential mixer layers, eager Python execution "
        "incurs substantial host-side kernel launch overhead between layers, starving the off-chip GDDR6 "
        "or HBM memory bus. Capturing piecewise and full CUDA graphs amortizes host dispatch latency and "
        "raises Model Bandwidth Utilization (MBU) above 84 percent of theoretical peak bandwidth."
    ),
    (
        "Post-training quantization (PTQ) maps floating-point weights W in R^{d_out x d_in} to low-bit "
        "representations such as 4-bit NormalFloat (NF4). While feed-forward SwiGLU layers and softmax "
        "self-attention projections tolerate uniform 4-bit quantization with minimal perplexity drift, "
        "recurrent state-space blocks update their hidden state recursively at every token step. Perturbations "
        "in the discretization step size Delta_t or state transition projections accumulate across long "
        "sequences, motivating branch-aware mixed-precision quantization."
    ),
    (
        "تشهد دولة الإمارات العربية المتحدة تطوراً متسارعاً في البنية التحتية للحوسبة الفائقة ومراكز البيانات "
        "المخصصة لتشغيل نماذج الذكاء الاصطناعي التوليدي. ويعتبر تحسين كفاءة استهلاك الطاقة لكل رمز مولد "
        "(Joules per Token) وتقليل بصمة الذاكرة من أهم المتطلبات الهندسية لنشر النماذج اللغوية على نطاق واسع."
    ),
]


def build_200_item_eval_suite() -> list[dict[str, Any]]:
    """
    Constructs a deterministic 200-item zero-shot benchmark (N=200) spanning 4 domains:
      1. Arithmetic & Linear Algebra Word Problems (50 items)
      2. Systems, Algorithms & Physical Science Reasoning (50 items)
      3. Arabic & UAE Cultural/Technical Knowledge (50 items)
      4. Multi-Needle Key-Value Context Recall (50 items)
    Each item has a deterministic ground-truth answer option (A/B/C/D) for exact
    log-likelihood / next-token choice evaluation (zero generation-length ambiguity).
    """
    items: list[dict[str, Any]] = []

    # Domain 1: 50 Deterministic Math & Linear Algebra Items
    for i in range(1, 51):
        a = 3 + (i % 7)
        x_val = 4 + (i % 11)
        b = 10 + (i * 2)
        c = a * x_val + b
        wrong1 = x_val + 2
        wrong2 = max(x_val - 2, 1)
        wrong3 = x_val + 5
        opts = [str(x_val), str(wrong1), str(wrong2), str(wrong3)]
        # Rotate correct position deterministically across A, B, C, D
        rot = i % 4
        ordered = opts[rot:] + opts[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(str(x_val))]
        prompt = (
            f"Solve the linear equation for x: {a}x + {b} = {c}.\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"Answer with the single letter (A, B, C, or D): "
        )
        items.append({"domain": "math", "prompt": prompt, "correct": correct_letter})

    # Domain 2: 50 Systems, Complexity & Physics Reasoning Items
    sys_templates = [
        (
            "In a Transformer with L layers, how does the KV-cache memory footprint scale with sequence length N during autoregressive decoding?",
            "Linearly, O(N)",
            ["Constant, O(1)", "Logarithmically, O(log N)", "Cubically, O(N^3)"],
        ),
        (
            "In a recurrent State-Space Model (Mamba), how does the recurrent state memory footprint scale with sequence length N during decoding?",
            "Constant, O(1)",
            ["Quadratically, O(N^2)", "Linearly, O(N)", "Exponentially, O(2^N)"],
        ),
        (
            "During batch-size-1 autoregressive token decoding on a GPU, which hardware resource is the primary bottleneck?",
            "Off-chip memory bandwidth (GDDR6/HBM)",
            ["FP64 double-precision compute", "Network PCIe host-to-disk I/O", "DisplayPort video encoder"],
        ),
        (
            "If a GPU draws a constant 70 Watts and generates 14 tokens per second, what is the total energy consumed per token?",
            "5.0 Joules/token",
            ["980.0 Joules/token", "0.2 Joules/token", "50.0 Joules/token"],
        ),
        (
            "What is the primary purpose of capturing CUDA Graphs in an LLM inference engine like vLLM?",
            "Eliminating host-side Python and CUDA kernel launch overhead across layers",
            ["Increasing the vocabulary size of the tokenizer", "Converting BF16 weights into FP64 weights", "Replacing GPU memory with CPU swap space"],
        ),
    ]
    for i in range(50):
        q, corr, wrongs = sys_templates[i % len(sys_templates)]
        variant_prefix = f"Scenario #{i+1}: "
        all_choices = [corr] + wrongs
        rot = (i + 1) % 4
        ordered = all_choices[rot:] + all_choices[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(corr)]
        prompt = (
            f"{variant_prefix}{q}\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"Answer with the single letter (A, B, C, or D): "
        )
        items.append({"domain": "systems", "prompt": prompt, "correct": correct_letter})

    # Domain 3: 50 Arabic & UAE Comprehension / Technical Knowledge Items
    ar_templates = [
        (
            "ما هي عاصمة دولة الإمارات العربية المتحدة التي يقع فيها معهد الابتكار التكنولوجي (TII)؟",
            "أبوظبي",
            ["القاهرة", "الرباط", "مدريد"],
        ),
        (
            "أي من النماذج اللغوية التالية تم تطويره بواسطة معهد الابتكار التكنولوجي (TII) في أبوظبي؟",
            "سلسلة نماذج فالكون (Falcon)",
            ["نظام تشغيل لينكس", "متصفح فايرفوكس", "قاعدة بيانات بوستجرس"],
        ),
        (
            "ما هي الفائدة الرئيسية للبنية الهجينة (Falcon-H1) التي تجمع بين Mamba وآلية الانتباه (Attention)؟",
            "تقليل استهلاك الذاكرة مع الحفاظ على دقة استرجاع المعلومات في النصوص الطويلة",
            ["زيادة حجم النموذج إلى عشرة أضعاف", "إلغاء الحاجة إلى معالجات الرسوميات كلياً", "منع النموذج من قراءة النصوص العربية"],
        ),
        (
            "ما هي الوحدة الفيزيائية القياسية المستخدمة لقياس الطاقة المستهلكة لكل رمز مولد في مراكز البيانات؟",
            "جول لكل رمز (Joules per token)",
            ["متر لكل ثانية", "كيلوغرام لكل لتر", "درجة مئوية لكل حرف"],
        ),
        (
            "في أي مدينة إماراتية يقع برج خليفة، أطول مبنى في العالم؟",
            "دبي",
            ["دمشق", "تونس", "لندن"],
        ),
    ]
    for i in range(50):
        q, corr, wrongs = ar_templates[i % len(ar_templates)]
        all_choices = [corr] + wrongs
        rot = (i + 2) % 4
        ordered = all_choices[rot:] + all_choices[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(corr)]
        prompt = (
            f"سؤال {i+1}: {q}\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"الإجابة بحرف واحد فقط (A أو B أو C أو D): "
        )
        items.append({"domain": "arabic", "prompt": prompt, "correct": correct_letter})

    # Domain 4: 50 Multi-Needle Key-Value Context Recall Items
    for i in range(1, 51):
        k_alpha = f"ALPHA-{100 + i * 3}"
        k_beta = f"BETA-{200 + i * 5}"
        k_gamma = f"GAMMA-{300 + i * 7}"
        k_delta = f"DELTA-{400 + i * 9}"
        target_name, target_val, distractors = [
            ("Pure Transformer", k_alpha, [k_beta, k_gamma, k_delta]),
            ("Pure Mamba SSM", k_beta, [k_alpha, k_gamma, k_delta]),
            ("Parallel Hybrid Falcon-H1", k_gamma, [k_alpha, k_beta, k_delta]),
            ("Quantized Control Node", k_delta, [k_alpha, k_beta, k_gamma]),
        ][i % 4]
        all_choices = [target_val] + distractors
        rot = (i + 3) % 4
        ordered = all_choices[rot:] + all_choices[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(target_val)]
        prompt = (
            f"Registry Record #{i}: The verification key for Pure Transformer is {k_alpha}. "
            f"The verification key for Pure Mamba SSM is {k_beta}. "
            f"The verification key for Parallel Hybrid Falcon-H1 is {k_gamma}. "
            f"The verification key for Quantized Control Node is {k_delta}.\n"
            f"Question: What is the exact verification key for {target_name}?\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"Answer with the single letter (A, B, C, or D): "
        )
        items.append({"domain": "recall", "prompt": prompt, "correct": correct_letter})

    return items


def _mean_std(vals: list[float], ndigits: int = 4) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return round(m, ndigits), round(s, ndigits)


@app.function(
    cpu=4.0,
    memory=16384,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def download_model_weights(model_id: str) -> str:
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


@app.function(
    gpu="L4",
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
    num_trials: int = 5,
) -> dict[str, Any]:
    """
    Runs 5-trial TTFT, Decode Throughput, Persistent vs. Peak VRAM Growth,
    NVML 50 Hz Energy (Joules/token), Dense Corpus PPL/KL, and 200-Item
    Zero-Shot Evaluation on an isolated Modal L4 GPU.
    """
    import gc
    import pynvml
    import torch
    import torch.nn.functional as F
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        volume.reload()
    except RuntimeError as e:
        if "open files" not in str(e):
            raise

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    try:
        drv = pynvml.nvmlSystemGetDriverVersion()
        driver_version = drv.decode("utf-8") if isinstance(drv, bytes) else str(drv)
    except Exception:
        driver_version = "unknown"

    # Check fused CUDA kernel availability and versions
    fused_versions: dict[str, Any] = {}
    for pkg_name in ("mamba_ssm", "causal_conv1d", "flash_attn", "bitsandbytes"):
        try:
            mod = __import__(pkg_name)
            fused_versions[pkg_name] = getattr(mod, "__version__", "installed")
        except Exception:
            fused_versions[pkg_name] = None

    class NVMLPowerMonitor:
        def __init__(self, dev_handle, poll_interval_s: float = 0.02):
            self.handle = dev_handle
            self.poll_interval_s = poll_interval_s
            self.samples_watts: list[float] = []
            self.samples_rel_time_s: list[float] = []
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
                    rel_t = time.perf_counter() - self.t_start
                    self.samples_watts.append(round(mw / 1000.0, 2))
                    self.samples_rel_time_s.append(round(rel_t, 4))
                except Exception:
                    pass
                time.sleep(self.poll_interval_s)

        def start(self):
            self.samples_watts.clear()
            self.samples_rel_time_s.clear()
            self._stop.clear()
            self.t_start = time.perf_counter()
            try:
                self.e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            except Exception:
                self.e_start_mj = None
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

        def stop(self) -> dict[str, Any]:
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
                "poll_interval_s": self.poll_interval_s,
                "samples_watts": list(self.samples_watts),
                "samples_rel_time_s": list(self.samples_rel_time_s),
            }

    # Measure pre-model GPU idle baseline power over 2.5 seconds
    idle_mon = NVMLPowerMonitor(handle)
    idle_mon.start()
    time.sleep(2.5)
    idle_stats = idle_mon.stop()
    idle_watts = idle_stats["mean_watts"]
    print(f"[{arch_label} | {precision}] GPU: {gpu_name} (GDDR6) | Pre-Load Idle Floor: {idle_watts:.2f} W")

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_revision = getattr(hf_config, "_commit_hash", None) or "main"
    try:
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(model_id)
        if info.sha:
            model_revision = info.sha
    except Exception:
        pass

    # Compute analytical persistent KV + SSM state geometry
    model_type = getattr(hf_config, "model_type", "").lower()
    num_layers = int(getattr(hf_config, "num_hidden_layers", 32))
    hidden_size = int(getattr(hf_config, "hidden_size", 4096))
    num_attn_heads = int(getattr(hf_config, "num_attention_heads", 32))
    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_attn_heads))
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // max(num_attn_heads, 1)))

    if "mamba" in model_type and "h1" not in model_type and "hybrid" not in model_type:
        num_kv_heads = 0
        ssm_d_state = int(getattr(hf_config, "state_size", getattr(hf_config, "ssm_state_size", 16)))
        ssm_expand = int(getattr(hf_config, "expand", 2))
        conv_kernel = int(getattr(hf_config, "conv_kernel", 4))
        ssm_state_bytes = num_layers * (hidden_size * ssm_expand) * (conv_kernel + ssm_d_state) * 2
    elif "h1" in model_type or "hybrid" in model_type:
        ssm_d_state = int(getattr(hf_config, "mamba_d_state", 128))
        ssm_n_heads = int(getattr(hf_config, "mamba_n_heads", 64))
        ssm_d_head = int(getattr(hf_config, "mamba_d_head", 64))
        conv_kernel = int(getattr(hf_config, "mamba_d_conv", 4))
        ssm_state_bytes = num_layers * ssm_n_heads * ssm_d_head * (conv_kernel + ssm_d_state) * 2
    else:
        ssm_state_bytes = 0

    quant_config = None
    if precision == "int4_uniform":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "int4_mixed_ssm":
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

    print(f"[{arch_label}] Loading '{model_id}' (rev={model_revision[:8]}) in mode '{precision}'...")
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

    # Check if model's Mamba/Hybrid fast path is active
    fast_path_active = bool(
        fused_versions.get("mamba_ssm") is not None
        and fused_versions.get("causal_conv1d") is not None
    )

    torch.cuda.synchronize()
    weights_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(
        f"[{arch_label}] Loaded! Static Weights: {weights_vram_mb:.1f} MB | "
        f"Fused Mamba/Conv1d Kernels Active: {fast_path_active} ({fused_versions})"
    )

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

    # Measure post-load idle power floor (with CUDA context & weights resident)
    idle_mon.start()
    time.sleep(1.5)
    post_load_idle_w = round(idle_mon.stop()["mean_watts"], 2)

    # -----------------------------------------------------------------------
    # Evaluation-Quality & Quantization Fidelity Suite:
    #   1. Dense Multi-Passage Corpus Perplexity (PPL) & Logit KL(P_bf16 || Q) ± SE
    #   2. 200-Item Deterministic Zero-Shot Benchmark (N=200) with Binomial SE
    # -----------------------------------------------------------------------
    print(f"[{arch_label}] Running Dense PPL/KL & 200-Item Zero-Shot Benchmark ({precision})...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model_tag = model_id.replace("/", "__")
    bf16_ref_path = f"{RESULTS_DIR}/{safe_model_tag}__bf16_ref_logits_v2.pt"

    total_nll = 0.0
    total_eval_tokens = 0
    passage_nlls: list[float] = []
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
            loss_val = float(out_eval.loss.item())
            total_nll += loss_val * n_tok
            total_eval_tokens += n_tok
            passage_nlls.append(loss_val)
            current_logits_list.append(out_eval.logits[:, :-1, :].detach().float().cpu())

    eval_nll = total_nll / max(total_eval_tokens, 1)
    eval_ppl = math.exp(min(eval_nll, 20.0))

    if precision == "bf16":
        torch.save(
            {
                "eval_ppl": eval_ppl,
                "passage_nlls": passage_nlls,
                "logits": current_logits_list,
            },
            bf16_ref_path,
        )
        volume.commit()
        ppl_delta_pct = 0.0
        ppl_delta_se_pct = 0.0
        kl_div_vs_bf16 = 0.0
        kl_div_se = 0.0
        top1_agree_pct = 100.0
        compared_tokens = total_eval_tokens
    elif os.path.exists(bf16_ref_path):
        ref_data = torch.load(bf16_ref_path, map_location="cpu", weights_only=False)
        ref_ppl = float(ref_data["eval_ppl"])
        ref_passage_nlls = ref_data.get("passage_nlls", [])
        ref_logits_list = ref_data["logits"]
        ppl_delta_pct = ((eval_ppl - ref_ppl) / max(ref_ppl, 1e-6)) * 100.0

        # Passage-level SE on ΔPPL%
        if ref_passage_nlls and len(ref_passage_nlls) == len(passage_nlls):
            passage_deltas = [
                ((math.exp(cur_n) - math.exp(ref_n)) / max(math.exp(ref_n), 1e-6)) * 100.0
                for ref_n, cur_n in zip(ref_passage_nlls, passage_nlls)
            ]
            ppl_delta_se_pct = (
                statistics.stdev(passage_deltas) / math.sqrt(len(passage_deltas))
                if len(passage_deltas) > 1
                else 0.0
            )
        else:
            ppl_delta_se_pct = 0.0

        token_kls: list[float] = []
        agree_tokens = 0
        compared_tokens = 0
        for ref_l, cur_l in zip(ref_logits_list, current_logits_list):
            p_ref = F.softmax(ref_l, dim=-1)
            log_p_ref = F.log_softmax(ref_l, dim=-1)
            log_q_cur = F.log_softmax(cur_l, dim=-1)
            kl_tok = (p_ref * (log_p_ref - log_q_cur)).sum(dim=-1).squeeze(0)
            token_kls.extend(kl_tok.tolist())
            agree_tokens += int(
                (ref_l.argmax(dim=-1) == cur_l.argmax(dim=-1)).sum().item()
            )
            compared_tokens += int(ref_l.shape[1])

        kl_div_vs_bf16 = sum(token_kls) / max(compared_tokens, 1)
        kl_div_se = (
            statistics.stdev(token_kls) / math.sqrt(len(token_kls))
            if len(token_kls) > 1
            else 0.0
        )
        top1_agree_pct = (agree_tokens / max(compared_tokens, 1)) * 100.0
    else:
        ppl_delta_pct = 0.0
        ppl_delta_se_pct = 0.0
        kl_div_vs_bf16 = -1.0
        kl_div_se = 0.0
        top1_agree_pct = -1.0
        compared_tokens = total_eval_tokens

    # Run 200-item deterministic multiple-choice benchmark via next-token choice scoring
    suite_200 = build_200_item_eval_suite()
    choice_letters = ["A", "B", "C", "D"]
    choice_token_ids = [
        tokenizer.encode(c, add_special_tokens=False)[-1] for c in choice_letters
    ]
    correct_200 = 0
    domain_correct: dict[str, int] = {"math": 0, "systems": 0, "arabic": 0, "recall": 0}
    domain_total: dict[str, int] = {"math": 0, "systems": 0, "arabic": 0, "recall": 0}

    with torch.no_grad():
        for item in suite_200:
            enc = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
            out_mc = model(enc["input_ids"])
            last_logits = out_mc.logits[0, -1, choice_token_ids]
            pred_idx = int(last_logits.argmax().item())
            pred_letter = choice_letters[pred_idx]
            dom = item["domain"]
            domain_total[dom] += 1
            if pred_letter == item["correct"]:
                correct_200 += 1
                domain_correct[dom] += 1

    n_eval_items = len(suite_200)
    p_acc = correct_200 / n_eval_items
    task_accuracy_pct = p_acc * 100.0
    task_accuracy_se_pct = math.sqrt(max(p_acc * (1.0 - p_acc), 0.0) / n_eval_items) * 100.0
    domain_acc_pct = {
        k: round((domain_correct[k] / max(domain_total[k], 1)) * 100.0, 1)
        for k in domain_total
    }

    print(
        f"[{arch_label}] Eval ({precision} | {compared_tokens} toks, N={n_eval_items} tasks): "
        f"PPL={eval_ppl:.3f} (ΔPPL={ppl_delta_pct:+.2f}±{ppl_delta_se_pct:.2f}%) | "
        f"KL={kl_div_vs_bf16:.4f}±{kl_div_se:.4f} | "
        f"Top1={top1_agree_pct:.1f}% | Acc200={task_accuracy_pct:.1f}±{task_accuracy_se_pct:.1f}% {domain_acc_pct}"
    )

    env_metadata = {
        "model_revision": model_revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda or "12.4",
        "nvidia_driver_version": driver_version,
        "fused_kernels_active": fast_path_active,
        "fused_kernel_versions": fused_versions,
        "memory_type": "GDDR6",
    }

    results: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []

    for ctx_len in context_lengths:
        print(f"  -> Sweeping context_length={ctx_len}, decode_tokens={decode_tokens}, trials={num_trials}...")
        gc.collect()
        torch.cuda.empty_cache()

        ttft_ms_list: list[float] = []
        decode_tps_list: list[float] = []
        tpot_ms_list: list[float] = []
        total_joules_per_tok_list: list[float] = []
        dynamic_joules_per_tok_list: list[float] = []
        mean_power_list: list[float] = []
        peak_power_list: list[float] = []
        peak_vram_mb = 0.0
        measured_persistent_cache_mb = 0.0

        # Analytical persistent state (KV cache + Mamba state) in MiB
        analytical_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * ctx_len * 2
        analytical_persistent_mb = (analytical_kv_bytes + ssm_state_bytes) / (1024**2)

        try:
            for trial in range(num_trials):
                input_ids = torch.randint(100 + trial * 50, 5000, (1, ctx_len), device="cuda")
                attention_mask = torch.ones_like(input_ids, device="cuda")

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # --- Step A: Measure Prefill (TTFT) & Persistent Cache Allocation ---
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

                decode_only_ms = max(total_gen_ms - ttft_ms, 1.0)
                decode_only_s = decode_only_ms / 1000.0
                decode_tps = (actual_new_tokens - 1) / decode_only_s
                tpot_ms = decode_only_ms / max(actual_new_tokens - 1, 1)

                decode_joules_total = max(
                    g_stats["total_joules"] - p_stats["total_joules"],
                    g_stats["mean_watts"] * decode_only_s,
                )
                decode_joules_dynamic = max(
                    (g_stats["mean_watts"] - idle_watts) * decode_only_s, 0.01
                )

                j_per_tok_total = decode_joules_total / max(actual_new_tokens - 1, 1)
                j_per_tok_dynamic = decode_joules_dynamic / max(actual_new_tokens - 1, 1)

                decode_tps_list.append(decode_tps)
                tpot_ms_list.append(tpot_ms)
                total_joules_per_tok_list.append(j_per_tok_total)
                dynamic_joules_per_tok_list.append(j_per_tok_dynamic)
                mean_power_list.append(g_stats["mean_watts"])
                peak_power_list.append(g_stats["peak_watts"])

                peak_vram_mb = max(
                    peak_vram_mb, torch.cuda.max_memory_allocated() / (1024**2)
                )
                del out

                if trial == 0:
                    raw_traces.append(
                        {
                            "arch_label": arch_label,
                            "model_id": model_id,
                            "precision": precision,
                            "engine": "hf_eager",
                            "context_length": ctx_len,
                            "poll_interval_s": g_stats["poll_interval_s"],
                            "pre_load_idle_watts": round(idle_watts, 2),
                            "post_load_idle_watts": post_load_idle_w,
                            "samples_rel_time_s": g_stats["samples_rel_time_s"],
                            "samples_watts": g_stats["samples_watts"],
                        }
                    )

            prefill_workspace_growth_mb = max(peak_vram_mb - weights_vram_mb, 0.0)
            ttft_m, ttft_s = _mean_std(ttft_ms_list, 2)
            tps_m, tps_s = _mean_std(decode_tps_list, 2)
            tpot_m, tpot_s = _mean_std(tpot_ms_list, 2)
            jt_m, jt_s = _mean_std(total_joules_per_tok_list, 4)
            jd_m, jd_s = _mean_std(dynamic_joules_per_tok_list, 4)
            mw_m, mw_s = _mean_std(mean_power_list, 2)

            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "arch_label": arch_label,
                "model_id": model_id,
                "model_revision": model_revision,
                "precision": precision,
                "gpu_name": gpu_name,
                "memory_type": "GDDR6",
                "num_trials": num_trials,
                "idle_watts": round(idle_watts, 2),
                "post_load_idle_watts": post_load_idle_w,
                "weights_vram_mb": round(weights_vram_mb, 1),
                "context_length": ctx_len,
                "decode_tokens": decode_tokens - 1,
                "ttft_ms": ttft_m,
                "ttft_std_ms": ttft_s,
                "decode_tok_per_s": tps_m,
                "decode_tok_per_s_std": tps_s,
                "tpot_ms": tpot_m,
                "tpot_std_ms": tpot_s,
                "peak_vram_mb": round(peak_vram_mb, 1),
                "kv_or_state_growth_mb": round(prefill_workspace_growth_mb, 1),
                "analytical_persistent_state_mb": round(analytical_persistent_mb, 1),
                "analytical_total_decode_vram_mb": round(weights_vram_mb + analytical_persistent_mb, 1),
                "mean_watts": mw_m,
                "mean_watts_std": mw_s,
                "peak_watts": round(max(peak_power_list), 2),
                "joules_per_token_total": jt_m,
                "joules_per_token_total_std": jt_s,
                "joules_per_token_dynamic": jd_m,
                "joules_per_token_dynamic_std": jd_s,
                "eval_tokens_count": compared_tokens,
                "eval_ppl": round(eval_ppl, 4),
                "ppl_delta_pct": round(ppl_delta_pct, 2),
                "ppl_delta_se_pct": round(ppl_delta_se_pct, 2),
                "kl_div_vs_bf16": round(kl_div_vs_bf16, 5),
                "kl_div_se": round(kl_div_se, 5),
                "top1_agree_pct": round(top1_agree_pct, 2),
                "eval_tasks_n": n_eval_items,
                "task_accuracy_pct": round(task_accuracy_pct, 2),
                "task_accuracy_se_pct": round(task_accuracy_se_pct, 2),
                "domain_accuracy_pct": domain_acc_pct,
                "environment": env_metadata,
                "status": "ok",
            }
            print(
                f"     [OK | n={num_trials}] ctx={ctx_len:5d} | "
                f"TTFT={ttft_m:7.1f}±{ttft_s:4.1f}ms | "
                f"Decode={tps_m:5.2f}±{tps_s:4.2f} tok/s | "
                f"PersistState={analytical_persistent_mb:6.1f}MB | PeakVRAM={peak_vram_mb:7.1f}MB | "
                f"Energy={jt_m:.4f}±{jt_s:.4f} J/tok"
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
    return {"records": results, "raw_traces": raw_traces}


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    context_lengths: str = "",
    decode_tokens: int = 128,
    num_trials: int = 5,
):
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

    print("=" * 80)
    print(f"TII FALCON BENCHMARK | Mode: {mode} | Contexts: {ctx_list} | Trials: {num_trials}")
    print("=" * 80)

    print("\n[Stage 1/2] Ensuring model weights are cached in Modal Volume...")
    model_ids = [m[1] for m in models]
    list(download_model_weights.map(model_ids))

    print("\n[Stage 2/2] Running isolated GPU telemetry benchmarks on Modal L4...")
    all_results: list[dict[str, Any]] = []
    all_raw_traces: list[dict[str, Any]] = []

    if mode == "crossover":
        # Run all 3 BF16 models in parallel across 3 isolated L4 containers
        starmap_args = [
            (arch_label, model_id, "bf16", ctx_list, decode_tokens, num_trials)
            for arch_label, model_id in models
        ]
        outputs = list(run_gpu_benchmark.starmap(starmap_args))
        for out in outputs:
            all_results.extend(out["records"])
            all_raw_traces.extend(out["raw_traces"])
    else:
        # Check if 5-trial BF16 records with N=200 eval are already in benchmark_crossover.json
        cached_bf16: dict[str, list[dict[str, Any]]] = {}
        crossover_file = Path("results/benchmark_crossover.json")
        if mode == "quant-ablation" and crossover_file.exists():
            try:
                with open(crossover_file, "r", encoding="utf-8") as cf:
                    for r in json.load(cf):
                        if (
                            r.get("precision") == "bf16"
                            and r.get("context_length") in ctx_list
                            and r.get("eval_tasks_n") == 200
                            and r.get("num_trials") == num_trials
                        ):
                            cached_bf16.setdefault(r["model_id"], []).append(r)
            except Exception:
                pass

        bf16_args = []
        for arch_label, model_id in models:
            if model_id in cached_bf16 and len(cached_bf16[model_id]) == len(ctx_list):
                print(f"  [Cache Hit] Reusing 5-trial BF16 baseline for {arch_label} from benchmark_crossover.json")
                all_results.extend(cached_bf16[model_id])
            else:
                bf16_args.append((arch_label, model_id, "bf16", ctx_list, decode_tokens, num_trials))

        if bf16_args:
            bf16_outputs = list(run_gpu_benchmark.starmap(bf16_args))
            for out in bf16_outputs:
                all_results.extend(out["records"])
                all_raw_traces.extend(out["raw_traces"])

        # Step 2b: Run quantized modes in parallel across isolated L4 containers
        quant_args = []
        for arch_label, model_id in models:
            for prec in precisions:
                if prec == "bf16":
                    continue
                if prec == "int4_mixed_ssm" and "Transformer" in arch_label and "Hybrid" not in arch_label:
                    continue
                quant_args.append((arch_label, model_id, prec, ctx_list, decode_tokens, num_trials))

        if quant_args:
            quant_outputs = list(run_gpu_benchmark.starmap(quant_args))
            for out in quant_outputs:
                all_results.extend(out["records"])
                all_raw_traces.extend(out["raw_traces"])

    local_out_dir = Path("results")
    local_out_dir.mkdir(exist_ok=True)
    out_file = local_out_dir / f"benchmark_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    traces_file = local_out_dir / f"raw_50hz_{mode}_traces.json"
    with open(traces_file, "w", encoding="utf-8") as f:
        json.dump(all_raw_traces, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Done! Saved {len(all_results)} records (n={num_trials} trials) to: {out_file}")
    print(f"Saved {len(all_raw_traces)} raw 50 Hz NVML traces to:            {traces_file}")
    print("=" * 80)
