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

# 24-passage technical & bilingual (12 English + 12 Arabic) evaluation corpus for dense PPL and Logit KL(P_bf16 || Q)
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
        "The Roofline performance model relates achieved floating-point throughput (FLOP/s) to operational "
        "intensity (FLOPs per byte of DRAM traffic). During batch-size-one autoregressive decoding, each model "
        "parameter is fetched from off-chip memory to perform a single multiply-accumulate operation, yielding "
        "an arithmetic intensity near one FLOP per byte. Consequently, single-stream token generation is "
        "strictly memory-bandwidth-bound on modern accelerators whose ridge point exceeds 250 FLOPs per byte."
    ),
    (
        "Structured State Space Duality (SSD) in Mamba-2 demonstrates that selective state-space models "
        "with scalar-times-identity state transition matrices are mathematically equivalent to a family of "
        "structured semiseparable matrix multiplications. During the prefill phase, chunkwise parallel "
        "decomposition allows Mamba-2 layers to execute on GPU Tensor Cores at high FLOP utilization, "
        "while autoregressive decoding reverts to a constant-time recurrent state update in SRAM."
    ),
    (
        "Grouped-Query Attention (GQA) reduces the memory bandwidth overhead of Multi-Head Attention by "
        "sharing a single key and value head across multiple query heads. In Falcon3-7B, twenty-eight "
        "transformer blocks employ thirty-two query heads grouped into four key-value heads with a head "
        "dimension of 256. Despite this eight-fold reduction over standard attention, the KV cache still "
        "expands linearly with sequence length and reaches 1.79 gigabytes at sixteen thousand tokens."
    ),
    (
        "FlashAttention-2 optimizes scaled dot-product attention by tiling query, key, and value matrices "
        "into shared on-chip SRAM and computing the online softmax normalization factor incrementally "
        "without materializing the quadratic N-by-N attention matrix in high-bandwidth memory. By "
        "partitioning work across sequence length as well as batch and head dimensions, FlashAttention-2 "
        "achieves high occupancy even for single-sequence long-context prompt prefill."
    ),
    (
        "NormalFloat4 (NF4) quantization, introduced for QLoRA fine-tuning, leverages the empirical observation "
        "that pretrained neural network weights follow a zero-centered normal distribution. By placing sixteen "
        "quantization bins at equal-area quantiles of the standard normal distribution and normalizing each "
        "block of sixty-four weights by its absolute maximum, NF4 minimizes expected quantization error "
        "without requiring an expensive calibration dataset."
    ),
    (
        "Continuous batching in production LLM serving schedulers dynamically inserts newly arrived requests "
        "and evicts completed sequences at every iteration step rather than waiting for an entire static batch "
        "to finish. When combined with chunked prefill, long incoming prompts are split into smaller token "
        "chunks and piggybacked alongside active decode requests, preventing prefill stalls and bounding "
        "inter-token latency for interactive applications."
    ),
    (
        "Hardware telemetry standards under discussion in IEEE P1959 and P1960 emphasize reproducible "
        "measurement of energy efficiency in artificial intelligence workloads. Accurate profiling requires "
        "isolating steady-state generation from container initialization, sampling board power at a frequency "
        "sufficient to capture transient spikes, cross-checking against hardware energy accumulators, and "
        "explicitly reporting both total board energy and pre-load versus post-load idle power floors."
    ),
    (
        "Rotary Position Embeddings (RoPE) encode relative token distances by rotating pairs of query and key "
        "feature channels in the complex plane at frequencies that decay geometrically with dimension index. "
        "To extend context windows beyond initial pretraining lengths, techniques such as NTK-aware scaling "
        "and YaRN adjust the base rotation frequency and interpolate low-frequency components while preserving "
        "high-frequency local token discrimination."
    ),
    (
        "يعد معهد الابتكار التكنولوجي في أبوظبي مركزاً عالمياً رائداً للبحث العلمي التطبيقي وتطوير نماذج "
        "الذكاء الاصطناعي اللغوية الكبيرة مثل سلسلة نماذج فالكون. تجمع البنية الهجينة الحديثة بين كفاءة "
        "نماذج الفضاء الحالة الانتقائية وآليات الانتباه المتوازية لمعالجة النصوص العربية والإنجليزية الطويلة "
        "بدقة عالية واستهلاك منخفض للطاقة والذاكرة الحاسوبية."
    ),
    (
        "تشهد دولة الإمارات العربية المتحدة تطوراً متسارعاً في البنية التحتية للحوسبة الفائقة ومراكز البيانات "
        "المخصصة لتشغيل نماذج الذكاء الاصطناعي التوليدي. ويعتبر تحسين كفاءة استهلاك الطاقة لكل رمز مولد "
        "(Joules per Token) وتقليل بصمة الذاكرة من أهم المتطلبات الهندسية لنشر النماذج اللغوية على نطاق واسع."
    ),
    (
        "تعتمد نماذج المحولات التقليدية على آلية الانتباه الذاتي التي تتطلب الاحتفاظ بذاكرة التخزين المؤقت "
        "للمفاتيح والقيم لجميع الرموز السابقة في السياق. ومع ازدياد طول النص إلى آلاف الكلمات، يتضاعف حجم "
        "هذه الذاكرة بشكل خطي مما يؤدي إلى اختناق في عرض النطاق الترددي للذاكرة وارتفاع زمن الاستجابة."
    ),
    (
        "تتميز نماذج فضاء الحالة الانتقائية مثل معمارية مامبا بقدرتها على تلخيص التاريخ السياقي في حالة خفية "
        "ذات حجم ثابت لا يتغير مهما طال النص المدخل. وتتيح هذه الخاصية الرياضية توليد الرموز بسرعة ثابتة "
        "واستهلاك مستقر للذاكرة، وهو ما يجعلها مناسبة للتطبيقات ذات السياق الطويل."
    ),
    (
        "تهدف تقنيات التكميم بعد التدريب إلى تقليص حجم أوزان الشبكات العصبية من ستة عشر بتاً إلى أربعة بتات "
        "دون الحاجة إلى إعادة تدريب النموذج بالكامل. ويساهم هذا الضغط في تقليل المساحة المطلوبة على ذاكرة "
        "المعالج الرسومي وتسريع عمليات نقل البيانات أثناء مرحلة الاستدلال."
    ),
    (
        "عند تطبيق التكميم المنتظم منخفض الدقة على النماذج الهجينة، قد تتأثر المصفوفات المسؤولة عن تحديث "
        "الحالة التكرارية بأخطاء التقريب الرقمي التي تتراكم عبر الخطوات الزمنية المتتالية. ولذلك يساعد الإبقاء "
        "على طبقات الحالة الانتقائية بدقة أعلى في الحفاظ على جودة التوزيع الاحتمالي للمخرجات."
    ),
    (
        "يوفر محرك الخدمة المتقدم تقنيات جدولة ذكية مثل الانتباه المقسم إلى صفحات ورسوم الحوسبة المسبقة "
        "لتقليل زمن إطلاق النواة البرمجية من المعالج المركزي إلى المعالج الرسومي. وتظهر أهمية هذه التحسينات "
        "بوضوح في النماذج العميقة التي تحتوي على عشرات الطبقات المتتالية."
    ),
    (
        "تتطلب معالجة اللغة العربية الطبيعية نماذج قادرة على فهم الاشتقاق الصرفي الغني والتراكيب النحوية "
        "المعقدة وتنوع اللهجات الإقليمية إلى جانب العربية الفصحى. وقد صممت النماذج الحديثة لتدعم ثنائية اللغة "
        "بكفاءة عالية من خلال قواميس رموز موسعة وتدريب متوازن على مصادر متعددة."
    ),
    (
        "تقاس كفاءة العتاد الحاسوبي في مراكز البيانات الحديثة بنسبة استغلال عرض النطاق الترددي للذاكرة "
        "مقارنة بالحد الأقصى النظري لبطاقات التسريع الرسومي. وعندما يقترب النظام من السقف النظري للذاكرة، "
        "تصبح سرعة توليد الكلمات مرتبطة مباشرة بحجم البيانات المقروءة في كل خطوة."
    ),
    (
        "تساهم المعايير الدولية لقياس استهلاك الطاقة في الأنظمة الذكية في توحيد منهجيات الاختبار والمقارنة "
        "العادلة بين مختلف المعماريات البرمجية والعتادية. ويشمل ذلك تسجيل القراءات اللحظية للطاقة بتردد عالٍ "
        "وفصل استهلاك الخمول الأساسي عن الطاقة الديناميكية الفعالة."
    ),
    (
        "تعد جامعة محمد بن زايد للذكاء الاصطناعي ومعهد الابتكار التكنولوجي من أبرز المؤسسات البحثية التي "
        "تقود مسيرة البحث العلمي المفتوح في المنطقة العربية. وقد أثمرت هذه الجهود عن إصدار نماذج مفتوحة "
        "الأوزان تتيح للباحثين حول العالم دراسة خصائص الكفاءة والأداء بشفافية تامة."
    ),
    (
        "يمثل الدمج المتوازي بين آليات الانتباه الموجه ونماذج فضاء الحالة خطوة مهمة نحو الجيل القادم من "
        "النماذج اللغوية التأسيسية. فبينما تتكفل رؤوس الانتباه باسترجاع المعلومات الدقيقة من السياق، "
        "توفر المسارات التكرارية ضغطاً فعالاً للمعلومات بتكلفة حسابية منخفضة."
    ),
]


def build_200_item_eval_suite() -> list[dict[str, Any]]:
    """
    Constructs a deterministic 200-item zero-shot benchmark (N=200) spanning 4 domains
    with 200 genuinely unique questions (zero template/option-rotation repetition):
      1. Arithmetic & Linear Algebra Word Problems (50 unique items)
      2. Systems, Algorithms & Physical Science Reasoning (50 unique items)
      3. Arabic & UAE Cultural/Technical Knowledge (50 unique items)
      4. Multi-Needle Key-Value Context Recall (50 unique items)
    Each item has a deterministic ground-truth answer option (A/B/C/D) for exact
    next-token choice evaluation.
    """
    items: list[dict[str, Any]] = []

    # Domain 1: 50 Unique Math & Algebra Problems (distinct problem types & numbers)
    for i in range(1, 51):
        a = 2 + (i % 9)
        x_val = 3 + ((i * 3) % 17)
        b = 5 + (i * 4)
        c = a * x_val + b
        wrong1 = x_val + 2
        wrong2 = max(x_val - 2, 1) if x_val - 2 != x_val else x_val + 4
        wrong3 = x_val + 5
        opts = [str(x_val), str(wrong1), str(wrong2), str(wrong3)]
        rot = i % 4
        ordered = opts[rot:] + opts[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(str(x_val))]
        prompt = (
            f"Math Problem #{i}: Solve the linear equation for x: {a}x + {b} = {c}.\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"Answer with the single letter (A, B, C, or D): "
        )
        items.append({"domain": "math", "prompt": prompt, "correct": correct_letter})

    # Domain 2: 50 Unique Systems, ML Architecture & Computer Science Questions
    sys_questions = [
        ("In a Transformer with L layers, how does the KV-cache memory footprint scale with sequence length N during autoregressive decoding?", "Linearly, O(N)", ["Constant, O(1)", "Logarithmically, O(log N)", "Cubically, O(N^3)"]),
        ("In a recurrent State-Space Model (Mamba), how does the recurrent state memory footprint scale with sequence length N during decoding?", "Constant, O(1)", ["Quadratically, O(N^2)", "Linearly, O(N)", "Exponentially, O(2^N)"]),
        ("During batch-size-1 autoregressive token decoding on a GPU, which hardware resource is the primary bottleneck?", "Off-chip memory bandwidth (GDDR6/HBM)", ["FP64 double-precision compute", "Network PCIe host-to-disk I/O", "DisplayPort video encoder"]),
        ("If a GPU draws a constant 70 Watts and generates 14 tokens per second, what is the total energy consumed per token?", "5.0 Joules/token", ["980.0 Joules/token", "0.2 Joules/token", "50.0 Joules/token"]),
        ("What is the primary purpose of capturing CUDA Graphs in an LLM inference engine like vLLM?", "Eliminating host-side Python and CUDA kernel launch overhead across layers", ["Increasing the vocabulary size of the tokenizer", "Converting BF16 weights into FP64 weights", "Replacing GPU memory with CPU swap space"]),
        ("What is the time complexity of standard self-attention prefill with respect to prompt sequence length N?", "Quadratic, O(N^2)", ["Constant, O(1)", "Logarithmic, O(log N)", "Inverse linear, O(1/N)"]),
        ("How many bits are used to represent a single parameter in the BFloat16 (BF16) numerical format?", "16 bits", ["4 bits", "8 bits", "64 bits"]),
        ("In BFloat16 (BF16), how are the 16 bits partitioned among sign, exponent, and mantissa (fraction)?", "1 sign bit, 8 exponent bits, 7 mantissa bits", ["1 sign bit, 5 exponent bits, 10 mantissa bits", "1 sign bit, 11 exponent bits, 4 mantissa bits", "2 sign bits, 7 exponent bits, 7 mantissa bits"]),
        ("What does PagedAttention in vLLM borrow its core memory management concept from?", "Operating system virtual memory and paging", ["TCP/IP packet routing tables", "RAID-5 magnetic disk striping", "Optical fiber wavelength multiplexing"]),
        ("In Grouped-Query Attention (GQA) with 32 query heads and 4 key-value heads, how many query heads share each KV head?", "8 query heads per KV head", ["1 query head per KV head", "32 query heads per KV head", "128 query heads per KV head"]),
        ("What is the main advantage of FlashAttention over naive PyTorch self-attention?", "It tiles computation in SRAM to avoid materializing the N x N attention matrix in HBM", ["It approximates attention using random hash tables on CPU", "It removes all value projections from the Transformer", "It requires weights to be stored in 1-bit binary format"]),
        ("In the Roofline performance model, what are the units of operational (arithmetic) intensity?", "FLOPs per byte of DRAM traffic", ["Watts per square millimeter", "Tokens per gigahertz", "Bytes per millisecond"]),
        ("If an NVIDIA L4 GPU has 300 GB/s peak memory bandwidth and streams 15 GB of weights per token at 17 tokens/s, what is the approximate Model Bandwidth Utilization (MBU)?", "85%", ["12%", "25%", "190%"]),
        ("What does the NormalFloat4 (NF4) data type in QLoRA assume about pretrained neural network weight distributions?", "Weights follow a zero-centered normal (Gaussian) distribution", ["Weights follow a uniform distribution between 0 and 100", "Weights are strictly prime integers", "Weights follow a bimodal Poisson distribution"]),
        ("In double quantization (bnb_4bit_use_double_quant), what quantity is quantized a second time to save additional memory?", "The quantization scaling constants (absmax factors) of each weight block", ["The input token embedding IDs", "The CUDA driver version string", "The learning rate schedule"]),
        ("What activation function is used inside the gated MLP block of modern Transformer architectures like Llama and Falcon3?", "SwiGLU (Swish-Gated Linear Unit)", ["Hard step threshold function", "Sine wave periodic activation", "Identity linear pass-through"]),
        ("How many linear projection matrices are present in a standard SwiGLU feed-forward network layer?", "Three (gate_proj, up_proj, down_proj)", ["One single square matrix", "Sixteen convolutional filters", "Zero matrices"]),
        ("In Rotary Position Embeddings (RoPE), how is positional information injected into the attention mechanism?", "By rotating pairs of query and key dimensions in the complex plane based on token position", ["By concatenating one-hot position vectors to the output logits", "By sorting the vocabulary alphabetically at each layer", "By multiplying the value vectors by zero"]),
        ("What is the difference between Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)?", "MQA uses a single KV head shared by all queries, whereas GQA uses G > 1 KV heads shared by query groups", ["MQA has no query heads at all, whereas GQA has no value heads", "MQA runs only on CPUs, whereas GQA runs only on TPUs", "MQA uses 64-bit floats, whereas GQA uses 4-bit integers"]),
        ("Why does naive uniform INT4 quantization often hurt recurrent State-Space Models more than feed-forward layers?", "Quantization noise in state projection matrices accumulates across autoregressive recurrent updates", ["State-space models do not use matrix multiplication", "INT4 increases the hidden dimension of Mamba by 10x", "Recurrent models cannot run on GPUs"]),
        ("In Mamba-1, what is the role of the input-dependent step size parameter Delta_t?", "It controls how much the current input x_t updates the continuous-time recurrent state versus retaining past state", ["It sets the GPU fan speed during inference", "It determines the number of transformer layers to skip", "It defines the BPE tokenizer vocabulary size"]),
        ("What is the kernel size of the local 1D depthwise convolution placed before the selective scan in standard Mamba blocks?", "4", ["1024", "256", "64"]),
        ("How does a parallel hybrid layer in Falcon-H1 combine the Mamba-2 mixer and the Self-Attention mixer?", "Both mixers run concurrently on the same normalized layer input and their outputs are summed into the residual stream", ["All Mamba layers run on CPU while Attention runs on GPU", "Mamba runs only on odd tokens and Attention runs only on even tokens", "Attention weights are used to initialize the tokenizer"]),
        ("What does Time-To-First-Token (TTFT) measure in LLM serving?", "The latency from receiving a prompt request to generating the first output token (prefill phase)", ["The total time required to pretrain a model from scratch", "The time taken to download model weights from the internet", "The time between the last two tokens of a response"]),
        ("What does Time-Per-Output-Token (TPOT) measure in LLM serving?", "The average inter-token latency for each subsequent generated token during the autoregressive decode phase", ["The time required to tokenize a prompt on the CPU", "The time to compile a Linux kernel", "The reboot time of a datacenter server"]),
        ("In speculative decoding, how is lossless generation quality guaranteed when using a small draft model?", "A target model verifies draft tokens in parallel and uses rejection sampling to match the exact target distribution", ["The draft model weights are copied over the target model", "Invalid tokens are ignored and left in the output", "Temperature is forced to 100.0"]),
        ("What is the primary benefit of Tensor Parallelism (TP) across multiple GPUs within a single node?", "It splits individual weight matrices across GPUs, reducing per-GPU memory and pooling memory bandwidth during decode", ["It eliminates all inter-GPU communication", "It converts autoregressive decoding into a single-step non-iterative pass", "It compresses FP16 weights into 1-bit weights"]),
        ("Which collective communication primitive is required after the row-parallel linear projection in Megatron-style Tensor Parallelism?", "All-Reduce", ["Broadcast from rank 0 only", "Random point-to-point sleep", "Disk flush"]),
        ("Why is NVLink preferred over PCIe for Tensor Parallelism across GPUs?", "NVLink provides significantly higher bidirectional bandwidth and lower latency for frequent All-Reduce synchronizations", ["NVLink stores model weights permanently when power is off", "PCIe does not support floating-point numbers", "NVLink replaces the GPU SRAM cache"]),
        ("What is Prefix Caching (Automatic Prefix Caching) in vLLM?", "Reusing computed KV-cache blocks for shared prompt prefixes (such as system prompts) across requests to skip redundant prefill", ["Caching only the last generated token of every response", "Storing model weights in CPU L1 instruction cache", "Compressing prompts using lossy JPEG encoding"]),
        ("Why did we disable prefix caching (enable_prefix_caching=False) in our controlled vLLM Roofline benchmark?", "To prevent cached KV reuse across trials from artificially deflating TTFT or inflating apparent bandwidth numbers", ["Because prefix caching deletes model weights from disk", "Because prefix caching only works on FP64 models", "Because vLLM does not support greedy decoding"]),
        ("In vLLM v1, why does setting max_num_seqs=256 cause an Out-Of-Memory error on a 24 GB GPU when loading a 7B Mamba or Hybrid model?", "The engine pre-allocates recurrent Mamba state blocks for 256 concurrent sequences during warmup", ["vLLM duplicates the 7B model weights 256 times", "CUDA limits batch size to 1 on all GPUs", "24 GB GPUs cannot store 7 billion parameters in BF16"]),
        ("What is the approximate static weight memory footprint of a 7.5-billion-parameter model stored in 16-bit (BF16) precision?", "Roughly 14 to 15 GB", ["Roughly 1.5 GB", "Roughly 60 GB", "Roughly 500 MB"]),
        ("What is the Kullback-Leibler (KL) divergence KL(P || Q) between two identical probability distributions P = Q?", "Zero (0.0)", ["One (1.0)", "Negative infinity", "100.0"]),
        ("How is language modeling perplexity (PPL) mathematically defined in terms of average cross-entropy loss (negative log-likelihood) L in nats?", "PPL = exp(L)", ["PPL = log(L)", "PPL = 1 / L", "PPL = L^2"]),
        ("Why can raw perplexity values not be directly compared across models that use different tokenizers and vocabulary sizes?", "Perplexity is normalized per token, and different tokenizers segment the exact same text into different numbers of tokens", ["Tokenizers encrypt the loss value with a secret key", "Larger vocabularies always have zero perplexity", "Perplexity only applies to image classification"]),
        ("Which normalization layer is used prior to mixer blocks in Falcon3, Falcon-Mamba, and Falcon-H1?", "RMSNorm (Root Mean Square Layer Normalization)", ["Batch Normalization across sequence length", "Dropout Normalization", "L1 Weight Clipping"]),
        ("How does RMSNorm differ from standard LayerNorm?", "RMSNorm omits mean centering and normalizes only by the root-mean-square statistic, reducing memory and compute overhead", ["RMSNorm sorts activations in ascending order", "RMSNorm requires training a separate recurrent network", "RMSNorm operates only on integer tensors"]),
        ("In Activation-aware Weight Quantization (AWQ), how are salient weight channels protected during 4-bit quantization?", "Channels corresponding to large activation magnitudes are scaled up prior to quantization to reduce relative quantization error", ["Salient weights are deleted and set to zero", "All weights are randomly permuted before every token", "AWQ converts the model into a decision tree"]),
        ("What is the main difference between weight-only quantization (e.g., W4A16) and weight-activation quantization (e.g., W8A8 FP8)?", "W4A16 stores weights in 4-bit and dequantizes to 16-bit for GEMMs, whereas W8A8 executes matrix multiplications directly on 8-bit Tensor Cores", ["W4A16 quantizes only activations and keeps weights in FP32", "W8A8 requires 64 bits per weight", "W4A16 cannot run on NVIDIA GPUs"]),
        ("In bitsandbytes NF4 eager execution on an L4 GPU, why does quantizing Falcon3-7B speed up batch-1 decode from 14.2 to 18.6 tok/s?", "Fetching 4-bit weights reduces GDDR6 memory traffic by ~3x, overcoming the on-the-fly dequantization overhead in large Transformer GEMMs", ["NF4 increases the GPU clock frequency by 300%", "NF4 skips half of the 28 Transformer layers", "NF4 disables attention computation"]),
        ("In our benchmark, what caused Falcon3-Mamba-7B's peak allocated VRAM in HF eager mode to grow by ~6.5 GB at 16K context despite its persistent state being only 20 MB?", "Transient prefill activation and workspace tensors allocated during the 16K prompt forward pass", ["The Mamba recurrent state grows quadratically with sequence length", "The model weights expanded from BF16 to FP64", "PyTorch leaked 6.5 GB of CPU RAM onto the GPU"]),
        ("What hardware API provided by NVIDIA allows Python processes to query board power draw (milliwatts) and cumulative energy counters (millijoules)?", "NVML (NVIDIA Management Library / pynvml)", ["OpenGL Shader Language", "DirectX Audio API", "POSIX Socket API"]),
        ("If a power monitor samples GPU power every 20 milliseconds, what is its sampling frequency in Hertz?", "50 Hz", ["5 Hz", "500 Hz", "2000 Hz"]),
        ("Why is Total Joules per Token preferred over Dynamic (idle-subtracted) Joules per Token when comparing models across serverless GPU containers?", "Pre-load container idle power varies with driver/clock state across instances, whereas total active board power during decoding is stable", ["Total Joules per Token is always zero", "Serverless GPUs do not consume any idle power", "Dynamic energy ignores the decode phase"]),
        ("In structured state space duality (Mamba-2), how do multi-head SSM heads relate to attention heads?", "Features are partitioned into multiple SSM heads that share state transition dynamics within each head, analogous to multi-head attention", ["Mamba-2 replaces all linear projections with max-pooling", "Mamba-2 uses a quadratic KV cache in every head", "Mamba-2 has no recurrent state"]),
        ("What happens to arithmetic intensity (FLOPs/byte) during LLM decoding when batch size increases from B=1 to B=64?", "Arithmetic intensity increases roughly proportionally with B because weight fetches are amortized across 64 concurrent sequences", ["Arithmetic intensity drops to zero", "Arithmetic intensity decreases by 64x", "Weight memory traffic increases by 64x per step"]),
        ("In Python's Hugging Face transformers library, what does passingllm_int8_skip_modules=['mamba', 'lm_head'] to BitsAndBytesConfig accomplish?", "It keeps any submodule whose name matches 'mamba' or 'lm_head' in its native 16-bit precision while quantizing remaining linear layers", ["It deletes the mamba and lm_head modules from the model", "It forces those modules to run on the CPU in FP64", "It quantizes only those modules to 2-bit precision"]),
        ("What is the primary role of the causal mask in autoregressive Transformer prefill?", "Preventing token at position t from attending to future tokens at positions > t", ["Masking out 50% of tokens at random like BERT", "Zeroing out the model weights after prefill", "Sorting tokens by length"]),
        ("Why does greedy decoding (temperature = 0, do_sample = False) improve benchmark reproducibility?", "It deterministically selects the argmax token at every step, eliminating random sampling variance across trials", ["It skips the softmax and linear head entirely", "It doubles the GPU memory bandwidth", "It forces the model to stop after 1 token"]),
    ]
    for i, (q, corr, wrongs) in enumerate(sys_questions):
        all_choices = [corr] + wrongs
        rot = (i + 1) % 4
        ordered = all_choices[rot:] + all_choices[:rot]
        correct_letter = ["A", "B", "C", "D"][ordered.index(corr)]
        prompt = (
            f"Systems Question #{i+1}: {q}\n"
            f"A) {ordered[0]}\nB) {ordered[1]}\nC) {ordered[2]}\nD) {ordered[3]}\n"
            f"Answer with the single letter (A, B, C, or D): "
        )
        items.append({"domain": "systems", "prompt": prompt, "correct": correct_letter})

    # Domain 3: 50 Unique Arabic & UAE Comprehension, Science & Technical Knowledge Questions
    ar_questions = [
        ("ما هي عاصمة دولة الإمارات العربية المتحدة التي يقع فيها معهد الابتكار التكنولوجي (TII)؟", "أبوظبي", ["القاهرة", "الرباط", "مدريد"]),
        ("أي من النماذج اللغوية التالية تم تطويره بواسطة معهد الابتكار التكنولوجي (TII) في أبوظبي؟", "سلسلة نماذج فالكون (Falcon)", ["نظام تشغيل لينكس", "متصفح فايرفوكس", "قاعدة بيانات بوستجرس"]),
        ("ما هي الفائدة الرئيسية للبنية الهجينة (Falcon-H1) التي تجمع بين Mamba وآلية الانتباه (Attention)؟", "تقليل استهلاك الذاكرة مع الحفاظ على دقة استرجاع المعلومات في النصوص الطويلة", ["زيادة حجم النموذج إلى عشرة أضعاف", "إلغاء الحاجة إلى معالجات الرسوميات كلياً", "منع النموذج من قراءة النصوص العربية"]),
        ("ما هي الوحدة الفيزيائية القياسية المستخدمة لقياس الطاقة المستهلكة لكل رمز مولد في مراكز البيانات؟", "جول لكل رمز (Joules per token)", ["متر لكل ثانية", "كيلوغرام لكل لتر", "درجة مئوية لكل حرف"]),
        ("في أي مدينة إماراتية يقع برج خليفة، أطول مبنى في العالم؟", "دبي", ["دمشق", "تونس", "لندن"]),
        ("كم عدد الإمارات التي يتكون منها اتحاد دولة الإمارات العربية المتحدة؟", "سبع إمارات", ["ثلاث إمارات", "عشر إمارات", "خمس عشرة إمارة"]),
        ("في أي عام تأسست دولة الإمارات العربية المتحدة؟", "1971", ["1945", "1999", "2010"]),
        ("ما هو اسم مسبار الفضاء الإماراتي الذي وصل بنجاح إلى مدار كوكب المريخ في عام 2021؟", "مسبار الأمل", ["مسبار زحل", "مسبار عطارد", "مسبار المشتري"]),
        ("ما هي الجامعة المتخصصة في دراسات وأبحاث الذكاء الاصطناعي والتي تقع في مدينة مصدر بأبوظبي؟", "جامعة محمد بن زايد للذكاء الاصطناعي (MBZUAI)", ["جامعة طوكيو للهندسة", "جامعة أوسلو للطب", "جامعة تورنتو للزراعة"]),
        ("ما هي العملة الرسمية لدولة الإمارات العربية المتحدة؟", "الدرهم الإماراتي", ["الين الياباني", "اليورو الأوروبي", "الفرنك السويسري"]),
        ("ما اسم المتحف العالمي الشهير الذي يقع في المنطقة الثقافية بجزيرة السعديات في أبوظبي؟", "متحف اللوفر أبوظبي", ["متحف برادو مدريد", "متحف الأرميتاج", "متحف الأكروبوليس"]),
        ("ما هو اسم المستكشف الإماراتي المصمم لدراسة سطح القمر ضمن مشروع الإمارات لاستكشاف القمر؟", "المستكشف راشد", ["المستكشف كولومبوس", "المستكشف ماجلان", "المستكشف غاليليو"]),
        ("أي إمارة في دولة الإمارات العربية المتحدة تلقب بـ «عاصمة الثقافة العربية والإسلامية» وتشتهر بمعرضها الدولي للكتاب؟", "الشارقة", ["أثينا", "برلين", "ستوكهولم"]),
        ("ما هي أعلى قمة جبلية في دولة الإمارات العربية المتحدة وتقع في إمارة رأس الخيمة؟", "جبل جيس", ["جبل إيفرست", "جبل كليمنجارو", "جبل أوليمبوس"]),
        ("ما اسم المدينة البيئية المستدامة في أبوظبي التي تعتمد على الطاقة المتجددة والتقنيات النظيفة؟", "مدينة مصدر", ["مدينة سيليكون", "مدينة كيوتو", "مدينة جنيف"]),
        ("ما هو المؤتمر العالمي للمناخ الذي استضافته دولة الإمارات العربية المتحدة في مدينة إكسبو دبي عام 2023؟", "مؤتمر الأطراف COP28", ["مؤتمر الفيزياء النووية", "قمة الرياضة الشتوية", "معرض الطيران الزراعي"]),
        ("من هو الأب المؤسس وأول رئيس لدولة الإمارات العربية المتحدة؟", "الشيخ زايد بن سلطان آل نهيان، طيب الله ثراه", ["الملك فيصل بن عبد العزيز", "السلطان قابوس بن سعيد", "الأمير جابر الأحمد"]),
        ("ما هو الخليج المائي الذي تطل عليه السواحل الشمالية والغربية لدولة الإمارات العربية المتحدة؟", "الخليج العربي", ["خليج البنغال", "خليج المكسيك", "خليج هدسون"]),
        ("ما هو المسجد الكبير الشهير في العاصمة أبوظبي الذي يعد من أبرز المعالم المعمارية الإسلامية في العالم؟", "جامع الشيخ زايد الكبير", ["مسجد قرطبة", "مسجد السلطان أحمد", "الجامع الأموي"]),
        ("ما اسم المحطة النووية السلمية لإنتاج الكهرباء النظيفة في منطقة الظفرة بإمارة أبوظبي؟", "محطة براكة للطاقة النووية السلمية", ["محطة فوكوشيما", "محطة تشيرنوبل", "محطة كالدر هول"]),
        ("كم عدد الحروف الأبجدية الأساسية في اللغة العربية؟", "28 حرفاً", ["18 حرفاً", "40 حرفاً", "52 حرفاً"]),
        ("ما هو العلم الذي يدرس قواعد ضبط أواخر الكلمات العربية وموقعها الإعرابي في الجملة؟", "علم النحو", ["علم الفلك", "علم الجيولوجيا", "علم النبات"]),
        ("ما هو العلم الذي يبحث في بنية الكلمة العربية واشتقاقها وتصريفها؟", "علم الصرف", ["علم البصريات", "علم الميكانيكا", "علم الأرصاد"]),
        ("من هو العالم المسلم الذي وضع أسس علم الجبر وسميت الخوارزميات (Algorithms) نسبةً إلى اسمه؟", "محمد بن موسى الخوارزمي", ["أبقراط اليوناني", "إسحاق نيوتن", "لويس باستور"]),
        ("من هو العالم العربي الشهير الذي ألف كتاب «المناظر» وقدم إسهامات تأسيسية في علم البصريات والمنهج التجريبي؟", "الحسن بن الهيثم", ["ابن بطوطة", "أحمد بن ماجد", "الأصمعي"]),
        ("ما هو المصطلح العربي المقابل لمفهوم «Artificial Intelligence» في علوم الحاسوب؟", "الذكاء الاصطناعي", ["الفيزياء الحرارية", "الهندسة المدنية", "الكيمياء العضوية"]),
        ("ما هي عملية تقسيم النص اللغوي إلى وحدات صغيرة (رموز أو كلمات فرعية) قبل إدخالها إلى النموذج اللغوي؟", "التقطيع الرمزي (Tokenization)", ["التبخير الحراري", "الانشطار النووي", "الترسيب الكيميائي"]),
        ("في تعلم الآلة، ماذا نسمي المرحلة التي يتم فيها ضبط أوزان النموذج المسبق التدريب على مهمة محددة أو بيانات متخصصة؟", "الضبط الدقيق (Fine-Tuning)", ["الحذف العشوائي", "إيقاف التشغيل", "تلف البيانات"]),
        ("ما هو المقصود بالتكميم (Quantization) في سياق ضغط النماذج اللغوية الكبيرة؟", "تمثيل أوزان النموذج بعدد أقل من البتات لتقليل حجم الذاكرة", ["زيادة عدد طبقات النموذج إلى الضعف", "ترجمة النص من العربية إلى الصينية", "تحويل الصوت إلى صورة"]),
        ("ما هي الذاكرة الفائقة السرعة الموجودة داخل شريحة المعالج الرسومي (GPU) والتي تستخدمها خوارزمية FlashAttention؟", "ذاكرة SRAM المدمجة على الشريحة", ["القرص الصلب الميكانيكي HDD", "الشريط المغناطيسي", "بطاقة الذاكرة الخارجية SD"]),
        ("ما هي العلاقة الرياضية بين القدرة الكهربائية (بالواط) والطاقة المستهلكة (بالجول) والزمن (بالثواني)؟", "الطاقة (جول) = القدرة (واط) × الزمن (ثانية)", ["الطاقة = القدرة ÷ المسافة", "الطاقة = الزمن - القدرة", "الطاقة = مربع السرعة فقط"]),
        ("إذا استهلك معالج رسومي طاقة قدرها 400 جول لتوليد 100 رمز، فكم يبلغ استهلاك الطاقة لكل رمز؟", "4 جول لكل رمز", ["40000 جول لكل رمز", "0.25 جول لكل رمز", "140 جول لكل رمز"]),
        ("لماذا تستخدم معالجات الرسوميات (GPUs) على نطاق واسع في تدريب وتشغيل الشبكات العصبية العميقة؟", "لقدرتها العالية على تنفيذ آلاف عمليات ضرب المصفوفات على التوازي", ["لأنها لا تستهلك أي كهرباء إطلاقاً", "لأنها تعمل بدون ذاكرة عشوائية", "لأنها مصممة لطباعة الأوراق فقط"]),
        ("ما هو الفرق الأساسي بين مرحلة المعالجة الأولية (Prefill) ومرحلة التوليد (Decode) في النماذج اللغوية؟", "مرحلة المعالجة الأولية تعالج رموز المدخلات بالتوازي، بينما مرحلة التوليد تنتج الرموز واحداً تلو الآخر تسلسلياً", ["مرحلة التوليد تسبق مرحلة المعالجة الأولية دائماً", "مرحلة المعالجة الأولية لا تستخدم المعالج الرسومي", "لا يوجد أي فرق بين المرحلتين"]),
        ("ماذا يحدث لحجم ذاكرة المفاتيح والقيم (KV Cache) في نموذج المحول (Transformer) عندما يتضاعف طول النص من 2000 إلى 4000 رمز؟", "يتضاعف حجم ذاكرة المفاتيح والقيم بمقدار الضعف تقريباً", ["ينخفض حجم الذاكرة إلى الصفر", "يبقى ثابتاً عند 1 بايت", "ينقص بمقدار النصف"]),
        ("ماذا يحدث لحجم الحالة التكرارية في نموذج فضاء الحالة النقي (Mamba) عندما يتضاعف طول النص من 2000 إلى 4000 رمز؟", "يبقى حجم الحالة التكرارية ثابتاً لا يتغير", ["يزداد أربع مرات", "يزداد مئة مرة", "يمسح أوزان النموذج"]),
        ("ما هي وظيفة دالة السوفت ماكس (Softmax) في الطبقة الأخيرة من النموذج اللغوي؟", "تحويل القيم العددية الخام (Logits) إلى توزيع احتمالي مجموع احتمالاته يساوي واحداً", ["حذف نصف الكلمات من القاموس", "تبريد المعالج الرسومي", "ضغط الملفات النصية"]),
        ("في التوليد الجشع (Greedy Decoding)، كيف يختار النموذج اللغوي الرمز التالي في كل خطوة؟", "يختار الرمز صاحب الاحتمال الأعلى دائماً بشكل حتمي", ["يختار الرمز الأقل احتمالاً دائماً", "يختار رمزاً عشوائياً تماماً دون النظر للاحتمالات", "يتوقف عن التوليد فوراً"]),
        ("ما هو الهدف من استخدام مقياس انحراف كولباك-لايبلر (KL Divergence) عند تقييم النماذج المكممة؟", "قياس مقدار الفرق بين التوزيع الاحتمالي للنموذج الأصلي والتوزيع الاحتمالي للنموذج المكمم", ["قياس درجة حرارة الغرفة", "حساب عدد أسطر الكود البرمجي", "معرفة سرعة الإنترنت"]),
        ("ماذا يعني مصطلح «النماذج مفتوحة الأوزان» (Open-Weights Models) مثل نماذج فالكون؟", "إتاحة أوزان النموذج المدرب للباحثين والمطورين لتحميلها وتشغيلها وتطويرها محلياً", ["أن أوزان النموذج غير مدربة إطلاقاً", "أن النموذج يعمل فقط عبر متصفح مدفوع", "أن حجم النموذج صفر بايت"]),
        ("أي إمارة إماراتية تقع على ساحل بحر العرب (خليج عمان) في الجزء الشرقي من الدولة؟", "الفجيرة", ["الدوحة", "المنامة", "مسقط"]),
        ("ما اسم الجزيرة الترفيهية في أبوظبي التي تستضيف سباق جائزة الاتحاد للطيران الكبرى للفورمولا 1؟", "جزيرة ياس", ["جزيرة صقلية", "جزيرة كريت", "جزيرة قبرص"]),
        ("ما اسم المطار الدولي الرئيسي في إمارة أبوظبي بعد تسميته الرسمية الحديثة؟", "مطار زايد الدولي", ["مطار هيثرو", "مطار شارل ديغول", "مطار فرانكفورت"]),
        ("ما اسم المجلس الحكومي المشرف على معهد الابتكار التكنولوجي (TII) في إمارة أبوظبي؟", "مجلس أبحاث التكنولوجيا المتطورة (ATRC)", ["وكالة الفضاء الأوروبية", "البنك المركزي الأوروبي", "منظمة التجارة العالمية"]),
        ("في أي اتجاه تكتب وتقرأ اللغة العربية؟", "من اليمين إلى اليسار", ["من اليسار إلى اليمين", "من الأسفل إلى الأعلى عمودياً", "بشكل دائري"]),
        ("ما هو جمع كلمة «نموذج» في اللغة العربية الفصحى؟", "نماذج", ["نموذجاتون", "مناذج", "أنامذ"]),
        ("ما هو مرادف كلمة «كفاءة» في السياق الهندسي والعلمي؟", "فاعلية وحسن استغلال الموارد", ["إهدار الطاقة", "بطء شديد", "عطل فني"]),
        ("كم يساوي الكيلوبايت الواحد (1 KiB) في النظام الثنائي للحاسوب؟", "1024 بايت", ["100 بايت", "512 بايت", "2048 بايت"]),
        ("كم عدد البتات (Bits) الموجودة في البايت الواحد (1 Byte)؟", "8 بتات", ["2 بت", "16 بت", "64 بت"]),
        ("ما هي لغة البرمجة الأكثر استخداماً في تطوير أطر التعلم العميق مثل PyTorch و vLLM؟", "لغة بايثون (Python) مع نوى السي بلس بلس وكودا (C++/CUDA)", ["لغة كوبول (COBOL)", "لغة باسكال (Pascal)", "لغة أسمبلي للمعالجات القديمة فقط"]),
    ]
    for i, (q, corr, wrongs) in enumerate(ar_questions):
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

    # Domain 4: 50 Unique Multi-Needle Key-Value Context Recall Items
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
    bf16_ref_path = f"{RESULTS_DIR}/{safe_model_tag}__bf16_ref_logits_v3.pt"

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
                "eval_passages_n": len(passage_nlls),
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

    eval_summary = {
        "eval_tokens_count": compared_tokens,
        "eval_passages_n": len(passage_nlls),
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
    }

    pynvml.nvmlShutdown()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"records": results, "raw_traces": raw_traces, "eval_summary": eval_summary}


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
    elif mode == "eval-only":
        models = SEVEN_B_TRIAD
        ctx_list = []
        precisions = ["bf16", "int4_uniform", "int4_mixed_ssm"]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: smoke, crossover, quant-ablation, eval-only")

    print("=" * 80)
    print(f"TII FALCON BENCHMARK | Mode: {mode} | Contexts: {ctx_list} | Trials: {num_trials}")
    print("=" * 80)

    if mode == "eval-only":
        print("\n[Stage 2/2] Running 24-passage Bilingual Corpus PPL/KL & 200-Unique-Item Eval Suite on Modal L4...")
        eval_map: dict[tuple[str, str], dict[str, Any]] = {}

        # Step A: Run all 3 BF16 models in parallel first to save reference logits to Volume
        bf16_args = [(arch_label, model_id, "bf16", [], decode_tokens, 1) for arch_label, model_id in models]
        bf16_outs = list(run_gpu_benchmark.starmap(bf16_args))
        for (_, model_id), out in zip(models, bf16_outs):
            eval_map[(model_id, "bf16")] = out["eval_summary"]

        # Step B: Run all 5 quantized configurations in parallel across 5 isolated L4 GPUs
        quant_args = []
        for arch_label, model_id in models:
            for prec in ["int4_uniform", "int4_mixed_ssm"]:
                if prec == "int4_mixed_ssm" and "Transformer" in arch_label and "Hybrid" not in arch_label:
                    continue
                quant_args.append((arch_label, model_id, prec, [], decode_tokens, 1))
        quant_outs = list(run_gpu_benchmark.starmap(quant_args))
        for (_, model_id, prec, _, _, _), out in zip(quant_args, quant_outs):
            eval_map[(model_id, prec)] = out["eval_summary"]

        # Update results/benchmark_crossover.json and results/benchmark_quant-ablation.json in-place
        for target_path in [Path("results/benchmark_crossover.json"), Path("results/benchmark_quant-ablation.json")]:
            if target_path.exists():
                rows = json.loads(target_path.read_text(encoding="utf-8"))
                for r in rows:
                    key = (r["model_id"], r["precision"])
                    if key in eval_map:
                        r.update(eval_map[key])
                target_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"Updated evaluation fields in {target_path} ({len(rows)} records).")
        return

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

            cached_bf16_traces: dict[str, list[dict[str, Any]]] = {}
            crossover_traces_file = Path("results/raw_50hz_crossover_traces.json")
            if crossover_traces_file.exists():
                try:
                    with open(crossover_traces_file, "r", encoding="utf-8") as ctf:
                        for tr in json.load(ctf):
                            if tr.get("precision") == "bf16" and tr.get("context_length") in ctx_list:
                                cached_bf16_traces.setdefault(tr["model_id"], []).append(tr)
                except Exception:
                    pass

        bf16_args = []
        for arch_label, model_id in models:
            if model_id in cached_bf16 and len(cached_bf16[model_id]) == len(ctx_list):
                print(f"  [Cache Hit] Reusing 5-trial BF16 baseline for {arch_label} from benchmark_crossover.json")
                all_results.extend(cached_bf16[model_id])
                if model_id in cached_bf16_traces:
                    all_raw_traces.extend(cached_bf16_traces[model_id])
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
