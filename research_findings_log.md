# Research & Engineering Findings Log
**Project:** *The 3-Architecture Crossover & Hybrid Quantization Study (`Falcon3` vs. `Falcon3-Mamba` vs. `Falcon-H1`)*  
**Author:** Harith Ibrahim ([ORCID: `0009-0008-2204-7324`](https://orcid.org/0009-0008-2204-7324) · [Google Scholar](https://scholar.google.co.uk/citations?user=D0ZI6BoAAAAJ))  
**GitHub Repository:** [`HarithSami/falcon-hybrid-crossover-benchmark`](https://github.com/HarithSami/falcon-hybrid-crossover-benchmark)  
**Purpose:** Living lab notebook documenting empirical findings, model architecture behaviors, quantization parameters, CUDA kernel/ABI discoveries, and hardware energy telemetry insights for our benchmark publication and open-source **vLLM Energy/MFU Instrumentation** release.

---

## 0. v2 Re-Run After External Review (2026-11-01)

Following external peer review, we executed a rigorous **5-trial re-run ($n=5$)** with randomized prompt sequences per trial, expanded our functional quality probe from 5 items to a 200-item templated suite across 4 domains, and synchronized 50 Hz NVML power sampling with hardware cumulative energy counter telemetry on isolated Modal NVIDIA L4 containers.

> [!IMPORTANT]
> **Single Source of Truth:** All authoritative numbers are documented in [`results/RESULTS.md`](results/RESULTS.md). The preliminary v1 tables below in Section 3 are preserved for historical provenance but are **superseded**.

### 0.1 Summary of Withdrawn v1 Claims
1. **"Uniform INT4 collapses Falcon3-Mamba accuracy 80% $\rightarrow$ 40%" — WITHDRAWN.**  
   The v1 observation was an artifact of an under-powered 5-question probe ($4/5 \rightarrow 2/5$). In the v2 evaluation ($N=200$ templated items), probe accuracy differences across precisions remain within $\sim 1\text{--}1.5$ standard errors for all three models (`Falcon3-Mamba-7B`: $48.0\% \rightarrow 53.0\% \rightarrow 47.5\%$; `Falcon-H1-7B`: $85.0\% \rightarrow 87.5\% \rightarrow 87.0\%$).
2. **"int4_mixed_ssm: zero accuracy loss / 100% retained" — WITHDRAWN.**  
   The coarse probe cannot statistically differentiate between quantization recipes; token-level KL divergence ($\text{KL} = 0.094 \pm 0.006$) and relative perplexity drift ($\Delta\text{PPL} = +3.36 \pm 1.83\%$) are non-zero.
3. **"int4_mixed_ssm recovers Mamba" as a compression recipe — REFRAMED AS CONTROL.**  
   Pure Mamba contains no non-recurrent attention or MLP blocks. Skipping `mamba` modules leaves static weights virtually untouched ($13,880.7\text{ MB}$ vs. $13,896.7\text{ MB}$). It is reported solely as a positive control confirming that the skip-list protects the SSM branch ($\text{KL} = 0.005$).
4. **"Falcon-H1 overtakes Falcon3-7B at 8K under vLLM" — WITHDRAWN.**  
   Under vLLM at 8K context, `Falcon-H1-7B` decode throughput trails `Falcon3-7B` by $\sim 2.4\%$ ($16.15 \pm 0.05$ vs. $16.55 \pm 0.02\text{ tok/s}$; $4.450$ vs. $4.348\text{ J/tok}$). `Falcon3-7B` leads `Falcon-H1-7B` at every measured context length in vLLM. However, `Falcon3-Mamba-7B` *does* lead `Falcon3-7B` at 8K ($17.27 \pm 0.01\text{ tok/s}$, $4.152\text{ J/tok}$).
5. **"Exact VRAM crossover at $N=2,048$" — SOFTENED TO NEAR-PARITY AROUND 2K.**  
   At 2048 context, `Falcon3-7B` peak VRAM is $14,900.5\text{ MB}$ vs. $14,902.9\text{ MB}$ for `Falcon-H1-7B` (a $2.4\text{ MB}$ difference at a single sweep point; nothing was measured between 512 and 4096).
6. **"47.8% faster TTFT for int4_mixed_ssm" — WITHDRAWN.**  
   The 512-token TTFT difference had large variance ($\text{std} = 212\text{ ms}$) and was measurement noise. At 2K and 8K context, prefill TTFT is virtually identical ($735.1\text{ ms}$ vs. $736.4\text{ ms}$ at 2K).
7. **"Fits on an 8 GB consumer GPU" — WITHDRAWN.**  
   Peak VRAM for `Falcon-H1-7B` `int4_mixed_ssm` reaches $8,080.5\text{ MB}$ at 8K context and $7,001\text{ MB}$ at 512 context, exceeding 8 GB on standard PyTorch allocators. It was not tested on physical 8 GB consumer cards.
8. **"300 GB/s HBM" — CORRECTED TO GDDR6.**  
   The NVIDIA L4 uses GDDR6 memory, not HBM. Additionally, the v1 "94.4% MBU" for `Falcon3-7B` under vLLM was inflated by prefix-cache reuse of the TTFT prompt; the corrected analytical MBU is $87.5\%$.
9. **"2.36x–2.54x vLLM speedup" — CORRECTED TO 1.8x–1.9x.**  
   With verified fused CUDA kernels active in HF eager, the baseline eager decode is $9.0\text{--}9.5\text{ tok/s}$ (rather than the unoptimized $7.0\text{ tok/s}$ in v1), resulting in a $1.8\text{--}1.9\times$ decode speedup under vLLM CUDA graphs.

### 0.2 What Replicated & New Supported Findings
- **HF Eager Reality:** In Hugging Face eager execution, `Falcon3-7B` is faster and more energy-efficient than both SSM-based models across every measured context length up to 16K ($16.43 \rightarrow 11.90\text{ tok/s}$ vs. $\sim 9.0\text{--}9.5\text{ tok/s}$). The eager deficit is host/kernel launch overhead across 44–64 layers, not inherent architecture efficiency.
- **CUDA Graph Acceleration in vLLM:** Capturing CUDA graphs in vLLM unlocks a **$1.8\text{--}1.9\times$ decode speedup** for Mamba ($9.38 \rightarrow 17.26\text{ tok/s}$ at 512; $9.51 \rightarrow 17.27\text{ tok/s}$ at 8K) and H1 ($9.02 \rightarrow 16.59\text{ tok/s}$ at 512; $8.96 \rightarrow 16.15\text{ tok/s}$ at 8K), compared to $1.07\text{--}1.17\times$ for Falcon3-7B ($16.43 \rightarrow 17.54\text{ tok/s}$ at 512; $14.19 \rightarrow 16.55\text{ tok/s}$ at 8K). This cuts total energy per token by $\sim 41\%$ (Mamba) and $\sim 39\text{--}42\%$ (H1).
- **vLLM 8K Leader:** At 8K context under vLLM, `Falcon3-Mamba-7B` is the fastest and most energy-efficient model ($17.27 \pm 0.01\text{ tok/s}$, $4.152\text{ J/tok}$), leading `Falcon3-7B` ($16.55 \pm 0.02\text{ tok/s}$, $4.348\text{ J/tok}$) by $4.4\%$ in speed and $4.5\%$ in energy.
- **Persistent State Scaling:** Analytical persistent state at 16K context is $1,792\text{ MB}$ for `Falcon3-7B`, $771\text{ MB}$ for `Falcon-H1-7B` ($2.3\times$ smaller), and strictly constant at $20\text{ MB}$ for `Falcon3-Mamba-7B`.
- **Peak VRAM Near-Parity:** Near-parity between `Falcon3-7B` and `Falcon-H1-7B` occurs around 2K context ($14,900.5\text{ MB}$ vs. $14,902.9\text{ MB}$). At 16K context, H1 peak VRAM is $1,671\text{ MB}$ lower ($17,038.4\text{ MB}$ vs. $18,709.2\text{ MB}$). Pure Mamba peak VRAM grows to $20,412.4\text{ MB}$ at 16K due to transient prefill workspace in PyTorch eager mode.
- **Quantization Trade-offs:** On `Falcon-H1-7B`, `int4_mixed_ssm` compresses weights from $14.48\text{ GB} \rightarrow 6.74\text{ GB}$ ($-53.4\%$). Compared to `int4_uniform`, it yields modestly lower KL divergence ($0.094 \pm 0.006$ vs. $0.111 \pm 0.009$) and lower $\Delta\text{PPL}$ ($+3.36\%$ vs. $+4.78\%$), but is slower and less energy-efficient at 8K context ($6.64\text{ tok/s}$, $8.162\text{ J/tok}$ vs. $6.66\text{ tok/s}$, $7.301\text{ J/tok}$). In `bitsandbytes`, NF4 slows SSM decode while accelerating `Falcon3-7B` ($14.19 \rightarrow 18.58\text{ tok/s}$).

---

## 1. Experimental Environment & Telemetry Calibration

### 1.1 Datacenter Telemetry Setup
- **Headless Datacenter Baseline:** 100% of benchmark execution runs on isolated Linux containers on **Modal Serverless (NVIDIA L4 24GB GDDR6, 300 GB/s bandwidth, 72W TDP)**.
- **Hardware Telemetry:** Direct NVML hardware cumulative energy counters (`nvmlDeviceGetTotalEnergyConsumption`) synchronized with 50 Hz (20ms interval) power polling (`vllm_energy_mfu_profiler.NVMLPowerMonitor`).
- **Driver & Frameworks:** NVIDIA driver 610.57.04, CUDA 12.4, PyTorch 2.5.1+cu124, transformers 4.57.6, mamba_ssm 2.3.1, causal_conv1d 1.7.0, flash_attn 2.8.3.post2, bitsandbytes 0.50.2, vLLM v1 (vLLM 0.32.0 in image). Fused Mamba kernels confirmed active at runtime.
- **Model Revisions:**
  - `tiiuae/Falcon3-7B-Instruct`: `1e57a0ecd176c7c139f289c60a74e57f887c3dfb`
  - `tiiuae/Falcon3-Mamba-7B-Instruct`: `79268d5c8e650ec0ec24aad2729bfc906f569580`
  - `tiiuae/Falcon-H1-7B-Instruct`: `41e72f27effbab80cd45b6e884688452253a3686`

### 1.2 Idle Power Reconciliation: Total vs. Dynamic Energy per Token
In early testing, an idle power baseline of 18.28–19.51 W was observed on initial containers. However, across our comprehensive 5-trial suite, container pre-load idle power varied from **13.5 W to 35.4 W** (and post-load idle with weights resident reached 36.1 W to 43.6 W), depending on container start state and power management transitions:
- `Falcon3-7B`: pre-load idle 13.5–16.4 W
- `Falcon3-Mamba-7B`: pre-load idle 33.4–35.4 W
- `Falcon-H1-7B`: pre-load idle 29.9–35.3 W

Because the pre-load idle floor is container-dependent, **Total Energy per Output Token (`joules_per_token_total`) is designated as the primary metric**:
$$E_{\text{total}} = \frac{E_{\text{decode}} - E_{\text{prefill}}}{N_{\text{new\_tokens}}}$$
Dynamic marginal energy ($E_{\text{dynamic}}$), which subtracts container idle floor, is tracked as a secondary internal reference only and is not used for cross-model comparisons.

---

## 2. Model Architecture & Quantization Configuration Findings

### 2.1 Branch-Aware Mixed-Precision Quantization (`int4_mixed_ssm`)
- **Concept:** In parallel hybrid architectures (`Falcon-H1`), Mamba-2 selective state-space layers update recurrent hidden states step-by-step ($h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$). Attention and SwiGLU MLP blocks are non-recurrent.
- **Skip Recipe:**
  ```python
  BitsAndBytesConfig(
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
  ```
- **Trade-off on Falcon-H1-7B:** Static weights drop from 14,477.3 MB to 6,739.7 MB (-53.4%). While it yields modestly lower token-level KL divergence than uniform INT4 ($0.094 \pm 0.006$ vs. $0.111 \pm 0.009$), it requires 1.87 GB more memory than uniform INT4 (4,874.2 MB) and does not improve decode throughput or energy on L4 GPUs under eager bitsandbytes.

---

## 3. Historical Telemetry Logs (v1, Superseded)

> [!WARNING]
> **v1, superseded — see [`results/RESULTS.md`](results/RESULTS.md) for authoritative v2 numbers (5-trial re-run).**
> The tables below are preserved for historical record from the initial 2-trial exploration.

### 3.1 Historical 7B BF16 Crossover Sweep (v1, Superseded)

> [!NOTE]
> Superseded by Section 1 of `results/RESULTS.md`. In v2 (5 trials), Falcon3-7B decode is 16.43 $\rightarrow$ 11.90 tok/s, Falcon3-Mamba-7B is 9.38 $\rightarrow$ 9.41 tok/s, and Falcon-H1-7B is 9.02 $\rightarrow$ 9.03 tok/s.

| Architecture (`model_id`) | Static Weights (MB) | Context | TTFT (ms) | Decode (`tok/s`) | Peak VRAM (MB) | KV / State Growth (MB) | Energy (`J/tok`) | Eval PPL | Zero-Shot Task Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Transformer (7B) (`Falcon3-7B`) | 14220.4 | 512 | 144.5 | 16.2 | 14456.9 | +236.5 | 4.4199 | 17.53 | 80% |
| Pure Transformer (7B) (`Falcon3-7B`) | 14220.4 | 2048 | 531.9 | 15.7 | 14864.9 | +644.6 | 4.5352 | 17.53 | 80% |
| Pure Transformer (7B) (`Falcon3-7B`) | 14220.4 | 4096 | 1172.7 | 15.0 | 15409.1 | +1188.7 | 4.7639 | 17.53 | 80% |
| Pure Transformer (7B) (`Falcon3-7B`) | 14220.4 | 8192 | 2466.9 | 14.1 | 16497.3 | +2276.9 | 5.0723 | 17.53 | 80% |
| Pure Transformer (7B) (`Falcon3-7B`) | 14220.4 | 16384 | 5373.3 | 11.8 | 18673.7 | +4453.4 | 6.0419 | 17.53 | 80% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 13896.7 | 512 | 252.9 | 7.0 | 14218.8 | +322.1 | 7.2381 | 8.94 | 80% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 13896.7 | 2048 | 800.0 | 6.8 | 14814.3 | +917.7 | 7.5640 | 8.94 | 80% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 13896.7 | 4096 | 1683.0 | 6.9 | 15608.4 | +1711.8 | 7.6713 | 8.94 | 80% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 13896.7 | 8192 | 3367.4 | 6.9 | 17196.6 | +3299.9 | 7.9367 | 8.94 | 80% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 13896.7 | 16384 | 6698.7 | 6.9 | 20372.9 | +6476.2 | 8.2630 | 8.94 | 80% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 14477.3 | 512 | 407.3 | 7.0 | 14640.3 | +163.1 | 7.4812 | 33.23 | 100% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 14477.3 | 2048 | 647.3 | 7.0 | 14869.2 | +391.9 | 7.7721 | 33.23 | 100% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 14477.3 | 4096 | 1597.7 | 7.1 | 15174.2 | +697.0 | 7.8917 | 33.23 | 100% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 14477.3 | 8192 | 3123.7 | 7.0 | 15784.4 | +1307.2 | 8.3321 | 33.23 | 100% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 14477.3 | 16384 | 6609.2 | 6.9 | 17004.8 | +2527.6 | 8.9602 | 33.23 | 100% |

### 3.2 Historical 7B Quantization Ablation (v1, Superseded)

> [!NOTE]
> Superseded by Section 2 of `results/RESULTS.md`. In v2 (5 trials, 200-item probe), no catastrophic accuracy collapse occurs; probe accuracy shifts are within 1–1.5 SE across all models.

| Architecture (`model_id`) | Precision | Static Weights (MB) | Eval PPL ($\Delta\%$) | Logit KL vs BF16 | Top-1 Agree | Task Acc (v1 5-item) | Decode @ 8K | Energy @ 8K |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Transformer (7B) (`Falcon3-7B`) | bf16 | 14220.4 | 17.53 (+0.0%) | 0.0000 | 100.0% | 80% | 14.1 | 5.0723 |
| Pure Transformer (7B) (`Falcon3-7B`) | int4_uniform | 4808.4 | 17.63 (+0.57%) | 0.1729 | 82.4% | 80% | 17.9 | 3.9659 |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | bf16 | 13896.7 | 8.94 (+0.0%) | 0.0000 | 100.0% | 80% | 6.9 | 7.9367 |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | int4_uniform | 4359.0 | 9.14 (+2.25%) | 0.0764 | 88.7% | 40% | 6.9 | 7.3498 |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | int4_mixed_ssm | 13880.7 | 8.92 (-0.17%) | 0.0056 | 98.0% | 80% | 7.0 | 8.6823 |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | bf16 | 14477.3 | 33.23 (+0.0%) | 0.0000 | 100.0% | 100% | 7.0 | 8.3321 |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | int4_uniform | 4874.2 | 33.94 (+2.13%) | 0.0981 | 86.1% | 100% | 6.0 | 8.5504 |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | int4_mixed_ssm | 6739.7 | 33.71 (+1.42%) | 0.0829 | 87.3% | 100% | 6.5 | 8.3616 |

### 3.3 Historical vLLM vs HF Eager Study (v1, Superseded)

> [!NOTE]
> Superseded by Section 3 of `results/RESULTS.md`. In v2, vLLM speedup is 1.8–1.9× for Mamba/H1; Falcon3-Mamba-7B leads at 8K while Falcon-H1-7B trails Falcon3-7B by ~2.4%. Analytical MBU is 87.5% for Falcon3, 84.2% for Mamba, and 84.5% for H1.

| Architecture (`model_id`) | Context | Engine | TTFT (ms) | Decode (`tok/s`) | Total Energy (`J/tok`) | Dynamic Energy (`J/tok`) | Achieved BW (`GB/s`) | MBU (`%`) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Transformer (7B) (`Falcon3-7B`) | 512 | hf_eager | 144.5 | 16.20 | 4.4199 | 3.4091 | 245.6 | 81.9% |
| Pure Transformer (7B) (`Falcon3-7B`) | 512 | vllm_cudagraph | 153.7 | 17.73 | 3.9989 | 2.9538 | 265.5 | 88.5% |
| Pure Transformer (7B) (`Falcon3-7B`) | 2048 | hf_eager | 531.9 | 15.68 | 4.5352 | 3.5121 | 244.4 | 81.5% |
| Pure Transformer (7B) (`Falcon3-7B`) | 2048 | vllm_cudagraph | 406.4 | 18.16 | 3.9460 | 2.9258 | 275.2 | 91.7% |
| Pure Transformer (7B) (`Falcon3-7B`) | 8192 | hf_eager | 2466.9 | 14.08 | 5.0723 | 3.9233 | 243.6 | 81.2% |
| Pure Transformer (7B) (`Falcon3-7B`) | 8192 | vllm_cudagraph | 1774.0 | 16.37 | 4.3848 | 3.2530 | 283.2 | 94.4% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 512 | hf_eager | 252.9 | 6.97 | 7.2381 | 2.8185 | 103.9 | 34.6% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 512 | vllm_cudagraph | 242.8 | 17.29 | 4.1135 | 2.1241 | 252.5 | 84.2% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 2048 | hf_eager | 800.0 | 6.84 | 7.5640 | 3.0598 | 106.2 | 35.4% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 2048 | vllm_cudagraph | 837.8 | 17.26 | 4.1805 | 2.1878 | 252.1 | 84.0% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 8192 | hf_eager | 3367.4 | 6.93 | 7.9367 | 3.4942 | 125.0 | 41.6% |
| Pure SSM / Mamba (7B) (`Falcon3-Mamba-7B`) | 8192 | vllm_cudagraph | 3391.7 | 17.30 | 4.1719 | 2.1834 | 252.6 | 84.2% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 512 | hf_eager | 407.3 | 7.02 | 7.4812 | 3.1718 | 107.8 | 35.9% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 512 | vllm_cudagraph | 170.2 | 16.59 | 4.2875 | 2.2860 | 254.6 | 84.8% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 2048 | hf_eager | 647.3 | 6.96 | 7.7721 | 3.4478 | 108.5 | 36.2% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 2048 | vllm_cudagraph | 730.6 | 17.68 | 4.0790 | 2.2010 | 272.5 | 90.8% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 8192 | hf_eager | 3123.7 | 6.96 | 8.3321 | 4.0081 | 115.2 | 38.4% |
| Parallel Hybrid (7B) (`Falcon-H1-7B`) | 8192 | vllm_cudagraph | 2141.8 | 16.51 | 4.3670 | 2.3564 | 273.3 | 91.1% |

---

## 4. Infrastructure, CUDA Kernels & Pitfalls Solved (Reproducibility Notes)

### 4.1 Reference PyTorch Fallback vs. Fused Mamba-2 CUDA Kernels (`mamba_ssm` & `causal_conv1d`)
- **Finding:** Without compiled `mamba_ssm` and `causal_conv1d` C++/CUDA extensions, Hugging Face `transformers` falls back to reference PyTorch loops (`causal_conv1d_fn`, `mamba_chunk_scan_combined`, `selective_state_update`), inflating dynamic scan memory by **3×** (`+353.0 MB` vs. `+115.8 MB` on `Falcon-H1-1.5B` at `ctx=512`).
- **ABI & Version Pitfalls Discovered During Container Build:**
  1. When building `causal-conv1d>=1.5.0` and `mamba-ssm>=2.2.4` via `pip install --no-build-isolation` on `nvidia/cuda:12.4.1-devel-ubuntu22.04`, `mamba-ssm`'s unpinned `torch` dependency caused `pip` to upgrade `torch` from `2.5.1` to `2.14.1` *after* compiling the `.so` wheels against `torch 2.5.1`.
  2. Furthermore, `transformers>=5.0.0` enforces `PyTorch >= 2.6` at import time.
- **Exact Working Container Recipe:** Pin `"torch==2.5.1"` across all `pip_install` layers and pin `"transformers>=4.51.0,<5.0.0"` (`transformers 4.57.6`) + `"huggingface_hub>=0.27.0,<1.0.0"`. This activates `The fast path for FalconH1 will be used when running the model on a GPU` with zero ABI or import errors.

### 4.2 Hugging Face Xet Background Logger vs. Modal Volume Reload
- **Finding:** `huggingface_hub` v1.3+ uses the Rust Xet transfer engine (`hf-xet`), which keeps an open background log file handle inside `$HF_HOME/xet/logs/xet_*.log`. When a warm Modal GPU container calls `volume.reload()`, Modal throws `ConflictError: there are open files preventing the operation`.
- **Fix:** Set `"HF_XET_LOG_DIR": "/tmp/hf_xet_logs"` in the container environment so Xet writes logs to container-local `/tmp` rather than the mounted network volume `/cache/hf`, and wrap `volume.reload()` in a `try/except RuntimeError` guard for reused containers.

### 4.3 BPE Tokenizer Space Cleanup Warning in `transformers>=5.x`
- **Finding:** `Falcon3` uses a `TokenizersBackend` BPE tokenizer where `clean_up_tokenization_spaces=True` emits a warning about stripping spaces before punctuation.
- **Fix:** Pass `clean_up_tokenization_spaces=False` during `tokenizer.decode()` in deterministic task evaluations.

### 4.4 Local ISP DNS Fallback for Modal gRPC (`src/run_modal_cli.py`)
- **Finding:** Intermittent local router/ISP DNS timeouts (`[Errno 11001] getaddrinfo failed` for `api.modal.com`) can interrupt long-running `modal run` sessions from Windows.
- **Fix:** Created [`src/run_modal_cli.py`](src/run_modal_cli.py) which patches `socket.getaddrinfo` to automatically resolve `*.modal.com` via Cloudflare DNS (`1.1.1.1`) whenever the local Windows DNS resolver fails.

### 4.5 vLLM v1 Mamba State Block Pre-Allocation & FlashInfer CUDA 12.4 JIT Fixes (`src/modal_vllm_benchmark.py`)
- **Finding 1 (`max_num_seqs` OOM on 24 GB GPUs for Mamba/Hybrid models):** In `vLLM` v1, `max_num_seqs` defaults to `256`. During `profile_cudagraph_memory`, `_init_minimal_kv_cache_for_profiling` pre-allocates `max_num_seqs` concurrent Mamba recurrent state blocks (`256` blocks $\approx$ `16.84 GiB` on `Falcon-H1-7B`), causing an immediate OOM on 24 GB L4 GPUs alongside the `14.68 GiB` model weights (or raising `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (244)` on `Falcon3-Mamba-7B`).
- **Fix 1:** Explicitly set `max_num_seqs=8` (or `16` for small-batch serving) and `gpu_memory_utilization=0.92` when initializing `LLM(...)` on 24 GB GPUs. This reduces Mamba state pre-allocation from `16.84 GiB` to `<0.6 GiB`.
- **Finding 2 (`flashinfer` JIT `--compress-mode=size` on CUDA 12.4):** `flashinfer>=0.7.0` passes `--compress-mode=size` (a CUDA 12.8+ `nvcc` flag) when JIT-compiling its top-k/top-p sampling kernel, failing with `nvcc fatal: Unknown option '--compress-mode=size'` on CUDA 12.4 images.
- **Fix 2:** Strip `--compress-mode=size` from `/usr/local/lib/python3.11/site-packages/flashinfer/jit/*.py` during image build and set `"VLLM_USE_FLASHINFER_SAMPLER": "0"`.

---

## 5. Roadmap Status
- [x] **Milestone 1:** Verify isolated Modal L4 NVML telemetry, persistent volume caching, and evaluation-quality delta suite (`PPL`, `KL_vs_BF16`, `Top1_Agree`, `TaskAcc`) on 1B/1.5B smoke models.
- [x] **Milestone 2:** Lock `torch==2.5.1` ABI + `transformers<5.0.0` during `mamba-ssm` / `causal-conv1d` wheel installation and record the exact **Fused CUDA Kernel vs. PyTorch Reference Speedup & VRAM Reduction** on `Falcon-H1`.
- [x] **Milestone 3:** Execute full 7B 3-Architecture Crossover Sweep (`Falcon3-7B-Instruct` vs. `Falcon3-Mamba-7B-Instruct` vs. `Falcon-H1-7B-Instruct`) across context lengths `512` to `16,384` ($n=5$ trials, `results/benchmark_crossover.json`).
- [x] **Milestone 4:** Execute 7B Quantization Ablation (`bf16` vs. `int4_uniform` vs. `int4_mixed_ssm`) via `--mode quant-ablation` ($n=5$ trials, `results/benchmark_quant-ablation.json`).
- [x] **Milestone 5:** Package and release the **Energy & MFU/MBU Instrumentation for vLLM** ([`src/vllm_energy_mfu_profiler.py`](src/vllm_energy_mfu_profiler.py), [`src/modal_vllm_benchmark.py`](src/modal_vllm_benchmark.py), [`src/recompute_hf_roofline.py`](src/recompute_hf_roofline.py), [`results/benchmark_vllm_mfu.json`](results/benchmark_vllm_mfu.json), and [`results/plots/benchmark_vllm_mfu_charts.png`](results/plots/benchmark_vllm_mfu_charts.png)).
- [x] **Milestone 6:** v2 5-trial re-run after external review; consolidate single source of truth in `results/RESULTS.md`.
