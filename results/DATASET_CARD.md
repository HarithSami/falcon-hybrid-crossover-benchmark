---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
- ar
license: apache-2.0
pretty_name: "Falcon 7B Triad Architecture Crossover, Quantization Ablation & vLLM Energy Telemetry Dataset"
tags:
- energy-efficiency
- green-ai
- llm-telemetry
- vllm
- mamba
- ssm
- hybrid-architecture
- falcon-h1
- falcon3
- quantization
- roofline
- modal
size_categories:
- 10K<n<100K
---

# Falcon 7B Triad Telemetry Dataset: Architecture Crossover, Branch-Aware Quantization & vLLM Energy

## Dataset Summary

This dataset contains empirical hardware telemetry, latency, energy consumption, and fidelity metrics collected across the **Technology Innovation Institute (TII) 7B architectural triad**:
1. **Pure Transformer (GQA):** [`tiiuae/Falcon3-7B-Instruct`](https://huggingface.co/tiiuae/Falcon3-7B-Instruct) (28 layers)
2. **Pure State-Space Model (SSM / Mamba-1):** [`tiiuae/Falcon3-Mamba-7B-Instruct`](https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Instruct) (64 mixer layers)
3. **Parallel Hybrid (Mamba-2 + Attention):** [`tiiuae/Falcon-H1-7B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct) (44 parallel hybrid layers)

All measurements were captured on isolated **NVIDIA L4 GPUs (24 GB GDDR6, 300 GB/s peak memory bandwidth, 72 W TDP)** via Modal serverless infrastructure with 50 Hz NVML board power polling and hardware cumulative energy counter integration. The measurement protocol was informed by **IEEE P1959/P1960** energy telemetry discussions.

This is the **v2 re-run dataset** (completed 2026-11-01), superseding earlier preliminary 2-trial runs with $n=5$ independent trials per configuration, full error bounds (mean $\pm$ sample std / standard error), and a 200-item templated evaluation probe.

---

## File Inventory & Descriptions

The `results/` directory contains structured JSON benchmarks, raw high-frequency power traces, and publication-ready plots:

| File | Description | Records / Format |
| :--- | :--- | :--- |
| `results/RESULTS.md` | Human-readable single source of truth containing consolidated tables, supported claims, and withdrawn v1 claims. | Markdown report |
| `results/benchmark_crossover.json` | 7B 3-architecture BF16 context sweep (512 to 16,384 tokens) in Hugging Face eager mode ($n=5$ trials). | JSON list (15 objects) |
| `results/benchmark_quant-ablation.json` | 7B quantization ablation comparing `bf16`, `int4_uniform` (NF4 across all linear layers), and `int4_mixed_ssm` (NF4 on Attention/MLP, native BF16 on Mamba SSM layers) at contexts 512, 2048, and 8192 ($n=5$ trials). | JSON list (24 objects) |
| `results/benchmark_vllm_mfu.json` | Engine comparison between Hugging Face eager execution and `vLLM` v1 with CUDA Graphs across contexts 512, 2048, and 8192 ($n=5$ trials), reporting throughput, energy, and analytical GDDR6 MBU/MFU. | JSON object with keys `hf_roofline` and `vllm_cudagraph` |
| `results/benchmark_smoke.json` | Initial verification run on smaller 1B/1.5B models (`Falcon3-1B-Instruct` and `Falcon-H1-1.5B-Deep-Instruct`) evaluating fused CUDA kernels vs reference PyTorch fallbacks. | JSON list (5 objects) |
| `results/raw_50hz_crossover_traces.json` | 50 Hz (20 ms interval) time-series power telemetry logs for all crossover benchmark configurations. | JSON list (15 trace objects) |
| `results/raw_50hz_quant-ablation_traces.json` | 50 Hz time-series power telemetry logs for all quantization ablation configurations. | JSON list (15 trace objects) |
| `results/raw_50hz_vllm_traces.json` | 50 Hz time-series power telemetry logs for all vLLM CUDA graph runs. | JSON list (9 trace objects) |
| `results/plots/` | Publication-ready 2x2 multi-panel plots generated from the benchmark JSONs in both high-res PNG (300 DPI) and vector PDF formats. | PNG and PDF files |

### Plot Files in `results/plots/`
- `benchmark_crossover_charts.png` / `.pdf`: 2x2 grid displaying (a) Decode throughput vs context length in HF eager, (b) Persistent state vs peak prefill $\Delta$VRAM, (c) Total decode energy per token, and (d) Prefill TTFT.
- `benchmark_quant-ablation_charts.png` / `.pdf`: 2x2 grid displaying (a) Within-model $\Delta$PPL% vs static weight footprint, (b) Token-level logit $\text{KL}(P_{\text{bf16}} \parallel Q)$, (c) 200-item zero-shot probe accuracy, and (d) Total decode energy per token at 8K context.
- `benchmark_vllm_mfu_charts.png` / `.pdf`: 2x2 grid displaying (a) Decode throughput (vLLM CUDA graphs vs HF eager), (b) GDDR6 Model Bandwidth Utilization (MBU %), (c) Total decode energy per token, and (d) Prefill TTFT.
- `benchmark_smoke_charts.png`: Initial 1B / 1.5B sanity check summary.

---

## Schema & Field Definitions

### `benchmark_crossover.json` & `benchmark_quant-ablation.json`
Each entry in the list contains the following attributes:

- `timestamp` *(str)*: ISO-8601 execution timestamp.
- `arch_label` *(str)*: Architecture descriptor (`Pure Transformer (7B)`, `Pure SSM / Mamba (7B)`, `Parallel Hybrid (7B)`).
- `model_id` *(str)*: Hugging Face repository identifier.
- `model_revision` *(str)*: Exact git commit hash of the model snapshot.
- `precision` *(str)*: Precision configuration (`bf16`, `int4_uniform`, or `int4_mixed_ssm`).
- `gpu_name` *(str)*: Accelerator model name (`NVIDIA L4`).
- `memory_type` *(str)*: GPU DRAM technology (`GDDR6`).
- `num_trials` *(int)*: Repetitions measured per condition ($n=5$).
- `idle_watts` *(float)*: Pre-load baseline idle board power (W) before loading weights or allocating CUDA context.
- `post_load_idle_watts` *(float)*: Post-load idle board power (W) with resident weights and CUDA runtime initialized.
- `weights_vram_mb` *(float)*: Static model weight memory footprint in MiB.
- `context_length` *(int)*: Prompt sequence length in tokens (512, 2048, 4096, 8192, 16384).
- `decode_tokens` *(int)*: Autoregressive token budget (128 generated; 127 measured decode steps).
- `ttft_ms` *(float)*: Prefill Time-To-First-Token in milliseconds (mean across $n=5$ trials).
- `ttft_std_ms` *(float)*: Sample standard deviation of TTFT across trials.
- `decode_tok_per_s` *(float)*: Steady-state autoregressive decode throughput in tokens per second (mean).
- `decode_tok_per_s_std` *(float)*: Sample standard deviation of decode throughput.
- `tpot_ms` *(float)*: Time-Per-Output-Token in milliseconds ($1000 / \text{decode\_tok\_per\_s}$).
- `tpot_std_ms` *(float)*: Sample standard deviation of TPOT.
- `peak_vram_mb` *(float)*: `torch.cuda.max_memory_allocated()` in MiB during generation (includes static weights, prefill workspace, and persistent state).
- `kv_or_state_growth_mb` *(float)*: Peak VRAM minus static weights in MiB.
- `analytical_persistent_state_mb` *(float)*: Exact theoretical persistent state footprint (KV cache + SSM conv1d/recurrent state) computed from architecture configuration parameters.
- `analytical_total_decode_vram_mb` *(float)*: Static weights plus analytical persistent state in MiB.
- `mean_watts` *(float)*: Mean board power draw during active decode phase (W).
- `mean_watts_std` *(float)*: Sample standard deviation of mean board power.
- `peak_watts` *(float)*: Peak instantaneous board power observed during decode (W).
- `joules_per_token_total` *(float)*: Total energy consumed per decode token ($E_{\text{total}} = (E_{\text{decode}} - E_{\text{prefill}}) / N_{\text{tokens}}$), integrated from NVML cumulative energy counters. **Primary energy metric.**
- `joules_per_token_total_std` *(float)*: Sample standard deviation of total J/tok.
- `joules_per_token_dynamic` *(float)*: Marginal dynamic energy per token after subtracting pre-load idle power. Reported as secondary metric due to container-to-container idle variance.
- `joules_per_token_dynamic_std` *(float)*: Sample standard deviation of dynamic J/tok.
- `eval_tokens_count` *(int)*: Number of scored tokens on the bilingual evaluation corpus.
- `eval_ppl` *(float)*: Exponentiated cross-entropy loss on the evaluation corpus.
- `ppl_delta_pct` *(float)*: Relative percentage perplexity difference relative to the model's own BF16 baseline: $(\text{PPL}_Q - \text{PPL}_{\text{bf16}}) / \text{PPL}_{\text{bf16}} \times 100\%$.
- `ppl_delta_se_pct` *(float)*: Passage-level standard error for $\Delta$PPL% ($n=6$ passages).
- `kl_div_vs_bf16` *(float)*: Mean token-level forward Kullback-Leibler divergence $\text{KL}(P_{\text{bf16}} \parallel Q)$ over next-token logits (nats).
- `kl_div_se` *(float)*: Token-level standard error of the KL divergence.
- `top1_agree_pct` *(float)*: Percentage of greedy argmax tokens identical between quantized and BF16 predictions.
- `eval_tasks_n` *(int)*: Total items in deterministic evaluation probe ($N=200$).
- `task_accuracy_pct` *(float)*: Overall accuracy percentage on the 200-item probe.
- `task_accuracy_se_pct` *(float)*: Binomial standard error of probe accuracy ($\sqrt{p(1-p)/N} \times 100\%$).
- `domain_accuracy_pct` *(dict)*: Accuracy breakdown across 4 domains (`math`, `systems`, `arabic_uae`, `key_value`).
- `environment` *(dict)*: Runtime library versions, driver, and CUDA details.
- `status` *(str)*: Execution status (`ok`).

### `benchmark_vllm_mfu.json`
Contains two top-level keys:
- `hf_roofline`: Analytical GDDR6 Roofline MBU and MFU derived from the HF eager crossover benchmark via `src/recompute_hf_roofline.py`.
- `vllm_cudagraph`: Direct measurements from `vLLM` v1 with CUDA graph capture across contexts 512, 2048, and 8192 ($n=5$ trials).

Key engine-specific fields include:
- `engine` *(str)*: `hf_eager` or `vllm_cudagraph`.
- `achieved_mem_bw_gbps` *(float)*: Analytical memory bandwidth achieved during decode:
  $$\text{BW} = \frac{(\text{Weights} + \text{KV}(\text{seq}) + 2 \times \text{SSM\_state}) \times \text{tok/s}}{10^9}$$
- `mbu_pct` *(float)*: Model Bandwidth Utilization (% of 300 GB/s peak GDDR6 bandwidth).
- `mfu_pct` *(float)*: Model FLOPs Utilization (% of 121 TFLOPS dense BF16 peak).

### `raw_50hz_*_traces.json`
Array of trace entries logged by `vllm_energy_mfu_profiler.NVMLPowerMonitor`:
- `arch_label` *(str)*: Model architecture label.
- `model_id` *(str)*: Model repository id.
- `precision` *(str)*: Precision mode.
- `engine` *(str)*: Serving engine (`hf_eager` or `vllm_cudagraph`).
- `context_length` *(int)*: Sequence length.
- `poll_interval_s` *(float)*: Polling period ($0.02$ seconds, corresponding to 50 Hz).
- `pre_load_idle_watts` *(float)*: Container idle power baseline before model loading.
- `post_load_idle_watts` *(float)*: Idle power with model loaded.
- `samples_rel_time_s` *(list[float])*: Monotonic elapsed time in seconds from decode start.
- `samples_watts` *(list[float])*: Instantaneous board power in Watts at each sample timestamp.

---

## Measurement Protocol

1. **Hardware Isolation:** Experiments ran in independent, headless Linux containers on Modal serverless infrastructure. Each model was evaluated in its own freshly instantiated container on an NVIDIA L4 GPU (24 GB GDDR6, 72 W TDP).
2. **Repetitions & Warmup:** All configurations were evaluated across $n=5$ independent trials. Each trial used a newly generated synthetic prompt. The first decode token was omitted from decode throughput calculations to isolate steady-state generation from prefill.
3. **Power & Energy Logging:** Board power was polled at 50 Hz (20 ms interval) via NVML (`nvmlDeviceGetPowerUsage`). Total energy consumption was integrated directly from hardware cumulative energy counters (`nvmlDeviceGetTotalEnergyConsumption`). Total J/tok is the primary reported energy metric.
4. **Analytical vs Peak Memory:**
   - **Analytical Persistent State:** Theoretical recurrent state (KV cache for Attention heads + conv1d and SSM recurrent state for Mamba layers) calculated exactly from configuration dimensions.
   - **Peak $\Delta$VRAM:** `torch.cuda.max_memory_allocated() - weights_vram_mb`, capturing transient prefill scratchpads and intermediate activations alongside persistent state.
5. **Quality Probe Protocol:**
   - 6-passage bilingual (English technical and Arabic/UAE cultural) corpus scored for perplexity ($\Delta\text{PPL}\%$ relative to own BF16 baseline) and token-level logit $\text{KL}(P_{\text{bf16}} \parallel Q)$.
   - 200-item templated multiple-choice evaluation probe (next-token A/B/C/D scoring) spanning 4 domains $\times$ 50 items: math (50 synthetic linear equations), systems (5 unique questions $\times$ 10 option permutations), Arabic/UAE knowledge (5 unique questions $\times$ 10 option permutations), and key-value recall (50 synthetic pairs).

---

## Scope & Limitations

1. **Templated Evaluation Probe:** The systems and Arabic/UAE domains consist of 5 unique questions rotated across 10 answer permutations. The probe is intended as a functional coarse sanity check to detect catastrophic breakdown rather than a comprehensive general capability benchmark. Binomial standard errors are reported for transparency but overstate item independence.
2. **Analytical MBU Metric:** Model Bandwidth Utilization (MBU %) is computed from an analytical model of weights and persistent state traffic divided by peak hardware bandwidth (300 GB/s). It does not measure raw hardware bus counters and excludes activation and logit memory traffic.
3. **Container Idle Floor Variation:** Pre-load idle power varied across serverless container allocations from 13.5 W to 35.4 W (post-load idle: 36.1 W to 43.6 W). Consequently, dynamic J/tok (which subtracts pre-load idle) is influenced by container initialization state and is reported only as a secondary metric; total J/tok is the primary metric.
4. **Batch Size Scope:** All measurements were conducted at batch size 1 (single-stream low-latency decode). Multi-tenant batched throughput and multi-sequence serving dynamics were not evaluated in the crossover sweep.

---

## Citation

```bibtex
@techreport{ibrahim2026falcon7bcrossover,
  title       = {Hardware, Energy, and Quantization Crossover in Parallel Hybrid State-Space Models: An Empirical Study on the Falcon 7B Triad},
  author      = {Ibrahim, Harith},
  year        = {2026},
  institution = {Independent Research},
  note        = {Empirical evaluation of Falcon3-7B, Falcon3-Mamba-7B, and Falcon-H1-7B on NVIDIA L4 GPUs informed by IEEE P1959/P1960 discussions}
}
```
