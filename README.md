# Falcon 7B Architecture Crossover, Branch-Aware Quantization & vLLM Energy/MBU Telemetry

[![Hardware: NVIDIA L4 24GB](https://img.shields.io/badge/GPU-NVIDIA%20L4%2024GB-76B900?logo=nvidia)](https://modal.com)
[![Models: TII Falcon 7B Triad](https://img.shields.io/badge/Models-Falcon3%20%7C%20Mamba%20%7C%20Falcon--H1-blue)](https://huggingface.co/tiiuae)
[![Telemetry: IEEE P1959/P1960 Aligned](https://img.shields.io/badge/Telemetry-Joules%2FToken%20%26%20Roofline%20MBU-orange)](#4-open-source-vllm-energy--roofline-mfumbu-profiler)
[![Author: Harith Ibrahim](https://img.shields.io/badge/Author-Harith%20Ibrahim-blue.svg)](https://orcid.org/0009-0008-2204-7324)
[![ORCID: 0009-0008-2204-7324](https://img.shields.io/badge/ORCID-0009--0008--2204--7324-green.svg)](https://orcid.org/0009-0008-2204-7324)
[![Google Scholar](https://img.shields.io/badge/Scholar-Harith%20Ibrahim-4285F4.svg?logo=google-scholar&logoColor=white)](https://scholar.google.co.uk/citations?user=D0ZI6BoAAAAJ)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**Author:** **Harith Ibrahim** (Machine Learning Systems Researcher, Dubai, UAE; Member & Contributor to IEEE P1959/P1960 Standards Working Groups)  
**Academic Profiles:** [ORCID: `0009-0008-2204-7324`](https://orcid.org/0009-0008-2204-7324) • [Google Scholar Profile](https://scholar.google.co.uk/citations?user=D0ZI6BoAAAAJ) • [GitHub Profile (@HarithSami)](https://github.com/HarithSami)

An empirical hardware-efficiency, post-training quantization, and serving-engine study across the **Technology Innovation Institute's (TII) 7B architectural triad**:
1. **Pure Transformer (GQA):** [`tiiuae/Falcon3-7B-Instruct`](https://huggingface.co/tiiuae/Falcon3-7B-Instruct) (`28` layers)
2. **Pure State-Space Model (SSM / Mamba-1):** [`tiiuae/Falcon3-Mamba-7B-Instruct`](https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Instruct) (`64` mixer layers)
3. **Parallel Hybrid (Mamba-2 + Attention):** [`tiiuae/Falcon-H1-7B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct) (`44` parallel hybrid layers)

All benchmarks were executed on **isolated NVIDIA L4 (24 GB GDDR6, 300 GB/s HBM, 72W TDP)** datacenter GPUs using 50 Hz NVML hardware energy integration (`pynvml`) aligned with **IEEE P1959/P1960** energy-efficiency standardization practices.

---

## Executive Summary: Four Headline Findings

1. **Exact 7B Memory Crossover at `ctx = 2,048` (`1.67 GB` VRAM Saved at `16K`):**
   Although `Falcon-H1-7B` has a slightly larger static `bf16` weight footprint (`14.48 GB` vs. `14.22 GB` for `Falcon3-7B`), its parallel Mamba-2 + reduced KV cache grows **43.2% slower** (`+2,527.6 MB` vs. `+4,453.4 MB` at `16K`). Total Peak VRAM reaches exact parity at **`ctx = 2,048`** (`14,869 MB` vs. `14,865 MB`) and saves **`1.67 GB` per concurrent stream** at `ctx = 16,384`.
2. **Catastrophic Reasoning Collapse in Pure Mamba Under Uniform INT4 (`80%` $\rightarrow$ `40%`):**
   Uniform 4-bit NormalFloat quantization (`int4_uniform`) on `Falcon3-Mamba-7B` causes **4× worse perplexity drift** (`+2.25%` vs. `+0.57%` on `Falcon3-7B`) and collapses zero-shot reasoning/Arabic accuracy from **80% down to 40%** because nearly 100% of pure Mamba's parameters reside inside recurrent state projections ($h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$).
3. **Branch-Aware Mixed Precision (`int4_mixed_ssm`) Unlocks `Falcon-H1-7B` on 8 GB GPUs with Zero Accuracy Loss:**
   Because `Falcon-H1-7B` decouples recurrent Mamba-2 heads from Attention and SwiGLU MLP blocks, keeping `mamba` projections in `bf16` while quantizing Attention + MLP to `int4` compresses static weights by **-53.5%** (`14.48 GB` $\rightarrow$ **`6.74 GB`**), reduces PPL degradation by **33.3%** (`+1.42%` vs. `+2.13%`), lowers logit $\text{KL}(P_{\text{bf16}} \parallel Q)$ by **15.5%** (`0.0829` vs. `0.0981`), cuts prefill TTFT by **47.8%** (`260.1 ms` vs. `497.9 ms`), and retains **100%** zero-shot task accuracy.
4. **`vLLM` CUDA Graphs Unlock a `2.36x–2.54x` Speedup, `91.1%` Roofline MBU, and `-47.6%` Energy/Token on Hybrid & SSM Models:**
   In eager Hugging Face (`hf_eager`), sequential Python kernel launches across 44–64 layers starve the L4 memory bus (`34.6%–38.4%` Model Bandwidth Utilization, `~7.0 tok/s`). Capturing **Piecewise + Full CUDA Graphs in `vLLM` v1** boosts `Falcon-H1-7B` MBU to **`84.8%–91.1%` (`254.6–273.3 GB/s`)** and `Falcon3-Mamba-7B` MBU to **`84.2%` (`252.6 GB/s`)**, slashing decode energy by **-47.6%** (`8.33` $\rightarrow$ **`4.37 J/tok`**) and **overtaking `Falcon3-7B` in both decode speed and energy efficiency at `ctx = 8,192`**.

---

## 1. 7B Architecture Crossover (`bf16`, Context `512` to `16,384`)

![7B Architecture Crossover Charts](results/plots/benchmark_crossover_charts.png)

| Architecture (`model_id`) | Context | TTFT (ms) | Decode (`tok/s`) | Peak VRAM (MB) | Dynamic State Growth (MB) | Total Energy (`J/tok`) | Dynamic Energy (`J/tok`) | Zero-Shot Task Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `512` | `144.5` | `16.2` | `14456.9` | `+236.5` | `4.4199` | `3.4091` | `80%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `2048` | `531.9` | `15.7` | `14864.9` | `+644.6` | `4.5352` | `3.5121` | `80%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `4096` | `1172.7` | `15.0` | `15409.1` | `+1188.7` | `4.7639` | `3.6973` | `80%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `8192` | `2466.9` | `14.1` | `16497.3` | `+2276.9` | `5.0723` | `3.9233` | `80%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `16384` | `5373.3` | `11.8` (`-27.2%`) | `18673.7` | `+4453.4` | `6.0419` (`+36.7%`) | `4.6465` | `80%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `512` | `252.9` | `7.0` | `14218.8` | `+322.1` | `7.2381` | `2.8185` | `80%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `2048` | `800.0` | `6.8` | `14814.3` | `+917.7` | `7.5640` | `3.0598` | `80%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `4096` | `1683.0` | `6.9` | `15608.4` | `+1711.8` | `7.6713` | `3.2061` | `80%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `8192` | `3367.4` | `6.9` | `17196.6` | `+3299.9` | `7.9367` | `3.4942` | `80%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `16384` | `6698.7` | `6.9` (`-0.4%`) | `20372.9` | `+6476.2` | `8.2630` | `3.8222` | `80%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `512` | `407.3` | `7.0` | `14640.3` | **`+163.1`** | `7.4812` | `3.1718` | **`100%`** |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `2048` | `647.3` | `7.0` | **`14869.2`** | **`+391.9`** | `7.7721` | `3.4478` | **`100%`** |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `4096` | `1597.7` | `7.1` | **`15174.2`** | **`+697.0`** | `7.8917` | `3.6288` | **`100%`** |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `8192` | `3123.7` | `7.0` | **`15784.4`** | **`+1307.2`** | `8.3321` | `4.0081` | **`100%`** |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `16384` | `6609.2` | `6.9` (`-1.2%`) | **`17004.8` (`-1.67 GB`)** | **`+2527.6` (`-43.2%`)** | `8.9602` | `4.6197` | **`100%`** |

---

## 2. Branch-Aware Quantization Ablation (`bf16` vs. `int4_uniform` vs. `int4_mixed_ssm`)

![7B Quantization Ablation Charts](results/plots/benchmark_quant-ablation_charts.png)

| Architecture (`model_id`) | Precision | Static Weights (MB) | Eval PPL ($\Delta\%$ vs. BF16) | Logit $\text{KL}(P_{\text{bf16}} \parallel Q)$ ($\downarrow$) | Top-1 Token Agree ($\uparrow$) | Zero-Shot Task Acc ($\uparrow$) | TTFT @ `512` (ms) | Decode @ `8192` (`tok/s`) | Energy @ `8192` (`J/tok`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `bf16` | `14220.4` | `17.53 (+0.00%)` | `0.0000` | `100.0%` | `80%` | `144.5` | `14.1` | `5.0723` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `int4_uniform` | `4808.4` | `17.63 (+0.57%)` | `0.1729` | `82.4%` | **`80%`** | `232.3` | **`17.9`** | **`3.9659`** |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `bf16` | `13896.7` | `8.94 (+0.00%)` | `0.0000` | `100.0%` | `80%` | `252.9` | `6.9` | `7.9367` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `int4_uniform` | `4359.0` | `9.14 (+2.25%)` | `0.0764` | `88.7%` | **`40%` (-50%!)** | `359.7` | `6.9` | `7.3498` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `int4_mixed_ssm` | `13880.7` | `8.92 (-0.17%)` | `0.0056` | `98.0%` | **`80%` (Recovered!)** | `251.6` | `7.0` | `8.6823` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `bf16` | `14477.3` | `33.23 (+0.00%)` | `0.0000` | `100.0%` | `100%` | `407.3` | `7.0` | `8.3321` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `int4_uniform` | `4874.2` | `33.94 (+2.13%)` | `0.0981` | `86.1%` | `100%` | `497.9` | `6.0` | `8.5504` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `int4_mixed_ssm` | **`6739.7` (-53.5%)** | **`33.71 (+1.42%)`** | **`0.0829` (-15.5%)** | **`87.3%`** | **`100%`** | **`260.1` (-47.8%)** | **`6.5`** | **`8.3616`** |

---

## 3. `vLLM` (CUDA Graphs) vs. `hf_eager`: Roofline MFU/MBU & Energy Crossover

![7B vLLM vs HF Roofline MFU/MBU Charts](results/plots/benchmark_vllm_mfu_charts.png)

| Architecture (`model_id`) | Context | Engine | TTFT (ms) | Decode (`tok/s`) | Total Energy (`J/tok`) | Dynamic Energy (`J/tok`) | Achieved HBM BW (`GB/s`) | Roofline MBU (`%`) | Roofline MFU (`%`) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `512` | `hf_eager` | `144.5` | `16.20` | `4.4199` | `3.4091` | `245.6` | `81.9%` | `0.20%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `512` | `vllm_cudagraph` | `153.7` | **`17.73`** | **`3.9989`** | **`2.9538`** | **`265.5`** | **`88.5%`** | `0.22%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `2048` | `hf_eager` | `531.9` | `15.68` | `4.5352` | `3.5121` | `244.4` | `81.5%` | `0.19%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `2048` | `vllm_cudagraph` | `406.4` | **`18.16`** | **`3.9460`** | **`2.9258`** | **`275.2`** | **`91.7%`** | `0.23%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `8192` | `hf_eager` | `2466.9` | `14.08` | `5.0723` | `3.9233` | `243.6` | `81.2%` | `0.17%` |
| **Pure Transformer (7B)** (`Falcon3-7B`) | `8192` | `vllm_cudagraph` | `1774.0` | **`16.37`** | **`4.3848`** | **`3.2530`** | **`283.2`** | **`94.4%`** | `0.28%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `512` | `hf_eager` | `252.9` | `6.97` | `7.2381` | `2.8185` | `103.9` | `34.6%` | `0.08%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `512` | `vllm_cudagraph` | `242.8` | **`17.29` (`2.48x`)** | **`4.1135` (`-43.2%`)** | **`2.1241`** | **`252.5`** | **`84.2%`** | `0.21%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `2048` | `hf_eager` | `800.0` | `6.84` | `7.5640` | `3.0598` | `106.2` | `35.4%` | `0.08%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `2048` | `vllm_cudagraph` | `837.8` | **`17.26` (`2.52x`)** | **`4.1805` (`-44.7%`)** | **`2.1878`** | **`252.1`** | **`84.0%`** | `0.21%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `8192` | `hf_eager` | `3367.4` | `6.93` | `7.9367` | `3.4942` | `125.0` | `41.6%` | `0.08%` |
| **Pure SSM / Mamba (7B)** (`Falcon3-Mamba-7B`) | `8192` | `vllm_cudagraph` | `3391.7` | **`17.30` (`2.50x`)** | **`4.1719` (`-47.4%`)** | **`2.1834`** | **`252.6`** | **`84.2%`** | `0.21%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `512` | `hf_eager` | `407.3` | `7.02` | `7.4812` | `3.1718` | `107.8` | `35.9%` | `0.09%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `512` | `vllm_cudagraph` | **`170.2` (`-58.2%`)** | **`16.59` (`2.36x`)** | **`4.2875` (`-42.7%`)** | **`2.2860`** | **`254.6`** | **`84.8%`** | `0.21%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `2048` | `hf_eager` | `647.3` | `6.96` | `7.7721` | `3.4478` | `108.5` | `36.2%` | `0.09%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `2048` | `vllm_cudagraph` | `730.6` | **`17.68` (`2.54x`)** | **`4.0790` (`-47.5%`)** | **`2.2010`** | **`272.5`** | **`90.8%`** | `0.22%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `8192` | `hf_eager` | `3123.7` | `6.96` | `8.3321` | `4.0081` | `115.2` | `38.4%` | `0.09%` |
| **Parallel Hybrid (7B)** (`Falcon-H1-7B`) | `8192` | `vllm_cudagraph` | **`2141.8` (`-31.4%`)** | **`16.51` (`2.37x`)** | **`4.3670` (`-47.6%`)** | **`2.3564`** | **`273.3`** | **`91.1%`** | `0.28%` |

---

## 4. Open-Source `vLLM` Energy & Roofline MFU/MBU Profiler

[`src/vllm_energy_mfu_profiler.py`](src/vllm_energy_mfu_profiler.py) is a zero-dependency, drop-in NVML telemetry and Roofline efficiency profiler for **Transformer, Mamba/SSM, and Parallel Hybrid (`Falcon-H1`)** models in `vLLM` or PyTorch.

### Quickstart Usage

```python
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from src.vllm_energy_mfu_profiler import VLLMEnergyMFUProfiler, build_architecture_profile

model_id = "tiiuae/Falcon-H1-7B-Instruct"
hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
weights_bytes = int(14477.3 * (1024**2))

profiler = VLLMEnergyMFUProfiler(device_index=0, poll_interval_s=0.02)
profiler.calibrate_idle_floor(duration_s=2.0)

arch_profile = build_architecture_profile(model_id, hf_config, weights_bytes, precision="bf16")

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    dtype="bfloat16",
    max_num_seqs=8,              # Prevents 256-block Mamba state OOM on 24GB GPUs
    gpu_memory_utilization=0.92,
)

with profiler.profile_generation(
    arch=arch_profile,
    weights_vram_bytes=weights_bytes,
    engine="vllm_cudagraph",
    precision="bf16",
    context_length=8192,
    decode_tokens=128,
) as session:
    _ = llm.generate(["Your prompt..."], SamplingParams(temperature=0.0, max_tokens=128))

print(session.report.to_dict())
```

---

## 5. Repository Structure & Reproducing on Modal

```text
├── paper/
│   ├── falcon_7b_hybrid_crossover_paper.md   # Full technical report (Markdown)
│   └── falcon_7b_hybrid_crossover_paper.tex  # LaTeX source for arXiv / workshop submission
├── outreach/
│   └── linkedin_and_hf_launch_kit.md         # LinkedIn deep-dive post & Hugging Face Hub cards
├── results/
│   ├── benchmark_smoke.json                  # 1B / 1.5B smoke & fused-kernel comparison
│   ├── benchmark_crossover.json              # 7B 3-architecture BF16 crossover (512 to 16K)
│   ├── benchmark_quant-ablation.json         # 7B BF16 vs. INT4 uniform vs. INT4 mixed SSM
│   ├── benchmark_vllm_mfu.json               # 7B vLLM CUDA-Graph vs. HF Eager Roofline MFU/MBU
│   └── plots/                                # High-resolution 4-panel publication charts (.png)
├── src/
│   ├── vllm_energy_mfu_profiler.py           # Open-source NVML Energy & Roofline MFU/MBU profiler
│   ├── modal_benchmark.py                    # Modal L4 serverless benchmark (smoke/crossover/quant)
│   ├── modal_vllm_benchmark.py               # Modal L4 vLLM CUDA-Graph Roofline benchmark
│   ├── plot_results.py                       # Markdown summary & 4-panel Matplotlib generator
│   └── run_modal_cli.py                      # Cloudflare DNS-resilient Modal CLI wrapper
└── research_findings_log.md                  # Living research & engineering notebook
```

### Running the Full Suite on Modal Serverless GPUs
```bash
pip install -r requirements.txt
modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN

# 1. 1B/1.5B Smoke Test
python src/run_modal_cli.py run src/modal_benchmark.py --mode smoke

# 2. 7B 3-Architecture BF16 Crossover (512 to 16,384 context)
python src/run_modal_cli.py run src/modal_benchmark.py --mode crossover

# 3. 7B Branch-Aware Quantization Ablation (bf16 vs. int4_uniform vs. int4_mixed_ssm)
python src/run_modal_cli.py run src/modal_benchmark.py --mode quant-ablation

# 4. 7B vLLM CUDA-Graph vs. HF Eager Roofline MFU/MBU Benchmark
python src/run_modal_cli.py run src/modal_vllm_benchmark.py

# 5. Regenerate All Plots
python src/plot_results.py results/benchmark_crossover.json
python src/plot_results.py results/benchmark_quant-ablation.json
python src/plot_results.py results/benchmark_vllm_mfu.json
```

---

## Citation

```bibtex
@techreport{ibrahim2026falcon7bcrossover,
  title       = {Hardware, Energy, and Quantization Crossover in Parallel Hybrid State-Space Models: An Empirical Study on the Falcon 7B Triad},
  author      = {Ibrahim, Harith},
  year        = {2026},
  institution = {Independent Research (IEEE P1959/P1960 Working Group Contributor)},
  note        = {Empirical evaluation of Falcon3-7B, Falcon3-Mamba-7B, and Falcon-H1-7B on NVIDIA L4 GPUs}
}
```
