# Results v2 — Single Source of Truth (5-trial re-run, 2026-11-01)

All numbers below come from `benchmark_crossover.json`, `benchmark_quant-ablation.json`,
`benchmark_vllm_mfu.json` (re-run after external review). **Any document (paper, README, HF card,
findings log, outreach) must quote only these numbers.** v1 numbers (2-trial, 5-question probe) are
superseded and several v1 headline claims did NOT replicate — see §6.

## 0. Setup / reproducibility
- GPU: NVIDIA L4, 24 GB **GDDR6** (not HBM), 300 GB/s peak memory bandwidth, 121 TFLOPS dense BF16, 72 W TDP. Modal serverless, one model per container.
- NVIDIA driver 610.57.04 · CUDA 12.4 · torch 2.5.1+cu124 · transformers 4.57.6 · mamba_ssm 2.3.1 · causal_conv1d 1.7.0 · flash_attn 2.8.3.post2 · bitsandbytes 0.50.2 · vLLM v1 (≥0.8.5; vLLM 0.32.0 installed in image).
- Fused Mamba kernels confirmed active at runtime for all HF eager runs (`fused_kernels_active: true`; "fast path for FalconH1" log line).
- Model revisions: Falcon3-7B-Instruct `1e57a0ecd176c7c139f289c60a74e57f887c3dfb`; Falcon3-Mamba-7B-Instruct `79268d5c8e650ec0ec24aad2729bfc906f569580`; Falcon-H1-7B-Instruct `41e72f27effbab80cd45b6e884688452253a3686`.
- Protocol: batch size 1, 128 decode tokens (127 measured after the first), greedy, **n = 5 trials**, mean ± std. Random synthetic prompts, distinct per trial. vLLM: CUDA graphs on, `enable_prefix_caching=False`, `max_num_seqs=8`, `gpu_memory_utilization=0.92`.
- Energy: NVML board power sampled at 50 Hz plus NVML cumulative energy counter. **Total J/tok is the primary metric.** Dynamic J/tok subtracts a pre-load idle floor that varied from 13.5 W to 35.4 W across containers (post-load idle 36–44 W), so dynamic J/tok is reported as secondary only. Raw traces: `results/raw_50hz_*_traces.json`.
- Memory: two quantities are reported. (1) **Analytical persistent state** = KV cache + SSM conv/recurrent state from model config. (2) **Peak ΔVRAM** = `torch.cuda.max_memory_allocated` − weights, which includes transient prefill workspace/activations.
- MBU is an **analytical estimate**: bytes/token = weights + KV(avg seq) + 2×SSM state, × decode tok/s ÷ 300 GB/s. It ignores activations, logits and kernel inefficiency, so treat it as a model-based estimate rather than a hardware counter.
- Quality: dense PPL and token-level KL(P_bf16‖Q) on a 6-passage bilingual (EN/AR) corpus — 769 to 1,223 scored tokens depending on tokenizer; SE for ΔPPL is passage-level (n=6), SE for KL is token-level. Plus a **200-item templated multiple-choice probe** (next-token A/B/C/D scoring) over 4 domains × 50: math (50 generated linear equations), systems (5 unique questions × 10 option rotations), Arabic/UAE (5 unique × 10 rotations), key-value recall (50 generated). **Systems and Arabic domains therefore have only 5 unique questions each — the probe is NOT an independent 200-question benchmark.** Binomial SE is reported but overstates independence. Treat accuracy as a coarse functional check.

## 1. HF eager BF16 crossover (n=5)
| Model | Ctx | TTFT ms | Decode tok/s | Persist. state MB (analytical) | Peak VRAM MB | Total J/tok |
|---|---:|---:|---:|---:|---:|---:|
| Falcon3-7B | 512 | 153.5±7.0 | 16.43±0.03 | 56 | 14492.4 | 4.376±0.034 |
| Falcon3-7B | 2048 | 552.0±6.2 | 15.91±0.02 | 224 | 14900.5 | 4.529±0.011 |
| Falcon3-7B | 4096 | 1247.5±7.8 | 15.30±0.01 | 448 | 15444.6 | 4.714±0.015 |
| Falcon3-7B | 8192 | 2674.1±23.1 | 14.19±0.03 | 896 | 16532.8 | 5.074±0.012 |
| Falcon3-7B | 16384 | 5695.5±20.5 | 11.90±0.04 | 1792 | 18709.2 | 6.052±0.027 |
| Falcon3-Mamba-7B | 512 | 225.0±5.2 | 9.38±0.03 | 20 | 14258.4 | 7.005±0.110 |
| Falcon3-Mamba-7B | 2048 | 819.9±8.6 | 9.45±0.04 | 20 | 14853.9 | 6.788±0.084 |
| Falcon3-Mamba-7B | 4096 | 1779.7±3.6 | 9.48±0.02 | 20 | 15648.0 | 7.075±0.045 |
| Falcon3-Mamba-7B | 8192 | 3605.4±56.5 | 9.51±0.06 | 20 | 17236.1 | 7.035±0.063 |
| Falcon3-Mamba-7B | 16384 | 7089.0±179.8 | 9.41±0.08 | 20 | 20412.4 | 7.045±0.185 |
| Falcon-H1-7B | 512 | 258.2±175.4 | 9.02±0.05 | 89 | 14674.1 | 7.044±0.213 |
| Falcon-H1-7B | 2048 | 680.4±12.5 | 9.06±0.08 | 155 | 14902.9 | 7.489±0.115 |
| Falcon-H1-7B | 4096 | 1593.0±101.3 | 9.11±0.05 | 243 | 15208.0 | 7.604±0.053 |
| Falcon-H1-7B | 8192 | 3259.3±46.1 | 8.96±0.03 | 419 | 15818.1 | 7.676±0.124 |
| Falcon-H1-7B | 16384 | 6892.5±73.0 | 9.03±0.07 | 771 | 17038.4 | 7.855±0.138 |

Static weights (MB): Falcon3-7B 14220.4 · Falcon3-Mamba-7B 13896.7 · Falcon-H1-7B 14477.3.

Supported statements:
- Falcon3-7B eager decode drops 27.6% (16.43→11.90 tok/s) and total J/tok rises 38.3% (4.38→6.05) from 512 to 16K. Mamba and H1 eager decode is flat within ~1% across 512–16K.
- **In HF eager, Falcon3-7B is faster and more energy-efficient than both SSM-based models at every measured context up to 16K.** The eager gap is dominated by framework/launch overhead (see §3), so Table 1 characterises HF eager execution, not the architectures.
- Analytical persistent state at 16K: Falcon3-7B 1792 MB, H1 771 MB (2.3× smaller), Mamba 20 MB (constant).
- Peak VRAM: Falcon3 vs H1 are at **near-parity around 2K** (14900.5 vs 14902.9 MB, a 2.4 MB difference at one sweep point; nothing measured between 512 and 4K). At 16K H1 peak is 1,671 MB lower (17038.4 vs 18709.2).
- Mamba peak ΔVRAM grows to ~6.5 GB at 16K while its persistent state is 20 MB → this growth is **transient prefill workspace** in the HF path, not recurrent state.

## 2. Quantization ablation (n=5 latency; quality metrics context-independent)
| Model | Precision | Weights MB | ΔPPL % ±SE | KL ±SE | Top-1 % | Probe acc % ±SE | Decode tok/s @8K | J/tok @8K | Peak VRAM @8K MB | TTFT @2K ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon3-7B | bf16 | 14220.4 | 0 | 0 | 100 | 70.0±3.2 | 14.19±0.03 | 5.074±0.012 | 16532.8 | 552.0 |
| Falcon3-7B | int4_uniform | 4808.4 | −3.31±1.99 | 0.175±0.020 | 83.0 | 64.0±3.4 | 18.58±0.09 | 3.851±0.021 | 7120.8 | 634.1 |
| Falcon3-Mamba-7B | bf16 | 13896.7 | 0 | 0 | 100 | 48.0±3.5 | 9.51±0.06 | 7.035±0.063 | 17236.1 | 819.9 |
| Falcon3-Mamba-7B | int4_uniform | 4359.0 | +3.29±1.15 | 0.074±0.004 | 88.5 | 53.0±3.5 | 7.29±0.02 | 6.635±0.021 | 7698.5 | 916.0 |
| Falcon3-Mamba-7B | int4_mixed_ssm | 13880.7 | −0.43±0.20 | 0.005±0.0002 | 97.3 | 47.5±3.5 | 7.12±0.03 | 8.271±0.035 | 17220.1 | 838.5 |
| Falcon-H1-7B | bf16 | 14477.3 | 0 | 0 | 100 | 85.0±2.5 | 8.96±0.03 | 7.676±0.124 | 15818.1 | 680.4 |
| Falcon-H1-7B | int4_uniform | 4874.2 | +4.78±2.19 | 0.111±0.009 | 84.8 | 87.5±2.3 | 6.66±0.02 | 7.301±0.080 | 6215.0 | 736.4 |
| Falcon-H1-7B | int4_mixed_ssm | 6739.7 | +3.36±1.83 | 0.094±0.006 | 86.0 | 87.0±2.4 | 6.64±0.05 | 8.162±0.086 | 8080.5 | 735.1 |

Supported statements:
- **No accuracy collapse.** Probe accuracy differences between precisions are within ~1–1.5 SE for every model (Mamba 48.0 → 53.0 → 47.5; H1 85.0 → 87.5 → 87.0). The v1 "80% → 40%" Mamba collapse (4/5 → 2/5) does not replicate.
- int4_mixed_ssm on Falcon3-Mamba-7B quantizes almost nothing (13,880.7 vs 13,896.7 MB) — it is essentially BF16 and is reported only as a control confirming that the skip-list protects the SSM path (KL 0.005).
- On Falcon-H1-7B, int4_mixed_ssm keeps Mamba projections in BF16 and NF4-quantizes attention/MLP: weights 14.48 → 6.74 GB (−53.4%). vs int4_uniform it has ~15% lower KL (0.094±0.006 vs 0.111±0.009; difference ≈1.5 combined SE — suggestive, not conclusive), lower ΔPPL (+3.36 vs +4.78, overlapping SE), slightly higher top-1 agreement (86.0 vs 84.8). It costs 1.87 GB more weight memory, and is **slower-or-equal and less energy-efficient** than uniform at 8K (8.16 vs 7.30 J/tok). TTFT is equal at 2K/8K (735 vs 736 ms; the 512 difference has std 212 ms and is noise).
- Peak VRAM for H1 int4_mixed_ssm reaches 8,080 MB at 8K and 7,001 MB at 512 → **"fits on an 8 GB GPU" is not supported** (not tested; peak already exceeds 8 GB at 8K on this allocator).
- bitsandbytes NF4 in HF eager slows the SSM models' decode (H1 9.0 → 6.6 tok/s, Mamba 9.5 → 7.1–7.3) while speeding up Falcon3-7B (14.2 → 18.6 tok/s at 8K, −24% J/tok).
- Falcon3-7B int4_uniform ΔPPL is negative (−3.3±2.0%) — an artifact of the small 6-passage corpus; KL (0.175) is the more reliable fidelity signal and is the largest of all models.

## 3. vLLM (CUDA graphs) vs HF eager (n=5, BF16)
| Model | Ctx | Engine | TTFT ms | Decode tok/s | Total J/tok | MBU % (analytical) |
|---|---:|---|---:|---:|---:|---:|
| Falcon3-7B | 512 | hf_eager | 153.5±7.0 | 16.43±0.03 | 4.376±0.034 | 82.0 |
| Falcon3-7B | 512 | vllm | 146.8±4.8 | 17.54±0.02 | 4.079±0.063 | 87.6 |
| Falcon3-7B | 2048 | hf_eager | 552.0±6.2 | 15.91±0.02 | 4.529±0.011 | 80.4 |
| Falcon3-7B | 2048 | vllm | 554.4±6.1 | 17.34±0.02 | 4.145±0.017 | 87.6 |
| Falcon3-7B | 8192 | hf_eager | 2674.1±23.1 | 14.19±0.03 | 5.074±0.012 | 75.0 |
| Falcon3-7B | 8192 | vllm | 2367.3±26.9 | 16.55±0.02 | 4.348±0.011 | 87.5 |
| Falcon3-Mamba-7B | 512 | hf_eager | 225.0±5.2 | 9.38±0.03 | 7.005±0.110 | 45.7 |
| Falcon3-Mamba-7B | 512 | vllm | 221.9±6.5 | 17.26±0.04 | 4.140±0.062 | 84.1 |
| Falcon3-Mamba-7B | 2048 | hf_eager | 819.9±8.6 | 9.45±0.04 | 6.788±0.084 | 46.0 |
| Falcon3-Mamba-7B | 2048 | vllm | 859.9±11.7 | 17.29±0.01 | 4.152±0.012 | 84.3 |
| Falcon3-Mamba-7B | 8192 | hf_eager | 3605.4±56.5 | 9.51±0.06 | 7.035±0.063 | 46.3 |
| Falcon3-Mamba-7B | 8192 | vllm | 3354.2±15.0 | 17.27±0.01 | 4.152±0.012 | 84.2 |
| Falcon-H1-7B | 512 | hf_eager | 258.2±175.4 | 9.02±0.05 | 7.044±0.213 | 46.1 |
| Falcon-H1-7B | 512 | vllm | 176.2±9.7 | 16.59±0.02 | 4.294±0.089 | 84.8 |
| Falcon-H1-7B | 2048 | hf_eager | 680.4±12.5 | 9.06±0.08 | 7.489±0.115 | 46.7 |
| Falcon-H1-7B | 2048 | vllm | 623.0±6.8 | 16.42±0.03 | 4.390±0.010 | 84.4 |
| Falcon-H1-7B | 8192 | hf_eager | 3259.3±46.1 | 8.96±0.03 | 7.676±0.124 | 47.9 |
| Falcon-H1-7B | 8192 | vllm | 2643.0±8.9 | 16.15±0.05 | 4.450±0.019 | 84.5 |

MFU is 0.18–0.22% everywhere (batch-1 decode is memory-bound); include the column for completeness.

Supported statements:
- vLLM CUDA graphs speed up decode **1.8–1.9×** for Mamba (9.38→17.26 at 512; 9.51→17.27 at 8K) and H1 (9.02→16.59; 8.96→16.15) vs **1.07–1.17×** for Falcon3-7B, cutting total J/tok by ~41% (Mamba) and ~39–42% (H1). This shows the eager gap is mostly host/launch overhead.
- Under vLLM at 8K: **Falcon3-Mamba-7B is fastest and most efficient** (17.27±0.01 tok/s, 4.152 J/tok), ahead of Falcon3-7B (16.55±0.02, 4.348) by 4.4% in speed and 4.5% in energy — a gap far larger than run-to-run std. **Falcon-H1-7B does NOT overtake Falcon3-7B** at 8K (16.15±0.05 vs 16.55±0.02 tok/s; 4.450 vs 4.348 J/tok); it trails by ~2.4%. Falcon3-7B leads H1 at every measured context in vLLM.
- Trend: Falcon3-7B vLLM decode falls 5.6% from 512 to 8K; Mamba is flat; H1 falls 2.7%. Extrapolating beyond 8K is not supported by data.
- Analytical MBU: Falcon3-7B ~87.5%, Mamba ~84.2%, H1 ~84.5% under vLLM; eager SSM models ~46%. These are model-based estimates (see §0).
- vLLM v1 config note (still valid): default `max_num_seqs=256` reserves Mamba state for 256 sequences and OOMs/rejects on a 24 GB L4 for both SSM models; `max_num_seqs=8` fixes it. FlashInfer JIT sampler needs `nvcc` and fails on CUDA 12.4 with `--compress-mode=size`; workaround: `VLLM_USE_FLASHINFER_SAMPLER=0` (+ strip flag).

## 4. Idle power reconciliation
Pre-load idle floor per container: Falcon3 13.5–16.4 W; Mamba 33.4–35.4 W; H1 29.9–35.3 W (varies by container start state). Post-load idle (weights + CUDA context resident): 36.1–43.6 W. Older documents' 16.16 W / 18.28–19.51 W values are from different containers in v1. Because the floor is container-dependent, dynamic J/tok is not compared across models.

## 5. What to call things
- Use "informed by IEEE P1959/P1960 working-group discussions" — not "aligned" or "compliant".
- "show", not "prove". "near-parity around 2K", not "exact crossover". "within noise" where SE overlaps.
- L4 memory is GDDR6.

## 6. v1 claims that are withdrawn
1. "Uniform INT4 collapses Falcon3-Mamba accuracy 80%→40%" — withdrawn (5-item probe; does not replicate at N=200 templated).
2. "int4_mixed_ssm: zero accuracy loss / 100% retained" — withdrawn; accuracy probe does not separate recipes; KL and ΔPPL are non-zero.
3. "int4_mixed_ssm recovers Mamba" as a compression recipe — reframed as a control (no compression).
4. "Falcon-H1 overtakes Falcon3-7B at 8K under vLLM" — withdrawn; H1 trails by 2.4%. (Falcon3-Mamba-7B does lead.)
5. "Exact VRAM crossover at N=2048" — softened to near-parity around 2K.
6. "47.8% faster TTFT for int4_mixed_ssm" — withdrawn (noise at 512; equal at 2K/8K).
7. "Fits/unlocks Falcon-H1 on 8 GB GPUs" — withdrawn (peak 7.0–8.1 GB; untested on 8 GB hardware).
8. "300 GB/s HBM" — corrected to GDDR6. "94.4% MBU" for Falcon3 vLLM — was inflated by prefix-cache reuse of the TTFT prompt; v2 is 87.5% (analytical).
9. "2.36–2.54× vLLM speedup" — v2 is 1.8–1.9× (v2 eager baseline is faster: 9.0–9.5 tok/s vs v1 7.0 tok/s; v1 eager numbers are superseded).
