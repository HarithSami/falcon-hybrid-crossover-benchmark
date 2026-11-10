"""
Recompute the `hf_roofline` section of results/benchmark_vllm_mfu.json from the
current 5-trial results/benchmark_crossover.json (analytical GDDR6 MBU/MFU).

Usage:
  python src/recompute_hf_roofline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vllm_energy_mfu_profiler import ArchitectureProfile  # noqa: E402

PEAK_BW_GBPS = 300.0  # NVIDIA L4 GDDR6
PEAK_BF16_TFLOPS = 121.0

PROFILES = {
    "tiiuae/Falcon3-7B-Instruct": ArchitectureProfile(
        "tiiuae/Falcon3-7B-Instruct", "transformer", 7_455_129_600, 2.0, 28, 4, 256, 0
    ),
    "tiiuae/Falcon3-Mamba-7B-Instruct": ArchitectureProfile(
        "tiiuae/Falcon3-Mamba-7B-Instruct", "mamba", 7_285_350_400, 2.0, 64, 0, 0, 64 * 8192 * (4 + 16) * 2
    ),
    "tiiuae/Falcon-H1-7B-Instruct": ArchitectureProfile(
        "tiiuae/Falcon-H1-7B-Instruct", "hybrid", 7_589_838_848, 2.0, 44, 4, 128, 44 * 64 * 64 * (4 + 128) * 2
    ),
}


def main() -> None:
    cross = json.loads(Path("results/benchmark_crossover.json").read_text(encoding="utf-8"))
    vpath = Path("results/benchmark_vllm_mfu.json")
    vdata = json.loads(vpath.read_text(encoding="utf-8"))

    enriched = []
    for r in cross:
        if r.get("status") != "ok":
            continue
        prof = PROFILES[r["model_id"]]
        w_bytes = int(r["weights_vram_mb"] * 1024**2)
        avg_seq = r["context_length"] + (r.get("decode_tokens", 127) // 2)
        bytes_tok = w_bytes + prof.analytical_kv_cache_bytes(avg_seq) + 2 * prof.ssm_state_bytes
        tps, tps_std = r["decode_tok_per_s"], r.get("decode_tok_per_s_std", 0.0)
        bw = bytes_tok * tps / 1e9
        bw_std = bytes_tok * tps_std / 1e9
        flops_tok = 2.0 * prof.num_params + 4.0 * prof.num_layers * prof.num_kv_heads * prof.head_dim * avg_seq
        tflops = flops_tok * tps / 1e12
        row = dict(r)
        row.update(
            engine="hf_eager",
            memory_type="GDDR6",
            achieved_mem_bw_gbps=round(bw, 2),
            achieved_hbm_bw_gbps=round(bw, 2),
            achieved_mem_bw_gbps_std=round(bw_std, 2),
            mbu_pct=round(bw / PEAK_BW_GBPS * 100, 2),
            mbu_pct_std=round(bw_std / PEAK_BW_GBPS * 100, 2),
            achieved_tflops=round(tflops, 2),
            mfu_pct=round(tflops / PEAK_BF16_TFLOPS * 100, 2),
        )
        enriched.append(row)

    vdata["hf_roofline"] = enriched
    vpath.write_text(json.dumps(vdata, indent=2), encoding="utf-8")
    print(f"Recomputed {len(enriched)} hf_eager roofline rows from 5-trial crossover data.")


if __name__ == "__main__":
    main()
