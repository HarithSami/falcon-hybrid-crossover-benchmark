"""
Generate 2x2 Vector PDF & High-Res PNG Plots with Error Bars from Modal Benchmark JSONs
=======================================================================================
Exports publication-ready 2x2 figures (both vector `.pdf` for LaTeX inclusion and `.png`
for GitHub/LinkedIn) with 9-10pt typography, error bars (mean ± std / SE), and
within-model relative ΔPPL% (avoiding raw cross-tokenizer PPL comparisons).

Usage:
  python src/plot_results.py results/benchmark_crossover.json
  python src/plot_results.py results/benchmark_quant-ablation.json
  python src/plot_results.py results/benchmark_vllm_mfu.json
"""

import json
import sys
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _format_k_ticks(x, pos=None):
    if x == 0:
        return "0"
    if x >= 1024:
        val = x / 1024
        return f"{val:.0f}K" if val.is_integer() else f"{val:.1f}K"
    elif x >= 1000:
        val = x / 1000
        return f"{val:.0f}K" if val.is_integer() else f"{val:.1f}K"
    return f"{int(x)}"


def _apply_pub_style(plt):
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "figure.titlesize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def generate_summary_and_plots(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "vllm_cudagraph" in raw_data:
        generate_vllm_comparison_plots(path, raw_data)
        return

    rows = [r for r in raw_data if r.get("status") == "ok"]
    if not rows:
        print("No valid benchmark records found.")
        return

    precisions_present = sorted(set(r["precision"] for r in rows))
    is_quant_ablation = len(precisions_present) > 1

    print(f"\n### Benchmark Summary ({path.name})\n")
    print(
        "| Architecture | Precision | Ctx | TTFT (ms) | Decode (tok/s) | Persist State (MB) | Peak VRAM (MB) | Energy (J/tok) | ΔPPL (%) | KL vs BF16 | Top-1 Agree | Acc (N=200) |"
    )
    print(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows:
        ttft_str = f"{r['ttft_ms']:.1f}±{r.get('ttft_std_ms', 0.0):.1f}"
        tps_str = f"{r['decode_tok_per_s']:.2f}±{r.get('decode_tok_per_s_std', 0.0):.2f}"
        pstate_mb = r.get("analytical_persistent_state_mb", r.get("kv_or_state_growth_mb", 0.0))
        jt_str = f"{r['joules_per_token_total']:.4f}±{r.get('joules_per_token_total_std', 0.0):.4f}"
        dppl_str = f"{r.get('ppl_delta_pct', 0.0):+.2f}±{r.get('ppl_delta_se_pct', 0.0):.2f}%"
        kl_str = f"{r.get('kl_div_vs_bf16', 0.0):.4f}±{r.get('kl_div_se', 0.0):.4f}"
        agree_str = f"{r.get('top1_agree_pct', 100.0):.1f}%"
        acc_str = f"{r.get('task_accuracy_pct', 0.0):.1f}±{r.get('task_accuracy_se_pct', 0.0):.1f}%"
        print(
            f"| {r['arch_label']} | `{r['precision']}` | {r['context_length']} | "
            f"{ttft_str} | {tps_str} | {pstate_mb:.1f} | {r['peak_vram_mb']:.1f} | "
            f"**{jt_str}** | {dppl_str} | {kl_str} | {agree_str} | {acc_str} |"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _apply_pub_style(plt)
    out_dir = path.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    if is_quant_ablation:
        _plot_quant_ablation_2x2(plt, rows, out_dir, path.stem)
    else:
        _plot_crossover_2x2(plt, rows, out_dir, path.stem)


def _plot_crossover_2x2(plt, rows: list[dict], out_dir: Path, stem: str):
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["arch_label"], []).append(r)
    for k in groups:
        groups[k].sort(key=lambda x: x["context_length"])

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0))
    ax_tps, ax_mem = axes[0, 0], axes[0, 1]
    ax_eng, ax_ttft = axes[1, 0], axes[1, 1]

    palette = {
        "Pure Transformer (7B)": ("#1f77b4", "o"),
        "Pure SSM / Mamba (7B)": ("#2ca02c", "s"),
        "Parallel Hybrid (7B)": ("#d62728", "^"),
        "Transformer (1B)": ("#1f77b4", "o"),
        "Hybrid Mamba-Attn (1.5B)": ("#d62728", "^"),
    }

    ctx_ticks = [0, 4096, 8192, 12288, 16384]

    for arch, grp in groups.items():
        color, marker = palette.get(arch, ("#333333", "o"))
        ctxs = [g["context_length"] for g in grp]
        tps = [g["decode_tok_per_s"] for g in grp]
        tps_err = [g.get("decode_tok_per_s_std", 0.0) for g in grp]
        pstate = [g.get("analytical_persistent_state_mb", g["kv_or_state_growth_mb"]) for g in grp]
        peak_grow = [g["kv_or_state_growth_mb"] for g in grp]
        j_tok = [g["joules_per_token_total"] for g in grp]
        j_err = [g.get("joules_per_token_total_std", 0.0) for g in grp]
        ttft = [g["ttft_ms"] for g in grp]
        ttft_err = [g.get("ttft_std_ms", 0.0) for g in grp]

        ax_tps.errorbar(ctxs, tps, yerr=tps_err, color=color, marker=marker, markersize=6, capsize=3, linewidth=2.0, label=arch)
        ax_mem.plot(ctxs, pstate, color=color, marker=marker, markersize=6, linestyle="-", linewidth=2.0)
        ax_mem.plot(ctxs, peak_grow, color=color, marker=marker, markersize=5, linestyle=":", linewidth=1.8, alpha=0.85)
        ax_eng.errorbar(ctxs, j_tok, yerr=j_err, color=color, marker=marker, markersize=6, capsize=3, linewidth=2.0, label=arch)
        ax_ttft.errorbar(ctxs, ttft, yerr=ttft_err, color=color, marker=marker, markersize=6, capsize=3, linewidth=2.0, label=arch)

    for ax in (ax_tps, ax_mem, ax_eng, ax_ttft):
        ax.set_xticks(ctx_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k_ticks))
        ax.set_xlim(-500, 17200)
        ax.grid(True, alpha=0.3)

    ax_tps.set_title("(a) Decode Throughput vs. Context Length (HF Eager)")
    ax_tps.set_xlabel("Context Length (Tokens)")
    ax_tps.set_ylabel("Decode Throughput (tok/s ± std)")
    ax_tps.set_ylim(7.5, 17.5)

    ax_mem.set_title("(b) Persistent State vs. Peak Prefill ΔVRAM")
    ax_mem.set_xlabel("Context Length (Tokens)")
    ax_mem.set_ylabel("Memory Above Static Weights (MB)")
    ax_mem.set_ylim(-300, 7500)
    # Clean 2-entry legend in upper left that doesn't crowd curves
    mem_handles = [
        Line2D([], [], color="#333333", linestyle="-", linewidth=2.0, label="Persistent State (exact KV/SSM)"),
        Line2D([], [], color="#333333", linestyle=":", linewidth=1.8, label="Peak Prefill ΔVRAM (workspace)"),
    ]
    ax_mem.legend(handles=mem_handles, loc="upper left", framealpha=0.92, fontsize=8.5)

    ax_eng.set_title("(c) Total Decode Energy per Token (HF Eager)")
    ax_eng.set_xlabel("Context Length (Tokens)")
    ax_eng.set_ylabel("Joules / Output Token (± std)")
    ax_eng.set_ylim(3.5, 8.8)

    ax_ttft.set_title("(d) Prefill Time-To-First-Token (TTFT)")
    ax_ttft.set_xlabel("Context Length (Tokens)")
    ax_ttft.set_ylabel("TTFT (ms ± std)")
    ax_ttft.set_ylim(-200, 7800)

    # Shared top legend for architectures across the whole figure
    shared_handles = [
        Line2D([], [], color=c, marker=m, linestyle="-", linewidth=2.0, markersize=7, label=a)
        for a, (c, m) in palette.items()
        if a in groups
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(shared_handles),
        frameon=True,
        framealpha=0.95,
        fontsize=9.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.5, h_pad=2.2)
    for ext in ("png", "pdf"):
        out_f = out_dir / f"{stem}_charts.{ext}"
        plt.savefig(out_f, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out_f}")
    plt.close(fig)


def _plot_quant_ablation_2x2(plt, rows: list[dict], out_dir: Path, stem: str):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0))
    ax_dppl, ax_kl = axes[0, 0], axes[0, 1]
    ax_acc, ax_eng = axes[1, 0], axes[1, 1]

    # Deduplicate per (arch_label, precision) for corpus metrics
    by_arch_prec: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_arch_prec.setdefault((r["arch_label"], r["precision"]), []).append(r)

    arch_order = ["Pure Transformer (7B)", "Pure SSM / Mamba (7B)", "Parallel Hybrid (7B)"]
    if not any(a in [k[0] for k in by_arch_prec] for a in arch_order):
        arch_order = sorted(set(k[0] for k in by_arch_prec))

    # Distinct precision palette (Slate, Amber, Purple) that does not collide with architecture palette
    prec_colors = {
        "bf16": "#34495e",          # Slate / Charcoal
        "int4_uniform": "#d97706",   # Amber / Gold
        "int4_mixed_ssm": "#7b3294", # Purple / Violet
    }

    arch_markers = {
        "Pure Transformer (7B)": "o",
        "Pure SSM / Mamba (7B)": "s",
        "Parallel Hybrid (7B)": "^",
    }

    # Panel (a): Relative Perplexity Drift (ΔPPL% ± SE) vs. Static Weight Footprint (GiB)
    for (arch, prec), grp in by_arch_prec.items():
        r0 = grp[0]
        ax_dppl.errorbar(
            [r0["weights_vram_mb"] / 1024.0],
            [r0.get("ppl_delta_pct", 0.0)],
            yerr=[r0.get("ppl_delta_se_pct", 0.0)],
            fmt=arch_markers.get(arch, "o"),
            markersize=8,
            color=prec_colors.get(prec, "#333333"),
            capsize=4,
            linewidth=1.8,
        )

    ax_dppl.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_dppl.set_title("(a) Relative Perplexity Drift (ΔPPL%) vs. Weights")
    ax_dppl.set_xlabel("Static Weight Footprint (GiB)")
    ax_dppl.set_ylabel("Relative ΔPPL vs. Own BF16 (% ± SE)")
    ax_dppl.set_ylim(-4.5, 5.5)
    ax_dppl.set_xlim(3.5, 15.5)
    ax_dppl.grid(True, alpha=0.3)

    # Panel (b): Token-Level Logit KL Divergence KL(P_bf16 || Q) ± SE (quantized modes)
    quant_precs = ["int4_uniform", "int4_mixed_ssm"]
    x_pos = range(len(arch_order))
    width = 0.32
    for idx, prec in enumerate(quant_precs):
        kls = []
        kl_ses = []
        for arch_idx, arch in enumerate(arch_order):
            grp = by_arch_prec.get((arch, prec), [])
            offset = arch_idx + (idx - 0.5) * width
            if grp:
                val = grp[0].get("kl_div_vs_bf16", 0.0)
                se = grp[0].get("kl_div_se", 0.0)
                ax_kl.bar(offset, val, width=width, yerr=se, capsize=3, color=prec_colors[prec], alpha=0.92)
            else:
                # Pure Transformer has no int4_mixed_ssm
                ax_kl.bar(offset, 0.04, width=width, fill=False, edgecolor="#cccccc", linestyle="--", linewidth=1.0)
                ax_kl.text(offset, 0.045, "N/A\n(no SSM)", ha="center", va="bottom", fontsize=8.0, color="#555555", fontweight="bold")

    ax_kl.set_xticks(list(x_pos))
    ax_kl.set_xticklabels([a.replace(" (7B)", "") for a in arch_order])
    ax_kl.set_title(r"(b) Token-Level Logit $\mathrm{KL}(P_{\mathrm{bf16}} \parallel Q)$ (± SE)")
    ax_kl.set_ylabel("Mean KL Divergence (nats)")
    ax_kl.set_ylim(0, 0.22)
    ax_kl.grid(True, axis="y", alpha=0.3)

    # Panel (c): 200-Item Zero-Shot Benchmark Accuracy (% ± Binomial SE)
    all_precs = ["bf16", "int4_uniform", "int4_mixed_ssm"]
    w3 = 0.24
    for idx, prec in enumerate(all_precs):
        for arch_idx, arch in enumerate(arch_order):
            grp = by_arch_prec.get((arch, prec), [])
            offset = arch_idx + (idx - 1.0) * w3
            if grp:
                acc = grp[0].get("task_accuracy_pct", 0.0)
                acc_se = grp[0].get("task_accuracy_se_pct", 0.0)
                ax_acc.bar(offset, acc, width=w3, yerr=acc_se, capsize=3, color=prec_colors[prec], alpha=0.92)
            else:
                # Pure Transformer has no int4_mixed_ssm
                ax_acc.bar(offset, 25.0, width=w3, fill=False, edgecolor="#cccccc", linestyle="--", linewidth=1.0)
                ax_acc.text(offset, 27.0, "N/A\n(no SSM)", ha="center", va="bottom", fontsize=8.0, color="#555555", fontweight="bold")

    ax_acc.set_xticks(list(x_pos))
    ax_acc.set_xticklabels([a.replace(" (7B)", "") for a in arch_order])
    ax_acc.set_ylim(0, 115)
    ax_acc.set_title("(c) 200-Item Zero-Shot Accuracy (% ± SE, N=200)")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, axis="y", alpha=0.3)

    # Panel (d): Long-Context Energy per Token (J/tok ± std) across Precisions
    max_ctx = max(r["context_length"] for r in rows)
    for idx, prec in enumerate(all_precs):
        for arch_idx, arch in enumerate(arch_order):
            grp = [g for g in by_arch_prec.get((arch, prec), []) if g["context_length"] == max_ctx]
            offset = arch_idx + (idx - 1.0) * w3
            if grp:
                j_val = grp[0]["joules_per_token_total"]
                j_err = grp[0].get("joules_per_token_total_std", 0.0)
                ax_eng.bar(offset, j_val, width=w3, yerr=j_err, capsize=3, color=prec_colors[prec], alpha=0.92)
            else:
                # Pure Transformer has no int4_mixed_ssm
                ax_eng.bar(offset, 2.0, width=w3, fill=False, edgecolor="#cccccc", linestyle="--", linewidth=1.0)
                ax_eng.text(offset, 2.2, "N/A\n(no SSM)", ha="center", va="bottom", fontsize=8.0, color="#555555", fontweight="bold")

    ax_eng.set_xticks(list(x_pos))
    ax_eng.set_xticklabels([a.replace(" (7B)", "") for a in arch_order])
    ax_eng.set_title(f"(d) Total Decode Energy at N={max_ctx} (J/tok ± std)")
    ax_eng.set_ylabel("Joules / Token (± std)")
    ax_eng.set_ylim(0, 10.5)
    ax_eng.grid(True, axis="y", alpha=0.3)

    # Clean shared top legend across the figure (precisions + architecture markers)
    prec_legend_handles = [
        Patch(facecolor=prec_colors["bf16"], label="BF16"),
        Patch(facecolor=prec_colors["int4_uniform"], label="INT4 Uniform (NF4)"),
        Patch(facecolor=prec_colors["int4_mixed_ssm"], label="INT4 Mixed SSM"),
    ]
    arch_legend_handles = [
        Line2D([], [], color="#333333", marker=arch_markers["Pure Transformer (7B)"], linestyle="", markersize=7, label="Transformer"),
        Line2D([], [], color="#333333", marker=arch_markers["Pure SSM / Mamba (7B)"], linestyle="", markersize=7, label="Mamba SSM"),
        Line2D([], [], color="#333333", marker=arch_markers["Parallel Hybrid (7B)"], linestyle="", markersize=7, label="Parallel Hybrid"),
    ]
    shared_handles = prec_legend_handles + arch_legend_handles
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=True,
        framealpha=0.95,
        fontsize=9.2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.5, h_pad=2.2)
    for ext in ("png", "pdf"):
        out_f = out_dir / f"{stem}_charts.{ext}"
        plt.savefig(out_f, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out_f}")
    plt.close(fig)


def generate_vllm_comparison_plots(path: Path, data: dict):
    hf_rows = [r for r in data.get("hf_roofline", []) if r["context_length"] in (512, 2048, 8192)]
    vllm_rows = data.get("vllm_cudagraph", [])

    print("\n### vLLM CUDA-Graph vs. Hugging Face Eager Roofline MFU/MBU Summary (n=5 Trials)\n")
    print(
        "| Architecture | Context | Engine | TTFT (ms) | Decode (tok/s) | Energy Total (J/tok) | Energy Dyn (J/tok) | GDDR6 BW (GB/s) | MBU (%) | MFU (%) |"
    )
    print(
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    combined = []
    for r in hf_rows:
        combined.append({**r, "engine": "hf_eager"})
    for r in vllm_rows:
        combined.append(r)

    combined.sort(key=lambda x: (x["arch_label"], x["context_length"], x["engine"]))
    for r in combined:
        ttft_s = f"{r['ttft_ms']:.1f}±{r.get('ttft_std_ms', 0.0):.1f}"
        tps_s = f"{r['decode_tok_per_s']:.2f}±{r.get('decode_tok_per_s_std', 0.0):.2f}"
        jt_s = f"{r['joules_per_token_total']:.4f}±{r.get('joules_per_token_total_std', 0.0):.4f}"
        jd_s = f"{r['joules_per_token_dynamic']:.4f}±{r.get('joules_per_token_dynamic_std', 0.0):.4f}"
        bw_val = r.get("achieved_mem_bw_gbps", r.get("achieved_hbm_bw_gbps", 0.0))
        bw_s = f"{bw_val:.1f}±{r.get('achieved_mem_bw_gbps_std', 0.0):.1f}"
        mbu_s = f"{r['mbu_pct']:.1f}±{r.get('mbu_pct_std', 0.0):.1f}%"
        mfu_s = f"{r['mfu_pct']:.2f}%"
        print(
            f"| {r['arch_label']} | {r['context_length']} | `{r['engine']}` | "
            f"{ttft_s} | **{tps_s}** | **{jt_s}** | {jd_s} | {bw_s} | **{mbu_s}** | {mfu_s} |"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _apply_pub_style(plt)
    out_dir = path.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0))
    ax_tps, ax_mbu = axes[0, 0], axes[0, 1]
    ax_eng, ax_ttft = axes[1, 0], axes[1, 1]

    # Model architecture palette matching Figures 1 & 3
    palette = {
        "Pure Transformer (7B)": "#1f77b4",
        "Pure SSM / Mamba (7B)": "#2ca02c",
        "Parallel Hybrid (7B)": "#d62728",
    }

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in combined:
        groups.setdefault((r["arch_label"], r["engine"]), []).append(r)

    vllm_ctx_ticks = [512, 2048, 8192]

    for (arch, eng), grp in groups.items():
        grp.sort(key=lambda x: x["context_length"])
        ctxs = [g["context_length"] for g in grp]
        tps = [g["decode_tok_per_s"] for g in grp]
        tps_e = [g.get("decode_tok_per_s_std", 0.0) for g in grp]
        j_tok = [g["joules_per_token_total"] for g in grp]
        j_e = [g.get("joules_per_token_total_std", 0.0) for g in grp]
        mbu = [g["mbu_pct"] for g in grp]
        mbu_e = [g.get("mbu_pct_std", 0.0) for g in grp]
        ttft = [g["ttft_ms"] for g in grp]
        ttft_e = [g.get("ttft_std_ms", 0.0) for g in grp]

        color = palette.get(arch, "#333333")
        style = "-" if eng == "vllm_cudagraph" else "--"
        marker = "o" if eng == "vllm_cudagraph" else "s"

        ax_tps.errorbar(ctxs, tps, yerr=tps_e, color=color, linestyle=style, marker=marker, markersize=6, capsize=3, linewidth=2.0)
        ax_mbu.errorbar(ctxs, mbu, yerr=mbu_e, color=color, linestyle=style, marker=marker, markersize=6, capsize=3, linewidth=2.0)
        ax_eng.errorbar(ctxs, j_tok, yerr=j_e, color=color, linestyle=style, marker=marker, markersize=6, capsize=3, linewidth=2.0)
        ax_ttft.errorbar(ctxs, ttft, yerr=ttft_e, color=color, linestyle=style, marker=marker, markersize=6, capsize=3, linewidth=2.0)

    for ax in (ax_tps, ax_mbu, ax_eng, ax_ttft):
        ax.set_xticks(vllm_ctx_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k_ticks))
        ax.set_xlim(200, 8500)
        ax.grid(True, alpha=0.3)

    ax_tps.set_title("(a) Decode Throughput: vLLM vs. HF Eager")
    ax_tps.set_xlabel("Context Length (Tokens)")
    ax_tps.set_ylabel("Throughput (tok/s ± std)")
    ax_tps.set_ylim(7.0, 19.5)

    ax_mbu.set_title("(b) GDDR6 Model Bandwidth Utilization (MBU %)")
    ax_mbu.set_xlabel("Context Length (Tokens)")
    ax_mbu.set_ylabel("% of L4 300 GB/s Peak (± std)")
    ax_mbu.axhline(100.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Peak Bandwidth (300 GB/s)")
    ax_mbu.set_ylim(38.0, 103.0)

    ax_eng.set_title("(c) Total Decode Energy per Token")
    ax_eng.set_xlabel("Context Length (Tokens)")
    ax_eng.set_ylabel("Joules / Output Token (± std)")
    ax_eng.set_ylim(3.5, 8.5)

    ax_ttft.set_title("(d) Prefill Time-To-First-Token (TTFT)")
    ax_ttft.set_xlabel("Context Length (Tokens)")
    ax_ttft.set_ylabel("TTFT (ms ± std)")
    ax_ttft.set_ylim(0, 4000)

    # Shared top legend: 3 architectures + 2 engine styles
    arch_handles = [
        Line2D([], [], color=palette["Pure Transformer (7B)"], marker="o", linestyle="-", linewidth=2.0, markersize=7, label="Pure Transformer"),
        Line2D([], [], color=palette["Pure SSM / Mamba (7B)"], marker="o", linestyle="-", linewidth=2.0, markersize=7, label="Pure SSM / Mamba"),
        Line2D([], [], color=palette["Parallel Hybrid (7B)"], marker="o", linestyle="-", linewidth=2.0, markersize=7, label="Parallel Hybrid"),
    ]
    engine_handles = [
        Line2D([], [], color="#333333", linestyle="-", marker="o", linewidth=2.0, markersize=6, label="vLLM (CUDA Graph)"),
        Line2D([], [], color="#333333", linestyle="--", marker="s", linewidth=1.8, markersize=6, label="HF Eager"),
    ]
    shared_handles = arch_handles + engine_handles
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=True,
        framealpha=0.95,
        fontsize=9.2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.5, h_pad=2.2)
    for ext in ("png", "pdf"):
        out_f = out_dir / f"{path.stem}_charts.{ext}"
        plt.savefig(out_f, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out_f}")
    plt.close(fig)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_smoke.json"
    generate_summary_and_plots(target)
