"""
vLLM & Hugging Face Energy, Roofline MFU, and Hybrid MBU Telemetry Profiler
===========================================================================
Open-source hardware telemetry instrumentation aligned with IEEE P1959/P1960
energy-efficiency principles. Designed for evaluating Pure Transformers,
Pure State-Space Models (Mamba / Mamba-2), and Parallel Hybrid architectures
(such as TII's Falcon-H1 series) under vLLM (CUDA Graphs / PagedAttention)
and Hugging Face Transformers.

Key Capabilities:
  1. Hardware NVML Energy Accumulator (`nvmlDeviceGetTotalEnergyConsumption`)
     paired with high-frequency 20ms background power polling.
  2. Idle Power Floor Calibration (`P_idle`) to separate:
       - Total Datacenter Energy per Token (`joules_per_token_total`)
       - Dynamic Marginal Switching Energy per Token (`joules_per_token_dynamic`)
  3. Roofline Model FLOPs Utilization (MFU %) & Model Bandwidth Utilization (MBU %)
     accounting for both dynamic Transformer KV-cache reads and fixed O(1)
     Mamba SSM recurrent state updates during autoregressive decoding.

Author: Harith Ibrahim
License: Apache-2.0
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 1. Datacenter GPU Hardware Specifications (BF16 Tensor Core Peak & HBM BW)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_bf16_tflops: float  # Dense BF16/FP16 Tensor Core TFLOP/s
    peak_hbm_bw_gbps: float  # Peak memory bandwidth in GB/s (10^9 bytes/s)
    tdp_watts: float


KNOWN_GPU_SPECS: dict[str, GPUSpec] = {
    "L4": GPUSpec("NVIDIA L4", peak_bf16_tflops=121.0, peak_hbm_bw_gbps=300.0, tdp_watts=72.0),
    "A10G": GPUSpec("NVIDIA A10G", peak_bf16_tflops=125.0, peak_hbm_bw_gbps=600.0, tdp_watts=300.0),
    "A100-SXM4-40GB": GPUSpec("NVIDIA A100 40GB", peak_bf16_tflops=312.0, peak_hbm_bw_gbps=1555.0, tdp_watts=400.0),
    "A100-SXM4-80GB": GPUSpec("NVIDIA A100 80GB", peak_bf16_tflops=312.0, peak_hbm_bw_gbps=2039.0, tdp_watts=400.0),
    "H100": GPUSpec("NVIDIA H100 SXM5", peak_bf16_tflops=989.0, peak_hbm_bw_gbps=3350.0, tdp_watts=700.0),
    "T4": GPUSpec("Tesla T4", peak_bf16_tflops=65.0, peak_hbm_bw_gbps=320.0, tdp_watts=70.0),
}


def resolve_gpu_spec(gpu_name: str) -> GPUSpec:
    """Matches an NVML device name string to its hardware Roofline specification."""
    for key, spec in KNOWN_GPU_SPECS.items():
        if key.lower() in gpu_name.lower():
            return spec
    # Conservative fallback (L4-class)
    return GPUSpec(gpu_name, peak_bf16_tflops=121.0, peak_hbm_bw_gbps=300.0, tdp_watts=72.0)


# ---------------------------------------------------------------------------
# 2. Architecture Parameter & KV/SSM State Calculator
# ---------------------------------------------------------------------------
@dataclass
class ArchitectureProfile:
    model_id: str
    arch_type: str  # "transformer", "mamba", or "hybrid"
    num_params: int
    bytes_per_param: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    ssm_state_bytes: int  # Fixed recurrent state footprint in bytes


def build_architecture_profile(
    model_id: str,
    hf_config: Any,
    weights_vram_bytes: int,
    precision: str = "bf16",
) -> ArchitectureProfile:
    """
    Extracts exact KV-cache and SSM state geometry from a Hugging Face PretrainedConfig
    (supports Falcon3, FalconMamba, and FalconH1 parallel hybrid configs).
    """
    model_type = getattr(hf_config, "model_type", "").lower()
    num_layers = int(getattr(hf_config, "num_hidden_layers", 32))
    hidden_size = int(getattr(hf_config, "hidden_size", 4096))
    num_attn_heads = int(getattr(hf_config, "num_attention_heads", 32))
    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_attn_heads))
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // max(num_attn_heads, 1)))

    bytes_per_param = 2.0 if precision == "bf16" else (0.55 if precision == "int4_uniform" else 0.85)
    est_params = int(weights_vram_bytes / max(bytes_per_param, 0.5))

    if "mamba" in model_type and "h1" not in model_type and "hybrid" not in model_type:
        arch_type = "mamba"
        num_kv_heads = 0
        ssm_d_state = int(getattr(hf_config, "state_size", getattr(hf_config, "ssm_state_size", 16)))
        ssm_expand = int(getattr(hf_config, "expand", 2))
        ssm_state_bytes = num_layers * hidden_size * ssm_expand * ssm_d_state * 2
    elif "h1" in model_type or "hybrid" in model_type:
        arch_type = "hybrid"
        ssm_d_state = int(getattr(hf_config, "mamba_d_state", 128))
        ssm_n_heads = int(getattr(hf_config, "mamba_n_heads", 64))
        ssm_d_head = int(getattr(hf_config, "mamba_d_head", 64))
        ssm_state_bytes = num_layers * ssm_n_heads * ssm_d_head * ssm_d_state * 2
    else:
        arch_type = "transformer"
        ssm_state_bytes = 0

    return ArchitectureProfile(
        model_id=model_id,
        arch_type=arch_type,
        num_params=est_params,
        bytes_per_param=bytes_per_param,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ssm_state_bytes=ssm_state_bytes,
    )


# ---------------------------------------------------------------------------
# 3. NVML Hardware Energy & Roofline MFU/MBU Monitor
# ---------------------------------------------------------------------------
@dataclass
class TelemetryReport:
    model_id: str
    engine: str
    precision: str
    gpu_name: str
    context_length: int
    decode_tokens: int
    batch_size: int
    duration_s: float
    decode_tok_per_s: float
    tpot_ms: float
    idle_watts: float
    mean_watts: float
    peak_watts: float
    total_joules: float
    joules_per_token_total: float
    joules_per_token_dynamic: float
    achieved_tflops: float
    mfu_pct: float
    achieved_hbm_bw_gbps: float
    mbu_pct: float
    arithmetic_intensity_flops_per_byte: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMEnergyMFUProfiler:
    """
    Context-managed hardware energy & Roofline MFU/MBU profiler for vLLM and
    Hugging Face inference workloads.
    """

    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.02):
        import pynvml

        self.pynvml = pynvml
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
        raw_name = self.pynvml.nvmlDeviceGetName(self.handle)
        self.gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        self.gpu_spec = resolve_gpu_spec(self.gpu_name)
        self.poll_interval_s = poll_interval_s

        self._samples_watts: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._e_start_mj: int | None = None
        self._e_end_mj: int | None = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self.idle_watts: float = 0.0

    def calibrate_idle_floor(self, duration_s: float = 2.0) -> float:
        """Measures the clean pre-load GPU idle power floor in Watts."""
        self.start()
        time.sleep(duration_s)
        stats = self.stop_raw()
        self.idle_watts = stats["mean_watts"]
        return self.idle_watts

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self._samples_watts.append(mw / 1000.0)
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def start(self) -> None:
        self._samples_watts.clear()
        self._stop_event.clear()
        self._t_start = time.perf_counter()
        try:
            self._e_start_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self._e_start_mj = None
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_raw(self) -> dict[str, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._t_end = time.perf_counter()
        try:
            self._e_end_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self._e_end_mj = None

        duration_s = max(self._t_end - self._t_start, 1e-6)
        mean_w = sum(self._samples_watts) / len(self._samples_watts) if self._samples_watts else 0.0
        peak_w = max(self._samples_watts) if self._samples_watts else 0.0

        if (
            self._e_start_mj is not None
            and self._e_end_mj is not None
            and self._e_end_mj > self._e_start_mj
        ):
            total_joules = (self._e_end_mj - self._e_start_mj) / 1000.0
        else:
            total_joules = mean_w * duration_s

        return {
            "duration_s": duration_s,
            "mean_watts": mean_w,
            "peak_watts": peak_w,
            "total_joules": total_joules,
        }

    def compute_report(
        self,
        raw_stats: dict[str, float],
        arch: ArchitectureProfile,
        weights_vram_bytes: int,
        engine: str,
        precision: str,
        context_length: int,
        decode_tokens: int,
        batch_size: int = 1,
    ) -> TelemetryReport:
        """
        Computes IEEE P1959/P1960 energy metrics alongside Roofline MFU (%) and
        Hybrid MBU (%) for a completed generation window.
        """
        duration_s = raw_stats["duration_s"]
        total_generated_tokens = max(decode_tokens * batch_size, 1)
        tps = total_generated_tokens / duration_s
        tpot_ms = (duration_s * 1000.0) / max(decode_tokens, 1)

        total_joules = raw_stats["total_joules"]
        mean_watts = raw_stats["mean_watts"]
        dynamic_joules = max((mean_watts - self.idle_watts) * duration_s, 0.01)

        j_tok_total = total_joules / total_generated_tokens
        j_tok_dyn = dynamic_joules / total_generated_tokens

        # --- Roofline FLOPs per token & MFU (%) ---
        # Forward pass FLOPs per token ≈ 2 * N_params + attention context FLOPs (4 * L * H_kv * d_k * seq_len)
        avg_seq_len = context_length + (decode_tokens // 2)
        attn_flops_per_tok = 4.0 * arch.num_layers * arch.num_kv_heads * arch.head_dim * avg_seq_len
        flops_per_tok = (2.0 * arch.num_params) + attn_flops_per_tok
        achieved_tflops = (flops_per_tok * tps) / 1e12
        mfu_pct = (achieved_tflops / self.gpu_spec.peak_bf16_tflops) * 100.0

        # --- Roofline Memory Bytes Transferred per Step & MBU (%) ---
        # At each decode step, the GPU streams:
        #   1. Static model weights (weights_vram_bytes)
        #   2. Active KV cache (2 * L * H_kv * d_k * avg_seq_len * 2 bytes * batch_size)
        #   3. Recurrent Mamba SSM state read+write (2 * ssm_state_bytes * batch_size)
        kv_bytes_per_step = (
            2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * avg_seq_len * 2 * batch_size
        )
        ssm_bytes_per_step = 2 * arch.ssm_state_bytes * batch_size
        total_bytes_per_step = weights_vram_bytes + kv_bytes_per_step + ssm_bytes_per_step

        steps_per_s = max(decode_tokens / duration_s, 1e-6)
        achieved_hbm_bw_gbps = (total_bytes_per_step * steps_per_s) / 1e9
        mbu_pct = (achieved_hbm_bw_gbps / self.gpu_spec.peak_hbm_bw_gbps) * 100.0

        arith_intensity = (flops_per_tok * batch_size) / max(total_bytes_per_step, 1)

        return TelemetryReport(
            model_id=arch.model_id,
            engine=engine,
            precision=precision,
            gpu_name=self.gpu_name,
            context_length=context_length,
            decode_tokens=decode_tokens,
            batch_size=batch_size,
            duration_s=round(duration_s, 4),
            decode_tok_per_s=round(tps, 2),
            tpot_ms=round(tpot_ms, 2),
            idle_watts=round(self.idle_watts, 2),
            mean_watts=round(mean_watts, 2),
            peak_watts=round(raw_stats["peak_watts"], 2),
            total_joules=round(total_joules, 3),
            joules_per_token_total=round(j_tok_total, 4),
            joules_per_token_dynamic=round(j_tok_dyn, 4),
            achieved_tflops=round(achieved_tflops, 2),
            mfu_pct=round(mfu_pct, 2),
            achieved_hbm_bw_gbps=round(achieved_hbm_bw_gbps, 2),
            mbu_pct=round(mbu_pct, 2),
            arithmetic_intensity_flops_per_byte=round(arith_intensity, 2),
        )

    def shutdown(self) -> None:
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass
