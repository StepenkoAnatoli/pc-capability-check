# pc-capability-check

Small cross-platform utility to report PC hardware and estimate local LLM inference and fine-tuning suitability.

## Quick start

### Requirements
- Python 3.x
- No third-party Python packages required

### Run (human-readable output)

```bash
python pc_capability_check.py
```

### Run (JSON output)

```bash
python pc_capability_check.py --json
```

### Help

```bash
python pc_capability_check.py --help
```

## Supported operating systems

- Linux
- macOS
- Windows

The tool attempts OS-specific hardware discovery and degrades gracefully if a field cannot be detected.

## Example output (human-readable)

```text
System Summary
==============
OS: Linux 6.8.0-1024-azure (x86_64)
Platform: Linux-6.8.0-1024-azure-x86_64-with-glibc2.39
CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
Logical cores: 2
Total RAM: 6.75 GB
Disk (root/system): 13.58 GB total, 12.90 GB free
GPU(s): Unavailable or not detected

LLM Capability Estimate
=======================
Mode: CPU-only or unknown GPU VRAM
Inference: Limited local inference; focus on tiny models (<3B)
Fine-tuning/QLoRA: Local fine-tuning generally not recommended without a capable GPU

Recommendations:
- If possible, add or upgrade a discrete GPU for significantly better results.
- For CPU-only setups, prioritize smaller quantized models.

Caveats:
- These are conservative estimates, not guarantees.
- Actual usability depends on model architecture, quantization, context length, and software stack.
- Integrated GPUs and shared memory can vary significantly in real performance.
```

## JSON output

`--json` prints machine-readable structured output with sections:
- `platform`
- `cpu`
- `memory`
- `disk`
- `gpu`
- `llm_capability`

Unavailable values are returned as `"Unavailable"` for text fields and `null` for unknown numeric fields.

## What is detected

- OS/platform details
- CPU model and logical core count
- Total RAM
- Root/system disk total and free space
- GPU list (vendor/name and VRAM when practical for the OS and available system tools)

## How recommendations are calculated

This tool uses transparent, conservative heuristics (not benchmarks):

- **Primary signal:** detected dedicated GPU VRAM
- **Secondary signal:** total system RAM for CPU-only/unknown-GPU scenarios
- Guidance distinguishes:
  - CPU-only or unknown-GPU paths
  - GPU inference suitability
  - Fine-tuning/QLoRA practicality

Approximate model classes used in guidance: **3B / 7B / 13B / 70B**.

## Limitations

- Hardware APIs vary by OS; some environments (minimal containers/VMs) expose limited information.
- Linux GPU VRAM is often unavailable without vendor tooling.
- Reported suitability is conservative and should not be treated as a guarantee.

## Privacy

All checks run locally on your machine. The script does not send hardware details to any external service.

## Troubleshooting

- If GPU fields are unavailable, verify OS-level tooling exists (`lspci`/`nvidia-smi` on Linux, `system_profiler` on macOS). On Windows, modern systems may not ship `wmic`; this tool then attempts a PowerShell/CIM fallback, but some environments may still report `Unavailable`.
- If running in a VM/container, some hardware data may be hidden by the host.
- Use `--json` to integrate with automation and inspect raw detected fields.
