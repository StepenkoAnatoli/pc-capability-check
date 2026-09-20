# pc-capability-check

Small Windows-focused utility to report PC hardware and estimate local LLM inference and fine-tuning suitability.

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

- Windows

This build is **Windows-only**.  
If run on non-Windows hosts, the tool degrades gracefully and labels Windows-specific hardware fields as unavailable.

## Example output (human-readable)

```text
System Summary
==============
OS: Windows 11 (AMD64)
Platform: Windows-11-10.0.22631-SP0
CPU: 13th Gen Intel(R) Core(TM) i7-13700H
Logical cores: 20
Total RAM: 31.72 GB
Disk (root/system): 953.87 GB total, 614.42 GB free
Support: Windows host detected.
GPU(s):
  1. NVIDIA | NVIDIA GeForce RTX 4070 Laptop GPU | VRAM: 8.00 GB

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

- Hardware APIs and command availability vary across Windows versions/configurations.
- On non-Windows hosts, hardware fields are intentionally marked unavailable in this Windows-only build.
- Reported suitability is conservative and should not be treated as a guarantee.

## Privacy

All checks run locally on your machine. The script does not send hardware details to any external service.

## Troubleshooting

- If GPU fields are unavailable on Windows, `wmic` may be missing; this tool attempts a PowerShell/CIM fallback.
- If running in a VM, some hardware data may be hidden by the host.
- Use `--json` to integrate with automation and inspect raw detected fields.
