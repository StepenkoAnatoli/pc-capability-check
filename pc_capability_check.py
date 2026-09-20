#!/usr/bin/env python3
"""Cross-platform PC capability checker for local LLM workloads."""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


UNAVAILABLE = "Unavailable"


def run_command(command: List[str], timeout: int = 3) -> str:
    """Run a command and return stdout text, or an empty string on failure."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def detect_cpu_model(system: str) -> str:
    model = platform.processor().strip()
    if model:
        return model

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        _, _, value = line.partition(":")
                        candidate = value.strip()
                        if candidate:
                            return candidate
        except OSError:
            pass
    elif system == "Darwin":
        model = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    elif system == "Windows":
        model = run_command(["wmic", "cpu", "get", "Name"])
        if model:
            lines = [line.strip() for line in model.splitlines() if line.strip() and line.strip().lower() != "name"]
            if lines:
                return lines[0]
        env_model = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if env_model:
            return env_model

    return UNAVAILABLE


def detect_total_ram_bytes(system: str) -> Optional[int]:
    if system == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (OSError, ValueError):
            return None

    if system == "Darwin":
        output = run_command(["sysctl", "-n", "hw.memsize"])
        if output.isdigit():
            return int(output)

    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            global_memory_status_ex = ctypes.windll.kernel32.GlobalMemoryStatusEx
            global_memory_status_ex.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
            global_memory_status_ex.restype = ctypes.c_int
            if global_memory_status_ex(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return None

    return None


def detect_disk_bytes(system: str) -> Dict[str, Optional[int]]:
    root_path = os.path.abspath(os.sep)
    if system == "Windows":
        system_drive = os.environ.get("SystemDrive", "C:").strip() or "C:"
        root_path = f"{system_drive}\\"
    try:
        usage = shutil.disk_usage(root_path)
        return {"total": int(usage.total), "free": int(usage.free)}
    except OSError:
        return {"total": None, "free": None}


def parse_bytes_from_text(value: str) -> Optional[int]:
    if not value:
        return None
    cleaned = value.replace(",", "").strip()
    match = re.search(r"(\d+)", cleaned)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def bytes_to_human(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return UNAVAILABLE
    if num_bytes < 0:
        return UNAVAILABLE
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.2f} {units[index]}"


def bytes_to_gib(num_bytes: Optional[int]) -> float:
    if num_bytes is None:
        return 0.0
    return num_bytes / (1024 ** 3)


def infer_vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "advanced micro devices" in lowered or "radeon" in lowered:
        return "AMD"
    if "intel" in lowered:
        return "Intel"
    return UNAVAILABLE


def normalize_gpu_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def normalize_gpu_match_key(name: str) -> str:
    text = normalize_gpu_name(name)
    if ":" in text and ("controller" in text or "display" in text):
        text = text.split(":", 1)[1].strip()
    text = text.replace("nvidia corporation", " ")
    text = text.replace("nvidia", " ")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"\[|\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_vendor_text(vendor_text: str) -> str:
    normalized = infer_vendor_from_name(vendor_text)
    if normalized != UNAVAILABLE:
        return normalized
    cleaned = vendor_text.split("(", 1)[0].strip()
    return cleaned or UNAVAILABLE


def parse_wmic_video_controller_output(output: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if not output:
        return gpus

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return gpus

    header = lines[0]
    lowered_header = header.lower()
    if "name" not in lowered_header or "adapterram" not in lowered_header:
        return gpus

    name_idx = lowered_header.find("name")
    ram_idx = lowered_header.find("adapterram")
    if name_idx < 0 or ram_idx < 0:
        return gpus

    for line in lines[1:]:
        if line.lower().strip() in {"name", "adapterram"}:
            continue
        match = re.match(r"^(.*?)(\d[\d,]*)\s*$", line.strip())
        if match:
            name_text = match.group(1).strip()
            ram_text = match.group(2).strip()
        elif ram_idx > name_idx:
            name_text = line[name_idx:ram_idx].strip()
            ram_text = line[ram_idx:].strip()
        else:
            ram_text = line[ram_idx:name_idx].strip()
            name_text = line[name_idx:].strip()
        if not name_text and not ram_text:
            continue
        ram_bytes = parse_bytes_from_text(ram_text)
        name = name_text or UNAVAILABLE
        gpus.append(
            {
                "vendor": infer_vendor_from_name(name),
                "name": name,
                "vram_bytes": ram_bytes,
                "vram": bytes_to_human(ram_bytes),
            }
        )
    return gpus


def parse_powershell_video_controller_output(output: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if not output:
        return gpus
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return gpus

    entries: List[Dict[str, Any]]
    if isinstance(payload, dict):
        entries = [payload]
    elif isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, dict)]
    else:
        return gpus

    for item in entries:
        name = str(item.get("Name", "")).strip() or UNAVAILABLE
        ram_value = item.get("AdapterRAM")
        ram_bytes = parse_bytes_from_text(str(ram_value)) if ram_value is not None else None
        gpus.append(
            {
                "vendor": infer_vendor_from_name(name),
                "name": name,
                "vram_bytes": ram_bytes,
                "vram": bytes_to_human(ram_bytes),
            }
        )
    return gpus


def detect_linux_gpus_sysfs() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    for card_path in sorted(glob.glob("/sys/class/drm/card*")):
        if not re.fullmatch(r"card\d+", os.path.basename(card_path)):
            continue
        device_path = os.path.join(card_path, "device")
        vendor_path = os.path.join(device_path, "vendor")
        class_path = os.path.join(device_path, "class")
        uevent_path = os.path.join(device_path, "uevent")

        try:
            with open(class_path, "r", encoding="utf-8", errors="ignore") as handle:
                class_code = handle.read().strip().lower()
        except OSError:
            continue

        if not class_code.startswith("0x03"):
            continue

        vendor = UNAVAILABLE
        try:
            with open(vendor_path, "r", encoding="utf-8", errors="ignore") as handle:
                vendor_id = handle.read().strip().lower()
            vendor = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x1022": "AMD", "0x8086": "Intel"}.get(vendor_id, UNAVAILABLE)
        except OSError:
            pass

        name = UNAVAILABLE
        try:
            with open(uevent_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("DRIVER="):
                        name = line.split("=", 1)[1].strip() or UNAVAILABLE
                        break
        except OSError:
            pass

        gpus.append({"vendor": vendor, "name": name, "vram_bytes": None, "vram": UNAVAILABLE})

    return gpus


def update_linux_vram_from_nvidia_smi(gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not output:
        return gpus

    detected: List[Dict[str, Any]] = []
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) != 2:
            continue
        name = parts[0] or UNAVAILABLE
        try:
            vram_mb = float(parts[1])
            vram_bytes = int(vram_mb * (1024 ** 2))
        except ValueError:
            vram_bytes = None
        detected.append(
            {
                "vendor": infer_vendor_from_name(name),
                "name": name,
                "vram_bytes": vram_bytes,
                "vram": bytes_to_human(vram_bytes),
            }
        )

    if not detected:
        return gpus

    merged = list(gpus)
    used_detected: List[bool] = [False] * len(detected)
    detected_by_name: Dict[str, List[int]] = {}
    for idx, item in enumerate(detected):
        key = normalize_gpu_name(item.get("name", ""))
        if not key:
            continue
        detected_by_name.setdefault(key, []).append(idx)

    for gpu in merged:
        if gpu.get("vendor") != "NVIDIA":
            continue
        name_key = normalize_gpu_name(str(gpu.get("name", "")))
        index_pool = detected_by_name.get(name_key, [])
        while index_pool and used_detected[index_pool[0]]:
            index_pool.pop(0)
        if not index_pool:
            continue
        matched_index = index_pool.pop(0)
        detected_gpu = detected[matched_index]
        gpu["name"] = detected_gpu["name"]
        gpu["vram_bytes"] = detected_gpu["vram_bytes"]
        gpu["vram"] = detected_gpu["vram"]
        used_detected[matched_index] = True

    unmatched_detected = [i for i, used in enumerate(used_detected) if not used]
    unmatched_merged = [
        i for i, gpu in enumerate(merged)
        if gpu.get("vendor") == "NVIDIA" and gpu.get("vram_bytes") is None
    ]

    if len(unmatched_detected) == 1 and len(unmatched_merged) == 1:
        detected_gpu = detected[unmatched_detected[0]]
        target_gpu = merged[unmatched_merged[0]]
        target_gpu["name"] = detected_gpu["name"]
        target_gpu["vram_bytes"] = detected_gpu["vram_bytes"]
        target_gpu["vram"] = detected_gpu["vram"]
        used_detected[unmatched_detected[0]] = True
    else:
        merged_key_map = {
            normalize_gpu_match_key(str(merged[index].get("name", ""))): index
            for index in unmatched_merged
        }
        for index in unmatched_detected:
            detected_gpu = detected[index]
            key = normalize_gpu_match_key(str(detected_gpu.get("name", "")))
            target_index = merged_key_map.get(key)
            if target_index is None:
                continue
            merged[target_index]["name"] = detected_gpu["name"]
            merged[target_index]["vram_bytes"] = detected_gpu["vram_bytes"]
            merged[target_index]["vram"] = detected_gpu["vram"]
            used_detected[index] = True

    for index, detected_gpu in enumerate(detected):
        if not used_detected[index]:
            merged.append(detected_gpu)

    return merged


def detect_gpus(system: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []

    if system == "Windows":
        output = run_command(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"])
        gpus.extend(parse_wmic_video_controller_output(output))
        if not gpus:
            ps_output = run_command(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
                ]
            )
            gpus.extend(parse_powershell_video_controller_output(ps_output))

    elif system == "Darwin":
        output = run_command(["system_profiler", "SPDisplaysDataType"])
        if output:
            blocks = output.split("\n\n")
            for block in blocks:
                name = UNAVAILABLE
                vendor = UNAVAILABLE
                vram_bytes: Optional[int] = None
                for raw_line in block.splitlines():
                    line = raw_line.strip()
                    if line.startswith("Chipset Model:"):
                        name = line.split(":", 1)[1].strip() or UNAVAILABLE
                    elif line.startswith("Vendor:"):
                        vendor_text = line.split(":", 1)[1].strip()
                        vendor = normalize_vendor_text(vendor_text)
                    elif line.startswith("VRAM"):
                        size_text = line.split(":", 1)[1].strip() if ":" in line else ""
                        match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", size_text, re.IGNORECASE)
                        if match:
                            amount = float(match.group(1))
                            unit = match.group(2).upper()
                            if unit == "GB":
                                vram_bytes = int(amount * (1024 ** 3))
                            else:
                                vram_bytes = int(amount * (1024 ** 2))
                if name != UNAVAILABLE or vendor != UNAVAILABLE:
                    gpus.append({
                        "vendor": vendor,
                        "name": name,
                        "vram_bytes": vram_bytes,
                        "vram": bytes_to_human(vram_bytes),
                    })

    elif system == "Linux":
        output = run_command(["lspci"])
        if output:
            for line in output.splitlines():
                text = line.strip()
                lowered = text.lower()
                if "vga" not in lowered and "3d controller" not in lowered and "display controller" not in lowered:
                    continue
                details = text.split(":", 2)[-1].strip()
                vendor = infer_vendor_from_name(details)
                gpus.append({
                    "vendor": vendor,
                    "name": details or UNAVAILABLE,
                    "vram_bytes": None,
                    "vram": UNAVAILABLE,
                })
        if not gpus:
            gpus.extend(detect_linux_gpus_sysfs())
        gpus = update_linux_vram_from_nvidia_smi(gpus)

    return gpus


def estimate_llm_capability(total_ram_bytes: Optional[int], gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_ram_gib = bytes_to_gib(total_ram_bytes)
    gpu_vram_gib = [bytes_to_gib(gpu.get("vram_bytes")) for gpu in gpus if gpu.get("vram_bytes") is not None]
    max_gpu_vram_gib = max(gpu_vram_gib) if gpu_vram_gib else 0.0

    recommendations: List[str] = []

    if max_gpu_vram_gib >= 48:
        mode = "GPU-accelerated"
        inference = "Strong local inference: 13B/34B comfortable; 70B possible with quantization/offload"
        finetuning = "QLoRA: 13B practical; 7B comfortable. Full fine-tuning remains expensive"
        recommendations.extend(
            [
                "Use 4-bit or 8-bit quantization for larger models.",
                "Consider 70B only with careful quantization, batch size control, and patience.",
            ]
        )
    elif max_gpu_vram_gib >= 24:
        mode = "GPU-accelerated"
        inference = "Good local inference: 7B/13B comfortable; larger models require aggressive quantization"
        finetuning = "QLoRA: 7B practical; 13B possible with tight settings"
        recommendations.extend(
            [
                "Prefer 7B/13B instruction-tuned models for reliable latency.",
                "Use gradient checkpointing and small batch sizes for QLoRA.",
            ]
        )
    elif max_gpu_vram_gib >= 12:
        mode = "GPU-accelerated"
        inference = "Entry-to-mid GPU inference: 3B/7B strong; 13B possible with 4-bit quantization"
        finetuning = "QLoRA: mostly 3B/7B with conservative settings"
        recommendations.extend(
            [
                "Use 4-bit quantization and smaller context windows for stability.",
                "Target 3B/7B models for best usability.",
            ]
        )
    elif max_gpu_vram_gib >= 6:
        mode = "GPU-limited"
        inference = "Small-model GPU inference: up to 3B/7B (quantized)"
        finetuning = "Fine-tuning is limited; tiny QLoRA experiments only"
        recommendations.extend(
            [
                "Focus on 3B models or heavily-quantized 7B models.",
                "Expect trade-offs in context length and response speed.",
            ]
        )
    else:
        mode = "CPU-only or unknown GPU VRAM"
        if total_ram_gib >= 64:
            inference = "CPU inference feasible for 7B; 13B may work slowly with quantization"
        elif total_ram_gib >= 32:
            inference = "CPU inference suitable for 3B/7B with quantization; expect slow throughput"
        elif total_ram_gib >= 16:
            inference = "CPU inference mostly for 1B/3B-class models"
        else:
            inference = "Limited local inference; focus on tiny models (<3B)"
        finetuning = "Local fine-tuning generally not recommended without a capable GPU"
        recommendations.extend(
            [
                "If possible, add or upgrade a discrete GPU for significantly better results.",
                "For CPU-only setups, prioritize smaller quantized models.",
            ]
        )

    caveats = [
        "These are conservative estimates, not guarantees.",
        "Actual usability depends on model architecture, quantization, context length, and software stack.",
        "Integrated GPUs and shared memory can vary significantly in real performance.",
    ]

    return {
        "mode": mode,
        "inference": inference,
        "finetuning": finetuning,
        "recommended_model_classes": ["3B", "7B", "13B", "70B"],
        "max_detected_gpu_vram_gib": round(max_gpu_vram_gib, 2),
        "total_ram_gib": round(total_ram_gib, 2) if total_ram_bytes is not None else None,
        "recommendations": recommendations,
        "caveats": caveats,
    }


def collect_system_report() -> Dict[str, Any]:
    system = platform.system() or UNAVAILABLE
    disk = detect_disk_bytes(system)
    gpus = detect_gpus(system)

    report = {
        "platform": {
            "system": system,
            "release": platform.release() or UNAVAILABLE,
            "version": platform.version() or UNAVAILABLE,
            "machine": platform.machine() or UNAVAILABLE,
            "platform": platform.platform() or UNAVAILABLE,
        },
        "cpu": {
            "model": detect_cpu_model(system),
            "logical_cores": os.cpu_count() if os.cpu_count() is not None else UNAVAILABLE,
        },
        "memory": {
            "total_bytes": detect_total_ram_bytes(system),
        },
        "disk": {
            "total_bytes": disk["total"],
            "free_bytes": disk["free"],
        },
        "gpu": {
            "detected": bool(gpus),
            "count": len(gpus),
            "gpus": gpus,
        },
    }

    report["memory"]["total"] = bytes_to_human(report["memory"]["total_bytes"])
    report["disk"]["total"] = bytes_to_human(report["disk"]["total_bytes"])
    report["disk"]["free"] = bytes_to_human(report["disk"]["free_bytes"])
    report["llm_capability"] = estimate_llm_capability(report["memory"]["total_bytes"], gpus)

    return report


def format_human_report(report: Dict[str, Any]) -> str:
    platform_info = report["platform"]
    cpu = report["cpu"]
    memory = report["memory"]
    disk = report["disk"]
    gpu = report["gpu"]
    llm = report["llm_capability"]

    lines = [
        "System Summary",
        "==============",
        f"OS: {platform_info['system']} {platform_info['release']} ({platform_info['machine']})",
        f"Platform: {platform_info['platform']}",
        f"CPU: {cpu['model']}",
        f"Logical cores: {cpu['logical_cores']}",
        f"Total RAM: {memory['total']}",
        f"Disk (root/system): {disk['total']} total, {disk['free']} free",
    ]

    if gpu["detected"]:
        lines.append("GPU(s):")
        for index, item in enumerate(gpu["gpus"], start=1):
            lines.append(
                f"  {index}. {item.get('vendor', UNAVAILABLE)} | {item.get('name', UNAVAILABLE)} | VRAM: {item.get('vram', UNAVAILABLE)}"
            )
    else:
        lines.append("GPU(s): Unavailable or not detected")

    lines.extend(
        [
            "",
            "LLM Capability Estimate",
            "=======================",
            f"Mode: {llm['mode']}",
            f"Inference: {llm['inference']}",
            f"Fine-tuning/QLoRA: {llm['finetuning']}",
            "",
            "Recommendations:",
        ]
    )

    for recommendation in llm["recommendations"]:
        lines.append(f"- {recommendation}")

    lines.append("")
    lines.append("Caveats:")
    for caveat in llm["caveats"]:
        lines.append(f"- {caveat}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report hardware capabilities and estimate local LLM inference/fine-tuning suitability "
            "(conservative heuristic guidance)."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = collect_system_report()
    except Exception as exc:  # pragma: no cover
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_human_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
