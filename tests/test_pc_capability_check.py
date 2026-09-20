import unittest
from unittest import mock
import io

import pc_capability_check as pcc


class CapabilityHeuristicTests(unittest.TestCase):
    def test_gpu_tier_24gb_supports_13b_and_qlora(self):
        gpus = [{"vendor": "NVIDIA", "name": "RTX", "vram_bytes": 24 * 1024**3, "vram": "24.00 GB"}]
        result = pcc.estimate_llm_capability(32 * 1024**3, gpus)

        self.assertEqual(result["mode"], "GPU-accelerated")
        self.assertIn("7B/13B", result["inference"])
        self.assertIn("QLoRA", result["finetuning"])

    def test_cpu_only_low_ram_is_limited(self):
        result = pcc.estimate_llm_capability(8 * 1024**3, [])

        self.assertEqual(result["mode"], "CPU-only or unknown GPU VRAM")
        self.assertIn("tiny models", result["inference"])
        self.assertIn("not recommended", result["finetuning"])


class FormattingTests(unittest.TestCase):
    def test_human_report_includes_sections_and_unavailable_gpu(self):
        report = {
            "platform": {
                "system": "Linux",
                "release": "6.x",
                "version": "#1",
                "machine": "x86_64",
                "platform": "Linux-6.x-x86_64",
            },
            "cpu": {"model": "Test CPU", "logical_cores": 8},
            "memory": {"total_bytes": 16 * 1024**3, "total": "16.00 GB"},
            "disk": {"total_bytes": 100 * 1024**3, "free_bytes": 60 * 1024**3, "total": "100.00 GB", "free": "60.00 GB"},
            "gpu": {"detected": False, "count": 0, "gpus": []},
            "llm_capability": {
                "mode": "CPU-only or unknown GPU VRAM",
                "inference": "CPU inference mostly for 1B/3B-class models",
                "finetuning": "Local fine-tuning generally not recommended without a capable GPU",
                "recommendations": ["Use small models."],
                "caveats": ["Heuristic only."],
            },
        }

        text = pcc.format_human_report(report)

        self.assertIn("System Summary", text)
        self.assertIn("LLM Capability Estimate", text)
        self.assertIn("GPU(s): Unavailable or not detected", text)
        self.assertIn("Recommendations:", text)
        self.assertIn("Caveats:", text)


class WindowsGpuParsingTests(unittest.TestCase):
    def test_parse_wmic_video_controller_output_name_and_ram_columns(self):
        output = (
            "Name                               AdapterRAM\n"
            "NVIDIA GeForce RTX 4090           25757220864\n"
            "Intel(R) UHD Graphics             1073741824\n"
        )
        gpus = pcc.parse_wmic_video_controller_output(output)

        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(gpus[0]["vendor"], "NVIDIA")
        self.assertEqual(gpus[0]["vram_bytes"], 25757220864)
        self.assertEqual(gpus[1]["vendor"], "Intel")
        self.assertEqual(gpus[1]["vram_bytes"], 1073741824)

    @mock.patch("pc_capability_check.run_command")
    def test_detect_gpus_windows_falls_back_when_wmic_unavailable(self, mock_run_command):
        def fake_run(cmd, timeout=3):
            if cmd[:2] == ["wmic", "path"]:
                return ""
            if cmd and cmd[0] == "powershell":
                return '[{"Name":"NVIDIA GeForce RTX 4090","AdapterRAM":25757220864}]'
            return ""

        mock_run_command.side_effect = fake_run
        gpus = pcc.detect_gpus("Windows")

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["vendor"], "NVIDIA")
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(gpus[0]["vram_bytes"], 25757220864)


class LinuxGpuDetectionTests(unittest.TestCase):
    @mock.patch("pc_capability_check.run_command")
    def test_detect_linux_gpus_from_lspci(self, mock_run_command):
        def fake_run(cmd, timeout=3):
            if cmd == ["lspci"]:
                return "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics"
            if cmd and cmd[0] == "nvidia-smi":
                return ""
            return ""

        mock_run_command.side_effect = fake_run
        gpus = pcc.detect_gpus("Linux")

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["vendor"], "Intel")
        self.assertIn("UHD Graphics", gpus[0]["name"])

    @mock.patch("pc_capability_check.open", create=True)
    @mock.patch("pc_capability_check.glob.glob")
    def test_sysfs_fallback_ignores_connector_paths(self, mock_glob, mock_open):
        mock_glob.return_value = ["/sys/class/drm/card0", "/sys/class/drm/card0-DP-1"]

        file_data = {
            "/sys/class/drm/card0/device/class": "0x030000",
            "/sys/class/drm/card0/device/vendor": "0x10de",
            "/sys/class/drm/card0/device/uevent": "DRIVER=nvidia\n",
        }

        def fake_open(path, *args, **kwargs):
            if path in file_data:
                return mock.mock_open(read_data=file_data[path])()
            raise OSError("missing")

        mock_open.side_effect = fake_open
        gpus = pcc.detect_linux_gpus_sysfs()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["vendor"], "NVIDIA")
        self.assertEqual(gpus[0]["name"], "nvidia")

    @mock.patch("pc_capability_check.run_command")
    def test_nvidia_smi_merge_matches_by_name(self, mock_run_command):
        mock_run_command.return_value = (
            "NVIDIA GeForce RTX 3090, 24576\n"
            "NVIDIA GeForce RTX 4090, 24564\n"
        )
        existing = [
            {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 4090", "vram_bytes": None, "vram": pcc.UNAVAILABLE},
            {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 3090", "vram_bytes": None, "vram": pcc.UNAVAILABLE},
            {"vendor": "Intel", "name": "Intel UHD", "vram_bytes": None, "vram": pcc.UNAVAILABLE},
        ]

        merged = pcc.update_linux_vram_from_nvidia_smi(existing)

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(merged[0]["vram_bytes"], 24564 * 1024**2)
        self.assertEqual(merged[1]["name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(merged[1]["vram_bytes"], 24576 * 1024**2)
        self.assertEqual(merged[2]["vendor"], "Intel")

    @mock.patch("pc_capability_check.run_command")
    def test_nvidia_smi_merge_handles_lspci_style_name(self, mock_run_command):
        mock_run_command.return_value = "NVIDIA GeForce RTX 4090, 24564\n"
        existing = [
            {
                "vendor": "NVIDIA",
                "name": "VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)",
                "vram_bytes": None,
                "vram": pcc.UNAVAILABLE,
            }
        ]

        merged = pcc.update_linux_vram_from_nvidia_smi(existing)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(merged[0]["vram_bytes"], 24564 * 1024**2)

    @mock.patch("pc_capability_check.run_command")
    def test_nvidia_smi_merge_handles_duplicate_gpu_names(self, mock_run_command):
        mock_run_command.return_value = (
            "NVIDIA GeForce RTX 4090, 24564\n"
            "NVIDIA GeForce RTX 4090, 24564\n"
        )
        existing = [
            {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 4090", "vram_bytes": None, "vram": pcc.UNAVAILABLE},
            {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 4090", "vram_bytes": None, "vram": pcc.UNAVAILABLE},
        ]

        merged = pcc.update_linux_vram_from_nvidia_smi(existing)

        self.assertEqual(len(merged), 2)
        self.assertIsNotNone(merged[0]["vram_bytes"])
        self.assertIsNotNone(merged[1]["vram_bytes"])


class CliMainTests(unittest.TestCase):
    @mock.patch("pc_capability_check.collect_system_report")
    def test_main_outputs_human_readable_by_default(self, mock_collect):
        mock_collect.return_value = {
            "platform": {"system": "Linux", "release": "x", "version": "x", "machine": "x86_64", "platform": "Linux-x"},
            "cpu": {"model": "CPU", "logical_cores": 4},
            "memory": {"total": "16.00 GB", "total_bytes": 16 * 1024**3},
            "disk": {"total": "100.00 GB", "free": "50.00 GB", "total_bytes": 100 * 1024**3, "free_bytes": 50 * 1024**3},
            "gpu": {"detected": False, "count": 0, "gpus": []},
            "llm_capability": {
                "mode": "CPU-only or unknown GPU VRAM",
                "inference": "Limited local inference; focus on tiny models (<3B)",
                "finetuning": "Local fine-tuning generally not recommended without a capable GPU",
                "recommendations": ["Use small quantized models."],
                "caveats": ["Heuristic only."],
            },
        }
        with mock.patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
            exit_code = pcc.main([])
        self.assertEqual(exit_code, 0)
        self.assertIn("System Summary", fake_stdout.getvalue())

    @mock.patch("pc_capability_check.collect_system_report")
    def test_main_outputs_json_with_flag(self, mock_collect):
        mock_collect.return_value = {"platform": {"system": "Linux"}}
        with mock.patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
            exit_code = pcc.main(["--json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"platform"', fake_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
