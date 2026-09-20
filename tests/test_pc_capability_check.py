import unittest

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


if __name__ == "__main__":
    unittest.main()
