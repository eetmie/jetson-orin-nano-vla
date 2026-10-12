"""Exercise clock verification without touching real host power settings."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class HostClockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = Path(__file__).resolve().parents[1] / "scripts/jetson-perf.sh"
        self.script = self.root / "jetson-perf.sh"
        self.script.write_text(source.read_text().replace("/sys/", str(self.root / "sys") + "/"))
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.command("nvpmodel", "echo 'NV Power Mode: MAXN_SUPER'; echo 2")
        self.command("jetson_clocks", "[ \"$1\" = --show ] || exit 99; echo 'EMC MinFreq=3199000000 MaxFreq=3199000000 CurrentFreq=3199000000'")
        self.write("devices/system/cpu/online", "0-5")
        for cpu in range(6):
            for bound in ("min", "max"):
                self.write(f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_{bound}_freq", "1728000")
        for bound in ("min", "max"):
            self.write(f"class/devfreq/17000000.gpu/{bound}_freq", "1020000000")
        self.write("devices/platform/gpu.0/tpc_pg_mask", "240")
        self.write("devices/platform/gpu.0/fbp_pg_mask", "2")

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def write(self, name, value):
        path = self.root / "sys" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    def verify(self):
        return subprocess.run(["bash", str(self.script), "--verify"], text=True,
            capture_output=True, env={**os.environ, "PATH": str(self.bin) + ":" + os.environ["PATH"]})

    def test_all_clocks_pass_without_setting_them(self):
        self.command("nvpmodel", "[ \"$1\" = -q ] || exit 99; echo 'NV Power Mode: MAXN_SUPER'; echo 2")
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cpu5 1728-1728", result.stdout)
        self.assertIn("clocks pinned", result.stdout)

    def test_unpinned_last_cpu_is_detected(self):
        self.write("devices/system/cpu/cpu5/cpufreq/scaling_min_freq", "729600")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cpu5 is not pinned", result.stderr)

    def test_wrong_power_mode_is_detected(self):
        self.command("nvpmodel", "echo 'NV Power Mode: 25W'; echo 1")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MAXN_SUPER was not selected", result.stderr)

    def test_saved_mode_does_not_hide_failed_tpc_mask(self):
        self.write("devices/platform/gpu.0/tpc_pg_mask", "0")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tpc_pg_mask=0", result.stderr)

    def test_failed_fbp_mask_is_detected(self):
        self.write("devices/platform/gpu.0/fbp_pg_mask", "0")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fbp_pg_mask=0", result.stderr)

    def test_unpinned_emc_is_detected(self):
        self.command("jetson_clocks", "echo 'EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=2133000000'")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EMC is not pinned", result.stderr)

    def test_missing_gpu_is_detected_without_waiting(self):
        gpu = self.root / "sys/class/devfreq/17000000.gpu"
        for path in gpu.iterdir(): path.unlink()
        gpu.rmdir()
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU did not initialise", result.stderr)
