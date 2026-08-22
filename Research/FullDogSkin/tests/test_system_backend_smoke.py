import os
import subprocess
import sys
import unittest
from pathlib import Path


class BackendSystemSmokeTests(unittest.TestCase):
    def test_existing_backend_smoke_script_passes(self):
        project_dir = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.setdefault("ENABLE_CONTENT_VALIDATOR", "0")

        result = subprocess.run(
            [sys.executable, "test_backend.py"],
            cwd=project_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(result.returncode, 0, combined_output)
        self.assertIn("ALL PYTHON TESTS PASSED", combined_output)


if __name__ == "__main__":
    unittest.main()
