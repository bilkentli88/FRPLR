from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryReproductionTests(unittest.TestCase):
    def test_raw_row_counts(self):
        expected = {
            "data/qwen/v060_mechanism_4k8k.jsonl": 256,
            "data/qwen/v061_factorial_4k8k.jsonl": 512,
            "data/qwen/v062_geometry_localization_4k8k.jsonl": 768,
            "data/gemma/v071_gemma_factorial_4k8k.jsonl": 512,
            "data/gemma/v072_gemma_geometry_localization_4k8k.jsonl": 768,
            "data/qwen/qwen_harmonized_robustness_v1.jsonl": 256,
            "data/gemma/gemma_harmonized_robustness_v1.jsonl": 256,
            "data/phi/phi_factorial_confirmatory_v1.jsonl": 512,
            "data/mistral/mistral_factorial_confirmatory_v1.jsonl": 512,
        }
        for relative_path, count in expected.items():
            rows = sum(
                bool(line.strip())
                for line in (ROOT / relative_path)
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(rows, count, relative_path)

    def test_reported_results_reproduce(self):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "analysis/reproduce_reported_results.py"),
                "--repo-root",
                str(ROOT),
            ],
            check=True,
        )

    def test_repository_excludes_development_artifacts(self):
        names = {path.name for path in ROOT.rglob("*") if path.is_file()}
        excluded = {
            "v050_fragmentation_scan.jsonl",
            "v070_gemma_fragmentation_scan.jsonl",
            "v070_gemma_engineering_audit.json",
            "quality_natural_validation_manifest_v1.json",
            "STAGE3_QUALITY_PROTOCOL.md",
        }
        self.assertTrue(names.isdisjoint(excluded))


if __name__ == "__main__":
    unittest.main()
