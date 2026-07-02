from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from consumer_pipeline.config import load_config
from consumer_pipeline.debias import PreviousMonthCorrector
from consumer_pipeline.information import InformationEnvironmentBuilder
from consumer_pipeline.orchestrator import ConsumerPipeline


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_all_regions_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            for region in ("jp", "eu", "us"):
                config = load_config(ROOT / "configs" / f"{region}.yaml")
                config["population"]["size"] = 120
                config["output"]["directory"] = temporary_directory
                result, output_path = ConsumerPipeline(config).run("2026-06", "2026-06-20")
                self.assertEqual(result.region, region.upper())
                self.assertEqual(result.population_size, 120)
                self.assertTrue(0 <= result.corrected_score <= 200)
                self.assertIn("news_items", result.environment)
                self.assertIn("indicators", result.environment)
                self.assertTrue((output_path / "result.json").exists())
                self.assertTrue((output_path / "population_responses.csv").exists())

    def test_information_excludes_future_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            news = root / "news.jsonl"
            indicators = root / "indicators.csv"
            news.write_text(
                '{"published_at":"2026-06-01","title":"known","sentiment":0.5}\n'
                '{"published_at":"2026-07-01","title":"future","sentiment":1.0}\n',
                encoding="utf-8",
            )
            indicators.write_text(
                "observed_at,name,z_score,weight\n"
                "2026-06-01,known,0.5,1\n"
                "2026-07-01,future,3.0,1\n",
                encoding="utf-8",
            )
            environment = InformationEnvironmentBuilder().build(
                "2026-06-20", news, indicators, news_weight=0.5, indicator_weight=0.5
            )
            self.assertEqual(len(environment.news_items), 1)
            self.assertEqual(len(environment.indicators), 1)
            self.assertEqual(environment.news_items[0].title, "known")

    def test_correction_uses_exact_previous_month(self) -> None:
        history = ROOT / "examples" / "us" / "history.csv"
        corrected, detail = PreviousMonthCorrector(0.5, 10.0).apply(100.0, "2026-06", history)
        self.assertEqual(detail.previous_month, "2026-05")
        self.assertEqual(detail.residual, 3.0)
        self.assertEqual(corrected, 101.5)

    def test_literal_credential_is_rejected(self) -> None:
        config = load_config(ROOT / "configs" / "us.yaml")
        config["credentials"]["model_api_key"] = "not-a-variable-name"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            import yaml

            serializable = {key: value for key, value in config.items() if key != "_config_dir"}
            yaml.safe_dump(serializable, handle)
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "environment variable name"):
                load_config(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
