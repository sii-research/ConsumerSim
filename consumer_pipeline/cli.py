from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import load_config
from .orchestrator import ConsumerPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JP/EU/US ConsumerSim monthly pipeline")
    parser.add_argument("--config", required=True, help="Path to a region YAML config")
    parser.add_argument("--month", required=True, help="Target month in YYYY-MM")
    parser.add_argument("--as-of", required=True, help="Latest allowed information date in YYYY-MM-DD")
    args = parser.parse_args()

    result, output_path = ConsumerPipeline(load_config(args.config)).run(args.month, args.as_of)
    print(json.dumps({"output": str(output_path), "result": asdict(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
