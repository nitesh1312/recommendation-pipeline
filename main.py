"""
Unified Main CLI Entrypoint for Recommendation System Pipeline.
Usage:
    python main.py --all
    python main.py --stage ingestion
    python main.py --stage validation
    python main.py --stage training
"""

import argparse
import sys
from src.orchestration.runner import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(description="End-to-End Recommendation Data Pipeline Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full end-to-end pipeline from ingestion to evaluation"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "data_ingestion",
            "data_validation",
            "data_preparation",
            "data_transformation",
            "feature_engineering",
            "model_training",
            "model_evaluation"
        ],
        help="Run a specific individual stage"
    )

    args = parser.parse_args()
    orchestrator = PipelineOrchestrator(config_path=args.config)

    if args.stage:
        res = orchestrator.run_stage(args.stage)
        print(f"\nStage Result: {res}")
        if res.get("status") == "FAILED":
            sys.exit(1)
    else:
        # Default behavior is to run all stages end-to-end
        summary = orchestrator.run_all()
        print("\n================ PIPELINE EXECUTION SUMMARY ================")
        print(f"Status: {summary['status']}")
        print(f"Total Duration: {summary['total_duration_seconds']}s")
        for stage, data in summary["stages"].items():
            print(f"  • {stage:22s}: {data['status']} ({data['duration_seconds']}s)")
        print("============================================================\n")
        if summary.get("status") == "FAILED":
            sys.exit(1)


if __name__ == "__main__":
    main()
