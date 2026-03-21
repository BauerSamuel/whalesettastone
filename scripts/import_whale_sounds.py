"""
Import and process whale sound WAV files (solo/group directories).

Path is configurable via env WHALE_SOUNDS_DIR or CLI:
    python scripts/import_whale_sounds.py [path]
If no path is given, runs the workflow on existing data/whale_codas.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import logging
import os

from src.config import BASE_DIR
from src.workflow import Workflow

RESULTS_DIR = BASE_DIR / "analysis_results"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Path: env > first CLI arg > None (use existing whale_codas)
    base_dir = os.environ.get("WHALE_SOUNDS_DIR")
    if not base_dir and len(sys.argv) > 1:
        base_dir = sys.argv[1]
    base_dir = Path(base_dir) if base_dir else None

    workflow = Workflow()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    solo_files = []
    group_files = []
    solo_success = False
    group_success = False

    if base_dir and base_dir.exists():
        solo_dir = base_dir / "solo_recordings"
        group_dir = base_dir / "group_recordings"
        logging.info("Looking for WAV files in: %s", base_dir)

        if solo_dir.exists():
            solo_files = list(solo_dir.glob("*.wav"))
            logging.info("Processing solo recordings (%d files)...", len(solo_files))
            solo_success = workflow.run_workflow(
                source_path=solo_dir,
                is_directory=True,
                visualize=True,
            )
        if group_dir.exists():
            group_files = list(group_dir.glob("*.wav"))
            logging.info("Processing group recordings (%d files)...", len(group_files))
            group_success = workflow.run_workflow(
                source_path=group_dir,
                is_directory=True,
                visualize=True,
            )
    else:
        if base_dir:
            logging.warning("Path not found: %s. Using existing data/whale_codas.", base_dir)
        logging.info("Running workflow on existing data/whale_codas...")
        solo_success = workflow.run_workflow(visualize=True)

    report = {
        "total_files": len(solo_files) + len(group_files),
        "solo_recordings": len(solo_files),
        "group_recordings": len(group_files),
        "solo_processing_success": solo_success,
        "group_processing_success": group_success,
    }
    report_file = results_dir / "processing_summary.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Summary written to %s. Plots in data/plots/", report_file)


if __name__ == "__main__":
    main()
