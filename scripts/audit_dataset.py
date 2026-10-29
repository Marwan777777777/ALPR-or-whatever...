#!/usr/bin/env python3
"""CLI utility to audit EALPR dataset, export JSON/Markdown reports, generate splits, and render QA visualizations."""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals for Arabic characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.data.audit import EALPRDatasetAuditor
from vehicle_intelligence.data.splitter import LeakageResistantSplitter
from vehicle_intelligence.data.visualizer import AnnotationVisualizer


def main():
    parser = argparse.ArgumentParser(description="Audit and validate Egyptian ALPR dataset.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master",
        help="Path to EALPR dataset root folder."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/audit_reports",
        help="Directory to save audit JSON and Markdown reports."
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="artifacts/visualizations",
        help="Directory to save visualization sample images."
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="artifacts/splits",
        help="Directory to save train/val/test splits."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of visual sample images to generate for QA."
    )

    args = parser.parse_args()
    logger = setup_logger("dataset_auditor")

    dataset_path = Path(args.dataset_root)
    logger.info(f"Starting audit on dataset root: {dataset_path}")

    # 1. Run audit
    auditor = EALPRDatasetAuditor(dataset_path)
    report = auditor.run_audit()

    json_path, md_path = auditor.save_reports(report, output_dir=args.output_dir)
    logger.info(f"Audit reports successfully generated:\n  - JSON: {json_path}\n  - Markdown: {md_path}")

    # 2. Generate visualization samples
    logger.info(f"Generating annotation QA visualizations in: {args.vis_dir}")
    vis_samples = AnnotationVisualizer.export_sample_visualizations(
        dataset_root=dataset_path,
        output_dir=args.vis_dir,
        num_samples=args.num_samples
    )
    logger.info(f"Saved {len(vis_samples)} visualization QA samples.")

    # 3. Generate leakage-resistant splits
    logger.info("Generating leakage-resistant, vehicle-grouped train/val/test splits...")
    splitter = LeakageResistantSplitter(
        dataset_root=dataset_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    manifest = splitter.generate_split()
    split_paths = splitter.save_split_files(manifest, output_dir=args.splits_dir)
    logger.info(f"Split files generated: {len(manifest.train_vehicles)} train, {len(manifest.val_vehicles)} val, {len(manifest.test_vehicles)} test vehicles.")
    logger.info(f"Leakage check passed: {manifest.leakage_check_passed}")

    # Print summary
    print("\n" + auditor.generate_markdown_report(report))


if __name__ == "__main__":
    main()
