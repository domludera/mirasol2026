"""
5-fold cross-validation: train + test each (model, fold) via main_moody.py.

Experiments are named "{model}_{task}_fold{k}" (e.g. tinyvit_3class_fold0).
Folds that already have predictions in outputs/preds_targets/ are skipped
unless --force is given.

Usage:
    python scripts/run_5fold.py all
    python scripts/run_5fold.py tinyvit htsat
    python scripts/run_5fold.py mobilenetv3 resnet18 --folds 0 1
    python scripts/run_5fold.py all --task binary
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.factory import SUPPORTED_MODELS
from scripts.experiment_utils import TASKS, predictions_exist, train_and_test

FOLDS = [0, 1, 2, 3, 4]


def run_model(model_name: str, task: str, folds: list, force: bool) -> None:
    t_model = time.time()
    print(f"\n{'#' * 60}\n# MODEL: {model_name} ({task})\n{'#' * 60}")

    for fold in folds:
        exp_name = f"{model_name}_{task}_fold{fold}"
        if not force and predictions_exist(exp_name):
            print(f"\n  SKIP: {exp_name} already has predictions")
            continue

        print(f"\n{'=' * 60}\nFOLD {fold} - Experiment: {exp_name}\n{'=' * 60}")
        t_fold = time.time()
        status = train_and_test(exp_name, model_type=model_name, esc_fold=fold, **TASKS[task])
        print(f"\n  Fold {fold}: {status} ({(time.time() - t_fold) / 60:.1f} min)")

    print(f"\n# {model_name} ({task}) completed in {(time.time() - t_model) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Run k-fold CV training and testing.")
    parser.add_argument("models", nargs="+", choices=SUPPORTED_MODELS + ("all",),
                        help="Models to run, or 'all'")
    parser.add_argument("--task", choices=sorted(TASKS), default="3class")
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="Re-run folds that already have predictions")
    args = parser.parse_args()

    models = list(SUPPORTED_MODELS) if "all" in args.models else args.models

    print("=" * 60)
    print(f"{len(args.folds)}-Fold Cross-Validation ({args.task})")
    print(f"Models:  {', '.join(models)}")
    print(f"Folds:   {args.folds}")
    print(f"Dataset: {TASKS[args.task]['dataset_npy_name']}")
    print("=" * 60)

    t_start = time.time()
    for model_name in models:
        run_model(model_name, args.task, args.folds, args.force)
    print(f"\n\nAll done! Total time: {(time.time() - t_start) / 3600:.1f} hours")


if __name__ == "__main__":
    main()
