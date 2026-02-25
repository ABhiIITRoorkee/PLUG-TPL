#Plug-TPL/main.py


import argparse
import os
import sys
import time
import subprocess
from typing import List


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_int_list(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: str) -> None:
    log("RUN " + " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise SystemExit(f"Command failed (exit={result.returncode}) after {elapsed:.2f}s")
    log(f"DONE ({elapsed:.2f}s)")


def build_common_child_args(args) -> List[str]:
    # These are passed to train/test/metrics scripts so they all see the same config
    child = [
        "--tpl_range", str(args.tpl_range),
        "--training_data_path", args.training_data_path,
        "--testing_data_path", args.testing_data_path,
        "--function_feature_path", args.function_feature_path,
        "--output_path", args.output_path,
        "--n_heads", str(args.n_heads),
        "--d_k", str(args.d_k),
        "--d_v", str(args.d_v),
        "--d_q", str(args.d_q),
        "--continue_training", str(args.continue_training),
        "--train_batch_size", str(args.train_batch_size),
        "--epoch", str(args.epoch),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--device", args.device,
        "--stage", args.stage,
    ]
    return child


def run_train(project_root: str, py_exec: str, args, fold: int) -> None:
    train_dataset = f"training_{fold}.json"
    cmd = [py_exec, "train_model.py"] + build_common_child_args(args) + [
        "--train_dataset", train_dataset,
        "--fold", str(fold),  # harmless for train_model, useful for consistency
    ]
    log(f"TRAIN fold={fold} dataset={train_dataset}")
    run_cmd(cmd, cwd=project_root)


def run_test(project_root: str, py_exec: str, args, fold: int, rm: int) -> None:
    test_dataset = f"testing_{fold}_{rm}.json"
    cmd = [py_exec, "test_model.py"] + build_common_child_args(args) + [
        "--test_dataset", test_dataset,
        "--fold", str(fold),  # metrics uses these later, passing here is fine
        "--rm", str(rm),
    ]
    log(f"TEST fold={fold} rm={rm} dataset={test_dataset}")
    run_cmd(cmd, cwd=project_root)


def run_metrics(project_root: str, py_exec: str, args, fold: int, rm: int) -> None:
    cmd = [py_exec, "metrics_single_file.py"] + build_common_child_args(args) + [
        "--fold", str(fold),
        "--rm", str(rm),
    ]
    log(f"METRICS fold={fold} rm={rm}")
    run_cmd(cmd, cwd=project_root)


def main():
    p = argparse.ArgumentParser(description="Unified pipeline runner for Plug-TPL / AttenTPL")

    # Pipeline control
    p.add_argument("--mode", type=str, default="all",
                   choices=["train", "test", "metrics", "all"],
                   help="Which stage(s) to run")
    p.add_argument("--project_root", type=str, default=".",
                   help="Path to Plug-TPL project root (where train_model.py exists)")
    p.add_argument("--python_exec", type=str, default=sys.executable,
                   help="Python executable to use for child scripts")

    # Single fold/rm OR batch lists
    p.add_argument("--fold", type=int, default=0, help="Single fold")
    p.add_argument("--rm", type=int, default=3, help="Single rm")
    p.add_argument("--folds", type=str, default="", help="Comma-separated folds, e.g. 0,1,2,3,4")
    p.add_argument("--rms", type=str, default="", help="Comma-separated rms, e.g. 1,3,5")

    # Shared args from utility.parse_args.py
    p.add_argument("--tpl_range", type=int, default=763)
    p.add_argument("--training_data_path", type=str, default="./training data/")
    p.add_argument("--testing_data_path", type=str, default="./testing data/")
    p.add_argument("--function_feature_path", type=str, default="./data/function_feature_dict.pt")
    p.add_argument("--output_path", type=str, default="./output/")

    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--d_k", type=int, default=763)
    p.add_argument("--d_v", type=int, default=763)
    p.add_argument("--d_q", type=int, default=763)
    p.add_argument("--continue_training", type=int, default=0)
    p.add_argument("--train_batch_size", type=int, default=2048)
    p.add_argument("--epoch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--stage", type=str, default="C", choices=["A", "B", "C"])

    args = p.parse_args()

    project_root = os.path.abspath(args.project_root)
    if not os.path.exists(os.path.join(project_root, "train_model.py")):
        raise SystemExit(f"train_model.py not found in project_root: {project_root}")

    # Resolve fold list and rm list
    folds = parse_int_list(args.folds) if args.folds else [args.fold]
    rms = parse_int_list(args.rms) if args.rms else [args.rm]

    log(f"project_root={project_root}")
    log(f"mode={args.mode} folds={folds} rms={rms} device={args.device} stage={args.stage}")

    # TRAIN
    if args.mode in ("train", "all"):
        # In all mode, train once per fold (not per rm)
        for fold in folds:
            run_train(project_root, args.python_exec, args, fold)

    # TEST
    if args.mode in ("test", "all"):
        for fold in folds:
            for rm in rms:
                run_test(project_root, args.python_exec, args, fold, rm)

    # METRICS
    if args.mode in ("metrics", "all"):
        for fold in folds:
            for rm in rms:
                run_metrics(project_root, args.python_exec, args, fold, rm)

    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()