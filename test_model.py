# Plug-TPL/test_model.py
import os
import json
import time
import re
from typing import Dict, List

import torch
import model_Atten_TPL
from utility.parse_args import arg_parse

args = arg_parse()

# -----------------------------
# Simple logger (START/HEARTBEAT/DONE)
# -----------------------------
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Load function features once
app_functions = torch.load(args.function_feature_path)

# Infer feature dim for safe fallback
try:
    _any_k = next(iter(app_functions.keys()))
    FEATURE_DIM = int(app_functions[_any_k].shape[0])
except Exception:
    FEATURE_DIM = 512  # fallback


def encode_test_data(data: Dict) -> Dict:
    tpl_list = data.get("tpl_list", [])
    if not tpl_list:
        tpl_list = [0]

    batch_function_feature = []
    batch_target_tpl = []
    batch_tpl_context = []
    batch_context_len = []

    tpl_context = tpl_list
    app_id = data["app_id"]

    # robust fetch (in case an app_id is missing in feature dict)
    ff = app_functions.get(app_id, None)
    if ff is None:
        # fallback zero vector if missing
        ff = torch.zeros((FEATURE_DIM,), dtype=torch.float32)
    function_feature = ff.to(args.device)

    context_len = len(tpl_list)

    for i in range(args.tpl_range):
        target_tpl = i + 1
        if target_tpl not in tpl_list:
            batch_target_tpl.append(target_tpl)
            batch_function_feature.append(function_feature)
            batch_tpl_context.append(tpl_context)
            batch_context_len.append(context_len)

    input_data = {
        "tpl_context": torch.tensor(batch_tpl_context).to(args.device),
        "function": torch.stack(batch_function_feature).to(args.device),
        "target_tpl": torch.tensor(batch_target_tpl).to(args.device),
        "context_len": torch.tensor(batch_context_len, dtype=torch.float32).to(args.device),
    }
    return input_data


def get_top_n_tpl(probability_list, top_n) -> List:
    p_list = sorted(probability_list, reverse=True)
    return [p_list[i][1] for i in range(min(top_n, len(p_list)))]


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def test_model(model_path: str, test_dataset: str, output_file: str) -> None:
    recommend_file = os.path.join(args.output_path, output_file)
    test_path = os.path.join(args.testing_data_path, test_dataset)

    os.makedirs(args.output_path, exist_ok=True)

    # Count lines for progress reporting
    total = _count_lines(test_path)

    log("TEST START")
    log(f"PID={os.getpid()}  CWD={os.getcwd()}")
    log(f"device={args.device}  tpl_range={args.tpl_range}  feature_dim={FEATURE_DIM}")
    log(f"function_feature_path={args.function_feature_path}")
    log(f"test_dataset={test_path}")
    log(f"model_path={model_path}")
    log(f"output_file={recommend_file}")
    t0 = time.time()

    # Load model
    model = model_Atten_TPL.AttenTPL()
    # safer load for CPU/GPU portability
    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state)
    model = model.to(args.device)
    model.eval()

    # Heartbeat controls
    heartbeat_every_n = 50          # print every N apps
    heartbeat_every_sec = 10.0      # also print every ~10 seconds
    last_hb_t = time.time()

    processed = 0

    with open(recommend_file, "w", encoding="utf-8") as write_recommend_fp, \
         open(test_path, "r", encoding="utf-8") as test_fp, \
         torch.no_grad():

        for line in test_fp:
            line = line.strip()
            if not line:
                continue

            test_obj = json.loads(line)
            inputs = encode_test_data(test_obj)

            outputs = model(inputs)
            outputs = outputs.view(-1).tolist()

            probability_list = []
            tpl_list = test_obj["tpl_list"]

            num = 0
            for i in range(args.tpl_range):
                target_tpl = i + 1
                if target_tpl not in tpl_list:
                    probability_list.append((outputs[num], target_tpl))
                    num += 1

            top_n_tpl = get_top_n_tpl(probability_list, 20)

            write_data = {
                "app_id": test_obj["app_id"],
                "removed_tpls": test_obj["removed_tpl_list"],
                "recommend_tpls": top_n_tpl,
                "tpl_list": test_obj["tpl_list"],
            }
            write_recommend_fp.write(json.dumps(write_data) + "\n")

            processed += 1

            # HEARTBEAT
            now = time.time()
            if (processed % heartbeat_every_n == 0) or ((now - last_hb_t) >= heartbeat_every_sec):
                elapsed = now - t0
                rate = processed / max(elapsed, 1e-9)
                pct = (processed / total * 100.0) if total > 0 else 0.0
                eta = (total - processed) / max(rate, 1e-9) if total > 0 else 0.0
                log(f"HEARTBEAT progress={processed}/{total} ({pct:.2f}%)  "
                    f"rate={rate:.2f} apps/s  elapsed={elapsed:.1f}s  eta={eta:.1f}s")
                last_hb_t = now

    elapsed = time.time() - t0
    log(f"TEST DONE processed={processed}/{total}  elapsed={elapsed:.2f}s")
    log(f"Saved recommendations -> {recommend_file}")


if __name__ == "__main__":
    model_dir = "model_Atten_TPL"
    test_dataset = args.test_dataset

    pattern = r"testing_(\d+)_(\d+)\.json"
    matches = re.match(pattern, test_dataset)
    if matches:
        fold = matches.group(1)
        rm = matches.group(2)
    else:
        # fallback if naming differs
        fold = "0"
        rm = "X"

    output_file = f"testing_Atten_TPL_{fold}_{rm}.json"
    model_ckpt = os.path.join(model_dir, f"model_{fold}.pth")

    test_model(model_ckpt, test_dataset, output_file)
