#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_json_gz(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_funcs_map(path: Path) -> Dict[int, List[str]]:
    obj = load_json_gz(path)
    return {int(k): [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
            for k, v in obj.items()}


def load_scores_map(path: Path) -> Dict[int, List[float]]:
    obj = load_json_gz(path)
    return {int(k): [float(x) for x in v] if isinstance(v, list) else []
            for k, v in obj.items()}


def weighted_pool(emb: torch.Tensor, scores: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    emb: [n,d], scores: [n] in [0,1]
    weights = softmax(scores / tau)
    """
    if emb.size(0) == 0:
        return torch.zeros((emb.size(1),), dtype=emb.dtype, device=emb.device)
    if scores.numel() != emb.size(0):
        # fallback to mean if misaligned
        return emb.mean(dim=0)
    w = torch.softmax(scores / tau, dim=0).unsqueeze(1)  # [n,1]
    return (w * emb).sum(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions_path", required=True)
    ap.add_argument("--scores_path", required=True)
    ap.add_argument("--out_pt_path", required=True)
    ap.add_argument("--encoder_name", default="sentence-transformers/distiluse-base-multilingual-cased-v2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--expected_dim", type=int, default=512)
    args = ap.parse_args()

    funcs_map = load_funcs_map(Path(args.functions_path))
    scores_map = load_scores_map(Path(args.scores_path))
    print(f"[OK] Loaded funcs: {len(funcs_map)}, scores: {len(scores_map)}")

    model = SentenceTransformer(args.encoder_name, device=args.device)

    test_vec = model.encode(["test"], convert_to_tensor=True)
    d = int(test_vec.shape[-1])
    print(f"[OK] Encoder dim = {d}")
    if d != args.expected_dim:
        raise SystemExit(f"Dim mismatch: encoder gives {d}, expected {args.expected_dim}")

    out: Dict[int, torch.Tensor] = {}
    empty = 0

    for idx, funcs in tqdm(funcs_map.items(), desc="encode weighted"):
        if not funcs:
            out[idx] = torch.zeros((args.expected_dim,), dtype=torch.float32)
            empty += 1
            continue

        emb = model.encode(funcs, batch_size=args.batch_size, convert_to_tensor=True)  # [n,512]
        sc = torch.tensor(scores_map.get(idx, []), device=emb.device, dtype=torch.float32)  # [n]

        pooled = weighted_pool(emb, sc, tau=args.tau).detach().to("cpu").float()
        out[idx] = pooled

    Path(args.out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_pt_path)
    print(f"[OK] Saved: {args.out_pt_path}")
    print(f"[INFO] apps with empty funcs: {empty} / {len(funcs_map)}")


if __name__ == "__main__":
    main()
