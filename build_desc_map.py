#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build index->description map from Atten-TPL app_info.json (JSONL).

Expected input JSONL fields per line:
{
  "index": <int>,
  "app_id": <package_name_str>,
  "href": <url>,
  "name": <app_name>,
  "category": <category>,
  "description": <text>
}

Output:
- desc_map: dict[int, str]  (serialized as JSON/JSON.GZ)
- optional meta_map: dict[int, dict] (name/category/href/package)
"""

import argparse
import gzip
import io
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Any


_WS_RE = re.compile(r"[ \t]+")


def _open_text(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    """Open plain or gzipped text file based on extension."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding)
    return path.open(mode=mode, encoding=encoding)


def clean_description(text: str, keep_newlines: bool = True) -> str:
    """Light cleaning: normalize whitespace but preserve meaning."""
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    if keep_newlines:
        # normalize spaces inside lines, keep line breaks
        lines = []
        for ln in text.splitlines():
            ln = _WS_RE.sub(" ", ln).strip()
            if ln != "":
                lines.append(ln)
        return "\n".join(lines)
    # collapse everything to single line
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join([_WS_RE.sub(" ", ln).strip() for ln in text.split("\n") if ln.strip()])
    return text.strip()


def load_maps(app_info_path: Path, keep_newlines: bool, min_desc_len: int) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]]]:
    desc_map: Dict[int, str] = {}
    meta_map: Dict[int, Dict[str, Any]] = {}

    duplicates = 0
    empty_desc = 0
    total = 0

    with _open_text(app_info_path, "rt", "utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed line but keep going
                continue

            if "index" not in obj:
                continue

            try:
                idx = int(obj["index"])
            except Exception:
                continue

            desc = clean_description(obj.get("description", ""), keep_newlines=keep_newlines)
            if len(desc) < min_desc_len:
                empty_desc += 1
                desc = ""  # store empty so we can detect coverage later

            # If duplicate index appears, keep the longer description
            if idx in desc_map:
                duplicates += 1
                if len(desc) > len(desc_map[idx]):
                    desc_map[idx] = desc
            else:
                desc_map[idx] = desc

            # meta (optional, cheap)
            meta_map[idx] = {
                "package": obj.get("app_id", ""),
                "name": obj.get("name", ""),
                "category": obj.get("category", ""),
                "href": obj.get("href", ""),
            }

    print(f"[OK] Read lines (non-empty JSON): {total}")
    print(f"[OK] Unique indices stored: {len(desc_map)}")
    print(f"[INFO] Duplicates encountered: {duplicates}")
    print(f"[INFO] Descriptions shorter than {min_desc_len} chars: {empty_desc}")

    return desc_map, meta_map


def save_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON keys must be strings, so cast int->str for portability
    obj_str_keys = {str(k): v for k, v in obj.items()}

    if str(out_path).endswith(".gz"):
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(obj_str_keys, f, ensure_ascii=False)
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj_str_keys, f, ensure_ascii=False)

    print(f"[OK] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app_info_path", type=str, required=True,
                    help="Path to JSONL app_info.json (optionally .gz)")
    ap.add_argument("--out_desc_path", type=str, required=True,
                    help="Output path for index->description JSON (.json or .json.gz)")
    ap.add_argument("--out_meta_path", type=str, default="",
                    help="Optional output path for index->meta JSON (.json or .json.gz)")
    ap.add_argument("--keep_newlines", type=int, default=1,
                    help="1 keeps newlines; 0 collapses to one line")
    ap.add_argument("--min_desc_len", type=int, default=1,
                    help="Descriptions shorter than this become empty string (coverage tracking)")

    args = ap.parse_args()

    app_info_path = Path(args.app_info_path)
    out_desc_path = Path(args.out_desc_path)
    out_meta_path = Path(args.out_meta_path) if args.out_meta_path else None

    desc_map, meta_map = load_maps(
        app_info_path=app_info_path,
        keep_newlines=bool(args.keep_newlines),
        min_desc_len=int(args.min_desc_len),
    )

    save_json(desc_map, out_desc_path)
    if out_meta_path is not None:
        save_json(meta_map, out_meta_path)


if __name__ == "__main__":
    main()
