"""Cache and align additional Open-Meteo previous-day-2 forecasts (ECMWF IFS, DWD ICON).

.venv/bin/python pipelines/multi_nwp/fetch.py --model ecmwf_ifs025
.venv/bin/python pipelines/multi_nwp/fetch.py --model icon_global

Same variables, 48-hour offset, 12-hour availability buffer and alignment as the GFS
cache (pv_pipeline/nwp_data.py), which is reused and left untouched.
Source: https://open-meteo.com/en/docs/previous-runs-api (CC BY 4.0 attribution).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
import common as C
import nwp_data as N

MODELS = {"ecmwf_ifs025": "ECMWF IFS 0.25°", "icon_global": "DWD ICON global"}


def cache_dir(model: str) -> Path:
    return ROOT / "data" / "external" / f"open_meteo_{model}_day2"


def protocol(model: str) -> dict:
    return {**N.PROTOCOL, "model": model, "attribution": f"Open-Meteo / {MODELS[model]}; CC BY 4.0"}


def acquire(model: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    raw_folder = cache_dir(model) / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    frames, chunks = [], []
    for month, query in N.month_queries(start, end):
        query = {**query, "models": model}
        raw_path, meta_path = raw_folder / f"{month}.json", raw_folder / f"{month}.metadata.json"
        if raw_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["query"] != query or N.sha256(raw_path) != meta["sha256"]:
                raise ValueError(f"Cache mismatch: {raw_path}")
            frame, _ = N.decode_response(raw_path.read_bytes())
        else:
            url = N.ENDPOINT + "?" + urllib.parse.urlencode(query)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        raw = response.read()
                    frame, decoded = N.decode_response(raw)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            raw_path.write_bytes(raw)
            N.write_json(meta_path, {"protocol": protocol(model), "query": query, "url": url,
                                     "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                     "sha256": N.sha256(raw_path), "hours": len(frame),
                                     "returned_latitude": decoded["latitude"],
                                     "returned_longitude": decoded["longitude"]})
            print(f"[{model}] {month}: {int(frame[N.FIELDS[0]].notna().sum())}/{len(frame)} hours", flush=True)
            time.sleep(0.3)
        frames.append(frame)
        chunks.append({"file": str(raw_path.relative_to(cache_dir(model))), "sha256": N.sha256(raw_path)})
    hourly = pd.concat(frames).sort_index()
    if hourly.index.has_duplicates:
        raise ValueError("Overlapping forecast chunks")
    return hourly, {"protocol": protocol(model), "start": start, "end": end, "chunks": chunks}


def prepare(model: str, prep_path: Path) -> None:
    with np.load(prep_path, allow_pickle=False) as source:
        data = {"dates": source["dates"], "cs_proxy": source["cs_proxy"]}
    end = str(pd.Timestamp(str(data["dates"][-1])) + pd.Timedelta(days=1))[:10]
    hourly, manifest = acquire(model, "2023-12-31", end)
    arrays, audit = N.align_forecasts(data, hourly)
    manifest.update(prep_sha256=N.sha256(prep_path), audit=audit)
    folder = cache_dir(model)
    N.write_json(folder / "manifest.json", manifest)
    arrays["config"] = json.dumps({"protocol": protocol(model), "prep_sha256": N.sha256(prep_path),
                                   "manifest_sha256": N.sha256(folder / "manifest.json")}, sort_keys=True)
    np.savez_compressed(folder / "forecasts.npz", **arrays)
    (folder / "forecasts.sha256").write_text(N.sha256(folder / "forecasts.npz") + "\n")
    print(json.dumps(audit, indent=2))


def load(model: str, prep_path: Path) -> dict:
    folder = cache_dir(model)
    if N.sha256(folder / "forecasts.npz") != (folder / "forecasts.sha256").read_text().strip():
        raise ValueError(f"{model} forecast cache checksum differs")
    with np.load(folder / "forecasts.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    config = json.loads(str(arrays.pop("config")))
    if config != {"protocol": protocol(model), "prep_sha256": N.sha256(prep_path),
                  "manifest_sha256": N.sha256(folder / "manifest.json")}:
        raise ValueError(f"{model} forecast cache fingerprint differs")
    return arrays


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--prep", type=Path, default=C.PREP_PATH)
    args = ap.parse_args()
    prepare(args.model, args.prep)
