"""Cache and align issue-time-safe Open-Meteo GFS previous-day-2 forecasts.

Availability timestamps are conservative inferred bounds, not observed run release
times. Original provider responses are retained with queries and SHA-256 hashes.
Source: https://open-meteo.com/en/docs/previous-runs-api (CC BY 4.0 attribution).
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import common as C

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ("shortwave_radiation", "cloud_cover", "temperature_2m",
             "relative_humidity_2m", "wind_speed_10m")
FIELDS = tuple(v + "_previous_day2" for v in VARIABLES)
UNITS = ("W/m²", "%", "°C", "%", "km/h")
FEATURE_NAMES = ("nwp_ghi_wm2", "nwp_cloud_pct", "nwp_temp_c", "nwp_rh_pct", "nwp_wind_kmh")
DEFAULT_CACHE = C.PROJECT_ROOT / "data" / "external" / "open_meteo_gfs_day2"
PROTOCOL = {"version": 1, "endpoint": ENDPOINT, "model": "gfs_global",
            "latitude": C.MUMBAI_LAT, "longitude": C.MUMBAI_LON,
            "offset_hours": 48, "availability_buffer_hours": 12,
            "availability_kind": "inferred_upper_bound_not_observed_publication",
            "variables": list(VARIABLES), "units": list(UNITS), "timezone": "UTC",
            "attribution": "Open-Meteo / NOAA GFS; CC BY 4.0",
            "documentation": "https://open-meteo.com/en/docs/previous-runs-api"}


class BudgetExpired(TimeoutError):
    pass


class Budget:
    def __init__(self, seconds=3600):
        if seconds <= 0:
            raise ValueError("Runtime budget must be positive")
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def check(self, reserve=0):
        if time.monotonic() + reserve >= self.deadline:
            raise BudgetExpired("Runtime budget exhausted; completed work can resume")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def month_queries(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for month in pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS"):
        last = month.replace(day=calendar.monthrange(month.year, month.month)[1])
        yield month.strftime("%Y-%m"), {
            "latitude": C.MUMBAI_LAT, "longitude": C.MUMBAI_LON,
            "start_date": max(start, month).strftime("%Y-%m-%d"),
            "end_date": min(end, last).strftime("%Y-%m-%d"),
            "hourly": ",".join(FIELDS), "models": "gfs_global", "timezone": "UTC"}


def decode_response(raw):
    data = json.loads(raw)
    if data.get("error"):
        raise ValueError(f"Provider error: {data.get('reason')}")
    if data.get("utc_offset_seconds") != 0 or data.get("timezone") not in ("GMT", "UTC"):
        raise ValueError("Forecast response must use UTC")
    hourly = data["hourly"]
    stamps = pd.to_datetime(hourly["time"], utc=True)
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("Forecast timestamps must be unique and chronological")
    columns = {}
    for field, unit in zip(FIELDS, UNITS):
        if data["hourly_units"].get(field) != unit:
            raise ValueError(f"Unexpected units for {field}")
        values = np.asarray(hourly[field], dtype=float)
        if values.shape != (len(stamps),):
            raise ValueError("Forecast variable lengths differ")
        columns[field] = values
    return pd.DataFrame(columns, index=stamps), data


def read_chunk(raw_path, metadata_path, query):
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol"] != PROTOCOL or metadata["query"] != query:
        raise ValueError(f"Incompatible forecast cache: {raw_path}")
    if sha256(raw_path) != metadata["sha256"]:
        raise ValueError(f"Forecast cache checksum differs: {raw_path}")
    frame, _ = decode_response(raw_path.read_bytes())
    return frame, metadata


def acquire(cache, start, end, budget):
    """Resume monthly downloads; surface network failures instead of using actuals."""
    cache = Path(cache)
    raw_folder = cache / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    frames, chunks, errors = [], [], []
    for month, query in month_queries(start, end):
        budget.check(reserve=25)
        raw_path, meta_path = raw_folder / f"{month}.json", raw_folder / f"{month}.metadata.json"
        if raw_path.exists() or meta_path.exists():
            if not (raw_path.exists() and meta_path.exists()):
                raise ValueError(f"Incomplete raw cache pair: {raw_path}")
            frame, metadata = read_chunk(raw_path, meta_path, query)
        else:
            url = ENDPOINT + "?" + urllib.parse.urlencode(query)
            raw = None
            for attempt in range(3):
                budget.check(reserve=22)
                try:
                    with urllib.request.urlopen(url, timeout=20) as response:
                        raw = response.read()
                    frame, response = decode_response(raw)
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    # DNS/sandbox failures must be surfaced for an escalated rerun.
                    if isinstance(getattr(exc, "reason", None), (socket.gaierror, PermissionError)):
                        raise
                    if attempt == 2:
                        errors.append({"month": month, "error": str(exc)})
                    else:
                        time.sleep(2 ** attempt)
            if raw is None or (errors and errors[-1]["month"] == month):
                continue
            tmp = raw_path.with_suffix(".json.tmp")
            tmp.write_bytes(raw)
            tmp.replace(raw_path)
            metadata = {"protocol": PROTOCOL, "query": query, "url": url,
                        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "sha256": sha256(raw_path),
                        "returned_latitude": response["latitude"],
                        "returned_longitude": response["longitude"],
                        "units": response["hourly_units"], "hours": len(frame)}
            write_json(meta_path, metadata)
            print(f"[acquire] {month}: {len(frame)} hours", flush=True)
            time.sleep(0.2)
        frames.append(frame)
        chunks.append({"file": str(raw_path.relative_to(cache)), "sha256": metadata["sha256"],
                       "metadata_sha256": sha256(meta_path)})
    if not frames:
        raise ValueError("No archived forecasts could be acquired")
    hourly = pd.concat(frames).sort_index()
    if hourly.index.has_duplicates:
        raise ValueError("Overlapping forecast cache chunks")
    return hourly, {"protocol": PROTOCOL, "start": start, "end": end,
                    "chunks": chunks, "download_errors": errors}


def availability_bounds(valid_times, issue_times, offset_hours=48, buffer_hours=12):
    valid = np.asarray(valid_times, dtype="datetime64[s]")
    issue = np.asarray(issue_times, dtype="datetime64[s]")
    bound = valid - np.timedelta64(offset_hours, "h") + np.timedelta64(buffer_hours, "h")
    if np.any(bound > issue):
        raise ValueError("Forecast inferred availability exceeds midnight issue time")
    return bound


def solar_disaggregate(hourly_ghi, hour_ids, cs):
    """Scale each six-slot hour by known clear-sky weights, before kt clipping."""
    cs = np.asarray(cs, dtype=float)
    ids = np.asarray(hour_ids)
    known_cs = np.maximum(np.nan_to_num(cs, nan=0.0), 0)
    groups, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=known_cs)
    factor = np.divide(6 * known_cs, totals[inverse], out=np.zeros_like(known_cs),
                       where=totals[inverse] > 0)
    values = np.asarray(hourly_ghi, float)
    result = np.maximum(values, 0) * factor
    result[counts[inverse] != 6] = np.nan
    # No solar proxy in an hour means night. Nonzero forecast radiation there is
    # flagged separately rather than creating light at night.
    return result, {"groups": groups, "inverse": inverse, "cs_totals": totals,
                    "counts": counts}


def align_forecasts(data, hourly):
    """Uses dates/known solar proxy only; never observed GHI or observed weather."""
    dates, cs = data["dates"], data["cs_proxy"]
    n, steps = cs.shape
    if steps != C.STEPS_PER_DAY or n != len(dates):
        raise ValueError("Expected date-aligned day-by-144 clear-sky proxy")
    # Pandas 3 may parse date strings at microsecond resolution. Explicit ns is
    # required before adding nanosecond ten-minute offsets to the integer clock.
    local = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(C.TZ).as_unit("ns")
    start = pd.DatetimeIndex((local.tz_convert("UTC").asi8[:, None] +
                              np.arange(steps) * 10 * 60 * 1_000_000_000).ravel(), tz="UTC")
    midpoint = start + pd.Timedelta(minutes=5)
    hour_end = start.floor("h") + pd.Timedelta(hours=1)
    issue = np.repeat(local.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[s]"), steps)
    bound = availability_bounds(hour_end.tz_localize(None).to_numpy(dtype="datetime64[s]"), issue)
    end_pos = hourly.index.get_indexer(hour_end)
    valid_end = end_pos >= 0
    ghi_hour = np.full(len(start), np.nan)
    ghi_hour[valid_end] = hourly[FIELDS[0]].to_numpy()[end_pos[valid_end]]
    solar, groups = solar_disaggregate(ghi_hour, hour_end.asi8, cs.ravel())
    features = [solar]
    lo = hourly.index.get_indexer(midpoint.floor("h"))
    hi = hourly.index.get_indexer(midpoint.ceil("h"))
    fraction = (midpoint - midpoint.floor("h")).total_seconds().to_numpy() / 3600
    valid = (lo >= 0) & (hi >= 0)
    for field in FIELDS[1:]:
        values = hourly[field].to_numpy()
        result = np.full(len(start), np.nan)
        result[valid] = values[lo[valid]] * (1 - fraction[valid]) + values[hi[valid]] * fraction[valid]
        features.append(result)
    features = np.stack(features, axis=-1).reshape(n, steps, len(FIELDS))
    available = np.isfinite(features).all(axis=(1, 2)) & np.isfinite(cs).all(axis=1)
    # Audit conservation on complete, positive-proxy hours, independently of truth.
    sums = np.bincount(groups["inverse"], weights=np.nan_to_num(solar, nan=0.0)) / 6
    source = np.bincount(groups["inverse"], weights=np.nan_to_num(ghi_hour, nan=0.0)) / 6
    check = (groups["counts"] == 6) & (groups["cs_totals"] > 0)
    if check.any() and not np.allclose(sums[check], np.maximum(source[check], 0), atol=1e-8):
        raise ValueError("Hourly radiation energy was not preserved")
    night_nonzero = np.isfinite(ghi_hour) & (ghi_hour > 1) & (groups["cs_totals"][groups["inverse"]] == 0)
    audit = {"available_days": int(available.sum()),
             "hourly_energy_max_error_wm2": float(np.max(abs(sums[check] - np.maximum(source[check], 0)))) if check.any() else 0.,
             "night_nonzero_source_slots": int(night_nonzero.sum()),
             "earliest_available_date": str(dates[np.flatnonzero(available)[0]]) if available.any() else None,
             "latest_available_date": str(dates[np.flatnonzero(available)[-1]]) if available.any() else None}
    return {"features": features.astype(np.float32), "ghi": solar.reshape(n, steps),
            "available": available, "availability_upper_bound_utc": bound.reshape(n, steps),
            "dates": dates, "feature_names": np.array(FEATURE_NAMES)}, audit


def prepare(cache, prep_path, budget):
    cache = Path(cache)
    with np.load(prep_path, allow_pickle=False) as source:
        data = {"dates": source["dates"], "cs_proxy": source["cs_proxy"]}
    start = "2023-12-31"
    end = str(pd.Timestamp(str(data["dates"][-1])) + pd.Timedelta(days=1))[:10]
    hourly, manifest = acquire(cache, start, end, budget)
    coverage = {}
    for field in FIELDS:
        valid = hourly[field].dropna()
        coverage[field] = {"available_hours": len(valid), "missing_hours": int(hourly[field].isna().sum()),
                           "first_utc": valid.index[0].isoformat() if len(valid) else None,
                           "last_utc": valid.index[-1].isoformat() if len(valid) else None}
    write_json(cache / "coverage.json", coverage)
    arrays, audit = align_forecasts(data, hourly)
    manifest.update(prep_sha256=sha256(prep_path), audit=audit)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Prepared weather cache differs; use a fresh cache directory")
    write_json(manifest_path, manifest)
    arrays["config"] = json.dumps({"protocol": PROTOCOL, "prep_sha256": sha256(prep_path),
                                   "manifest_sha256": sha256(manifest_path)}, sort_keys=True)
    target = cache / "forecasts.npz"
    if not target.exists():
        temp = cache / "forecasts.tmp.npz"
        np.savez_compressed(temp, **arrays)
        temp.replace(target)
    digest_path = cache / "forecasts.sha256"
    if not digest_path.exists():
        # Recover only after proving an existing derived cache equals a fresh
        # transform of its checksummed raw source, never bless edited predictions.
        with np.load(target, allow_pickle=False) as saved:
            for key, value in arrays.items():
                expected = np.asarray(value)
                matches = (np.array_equal(saved[key], expected, equal_nan=True)
                           if expected.dtype.kind in "fc" else np.array_equal(saved[key], expected))
                if not matches:
                    raise ValueError("Existing aligned forecast cache differs from raw sources")
        digest_path.write_text(sha256(target) + "\n")
    return load_forecasts(cache, prep_path)


def load_forecasts(cache, prep_path):
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    if sha256(cache / "forecasts.npz") != (cache / "forecasts.sha256").read_text().strip():
        raise ValueError("Aligned forecast cache checksum differs")
    with np.load(cache / "forecasts.npz", allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    config = json.loads(str(arrays.pop("config")))
    if config != {"protocol": PROTOCOL, "prep_sha256": sha256(prep_path),
                  "manifest_sha256": sha256(manifest_path)}:
        raise ValueError("Forecast cache configuration/data fingerprint differs")
    return arrays, json.loads(manifest_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, default=C.PREP_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    _, manifest = prepare(args.cache, args.prep, Budget(args.max_seconds))
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
