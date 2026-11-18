"""gta6-twitch-ccu-forecast-01 entry: comparables-regression forecast.

Model (fit per case, from the case's own comparables — nothing is hardcoded
to a particular launch):
  For each hour h = 2..24, every comparable launch gives one point,
  log(viewers(h) / viewers(1)) against log(viewers(1)) and against
  log(channels(1)). Two least-squares lines are fit per hour; their
  predictions at this case's hour-1 print are averaged in log space. That
  average is then blended 75/25 with the median log-ratio across
  comparables (a hedge: the regressions say a launch day's level is nearly
  independent of its hour-1 print, so a pure regression would cap a
  record launch at a typical level; the blend lets a very large hour-1
  print lift the forecast). Bands are the empirical percentiles of the
  viewers-regression residuals, shrunk 60% toward the centre.

  Why this shape: across the 38 comparables, the raw log-ratio spread at
  late hours is ~2.2 (a mix of rolling-midnight console launches and
  global/PC launches), but after regressing on log hour-1 viewers the
  residual spread falls to ~0.8 — the day's level is nearly independent
  of hour 1 (log peak = 11.67 + 0.07 * log viewers(1)), so the ratio is
  mostly 1/viewers(1). Early hours (2-6) have slope ~0: the ratio is
  ~0.75 whatever hour 1 was, which the per-hour fit captures too.

  Practice (4 cases, answers fetched from SullyGnome GameViewers):
  house baseline 0.387097; this model 0.179438 (typical miss ~31%,
  90% coverage ~0.91). Per case: p00 0.183, p01 0.187, p02 0.134, p03 0.214.

Stdlib only, no network, deterministic (no randomness; iteration order
fixed by the input file). Reads every <id>.in in the input directory and
writes <id>.out to the output directory.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (starter's spread, used only
# by the fallback when a case carries no usable comparables).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15
# Bands: empirical residual percentiles shrunk this far toward the centre.
SHRINK = 0.6
# Centre: weight of the median-ratio hedge (the rest is the regressions).
W_MEDIAN = 0.25
# A regression needs at least this many comparables for an hour, else fall back.
MIN_POINTS = 10


def round2(x):
    """x rounded to 2 decimals, half up (same convention as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def median(vals):
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def percentile(sorted_vals, q):
    """Linear-interpolated percentile of a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def ols(points):
    """Least squares y = a + b*x over [(x, y)]; returns (a, b)."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        return my, 0.0
    b = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
    return my - b * mx, b


def release_day_rows(comp):
    """{hour: row} for the comparable's release day (hour 1 = 00:00 UTC)."""
    day = comp["day_start"][:10]
    rows = {}
    for row in comp["hourly"]:
        if row["from"][:10] == day:
            rows[int(row["from"][11:13]) + 1] = row
    return rows


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first.get("viewers") or 0)
    log_p1 = math.log(p1)
    ch1 = first.get("channels") or 0
    log_ch1 = math.log(ch1) if ch1 and ch1 > 0 else None

    # Per hour: (log viewers(1), log ratio) and (log channels(1), log ratio)
    # over comparables with a valid positive hour-1 and hour-h.
    by_hour_v = {h: [] for h in range(2, 25)}
    by_hour_c = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables") or []:
        rows = release_day_rows(comp)
        r1 = rows.get(1)
        if not r1:
            continue
        v1 = r1["viewers"]
        c1 = r1.get("channels") or 0
        if not v1 or v1 <= 0:
            continue
        for h in range(2, 25):
            rh = rows.get(h)
            if not rh:
                continue
            vh = rh["viewers"]
            if not vh or vh <= 0:
                continue
            by_hour_v[h].append((math.log(v1), math.log(vh / v1)))
            if c1 and c1 > 0:
                by_hour_c[h].append((math.log(c1), math.log(vh / v1)))

    hours = []
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    for row in sorted(rows, key=lambda r: r["hour"]):
        h = row["hour"]
        pts_v = by_hour_v[h]
        pts_c = by_hour_c[h]
        if len(pts_v) >= MIN_POINTS:
            a1, b1 = ols(pts_v)
            center_reg = a1 + b1 * log_p1
            if log_ch1 is not None and len(pts_c) >= MIN_POINTS:
                a2, b2 = ols(pts_c)
                center_reg = 0.5 * center_reg + 0.5 * (a2 + b2 * log_ch1)
            resid = sorted(y - (a1 + b1 * x) for x, y in pts_v)
            log_ratios = sorted(y for _, y in pts_v)
            center = (1 - W_MEDIAN) * center_reg + W_MEDIAN * median(log_ratios)
            qs = [center + percentile(resid, q) * SHRINK for q in LEVELS]
        elif pts_v:
            # Too few comparables for a fit: median ratio, empirical bands.
            vals = sorted(y for _, y in pts_v)
            med = median(vals)
            qs = [med + (percentile(vals, q) - med) * SHRINK for q in LEVELS]
        else:
            # No comparables at all: the starter's persistence forecast.
            qs = [z * SIGMA * math.sqrt(h - 1) for z in Z]
        out_q = []
        prev = 0.0
        for x in qs:
            v = p1 * math.exp(x)
            v = min(1e9, max(1.0, v))
            v = round2(v)
            if v < prev:  # keep quantiles non-decreasing after rounding
                v = prev
            prev = v
            out_q.append(v)
        hours.append({"hour": h, "q": out_q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
