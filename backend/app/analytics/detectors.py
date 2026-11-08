"""Univariate anomaly detectors. Each turns Features into a Finding with a score in THRESHOLD UNITS
(1.0 = at the detector's threshold, 2.0 = twice as far), an expected value and a plain-language explanation.
They return None when there is not enough data to judge (never a guess).

  robust_z           distance of the bucket mean from the rolling median, in robust standard deviations
  iqr                distance outside the interquartile fences
  rate_of_change     the 10-minute change compared with the baseline's 10-minute changes (sudden jumps / drops);
                     a change that brings the value back towards its baseline (recovery) scores 0
  forecast_residual  error of a Holt one-step forecast compared with past forecast errors (spikes, oscillation);
                     a value closer to its baseline than the (overshooting) forecast scores 0
  transitions        state metrics: number of state changes in the window (flapping, repeated door openings)
"""

from dataclasses import dataclass

import numpy as np

from app.analytics.features import Features, robust_scale

MAX_SCORE = 50.0
MIN_HISTORY_POINTS = 20


@dataclass
class Finding:
    detector: str
    score: float
    observed: float | None
    expected: float | None
    direction: str | None  # high | low | None
    detail: str


def _cap(s: float) -> float:
    return float(min(MAX_SCORE, max(0.0, s)))


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "n/a"
    digits = 0 if abs(v) >= 100 else (1 if abs(v) >= 10 else 2)
    return f"{v:.{digits}f}{'' if unit in ('state', 'ratio') else ' ' + unit}"


def robust_z(f: Features, p: dict, floor: float, unit: str, label: str) -> Finding | None:
    if f.value is None or f.median is None or f.n_history < MIN_HISTORY_POINTS:
        return None
    scale = robust_scale(f.mad, floor)
    z = (f.value - f.median) / scale
    return Finding("robust_z", _cap(abs(z) / p["threshold"]), f.value, f.median, "high" if z > 0 else "low",
                   f"{label} {_fmt(f.value, unit)} is {abs(z):.1f} robust standard deviations {'above' if z > 0 else 'below'}"
                   f" its recent median {_fmt(f.median, unit)} (threshold {p['threshold']:g})")


def iqr(f: Features, p: dict, floor: float, unit: str, label: str) -> Finding | None:
    if f.value is None or f.q1 is None or f.n_history < MIN_HISTORY_POINTS:
        return None
    spread = max(f.q3 - f.q1, floor)
    k = p["k"]
    if f.value > f.q3:
        dist, direction, fence = f.value - f.q3, "high", f.q3 + k * spread
    elif f.value < f.q1:
        dist, direction, fence = f.q1 - f.value, "low", f.q1 - k * spread
    else:
        return Finding("iqr", 0.0, f.value, f.median, None, f"{label} inside its interquartile range")
    return Finding("iqr", _cap(dist / (k * spread)), f.value, f.median, direction,
                   f"{label} {_fmt(f.value, unit)} is outside the {'upper' if direction == 'high' else 'lower'} fence "
                   f"{_fmt(fence, unit)} (Q1 {_fmt(f.q1, unit)}, Q3 {_fmt(f.q3, unit)}, k={k:g})")


def rate_of_change(f: Features, p: dict, floor: float, unit: str, label: str) -> Finding | None:
    if f.roc is None or f.roc_history is None or len(f.roc_history) < MIN_HISTORY_POINTS:
        return None
    if f.value is not None and f.median is not None and abs(f.value - f.median) < abs(f.value - f.roc - f.median):
        # the change brought the value back towards its baseline: that is recovery, not an anomaly
        return Finding("rate_of_change", 0.0, f.value, f.median, None, f"{label} is returning towards its baseline")
    med = float(np.median(f.roc_history))
    scale = robust_scale(float(np.median(np.abs(f.roc_history - med))), floor)
    z = (f.roc - med) / scale
    return Finding("rate_of_change", _cap(abs(z) / p["threshold"]), f.value, (f.value - f.roc + med) if f.value is not None
                   else None, "high" if z > 0 else "low",
                   f"{label} changed by {f.roc:+.3g} {unit if unit not in ('state', 'ratio') else ''} in the last window "
                   f"({f.roc_per_h:+.3g}/h); typical change is {med:+.3g} ± {scale:.3g} (|z| {abs(z):.1f}, threshold "
                   f"{p['threshold']:g})".replace("  ", " "))


def forecast_residual(f: Features, p: dict, floor: float, unit: str, label: str) -> Finding | None:
    if f.residual is None or f.residual_history is None or len(f.residual_history) < MIN_HISTORY_POINTS:
        return None
    if f.value is not None and f.median is not None and f.forecast is not None and \
            abs(f.value - f.median) < abs(f.forecast - f.median):
        # the forecast overshoots (e.g. after a step) while the value is back near its baseline: recovery
        return Finding("forecast_residual", 0.0, f.value, f.forecast, None, f"{label} is back near its baseline")
    med = float(np.median(f.residual_history))
    scale = robust_scale(float(np.median(np.abs(f.residual_history - med))), floor)
    z = (f.residual - med) / scale
    return Finding("forecast_residual", _cap(abs(z) / p["threshold"]), f.value, f.forecast,
                   "high" if z > 0 else "low",
                   f"{label} {_fmt(f.value, unit)} differs from the forecast {_fmt(f.forecast, unit)} by {f.residual:+.3g}; "
                   f"typical forecast error ± {scale:.3g} (|z| {abs(z):.1f}, threshold {p['threshold']:g})")


def transitions(f: Features, p: dict, floor: float, unit: str, label: str) -> Finding | None:
    if f.transitions is None or f.n_history < MIN_HISTORY_POINTS:
        return None
    limit = p["max_per_window"]
    return Finding("transitions", _cap(f.transitions / limit), f.value, None, "high",
                   f"{label} changed state {f.transitions} times in the baseline window (limit {limit})")


UNIVARIATE = {"robust_z": robust_z, "iqr": iqr, "rate_of_change": rate_of_change,
              "forecast_residual": forecast_residual, "transitions": transitions}
