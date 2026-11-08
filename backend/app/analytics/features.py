"""Feature engineering on a fixed time grid, identical for training (replayed history) and real-time inference.

Each (equipment, sensor, metric) series:
  - averages readings into `grid_s` buckets of MEASUREMENT time (irregular sampling -> regular grid); a bucket
    closes when a reading for a later bucket arrives
  - marks empty buckets as gaps (NaN) instead of interpolating; gaps longer than `max_gap_buckets` reset the forecast
    and block rate-of-change across them, so a reconnecting sensor never looks like a sudden change
  - keeps the last `baseline_buckets` buckets; buckets that were flagged anomalous stay out of the baseline (so a
    persistent fault does not become "normal") unless they exceed `anomalous_share_relearn` of the window
  - maintains a Holt (level + trend) one-step forecast and its residuals
Duplicate readings never reach this layer (the ingestion pipeline drops them); late readings are ignored.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np


@dataclass
class Features:
    """Features of the most recently closed bucket, relative to the baseline before it."""

    t: datetime
    value: float | None  # bucket mean (None = gap)
    n_history: int  # valid baseline buckets
    availability: float  # valid share of the baseline window (sensor data availability)
    median: float | None = None
    mad: float | None = None
    q1: float | None = None
    q3: float | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    pct_deviation: float | None = None  # (value - median) / |median|
    roc: float | None = None  # change over `short_buckets`
    roc_per_h: float | None = None
    roc_history: np.ndarray | None = None  # baseline changes over the same span
    forecast: float | None = None
    residual: float | None = None
    residual_history: np.ndarray | None = None
    transitions: int | None = None  # state metrics: changes within the baseline window

    def as_dict(self) -> dict:
        out = {}
        for k in ("value", "median", "mad", "q1", "q3", "mean", "std", "min", "max", "pct_deviation", "roc",
                  "roc_per_h", "forecast", "residual", "availability", "n_history", "transitions"):
            v = getattr(self, k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k] = round(float(v), 5) if isinstance(v, float) else v
        return out


@dataclass
class SeriesState:
    grid_s: int
    baseline: int
    short: int
    max_gap: int
    relearn_share: float = 0.5
    alpha: float = 0.3
    beta: float = 0.05
    values: deque = field(default_factory=deque)  # bucket means (NaN = gap)
    flags: deque = field(default_factory=deque)  # bucket was part of an anomaly
    last_index: int | None = None  # index of the last CLOSED bucket
    cur_index: int | None = None
    cur_sum: float = 0.0
    cur_n: int = 0
    level: float | None = None
    trend: float = 0.0
    gap_run: int = 0
    residuals: deque = field(default_factory=deque)
    last_residual: float | None = None
    last_forecast: float | None = None

    @property
    def capacity(self) -> int:
        return self.baseline + 1

    def _push(self, value: float) -> None:
        self.values.append(value)
        self.flags.append(False)
        while len(self.values) > self.capacity:
            self.values.popleft()
            self.flags.popleft()
        self._holt(value)

    def _holt(self, x: float) -> None:
        self.last_residual = self.last_forecast = None
        if math.isnan(x):
            self.gap_run += 1
            if self.gap_run > self.max_gap:
                self.level, self.trend = None, 0.0
            return
        self.gap_run = 0
        if self.level is None:
            self.level, self.trend = x, 0.0
            return
        forecast = self.level + self.trend
        self.last_forecast, self.last_residual = forecast, x - forecast
        level = self.alpha * x + (1 - self.alpha) * forecast
        self.trend = self.beta * (level - self.level) + (1 - self.beta) * self.trend
        self.level = level

    def add(self, ts: datetime, value: float) -> int:
        """Add a reading; returns how many buckets closed (0 when the reading falls in the open bucket)."""
        idx = int(ts.timestamp() // self.grid_s)
        if self.cur_index is None:
            self.cur_index, self.cur_sum, self.cur_n = idx, value, 1
            return 0
        if idx < self.cur_index:
            return 0  # late reading: the bucket is closed
        if idx == self.cur_index:
            self.cur_sum += value
            self.cur_n += 1
            return 0
        closed = 0
        self._close_current()
        closed += 1
        gaps = min(idx - self.cur_index - 1, self.capacity)
        for _ in range(gaps):
            self._push(float("nan"))
        self.cur_index, self.cur_sum, self.cur_n = idx, value, 1
        return closed

    def _close_current(self) -> None:
        mean = self.cur_sum / self.cur_n
        self._push(mean)
        if self.last_residual is not None:
            self.residuals.append(self.last_residual)
            while len(self.residuals) > self.baseline:
                self.residuals.popleft()
        self.last_index = self.cur_index

    def reset_forecast(self) -> None:
        """Forget the Holt state (after an expected interruption such as a mains outage) so the forecast restarts
        from the next value instead of carrying an artificial trend across the interruption."""
        self.level, self.trend = None, 0.0

    def flag_last(self) -> None:
        if self.flags:
            self.flags[-1] = True

    def bucket_time(self) -> datetime:
        return datetime.fromtimestamp((self.last_index + 1) * self.grid_s, UTC)

    def features(self, state_metric: bool = False) -> Features | None:
        if not self.values or self.last_index is None:
            return None
        vals = np.fromiter(self.values, float)
        flags = np.fromiter(self.flags, bool)
        x = vals[-1]
        hist, hflags = vals[:-1], flags[:-1]
        valid = ~np.isnan(hist)
        n_valid = int(valid.sum())
        f = Features(t=self.bucket_time(), value=None if math.isnan(x) else float(x), n_history=n_valid,
                     availability=float(valid.mean()) if len(hist) else 0.0)
        if state_metric:
            seq = vals[~np.isnan(vals)]
            f.transitions = int(np.count_nonzero(np.diff(seq))) if len(seq) > 1 else 0
            return f
        use = valid & ~hflags
        if valid.sum() and (hflags & valid).sum() / valid.sum() > self.relearn_share:
            use = valid  # the shifted level has become the new normal
        base = hist[use]
        if len(base):
            q1, med, q3 = np.percentile(base, [25, 50, 75])
            f.median, f.q1, f.q3 = float(med), float(q1), float(q3)
            f.mad = float(np.median(np.abs(base - med)))
            f.mean, f.std = float(base.mean()), float(base.std())
            f.min, f.max = float(base.min()), float(base.max())
            if f.value is not None and abs(med) > 1e-9:
                f.pct_deviation = (f.value - f.median) / abs(f.median)
        k = self.short
        if len(vals) > k and not math.isnan(x):
            window = vals[-k - 1:]
            if np.isnan(window).sum() <= self.max_gap and not math.isnan(window[0]):
                f.roc = float(x - window[0])
                f.roc_per_h = f.roc * 3600 / (k * self.grid_s)
        if len(hist) > k:
            diffs = hist[k:] - hist[:-k]
            ok = ~np.isnan(diffs) & ~hflags[k:] & ~hflags[:-k]
            f.roc_history = diffs[ok]
        if self.last_residual is not None:
            f.forecast, f.residual = self.last_forecast, self.last_residual
            f.residual_history = np.fromiter(list(self.residuals)[:-1], float) if len(self.residuals) > 1 else None
        return f


def robust_scale(mad: float | None, floor: float) -> float:
    """1.4826 * MAD (the normal-consistent spread), never below the metric's floor."""
    return max(1.4826 * (mad or 0.0), floor)
