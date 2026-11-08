"""Train Isolation Forest models per equipment category (offline; separate from real-time inference).

Training data = stored telemetry replayed through the SAME feature code the engine uses live (AnalyticsEngine with
a feature sink), so training and inference features cannot drift apart. Periods that the context rules explain
(mains outage and recharge) are excluded. A model is trained for one data source and only ever scores equipment of
that source: a model trained on simulated data is never applied to physical equipment.

The last `holdout` share (time-ordered) is held out; its flagged rate at the chosen threshold is recorded as the
false-alarm rate on presumed-normal data. That is a property of THIS data, not real-world accuracy.

From backend/:
    python -m app.analytics.train --data-source simulated --days 7
    python -m app.analytics.train --category battery --data-source physical --days 30 --no-activate
"""

import argparse
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import select

from app.analytics.config import config_hash
from app.analytics.engine import AnalyticsEngine
from app.analytics.models import AnomalyModel
from app.core.config import BACKEND_DIR
from app.telemetry.models import Measurement, Sensor

logger = logging.getLogger(__name__)
MODELS_DIR = BACKEND_DIR / "models" / "anomaly"
MIN_SAMPLES = 300


def replay(sessions, engine: AnalyticsEngine, start: datetime, end: datetime, data_source: str) -> int:
    """Feed stored good readings (time-ordered) into an engine. Returns the number of readings."""
    with sessions() as s:
        sensors = {x.sensor_id: x for x in s.scalars(select(Sensor))}
        q = (select(Measurement.sensor_id, Measurement.metric, Measurement.value, Measurement.ts)
             .where(Measurement.ts >= start, Measurement.ts <= end, Measurement.quality == "good",
                    Measurement.data_source == data_source)
             .order_by(Measurement.ts, Measurement.id).execution_options(yield_per=5000))
        n = 0
        for sid, metric, value, ts in s.execute(q):
            sensor = sensors.get(sid)
            if sensor is None:
                continue
            engine.on_reading(sid, sensor.equipment_id, sensor.category, data_source, metric, value, ts)
            n += 1
        return n


def collect(sessions, cfg: dict, start: datetime, end: datetime, data_source: str) -> dict[str, dict]:
    """Multivariate feature vectors per category: {category: {"names", "X", "t", "equipment"}}."""
    out: dict[str, dict] = defaultdict(lambda: {"names": None, "rows": [], "t": [], "equipment": set()})

    def sink(category, src, equipment, t, names, x):
        d = out[category]
        d["names"] = names
        d["rows"].append(x)
        d["t"].append(t)
        d["equipment"].add(equipment)

    engine = AnalyticsEngine(cfg, store=None, mv_sink=sink)
    n = replay(sessions, engine, start, end, data_source)
    logger.info("Replayed %d readings", n)
    return {c: {"names": d["names"], "X": np.vstack(d["rows"]), "t": d["t"], "equipment": sorted(d["equipment"])}
            for c, d in out.items() if d["rows"]}


def fit(X: np.ndarray, quantile: float, seed: int, holdout: float = 0.2, n_estimators: int = 200) -> dict:
    from sklearn.ensemble import IsolationForest

    n_train = int(len(X) * (1 - holdout))
    train, test = X[:n_train], X[n_train:]
    model = IsolationForest(n_estimators=n_estimators, max_samples="auto", contamination="auto", random_state=seed,
                            n_jobs=1).fit(train)
    a_train = -model.score_samples(train)
    threshold, center = float(np.quantile(a_train, quantile)), float(np.median(a_train))
    a_test = -model.score_samples(test) if len(test) else np.array([])
    median = np.median(train, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(train - median), axis=0), 1e-6)
    return {"model": model, "threshold": threshold, "center": center, "median": median, "scale": scale,
            "evaluation": {"train_samples": int(len(train)), "holdout_samples": int(len(test)),
                           "train_flagged_rate": float((a_train >= threshold).mean()),
                           "holdout_flagged_rate": float((a_test >= threshold).mean()) if len(test) else None,
                           "score_center": center, "score_threshold": threshold,
                           "note": "holdout = latest data, presumed normal; flagged rate ~ false-alarm rate on this data"}}


def train(sessions, analytics_store, cfg: dict, data_source: str, start: datetime, end: datetime,
          categories: list[str] | None = None, seed: int = 42, activate: bool = True,
          models_dir: Path = MODELS_DIR) -> list[AnomalyModel]:
    import joblib

    q = cfg["detectors"].get("isolation_forest", {}).get("quantile", 0.999)
    data = collect(sessions, cfg, start, end, data_source)
    models_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for category, d in sorted(data.items()):
        if categories and category not in categories:
            continue
        if len(d["X"]) < MIN_SAMPLES:
            logger.warning("%s: only %d samples (need %d); not trained", category, len(d["X"]), MIN_SAMPLES)
            continue
        r = fit(d["X"], q, seed)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        path = models_dir / f"{category}_{data_source}_iforest_{version}.joblib"
        joblib.dump({"model": r["model"], "features": d["names"], "median": r["median"], "scale": r["scale"],
                     "threshold": r["threshold"], "center": r["center"]}, path)
        row = AnomalyModel(category=category, data_source=data_source, algorithm="isolation_forest", version=version,
                           artifact_path=str(path.relative_to(BACKEND_DIR)) if path.is_relative_to(BACKEND_DIR)
                           else str(path),
                           config_hash=config_hash(cfg), features=d["names"],
                           params={"n_estimators": 200, "max_samples": "auto", "random_state": seed},
                           thresholds={"quantile": q, "score_threshold": r["threshold"], "score_center": r["center"]},
                           training={"start": start.isoformat(), "end": end.isoformat(), "data_source": data_source,
                                     "samples": int(len(d["X"])), "equipment": d["equipment"]},
                           evaluation=r["evaluation"])
        saved.append(analytics_store.add_model(row, activate=activate))
        logger.info("%s: trained on %d samples, holdout flagged %.4f", category, len(d["X"]),
                    r["evaluation"]["holdout_flagged_rate"] or 0)
    return saved


def main() -> None:
    from app.analytics.config import load
    from app.analytics.store import AnalyticsStore
    from app.core.config import get_settings
    from app.telemetry import db

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-source", choices=["simulated", "physical"], required=True)
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--category", action="append", help="limit to these categories (repeatable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-activate", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    url, kind = db.database_url(settings)
    engine = db.make_engine(url)
    db.upgrade(engine)
    sessions = db.session_factory(engine)
    astore = AnalyticsStore(sessions)
    active = astore.active_config()
    cfg = load()[0] if active is None else __import__("app.analytics.config", fromlist=["parse"]).parse(active.content)
    end = datetime.now(UTC)
    rows = train(sessions, astore, cfg, a.data_source, end - timedelta(days=a.days), end, a.category, a.seed,
                 not a.no_activate)
    print(f"Database: {kind}. Trained {len(rows)} model(s): " + ", ".join(f"{r.category}#{r.id}" for r in rows))
    print("Restart the API or POST /api/analytics/models/reload to load them.")


if __name__ == "__main__":
    main()
