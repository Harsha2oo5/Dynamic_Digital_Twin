"""Database access for analytics (anomalies, health history, recommendations, models, configs)."""

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.analytics.models import Anomaly, AnomalyConfig, AnomalyModel, EquipmentHealth, MaintenanceRecommendation
from app.telemetry.models import utcnow
from app.telemetry.store import NotFound


class AnalyticsStore:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    # --- anomalies --------------------------------------------------------------------------------
    def save_anomaly(self, a: Anomaly) -> Anomaly:
        with self.sessions() as s:
            merged = s.merge(a)
            s.commit()
            return merged

    def get_anomaly(self, anomaly_id: int) -> Anomaly | None:
        with self.sessions() as s:
            return s.get(Anomaly, anomaly_id)

    def unresolved_anomalies(self) -> list[Anomaly]:
        with self.sessions() as s:
            return list(s.scalars(select(Anomaly).where(Anomaly.status != "resolved")))

    def recent_anomalies(self, since: datetime, equipment_id: str | None = None) -> list[Anomaly]:
        """Anomalies active at any time since `since` (opened after it, or still active / ended after it)."""
        with self.sessions() as s:
            q = select(Anomaly).where((Anomaly.last_detected_at >= since) | (Anomaly.condition_active.is_(True)))
            if equipment_id:
                q = q.where(Anomaly.equipment_id == equipment_id)
            return list(s.scalars(q.order_by(Anomaly.first_detected_at)))

    def anomalies_page(self, limit: int, offset: int, **f) -> tuple[int, list[Anomaly]]:
        with self.sessions() as s:
            q = select(Anomaly)
            for col in ("equipment_id", "sensor_id", "category", "severity", "status", "metric", "detector",
                        "data_source", "kind"):
                if f.get(col):
                    q = q.where(getattr(Anomaly, col) == f[col])
            if f.get("active") is not None:
                q = q.where(Anomaly.condition_active.is_(bool(f["active"])))
            if f.get("start"):
                q = q.where(Anomaly.last_detected_at >= f["start"])
            if f.get("end"):
                q = q.where(Anomaly.first_detected_at <= f["end"])
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = list(s.scalars(q.order_by(Anomaly.first_detected_at.desc(), Anomaly.id.desc())
                                  .limit(limit).offset(offset)))
            return total, rows

    def set_anomaly_status(self, anomaly_id: int, status: str, by: str, note: str | None) -> Anomaly:
        with self.sessions() as s:
            a = s.get(Anomaly, anomaly_id)
            if a is None:
                raise NotFound(f"anomaly {anomaly_id} not found")
            now = utcnow()
            if status == "acknowledged":
                if a.status == "resolved":
                    raise ValueError("anomaly is already resolved")
                a.status, a.acknowledged_at, a.acknowledged_by = "acknowledged", now, by
            else:
                a.status, a.resolved_at, a.resolved_by = "resolved", now, by
            if note:
                a.resolution_note = note
            s.commit()
            return a

    def anomaly_counts_by_day(self, equipment_id: str, since: datetime) -> list[datetime]:
        with self.sessions() as s:
            return list(s.scalars(select(Anomaly.first_detected_at)
                                  .where(Anomaly.equipment_id == equipment_id, Anomaly.first_detected_at >= since)))

    # --- health -----------------------------------------------------------------------------------
    def add_health(self, rows: list[dict]) -> None:
        if rows:
            with self.sessions() as s:
                s.add_all(EquipmentHealth(**r) for r in rows)
                s.commit()

    def health_history(self, equipment_id: str, since: datetime) -> list[EquipmentHealth]:
        with self.sessions() as s:
            return list(s.scalars(select(EquipmentHealth).where(EquipmentHealth.equipment_id == equipment_id,
                                                                EquipmentHealth.at >= since)
                                  .order_by(EquipmentHealth.at)))

    # --- recommendations --------------------------------------------------------------------------
    def active_recommendations(self, equipment_id: str | None = None) -> list[MaintenanceRecommendation]:
        with self.sessions() as s:
            q = select(MaintenanceRecommendation).where(MaintenanceRecommendation.status == "active")
            if equipment_id:
                q = q.where(MaintenanceRecommendation.equipment_id == equipment_id)
            return list(s.scalars(q.order_by(MaintenanceRecommendation.created_at)))

    def save_recommendation(self, r: MaintenanceRecommendation) -> MaintenanceRecommendation:
        with self.sessions() as s:
            merged = s.merge(r)
            s.commit()
            return merged

    def recommendations_page(self, limit: int, offset: int, **f) -> tuple[int, list[MaintenanceRecommendation]]:
        with self.sessions() as s:
            q = select(MaintenanceRecommendation)
            for col in ("equipment_id", "category", "status", "evidence_level", "data_source"):
                if f.get(col):
                    q = q.where(getattr(MaintenanceRecommendation, col) == f[col])
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = list(s.scalars(q.order_by(MaintenanceRecommendation.updated_at.desc()).limit(limit).offset(offset)))
            return total, rows

    def acknowledge_recommendation(self, rec_id: int, by: str) -> MaintenanceRecommendation:
        with self.sessions() as s:
            r = s.get(MaintenanceRecommendation, rec_id)
            if r is None:
                raise NotFound(f"recommendation {rec_id} not found")
            r.acknowledged, r.acknowledged_by, r.acknowledged_at = True, by, utcnow()
            s.commit()
            return r

    # --- models -----------------------------------------------------------------------------------
    def add_model(self, m: AnomalyModel, activate: bool = True) -> AnomalyModel:
        with self.sessions() as s:
            if activate:
                s.execute(update(AnomalyModel).where(AnomalyModel.category == m.category,
                                                     AnomalyModel.data_source == m.data_source,
                                                     AnomalyModel.algorithm == m.algorithm).values(active=False))
                m.active = True
            s.add(m)
            s.commit()
            return m

    def models(self, active_only: bool = False) -> list[AnomalyModel]:
        with self.sessions() as s:
            q = select(AnomalyModel).order_by(AnomalyModel.trained_at.desc())
            if active_only:
                q = q.where(AnomalyModel.active.is_(True))
            return list(s.scalars(q))

    def activate_model(self, model_id: int) -> AnomalyModel:
        with self.sessions() as s:
            m = s.get(AnomalyModel, model_id)
            if m is None:
                raise NotFound(f"model {model_id} not found")
            s.execute(update(AnomalyModel).where(AnomalyModel.category == m.category,
                                                 AnomalyModel.data_source == m.data_source,
                                                 AnomalyModel.algorithm == m.algorithm).values(active=False))
            m.active = True
            s.commit()
            return m

    # --- configs ----------------------------------------------------------------------------------
    def active_config(self) -> AnomalyConfig | None:
        with self.sessions() as s:
            return s.scalar(select(AnomalyConfig).where(AnomalyConfig.active.is_(True)))

    def add_config(self, content: str, config_hash: str, by: str | None, note: str | None) -> AnomalyConfig:
        with self.sessions() as s:
            version = (s.scalar(select(func.max(AnomalyConfig.version))) or 0) + 1
            s.execute(update(AnomalyConfig).values(active=False))
            c = AnomalyConfig(version=version, content=content, config_hash=config_hash, active=True, created_by=by,
                              note=note)
            s.add(c)
            s.commit()
            return c

    def configs(self) -> list[AnomalyConfig]:
        with self.sessions() as s:
            return list(s.scalars(select(AnomalyConfig).order_by(AnomalyConfig.version.desc())))
