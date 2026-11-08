"""Track lifecycle, session-level equipment identity, inventory de-duplication, model management and the
pseudo-label training path. Synthetic moving-box sequences, tmp_path only; the real model is used only where
an API round trip is checked."""

import json
import time
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.identification.equipment_identity import (
    EquipmentIdentity,
    IdentityConfig,
    appearance_histogram,
    appearance_similarity,
    class_abbreviation,
)
from app.main import create_app
from app.schemas.detection import FrameResult
from app.services.object_registry import ObjectRegistry
from app.training import taxonomy
from app.training.prepare_dataset import PSEUDO_DATASET_MARKER, check_pseudo_run, prepare_pseudo
from app.training.pseudo_label.outputs import MANIFEST, MARKER
from app.training.review_store import label_sha256
from app.training.train import build_model_card, check_training_ready, promote_versioned
from app.vision.detector import RawDetections
from app.vision.processor import build_detections
from app.vision.track_manager import ACTIVE, LOST, NEW, TrackingSession, TrackManager
from app.vision.tracker import ObjectTracker, load_tracker_args
from tests.conftest import make_settings
from tests.test_training_pipeline import dataset, kit  # noqa: F401  (fixtures)

NAMES = {0: "radio_unit", 1: "antenna", 2: "battery"}
TRACKER_CFG = "configs/trackers/bts_bytetrack.yaml"


def raw(boxes, confs, classes, w=640, h=480) -> RawDetections:
    return RawDetections(xyxy=np.array(boxes, np.float32).reshape(-1, 4), conf=np.array(confs, np.float32),
                         cls=np.array(classes, int), frame_width=w, frame_height=h, inference_ms=10.0)


def frame_with(boxes_colors, w=640, h=480) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        img[int(y1):int(y2), int(x1):int(x2)] = color
    return img


def session(sid="s1", frame_rate=10.0, **kw) -> TrackingSession:
    from app.core.config import BACKEND_DIR

    return TrackingSession(load_tracker_args(str(BACKEND_DIR / TRACKER_CFG)), sid, NAMES, frame_rate=frame_rate, **kw)


def run_sequence(sess, boxes_per_frame, classes=None, dt=0.1, start=0.0, color=(0, 0, 220)):
    """Feed frames; boxes_per_frame[i] is a list of boxes (possibly empty). Returns infos per frame."""
    out = []
    for i, boxes in enumerate(boxes_per_frame):
        cls = classes[i] if classes else [0] * len(boxes)
        r = raw(boxes or np.zeros((0, 4)), [0.85] * len(boxes), cls)
        frame = frame_with([(b, color) for b in boxes])
        out.append(sess.step(r, frame, start + i * dt))
    return out


def moving(n, x0=100, y0=100, size=80, step=4):
    return [[[x0 + i * step, y0, x0 + i * step + size, y0 + size]] for i in range(n)]


# --- lifecycle ---------------------------------------------------------------------------------------------
def test_lifecycle_new_active_lost_removed():
    sess = session(lost_timeout_s=1.0)
    infos = run_sequence(sess, moving(6))
    states = [inf[1][0].track_state for inf in infos]
    assert states[:2] == [NEW, NEW] and states[2:] == [ACTIVE] * 4  # min_hits = 3
    assert infos[1][1][0].equipment_id is None and infos[2][1][0].equipment_id is not None
    # disappears: lost inside the timeout, removed after it
    run_sequence(sess, [[]] * 3, start=0.6)
    tid = next(iter(sess.manager.tracks))
    assert sess.manager.tracks[tid].state == LOST
    run_sequence(sess, [[]] * 2, start=2.0)
    assert not sess.manager.tracks and sess.counts()["removed"] == 1


def test_tentative_track_never_gets_an_equipment_id():
    sess = session(lost_timeout_s=0.5)
    infos = run_sequence(sess, moving(2) + [[]] * 10)
    assert all(i.equipment_id is None for f in infos for i in f[1] if i)
    assert sess.counts()["equipment_items"] == 0


def test_one_moving_item_keeps_one_equipment_id():
    sess = session()
    infos = run_sequence(sess, moving(30, step=6))
    ids = {i.equipment_id for f in infos for i in f[1] if i and i.equipment_id}
    assert len(ids) == 1 and next(iter(ids)).endswith("@s1") and next(iter(ids)).startswith("RU-")


def test_class_vote_smooths_flicker():
    sess = session()
    classes = [[0], [0], [1], [0], [0], [1], [0], [0]]  # radio unit, flickering to antenna
    infos = run_sequence(sess, moving(8), classes=classes)
    last = infos[-1][1][0]
    assert NAMES[last.stable_class_id] == "radio_unit" and 0.6 < last.class_consistency < 0.9
    dets = build_detections(raw([[1, 2, 3, 4]], [0.5], [1]), NAMES, np.array([last.track_id]), None, [last])
    assert dets[0].class_name == "antenna" and dets[0].stable_class == "radio_unit"  # raw class kept as-is


def test_sessions_are_isolated():
    a, b = session("aaaa"), session("bbbb")
    ia = run_sequence(a, moving(5))
    ib = run_sequence(b, moving(5))
    ea, eb = ia[-1][1][0].equipment_id, ib[-1][1][0].equipment_id
    assert ea == "RU-001@aaaa" and eb == "RU-001@bbbb"
    run_sequence(a, [[]] * 50, start=1.0)  # session a losing its item does not touch b
    assert b.manager.tracks and b.manager.tracks[next(iter(b.manager.tracks))].state == ACTIVE


def test_relink_after_tracker_id_change():
    """Item leaves the frame for longer than the tracker buffer and comes back: a new tracker ID, same item."""
    sess = session()  # tracker buffer ~3 s
    first = run_sequence(sess, moving(5))
    eid, tid = first[-1][1][0].equipment_id, first[-1][1][0].track_id
    run_sequence(sess, [[]] * 40, start=0.5)  # 4 s gap > buffer -> the tracker drops the track
    back = run_sequence(sess, [[[130, 100, 210, 180]]] * 5, start=4.5)
    info = back[-1][1][0]
    assert info.track_id != tid and info.equipment_id == eid and info.relinked
    assert sess.counts()["relinks"] == 1 and sess.counts()["equipment_items"] == 1


def test_no_relink_for_other_class_or_far_dissimilar_item():
    ident = EquipmentIdentity("s", IdentityConfig(relink_window_s=10))
    red = appearance_histogram(frame_with([((0, 0, 50, 50), (0, 0, 220))]), np.array([0, 0, 50, 50]))
    green = appearance_histogram(frame_with([((0, 0, 50, 50), (0, 200, 0))]), np.array([0, 0, 50, 50]))
    assert appearance_similarity(red, red) == pytest.approx(1.0) and appearance_similarity(red, green) < 0.2
    e1, _ = ident.assign("radio_unit", np.array([0, 0, 50, 50]), red, 640, 480, 0.0)
    ident.lost(e1, "radio_unit", np.array([0, 0, 50, 50]), red, 1.0)
    e2, rel = ident.assign("antenna", np.array([0, 0, 50, 50]), red, 640, 480, 2.0)
    assert not rel and e2.startswith("ANT-")  # other class
    e3, rel = ident.assign("radio_unit", np.array([500, 400, 550, 450]), green, 640, 480, 2.0)
    assert not rel  # far away and different colour
    e4, rel = ident.assign("radio_unit", np.array([10, 0, 60, 50]), red, 640, 480, 3.0)
    assert rel and e4 == e1
    ident.lost(e4, "radio_unit", np.array([10, 0, 60, 50]), red, 3.0)
    e5, rel = ident.assign("radio_unit", np.array([10, 0, 60, 50]), red, 640, 480, 30.0)
    assert not rel  # outside the re-link window


def test_class_abbreviations():
    assert class_abbreviation("power_supply_unit") == "PSU" and class_abbreviation("person") == "PER"


def test_tracker_buffer_is_scaled_to_time():
    args = load_tracker_args(TRACKER_CFG)
    assert ObjectTracker(args, frame_rate=30)._scaled_args().track_buffer == args.track_buffer
    assert ObjectTracker(args, frame_rate=5)._scaled_args().track_buffer == round(args.track_buffer * 5 / 30)
    assert ObjectTracker(args).buffer_s == pytest.approx(args.track_buffer / 30)


def test_botsort_config_loads_and_tracks():
    sess_args = load_tracker_args("configs/trackers/bts_botsort.yaml")
    t = ObjectTracker(sess_args, frame_rate=10)
    ids = [t.update(raw(b, [0.9], [0]), frame_with([(b[0], (0, 0, 200))]))[0] for b in moving(6)]
    assert ids[-1] >= 0 and len({i for i in ids if i >= 0}) == 1


# --- inventory ------------------------------------------------------------------------------------------------
def _frame(dets, ts) -> FrameResult:
    return FrameResult(frame_id=0, timestamp=ts, device="cpu", fps=10, inference_ms=100, frame_width=640,
                       frame_height=480, detections=dets)


def test_registry_deduplicates_by_equipment_id():
    sess = session("vid1")
    reg = ObjectRegistry(lost_after_s=2.0)
    reg.source_started("video:vid1", "video", "clip", sess)
    t0 = datetime.now() - timedelta(seconds=1)
    seq = moving(5) + [[]] * 40 + [[[130, 100, 210, 180]]] * 5
    for i, boxes in enumerate(seq):
        r = raw(boxes or np.zeros((0, 4)), [0.8] * len(boxes), [0] * len(boxes))
        ids, infos = sess.step(r, frame_with([(b, (0, 0, 220)) for b in boxes]), i * 0.1)
        reg.update("video:vid1", "video", _frame(build_detections(r, NAMES, ids, None, infos),
                                                 t0 + timedelta(seconds=i * 0.04)))
    objs = reg.list_objects()
    assert len(objs) == 1  # one item despite two tracker IDs; tentative frames not registered
    o = objs[0]
    assert o.equipment_id == "RU-001@vid1" and len(o.track_ids) == 2 and o.relinked == 1
    assert o.object_id == "video:vid1:RU-001@vid1" and reg.get(o.object_id) is not None
    assert o.class_name == "radio_unit" and o.duration_s >= 0 and o.track_state == "active"
    assert reg.list_objects(class_name="antenna") == [] and len(reg.list_objects(session="vid1")) == 1
    summary = reg.summary()
    assert summary.total_items == 1 and summary.by_class[0]["class_name"] == "radio_unit"
    s = reg.sessions()[0]
    assert s.equipment_items == 1 and s.live and s.tracking["relinks"] == 1
    reg.source_stopped("video:vid1")
    assert reg.get(o.object_id).track_state == "removed" and not reg.sessions()[0].live


# --- pseudo-label training path ----------------------------------------------------------------------------------
def _pseudo_run(tmp_path, manifest, skip_every=7):
    run = tmp_path / "pseudo_run"
    (run / "labels").mkdir(parents=True)
    (run / MARKER).write_text("not ground truth")
    (run / "classes.txt").write_text("\n".join(taxonomy.CLASS_NAMES.values()) + "\n")
    images = {}
    for n, (key, row) in enumerate(sorted(manifest.items())):
        trainable = n % skip_every != 0
        entry = {"file": row["file_name"], "original_path": row["original_path"], "trainable": trainable}
        if trainable:
            text = f"{row['folder_class_id']} 0.5 0.5 0.4 0.4\n"
            (run / "labels" / f"{key}.txt").write_text(text, encoding="utf-8")
            entry["label_sha256"] = label_sha256(text)
        images[key] = entry
    (run / MANIFEST).write_text(json.dumps({"label_source": "pseudo", "ground_truth": False, "run": "t",
                                            "config_hash": "abc", "disclaimer": "NOT ground truth",
                                            "images": images}))
    return run


def test_prepare_pseudo_end_to_end(tmp_path, dataset, kit):  # noqa: F811
    run = _pseudo_run(tmp_path, kit)
    out = tmp_path / "ds"
    s = prepare_pseudo(run, tmp_path / "manifest.csv", dataset, out, strict=False)
    assert s["label_source"] == "pseudo" and s["ground_truth"] is False and s["pseudo_run"] == "t"
    assert (out / PSEUDO_DATASET_MARKER).is_file() and s["post_build_problems"] == []
    n_train = sum(1 for e in json.loads((run / MANIFEST).read_text())["images"].values() if e["trainable"])
    assert sum(s["images"].values()) == n_train  # only trainable images
    data = out / "data.yaml"
    assert any("pseudo-labels" in p for p in check_training_ready(data))  # refused without opt-in
    assert check_training_ready(data, allow_pseudo=True) == []
    victim = next((out / "labels" / "train").glob("*.txt"))
    victim.write_text("0 0.5 0.5 0.1 0.1\n")
    assert any("modified" in p for p in check_training_ready(data, allow_pseudo=True))


def test_pseudo_run_tampering_is_refused(tmp_path, dataset, kit):  # noqa: F811
    run = _pseudo_run(tmp_path, kit)
    label = next((run / "labels").glob("*.txt"))
    label.write_text("1 0.5 0.5 0.2 0.2\n")
    _, _, errors = check_pseudo_run(run, kit)
    assert any("modified after generation" in e for e in errors)
    (run / MARKER).unlink()
    assert "not a pseudo-label run" in check_pseudo_run(run, kit)[2][0]


def test_model_card_records_pseudo_provenance(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    summary = {"run": "r", "mode": "pilot", "results": {"best": {"metrics/mAP50(B)": 0.4}}, "train_args": {}}
    prov = {"label_source": "pseudo", "ground_truth": False, "pseudo_run": "full_v1"}
    t1 = promote_versioned(best, summary, prov, tmp_path / "data.yaml", models_dir=tmp_path / "custom")
    t2 = promote_versioned(best, summary, prov, tmp_path / "data.yaml", models_dir=tmp_path / "custom")
    assert (t1.name, t2.name) == ("digital_twin_v1.pt", "digital_twin_v2.pt")
    card = json.loads(t1.with_suffix(".model_card.json").read_text())
    assert card["label_source"] == "pseudo" and card["ground_truth"] is False and card["version"] == 1
    assert "NOT accuracy" in card["metrics_meaning"] and card["validation_metrics"] == {"mAP50": 0.4}
    assert any("pseudo-labels" in x for x in card["limitations"])
    assert build_model_card(best, {}, {"label_source": "reviewed", "ground_truth": True},
                            tmp_path)["limitations"] == []


# --- API: model management, sessions, inventory -------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("lifecycle")))
    with TestClient(app) as c:
        yield c


def test_model_versions_and_select(client):
    versions = client.get("/api/model/versions").json()
    pre = next(v for v in versions if v["name"] == "yolo11n.pt")
    assert pre["active"] and pre["label_source"] == "coco" and pre["model_type"] == "pretrained"
    r = client.post("/api/model/select", json={"model_id": pre["model_id"]})
    assert r.status_code == 200 and r.json()["active"]
    assert client.post("/api/model/select", json={"model_id": "../../etc/passwd.pt"}).status_code == 422
    assert client.post("/api/model/select", json={"model_id": "custom/missing.pt"}).status_code == 422
    assert client.get("/api/model/info").json()["model_name"] == "yolo11n.pt"  # unchanged after failures
    perf = client.get("/api/model/performance").json()
    assert perf["label_source"] == "coco" and "runtime" in perf


def test_broken_checkpoint_keeps_current_model(client, tmp_path):
    models_dir = client.app.state.service.models.models_dir
    broken = models_dir / "custom" / "zz_broken_test.pt"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not a checkpoint")
    try:
        r = client.post("/api/model/select", json={"model_id": "custom/zz_broken_test.pt"})
        assert r.status_code == 422 and "stays active" in r.json()["detail"]
        assert client.get("/api/health").json()["model_loaded"] is True
    finally:
        broken.unlink()


def test_video_produces_session_identity_and_inventory(client, bus_video):
    with bus_video.open("rb") as f:
        job = client.post("/api/detect/video", files={"file": ("pan.mp4", f, "video/mp4")}).json()
    for _ in range(600):
        st = client.get(f"/api/detect/video/{job['job_id']}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "completed"
    frames = client.get(f"/api/detect/video/{job['job_id']}/results").json()["frames"]
    bus = [d for f in frames for d in f["detections"] if d["class_name"] == "bus"]
    assert bus[0]["track_state"] == "new" and bus[-1]["track_state"] == "active"
    assert len({d["equipment_id"] for d in bus if d["equipment_id"]}) == 1
    sid = job["job_id"]
    tracks = client.get(f"/api/sessions/{sid}/tracks").json()
    assert tracks and all(t["equipment_id"].endswith(f"@{sid}") for t in tracks)
    assert sum(t["class_name"] == "bus" for t in tracks) == 1
    sessions = client.get("/api/sessions").json()
    assert any(s["source_id"] == f"video:{job['job_id']}" and not s["live"] for s in sessions)
    summary = client.get("/api/inventory/summary").json()
    assert summary["total_items"] >= len(tracks) and "not permanent" in summary["note"]
    assert client.get("/api/objects", params={"class": "bus", "session": sid}).json()
    assert client.get("/api/sessions/nope/tracks").status_code == 404
