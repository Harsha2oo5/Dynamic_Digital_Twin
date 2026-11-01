import cv2
import numpy as np

from app.training.dataset_validator import parse_yolo_label, validate_dataset


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((20, 30, 3), np.uint8))


def test_unlabelled_folder_per_class_dataset(tmp_path):
    _img(tmp_path / "antenna" / "a.png")
    _img(tmp_path / "radio" / "b.jpg")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "._a.png").write_bytes(b"meta")
    (tmp_path / "radio" / "._b.jpg").write_bytes(b"meta")

    report = validate_dataset(tmp_path)
    assert report["total_images"] == 2
    assert report["labelled_images"] == 0
    assert report["annotation_format"] == "none"
    assert report["macos_metadata_files"] == 2
    assert report["folders"][0]["min_resolution"] == [30, 20]
    assert not report["ready_for_detection_training"]


def test_yolo_dataset_with_errors_and_duplicates(tmp_path):
    for name in ("x", "y"):
        _img(tmp_path / "images" / "train" / f"{name}.jpg")  # identical bytes -> duplicates
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "labels" / "train" / "y.txt").write_text("3 0.5 0.5 0.2 1.4\n")
    (tmp_path / "data.yaml").write_text("names: [a, b]\n")

    report = validate_dataset(tmp_path)
    assert report["annotation_format"] == "yolo"
    assert report["labelled_images"] == 2
    assert len(report["duplicate_groups"]) == 1
    assert len(report["label_errors"]) == 2  # class out of range + coordinate > 1
    assert not report["ready_for_detection_training"]


def test_parse_yolo_label(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("1 0.1 0.2 0.3 0.4\nbad line\n")
    boxes, errors = parse_yolo_label(p, num_classes=2)
    assert boxes == [(1, 0.1, 0.2, 0.3, 0.4)]
    assert len(errors) == 1
