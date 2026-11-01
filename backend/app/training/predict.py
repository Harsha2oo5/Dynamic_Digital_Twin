"""Run a checkpoint on images (CPU): save annotated images and, optionally, YOLO-format box
proposals for *model-assisted annotation*.

Proposals are NOT training labels. Import them into CVAT / Label Studio as pre-annotations and
have a person correct every image before exporting. validate_annotations treats an export like
any other, so unreviewed proposals must never be fed to prepare_dataset directly.

Usage:
    python -m app.training.predict --model models/custom/digital_twin_best.pt --source "../digital twin project datasets/antenna" \
        --out ../experiments/predictions/antenna [--proposals]
"""

import argparse
import json
import sys
from pathlib import Path

from app.core.device import resolve_device
from app.training.dataset_validator import IMAGE_EXTS, is_macos_metadata


def to_yolo_lines(xywhn, cls, conf, min_conf: float) -> list[str]:
    return [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for (x, y, w, h), c, p in zip(xywhn, cls, conf) if p >= min_conf]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a YOLO checkpoint on CPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory (read-only)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--proposals", action="store_true", help="Also write YOLO .txt pre-annotations for review")
    args = parser.parse_args()

    if args.out.exists():
        sys.exit(f"Refusing to overwrite {args.out}")
    files = [args.source] if args.source.is_file() else sorted(
        p for p in args.source.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not is_macos_metadata(p))
    if not files:
        sys.exit("No images found")

    import cv2
    from ultralytics import YOLO

    from app.training.audit_dataset import read_image

    model = YOLO(str(args.model))
    (args.out / "annotated").mkdir(parents=True)
    if args.proposals:
        (args.out / "proposals").mkdir()
        (args.out / "proposals" / "classes.txt").write_text("\n".join(model.names[i] for i in sorted(model.names)) + "\n",
                                                            encoding="utf-8")
    results = []
    for path in files:
        img = read_image(path)
        if img is None:
            print(f"  skipped unreadable {path}")
            continue
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=resolve_device("cpu"), verbose=False)[0]
        cv2.imencode(".jpg", r.plot())[1].tofile(str(args.out / "annotated" / f"{path.stem}.jpg"))
        boxes = r.boxes.cpu().numpy()
        if args.proposals:
            lines = to_yolo_lines(boxes.xywhn, boxes.cls, boxes.conf, args.conf)
            (args.out / "proposals" / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                    encoding="utf-8")
        results.append({"image": str(path), "detections": [
            {"class_name": model.names[int(c)], "confidence": round(float(p), 4), "xyxy": [round(float(v), 1) for v in b]}
            for b, c, p in zip(boxes.xyxy, boxes.cls, boxes.conf)]})
    (args.out / "predictions.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"{len(results)} images processed -> {args.out}")
    if args.proposals:
        print("Proposals are unreviewed pre-annotations: import, correct every image, then export.")


if __name__ == "__main__":
    main()
