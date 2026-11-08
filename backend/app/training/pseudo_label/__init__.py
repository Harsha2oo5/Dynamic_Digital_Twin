"""Automated pseudo-labelling: Grounding DINO regions -> CLIP classification -> evidence fusion -> YOLO labels.

Pseudo-labels are machine-generated and are NOT ground truth. Every box keeps its full provenance (both models'
raw outputs, the fusion terms, the decision and its reason), and every run writes PSEUDO_LABELS_NOT_GROUND_TRUTH.txt
plus a hashed pseudo_manifest.json. Training on them is a separate, explicitly labelled path
(`prepare_dataset --pseudo`), and models trained that way are tagged `label_source: pseudo`.

Usage (from backend/):
    python -m app.training.pseudo_label.run --name smoke_21 --limit 21
    python -m app.training.pseudo_label.run --name full_v1                  # all images of annotation/manifest.csv
"""
