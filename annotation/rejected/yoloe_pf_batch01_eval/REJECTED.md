# Rejected candidate set: yoloe_pf_batch01_eval

Rejected on 2026-11-01. **Do not import `cvat_import/*.zip` into CVAT.**

The prompt-free YOLOE (yoloe-11l-seg-pf.pt, 4585-term built-in vocabulary) does not recognise the
Digital Twin components:

- The folder's own component was proposed in only 10/175 images (1/175 at confidence >= 0.5).
- 116/175 images had no candidate at all; 0/175 qualified as high confidence.
- Most frequent raw labels: "darkness", "storage box" (batteries), "bottle" (cables), "screw", "hinge".
- Sample (IMG_6557, cable folder): "connector" boxes land on a junction box and clamps; the cables,
  antenna and radio unit are missed.

Pre-annotating with these boxes would cost reviewers more time than drawing boxes from scratch.
Kept for traceability; the marker file still blocks it from validation/training.
