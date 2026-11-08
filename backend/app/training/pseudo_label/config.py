"""Pseudo-labelling configuration: defaults, YAML overrides, and a stable hash saved with every run.

Defaults are informed by the zero-shot diagnostics (experiments/zero_shot_bts/diagnostics/class_wise_report.md):
  - CLIP prompt set `detailed_ensemble` and 20 % crop padding gave the highest folder consistency;
  - Grounding DINO's own class matched the image's folder for only ~16 % of primary boxes, so it is a weak vote;
  - boxes under 5 % of the photo matched their folder at chance level (14.8 %), so they need stronger evidence.
None of these numbers are accuracy; there is no ground truth.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from app.core.config import BACKEND_DIR, PROJECT_DIR
from app.training.taxonomy import CLASS_NAMES

CLASSES = list(CLASS_NAMES.values())
DEFAULT_CONFIG_FILE = BACKEND_DIR / "configs" / "pseudo_label" / "default.yaml"

# Grounding DINO prompts (same as the stored zero-shot run, so stored detections stay comparable).
GDINO_PROMPTS: dict[str, list[str]] = {
    "antenna": ["telecom antenna", "cellular antenna", "panel antenna"],
    "router": ["network router", "telecom router chassis"],
    "cable": ["coaxial cable", "electrical cable", "cable bundle"],
    "battery": ["backup battery", "battery module"],
    "power_supply_unit": ["power supply unit", "rectifier module"],
    "equipment_cabinet": ["equipment cabinet", "outdoor telecom cabinet"],
    "radio_unit": ["remote radio unit", "rru"],
}

# CLIP descriptions: the `detailed_ensemble` set from the diagnostics (best folder consistency).
CLIP_PROMPTS: dict[str, list[str]] = {
    "antenna": ["a white panel sector antenna of a cellular base station",
                "a tall flat telecom antenna mounted on a mast or rooftop",
                "an omnidirectional or ceiling dome antenna with a coaxial connector",
                "a BTS antenna used for mobile network coverage"],
    "router": ["a rack-mounted telecom router chassis with network ports and line cards",
               "a cell site transmission router with fibre and ethernet interfaces",
               "a carrier-grade network router used at a base station",
               "a metal router box with many small network ports and status LEDs"],
    "cable": ["a coaxial feeder cable with a connector used in a base station",
              "a bundle of black power and fibre cables",
              "a jumper cable connecting telecom equipment",
              "a coiled telecom cable on a white background"],
    "battery": ["a telecom backup battery module with positive and negative terminals",
                "a lithium battery pack for base station power backup",
                "a row of lead-acid batteries in a telecom cabinet",
                "a rack-mounted battery unit with handles"],
    "power_supply_unit": ["a telecom rectifier power supply module", "a DC power supply unit for a base station",
                          "telecom power distribution equipment with breakers",
                          "a power shelf with several hot-swappable rectifier modules and cooling fans"],
    "equipment_cabinet": ["a large outdoor telecom equipment cabinet with doors",
                          "a metal enclosure that houses base station equipment",
                          "an open telecom cabinet with racks of equipment inside",
                          "a standing indoor equipment rack cabinet"],
    "radio_unit": ["a remote radio unit used in a cellular base station", "an outdoor telecom radio frequency unit",
                   "a BTS remote radio head with cooling fins", "an RRU mounted near a cellular antenna"],
}

# A second, independent prompt set (the original pipeline's). Used only to measure prompt stability.
CLIP_ALT_PROMPTS: dict[str, list[str]] = {
    "antenna": ["a photo of a telecom panel antenna", "a photo of a cellular base station antenna",
                "a photo of a white sector antenna on a mast"],
    "router": ["a photo of a telecom network router with ports and line cards", "a photo of a rack-mounted network router",
               "a photo of a transmission router chassis"],
    "cable": ["a photo of a coaxial cable", "a photo of a bundle of electrical cables",
              "a photo of a feeder cable with a connector"],
    "battery": ["a photo of a telecom backup battery", "a photo of a rack of battery modules",
                "a photo of a lead-acid battery block"],
    "power_supply_unit": ["a photo of a telecom power supply unit", "a photo of a rectifier module",
                          "a photo of a DC power shelf"],
    "equipment_cabinet": ["a photo of a telecom equipment cabinet", "a photo of an outdoor base station cabinet",
                          "a photo of an open equipment enclosure"],
    "radio_unit": ["a photo of a remote radio unit", "a photo of a finned metal RRU box with connectors",
                   "a photo of a base station radio unit"],
}

CLIP_NEGATIVES: list[str] = [
    "a photo of a wall", "a photo of the sky", "a photo of a person", "a photo of a tree", "a screenshot of text",
    "a photo of a floor", "a blurry photo of nothing in particular", "a photo of a cardboard box", "a photo of a hand",
]


def _per_class(value: float) -> dict[str, float]:
    return {c: value for c in CLASSES}


@dataclass
class PseudoLabelConfig:
    # --- inputs ---
    dataset_root: str = str(PROJECT_DIR / "digital twin project datasets")
    manifest: str = str(PROJECT_DIR / "annotation" / "manifest.csv")
    # "stored": reuse Grounding DINO boxes of an earlier zero-shot run; "live": run Grounding DINO now
    detections: str = "stored"
    stored_run: str = str(PROJECT_DIR / "experiments" / "zero_shot_bts" / "runs" / "full_543_gdino_clip")
    # --- models (all CPU, all already cached on the dev machine) ---
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    clip_model: str = "openai/clip-vit-base-patch32"
    sam_weights: str = str(PROJECT_DIR / "experiments" / "zero_shot_bts" / "weights" / "mobile_sam.pt")
    use_sam: bool = False
    # --- Grounding DINO (live mode) ---
    gdino_short_side: int = 512
    gdino_long_side: int = 896
    letterbox_crop: bool = True
    box_threshold: float = 0.25  # the stored run used 0.25, so lower values only apply in live mode
    proposal_nms_iou: float = 0.5  # class-agnostic suppression across prompts
    gdino_prompts: dict = field(default_factory=lambda: {k: list(v) for k, v in GDINO_PROMPTS.items()})
    # --- CLIP ---
    clip_prompts: dict = field(default_factory=lambda: {k: list(v) for k, v in CLIP_PROMPTS.items()})
    clip_alt_prompts: dict = field(default_factory=lambda: {k: list(v) for k, v in CLIP_ALT_PROMPTS.items()})
    clip_negatives: list = field(default_factory=lambda: list(CLIP_NEGATIVES))
    crop_padding: float = 0.20
    clip_logit_scale: float = 100.0  # CLIP's own temperature: probabilities = softmax(scale * cosine)
    clip_support_top_k: int = 3  # the final class must be among CLIP's top-k
    background_margin: float = 0.0  # reject when best negative similarity > best class similarity + margin
    # --- evidence fusion: log p = w_clip log p_clip + w_gdino log p_gdino + log prior ---
    w_clip: float = 1.0
    w_gdino: float = 0.5
    gdino_vote_smoothing: float = 0.05
    # Folder prior: a SOFT multiplicative boost for the image's folder class, scaled up with box size
    # (large boxes in these single-product photos are usually the product). Never the sole evidence:
    # CLIP must still rank the class in its top-k. Set folder_prior_boost to 1.0 to switch it off.
    folder_prior_boost: float = 2.0
    folder_prior_full_area: float = 0.20  # box share of the photo at which the full boost applies
    # Visual prototypes (prototypes.py): CLIP image embeddings of large "whole product" boxes per folder,
    # leave-near-duplicate-group-out. Folders act as image-level weak labels here as well.
    use_prototypes: bool = True
    w_proto: float = 2.0
    proto_logit_scale: float = 30.0
    proto_anchor_min_area: float = 0.20
    proto_min_anchors: int = 5
    # --- filtering ---
    min_area_share: float = 0.003  # of the photo; smaller boxes are rejected
    tiny_area_share: float = 0.05  # boxes below this need `tiny_extra_confidence` more
    tiny_extra_confidence: float = 0.10
    min_gdino_confidence: dict = field(default_factory=lambda: _per_class(0.27))
    min_final_confidence: dict = field(default_factory=lambda: _per_class(0.45))
    min_final_margin: float = 0.10
    require_flip_stable: bool = True  # the class must survive a horizontal flip of the crop
    # --- suppression ---
    duplicate_iou: float = 0.60  # same class
    conflict_iou: float = 0.80  # different classes, same object
    part_containment: float = 0.85  # a box this much inside a larger accepted box ...
    part_max_area_ratio: float = 0.50  # ... and at most this fraction of its size is a part of it
    container_classes: list = field(default_factory=lambda: ["equipment_cabinet"])  # may contain other components
    # classes exempt from the part rule. Empty by default: a real cable usually leaves the box it plugs into, so
    # only cable boxes >= part_containment inside another object (rack ears, handles) are treated as parts
    allowed_inside_any: list = field(default_factory=list)
    nestable_classes: list = field(default_factory=lambda: ["battery", "power_supply_unit"])  # module in rack/shelf
    offline: bool = True  # load models from the local Hugging Face cache only; never download
    # --- image-level training decision ---
    blocking_uncertain_area: float = 0.05  # an uncovered UNCERTAIN box this large excludes the image
    covered_iou: float = 0.50
    # --- outputs ---
    preview_max_side: int = 900
    write_previews: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Stable hash of every setting that affects the labels (paths and output options excluded)."""
        d = {k: v for k, v in self.to_dict().items()
             if k not in {"dataset_root", "manifest", "stored_run", "preview_max_side", "write_previews", "offline"}}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        problems = []
        if self.detections not in ("stored", "live"):
            problems.append(f"detections must be 'stored' or 'live', got {self.detections!r}")
        for name in ("gdino_prompts", "clip_prompts", "clip_alt_prompts", "min_gdino_confidence",
                     "min_final_confidence"):
            missing = set(CLASSES) - set(getattr(self, name))
            extra = set(getattr(self, name)) - set(CLASSES)
            if missing or extra:
                problems.append(f"{name}: keys must be exactly the seven classes (missing {sorted(missing)}, "
                                f"unknown {sorted(extra)})")
        for name in self.container_classes + self.allowed_inside_any + self.nestable_classes:
            if name not in CLASSES:
                problems.append(f"unknown class {name!r} in container/allowed lists")
        if self.folder_prior_boost < 1.0:
            problems.append("folder_prior_boost must be >= 1 (1 = off)")
        return problems


def load_config(path: Path | None = None, overrides: dict | None = None) -> PseudoLabelConfig:
    """Defaults <- YAML file (if any) <- explicit overrides. Unknown keys are an error."""
    data: dict = {}
    if path is not None:
        data.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    data.update(overrides or {})
    known = {f.name for f in fields(PseudoLabelConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown pseudo-label config keys: {sorted(unknown)}")
    cfg = PseudoLabelConfig()
    for k, v in data.items():
        cur = getattr(cfg, k)
        if isinstance(cur, dict) and isinstance(v, dict) and k.startswith("min_"):
            v = {**cur, **v}  # per-class thresholds may override only some classes
        setattr(cfg, k, v)
    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid pseudo-label config: " + "; ".join(problems))
    return cfg
