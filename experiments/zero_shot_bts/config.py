"""Experiment configuration: models, prompts, thresholds. Saved verbatim with every run."""

from dataclasses import asdict, dataclass, field

from app.training.taxonomy import CLASS_NAMES

# Grounding DINO prompts (detection). Short noun phrases ground best; several per class.
GDINO_PROMPTS: dict[int, list[str]] = {
    0: ["telecom antenna", "cellular antenna", "panel antenna"],
    1: ["network router", "telecom router chassis"],
    2: ["coaxial cable", "electrical cable", "cable bundle"],
    3: ["backup battery", "battery module"],
    4: ["power supply unit", "rectifier module"],
    5: ["equipment cabinet", "outdoor telecom cabinet"],
    6: ["remote radio unit", "rru"],
}

# CLIP descriptions (classification). Prompt ensembling: embeddings of all descriptions are averaged per class.
CLIP_DESCRIPTIONS: dict[int, list[str]] = {
    0: ["a photo of a telecom panel antenna", "a photo of a cellular base station antenna",
        "a photo of a white sector antenna on a mast"],
    1: ["a photo of a telecom network router with ports and line cards", "a photo of a rack-mounted network router",
        "a photo of a transmission router chassis"],
    2: ["a photo of a coaxial cable", "a photo of a bundle of electrical cables", "a photo of a feeder cable with a connector"],
    3: ["a photo of a telecom backup battery", "a photo of a rack of battery modules", "a photo of a lead-acid battery block"],
    4: ["a photo of a telecom power supply unit", "a photo of a rectifier module", "a photo of a DC power shelf"],
    5: ["a photo of a telecom equipment cabinet", "a photo of an outdoor base station cabinet",
        "a photo of an open equipment enclosure"],
    6: ["a photo of a remote radio unit", "a photo of a finned metal RRU box with connectors",
        "a photo of a base station radio unit"],
}

# Negative / "none of the seven" descriptions: if one of these wins, the crop is UNKNOWN instead of being
# forced into a component class.
CLIP_NEGATIVES: list[str] = [
    "a photo of a wall", "a photo of the sky", "a photo of a person", "a photo of a tree",
    "a screenshot of text", "a photo of a floor", "a blurry photo of nothing in particular",
    "a photo of a cardboard box", "a photo of a hand",
]

UNKNOWN, UNCERTAIN = "UNKNOWN", "UNCERTAIN"


@dataclass
class Config:
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    clip_model: str = "openai/clip-vit-base-patch32"
    sam_weights: str = "mobile_sam.pt"  # Ultralytics MobileSAM; only used with --sam
    use_clip: bool = True
    use_sam: bool = False
    letterbox_crop: bool = True  # run detection on the photo inside phone screenshots
    gdino_short_side: int = 512
    gdino_long_side: int = 896
    box_threshold: float = 0.25  # Grounding DINO detection confidence (max token probability)
    nms_iou: float = 0.5  # class-agnostic duplicate suppression across prompts
    min_box_px: float = 8.0
    crop_padding: float = 0.05  # context added around a box before CLIP
    clip_min_similarity: float = 0.20  # below this the best class is UNCERTAIN
    clip_min_margin: float = 0.005  # top1 - top2 below this is UNCERTAIN
    sam_min_fill: float = 0.15  # mask must cover this fraction of the box to replace it
    batch_size: int = 8  # images per progress/flush step
    gdino_prompts: dict = field(default_factory=lambda: {k: list(v) for k, v in GDINO_PROMPTS.items()})
    clip_descriptions: dict = field(default_factory=lambda: {k: list(v) for k, v in CLIP_DESCRIPTIONS.items()})
    clip_negatives: list = field(default_factory=lambda: list(CLIP_NEGATIVES))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_names"] = dict(CLASS_NAMES)
        return d
