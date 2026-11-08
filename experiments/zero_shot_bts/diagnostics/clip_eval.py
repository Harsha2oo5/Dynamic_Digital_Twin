"""CLIP re-scoring for the diagnostics: prompt sets, crop variants, cached embeddings, scoring.

The pipeline's ClipClassifier is not modified or reused; this module loads the same checkpoint separately so
crops and prompts can be varied while the Grounding DINO boxes stay exactly as recorded."""

import numpy as np

from app.training.taxonomy import CLASS_NAMES
from experiments.zero_shot_bts.config import CLIP_DESCRIPTIONS, CLIP_NEGATIVES

CLASSES = list(CLASS_NAMES.values())

READABLE = {"antenna": "antenna", "router": "network router", "cable": "cable", "battery": "battery",
            "power_supply_unit": "power supply unit", "equipment_cabinet": "equipment cabinet", "radio_unit": "radio unit"}

SHORT = {
    "antenna": ["a photo of an antenna", "a photo of a panel antenna", "a photo of a cellular antenna"],
    "router": ["a photo of a router", "a photo of a network router", "a photo of a telecom router"],
    "cable": ["a photo of a cable", "a photo of a coaxial cable", "a photo of cables"],
    "battery": ["a photo of a battery", "a photo of a battery module", "a photo of a backup battery"],
    "power_supply_unit": ["a photo of a power supply", "a photo of a rectifier", "a photo of a power module"],
    "equipment_cabinet": ["a photo of a cabinet", "a photo of an equipment cabinet", "a photo of a telecom cabinet"],
    "radio_unit": ["a photo of a radio unit", "a photo of an RRU", "a photo of a remote radio head"],
}

DETAILED = {
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

ORIGINAL = {CLASS_NAMES[k]: v for k, v in CLIP_DESCRIPTIONS.items()}
NAME_ONLY = {c: [f"a photo of a {READABLE[c]}"] for c in CLASSES}
ALL = {c: ORIGINAL[c] + SHORT[c] + DETAILED[c] for c in CLASSES}

# (name, prompt set, aggregation). "mean" = class-level ensembling (average of normalised prompt embeddings, as
# the pipeline does); "max" = best single prompt per class (no ensembling).
PROMPT_CONFIGS = [
    ("name_only", NAME_ONLY, "mean"),
    ("short_ensemble", SHORT, "mean"),
    ("original_ensemble", ORIGINAL, "mean"),  # = the pipeline
    ("original_max", ORIGINAL, "max"),
    ("detailed_ensemble", DETAILED, "mean"),
    ("detailed_max", DETAILED, "max"),
    ("all_ensemble", ALL, "mean"),
]
PADDINGS = [0.0, 0.05, 0.10, 0.20]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def class_similarities(img: np.ndarray, prompt_embs: dict[str, np.ndarray], how: str) -> np.ndarray:
    """img: (N, D) normalised image embeddings; prompt_embs: class -> (P, D) normalised text embeddings.
    Returns (N, 7) cosine similarities in CLASSES order."""
    cols = []
    for c in CLASSES:
        p = prompt_embs[c]
        if how == "mean":
            cols.append(img @ normalize(p.mean(axis=0)))
        elif how == "max":
            cols.append((img @ p.T).max(axis=1))
        else:
            raise ValueError(how)
    return np.stack(cols, axis=1)


def rankings(sims: np.ndarray) -> list[list[str]]:
    return [[CLASSES[j] for j in np.argsort(-row, kind="stable")] for row in sims]


def decided_share(sims: np.ndarray, neg: np.ndarray, min_sim: float = 0.20, min_margin: float = 0.005) -> dict:
    """Shares of UNKNOWN / UNCERTAIN / decided under the pipeline's rules (for comparing prompt sets)."""
    s = np.sort(sims, axis=1)
    top1, margin = s[:, -1], s[:, -1] - s[:, -2]
    unknown = neg > top1
    uncertain = ~unknown & ((top1 < min_sim) | (margin < min_margin))
    n = len(sims)
    return {"unknown": round(float(unknown.mean()), 4) if n else None,
            "uncertain": round(float(uncertain.mean()), 4) if n else None,
            "decided": round(float((~unknown & ~uncertain).mean()), 4) if n else None,
            "median_margin": round(float(np.median(margin)), 4) if n else None}


class ClipEmbedder:
    def __init__(self, model_id: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_grad_enabled(False)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).eval()

    def text(self, texts: list[str]) -> np.ndarray:
        inp = self.processor(text=texts, return_tensors="pt", padding=True)
        return normalize(self.model.get_text_features(**inp).numpy())

    def images(self, crops_bgr: list[np.ndarray], batch: int = 32) -> np.ndarray:
        import cv2
        from PIL import Image

        out = []
        for i in range(0, len(crops_bgr), batch):
            pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr[i:i + batch]]
            out.append(self.model.get_image_features(**self.processor(images=pils, return_tensors="pt")).numpy())
        return normalize(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

    def prompt_embeddings(self, prompts: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return {c: self.text(prompts[c]) for c in CLASSES}


NEGATIVES = list(CLIP_NEGATIVES)
