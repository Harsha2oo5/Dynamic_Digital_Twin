"""Visual class prototypes from CLIP image embeddings (weak supervision, leave-group-out).

CLIP's text side barely knows BTS equipment (radio units are usually ranked 4th-5th against their own
descriptions), but its image embeddings separate look-alike boxes much better. A class prototype is the mean
embedding of *anchor* boxes: the largest box of each image of that class's folder, if it covers at least
`proto_anchor_min_area` of the photo (in these mostly single-product photos that box is usually the product).

Folders are image-level weak labels, so every image is scored against prototypes built WITHOUT its own
near-duplicate group (leave-group-out): an image never votes for itself or its re-captures.
"""

from collections import defaultdict

import numpy as np

from app.training.pseudo_label.config import CLASSES, PseudoLabelConfig


def anchor_index(entries: list[dict], cfg: PseudoLabelConfig) -> list[tuple[int, int, str, int]]:
    """(entry index, box index, folder class, near-duplicate group) of every anchor box."""
    out = []
    for i, e in enumerate(entries):
        fc = e["row"].get("folder_class_name")
        if fc not in CLASSES or not e["boxes"] or len(e["emb"]) != len(e["boxes"]):
            continue  # no embeddings (e.g. scores reused from a run without a cache): cannot be an anchor
        j = max(range(len(e["boxes"])), key=lambda k: e["boxes"][k]["area_share"])
        if e["boxes"][j]["area_share"] >= cfg.proto_anchor_min_area:
            out.append((i, j, fc, int(e["row"].get("near_duplicate_group", -1))))
    return out


class Prototypes:
    def __init__(self, entries: list[dict], cfg: PseudoLabelConfig):
        """`entries[i]` has "row" (manifest row), "boxes" (with area_share) and "emb" ((N, D) normalised)."""
        self.cfg = cfg
        dim = next((e["emb"].shape[1] for e in entries if len(e["emb"])), 512)
        self.total = {c: np.zeros(dim, np.float64) for c in CLASSES}
        self.count = dict.fromkeys(CLASSES, 0)
        self.by_group: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(dim, np.float64))
        self.group_count: dict[tuple[str, int], int] = defaultdict(int)
        for i, j, c, g in anchor_index(entries, cfg):
            v = entries[i]["emb"][j]
            self.total[c] += v
            self.count[c] += 1
            self.by_group[(c, g)] += v
            self.group_count[(c, g)] += 1

    def anchors(self) -> dict[str, int]:
        return dict(self.count)

    def matrix_for(self, group: int) -> tuple[np.ndarray, list[str]]:
        """Normalised prototypes excluding `group`; classes with too few anchors are left out."""
        rows, names = [], []
        for c in CLASSES:
            n = self.count[c] - self.group_count.get((c, group), 0)
            if n < self.cfg.proto_min_anchors:
                continue
            v = self.total[c] - self.by_group.get((c, group), 0.0)
            rows.append(v / (np.linalg.norm(v) or 1.0))
            names.append(c)
        return (np.stack(rows) if rows else np.zeros((0, len(self.total[CLASSES[0]])))), names

    def similarities(self, emb: np.ndarray, group: int) -> list[dict[str, float] | None]:
        """Per embedding row: cosine similarity to every available prototype (None if fewer than 2 exist)."""
        m, names = self.matrix_for(group)
        if len(names) < 2 or len(emb) == 0:
            return [None] * len(emb)
        sims = emb @ m.T
        return [dict(zip(names, map(float, row))) for row in sims]
