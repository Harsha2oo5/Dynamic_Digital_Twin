"""The Digital Twin component taxonomy: the single source of truth for class IDs and names.

The folder mapping is confirmed by the project team. Folder names identify which
component an image was *collected for*; they are not bounding boxes.
"""

from collections.abc import Mapping

# YOLO class id -> class name (order is the label-id order; never reorder existing entries)
CLASS_NAMES: dict[int, str] = {
    0: "antenna",
    1: "router",
    2: "cable",
    3: "battery",
    4: "power_supply_unit",
    5: "equipment_cabinet",
    6: "radio_unit",
}

# dataset folder (relative to the dataset root, first path component) -> class id
FOLDER_TO_CLASS: dict[str, int] = {
    "antenna": 0,
    "Archive (7)": 1,
    "Archive (6)": 2,
    "Archive (5)": 3,
    "Archive (4)": 4,
    "Equipment Cabinet": 5,
    "radio unit": 6,
}

NUM_CLASSES = len(CLASS_NAMES)
CLASS_IDS: dict[str, int] = {name: cid for cid, name in CLASS_NAMES.items()}


def class_for_folder(top_level_folder: str) -> int | None:
    return FOLDER_TO_CLASS.get(top_level_folder)


def names_match_taxonomy(names: Mapping[int, str] | list[str]) -> bool:
    """True if a model's / data.yaml's class names equal the taxonomy exactly (same ids, same names)."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()} == CLASS_NAMES
