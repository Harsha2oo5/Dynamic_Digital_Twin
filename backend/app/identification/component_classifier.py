"""Component identification layer.

Kept separate from detection so the physical-component logic can evolve independently
(e.g. a registry with permanent component IDs, or a second-stage classifier).

- Pretrained (COCO) model: detections are generic objects, so component_type is None.
- Custom model: the class name produced by the fine-tuned model *is* the component type.
  Class names come from the checkpoint itself; nothing is hard-coded here.
"""


class ComponentClassifier:
    def __init__(self, model_type: str, class_names: dict[int, str]):
        self.enabled = model_type == "custom"
        self.class_names = class_names

    def component_type(self, class_id: int) -> str | None:
        if not self.enabled:
            return None
        return self.class_names.get(class_id)
