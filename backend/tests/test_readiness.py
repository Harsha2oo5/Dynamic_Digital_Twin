"""Readiness report: computed from review state and exports, gate blocks until reviewed data exists."""

from app.training.readiness import pilot_gate, review_states
from app.training.review_store import REVIEWED, ReviewStore
from tests.test_review_workflow import PER_CLASS, _review_all, env  # noqa: F401 - fixture reuse


def test_gate_blocks_without_reviews_and_counts_reviewed_boxes(env):  # noqa: F811
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    reviews = review_states(env["review"].parent)
    gate = pilot_gate(reviews, [], [])
    assert not gate["ready"] and gate["reviewed_images"] == 0
    assert any("0 human-reviewed images" in b for b in gate["blocking"])
    assert gate["unreviewed_staged_images"] == 7 * PER_CLASS
    assert reviews[0]["candidate_boxes_unverified"] == 2 * 7 * PER_CLASS and reviews[0]["human_actions_logged"] == 0

    _review_all(store)
    reviews = review_states(env["review"].parent)
    assert reviews[0]["reviewed_images"] == 7 * PER_CLASS and reviews[0]["human_actions_logged"] > 0
    gate = pilot_gate(reviews, [{"valid": True}], [{"training_gate_problems": []}])
    # 2 boxes per image -> 12 per class (radio_unit keeps its extra candidate): every class clears the threshold
    assert gate["ready"], gate["blocking"]
    assert all(n >= 10 for n in gate["reviewed_boxes_per_class"].values())
    assert sum(i["status"] == REVIEWED for i in store.images) == gate["reviewed_images"]
