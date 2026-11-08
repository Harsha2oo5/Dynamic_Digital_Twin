"""Evidence fusion and per-box decisions (pure functions, no models).

For one candidate box the evidence is:
  p_clip   softmax(logit_scale * cosine similarities) of the crop against the seven class descriptions
  p_gdino  Grounding DINO's vote: the best confidence among the prompts (of each class) that fired on this region
  p_proto  softmax(proto_logit_scale * cosine) of the crop's CLIP image embedding against visual class prototypes
           (leave-group-out, see prototypes.py); optional
  prior    optional soft boost for the image's folder class, scaled with box size

  log p_final(c) = w_clip log p_clip(c) + w_gdino log p_gdino(c) + w_proto log p_proto(c) + log prior(c)  (normalised)

Visual support rule: the final class must be in CLIP's text top-k OR be the top prototype match; the folder prior
alone can never decide a class.

Decision tiers:
  ACCEPTED   written as a YOLO pseudo-label
  UNCERTAIN  kept in the records with the reason, never written as a label
  REJECTED   not an object of the seven classes (background, too small, part, duplicate, low detector score)
"""

import math

from app.training.pseudo_label.config import CLASSES, PseudoLabelConfig

ACCEPTED, UNCERTAIN, REJECTED = "ACCEPTED", "UNCERTAIN", "REJECTED"


def softmax(scores: dict[str, float], scale: float) -> dict[str, float]:
    m = max(scores.values())
    e = {c: math.exp(scale * (s - m)) for c, s in scores.items()}
    z = sum(e.values())
    return {c: v / z for c, v in e.items()}


def gdino_votes(gdino_class: str, gdino_conf: float, alternatives: list[dict], smoothing: float) -> dict[str, float]:
    """Per-class vote from the prompts that fired on this region (the box itself plus suppressed alternatives)."""
    v = {c: 0.0 for c in CLASSES}
    v[gdino_class] = max(v[gdino_class], gdino_conf)
    for a in alternatives:
        if a.get("gdino_class") in v:
            v[a["gdino_class"]] = max(v[a["gdino_class"]], float(a.get("gdino_confidence", 0.0)))
    z = sum(x + smoothing for x in v.values())
    return {c: (x + smoothing) / z for c, x in v.items()}


def folder_prior(folder_class: str | None, area_share: float, cfg: PseudoLabelConfig) -> dict[str, float]:
    """Multiplicative factors (1 = neutral). The folder's class is boosted by up to `folder_prior_boost`."""
    prior = {c: 1.0 for c in CLASSES}
    if folder_class in prior and cfg.folder_prior_boost > 1.0:
        scale = min(1.0, area_share / cfg.folder_prior_full_area) if cfg.folder_prior_full_area > 0 else 1.0
        prior[folder_class] = 1.0 + (cfg.folder_prior_boost - 1.0) * scale
    return prior


def proto_probs(proto_sims: dict[str, float] | None, scale: float) -> dict[str, float] | None:
    """Softmax over the prototypes that exist; classes without a prototype get the lowest probability (no support)."""
    if not proto_sims:
        return None
    p = softmax(proto_sims, scale)
    floor = min(p.values())
    full = {c: p.get(c, floor) for c in CLASSES}
    z = sum(full.values())
    return {c: v / z for c, v in full.items()}


def fuse(p_clip: dict[str, float], p_gdino: dict[str, float], prior: dict[str, float],
         cfg: PseudoLabelConfig, p_proto: dict[str, float] | None = None) -> dict[str, float]:
    logits = {c: cfg.w_clip * math.log(max(p_clip[c], 1e-12)) + cfg.w_gdino * math.log(max(p_gdino[c], 1e-12))
              + math.log(prior[c]) for c in CLASSES}
    if p_proto is not None and cfg.w_proto > 0:
        for c in CLASSES:
            logits[c] += cfg.w_proto * math.log(max(p_proto[c], 1e-12))
    return softmax(logits, 1.0)


def ranked(p: dict[str, float]) -> list[str]:
    return sorted(p, key=lambda c: -p[c])


def classify_box(det: dict, clip_sims: dict[str, float], negative_sim: float, area_share: float,
                 folder_class: str | None, cfg: PseudoLabelConfig, flip_sims: dict[str, float] | None = None,
                 alt_sims: dict[str, float] | None = None, proto_sims: dict[str, float] | None = None,
                 flip_proto_sims: dict[str, float] | None = None) -> dict:
    """Decide one box. `det` holds gdino_class, gdino_confidence and alternatives. Returns the decision record
    (tier, final class, confidences, every fusion term, reason). Containment / duplicate rules run later."""
    p_clip = softmax(clip_sims, cfg.clip_logit_scale)
    p_gd = gdino_votes(det["gdino_class"], det["gdino_confidence"], det.get("alternatives", []),
                       cfg.gdino_vote_smoothing)
    prior = folder_prior(folder_class, area_share, cfg)
    p_pr = proto_probs(proto_sims, cfg.proto_logit_scale) if cfg.use_prototypes else None
    p = fuse(p_clip, p_gd, prior, cfg, p_pr)
    no_prior = fuse(p_clip, p_gd, {c: 1.0 for c in CLASSES}, cfg, p_pr)
    order = ranked(p)
    c1, c2 = order[0], order[1]
    clip_order = ranked(p_clip)
    proto_top1 = ranked(p_pr)[0] if p_pr else None
    best_sim = max(clip_sims.values())

    rec = {
        "final_class": c1,
        "final_confidence": round(p[c1], 4),
        "final_margin": round(p[c1] - p[c2], 4),
        "runner_up": c2,
        "p_final": {c: round(v, 4) for c, v in p.items()},
        "p_clip": {c: round(v, 4) for c, v in p_clip.items()},
        "p_gdino": {c: round(v, 4) for c, v in p_gd.items()},
        "folder_prior": {c: round(v, 3) for c, v in prior.items() if v != 1.0},
        "clip_similarities": {c: round(v, 4) for c, v in clip_sims.items()},
        "clip_negative_similarity": round(negative_sim, 4),
        "clip_top1": clip_order[0],
        "clip_rank_of_final": clip_order.index(c1) + 1,
        "gdino_top1": det["gdino_class"],
        "p_proto": {c: round(v, 4) for c, v in p_pr.items()} if p_pr else None,
        "proto_similarities": {c: round(v, 4) for c, v in proto_sims.items()} if p_pr else None,
        "proto_top1": proto_top1,
        "folder_prior_decisive": ranked(no_prior)[0] != c1,
        "area_share": round(area_share, 4),
    }
    if flip_sims is not None:
        flip_pr = proto_probs(flip_proto_sims, cfg.proto_logit_scale) if (cfg.use_prototypes and flip_proto_sims) else p_pr
        rec["flip_class"] = ranked(fuse(softmax(flip_sims, cfg.clip_logit_scale), p_gd, prior, cfg, flip_pr))[0]
        rec["flip_stable"] = rec["flip_class"] == c1
    if alt_sims is not None:
        rec["alt_prompt_class"] = ranked(fuse(softmax(alt_sims, cfg.clip_logit_scale), p_gd, prior, cfg, p_pr))[0]
        rec["alt_prompt_stable"] = rec["alt_prompt_class"] == c1

    min_final = cfg.min_final_confidence[c1] + (cfg.tiny_extra_confidence if area_share < cfg.tiny_area_share else 0.0)
    if area_share < cfg.min_area_share:
        tier, reason = REJECTED, f"too small ({area_share:.4f} of the photo < {cfg.min_area_share})"
    elif negative_sim > best_sim + cfg.background_margin:
        tier, reason = REJECTED, f"background description wins ({negative_sim:.3f} > {best_sim:.3f})"
    elif det["gdino_confidence"] < cfg.min_gdino_confidence[c1]:
        tier, reason = REJECTED, (f"detector confidence {det['gdino_confidence']:.3f} < "
                                  f"{cfg.min_gdino_confidence[c1]} for {c1}")
    elif rec["clip_rank_of_final"] > cfg.clip_support_top_k and proto_top1 != c1:
        tier, reason = UNCERTAIN, (f"no CLIP support: {c1} is CLIP text rank {rec['clip_rank_of_final']}"
                                   + (f" and not the top prototype ({proto_top1})" if proto_top1 else ""))
    elif p[c1] < min_final:
        tier, reason = UNCERTAIN, f"final confidence {p[c1]:.3f} < {min_final:.2f}"
    elif p[c1] - p[c2] < cfg.min_final_margin:
        tier, reason = UNCERTAIN, f"margin to {c2} {p[c1] - p[c2]:.3f} < {cfg.min_final_margin}"
    elif cfg.require_flip_stable and rec.get("flip_stable") is False:
        tier, reason = UNCERTAIN, f"class changes to {rec['flip_class']} on a flipped crop"
    else:
        tier, reason = ACCEPTED, None
    rec["tier"], rec["reason"] = tier, reason
    return rec
