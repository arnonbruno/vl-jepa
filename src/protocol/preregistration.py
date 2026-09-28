"""Thresholds fixed before any confirmation run."""

from __future__ import annotations

PREREGISTRATION = {
    "status": "preregistered_not_unblinded",
    "confirmation_runs_executed": 0,
    "seeds": [0, 1, 2],
    "primary_compositional_min_gain_points": 2.0,
    "paired_ci_must_exclude_zero": True,
    "max_retrieval_r1_degradation_points": 1.0,
    "protected_r1_endpoints": [
        "coco_karpathy_test_i2t_r1",
        "coco_karpathy_test_t2i_r1",
        "flickr30k_karpathy_test_i2t_r1",
        "flickr30k_karpathy_test_t2i_r1",
    ],
    "metrics_are_not_averaged_into_one_score": True,
    "winoground_alone_is_not_a_significance_claim": True,
}
