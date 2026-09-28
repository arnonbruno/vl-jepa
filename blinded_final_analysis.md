# Blinded final analysis

Status: `preregistered_not_unblinded`.
Confirmation runs executed: 0.
Seeds if confirmation is launched: [0, 1, 2].

Thresholds are stopping rules. Confirmation scores do not exist yet.

- At least 2.0 points on one declared compositional endpoint.
- The paired confidence interval on that difference excludes zero.
- No more than 1.0 point of degradation
  on each protected retrieval R@1 versus the strongest matched baseline:
  - `coco_karpathy_test_i2t_r1`
  - `coco_karpathy_test_t2i_r1`
  - `flickr30k_karpathy_test_i2t_r1`
  - `flickr30k_karpathy_test_t2i_r1`

Metrics stay separate. Winoground alone is not a significance claim.
Tuning after these endpoints are opened is development evidence.
