# Mechanism screen

M0–M5 are defined and were not trained. Shared exposure ledger:
`mechanism_screen_shared_exposure`.

| ID | Target | Supervision | Purpose |
|---|---|---|---|
| M0 | original_fixed | matched_non_role_qa | Conditional baseline |
| M1 | calibrated_fixed | matched_non_role_qa | Target intervention |
| M2 | original_fixed | grounded_role_qa | Grounding intervention |
| M3 | calibrated_fixed | grounded_role_qa | Interaction of calibration and grounded role questions |
| M4 | original_fixed | matched_compositional_declarative | Check whether ordinary semantic supervision explains a gain |
| M5 | original_fixed | role_qa_answers_permuted_within_groups | Diagnostic for reliance on valid grounding labels; not a competitive baseline |

Residual adapter: 512 → 128 → 512, last linear layer zero, identity at init.
Freeze it before comparing predictors. Loss: paraphrase-versus-negative
hinge plus an anchor penalty. No adapter was fit.

M3 has to beat the strongest matched baseline, not only M0.
M5 is a label permutation. Status: not_trained.
