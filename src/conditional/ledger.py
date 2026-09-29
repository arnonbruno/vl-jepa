"""Caps for the research program. Authorization is not training.

The initial program is 6 target-screen runs, 6 mechanism-screen runs, 9
confirmation runs, and 6 second-backbone runs. Confirmation waits for a
development signal. The second backbone waits for a proceed decision. Nothing
here launches those runs, and the whole set cannot be authorized at once.
"""

from __future__ import annotations

STAGE_CAPS = {
    "target_screen": 6,
    "mechanism_screen": 6,
    "confirmation": 9,
    "second_backbone": 6,
}

GQA_SCREEN_IMAGE_CAP = 5000
GQA_SCREEN_PAIR_CAP = 20000


class LaunchRefused(RuntimeError):
    pass


class ComputeLedger:
    def __init__(self) -> None:
        self.authorized = {stage: 0 for stage in STAGE_CAPS}
        self.development_signal = False
        self.confirmation_decision: str | None = None

    def note_development_signal(self, present: bool) -> None:
        self.development_signal = bool(present)

    def note_confirmation_decision(self, decision: str) -> None:
        if decision not in {"proceed", "stop"}:
            raise ValueError("confirmation decision must be proceed or stop")
        self.confirmation_decision = decision

    def authorize(self, stage: str, count: int) -> dict:
        if stage not in STAGE_CAPS:
            raise KeyError(f"unknown stage {stage!r}")
        if count < 1:
            raise ValueError("authorization count must be positive")
        if stage == "confirmation" and not self.development_signal:
            raise LaunchRefused("confirmation waits for a development signal")
        if stage == "second_backbone" and self.confirmation_decision != "proceed":
            raise LaunchRefused("a second backbone waits for a confirmation proceed decision")
        cap = STAGE_CAPS[stage]
        if self.authorized[stage] + count > cap:
            raise LaunchRefused(f"{stage} cap is {cap}; refusing {count} more runs")
        self.authorized[stage] += count
        return {
            "stage": stage,
            "authorized_total": self.authorized[stage],
            "cap": cap,
            "runs_executed": 0,
        }

    def authorize_initial_program(self) -> None:
        raise LaunchRefused("do not launch all 27 planned training runs in advance")


def assert_gqa_screening_cap(*, images: int, pairs: int) -> None:
    if images < 0 or pairs < 0:
        raise ValueError("GQA screening counts must be non-negative")
    if images > GQA_SCREEN_IMAGE_CAP or pairs > GQA_SCREEN_PAIR_CAP:
        raise ValueError(
            f"GQA screening cap is {GQA_SCREEN_IMAGE_CAP} images and "
            f"{GQA_SCREEN_PAIR_CAP} question-answer pairs"
        )
