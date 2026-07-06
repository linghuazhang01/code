"""Legacy General-Reasoner verifier worker compatibility shim."""


class RewardModelWorker:
    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError(
            "Legacy General-Reasoner verifier worker was removed when grpo/ "
            "was reset for M2RL-style IF/Science GRPO."
        )


__all__ = ["RewardModelWorker"]
