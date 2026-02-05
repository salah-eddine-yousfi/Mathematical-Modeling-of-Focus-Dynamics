from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class EvidenceConfig:
    H: float = 0.03
    lambda_on: float = 0.40
    lambda_off: float = 1.10
    lambda_unknown: float = 0.0
    threshold: float = 0.6
    L0: float = 0.0


@dataclass
class EvidenceStepOutput:
    llr: float
    L: float
    p: float
    state_filtered: int
    is_unknown: bool


def sigmoid(u: float) -> float:
    if u >= 0:
        z = math.exp(-u)
        return 1.0 / (1.0 + z)
    z = math.exp(u)
    return z / (1.0 + z)


def psi(L_prev: float, H: float) -> float:
    if not (0.0 < H < 0.5):
        raise ValueError(f"H doit être dans (0, 0.5). Reçu: H={H}")

    A = (1.0 - H) / H
    logA = math.log(A)

    def log_sum_exp_A(t: float) -> float:
        d = t - logA
        if d > 50:
            return logA + d
        return logA + math.log1p(math.exp(d))

    return L_prev + log_sum_exp_A(-L_prev) - log_sum_exp_A(+L_prev)


class EvidenceAccumulator:
    def __init__(self, cfg: EvidenceConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.L = float(self.cfg.L0)
        self.run_yes = 0
        self.run_no = 0

    def _llr_from_state(self, x: int | None) -> tuple[float, bool]:
        if x is None:
            self.run_yes = 0
            self.run_no = 0
            return float(self.cfg.lambda_unknown), True

        if x == 1:
            self.run_yes += 1
            self.run_no = 0
            if self.run_yes == 1:
                return 0.10, False
            if self.run_yes == 2:
                return 0.25, False
            return float(self.cfg.lambda_on), False

        if x == 0:
            self.run_no += 1
            self.run_yes = 0
            if self.run_no == 1:
                return -0.25, False
            return -float(self.cfg.lambda_off), False

        raise ValueError(f"x invalide: {x}")

    def step(self, x: int | None) -> EvidenceStepOutput:
        llr, is_unknown = self._llr_from_state(x)
        self.L = psi(self.L, self.cfg.H) + llr
        p = sigmoid(self.L)
        state_filtered = 1 if p >= self.cfg.threshold else 0

        return EvidenceStepOutput(
            llr=llr,
            L=self.L,
            p=p,
            state_filtered=state_filtered,
            is_unknown=is_unknown,
        )
