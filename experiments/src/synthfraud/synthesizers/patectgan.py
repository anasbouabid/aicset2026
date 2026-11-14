"""PATE-CTGAN via SmartNoise (PATE [jordon2019pategan] + CTGAN [xu2019modeling]).

Not the ICLR PATE-GAN architecture. SmartNoise trains until the requested
ε is spent; the unused CTGAN `epochs` knob does not stop the loop.
Runs on CPU here (the library forces CPU when CUDA is absent; no MPS path).
"""

from __future__ import annotations

import pandas as pd

from synthfraud.log import log
from synthfraud.schema import Schema


class PATECTGANSynthesizer:
    def __init__(
        self,
        seed: int,
        schema: Schema,
        epsilon: float = 10.0,
        preprocessor_eps: float = 1.0,
        epochs: int = 300,
        delta: float | None = None,
        sample_per_teacher: int = 10_000,
        teacher_iters: int = 5,
        student_iters: int = 5,
        batch_size: int = 500,
        pac: int = 1,
    ) -> None:
        self.seed = seed
        self.schema = schema
        self.epsilon = float(epsilon)
        self.preprocessor_eps = float(preprocessor_eps)
        self.epochs = int(epochs)
        self.delta = delta
        self.sample_per_teacher = int(sample_per_teacher)
        self.teacher_iters = int(teacher_iters)
        self.student_iters = int(student_iters)
        self.batch_size = int(batch_size)
        self.pac = int(pac)
        self._model = None
        self.accounting: dict | None = None

    def fit(self, train: pd.DataFrame) -> None:
        try:
            from snsynth import Synthesizer
        except ImportError as exc:
            raise ImportError("PATE-CTGAN needs `pip install -e '.[dp]'`") from exc
        n_teachers = int(len(train) / self.sample_per_teacher) + 1
        log(
            f"PATE-CTGAN fit ε={self.epsilon:g} preprocessor_ε={self.preprocessor_eps:g} "
            f"sample_per_teacher={self.sample_per_teacher} (~{n_teachers} teachers) "
            f"n={len(train):,}  (loop stops when ε is spent, not at epochs={self.epochs})"
        )
        kwargs = {
            "epsilon": self.epsilon,
            "verbose": True,
            "epochs": self.epochs,
            "pac": self.pac,
            "cuda": False,
            "batch_size": self.batch_size,
            "sample_per_teacher": self.sample_per_teacher,
            "teacher_iters": self.teacher_iters,
            "student_iters": self.student_iters,
        }
        if self.delta is not None:
            kwargs["delta"] = self.delta
        model = Synthesizer.create("patectgan", **kwargs)
        discrete = list(self.schema.categorical) + [self.schema.label]
        model.fit(
            train[self.schema.all_cols],
            preprocessor_eps=self.preprocessor_eps,
            categorical_columns=discrete,
            continuous_columns=list(self.schema.numeric),
        )
        self._model = model
        spent = None
        if hasattr(model, "_transformer") and getattr(model._transformer, "odometer", None):
            try:
                spent = tuple(model._transformer.odometer.spent)
            except Exception:
                spent = None
        spent_py = [float(x) for x in spent] if spent is not None else None
        delta = getattr(model, "delta", self.delta)
        self.accounting = {
            "mechanism": "patectgan",
            "epsilon_requested": self.epsilon,
            "preprocessor_eps": self.preprocessor_eps,
            "delta": None if delta is None else float(delta),
            "sample_per_teacher": self.sample_per_teacher,
            "n_teachers_approx": n_teachers,
            "preprocessor_odometer": spent_py,
            "cite": "jordon2019pategan",
            "note": (
                "SmartNoise PATE-CTGAN, not ICLR PATE-GAN. "
                "Training stops on spent ε. Library shares one discriminator "
                "object across teachers (snsynth 1.0.8)."
            ),
        }
        log(f"PATE-CTGAN fit finished accounting={self.accounting}")

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("call fit() before sample()")
        raw = self._model.sample(n)
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(raw, columns=self.schema.all_cols)
        frame = raw[self.schema.all_cols].copy()
        if self.schema.name == "paysim":
            frame[self.schema.label] = frame[self.schema.label].round().astype(int)
            frame["step"] = frame["step"].round().clip(lower=1).astype(int)
            frame["type"] = frame["type"].astype(str)
        return frame
