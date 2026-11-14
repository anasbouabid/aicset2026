from __future__ import annotations

import inspect

from synthfraud.schema import Schema
from synthfraud.synthesizers.identity import IdentitySynthesizer
from synthfraud.synthesizers.count_models import (
    ClassConditionalCountCopulaSynthesizer,
    ClassConditionalCountHistSynthesizer,
)
from synthfraud.synthesizers.independent import (
    ClassConditionalCountBlockSynthesizer,
    ClassConditionalIndependentSynthesizer,
    IndependentSynthesizer,
)

SYNTHESIZERS = {
    "identity": IdentitySynthesizer,
    "independent": IndependentSynthesizer,
    "independent_classcond": ClassConditionalIndependentSynthesizer,
    "independent_classcond_countblock": ClassConditionalCountBlockSynthesizer,
    "independent_classcond_countcopula": ClassConditionalCountCopulaSynthesizer,
    "independent_classcond_counthist": ClassConditionalCountHistSynthesizer,
}


def build_synthesizer(name: str, seed: int, schema: Schema, **kwargs):
    extra = {k: v for k, v in kwargs.items() if v is not None}
    if name == "ctgan":
        from synthfraud.synthesizers.ctgan import CTGANSynthesizer

        return CTGANSynthesizer(seed=seed, schema=schema, **extra)
    if name == "tvae":
        from synthfraud.synthesizers.tvae import TVAESynthesizer

        return TVAESynthesizer(seed=seed, schema=schema, **extra)
    if name in ("tabddpm", "ddpm"):
        from synthfraud.synthesizers.tabddpm import TabDDPMSynthesizer

        return TabDDPMSynthesizer(seed=seed, schema=schema, **extra)
    if name == "findiff":
        from synthfraud.synthesizers.findiff import FinDiffSynthesizer

        return FinDiffSynthesizer(seed=seed, schema=schema, **extra)
    if name == "dp_independent":
        from synthfraud.synthesizers.dp import DPIndependentSynthesizer

        return DPIndependentSynthesizer(seed=seed, schema=schema, **extra)
    if name == "dp_classcond":
        from synthfraud.synthesizers.dp import DPClassConditionalSynthesizer

        return DPClassConditionalSynthesizer(seed=seed, schema=schema, **extra)
    if name == "patectgan":
        from synthfraud.synthesizers.patectgan import PATECTGANSynthesizer

        return PATECTGANSynthesizer(seed=seed, schema=schema, **extra)
    try:
        cls = SYNTHESIZERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown synthesizer {name!r}; choose from {sorted(SYNTHESIZERS) + ['ctgan', 'tvae', 'tabddpm', 'findiff', 'dp_independent', 'dp_classcond', 'patectgan']}"
        ) from exc
    params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in extra.items() if k in params}
    return cls(seed=seed, schema=schema, **accepted)
