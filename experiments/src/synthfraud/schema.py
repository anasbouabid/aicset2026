"""Column roles and public (C_pub) constraints. Bank-specific rules do not live here."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FRAUD_TYPES = ("TRANSFER", "CASH_OUT")
PAYSIM_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


@dataclass(frozen=True)
class Schema:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    label: str

    @property
    def feature_cols(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    @property
    def all_cols(self) -> list[str]:
        return self.feature_cols + [self.label]

    def drop_numeric(self, cols: tuple[str, ...] | list[str]) -> Schema:
        drop = set(cols)
        numeric = tuple(c for c in self.numeric if c not in drop)
        if not numeric and not self.categorical:
            raise ValueError("drop_numeric would leave no features")
        return Schema(
            name=self.name,
            numeric=numeric,
            categorical=self.categorical,
            label=self.label,
        )


TOY = Schema(
    name="toy",
    numeric=("amount", "hour", "n_tx_24h"),
    categorical=("merchant_cat", "country", "channel"),
    label="is_fraud",
)

PAYSIM = Schema(
    name="paysim",
    numeric=(
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ),
    categorical=("type",),
    label="isFraud",
)

# Compact IEEE-CIS T1 schema: public typology + count/delay + a low-missing V block.
# V307–V321 is a complete-ish aggregate block on labeled train (see diagnostic JSON).
IEEE_CIS_PRODUCTS = ("W", "H", "C", "S", "R")
IEEE_CIS_CARD4 = ("visa", "mastercard", "american express", "discover", "missing")
IEEE_CIS_CARD6 = ("debit", "credit", "debit or credit", "charge card", "missing")
IEEE_CIS_DEVICES = ("desktop", "mobile", "missing")

IEEE_CIS = Schema(
    name="ieee_cis",
    numeric=(
        "TransactionAmt",
        "TransactionDT",
        "C1",
        "C2",
        "C13",
        "D1",
        "V307",
        "V310",
        "V314",
        "V317",
        "V320",
    ),
    categorical=("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"),
    label="isFraud",
)


# Matches the real-data counts HGB (C1, C2, C13 + ProductCD). Used as a joint
# bootstrap block in independent_classcond_countblock.
IEEE_CIS_COUNT_NUMERIC = ("C1", "C2", "C13")
IEEE_CIS_COUNT_CAT = "ProductCD"
IEEE_CIS_COUNT_BLOCK = IEEE_CIS_COUNT_NUMERIC + (IEEE_CIS_COUNT_CAT,)


def ieee_signal_slices() -> dict[str, Schema]:
    """Compact IEEE-CIS views for U_h source-of-signal (not extra C_pub)."""
    public_cat = ("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType")
    public_num = ("TransactionAmt", "TransactionDT")
    count_num = ("C1", "C2", "C13")
    return {
        "counts": Schema(
            name="ieee_cis_counts",
            numeric=count_num,
            categorical=("ProductCD",),
            label=IEEE_CIS.label,
        ),
        "no_counts": IEEE_CIS.drop_numeric(count_num),
        "public": Schema(
            name="ieee_cis_public",
            numeric=public_num,
            categorical=public_cat,
            label=IEEE_CIS.label,
        ),
    }


# Toy-only draw lists (not PaySim).
MERCHANT_CATS = ("grocery", "fuel", "electronics", "travel", "atm")
COUNTRIES = ("FR", "DE", "ES", "US", "NG")
CHANNELS = ("pos", "ecom", "app")
MAX_AMOUNT_TOY = 20_000.0


def constraint_violations(schema: Schema, row: dict) -> list[str]:
    if schema.name == "paysim":
        return _paysim_violations(row)
    if schema.name == "ieee_cis":
        return _ieee_cis_violations(row)
    return _toy_violations(row)


def constraint_mask(schema: Schema, frame: pd.DataFrame) -> pd.Series:
    """True where the row satisfies C_pub (vectorized)."""
    if schema.name == "paysim":
        amount_ok = frame["amount"].astype(float) >= 0
        step_ok = frame["step"].astype(int) >= 1
        type_ok = frame["type"].astype(str).isin(PAYSIM_TYPES)
        fraud = frame["isFraud"].astype(int).eq(1)
        typology_ok = ~fraud | frame["type"].astype(str).isin(FRAUD_TYPES)
        return amount_ok & step_ok & type_ok & typology_ok
    if schema.name == "ieee_cis":
        amount_ok = frame["TransactionAmt"].astype(float) >= 0
        dt_ok = frame["TransactionDT"].astype(float) >= 0
        prod_ok = frame["ProductCD"].astype(str).isin(IEEE_CIS_PRODUCTS)
        card4_ok = frame["card4"].astype(str).isin(IEEE_CIS_CARD4)
        card6_ok = frame["card6"].astype(str).isin(IEEE_CIS_CARD6)
        device_ok = frame["DeviceType"].astype(str).isin(IEEE_CIS_DEVICES)
        return amount_ok & dt_ok & prod_ok & card4_ok & card6_ok & device_ok
    hits = [
        len(constraint_violations(schema, rec)) == 0
        for rec in frame.to_dict(orient="records")
    ]
    return pd.Series(hits, index=frame.index)


def public_certificate(schema: Schema, row: dict) -> str:
    """Auditor label from C_pub only — never a production threshold."""
    if schema.name == "paysim":
        if int(row["isFraud"]) != 1:
            return "legit"
        kind = str(row["type"])
        if kind == "TRANSFER":
            return "typology_transfer"
        if kind == "CASH_OUT":
            return "typology_cash_out"
        return "uncertified_fraud"
    if schema.name == "ieee_cis":
        if int(row["isFraud"]) != 1:
            return "legit"
        prod = str(row["ProductCD"])
        if prod in IEEE_CIS_PRODUCTS:
            return f"typology_product_{prod}"
        return "uncertified_fraud"
    if int(row[schema.label]) != 1:
        return "legit"
    return "fraud"


def certificate_coverage(schema: Schema, frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in frame.to_dict(orient="records"):
        key = public_certificate(schema, rec)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _toy_violations(row: dict) -> list[str]:
    hits: list[str] = []
    amount = float(row["amount"])
    hour = int(row["hour"])
    if amount < 0 or amount > MAX_AMOUNT_TOY:
        hits.append("amount_bounds")
    if hour < 0 or hour > 23:
        hits.append("hour_bounds")
    if "n_tx_24h" in row:
        if int(row["n_tx_24h"]) < 0:
            hits.append("velocity_nonneg")
    if row.get("merchant_cat") == "atm" and row.get("channel") == "ecom":
        hits.append("atm_not_ecom")
    return hits


def _ieee_cis_violations(row: dict) -> list[str]:
    """Public IEEE-CIS bounds / typology — not a production scorer."""
    hits: list[str] = []
    if float(row["TransactionAmt"]) < 0:
        hits.append("amount_nonneg")
    if float(row["TransactionDT"]) < 0:
        hits.append("dt_nonneg")
    if str(row["ProductCD"]) not in IEEE_CIS_PRODUCTS:
        hits.append("product_known")
    if str(row["card4"]) not in IEEE_CIS_CARD4:
        hits.append("card4_known")
    if str(row["card6"]) not in IEEE_CIS_CARD6:
        hits.append("card6_known")
    if str(row["DeviceType"]) not in IEEE_CIS_DEVICES:
        hits.append("device_known")
    return hits


def _paysim_violations(row: dict) -> list[str]:
    """Public PaySim physics / typology — not a production scorer."""
    hits: list[str] = []
    if float(row["amount"]) < 0:
        hits.append("amount_nonneg")
    if int(row["step"]) < 1:
        hits.append("step_positive")
    if str(row["type"]) not in PAYSIM_TYPES:
        hits.append("type_known")
    if int(row["isFraud"]) == 1 and str(row["type"]) not in FRAUD_TYPES:
        hits.append("fraud_only_transfer_cashout")
    return hits


def public_evasion(schema: Schema, fraud):
    """Constraint-legal perturbation used to score adversarial utility (Risk B)."""
    attacked = fraud.copy()
    if schema.name == "paysim" or "type" in attacked.columns:
        attacked["amount"] = attacked["amount"] * 0.25
        attacked["type"] = "PAYMENT"
    elif schema.name == "ieee_cis":
        attacked["TransactionAmt"] = attacked["TransactionAmt"].astype(float) * 0.25
        attacked["ProductCD"] = "W"
    else:
        attacked["amount"] = attacked["amount"].clip(lower=1.0) * 0.25
        if "hour" in attacked.columns:
            attacked["hour"] = 12
    return attacked
