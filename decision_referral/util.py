import os
import re
from typing import List, Optional, Tuple

import pandas as pd

from decision_referral import io_util, types
from decision_referral.types import DecisionReferralResult


NAME_RAD = "Rad."
NAME_DR = "DR"
NAME_AI = "AI"

SENSITIVITY = "sensitivity"
SPECIFICITY = "specificity"

DELTA_SENSITIVITY_COLUMN = f"Δ {SENSITIVITY} (p-value)"
DELTA_SPECIFICITY_COLUMN = f"Δ {SPECIFICITY} (p-value)"


def get_operating_pair_name(min_sensitivity: Optional[float], min_specificity: Optional[float]):
    assert (
        min_sensitivity is not None or min_specificity is not None
    ), f"Got min_sensitivity={min_sensitivity}, min_specificity={min_specificity}. At least one of them has to be set."
    if min_sensitivity is None:
        return f"SN@{min_specificity}"
    elif min_specificity is None:
        return f"NT@{min_sensitivity}"
    else:
        return f"NT@{min_sensitivity}+SN@{min_specificity}"


def get_sensitivity_and_specificity_from_operating_pair(name: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse (min_sensitivity, min_specificity) from operating pair name

    Examples:
        min_sensitivity, None <- "NT@{min_sensitivity}"
        None, min_specificity <- "SN@{min_specificity}"
        min_sensitivity, min_specificity <- "NT@{min_sensitivity}+SN@{min_specificity}"

    """
    m = re.search(r"^(NT@(?P<sensitivity>\d{1}\.\d+))?(\+)?(SN@(?P<specificity>\d{1}\.\d+))?$", name)
    if m is not None:
        min_sensitivity = float(m.group("sensitivity")) if m.group("sensitivity") is not None else None
        min_specificity = float(m.group("specificity")) if m.group("specificity") is not None else None
        return min_sensitivity, min_specificity
    else:
        raise ValueError(f"Can't parse sensitivity and/or specificity from {name}.")


def select_operating_pairs(df_results: pd.DataFrame, names_to_select: List[str]) -> pd.DataFrame:
    assert all(
        op in df_results.index for op in names_to_select
    ), f"Not all operating pairs {names_to_select} exist in data frame."
    return df_results.reindex(names_to_select)


def get_validation_dir(validation_dataset: types.Dataset = types.Dataset.INTERNAL_VALIDATION_SET) -> str:
    """Construct dir in which decision referral data for a model / validation combo is stored."""
    return os.path.join(io_util.RESULTS_DIR, validation_dataset.name.lower())


def get_test_dir(validation_dataset: types.Dataset, test_set: types.Dataset) -> str:
    """Get path to test directory."""
    return os.path.join(get_validation_dir(validation_dataset), "test_sets", test_set.name.lower())


def get_decision_referral_dir(dataset_dir: str) -> str:
    """Get the directory under which decision referral artefacts are stored"""
    return os.path.join(dataset_dir, "decision_referral")


def get_plots_dir(dr_dir: str) -> str:
    """Get the plot artefact directory from a given decision referral dir (i.e. ends in '/decision_referral')"""
    return os.path.join(os.path.split(dr_dir)[0], "plots")


def get_dataset_name_from_dir(dr_dir: str) -> str:
    """Simplistic util to get the name of the decision referral dataset from a given directory"""
    return os.path.split(dr_dir.replace("/decision_referral", "").replace("/plots", ""))[-1]


def get_max_sensitive_geq_specific_op(decision_referral_result: DecisionReferralResult) -> str:
    """Based on validation data results, determine an operating point that achieves the best
    sensitivity improvement without sacrificing specificity.
    """
    df = decision_referral_result.to_data_frame()
    df = df.sort_values(by=["delta_sensitivity"], ascending=True, inplace=False)
    df = df[df.delta_specificity >= 0.0]
    return df.iloc[-1].name


def generate_operating_pairs_list(min_sensitivities: List[float], min_specificities: List[float]) -> List[str]:
    # only NP
    ops = [get_operating_pair_name(min_sens, None) for min_sens in min_sensitivities]
    # only SN
    ops.extend([get_operating_pair_name(None, min_spec) for min_spec in min_specificities])
    # Both of NP & SN
    ops.extend(
        [
            get_operating_pair_name(min_sens, min_spec)
            for min_sens in min_sensitivities
            for min_spec in min_specificities
        ]
    )
    return ops


MIN_SENSITIVITIES = [0.99, 0.98, 0.97, 0.95]
MIN_SPECIFICITIES = [0.99, 0.98, 0.97, 0.95]
OPERATING_PAIRS = generate_operating_pairs_list(MIN_SENSITIVITIES, MIN_SPECIFICITIES)
