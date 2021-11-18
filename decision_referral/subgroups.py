import logging
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
import numpy as np

from decision_referral import (
    io_util,
    statistics_util,
    types,
    util,
)
from decision_referral.core import DecisionReferral
from decision_referral.types import ValueWithCI
from decision_referral.util import (
    NAME_AI,
    NAME_RAD,
    NAME_DR,
    SENSITIVITY,
    SPECIFICITY,
)


class SubgroupMasks:
    STRATIFICATIONS = [
        "Screening site",
        "Manufacturer",
        "Breast density",
        "Size (mm)",
        "Core Needle Biopsy score",
        "Finding",
    ]

    def __init__(self, df: pd.DataFrame):
        subgroup_cols = [c for c in df.columns if c[0] in self.STRATIFICATIONS]
        subgroup_df = df[subgroup_cols]
        if not isinstance(subgroup_df.columns, pd.MultiIndex):
            # Make sure we have a multi-index
            subgroup_df.columns = pd.MultiIndex.from_tuples(subgroup_df.columns)
        self._df = subgroup_df

    @classmethod
    def from_disk(cls, dataset: types.Dataset):
        filename = os.path.join(io_util.INPUT_DIR, dataset.name.lower() + ".h5")
        return cls(pd.read_hdf(filename, key="table"))

    def __getitem__(self, stratification: str):
        assert stratification in self.STRATIFICATIONS, f"{stratification} not in {self.STRATIFICATIONS}"
        return self._df[stratification]


class SubgroupResult(NamedTuple):
    method: str  # NAME_RAD, NAME_AI, NAME_DR
    stratification: Optional[str]  # e.g. density or None (for the average sensitivity)
    stratum: Optional[str]  # e.g. ACR-A
    value_with_ci: ValueWithCI  # sensitivity or specificity
    n_studies: int
    delta: Optional[float]  # difference in sensitivity or specificity wrt radiologist
    p_value: Optional[float]  # p_value for the difference in sensitivity or specificity wrt radiologist


class SubgroupResults(List[SubgroupResult]):

    NAME = "<ValueName>"

    def to_data_frame(self) -> pd.DataFrame():
        df = pd.DataFrame(self)
        # Format the columns
        df[f"{self.NAME} (95% CI)"] = df.value_with_ci.map(lambda x: f"{x.value:.1%} ({x.ci_low:.1%}, {x.ci_upp:.1%})")
        # LDH: "p-values should be given to two significant figures,
        #       but no longer than 4 decimal places (e.g. p<0.0001)."
        df[f"Δ {self.NAME} (P value)"] = df.delta.map(lambda x: f"{x:.1%} " if x is not None else "") + df.p_value.map(
            lambda x: "(p=" + f"{round(x, 4):.2g}"[:6] + ")" if x is not None else ""
        )
        df.drop(
            axis=1,
            columns=[
                "value_with_ci",
                "delta",
                "p_value",
            ],
            inplace=True,
        )
        return df

    def get_strata(self, method: str, stratification: Optional[str]) -> Union[SubgroupResult, List[SubgroupResult]]:
        strata = [sp for sp in self if sp.method == method and sp.stratification == stratification]
        if stratification is None:
            assert len(strata) == 1, (
                f"Without stratification, we expect only one sensitivity (the average). " f"strata={strata}."
            )
            return strata[0]
        else:
            return strata


class SubgroupSensitivities(SubgroupResults):
    NAME = "Sensitivity"


class SubgroupSpecificities(SubgroupResults):
    NAME = "Specificity"


def compute_subgroup_results(
    subgroup_masks: SubgroupMasks,
    decision_referral: DecisionReferral,
    operating_point: str,
    run_hypothesis_tests: bool = True,
) -> Tuple[SubgroupSensitivities, SubgroupSpecificities]:

    min_sensitivity, min_specificity = util.get_sensitivity_and_specificity_from_operating_pair(operating_point)
    result_tuple = decision_referral.result.get_result_tuple(min_sensitivity, min_specificity)

    # Precompute as we need it for every stratum
    y_rad_plus_ai = decision_referral.combine_rad_and_ai_predictions(
        result_tuple.selection_unconfident, result_tuple.lower_threshold, result_tuple.upper_threshold
    )

    # Initialize with average values
    subgroup_sensitivities = compute_sensitivity_per_subgroup(
        decision_referral, y_rad_plus_ai, stratification=None, run_hypothesis_tests=run_hypothesis_tests
    )
    subgroup_specificities = compute_specificity_per_subgroup(
        decision_referral, y_rad_plus_ai, stratification=None, run_hypothesis_tests=run_hypothesis_tests
    )

    # For sensitivity, we go through all subgroups
    for stratifier_name in subgroup_masks.STRATIFICATIONS:
        stratifier_masks = get_masks_of_positives(decision_referral.y_true, subgroup_masks[stratifier_name])
        if stratifier_masks == {}:
            logging.debug(
                f"Not enough metadata available to perform {stratifier_name} stratification. "
                f"This can happen e.g. for stratifiers that require region level annotations which is "
                f"not available for all of our partner datasets."
            )
            continue

        for stratum_name, stratum_mask in stratifier_masks.items():
            subgroup_sensitivities.extend(
                compute_sensitivity_per_subgroup(
                    decision_referral,
                    y_rad_plus_ai=y_rad_plus_ai,
                    stratification=stratifier_name,
                    stratum=stratum_name,
                    mask=stratum_mask,
                    run_hypothesis_tests=run_hypothesis_tests,
                )
            )

    # For specificity, we are only interested in the stratification over manufacturers
    MANUFACTURER = "Manufacturer"
    for stratum_name, stratum_mask in get_masks_of_negatives(
        decision_referral.y_true, subgroup_masks[MANUFACTURER]
    ).items():
        subgroup_specificities.extend(
            compute_specificity_per_subgroup(
                decision_referral,
                y_rad_plus_ai=y_rad_plus_ai,
                stratification=MANUFACTURER,
                stratum=stratum_name,
                mask=stratum_mask,
                run_hypothesis_tests=run_hypothesis_tests,
            )
        )

    return SubgroupSensitivities(subgroup_sensitivities), SubgroupSpecificities(subgroup_specificities)


def compute_sensitivity_per_subgroup(
    decision_referral: DecisionReferral,
    y_rad_plus_ai: np.ndarray,
    stratification: Optional[str],
    stratum: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    run_hypothesis_tests: bool = True,
) -> List[SubgroupResult]:
    return _compute_metric_per_subgroup(
        SENSITIVITY,
        decision_referral,
        y_rad_plus_ai,
        stratification,
        stratum,
        mask,
        run_hypothesis_tests,
    )


def compute_specificity_per_subgroup(
    decision_referral: DecisionReferral,
    y_rad_plus_ai: np.ndarray,
    stratification: Optional[str],
    stratum: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    run_hypothesis_tests: bool = True,
) -> List[SubgroupResult]:
    return _compute_metric_per_subgroup(
        SPECIFICITY,
        decision_referral,
        y_rad_plus_ai,
        stratification,
        stratum,
        mask,
        run_hypothesis_tests,
    )


def _compute_metric_per_subgroup(
    metric_name: str,
    decision_referral: DecisionReferral,
    y_rad_plus_ai: np.ndarray,
    stratification: Optional[str],
    stratum: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    run_hypothesis_tests: bool = True,
) -> List[SubgroupResult]:

    n_pred = decision_referral.y_rad.shape[1]

    # apply optional mask
    y_rad_selected = _maybe_filter(decision_referral.y_rad, mask)
    y_ai_selected = _maybe_filter(decision_referral.y_ai, mask)
    # We mimic the AI predictions to also come from two radiologists
    y_ai_selected = np.repeat(y_ai_selected[:, None], repeats=n_pred, axis=1)

    y_rad_plus_ai_selected = _maybe_filter(y_rad_plus_ai, mask)
    y_true_selected = _maybe_filter(decision_referral.y_true, mask)
    weights_selected = _maybe_filter(decision_referral.weights, mask)

    if metric_name == SENSITIVITY:
        n_studies = int(y_true_selected.sum())
        metric_fn = statistics_util.compute_sensitivity_vectorized
        metric_with_ci_fn = statistics_util.compute_sensitivity_from_pooled_predictions
    elif metric_name == SPECIFICITY:
        n_studies = int((~y_true_selected).sum())
        metric_fn = statistics_util.compute_specificity_vectorized
        metric_with_ci_fn = statistics_util.compute_specificity_from_pooled_predictions
    else:
        raise ValueError(f"Metric {metric_name} not supported.")

    # Compute deltas such that we have those available even if we don't run hypothesis tests
    def _compute_delta(y_pred: np.ndarray) -> float:
        y_true_repeated = np.repeat(y_true_selected, n_pred)
        weights_repeated = np.repeat(weights_selected, n_pred)
        return metric_fn(y_true_repeated, y_pred.flatten(), weights_repeated) - metric_fn(
            y_true_repeated, y_rad_selected.flatten(), weights_repeated
        )

    delta_dr = _compute_delta(y_rad_plus_ai_selected)
    delta_ai = _compute_delta(y_ai_selected)

    if run_hypothesis_tests:
        test_result_ai = statistics_util.permutation_test(
            y_true=y_true_selected,
            y_rad_plus_ai=y_ai_selected,
            y_rad=y_rad_selected,
            sample_weights=weights_selected,
            metric_fns={metric_name: metric_fn},
        )
        assert delta_ai == test_result_ai[metric_name].delta

        test_result_rad_plus_ai = statistics_util.permutation_test(
            y_true=y_true_selected,
            y_rad_plus_ai=y_rad_plus_ai_selected,
            y_rad=y_rad_selected,
            sample_weights=weights_selected,
            metric_fns={metric_name: metric_fn},
        )
        assert delta_dr == test_result_rad_plus_ai[metric_name].delta

    return [
        SubgroupResult(
            NAME_RAD,
            stratification,
            stratum,
            value_with_ci=metric_with_ci_fn(
                y_true=y_true_selected,
                y_pred=y_rad_selected,
                sample_weights=weights_selected,
                n_resamples=decision_referral.n_bootstrap,
            ),
            n_studies=n_studies,
            delta=None,
            p_value=None,
        ),
        SubgroupResult(
            NAME_AI,
            stratification,
            stratum,
            value_with_ci=metric_with_ci_fn(
                y_true=y_true_selected,
                y_pred=y_ai_selected,
                sample_weights=weights_selected,
                n_resamples=decision_referral.n_bootstrap,
            ),
            n_studies=n_studies,
            delta=delta_ai,
            p_value=test_result_ai[metric_name].p_value if run_hypothesis_tests else None,
        ),
        SubgroupResult(
            NAME_DR,
            stratification,
            stratum,
            value_with_ci=metric_with_ci_fn(
                y_true=y_true_selected,
                y_pred=y_rad_plus_ai_selected,
                sample_weights=weights_selected,
                n_resamples=decision_referral.n_bootstrap,
            ),
            n_studies=n_studies,
            delta=delta_dr,
            p_value=test_result_rad_plus_ai[metric_name].p_value if run_hypothesis_tests else None,
        ),
    ]


def _maybe_filter(array: np.ndarray, mask: Optional[np.ndarray]):
    return array[mask] if mask is not None else array


def get_masks_of_positives(y_true: np.ndarray, df_masks: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Filter the masks down to using only the positives

    This removes non-cancerous samples such as benign radiological findings
    """
    return {col: (df_masks[col].values & y_true) for col in df_masks}


def get_masks_of_negatives(y_true: np.ndarray, df_masks: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Filter the masks down to using only the negatives"""
    return {col: (df_masks[col].values & ~y_true) for col in df_masks}
