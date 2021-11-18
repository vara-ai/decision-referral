from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from decision_referral import util
from decision_referral.types import DecisionReferralResult


# (normal triaging threshold, safety net threshold)
ThresholdPair = Tuple[Optional[float], Optional[float]]


def compute_thresholds(
    operating_pairs: Iterable[str],
    y_score_nt: np.ndarray,
    y_score_sn: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    rad_sensitivity: Optional[float],
) -> Dict[str, ThresholdPair]:
    """
    For each operating pair name, compute the necessary thresholds.

    Args:
        operating_pairs: List of operating pairs as formatted by `util.get_operating_pair_name`.
        y_score_nt: model scores for normal triaging
        y_score_sn: model scores for safety net
        y_true: ground truth labels
        weights: weights
        rad_sensitivity: The sensitivity achieved by radiologist. Used for configuring standalone AI

    Returns:
        Dictionary of operating pair to threshold pair.

    """
    selection_pos = y_true == True  # noqa
    selection_neg = ~selection_pos

    y_score_pos = y_score_nt[selection_pos]
    y_score_neg = y_score_sn[selection_neg]

    weights_pos = weights[selection_pos]
    weights_neg = weights[selection_neg]

    min_sensitivities = {util.get_sensitivity_and_specificity_from_operating_pair(op)[0] for op in operating_pairs}
    if rad_sensitivity is not None:
        # Add the radiologist sensitivity for configuring standalone AI
        min_sensitivities |= {rad_sensitivity}
    min_specificities = {util.get_sensitivity_and_specificity_from_operating_pair(op)[1] for op in operating_pairs}

    valid_min_sensitivities = np.array([sens for sens in min_sensitivities if sens is not None])
    valid_min_specificities = np.array([spec for spec in min_specificities if spec is not None])

    thresholds_by_sensitivity = {
        sens: thresh
        for sens, thresh in zip(
            valid_min_sensitivities,
            calculate_normal_triaging_threshold(y_score_pos, valid_min_sensitivities, weights_pos),
        )
    }

    thresholds_by_specificity = {
        spec: thresh
        for spec, thresh in zip(
            valid_min_specificities,
            calculate_safety_net_threshold(y_score_neg, valid_min_specificities, weights_neg),
        )
    }

    thresholds_by_operating_pairs = dict()
    for op in operating_pairs:
        min_sensitivity, min_specificity = util.get_sensitivity_and_specificity_from_operating_pair(op)
        thresholds_by_operating_pairs[op] = (
            thresholds_by_sensitivity[min_sensitivity] if min_sensitivity is not None else None,
            thresholds_by_specificity[min_specificity] if min_specificity is not None else None,
        )
    if rad_sensitivity is not None:
        # Add the radiologist sensitivity for configuring standalone AI
        thresholds_by_operating_pairs[util.NAME_AI] = (thresholds_by_sensitivity[rad_sensitivity], None)

    return thresholds_by_operating_pairs


def get_thresholds_by_operating_pairs_from_result(
    decision_referral_result: DecisionReferralResult,
) -> Dict[str, ThresholdPair]:
    """Extract thresholds from decision referral result."""
    thresholds_by_operating_pairs = {}
    for results_tuple in decision_referral_result:
        if results_tuple.name not in {util.NAME_RAD}:
            thresholds_by_operating_pairs[results_tuple.name] = (
                results_tuple.lower_threshold,
                results_tuple.upper_threshold,
            )
    return thresholds_by_operating_pairs


def calculate_normal_triaging_threshold(
    y_score_pos: np.ndarray, operating_sensitivity: Union[float, np.ndarray], weights: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """Calculate the largest threshold which produces a sensitivity greater than, or equal to,
    `operating_sensitivity`

    Args:
        y_score_pos (n_positives,): the probs of the positive studies considered
        operating_sensitivity: target sensitivity/sensitivities, must be in (0, 1]
        weights (n_positives,): optional weights for each sample for inverse probability weighing.

    Returns:
        threshold(s)
    """
    assert np.all((0 < operating_sensitivity) & (operating_sensitivity <= 1))

    # for operating sensitivity of 1, i.e. not using the normal preselection at all so
    # that there is no chance of false negatives
    # setting the threshold to the lowest score encountered does not generalize,
    # therefore we just set it to 0
    if isinstance(operating_sensitivity, np.ndarray):
        thresholds = np.zeros(len(operating_sensitivity))

        valid_sensitivities = operating_sensitivity < 1.0

        thresholds[valid_sensitivities] = calculate_proportion_threshold(
            y_score_pos,
            target_proportion=operating_sensitivity[valid_sensitivities],
            weights=weights,
            bigger_than=True,
        )
        return thresholds
    else:
        if operating_sensitivity == 1.0:
            return 0.0
        else:
            return calculate_proportion_threshold(
                y_score_pos, target_proportion=operating_sensitivity, weights=weights, bigger_than=True
            )


def calculate_safety_net_threshold(
    y_score_neg: np.ndarray, operating_specificity: Union[float, np.ndarray], weights: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """Calculate the threshold(s) that should be used for the safety net feature.
    This is defined as the lowest threshold for which the specificity is at least the `operating_specificity`.
    (taken that all studies > threshold get classified as positive)
    This is the standard set by our regulatory team.

    Note: we ignore the case that two negatives have exactly the same probability. This is extremely unlikely in
    practice

    Args:
        y_score_neg (n_negatives,): the cancer probability assigned to each normal exam in the dataset
        operating_specificity: 1 - maximum false positive rate(s) we tolerate
        weights (n_negatives,): optional weights for each sample for inverse probability weighing.


    Returns:
        the threshold(s) in the interval (0, 1). Studies with score higher than this should be flagged by the safety net
        If the FP rate constraint is not met at any of the thresholds, return a threshold of 1.
    """

    # for operating specificity of 1, i.e. not using the safety net at all so
    # that there is no chance of false positives
    # setting the threshold to the highest score encountered does not generalize,
    # therefore we just set it to 1
    if isinstance(operating_specificity, np.ndarray):
        thresholds = np.ones(len(operating_specificity))
        valid_specificities = operating_specificity < 1.0

        thresholds[valid_specificities] = calculate_proportion_threshold(
            y_score_neg,
            target_proportion=operating_specificity[valid_specificities],
            weights=weights,
            bigger_than=False,
        )
        return thresholds
    else:
        if operating_specificity == 1.0:
            return 1.0
        else:
            return calculate_proportion_threshold(
                y_score_neg, target_proportion=operating_specificity, weights=weights, bigger_than=False
            )


def calculate_proportion_threshold(
    scores: np.ndarray,
    target_proportion: Union[float, np.ndarray],
    bigger_than: bool,
    weights: Optional[np.ndarray] = None,
):
    """Calculates the threshold(s) for which we have the target proportion of
     (weighted) samples that are smaller than (or bigger than, depending on the
    flag) the threshold"""
    if weights is None:
        weights = np.ones_like(scores)
    assert len(weights) == len(scores), f"Got different number of weights ({len(weights)}) and samples ({len(scores)})."

    score_order = np.argsort(scores)
    if bigger_than:
        score_order = score_order[::-1]
    thresholds = scores[score_order]
    # if the specificity is above the highest one reachable
    sorted_weights = weights[score_order]

    # calculate the sensitivity that would be produced if the operating point was set to each of these values
    proportions = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    return thresholds[np.searchsorted(proportions, target_proportion)]
