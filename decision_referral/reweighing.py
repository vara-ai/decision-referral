"""Inverse probability weighting as described in eMethods 3. Sample weights"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


LabelsDict = Dict[str, np.ndarray]


########################################################################################################################
# Rates for different subtypes over a representative screening population.

# https://fachservice.mammo-programm.de/download/evaluationsberichte/Jahresbericht-Evaluation_2018.pdf
SCREEN_DETECTED_CANCER_FRACTION = 5.9 / 1000
# The recall rate of 2.9% actually just refers to followup screening rounds. The combined value for first and followup
# is 4.1%. Using 4.1% has an almost negligible effect on the results and does not change their interpretation.
RECALL_FRACTION = 29 / 1000
BIOPSY_FRACTION = 11 / 1000  # See Table 1, biopsy_fraction = n_has_biopsy / n_total
# Take the value given to us from ANONYMIZED
CONSENSUS_FRACTION = 0.122
########################################################################################################################


@dataclass(frozen=True)
class PopulationStats:
    """
    screen_detected_cancer_fraction: the fraction of all screening studies with a screen detected cancer
    biopsy_fraction: the fraction of all screening studies that got a biopsy
    recall_fraction: the fraction of all screening studies that were recalled
    consensus_fraction: the fraction of all screening studies that went into consensus conference
    """

    screen_detected_cancer_fraction: float
    biopsy_fraction: float
    recall_fraction: float
    consensus_fraction: float

    @property
    def no_consensus_fraction(self) -> float:
        return 1 - self.consensus_fraction


POPULATION_STATS = PopulationStats(
    screen_detected_cancer_fraction=SCREEN_DETECTED_CANCER_FRACTION,
    biopsy_fraction=BIOPSY_FRACTION,
    recall_fraction=RECALL_FRACTION,
    consensus_fraction=CONSENSUS_FRACTION,
)

SCREEN_DETECTED_CANCER_PRIORS = "screen_detected_cancer_priors"
NOT_CANCER = "not_cancer"
SCREEN_DETECTED_CANCERS = "screen_detected_cancers"
WAS_BIOPSIED = "was_biopsied"
WAS_REFERRED = "was_referred"
WENT_TO_CONSENSUS = "went_to_consensus"

REQUIRED_LABELS = {
    NOT_CANCER,
    WENT_TO_CONSENSUS,
    WAS_REFERRED,
    WAS_BIOPSIED,
    SCREEN_DETECTED_CANCERS,
    SCREEN_DETECTED_CANCER_PRIORS,
}


class LabelsNotAvailableError(Exception):
    """Label set such as WENT_TO_CONSENSUS or MISSED_INTERVAL_CANCERS or ... not available"""

    pass


@dataclass(frozen=True)
class ReweighingSubgroup:
    """
    Represents one subgroup of interest for our reweighing logic.
    """

    # Name of the subgroup, e.g. "cancers"
    name: str
    # Fraction this subgroup represents in a screening population, e.g. 0.0059
    fraction: float
    # Boolean mask, where True represents the members of this subgroup in the to-be-reweighed dataset.
    mask: np.ndarray

    @property
    def nr_samples_overall(self) -> int:
        return len(self.mask)

    @property
    def nr_samples_in_subgroup(self) -> int:
        return self.mask.sum()

    @property
    def percentage_in_dataset(self) -> float:
        return self.nr_samples_in_subgroup / self.nr_samples_overall

    @property
    def weight(self) -> float:
        return self.fraction / self.percentage_in_dataset


def suspiciousness_stages_reweighing_spec(
    labels_dict: LabelsDict, pop_stats: PopulationStats
) -> List[ReweighingSubgroup]:
    """
    Reweighing spec that only considers the five standard suspiciousness rounds (cancer, biopsy, recall, cc, no cc),
    but does not consider the first/followup metadata.
    """
    return [
        ReweighingSubgroup(
            name="screen_detected_cancer",
            fraction=pop_stats.screen_detected_cancer_fraction,
            mask=labels_dict[SCREEN_DETECTED_CANCERS],
        ),
        ReweighingSubgroup(
            name="biopsy_no_screen_detected_cancer",
            fraction=pop_stats.biopsy_fraction - pop_stats.screen_detected_cancer_fraction,
            mask=labels_dict[WAS_BIOPSIED] & labels_dict[NOT_CANCER],
        ),
        ReweighingSubgroup(
            name="recall_no_biopsy",
            fraction=pop_stats.recall_fraction - pop_stats.biopsy_fraction,
            mask=labels_dict[WAS_REFERRED] & ~labels_dict[WAS_BIOPSIED],
        ),
        ReweighingSubgroup(
            name="cc_no_recall",
            fraction=pop_stats.consensus_fraction - pop_stats.recall_fraction,
            mask=labels_dict[WENT_TO_CONSENSUS] & ~labels_dict[WAS_REFERRED],
        ),
        ReweighingSubgroup(
            name="no_cc",
            fraction=1.0 - pop_stats.consensus_fraction,
            mask=~labels_dict[WENT_TO_CONSENSUS],
        ),
    ]


def calculate_representative_weights(
    labels_dict: LabelsDict,
    reweighing_spec: Callable[
        [LabelsDict, PopulationStats], List[ReweighingSubgroup]
    ] = suspiciousness_stages_reweighing_spec,
    pop_stats: PopulationStats = POPULATION_STATS,
) -> Tuple[np.ndarray, Dict]:
    """
    Produce a set of sample weights that correct for unrepresentative dataset collection. Currently, we control
    for the following stages of suspiciousness (ordered increasingly):
     * no consensus conference
     * consensus conference
     * recall
     * biopsy
     * screen detected cancers

    The fractions that are passed externally, come from literature and in-house analysis and are usually defined on
    subgroups that are partially overlapping. This function takes care of calculating more fine-grained weights for
    non-overlapping subgroups as illustrated below.

    In the following, we visualise with exemplary numbers for consensus_fraction = 12%, recall_fraction = 3%,
    biopsy_fraction = 1.1%, and screen_detected_cancer_fraction = 0.59% the composition of a screening population in
    terms of its suspiciousness stages. We consider a single round of screening until up to, but excluding the next
    round.

    | normal_no_consensus | consensus                                                                |
      0.88                + 0.12                                                                     = 1.0
    | normal_no_consensus | consensus_no_recall | recall                                             |
      0.88                + 0.09                + 0.03                                               = 1.0
    | normal_no_consensus | consensus_no_recall | recall_no_biopsy | biopsy                          |
      0.88                + 0.09                + 0.019            + 0.011                           = 1.0
    | normal_no_consensus | consensus_no_recall | recall_no_biopsy | biopsy_no_sd_cancer | sd_cancer |
      0.88                + 0.09                + 0.019            + 0.0051               + 0.0059   = 1.0

    Args:
        labels_dict: as calculated by `DatasetMetadata.labels_dict`
        reweighing_spec: Function returning a list of subgroups to be considered for reweighing.
        pop_stats: screening population statistics

    Returns:
        array containing a single weight for each sample, which can be passed into `sklearn.metrics.roc_curve` etc. and
        a summary of reweighing groups

    Raises:
        `AssertionError` if the subgroups are either not distinct or not complete, or if the subgroups' fractions don't
            add to exactly 1.
        `LabelsNotAvailableError` if any of the necessary keys in labels_dict are not available.
    """

    reweighing_groups = reweighing_spec(labels_dict, pop_stats)

    assert all(
        sum(group.mask for group in reweighing_groups) == 1
    ), "Subgroups are either not distinct or not complete."
    sum_all_groups = sum(group.fraction for group in reweighing_groups)
    assert sum_all_groups == 1, f"Subgroup fractions must add to 1, but instead add to {sum_all_groups}."

    reweighing_summary = {
        subgroup.name: {
            "n": int(subgroup.nr_samples_in_subgroup),
            "percentage_in_dataset": f"{subgroup.percentage_in_dataset:.2%}",
            "percentage_should_be": f"{subgroup.fraction:.2%}",
            "weight": f"{subgroup.weight:.5f}",
        }
        for subgroup in reweighing_groups
    }

    conditions = {subgroup.weight: subgroup.mask for subgroup in reweighing_groups}

    # calculate_weights_from_boolean_conditions ensures that all samples are included in exactly one group
    return calculate_weights_from_boolean_conditions(conditions), reweighing_summary


def calculate_weights_from_boolean_conditions(conditions_dict: Dict[float, np.ndarray]) -> np.ndarray:
    """Given a mutually exclusive set of conditions and corresponding weights, assign a single weight to each sample

    e.g. if we have three samples

    conditions = {
        0.1: np.array([True, False, True, False]),
        0.2: np.array([False, True, False, False]),
        0.3: np.array([False, False, False, True]),
    }

    calculate_weights_from_binary_conditions(conditions) = np.array([0.1, 0.2, 0.1, 0.3])


    Args:
        conditions_dict: `weight_value` -> `boolean_mask`. If `boolean_mask` is True at a given index, the sample at
            that index will be assigned a weight of `weight_value`.

    Returns:
        an array of weights (floats) where each value matches the key of the only value condition array for which
        there is a true at that index

    Raises:
        AssertionError: If the conditions are not the same length, are not boolean type or are not mutually exclusive.
            The final condition is required to avoid ambiguity.
    """
    # first ensure that the conditions are boolean, 1D, and the same length
    assert all(condition_array.dtype == bool for condition_array in conditions_dict.values()), (
        f"Expected boolean conditions, got "
        f"dtypes {set(condition_array.dtype for condition_array in conditions_dict.values())}"
    )

    condition_shapes = [condition_array.shape for condition_array in conditions_dict.values()]
    assert len(set(condition_shapes)) == 1, "Conditions are not all the same length!"
    assert len(condition_shapes[0]) == 1, "Conditions are not 1 dimensional!"

    # now apply each of the masks sequentially, asserting mutual exclusivity along the way
    sample_weights = np.zeros(condition_shapes[0], dtype=float)
    weights_set = np.zeros(condition_shapes[0], dtype=bool)

    for weight, condition_array in conditions_dict.items():
        # check none of the weights have been set already
        assert not (
            condition_array & weights_set
        ).any(), "Specified conditions are not mutually exclusive on this data!"

        # update the weights and the record of which weights have been set
        sample_weights[condition_array] = weight
        weights_set = weights_set | condition_array

    # make sure all the weights have been set
    assert weights_set.all(), "Not all of the samples are covered by the specified conditions!"

    return sample_weights
