from collections import defaultdict
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_curve
import tensorflow as tf

from decision_referral.types import OperatingPoint, ValueWithCI


def determine_best_op_from_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: bool,
    sample_weight: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    name: str = "",
) -> OperatingPoint:
    """Determines the best operating point from receiver-operating-characteristics

    Currently, "best" is defined by "maximising the area under the operating point".
    Other choices might be desirable in the future. E.g. finding the best op by
    passing in a desired minimum sensitivity or specificity e.g. from radiologist
    reference values.

    Args:
        y_true: (n_samples,)
        y_score: (n_samples,)
        pos_label: values of positive labels
        sample_weight: (n_samples,), optional
        n_bootstrap: optional number of bootstrap samples
        name: for the operating point

    Returns:
        operating_point: sensitivity, specificity and ppv with CIs

    """
    fdr, tdr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=pos_label, sample_weight=sample_weight)

    area_under_op = np.array([t * (1 - f) for t, f in zip(tdr, fdr)])
    best_threshold = thresholds[np.argmax(area_under_op)]

    y_pred = np.empty_like(y_true)
    y_pred[y_score >= best_threshold] = True
    y_pred[y_score < best_threshold] = False

    # No pooling happening here, but we use the function in order to get CIs via bootstrapping
    return compute_operating_point_from_pooled_predictions(
        y_true, y_pred, sample_weight, n_resamples=n_bootstrap, name=name
    )


def compute_sensitivity_vectorized(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """Vectorized, fast implementation without boilerplate to be used for resampling based CIs and hypothesis tests

    Args:
        y_true (n_samples,) or (n_resamples, n_samples): boolean, if used with resampling set axis=1
        y_pred (n_samples,) or (n_resamples, n_samples): boolean predictions, if used with resampling set axis=1
        weights (n_samples,) or (n_resamples, n_samples): float weights, if used with resampling set axis=1
        axis: set to 1 / -1 if metric aggregation over n_samples shall happen in parallel for n_resamples.

    Returns:
        sensitivity:
            scalar if input is of shape (n_samples,);
            (n_resamples,) if input is of shape (n_resamples, n_samples) and aggregation is performed over axis=1

    """
    return ((y_true & y_pred) * weights).sum(axis) / (y_true * weights).sum(axis)


def compute_specificity_vectorized(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """Vectorized, fast implementation without boilerplate to be used for resampling based CIs and hypothesis tests

    Args:
        y_true (n_samples,) or (n_resamples, n_samples): boolean, if used with resampling set axis=1
        y_pred (n_samples,) or (n_resamples, n_samples): boolean predictions, if used with resampling set axis=1
        weights (n_samples,) or (n_resamples, n_samples): float weights, if used with resampling set axis=1
        axis: set to 1 / -1 if metric aggregation over n_samples shall happen in parallel for n_resamples.

    Returns:
        specificity:
            scalar if input is of shape (n_samples,);
            (n_resamples,) if input is of shape (n_resamples, n_samples) and aggregation is performed over axis=1

    """
    negatives = y_true == False  # noqa E712
    return ((negatives & (y_pred == False)) * weights).sum(axis) / (negatives * weights).sum(axis)  # noqa E712


def compute_ppv_vectorized(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """Vectorized, fast implementation without boilerplate to be used for resampling based CIs and hypothesis tests

    Args:
        y_true (n_samples,) or (n_resamples, n_samples): boolean, if used with resampling set axis=1
        y_pred (n_samples,) or (n_resamples, n_samples): boolean predictions, if used with resampling set axis=1
        weights (n_samples,) or (n_resamples, n_samples): float weights, if used with resampling set axis=1
        axis: set to 1 / -1 if metric aggregation over n_samples shall happen in parallel for n_resamples.

    Returns:
        ppv:
            scalar if input is of shape (n_samples,);
            (n_resamples,) if input is of shape (n_resamples, n_samples) and aggregation is performed over axis=1

    """
    true_positives = ((y_true & y_pred) * weights).sum(axis)
    false_positives = (((y_true == False) & y_pred) * weights).sum(axis)  # noqa E712
    return true_positives / (true_positives + false_positives)


def value_and_ci_from_array(central_value: float, samples: np.ndarray, alpha: float) -> ValueWithCI:
    ci_low, ci_upp = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return ValueWithCI(central_value, ci_low, ci_upp)


def compute_operating_point_from_pooled_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    name: str = "",
) -> OperatingPoint:
    """
    Compute sensitivity (recall), specificity and ppv (precision) from an array of boolean labels and predictions
    together with bootstrapped CIs.

    Bootstrapping is performed via a customised, vectorized implementation to speed things up significantly.

    Args:
        y_true (n_samples,): boolean label array
        y_pred (n_samples,) or (n_samples, n_pred): boolean prediction array from one or more predictors such as readers
            if multiple predictions are provided per sample, those are pooled
        sample_weights (n_samples,): Optional weights from for instance inverse probability weighting
                        which compensate for oversampled subsets of the data.
                        see for instance
                        McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H.,
                        Back, T., Chesus, M., Corrado, G.C., Darzi, A. and Etemadi, M., 2020. International
                        evaluation of an AI system for breast cancer screening. Nature, 577(7788), pp.89-94.
        n_resamples: the number of samples with replacement for the bootstrap CIs, set to zero in order to skip the
            computation of confidence intervals
        alpha: the significance level
        name: for the operating point

    Returns:
        sensitivity with CI
        specificity with CI
        ppv with CI

    """

    if y_pred.ndim == 2:
        n_samples, n_pred = y_pred.shape
        # Compute the average metrics by concatenating the predictions
        sens = compute_sensitivity_vectorized(
            np.repeat(y_true, n_pred), y_pred.flatten(), np.repeat(sample_weights, n_pred)
        )
        spec = compute_specificity_vectorized(
            np.repeat(y_true, n_pred), y_pred.flatten(), np.repeat(sample_weights, n_pred)
        )
        ppv = compute_ppv_vectorized(np.repeat(y_true, n_pred), y_pred.flatten(), np.repeat(sample_weights, n_pred))
    else:
        sens = compute_sensitivity_vectorized(y_true, y_pred, sample_weights)
        spec = compute_specificity_vectorized(y_true, y_pred, sample_weights)
        ppv = compute_ppv_vectorized(y_true, y_pred, sample_weights)

    if n_resamples > 0:
        y_true_resampled, y_pred_resampled, weights_resampled = generate_resamples(
            y_true, y_pred, sample_weights, n_resamples
        )

        sens_resampled = compute_sensitivity_vectorized(y_true_resampled, y_pred_resampled, weights_resampled, axis=1)
        sensitivity_with_ci = value_and_ci_from_array(sens, sens_resampled, alpha)

        spec_resampled = compute_specificity_vectorized(y_true_resampled, y_pred_resampled, weights_resampled, axis=1)
        specificity_with_ci = value_and_ci_from_array(spec, spec_resampled, alpha)

        ppv_resampled = compute_ppv_vectorized(y_true_resampled, y_pred_resampled, weights_resampled, axis=1)
        ppv_with_ci = value_and_ci_from_array(ppv, ppv_resampled, alpha)

    else:
        sensitivity_with_ci = ValueWithCI(value=sens, ci_low=None, ci_upp=None)
        specificity_with_ci = ValueWithCI(value=spec, ci_low=None, ci_upp=None)
        ppv_with_ci = ValueWithCI(value=ppv, ci_low=None, ci_upp=None)

    return OperatingPoint(name=name, sensitivity=sensitivity_with_ci, specificity=specificity_with_ci, ppv=ppv_with_ci)


def compute_sensitivity_from_pooled_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> ValueWithCI:
    """
    Compute sensitivity (recall) from an array of boolean labels and predictions together with bootstrapped CIs.

    Bootstrapping is performed via a customised, vectorized implementation to speed things up significantly.

    Args:
        y_true (n_samples,): boolean label array
        y_pred (n_samples,) or (n_pred, n_samples): boolean prediction array from one or more predictors such as readers
            if multiple predictions are provided per sample, those are pooled
        sample_weights (n_samples,): Optional weights from for instance inverse probability weighting
                        which compensate for oversampled subsets of the data.
                        see for instance
                        McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H.,
                        Back, T., Chesus, M., Corrado, G.C., Darzi, A. and Etemadi, M., 2020. International
                        evaluation of an AI system for breast cancer screening. Nature, 577(7788), pp.89-94.
        n_resamples: the number of samples with replacement for the bootstrap CIs; set to zero in order to skip CI
            computation
        alpha: the significance level

    Returns:
        sensitivity with CI
    """
    return _compute_metric_from_pooled_predictions(
        compute_sensitivity_vectorized,
        y_true,
        y_pred,
        sample_weights,
        n_resamples,
        alpha,
    )


def compute_specificity_from_pooled_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> ValueWithCI:
    """
    Compute specificity from an array of boolean labels and predictions together with bootstrapped CIs.

    Bootstrapping is performed via a customised, vectorized implementation to speed things up significantly.

    Args:
        y_true (n_samples,): boolean label array
        y_pred (n_samples,) or (n_pred, n_samples): boolean prediction array from one or more predictors such as readers
            if multiple predictions are provided per sample, those are pooled
        sample_weights (n_samples,): Optional weights from for instance inverse probability weighting
                        which compensate for oversampled subsets of the data.
                        see for instance
                        McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H.,
                        Back, T., Chesus, M., Corrado, G.C., Darzi, A. and Etemadi, M., 2020. International
                        evaluation of an AI system for breast cancer screening. Nature, 577(7788), pp.89-94.
        n_resamples: the number of samples with replacement for the bootstrap CIs; set to zero in order to skip CI
            computation
        alpha: the significance level

    Returns:
        specificity with CI
    """
    return _compute_metric_from_pooled_predictions(
        compute_specificity_vectorized,
        y_true,
        y_pred,
        sample_weights,
        n_resamples,
        alpha,
    )


def _compute_metric_from_pooled_predictions(
    compute_metric_vectorized_fn: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> ValueWithCI:
    """
    Compute a metric from an array of boolean labels and predictions together with bootstrapped CIs.

    Bootstrapping is performed via a customised, vectorized implementation to speed things up significantly.

    Args:
        compute_metric_vectorized_fn: either of compute_{sensitivity, specificity, ...}_vectorized
        y_true (n_samples,): boolean label array
        y_pred (n_samples,) or (n_pred, n_samples): boolean prediction array from one or more predictors such as readers
            if multiple predictions are provided per sample, those are pooled
        sample_weights (n_samples,): Optional weights from for instance inverse probability weighting
                        which compensate for oversampled subsets of the data.
                        see for instance
                        McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H.,
                        Back, T., Chesus, M., Corrado, G.C., Darzi, A. and Etemadi, M., 2020. International
                        evaluation of an AI system for breast cancer screening. Nature, 577(7788), pp.89-94.
        n_resamples: the number of samples with replacement for the bootstrap CIs; set to zero in order to skip CI
            computation
        alpha: the significance level

    Returns:
        metric with CI
    """
    if y_pred.ndim == 2:
        n_samples, n_pred = y_pred.shape
        # Compute the average metric by concatenating the predictions
        metric = compute_metric_vectorized_fn(
            np.repeat(y_true, n_pred), y_pred.flatten(), np.repeat(sample_weights, n_pred)
        )
    else:
        metric = compute_metric_vectorized_fn(y_true, y_pred, sample_weights)

    if n_resamples > 0:
        y_true_resampled, y_pred_resampled, weights_resampled = generate_resamples(
            y_true, y_pred, sample_weights, n_resamples
        )
        metric_resampled = compute_metric_vectorized_fn(y_true_resampled, y_pred_resampled, weights_resampled, axis=1)
        metric_with_ci = value_and_ci_from_array(metric, metric_resampled, alpha)
    else:
        metric_with_ci = ValueWithCI(value=metric, ci_low=None, ci_upp=None)

    return metric_with_ci


def generate_resamples(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weights: np.ndarray, n_resamples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to generate samples with replacement for bootstrapping and optionally pool from predictions

    Args:
        y_true (n_samples,)
        y_pred (n_samples,) or (n_samples, n_pred)
        sample_weights (n_samples,)
        n_resamples: the number of samples to draw

    Returns:
        y_true_resampled (n_resamples, n_samples)
        y_pred_resampled (n_resamples, n_samples): if multiple predictions per sample are provided, pooling happens
            before every resampling step
        sample_weights_resampled (n_resamples, n_samples)

    """
    n_samples = len(y_true)

    # Bootstrap resamples
    resamples = np.random.randint(low=0, high=n_samples, size=(n_resamples, n_samples))

    # (n_resamples, n_samples)
    y_true_resampled = np.array([y_true[resample] for resample in resamples])
    weights_resampled = np.array([sample_weights[resample] for resample in resamples])

    if y_pred.ndim == 2:
        n_pred = y_pred.shape[1]
        assert y_pred.shape == (
            n_samples,
            n_pred,
        ), f"Inconsistent shape of y_pred {y_pred.shape}, expecting {(n_samples, n_pred)}."
        # (n_resamples, n_samples, n_pred)
        pred_samples = tf.keras.utils.to_categorical(
            y=np.random.randint(low=0, high=n_pred, size=(n_resamples, n_samples)),
            num_classes=n_pred,
            dtype=y_true.dtype,
        )
        if n_samples == 1:
            # Edge case for local data/small samples: Keras' `to_categorical` is removing trailing singleton dimensions
            # because it is usually meant to convert from an integer class vector to a binary class matrix.
            pred_samples = pred_samples[:, None, :]
        assert pred_samples.shape == (n_resamples, n_samples, n_pred)
        y_pred_resampled = np.array(
            [y_pred[pred_sample][resample] for pred_sample, resample in zip(pred_samples, resamples)]
        )
    else:
        y_pred_resampled = np.array([y_pred[resample] for resample in resamples])

    return y_true_resampled, y_pred_resampled, weights_resampled


class PermutationTestResult(NamedTuple):
    delta: float
    p_value: float


def permutation_test(
    y_true: np.ndarray,
    y_rad_plus_ai: np.ndarray,
    y_rad: np.ndarray,
    sample_weights: np.ndarray,
    metric_fns: Dict[str, Callable],
    n_permutations: int = 10000,
) -> Dict[str, PermutationTestResult]:
    """Computes a p-value by comparing the observed difference in a metric with the randomization distribution

    Args:
        y_true: (n_samples,), boolean labels
        y_rad_plus_ai: (n_samples, n_reads) boolean predictions for each study and read combination
        y_rad: (n_samples, n_reads) boolean predictions for each study and reader
        sample_weights: (n_samples,) float, per sample weights from inverse probability weighing
        metric_fns: each keyed callable should accept `y_true`, `y_pred`, `sample_weights` as arguments
            CAUTION: passing in multiple functions may require correcting for the multiple comparisons problem
                https://en.wikipedia.org/wiki/Multiple_comparisons_problem
                Though the correction due to multiple comparisons may happen externally and this function is still
                helpful to avoid redundant resampling.
        n_permutations: the number of samples to construct the empirical randomization distribution from. For every
            permutation, the same read (per study) is pooled from y_rad_plus_ai and y_rad

    Returns:
        {"metric_name": PermutationTestResult(delta, p_value), ...}:
            for each metric_fn the observed difference as well as the two-sided p-value

    References:
        Foundations in chapter 15, and optionally 16 of the book:
            Efron, Tibshirani (1994): An Introduction to the Bootstrap
        An application for our use case (compare AI system vs. reader performances on paired data) is given in:
            Mckinney, S. M. et al. (2020). International evaluation of an AI system for breast cancer screening.
                Nature, 577(January). https://doi.org/10.1038/s41586-019-1799-6

    """

    n_samples, n_pred = y_rad.shape
    n_groups = 2  # We have predictions from two groups to compare: y_rad_plus_ai vs. y_rad
    assert y_rad_plus_ai.shape == (
        n_samples,
        n_pred,
    ), f"Expecting both y_rad_plus_ai {y_rad_plus_ai.shape} and y_rad {y_rad.shape} to have the same shape."

    # 1. Generate sampling matrix (n_permutations, n_samples, n_pred) to select from pool of predictions / readers
    choose_pred_samples = tf.keras.utils.to_categorical(
        y=np.random.randint(low=0, high=n_pred, size=(n_permutations, n_samples)),
        num_classes=n_pred,
        dtype=y_true.dtype,
    )
    # 2. Generate a choice array (n_permutations, n_samples, n_group) for the permutations
    choose_group_samples = tf.keras.utils.to_categorical(
        y=np.random.randint(low=0, high=n_groups, size=(n_permutations, n_samples)),
        num_classes=n_groups,
        dtype=y_true.dtype,
    )
    # 3. Perform prediction pooling followed by permutation (random reassignment to one of the two groups)
    y_pred_group = np.stack([y_rad_plus_ai, y_rad], axis=-1)  # (n_samples, n_pred, n_group)
    # (n_permutations, n_samples)
    y_rad_plus_ai_permuted = np.array(
        [
            y_pred_group[choose_pred][choose_group]
            for choose_pred, choose_group in zip(choose_pred_samples, choose_group_samples)
        ]
    )
    # (n_permutations, n_samples) pooling the exact same prediction per permutation, but choosing the other group
    y_rad_permuted = np.array(
        [
            y_pred_group[choose_pred][~choose_group]
            for choose_pred, choose_group in zip(choose_pred_samples, choose_group_samples)
        ]
    )
    # Same labels and weights for every permutation, we just precompute for reuse for vectorized metric_fns
    y_true_permuted = np.tile(y_true, (n_permutations, 1))  # (n_permutations, n_samples)
    weights_permuted = np.tile(sample_weights, (n_permutations, 1))  # (n_permutations, n_samples)
    #  4. apply metric_fns in a vectorised fashion
    delta_randomized = defaultdict()
    for metric_name, metric_fn in metric_fns.items():
        delta_randomized[metric_name] = metric_fn(
            y_true_permuted, y_rad_plus_ai_permuted, weights_permuted, axis=1
        ) - metric_fn(y_true_permuted, y_rad_permuted, weights_permuted, axis=1)

    # For the observed delta, we compute the average for which we have to repeat the samples for every `pred`
    y_true_repeated = np.repeat(y_true, n_pred)  # (n_samples * n_pred,)
    weights_repeated = np.repeat(sample_weights, n_pred)  # (n_samples * n_pred,)

    delta_observed = {
        metric_name: metric_fn(y_true_repeated, y_rad_plus_ai.flatten(), weights_repeated)
        - metric_fn(y_true_repeated, y_rad.flatten(), weights_repeated)
        for metric_name, metric_fn in metric_fns.items()
    }

    p_values = {k: sum(abs(delta_randomized[k]) >= abs(delta_observed[k])) / n_permutations for k in delta_observed}

    return {k: PermutationTestResult(delta=delta_observed[k], p_value=p_values[k]) for k in delta_observed}
