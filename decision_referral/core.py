import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import delayed, Parallel
from omegaconf import MISSING

from conf import lib as config_lib
from decision_referral import statistics_util, thresholds, types as dr_types, util as dr_util
from decision_referral.types import OperatingPoint

GROUP = "dr_config"


@config_lib.register_config(name="base_decision_referral")
@dataclass
class DecisionReferralConfig:
    # Each operating pair in the list is defined by a name in one of the following 3 schema:
    # 'NT@{sensitivity}', 'SN@{specificity}', 'NT@{sensitivity}+SN@{specificity}'
    operating_pairs: List[str] = MISSING

    # The number of bootstrap samples to draw in order to generate confidence bounds
    n_bootstrap: int = MISSING

    # Whether to run tests that assess the statistical significance of the difference
    # in metrics between radiologist plus AI and radiologist alone.
    run_hypothesis_tests: bool = MISSING

    reweigh_samples: bool = True

    # The number of CPUs to use for computing different decision referral configurations
    # in parallel. By default, parallelization is turned off to limit the memory usage for
    # large datasets. Use -1 to use all CPUs.
    n_jobs: int = 1


@config_lib.register_config(group=GROUP)
class Maximal(DecisionReferralConfig):
    operating_pairs = dr_util.OPERATING_PAIRS
    n_bootstrap = 1000
    run_hypothesis_tests = True


@config_lib.register_config(group=GROUP)
class Minimal(DecisionReferralConfig):
    operating_pairs = ["NT@0.97+SN@0.98"]
    n_bootstrap = 10
    run_hypothesis_tests = False


class DecisionReferralUndefinedException(Exception):
    # Raised when decision referral logic can't be computed, as normal triaging and safety net would overlap.
    pass


class DecisionReferral:
    """Stratifies dataset by confident vs. unconfident predictions, across a range of sensitivities/specificities

    and performs a retrospective comparison of screening performance metrics comparing a single reader against algorithm
    plus single reader. For a two reader setup, we sample from / average the two in order to simulate the effect
    of decision referral on a single reader.

    This analyzer allows a two stage procedure as follows:
    (1) Run it without specifying the optional kwarg `thresholds_by_operating_pairs` and the
    thresholds will be computed based on the `min_sensitivities` and `min_specificities` of the respective operating
    pair on dataset A (e.g. some INTERNAL_VALIDATION_SET).
    (2) Use the resulting `thresholds_by_operating_pair` attribute in order to initialize another analyzer on a
    different dataset B (e.g. some INTERNAL_TEST_SET) to check whether the thresholds generalise to another dataset in
    terms of performance.

    """

    def __init__(
        self,
        dr_inputs: dr_types.DecisionReferralInputs,
        cfg: DecisionReferralConfig,
        thresholds_by_operating_pairs: Optional[Dict[str, thresholds.ThresholdPair]] = None,
    ):
        """

        Args:
            dr_inputs: everything that is needed for decision referral, per dataset. That is ground truth cancer labels,
                model assessments, radiologist assessments, ...
            cfg: the configuration object for all things decision referral
            thresholds_by_operating_pairs: optional map from operating pair name to thresholds.
                Pass thresholds when using test data
        """
        self.dr_inputs = dr_inputs
        self.cfg = cfg

        self.y_true = self.dr_inputs.y_true
        # (n_samples, n_reads)
        self.y_rad = np.c_[self.dr_inputs.y_read_1, self.dr_inputs.y_read_2]

        self.n_bootstrap = cfg.n_bootstrap
        self.run_hypothesis_tests = cfg.run_hypothesis_tests
        # We intentionally take the provided weights on the whole dataset such that the dataset as a whole is
        # representative for the screening population
        if cfg.reweigh_samples:
            self.weights = self.dr_inputs.weights
        else:
            self.weights = np.ones_like(self.y_true)

        # We need the radiologist sensitivity to configure standalone AI
        self.rad_performance = self._get_rad_performance()

        if thresholds_by_operating_pairs is None:
            # We don't know the thresholds yet and compute them on the data
            thresholds_by_operating_pairs = thresholds.compute_thresholds(
                cfg.operating_pairs,
                self.dr_inputs.y_score_nt,
                self.dr_inputs.y_score_sn,
                self.dr_inputs.y_true,
                self.weights,
                self.rad_performance.sensitivity.value,
            )
        else:
            # Thresholds were computed a priori and passed in.
            assert all(
                [op in thresholds_by_operating_pairs for op in cfg.operating_pairs]
            ), "Didn't get thresholds for all operating pairs."
            assert dr_util.NAME_AI in thresholds_by_operating_pairs, "Didn't get a threshold for standalone AI."

        self.thresholds_by_operating_pairs = thresholds_by_operating_pairs

    @functools.cached_property
    def result(self) -> dr_types.DecisionReferralResult:
        return dr_types.DecisionReferralResult(self._compute_result_tuples())

    @classmethod
    def from_result(
        cls,
        dr_inputs: dr_types.DecisionReferralInputs,
        cfg: DecisionReferralConfig,
        decision_referral_result_val: dr_types.DecisionReferralResult,
    ) -> "DecisionReferral":
        """Initialise decision referral with thresholds inferred from validation data result."""
        return cls(
            dr_inputs,
            cfg=cfg,
            thresholds_by_operating_pairs=thresholds.get_thresholds_by_operating_pairs_from_result(
                decision_referral_result_val
            ),
        )

    def _compute_result_tuples(self) -> List[dr_types.ResultsTuple]:
        """
        For each operating point or pair, split the dataset into a confident and an unconfident regime and build
        predictions of the different parties (radiologist, model(s), radiologist + model(s)).
        """
        # Initialize results with raw radiologist and standalone AI results
        results = [self.rad_performance, self._get_ai_performance()]

        def _compute_rad_plus_ai_result(
            op_name: str, lower_threshold: float, upper_threshold: float
        ) -> Optional[dr_types.ResultsTuple]:
            try:
                (selection_confident, y_score_combined) = self.determine_decision_referral_selection_and_scores(
                    lower_threshold, upper_threshold
                )
            except DecisionReferralUndefinedException:
                return None
            min_sensitivity, min_specificity = dr_util.get_sensitivity_and_specificity_from_operating_pair(op_name)
            return self._get_rad_plus_ai_performance(
                min_sensitivity,
                min_specificity,
                lower_threshold,
                upper_threshold,
                ~selection_confident,
                self.rad_performance,
            )

        results += [
            rad_plus_ai_result
            # we use the threading backend because the closure can't be pickled
            # and most computations are numpy anyway which releases the GIL
            for rad_plus_ai_result in Parallel(n_jobs=self.cfg.n_jobs, backend="threading", verbose=10)(
                delayed(_compute_rad_plus_ai_result)(op, lower_threshold, upper_threshold)
                for op, (lower_threshold, upper_threshold) in self.thresholds_by_operating_pairs.items()
            )
            if rad_plus_ai_result is not None
        ]

        return results

    def determine_decision_referral_selection_and_scores(
        self, lower_threshold: Optional[float], upper_threshold: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine where the model is confident enough to make predictions and compute the combined model scores.

        Args:
            lower_threshold: lower threshold for the normal triaging, if None, normal triaging does not apply
            upper_threshold: upper threshold for the safety net, if None, safety does not apply

        Returns:
            Tuple containing:
                selection_confident: Boolean array indicating where the model is confident enough
                y_score_combined: Combined prediction array of normal triaging and safety net model
        """
        # For each algorithmic operating point, generate two strata:
        # (1) for algorithmic regime, (2) for decision referral regime.

        # Initialize with normal triaging & safety net turned off
        selection_sens, selection_spec = np.zeros_like(self.y_true), np.zeros_like(self.y_true)
        if lower_threshold is not None:
            selection_sens = self.dr_inputs.y_score_nt < lower_threshold
        if upper_threshold is not None:
            selection_spec = self.dr_inputs.y_score_sn > upper_threshold
        if any(selection_sens & selection_spec):
            raise DecisionReferralUndefinedException(
                "There are cases above the safety net and below the normal triaging threshold."
            )
        selection_confident = selection_sens ^ selection_spec
        # Combine predictions from both models into a single array for plotting & analysis purposes. Assume that the
        # normal triaging model takes over in the unconfident regime (for evaluation purpose only).
        y_score_combined = np.copy(self.dr_inputs.y_score_nt)
        # Overwrite predictions in specific regime
        y_score_combined[selection_spec] = self.dr_inputs.y_score_sn[selection_spec]
        return selection_confident, y_score_combined

    def _get_rad_performance(self) -> dr_types.ResultsTuple:
        """Get average performance of radiologists"""

        rad_op = statistics_util.compute_operating_point_from_pooled_predictions(
            y_true=self.y_true,
            y_pred=self.y_rad,
            sample_weights=self.weights,
            n_resamples=self.n_bootstrap,
            name=dr_util.NAME_RAD,
        )
        return dr_types.ResultsTuple(
            name=dr_util.NAME_RAD,
            sensitivity=rad_op.sensitivity,
            specificity=rad_op.specificity,
            ppv=rad_op.ppv,
            delta=None,
            # By definition, the algorithm was confident nowhere as the radiologist takes all decisions
            selection_confident=np.zeros_like(self.y_true, dtype=np.bool),
        )

    def _get_ai_performance(self) -> dr_types.ResultsTuple:
        """Get standalone AI performance."""

        # So far, normal triaging models achieve higher AUC for standalone usage,
        # this might change in the future
        y_score = self.dr_inputs.y_score_nt

        # Exemplary operating point for standalone AI usage
        # 1. Pop AI threshold as we don't need it down the road
        # 2. 0th because we configure by sensitivity
        threshold = self.thresholds_by_operating_pairs.pop(dr_util.NAME_AI)[0]
        self.y_ai = y_score >= threshold

        # Repeat the AI predictions to match the number of reads for the permutation test and OP computation
        y_ai_repeated = np.repeat(
            self.y_ai[:, None],
            repeats=self.y_rad.shape[1],
            axis=1,
        )

        # Pooling wouldn't be needed here, but we use the function in order to get CIs via bootstrapping
        ai_op = statistics_util.compute_operating_point_from_pooled_predictions(
            self.y_true, y_ai_repeated, self.weights, n_resamples=self.n_bootstrap, name=dr_util.NAME_AI
        )

        return dr_types.ResultsTuple(
            name=dr_util.NAME_AI,
            sensitivity=ai_op.sensitivity,
            specificity=ai_op.specificity,
            ppv=ai_op.ppv,
            delta=self.get_human_delta(
                ai_op,
                self.rad_performance,
                y_ai_repeated,
            ),
            # By definition, the algorithm was confident everywhere as the AI takes all decisions
            selection_confident=np.ones_like(self.y_true, dtype=np.bool),
            rule_out=statistics_util.compute_specificity_vectorized(
                y_true=self.y_true, y_pred=self.y_ai, weights=self.weights
            ),
            min_sensitivity=self.rad_performance.sensitivity.value,
            min_specificity=None,
            lower_threshold=threshold,
            upper_threshold=None,
        )

    def _get_rad_plus_ai_performance(
        self,
        min_sensitivity: Optional[float],
        min_specificity: Optional[float],
        lower_threshold: Optional[float],
        upper_threshold: Optional[float],
        selection_unconfident: np.ndarray,
        rad_performance: dr_types.ResultsTuple,
    ) -> dr_types.ResultsTuple:
        """Get performance of radiologist + algorithm"""
        y_rad_plus_ai = self.combine_rad_and_ai_predictions(selection_unconfident, lower_threshold, upper_threshold)

        dr_op = statistics_util.compute_operating_point_from_pooled_predictions(
            y_true=self.y_true,
            y_pred=y_rad_plus_ai,
            sample_weights=self.weights,
            n_resamples=self.n_bootstrap,
            name=dr_util.get_operating_pair_name(min_sensitivity, min_specificity),
        )

        # Rule out
        if lower_threshold is not None:
            rule_out = statistics_util.compute_specificity_vectorized(
                y_true=self.y_true, y_pred=self.dr_inputs.y_score_nt >= lower_threshold, weights=self.weights
            )
        else:
            rule_out = 0.0

        return dr_types.ResultsTuple(
            name=dr_util.get_operating_pair_name(min_sensitivity, min_specificity),
            sensitivity=dr_op.sensitivity,
            specificity=dr_op.specificity,
            ppv=dr_op.ppv,
            delta=self.get_human_delta(dr_op, rad_performance, y_rad_plus_ai),
            selection_confident=~selection_unconfident,
            rule_out=rule_out,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )

    def combine_rad_and_ai_predictions(
        self,
        selection_unconfident: np.ndarray,
        lower_threshold: Optional[float],
        upper_threshold: Optional[float],
    ) -> np.ndarray:
        """Build array of predictions: For each sample we take either the radiologist or model decision.

        Args:
            selection_unconfident: (n_studies,) boolean array; True for unconfident regime where
                algorithmic predictions are not confident enough and decisions are
                referred from algorithm to radiologist
            lower_threshold: for normal triaging
            upper_threshold: for safety net

        Returns:
            y_rad_plus_ai: boolean predictions from radiologist and model in respective regimes
                of size (n_studies, 2); we combine the model with each reader assessment for downstream analysis.

        """
        assert not (
            (lower_threshold is None) and (upper_threshold is None)
        ), "At least one threshold must be set, otherwise it does not make sense to combine RAD & AI predictions."

        # (n_samples, 2), one column for each read
        y_rad_plus_ai = np.zeros_like(self.y_rad, dtype=np.bool)

        # In the confident prediction regime, we assume the algorithmic predictions are accepted throughout
        if lower_threshold is not None:
            y_rad_plus_ai[self.dr_inputs.y_score_nt < lower_threshold] = False
        if upper_threshold is not None:
            y_rad_plus_ai[self.dr_inputs.y_score_sn > upper_threshold] = True

        # In the unconfident prediction regime, decisions are referred to radiologists
        y_rad_plus_ai[selection_unconfident] = self.y_rad[selection_unconfident]

        return y_rad_plus_ai

    def get_human_delta(
        self,
        op: OperatingPoint,
        rad_performance: dr_types.ResultsTuple,
        y_rad_plus_ai: Optional[np.ndarray] = None,
    ) -> dr_types.HumanDelta:

        delta_sensitivity = op.sensitivity.value - rad_performance.sensitivity.value
        delta_specificity = op.specificity.value - rad_performance.specificity.value
        delta_ppv = op.ppv.value - rad_performance.ppv.value

        delta_sensitivity_p_value = None
        delta_specificity_p_value = None

        if y_rad_plus_ai is not None and self.run_hypothesis_tests:
            test_result = statistics_util.permutation_test(
                self.y_true,
                y_rad_plus_ai=y_rad_plus_ai,
                y_rad=self.y_rad,
                sample_weights=self.weights,
                metric_fns={
                    dr_util.SENSITIVITY: statistics_util.compute_sensitivity_vectorized,
                    dr_util.SPECIFICITY: statistics_util.compute_specificity_vectorized,
                },
            )
            assert test_result[dr_util.SENSITIVITY].delta == delta_sensitivity, (
                f"Unexpected difference in delta sensitivity: "
                f"{test_result[dr_util.SENSITIVITY].delta} vs. {delta_sensitivity}"
            )
            assert test_result[dr_util.SPECIFICITY].delta == delta_specificity, (
                f"Unexpected difference in delta specificity: "
                f"{test_result[dr_util.SPECIFICITY].delta} vs. {delta_specificity}"
            )
            delta_sensitivity_p_value = test_result[dr_util.SENSITIVITY].p_value
            delta_specificity_p_value = test_result[dr_util.SPECIFICITY].p_value

        return dr_types.HumanDelta(
            is_super_human=(delta_sensitivity > 0) and (delta_specificity > 0) and (delta_ppv > 0),
            sensitivity=delta_sensitivity,
            specificity=delta_specificity,
            ppv=delta_ppv,
            sensitivity_p_value=delta_sensitivity_p_value,
            specificity_p_value=delta_specificity_p_value,
        )
