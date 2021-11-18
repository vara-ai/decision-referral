"""
Run the decision referral evaluation in standalone fashion on a validation (and test) dataset.

The script here tests whether decision referral performance translates from a validation dataset
to a test dataset (using the thresholds determined on the validation set).
"""
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import hydra
from omegaconf import MISSING, OmegaConf
import numpy as np

from conf import lib as config_lib
from decision_referral.core import DecisionReferral
from decision_referral import core as dr, io_util, subgroups, types, util
from decision_referral.types import DecisionReferralInputs, DecisionReferralResult


@config_lib.register_config(name="base_evaluate")
@dataclass
class EvaluateConfig:
    threshold_setting_dataset: types.Dataset = MISSING
    test_dataset: Optional[types.Dataset] = None
    analyze_subgroups: bool = True
    seed: Optional[int] = None
    decision_referral: dr.DecisionReferralConfig = MISSING


@hydra.main(config_path="conf", config_name="evaluate")
def run_decision_referral_analysis_process(cfg: EvaluateConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # 1. Run the analysis on the VALIDATION set (or load results if we've already done it)
    validation_dir = util.get_validation_dir(cfg.threshold_setting_dataset)
    validation_save_dir = util.get_decision_referral_dir(validation_dir)
    if os.path.exists(validation_save_dir):
        decision_referral_result_val = DecisionReferralResult.load(validation_save_dir)
        example_op = util.get_max_sensitive_geq_specific_op(decision_referral_result_val)
    else:
        decision_referral_val = DecisionReferral(
            DecisionReferralInputs.from_disk(cfg.threshold_setting_dataset),
            cfg=cfg.decision_referral,
        )
        decision_referral_result_val = decision_referral_val.result
        example_op = util.get_max_sensitive_geq_specific_op(decision_referral_result_val)
        save_decision_referral(decision_referral_val, example_op, validation_save_dir, cfg)
        if cfg.analyze_subgroups:
            analyze_subgroups(
                subgroups.SubgroupMasks.from_disk(cfg.threshold_setting_dataset),
                decision_referral_val,
                example_op,
                validation_save_dir,
                cfg.decision_referral.run_hypothesis_tests,
            )

    # 2. Use the thresholds determined on the validation set to do decision referral on test set
    if cfg.test_dataset is not None:
        test_save_dir = os.path.join(
            util.get_test_dir(cfg.threshold_setting_dataset, cfg.test_dataset), "decision_referral"
        )
        if not os.path.exists(test_save_dir):
            decision_referral_test = DecisionReferral.from_result(
                DecisionReferralInputs.from_disk(cfg.test_dataset),
                cfg=cfg.decision_referral,
                decision_referral_result_val=decision_referral_result_val,
            )
            save_decision_referral(decision_referral_test, example_op, test_save_dir, cfg)
            if cfg.analyze_subgroups:
                analyze_subgroups(
                    subgroups.SubgroupMasks.from_disk(cfg.test_dataset),
                    decision_referral_test,
                    example_op,
                    test_save_dir,
                    cfg.decision_referral.run_hypothesis_tests,
                )
        else:
            raise Exception("Evaluation has already been computed for your selected params")


def save_decision_referral(
    decision_referral: DecisionReferral, example_op: str, output_dir: str, cfg: EvaluateConfig
) -> None:
    """Saves decision referral results and all other artefacts needed for downstream computation."""

    os.makedirs(output_dir, exist_ok=True)

    config_lib.save_config(output_dir, cfg)

    # save important decision referral attributes (the whole object is >1GB & not necessary)
    io_util.save_to_pickle(
        decision_referral.weights, os.path.join(output_dir, "weights.pkl"), protocol=pickle.HIGHEST_PROTOCOL
    )
    io_util.save_to_pickle(
        decision_referral.y_true, os.path.join(output_dir, "y_true.pkl"), protocol=pickle.HIGHEST_PROTOCOL
    )
    io_util.save_to_pickle(
        decision_referral.y_rad, os.path.join(output_dir, "y_rad.pkl"), protocol=pickle.HIGHEST_PROTOCOL
    )
    io_util.save_to_pickle(
        decision_referral.dr_inputs.y_score_nt,
        os.path.join(output_dir, "y_score_normal_triaging.pkl"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    io_util.save_to_pickle(example_op, os.path.join(output_dir, "example_op.pkl"), protocol=pickle.HIGHEST_PROTOCOL)

    # save decision referral result
    decision_referral_result = decision_referral.result
    decision_referral_result.save(output_dir)

    # compute & save combined y_scores
    if example_op == util.NAME_AI:
        # so far just occasionally happens for small test data
        y_score_combined = decision_referral.dr_inputs.y_score_sn
    else:
        min_sensitivity, min_specificity = util.get_sensitivity_and_specificity_from_operating_pair(example_op)
        result_tuple = decision_referral_result.get_result_tuple(min_sensitivity, min_specificity)
        _, y_score_combined = decision_referral.determine_decision_referral_selection_and_scores(
            result_tuple.lower_threshold, result_tuple.upper_threshold
        )
    io_util.save_to_pickle(
        y_score_combined,
        os.path.join(output_dir, "y_score_combined.pkl"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def analyze_subgroups(
    subgroup_masks: subgroups.SubgroupMasks,
    decision_referral: DecisionReferral,
    example_op: str,
    output_dir: str,
    run_hypothesis_tests: bool,
) -> None:
    """Compute subgroup sensitivities with optional p-values and save the results"""
    subgroup_sensitivities, subgroup_specificities = subgroups.compute_subgroup_results(
        subgroup_masks,
        decision_referral,
        example_op,
        run_hypothesis_tests=run_hypothesis_tests,
    )
    io_util.save_to_pickle(
        subgroup_sensitivities,
        os.path.join(output_dir, "subgroup_sensitivities.pkl"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    io_util.save_to_pickle(
        subgroup_specificities,
        os.path.join(output_dir, "subgroup_specificities.pkl"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    run_decision_referral_analysis_process()
