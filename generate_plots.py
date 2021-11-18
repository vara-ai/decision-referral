"""
Uses precomputed decision referral results to generate plots and artefacts for use
in in-house analysis, publications, and regulatory submissions.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import hydra
from omegaconf import MISSING, OmegaConf

from decision_referral import figures, io_util, output_table, types
from conf import lib as config_lib

from decision_referral.subgroups import SubgroupSensitivities, SubgroupSpecificities
from decision_referral.types import DecisionReferralResult
from decision_referral.util import (
    get_dataset_name_from_dir,
    get_decision_referral_dir,
    get_plots_dir,
    get_sensitivity_and_specificity_from_operating_pair,
    get_test_dir,
    get_validation_dir,
    NAME_AI,
    NAME_RAD,
    NAME_DR,
)


@config_lib.register_config(name="base_generate_plots")
@dataclass
class GeneratePlotsConfig:
    threshold_setting_dataset: types.Dataset = MISSING
    test_dataset: Optional[types.Dataset] = None
    # operating_pairs: list of operating pairs to use for plotting
    # (must be a subset of the operating pairs passed to evaluate.py)
    operating_pairs: List[str] = MISSING
    # n_bootstrap: number of bootstrap samples to use plotting ROC curve CIs
    n_bootstrap: int = MISSING


def format_drr_df(df):
    df = df.sort_values(by=["sensitivity_value"], ascending=True)
    df = output_table.format_metric_and_ci_columns(df)
    df = output_table.format_delta_and_p_value_columns(df)
    df["triaging performance"] = df.apply(lambda x: f"{x.rule_out:.1%}", axis=1)
    df = df.drop(
        axis=1,
        columns=[
            "delta",
            "ppv_value",
            "ppv_ci_low",
            "ppv_ci_upp",
            "delta_ppv",
            "selection_confident",
            "rule_out",
            "min_sensitivity",
            "min_specificity",
            "lower_threshold",
            "upper_threshold",
            "is_super_human",
        ],
    )
    return df


def generate_results_table_subgroups(
    subgroup_sensitivities: SubgroupSensitivities,
    subgroup_specificities: SubgroupSpecificities,
    op_name: str,
    output_dir: str,
) -> None:
    subgroup_sensitivities.to_data_frame().to_csv(
        os.path.join(output_dir, f"subgroup_sensitivities_{op_name.lower()}.csv"), index=False
    )
    subgroup_specificities.to_data_frame().to_csv(
        os.path.join(output_dir, f"subgroup_specificities_{op_name.lower()}.csv"), index=False
    )


def generate_plots(input_dir: str, cfg: GeneratePlotsConfig) -> None:
    """Generate plots & other artefacts for publication/regulatory/analysis of decision referral
    Args:
        input_dir: a directory where decision referral artefacts were saved (i.e. .../decision_referral)
        cfg: specifies n_bootstrap samples for ROC CIs and the operating pairs to be used for tables and figures
    """

    # work out directories + whether we've already generated plots before
    dataset_name = get_dataset_name_from_dir(input_dir)
    if not os.path.exists(input_dir):
        raise Exception(f"You must run `evaluate.py` on {dataset_name} before you can generate decision referral plots")
    output_dir = get_plots_dir(input_dir)
    if os.path.exists(output_dir):
        logging.info(f"Plots have already been generated for {dataset_name} ({output_dir})")
        return

    os.makedirs(output_dir, exist_ok=True)
    config_lib.save_config(output_dir, cfg)

    # load decision referral result & filter to subset of operating pairs
    decision_referral_result = DecisionReferralResult.load(input_dir)
    filter_set = set(cfg.operating_pairs) | {NAME_RAD, NAME_DR, NAME_AI}
    decision_referral_result_filtered = decision_referral_result.filter(filter_set)

    # load all other decision referral attributes
    weights = io_util.load_from_pickle(os.path.join(input_dir, "weights.pkl"))
    y_true = io_util.load_from_pickle(os.path.join(input_dir, "y_true.pkl"))
    y_rad = io_util.load_from_pickle(os.path.join(input_dir, "y_rad.pkl"))
    y_score_normal_triaging = io_util.load_from_pickle(os.path.join(input_dir, "y_score_normal_triaging.pkl"))
    y_score_combined = io_util.load_from_pickle(os.path.join(input_dir, "y_score_combined.pkl"))
    example_op = io_util.load_from_pickle(os.path.join(input_dir, "example_op.pkl"))

    # Figure 3 & 4 A
    figures.plot_system_performance(
        y_true,
        y_score_normal_triaging,
        weights,
        decision_referral_result_filtered,
        rad_plus_ai_example_op=example_op,
        output_dir=output_dir,
        n_bootstrap=cfg.n_bootstrap,
    )

    # Figure 3 & 4 B
    min_sensitivity, min_specificity = get_sensitivity_and_specificity_from_operating_pair(example_op)
    figures.roc_plot_confident(
        decision_referral_result,  # NB: can't use the filtered drr because it might not have example_op
        y_score_combined,
        y_true,
        y_rad,
        weights,
        min_sensitivity,
        min_specificity,
        output_dir=output_dir,
        n_bootstrap=cfg.n_bootstrap,
    )

    # generate results table (just a CSV of the results dataframe)
    decision_referral_result_filtered.to_csv(output_dir, format_func=format_drr_df)

    # subgroup analysis
    try:
        subgroup_sensitivities = io_util.load_from_pickle(os.path.join(input_dir, "subgroup_sensitivities.pkl"))
        subgroup_specificities = io_util.load_from_pickle(os.path.join(input_dir, "subgroup_specificities.pkl"))
    except FileNotFoundError:
        logging.warning(f"Subgroup artefacts not found in {input_dir}")
        return
    # Figure 5
    figures.subgroup_performance_plot(subgroup_sensitivities, min_sensitivity, min_specificity, output_dir=output_dir)
    generate_results_table_subgroups(
        subgroup_sensitivities, subgroup_specificities, op_name=example_op, output_dir=output_dir
    )

    logging.info(f"Plots successfully saved to {output_dir}")


@hydra.main(config_path="conf", config_name="generate-plots")
def main(cfg: GeneratePlotsConfig):
    print(OmegaConf.to_yaml(cfg))

    threshold_setting_dr_dir = get_decision_referral_dir(get_validation_dir(cfg.threshold_setting_dataset))
    generate_plots(threshold_setting_dr_dir, cfg)

    if cfg.test_dataset is not None:
        test_dr_dir = get_decision_referral_dir(get_test_dir(cfg.threshold_setting_dataset, cfg.test_dataset))
        generate_plots(test_dr_dir, cfg)


if __name__ == "__main__":
    main()
