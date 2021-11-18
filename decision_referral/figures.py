import logging
import os
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from decision_referral import plot_util, statistics_util, subgroups
from decision_referral.plot_util import format_error_bar
from decision_referral.types import DecisionReferralResult, ValueWithCI
from decision_referral.util import (
    get_operating_pair_name,
    NAME_AI,
    NAME_RAD,
    NAME_DR,
)


COLOR_RAD = "#222130"  # as in workflow figure for radiologist
COLOR_AI = "#7273fb"  # as in workflow figure for AI
COLOR_RAD_PLUS_AI_EXAMPLE = "g"
COLOR_AI_STANDALONE = "y"

# Style all plots the same way
plot_context = "paper"
plot_font_scale = 1.0


@plot_util.seaborn_context(plot_context, plot_font_scale)
def plot_system_performance(
    y_true: np.ndarray,
    y_score_normal_triaging: np.ndarray,
    weights: np.ndarray,
    operating_points: DecisionReferralResult,
    rad_plus_ai_example_op: str,
    output_dir: str,
    n_bootstrap: int = 1000,
):
    """Plot in ROC space that compares the AI standalone, Rad & Rad+AI performance on the entire dataset"""
    # Sort the ops (motivated by the legend order)
    operating_points = sorted(
        operating_points, key=lambda op: {NAME_RAD: 0, NAME_AI: 1, rad_plus_ai_example_op: 2}.get(op.name, 3)
    )

    COLOR_MAP = {
        NAME_RAD: COLOR_RAD,
        NAME_AI: COLOR_AI_STANDALONE,
        rad_plus_ai_example_op: COLOR_RAD_PLUS_AI_EXAMPLE,
    }

    def _draw_ops(ax: plt.Axes, with_error_bars: bool):
        first_other_op = True
        for op in operating_points:
            # Legend for just some OPs to avoid cluttering the plot
            label = None
            op_string = f"{op.sensitivity.value:.3f}, {op.specificity.value:.3f}"
            if op.name == NAME_RAD:
                label = f"Radiologist: {op_string}"
            elif op.name == NAME_AI:
                label = f"AI stand-alone: {op_string}"
            elif op.name == rad_plus_ai_example_op:
                label = f"Decision referral (example): {op_string}"
            elif first_other_op:
                label = "Decision referral (others)"
                first_other_op = False
            ax.errorbar(
                x=1 - op.specificity.value,
                y=op.sensitivity.value,
                xerr=format_error_bar(
                    ValueWithCI(1 - op.specificity.value, 1 - op.specificity.ci_low, 1 - op.specificity.ci_upp)
                )
                if with_error_bars
                else None,
                yerr=format_error_bar(op.sensitivity) if with_error_bars else None,
                fmt="o",
                color=COLOR_MAP[op.name] if op.name in COLOR_MAP else COLOR_AI,
                label=label,
            )

    # Make one plot with the full axis, and then add a zoomed insert of the same
    fig = plt.figure()

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    title = f"All data\nPositive exams, n={n_pos:,}; Negative exams, n={n_neg:,}"

    # ROC curve
    plot_util.roc_curve_plot(
        y_true,
        y_score_normal_triaging,
        sample_weight=weights,
        legend_prefix="AI ",
        pos_label=True,  # True <=> corresponds to cancer
        color=COLOR_AI,
        title=title,
        n_bootstrap=n_bootstrap,
        fig=fig,
    )
    # Operating points with error bars
    ax = plt.gca()
    _draw_ops(ax, with_error_bars=n_bootstrap > 0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend()
    # Now add an insert to zoom into the operating points
    axins = ax.inset_axes([0.5, 0.45, 0.47, 0.47])
    _draw_ops(axins, with_error_bars=n_bootstrap > 0)
    ax.indicate_inset_zoom(axins)

    plot_util.save_mpl_figure(
        os.path.join(output_dir, "system_performance.png"), fig, close_fig=False, format="png", dpi=300
    )
    plot_util.save_mpl_figure(os.path.join(output_dir, "system_performance.svg"), fig, format="svg")


def roc_plot_confident(
    decision_referral_result: DecisionReferralResult,
    y_score_combined: np.ndarray,
    y_true: np.ndarray,
    y_rad: np.ndarray,
    weights: np.ndarray,
    min_sensitivity: Optional[float],
    min_specificity: Optional[float],
    output_dir: str,
    n_bootstrap: int = 1000,
):
    """
    Generates a ROC plot in the confident regime for the given operating pair.

    Args:
        decision_referral_result: DecisionReferralResult object.
        y_score_combined: from normal triaging and safety net model in respective regimes
        y_true: ground truth labels
        y_rad: radiologist assessments
        weights: inverse probability weights
        min_sensitivity: normal triaging sensitivity for the operating pair
        min_specificity: safety net specificity for the operating pair
        output_dir: where to store the plots
        n_bootstrap: number of resamples for CIs
    """
    name_confident = f"Subset assessed by AI ({min_sensitivity:.1%}, {min_specificity:.1%})"
    result_tuple = decision_referral_result.get_result_tuple(min_sensitivity, min_specificity)

    try:
        roc_plot(
            y_score_combined,
            y_true,
            y_rad,
            weights,
            output_dir,
            name_confident,
            result_tuple.selection_confident,
            n_bootstrap=n_bootstrap,
        )
    except plot_util.NotEnoughClassesError:
        logging.warning(
            "Don't have samples from both classes in order to compute ROC curves. "
            "This is expected for local test data."
        )


@plot_util.seaborn_context(plot_context, plot_font_scale)
def roc_plot(
    y_score: np.ndarray,
    y_true: np.ndarray,
    y_rad: np.ndarray,
    weights: np.ndarray,
    output_dir: str,
    name: str,
    selection: Optional[np.ndarray] = None,
    show_ai_op: bool = False,
    n_bootstrap: int = 1000,
):
    """Generate a ROC plot for the given model scores and selection

    Args:
        y_score: model scores for the whole dataset, will be sub-sampled via `selection`
        y_true: ground truth labels
        y_rad: radiologist assessments
        weights: inverse probability weights
        output_dir: folder to dump the figure to
        name: used in figure title and filename
        selection: Optional mask for which samples to plot
        show_ai_op: Whether to show the operating point of the algorithm. Defaults to False.
        n_bootstrap: number of resamples for CIs
    """
    if selection is None:
        selection = np.array([True] * len(y_score))

    # Get the operating point of radiologist(s)
    operating_points = [
        statistics_util.compute_operating_point_from_pooled_predictions(
            y_true=y_true[selection],
            y_pred=y_rad[selection],
            sample_weights=weights[selection],
            n_resamples=n_bootstrap,
            name="Radiologist",
        )
    ]
    operating_point_colors = [COLOR_RAD]

    if show_ai_op:
        operating_points.append(
            statistics_util.determine_best_op_from_roc(
                y_true=y_true[selection],
                y_score=y_score[selection],
                pos_label=True,
                sample_weight=weights[selection],
                n_bootstrap=n_bootstrap,
                name=NAME_AI,
            )
        )
        operating_point_colors.append(COLOR_AI)

    n_pos_selection = sum(y_true[selection] == True)  # noqa
    n_neg_selection = sum(y_true[selection] == False)  # noqa
    title = f"{name}\nPositive exams, n={n_pos_selection:,}; Negative exams, n={n_neg_selection:,}"

    # ROC curve
    fig, _ = plot_util.roc_curve_plot(
        y_true[selection],
        y_score[selection],
        sample_weight=weights[selection],
        legend_prefix="AI ",
        pos_label=True,  # True <=> corresponds to cancer
        color=COLOR_AI,
        title=title,
        xlabel="1 - specificity",
        ylabel="Sensitivity",
        n_bootstrap=n_bootstrap,
        operating_points=operating_points,
        operating_point_colors=operating_point_colors,
    )
    filename_suffix = f"{name.lower()}".replace(" ", "_")
    plot_util.save_mpl_figure(
        os.path.join(output_dir, f"roc_curve_{filename_suffix}.png"),
        fig,
        close_fig=False,
        format="png",
        dpi=300,
    )
    plot_util.save_mpl_figure(
        os.path.join(output_dir, f"roc_curve_{filename_suffix}.svg"),
        fig,
        format="svg",
    )


@plot_util.seaborn_context(plot_context, plot_font_scale)
def subgroup_performance_plot(
    subgroup_sensitivities: subgroups.SubgroupSensitivities,
    min_sensitivity: Optional[float],
    min_specificity: Optional[float],
    output_dir: str,
    stratifications: List[str] = subgroups.SubgroupMasks.STRATIFICATIONS,
):
    """Plots the performance on subgroups in terms of their impact on screening metrics"""
    op_name = get_operating_pair_name(min_sensitivity, min_specificity)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
    fig.suptitle(f"{op_name}")

    for stratification, ax in zip(stratifications, axs.flatten()):
        sens_rad_stratified = subgroup_sensitivities.get_strata(method=NAME_RAD, stratification=stratification)
        sens_ai_stratified = subgroup_sensitivities.get_strata(method=NAME_AI, stratification=stratification)
        sens_rad_plus_ai_stratified = subgroup_sensitivities.get_strata(method=NAME_DR, stratification=stratification)

        assert [s.stratum for s in sens_rad_stratified] == [
            s.stratum for s in sens_rad_plus_ai_stratified
        ], f"Misalignment between strata: {sens_rad_stratified} vs. {sens_rad_plus_ai_stratified}."
        x_labels = [f"{s.stratum}\n(n={s.n_studies})" for s in sens_rad_plus_ai_stratified]

        x = np.arange(len(x_labels))
        X_MIN = x.min() - 0.5
        X_MAX = x.max() + 0.5

        # bar layout
        WIDTH = 0.25
        x_rad = x - WIDTH
        x_ai = x
        x_dr = x + WIDTH

        ax.bar(
            x=x_rad,
            height=[s.value_with_ci.value for s in sens_rad_stratified],
            yerr=format_error_bar([s.value_with_ci for s in sens_rad_stratified]),
            width=WIDTH,
            color=COLOR_RAD,
            label="Radiologist",
        )
        ax.bar(
            x=x_ai,
            height=[s.value_with_ci.value for s in sens_ai_stratified],
            yerr=format_error_bar([s.value_with_ci for s in sens_ai_stratified]),
            width=WIDTH,
            color=COLOR_AI_STANDALONE,
            label="AI stand-alone",
        )
        ax.bar(
            x=x_dr,
            height=[s.value_with_ci.value for s in sens_rad_plus_ai_stratified],
            yerr=format_error_bar([s.value_with_ci for s in sens_rad_plus_ai_stratified]),
            width=WIDTH,
            # Idea for improvement: we could visualize the fraction of studies assessed by rad & AI respectively
            # via colors. Not doing that for now as it makes the plot more complex.
            color=COLOR_RAD_PLUS_AI_EXAMPLE,
            label="Decision referral",
        )

        # Indicate statistical significance (p values are available throughout or not at all)
        if sens_ai_stratified[0].p_value is not None:
            for x_i, s_i in zip(x_ai, sens_ai_stratified):
                plot_util.add_statistical_annotation(ax, s_i.p_value, x=x_i, y=s_i.value_with_ci.ci_upp)
            for x_i, s_i in zip(x_dr, sens_rad_plus_ai_stratified):
                plot_util.add_statistical_annotation(ax, s_i.p_value, x=x_i, y=s_i.value_with_ci.ci_upp)

        ax.hlines(
            y=subgroup_sensitivities.get_strata(method=NAME_RAD, stratification=None).value_with_ci.value,
            xmin=X_MIN,
            xmax=X_MAX,
            color=COLOR_RAD,
            linestyles="solid",
            label="Rad. (all studies)",
        )
        ax.hlines(
            y=subgroup_sensitivities.get_strata(method=NAME_AI, stratification=None).value_with_ci.value,
            xmin=X_MIN,
            xmax=X_MAX,
            color=COLOR_AI_STANDALONE,
            linestyles="dotted",
            label="AI (all studies)",
        )
        ax.hlines(
            y=subgroup_sensitivities.get_strata(method=NAME_DR, stratification=None).value_with_ci.value,
            xmin=X_MIN,
            xmax=X_MAX,
            color=COLOR_AI,
            linestyles="dashed",
            label="DR (all studies)",
        )

        ax.set_ylabel("Sensitivity")
        ax.set_title(f"Sensitivity by {stratification}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90 if stratification in ["Finding", "Core Needle Biopsy score"] else 0)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc="lower center")

    plt.tight_layout()

    plot_util.save_mpl_figure(
        os.path.join(output_dir, f"subgroup_sensitivities_{op_name.lower()}.png"),
        fig,
        close_fig=False,
        format="png",
        dpi=300,
    )
    plot_util.save_mpl_figure(
        os.path.join(output_dir, f"subgroup_sensitivities_{op_name.lower()}.svg"),
        fig,
        format="svg",
    )
