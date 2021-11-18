import pandas as pd
import numpy as np

from decision_referral.util import (
    DELTA_SENSITIVITY_COLUMN,
    DELTA_SPECIFICITY_COLUMN,
)


def format_metric_and_ci_columns(df: pd.DataFrame) -> pd.DataFrame:
    """String format all metric columns which carry confidence intervals for visualizing tables

    Args:
        df: as obtained from DecisionReferralResult.to_data_frame()

    Returns:
        df: with '{sensitivity, specificity}_{value, ci_low, ci_upp}' columns removed
            and '{sensitivity, specificity} (95% CI)' columns added
    """
    df["sensitivity (95% CI)"] = df.apply(
        lambda x: f"{x.sensitivity_value:.1%} ({x.sensitivity_ci_low:.1%}, {x.sensitivity_ci_upp:.1%})",
        axis=1,
    )
    df["specificity (95% CI)"] = df.apply(
        lambda x: f"{x.specificity_value:.1%} ({x.specificity_ci_low:.1%}, {x.specificity_ci_upp:.1%})",
        axis=1,
    )
    df.drop(
        axis=1,
        columns=[
            "sensitivity_value",
            "sensitivity_ci_low",
            "sensitivity_ci_upp",
            "specificity_value",
            "specificity_ci_low",
            "specificity_ci_upp",
        ],
        inplace=True,
    )
    return df


def format_delta_and_p_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """String format all delta metric columns and accompanying p-values for visualizing tables

    Args:
        df: as obtained from DecisionReferralResult.to_data_frame()

    Returns:
        df: with '{sensitivity, specificity, ppv}_{delta, p_value}' columns removed
            and 'Δ_{SENSITIVITY, SPECIFICITY} (p-value)' columns added
    """

    # LDH: "p-values should be given to two significant figures,
    #       but no longer than 4 decimal places (e.g. p<0.0001)."

    df[DELTA_SENSITIVITY_COLUMN] = df.delta_sensitivity.map(
        lambda x: f"{x:.1%} " if not np.isnan(x) else ""
    ) + df.delta_sensitivity_p_value.map(lambda x: "(p=" + f"{round(x, 4):.2g}"[:6] + ")" if not np.isnan(x) else "")
    df[DELTA_SPECIFICITY_COLUMN] = df.delta_specificity.map(
        lambda x: f"{x:.1%} " if not np.isnan(x) else ""
    ) + df.delta_specificity_p_value.map(lambda x: "(p=" + f"{round(x, 4):.2g}"[:6] + ")" if not np.isnan(x) else "")
    df.drop(
        axis=1,
        columns=[
            "delta_sensitivity",
            "delta_sensitivity_p_value",
            "delta_specificity",
            "delta_specificity_p_value",
        ],
        inplace=True,
    )
    return df
