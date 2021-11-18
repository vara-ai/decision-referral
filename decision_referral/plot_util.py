import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import matplotlib as mpl

# for use of matplotlib without GUI:
# https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server#4935945
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

from decision_referral.types import OperatingPoint, ValueWithCI

MatplotlibColor = Union[str, Tuple[float, float, float]]

# To enable text as text export for SVG files
plt.rcParams['svg.fonttype'] = 'none'

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


def seaborn_context(context=None, font_scale=1.0, rc=None):
    """Allows to specify a seaborn context as a decorator

    Usage:

    @seaborn_context(context="paper", font_scale=1.5)
    def plotting_function(...):

    is equivalent to, but saves indentation levels:

    def plotting_function(...):
        with sns.plotting_context(context="paper", font_scale=1.5):
            ...

    Args:
        same as for sns.plotting_context

    """

    def decorator_sns_context(plot_func):
        @functools.wraps(plot_func)
        def wrapper_sns_context(*args, **kwargs):
            with sns.plotting_context(context=context, font_scale=font_scale, rc=rc):
                value = plot_func(*args, **kwargs)
            return value

        return wrapper_sns_context

    return decorator_sns_context


class NotEnoughClassesError(Exception):
    pass


class NotEnoughSamplesError(Exception):
    pass


SCALAR_ROC_KEYS = ["auc", "n_pos", "n_neg", "auc_low", "auc_high"]


@dataclass
class ROCCurveData:
    fpr: np.ndarray  # false positive rate, equals 1 - specificity
    tpr: np.ndarray  # true positive rate, sensitivity
    thresholds: np.ndarray
    auc: float
    n_pos: int
    n_neg: int
    n_bootstrap: Optional[int] = None
    auc_low: Optional[float] = None
    auc_high: Optional[float] = None
    tpr_low: Optional[np.ndarray] = None
    tpr_high: Optional[np.ndarray] = None

    def scalar_dict(self):
        return {key: getattr(self, key) for key in SCALAR_ROC_KEYS}


def roc_curve_compute(y_true, y_score, pos_label=True, n_bootstrap=0, sample_weight=None) -> ROCCurveData:
    """Compute receiver operating characteristic (ROC)

    Args:
        y_true (array of shape (n_samples,)): True binary labels in range {0, 1} or {-1, 1}.  If labels are not
                                              binary, pos_label should be explicitly given.
        y_score (array of shape (n_samples,)): Target scores, can either be probability estimates of the positive
                                               class or confidence values.
        pos_label (int or bool): Label considered as positive and others are considered negative.
        n_bootstrap (int): the number of bootstrap samples for the confidence intervals.
        sample_weight (array of shape (n_samples,)): Optional set of sample weights

    """
    if sample_weight is not None:
        assert y_true.shape == sample_weight.shape
    else:
        sample_weight = np.ones(y_true.shape)
    assert y_score.ndim == 1, "y_score should be of shape (n_samples,)"
    assert len(y_true) == len(y_score), "y_true and y_score must both be n_samples long"

    if len(set(y_true)) == 1:
        raise NotEnoughClassesError(f"Only one class found in y_true, {set(y_true)}. Need two classes to compute ROC")

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    roc_auc = roc_auc_score(y_true, y_score, sample_weight=sample_weight)

    if n_bootstrap > 0:
        # We need to define a lambda here, as the sample weight has to be part of the bootstrapping (see below).
        # Arguments prefixed with underscore to not shadow variables from outer scope.
        roc_auc_score_func = lambda _y_true, _y_score, _sample_weight: roc_auc_score(
            _y_true, _y_score, sample_weight=_sample_weight
        )

        try:
            # If there is only 1 class, for instance if all ROIs are normal, ROC is undefined
            # we cannot catch this with a simple if clause, because due to bootstrapping this
            # can happen non-deterministically
            low, high = bootstrap(
                [y_true, y_score, sample_weight], roc_auc_score_func, n_resamples=n_bootstrap, alpha=0.05
            )

            fpr_low, tpr_low, _ = roc_curve(
                y_true[low.index],
                y_score[low.index],
                pos_label=pos_label,
                sample_weight=sample_weight[low.index] if sample_weight is not None else None,
            )
            fpr_high, tpr_high, _ = roc_curve(
                y_true[high.index],
                y_score[high.index],
                pos_label=pos_label,
                sample_weight=sample_weight[high.index] if sample_weight is not None else None,
            )
            interpolate_low_tpr = interpolate.interp1d(fpr_low, tpr_low, kind="nearest")
            interpolate_high_tpr = interpolate.interp1d(fpr_high, tpr_high, kind="nearest")

            return ROCCurveData(
                fpr=fpr,
                tpr=tpr,
                thresholds=thresholds,
                auc=roc_auc,
                n_pos=n_pos,
                n_neg=n_neg,
                n_bootstrap=n_bootstrap,
                tpr_low=interpolate_low_tpr(fpr),
                tpr_high=interpolate_high_tpr(fpr),
                auc_low=low.value,
                auc_high=high.value,
            )

        except ValueError:
            # if the bootstrapping process failed, we return ROC data without bootstrapping
            pass

    return ROCCurveData(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=roc_auc,
        n_pos=n_pos,
        n_neg=n_neg,
    )


def _draw_single_roc_curve(roc_data: ROCCurveData, legend_prefix: str, color):
    if roc_data.n_bootstrap is not None:
        legend = f"{legend_prefix}AUC: {roc_data.auc:0.3f} (95% CI:{roc_data.auc_low:0.3f}-{roc_data.auc_high:0.3f})"
    else:
        legend = f"{legend_prefix}AUC: {roc_data.auc:0.3f}"

    if roc_data.n_bootstrap is not None:
        plt.fill_between(roc_data.fpr, roc_data.tpr_low, roc_data.tpr_high, color=color, alpha=0.3)

    plt.plot(roc_data.fpr, roc_data.tpr, color=color, label=legend, linewidth=2)


def draw_roc_curves(
    roc_data: List[ROCCurveData],
    legends: List[str],
    colors: Optional[List[MatplotlibColor]] = None,
    title: str = "",
    xlabel: str = "1 - specificity",
    ylabel: str = "sensitivity",
    fig: Optional[mpl.figure.Figure] = None,
):
    """Plot receiver operating characteristic (ROC)

    Args:
        roc_data: List of ROCCurveData structures from `roc_curve_compute`
        legends: list of names for each curve
        colors: list of colors for each curve
        title: figure title
        xlabel: x-axis label
        ylabel: y-axis label
        fig: optionally create figure outside the function in order to get control over its size etc.
    """
    if fig is None:
        fig = plt.figure()
    plt.title(title)

    if colors is None:
        colors = sns.color_palette(n_colors=len(roc_data))

    for curve, legend, color in zip(roc_data, legends, colors):
        _draw_single_roc_curve(curve, legend, color)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right")

    return fig


def draw_operating_points(
    operating_points: List[OperatingPoint], fig: plt.Figure, mode: str = "roc", colors: Optional[List[Any]] = None
):
    """Plot a list of operating points into a ROC or PR curve

    Args:
        operating_points: e.g. 1st/2nd readers and more
        fig: the figure to add the plot to. Note that this function here does not use the `fig` explicitly,
            it is automatically used by matplotlib if fig is the "current figure". Hence, the caller has to make
            sure that this is the case.
        mode: "roc"/"pr" to plot the operating point into a ROC/PR curve respectively
        colors: if provided, one for each operating point. For allowed types see:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    Notes:
        Precision/Recall aren't used that much in the medical world. Equivalent terms
        are:
            precision <=> ppv
            recall <=> sensitivity

    """
    assert plt.gcf() == fig, "Make sure fig is the current figure such that it is used for plotting here."
    SUPPORTED_MODES = {"roc", "pr"}
    assert mode in SUPPORTED_MODES, f"Got {mode}, expected {SUPPORTED_MODES}."

    if colors is None:
        colors = sns.color_palette(n_colors=len(operating_points))
    else:
        assert len(colors) == len(
            operating_points
        ), f"Need exactly one color ({len(colors)}) for each operating point ({len(operating_points)})."

    def get_x_y(op: OperatingPoint, mode: str):
        return {"roc": (1 - op.specificity.value, op.sensitivity.value), "pr": (op.sensitivity.value, op.ppv.value)}[
            mode
        ]

    def get_label(op: OperatingPoint, mode: str):
        return {
            "roc": f"{op.name}: {op.sensitivity.value:.3f}, {op.specificity.value:.3f}",
            "pr": f"{op.name}: {op.ppv.value:.3f}, {op.sensitivity.value:.3f}",
        }[mode]

    for i, op in enumerate(operating_points):
        x, y = get_x_y(op, mode)
        if x is None or y is None:
            # We don't always have the necessary data to compute everything we need
            continue
        else:
            plt.scatter(
                x,
                y,
                s=50,
                color=colors[i],
                marker="X",
                label=get_label(op, mode),
            )


def roc_curve_plot(
    y_true,
    y_score,
    sample_weight=None,
    pos_label=True,
    legend_prefix="",
    n_bootstrap=0,
    color: Optional[MatplotlibColor] = None,
    title="",
    xlabel=None,
    ylabel=None,
    outfile=None,
    operating_points: Optional[List[OperatingPoint]] = None,
    operating_point_colors: Optional[List[Any]] = None,
    fig: Optional[mpl.figure.Figure] = None,
) -> Tuple[mpl.figure.Figure, ROCCurveData]:
    """Compute and plot receiver operating characteristic (ROC)

    Args:
        y_true (array of shape (n_samples,)): True binary labels in range {0, 1} or {-1, 1}.  If labels are not
                                              binary, pos_label should be explicitly given.
        y_score (array of shape (n_samples,)): Target scores, can either be probability estimates of the positive
                                               class or confidence values.
        sample_weight (array of shape (n_samples,)): optional weight for each sample
        pos_label (int or bool): Label considered as positive and others are considered negative.
        legend_prefix (string, by default empty): for the plot legend 'legend_prefix (auc=XX)'
        n_bootstrap (int): the number of bootstrap samples for the confidence intervals.
        color (string): for curves and confidence intervals.
        title (string): figure title
        xlabel: x-axis label
        ylabel: y-axis label
        outfile (string): if provided, save figure using the filename extension to determine the format.
        operating_points: if provided, draw the operating_points into the ROC figure.
        operating_point_colors: optional colors for operating points, one for each. For allowed types, see:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
        fig: optionally create figure outside the function in order to get control over its size etc.

    Returns:
        fig: the figure with the plot
        roc_data: A dict with the ROC data used to plot the curve.

    """

    # compute the ROC curve and get its parameters
    roc_data = roc_curve_compute(y_true, y_score, pos_label, n_bootstrap, sample_weight)

    if fig is None:
        # Create one figure into which both the ROC curves and the operating points are drawn.
        fig = plt.figure()

    if operating_points is not None:
        draw_operating_points(operating_points, fig, colors=operating_point_colors)

    # plot the ROC curve with the parameters and our plotting options
    fig = draw_roc_curves([roc_data], [legend_prefix], [color], title, xlabel, ylabel, fig)

    if outfile is not None:
        save_mpl_figure(outfile, fig)
        logging.info("ROC curve output to {}".format(outfile))

    return fig, roc_data


def generate_bootstrap_samples(
    data: List, fun: Callable[..., float], n_resamples: int
) -> Tuple[List[float], np.ndarray]:
    """Compute confidence interval for values of function fun

    Args:
        data (list): the arguments to fun
        fun (function): the function for which to compute the bootstrapped statistics, must return a float.
        n_resamples (int): the number of bootstrap samples to draw.

    Returns:
        Tuple containing:
            * A list of bootstrap sample results, i.e. the return value of `fun` for each bootstrap sample
            * The indices representing the different bootstrap samples

    """
    assert isinstance(data, list)
    n_samples = len(data[0])
    idx = np.random.randint(0, n_samples, (n_resamples, n_samples))

    def select(data, sample):
        return [d[sample] for d in data]

    def evaluate(sample):
        return fun(*select(data, sample))

    return list(map(evaluate, idx)), idx


class BootstrapBound(NamedTuple):
    value: float
    index: np.ndarray


def bootstrap(
    data: List, fun: Callable, n_resamples: int = 10000, alpha: float = 0.05
) -> Tuple[BootstrapBound, BootstrapBound]:
    values, idx = generate_bootstrap_samples(data, fun, n_resamples)

    # NOTE(christian): mergesort was found to perform best.
    idx = idx[np.argsort(values, axis=0, kind="mergesort")]
    values = np.sort(values, axis=0, kind="mergesort")

    low = BootstrapBound(value=values[int((alpha / 2.0) * n_resamples)], index=idx[int((alpha / 2.0) * n_resamples)])
    high = BootstrapBound(
        value=values[int((1 - alpha / 2.0) * n_resamples)], index=idx[int((1 - alpha / 2.0) * n_resamples)]
    )

    return low, high


def save_mpl_figure(filename, fig, close_fig=True, **kwargs):
    """Save matplotlib figure. By default, the figure is then closed and its memory reclaimed

    Args:
        filename (string): cloud or local absolute path including the extension such as ".png"
        fig (matplotlib.pyplot.Figure)
        close_fig (bool): close the figure and free memory after saving (default True)

        other kwargs are forwarded to `Figure.savefig`

    """
    with open(filename, "wb") as f:
        fig.savefig(f, **kwargs)

    # the following line is needed to ensure that the memory is released.
    # https://stackoverflow.com/questions/2364945/matplotlib-runs-out-of-memory-when-plotting-in-a-loop
    # has an explanation. Without it, we get out of memory errors in loops.
    if close_fig:
        plt.close(fig)


def format_error_bar(value_with_ci: Union[ValueWithCI, List[ValueWithCI]]):
    """Format as required by plt.errorbar"""
    if type(value_with_ci) != list:
        value_with_ci = [value_with_ci]

    lower = [x.value - x.ci_low for x in value_with_ci]
    upper = [x.ci_upp - x.value for x in value_with_ci]
    return np.array([lower, upper])


def get_significance_symbol(p_value: float):
    if p_value <= 0.0001:
        return "****"
    elif 0.0001 < p_value <= 0.001:
        return "***"
    elif 0.001 < p_value <= 0.01:
        return "**"
    elif 0.01 < p_value <= 0.05:
        return "*"
    else:
        return "ns"


def add_statistical_annotation(ax: plt.Axes, p_value: float, x: float, y: float):
    ax.text(x=x, y=y, s=get_significance_symbol(p_value), ha="center")
