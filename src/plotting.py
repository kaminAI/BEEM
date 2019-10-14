import datetime

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker
from scipy import interp
import numpy as np
import scipy.constants as sc
import seaborn as sns
from matplotlib import pyplot as plt

# Turns off the retarded error when one plots multiple color scatters
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_roc_curve(y_true, y_scores, filename=None, save_me=False):
    """
    Plot receiver-operating curve.

    Parameters
    ----------
    y_true : array-like
        List or array containing the TRUE labels
    y_scores : array-like
        List or array containing the probabilities of the predicted labels
    filename : str
        Descriptive filename
    """

    # Main calculations here
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    # Calculate area under the ROC curve here
    auc = np.trapz(tpr, fpr)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lw = 2
    ax.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc)
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Curve")
    ax.legend(loc="lower right")

    if save_me:
        assert filename is not None
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../illustrations/roc_curve-" + filename + "-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )

    plt.show()


def plot_raw_gp_data(X, Y, with_labels=False, gt_allocation=None, save_me=False):

    plt.style.use("default")

    base_size = 5
    plt.rc("text", usetex=True)
    matplotlib.rcParams.update({"font.size": 16})
    matplotlib.rcParams["figure.figsize"] = [base_size, base_size / sc.golden]
    matplotlib_axes_logger.setLevel("ERROR")
    sns.set_style("ticks", {"grid.linestyle": "--"})

    assert type(X) is np.ndarray
    assert type(Y) is np.ndarray

    fig, ax = plt.subplots(figsize=matplotlib.rcParams["figure.figsize"])

    if with_labels:
        current_palette = sns.color_palette("muted")
        assert gt_allocation is not None
        for i, j in enumerate(gt_allocation):
            ax.scatter(X[i], Y[i], c=current_palette[int(j)], alpha=0.75)

        # Bespoke legend
        pop1 = mpatches.Patch(color=current_palette[0], label="Population 1")
        pop2 = mpatches.Patch(color=current_palette[1], label="Population 2")
        ax.legend(handles=[pop1, pop2], framealpha=1)

    else:
        ax.scatter(X, Y, alpha=0.5, c="k", marker="x")

    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.grid(True)

    if save_me:
        now = datetime.datetime.now()
        fig.savefig(
            "../illustrations/gp_data_association-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf", bbox_inches="tight"
        )

    plt.show()


def plot_purity(mets, omgp=False, save_me=False):
    base_size = 6
    plt.rc("text", usetex=True)
    matplotlib.rcParams.update({"font.size": 14})
    matplotlib.rcParams["figure.figsize"] = [base_size, 0.4 * base_size]  # / sc.golden]
    matplotlib_axes_logger.setLevel("ERROR")
    sns.set_style("ticks", {"grid.linestyle": "--"})

    current_palette = sns.color_palette("deep")
    fig, ax = plt.subplots(figsize=matplotlib.rcParams["figure.figsize"])
    ax.grid(True)

    met_names = ["Purity"]

    mean = np.vstack(mets).mean(axis=0)
    std = np.vstack(mets).std(axis=0)
    xx = np.arange(len(mean))

    plt.fill_between(xx, mean - 2 * std, mean + 2 * std, alpha=0.2, color=current_palette[0])
    plt.plot(xx, mean, lw=2, alpha=1, c=current_palette[0], label=met_names[0])
    ax.set_xlim(0, len(mean) - 1)

    if omgp:
        loc = plticker.MultipleLocator(base=20.0)
    else:
        loc = plticker.MultipleLocator(base=2.0)  # this locator puts ticks at regular intervals

    ax.set_ylim(0.4, 1.0)
    ax.xaxis.set_major_locator(loc)
    ax.set_ylabel("Purity $[-]$")

    if omgp:
        ax.set_xlabel("Iterations of the marginalised variational bound")
    else:
        ax.set_xlabel("Number of Boltzmann updates")

    if save_me:
        now = datetime.datetime.now()
        fig.savefig(
            "../illustrations/purity_data_association_results-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )

    plt.show()


def roc_curve_with_error_bounds(y_true, y_scores, save_me=False):
    """
    Plot receiver-operating curve.

    Parameters
    ----------
    y_true : array-like
        List or array containing the TRUE labels
    y_scores : array-like
        List or array containing the probabilities of the predicted labels
    filename : str
        Descriptive filename
    """

    base_size = 6
    plt.rc("text", usetex=True)
    matplotlib.rcParams.update({"font.size": 14})

    assert type(y_scores) is list
    len_first = len(y_scores[0]) if y_scores else None
    assert all(len(i) == len_first for i in y_scores)
    assert len(y_true) == len_first

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    matplotlib.rcParams.update({"font.size": 14})
    lw = 2

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, len(y_true))

    # Plot all ROC curves here
    for y_score in y_scores:
        # Main calculations here
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        # Calculate area under the ROC curve here
        auc = np.trapz(tpr, fpr)
        aucs.append(auc)
        tprs.append(interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    assert len(mean_tpr) == len(mean_fpr)
    mean_tpr[-1] = 1.0
    mean_auc = np.trapz(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Mean ROC Curve
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="darkorange",
        lw=lw,
        label="Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (1 - mean_auc, std_auc),
    )

    # Chance
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, label="Chance", linestyle="--")

    # Fill between
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="darkorange", alpha=0.2, label=r"$\pm$ 2 std. dev.")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    if save_me:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../illustrations/roc_curve_with_bounds-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf", bbox_inches="tight"
        )

    plt.show()


def plot_metrics(mets, gp=False, save_me=False):

    assert type(mets) is list

    base_size = 6
    plt.rc("text", usetex=True)
    matplotlib.rcParams.update({"font.size": 14})
    matplotlib.rcParams["figure.figsize"] = [base_size, 0.4 * base_size]  # / sc.golden]
    matplotlib_axes_logger.setLevel("ERROR")
    sns.set_style("ticks", {"grid.linestyle": "--"})

    current_palette = sns.color_palette("deep")
    fig, ax = plt.subplots(figsize=matplotlib.rcParams["figure.figsize"])
    ax.grid(True)

    met_names = ["AUC", "Specificity", "Precision"]

    for i, j in enumerate(mets):
        mean = np.vstack(j).mean(axis=0)
        std = np.vstack(j).std(axis=0)
        xx = np.arange(len(mean))

        plt.fill_between(xx, mean - 2 * std, mean + 2 * std, alpha=0.2, color=current_palette[i])
        plt.plot(xx, mean, lw=2, alpha=1, c=current_palette[i], label=met_names[i])

    ax.set_xlim(0, len(mean) - 1)

    if gp:
        loc = plticker.MultipleLocator(base=20.0)
    else:
        loc = plticker.MultipleLocator(base=2.0)  # this locator puts ticks at regular intervals

    ax.set_ylim(0.3, 1.0)

    ax.xaxis.set_major_locator(loc)

    if len(mets) == 1:
        ax.set_ylabel("AUC $[-]$")
    else:
        ax.set_ylabel("Magnitude $[-]$")
        ax.legend(loc="lower right", ncol=1, framealpha=1)

    if gp:
        ax.set_xlabel("Iterations of the marginalised variational bound")
    else:
        ax.set_xlabel("Number of Boltzmann updates")

    if save_me:
        now = datetime.datetime.now()
        fig.savefig(
            "../illustrations/gp_data_association_results-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )

    plt.show()

