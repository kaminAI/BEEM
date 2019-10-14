import numpy as np
from numpy.random import multinomial


def make_equal_bin_sizes(X, n_bins: int) -> list:
    """
    Helper function to get equal sized bins for initialization.

    Parameters
    ----------
    X : array-like
        Contains the observations
    n_bins : int
        Number of bins to consider

    Returns
    -------
    list
        A list containg n_bins sets of observations
    """
    k, m = divmod(len(X), n_bins)
    return list(X[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_bins))


def greedy_sampler(prob_mtrx):
    """
    Greedy sampler, takes a "value matrix" and reassigns each sample to the max value generative model

    Parameters
    ----------
    prob_mtrx : array-like
        Probablity that _each_ observations belongs to a particular model.

    Returns
    -------
    list
        List of new indices
    """
    amax = np.argmax(prob_mtrx, axis=1)  # Get index of max val per row
    indx_bins = [np.where(amax == i)[0] for i in range(prob_mtrx.shape[-1])]
    return indx_bins


def boltzmann_sampling(prob_mtrx):
    """
    Not really boltzmann sampling, just sampling from a multinomial distribution.
    If the probabilities are generated from "boltzmann" this is boltzmann sampling.

    Parameters
    ----------
    prob_mtrx : array-like
        Probablity that _each_ observations belongs to a particular model.

    Returns
    -------
    list
        List of new indices

    Misc
    ----
    TODO: replace this for the scipy version
    """
    samples = np.asarray([multinomial(1, p) for p in prob_mtrx])
    indx_bins = [np.where(samples[:, i])[0] for i in range(samples.shape[-1])]
    return indx_bins


def boltzmann(logP_mtrx, tau):
    """
    Generates boltzmann distributions from value matrix (high value = hight probability).

    Parameters
    ----------
    logP_mtrx : array-like
        log-probability matrix
    tau : float
        Temperature parameter

    Returns
    -------
    array-like
        Probablity that _each_ observations belongs to a particular model.
    """
    score = np.sum(np.max(logP_mtrx, axis=1))
    # + np.min(logP_mtrx,axis=1).reshape(-1,1) #To avoid over/underflow
    z = logP_mtrx - np.max(logP_mtrx, axis=1).reshape(-1, 1)
    # z=logP_mtrx
    P_mtrx = np.exp(z / tau)
    prob_mtrx = P_mtrx / np.sum(P_mtrx, axis=1)[:, None]
    assert all(np.abs(np.sum(prob_mtrx, axis=1) - 1.0) < 0.01), "probabilites dont sum to 1"
    return prob_mtrx, score


def boltzmann_global_opt(logP_mtrx, tau):
    """
    Generates boltzmann distributions from value matrix (high value = hight probability).

    Parameters
    ----------
    logP_mtrx : array-like
        log-probability matrix
    tau : float
        Temperature parameter

    Returns
    -------
    array-like
        Probablity that _each_ observations belongs to a particular model.
    """
    score = np.sum(np.max(logP_mtrx, axis=1))

    def func(tau):
        tmp = (logP_mtrx - np.max(logP_mtrx, axis=1).reshape(-1, 1)) / tau
        return np.sum(np.max(tmp, axis=1))

    P_mtrx = np.exp(func(tau))

    prob_mtrx = P_mtrx / np.sum(P_mtrx, axis=1)[:, None]
    assert all(np.abs(np.sum(prob_mtrx, axis=1) - 1.0) < 0.01), "probabilites dont sum to 1"
    return prob_mtrx, tau, score


def score_mixture_likelihood(logP_mtrx):
    """
    Calculates the maximum likelihood assignment score.

    Example: Consider 1 sample and 3 generative models.
    the maximum likelihood assignment score is the likeliehood of the
    model with highest likeliehood to have generated the sample

    :param logP_mtrx [n_columns, n_model]:
    :return: maximum likelihood assignment score
    """
    score = np.mean(np.max(logP_mtrx, axis=1))
    return score


def assign(logP_mtrx):
    """
    Simple helper to get indices of best subcomponent fit / predict
    :param logP_mtrx:
    :return: indices of maximum likeliehood generative model
    """

    y = np.argmax(logP_mtrx, axis=1)
    return y
