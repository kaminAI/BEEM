import copy

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.utils import shuffle
from sklearn.utils.fixes import logsumexp

from .utils import assign, boltzmann, boltzmann_sampling
from .utils import make_equal_bin_sizes as equalbins
from .utils import score_mixture_likelihood


class BeemGMM:
    def __init__(self, n_components, tau=1.4, decay=0.97, patience=50, init="random", use_prior=False):

        self.n_components = n_components
        self.tau = tau
        self.decay = decay
        self.patience = patience
        self.init = init
        self.components = [GMM(n_components=1) for i in range(n_components)]
        self.pi = np.ones(n_components) / n_components
        self.use_prior = use_prior

    def _estimate_log_resp(self, X, tau, use_prior=False):
        logP_mtrx = self.predict_logP_mtrx(X)
        if use_prior:
            log_weights = np.log(self.pi)
        else:
            log_weights = np.log(np.ones(self.n_components) / self.n_components)

        weighted_logP_mtrx = logP_mtrx + log_weights

        weighted_logP_mtrx = weighted_logP_mtrx * (1 / tau)

        log_prob_norm = logsumexp(weighted_logP_mtrx, axis=1)

        with np.errstate(under="ignore"):
            log_resp = weighted_logP_mtrx - log_prob_norm[:, np.newaxis]

        return log_resp

    def predict_proba(self, X, tau=None, use_prior=None):
        if tau is None:
            tau = self.tau
        if use_prior is None:
            use_prior = self.use_prior

        log_resp = self._estimate_log_resp(X, tau, use_prior)

        return np.exp(log_resp)

    def predict_logP_mtrx(self, X):
        logP_mtrx = np.zeros([len(X), self.n_components])
        for i in range(self.n_components):
            logP_mtrx[:, i] = self.components[i]._estimate_weighted_log_prob(X).reshape(-1)

        return logP_mtrx

    def predict(self, X, tau=None, use_prior=None):

        if tau is None:
            tau = self.tau
        if use_prior is None:
            use_prior = self.use_prior

        log_resp = self._estimate_log_resp(X, tau, use_prior)

        return np.argmax(log_resp, axis=1)

    def pi_update(self, X, tau):
        r = np.exp(self._estimate_log_resp(X, tau, use_prior=True))
        self.pi = np.sum(r, axis=0) / len(r)

    def fit(self, X):
        n_worse = 0

        # Make initial sample <-> model assignment
        sample_indices = np.arange(0, len(X))
        if self.init.lower() == "random":
            bins = equalbins(sample_indices, n_bins=self.n_components)
        elif self.init.lower() == "kmeans":
            km = KMeans(n_clusters=self.n_components)
            km.fit(X)
            y_km = km.predict(X)
            bins = [sample_indices[y_km == i] for i in range(self.n_components)]

        # init
        logP_mtrx = np.empty([len(X), self.n_components])
        best_score = -np.inf
        iters = 0
        scores = []
        homo_scores = []
        TAU = []
        bincounts = []
        while n_worse < self.patience:
            TAU.append(self.tau)
            iters += 1
            for i, bin in enumerate(bins):
                if len(bin) >= 2:
                    self.components[i].fit(X[bin])  # Estimate 1 mix from samples in bin

            logP_mtrx = self.predict_logP_mtrx(X)  # compute loglikeliehood for all samples

            score = score_mixture_likelihood(logP_mtrx)
            scores.append(score)

            self.pi_update(X, tau=1)  # OBS TAU = 1 for now

            p_mtrx = self.predict_proba(X)
            # print(p_mtrx[0].round(4))
            bins = boltzmann_sampling(p_mtrx)

            spb = np.asarray([len(bin) for bin in bins])
            bincounts.append(spb)

            if score <= best_score:
                n_worse += 1
            else:
                best_score = score
                best_components = copy.copy(self.components)
                n_worse = 0
            self.tau *= self.decay
            self.tau = np.max([self.tau, 0.1])

        self.components = best_components
        return scores, TAU, bincounts
