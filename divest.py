"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from scipy.stats import entropy
import scipy.spatial.distance as dist
import itertools
from functools import partial
from scipy.optimize import minimize
import scipy.sparse as sps
from scipy.sparse.csgraph import minimum_spanning_tree

class GHPEstimator(BaseEstimator):
    """ A MST-based estimator for Generalized Henze-Penrose divergence.

    Parameters
    ----------
    distributions: list, optional (default None)
        List of callable pdfs. If None are given, then the MST is used to
        estimate the integrals. Otherwise, an MCMC approach is utilized.
    priors: array-like, optional (default=None)
        array of priors corresponding to the distributions. If none are given,
        then the data is used to estimate the parameters.
    """
    def __init__(self, distributions=None, priors=None):
        self.distributions = distributions
        self.priors = priors

    def fit(self, X, y):
        """ Approximate the MST integral estimates for all pairs of class labels

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.labels, counts = np.unique(y, return_counts=True)
        self.num_labels = self.labels.shape[0]

        if self.priors is None:
            self.priors = counts / n_samples

        self.gfr_ints = np.zeros((self.num_labels, self.num_labels))

        if self.distributions is None:
            A = dist.squareform(dist.pdist(X, 'euclidean'))
            MST = minimum_spanning_tree(A).toarray()

            x1, x2 = MST.nonzero()
            self.label_to_index = dict(zip(self.labels, range(self.num_labels)))

            for l1, l2 in itertools.combinations(self.labels, r=2):
                if l1 == l2:
                    continue
                i1 = self.label_to_index[l1]
                i2 = self.label_to_index[l2]
                self.gfr_ints[i1, i2] += sum((y[x1] == l1) & (y[x2] == l2))
                self.gfr_ints[i1, i2] += sum((y[x1] == l2) & (y[x2] == l1))
                self.gfr_ints[i1, i2] /= 2 * n_samples
                self.gfr_ints[i2, i1] = self.gfr_ints[i1, i2]
        else:
            for (p1, f1, l1, i1), (p2, f2, l2, i2) in itertools.combinations(zip(self.priors,
                                                                         self.distributions,
                                                                         self.labels, range(self.num_labels)), 2):
                est1 = (f1(X[y == l2]) /
                        sum([p * f(X[y == l2]) for (p, f) in zip(self.priors, self.distributions)])).sum() / np.sum(y == l2)

                est2 = (f2(X[y == l1]) /
                        sum([p * f(X[y == l1]) for (p, f) in zip(self.priors, self.distributions)])).sum() / np.sum(y == l1)

                self.gfr_ints[i1,i2] = p1 * p2 * (est1 + est2) / 2
                self.gfr_ints[i2,i1] = p1 * p2 * (est1 + est2) / 2
        # Return the estimator
        return self

    def get_HP_divergence(self):
        """ Obtain the estimated HP divergence (only implemented for num_labels=2)

        Parameters
        ----------

        Returns
        -------
        hp_divergence: float
            Estimate of the HP divergence
        """
        raise NotImplementedError(
            "THis is not implemented yet.")

def recursive_bayes_estimator(X, y, distributions=None, priors=None):


    if len(distributions) == 2:
        b_est = BayesErrorEstimator(method='GHP', conditionals=distributions, priors=priors)
        b_est.fit(X, y)
        lb, ub = b_est.get_bayes_bounds()
        return lb, ub

    M = len(distributions)
    lbs = np.zeros(M)
    ubs = np.zeros(M)
    for i in range(len(distributions)):
        #Compute the bound, when removing i:
        priors_without_i = priors[np.arange(len(priors))!=i]
        priors_without_i = priors_without_i / np.linalg.norm(priors_without_i, ord=1)
        lb, ub = recursive_bayes_estimator(X[y != i], y[y != i], distributions=distributions[:i] + distributions[i+1:],
                                           priors=priors_without_i)
        lbs[i] = lb
        ubs[i] = ub

    lb = (M - 1) / ((M - 2) * M) * np.sum((1 - priors) * lbs)

    def ub_function(alpha, priors, ubs, M):
        return 1 / (M - 2*alpha) * np.sum((1 - priors) * ubs) + (1 - alpha) / (M - 2 * alpha)

    ub_opt = partial(ub_function, priors=priors, ubs=ubs, M=M)
    result = minimize(ub_opt, x0=0.5, bounds=[(0,1)])
    alpha_opt = result.x[0]

    ub = 1 / (M - 2*alpha_opt) * np.sum((1 - priors) * ubs) + (1 - alpha_opt) / (M - 2 * alpha_opt)

    return lb, ub


class JSEstimator(BaseEstimator):
    """ An estimator for generalized Jensen-Shannon divergence.

    Parameters
    ----------
    distributions: list, optional (default None)
        List of callable pdfs. If None are given, then the MST is used to
        estimate the integrals. Otherwise, an MCMC approach is utilized.
    priors: array-like, optional (default=None)
        array of priors corresponding to the distributions. If none are given,
        then the data is used to estimate the parameters.
    """
    def __init__(self, distributions=None, priors=None):
        self.distributions = distributions
        self.priors = priors


    def fit(self, X, y):
        """ Approximate the generalized JS divergence.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.labels, counts = np.unique(y, return_counts=True)
        self.num_labels = self.labels.shape[0]

        if self.priors is None:
            self.priors = counts / n_samples

        if self.distributions is None:
            raise NotImplementedError(
                "No method for estimating JS divergence is implemented.")

        else:
            sum1 = -np.log2(sum([p * f(X) for p, f in zip(self.priors, self.distributions)])).sum() / n_samples

            sum2 = 0
            for l, p, f in zip(self.labels, self.priors, self.distributions):
                sum2 += p * -np.log2(f(X[y == l, :])).sum() / np.sum(y == l)

            self.js_est = sum1 - sum2

        # Return the estimator
        return self

    def get_js_estimate(self):
        """ Returns the estimate for generalized Jensen-Shannon divergence.

        Parameters
        ----------
        None

        Returns
        -------
        js_estimate : float
            Estimate of the generalized Jensen-Shannon divergence.
        """
        return self.js_est


class BayesErrorEstimator(BaseEstimator):
    """ An Estimator for the Bayes Error (and bounds) in a classification problem.

    Parameters
    ----------
    method: str, optional
        Method of estimation. Default is 'mst'.
    """
    def __init__(self, method='mst', priors=None, conditionals=None):
        self.method = method
        self.priors = priors
        self.conditionals = conditionals


    def fit(self, X, y):
        """ Estimates the Bayes error estimates and/or bounds.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The feature matrix for the classification problem.
        y : array_like, shape = [n_samples]
            The class labels for the classification problem.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y)
        self.n_samples, self.n_features = X.shape
        self.num_labels = np.unique(y).shape[0]
        self.label_types = np.unique(y)

        if self.method == 'GHP':
            ghp_est = GHPEstimator(priors=self.priors, distributions=self.conditionals)
            ghp_est.fit(X, y)
            ghp_sum = np.sum(np.triu(ghp_est.gfr_ints, 1))
            self.bayes_upper = 2 * ghp_sum

            self.bayes_lower = (self.num_labels - 1) / self.num_labels * (
                1 - max(1 - 2 * self.num_labels / (self.num_labels - 1) * ghp_sum, 0)**(0.5))
            self.bayes_est = None

        elif self.method == 'MCMC':
            if self.conditionals is None:
                raise ValueError(
                    "You must pass valid conditional distributions for direct MCMC estimation.")
            if self.priors is None:
                _, self.priors = np.unique(y, return_counts=True)
                self.priors /= self.num_labels

            marginals = []
            for p, f in zip(self.priors, self.conditionals):
                marginals.append(p * f(X))
            marginal_mtx = np.array(marginals)

            est_labels = np.argmax(marginal_mtx, axis=0)
            self.bayes_est = (est_labels != y).sum() / self.n_samples

            self.bayes_upper = None
            self.bayes_lower = None

        elif self.method == 'JS':
            js_est = JSEstimator(priors=self.priors, distributions=self.conditionals)
            js_est.fit(X, y)

            js_estimate = js_est.get_js_estimate()
            self.bayes_upper = 0.5 * (entropy(self.priors, base=2) - js_estimate)
            self.bayes_lower = 1 / (4 * (self.num_labels - 1)) * (entropy(self.priors, base=2) - js_estimate)**2
            self.bayes_est = None

        elif self.method == 'PW':
            hp_est = GHPEstimator()
            self.bayes_upper = 0
            self.bayes_lower = 0

            if self.priors is None:
                _, counts = np.unique(y, return_counts=True)
                self.priors = counts / self.num_labels

            if self.conditionals is None:
                for (l1, p1), (l2, p2) in itertools.combinations(zip(self.label_types, self.priors), 2):
                    hp_est = GHPEstimator()
                    hp_est = hp_est.fit(X[(y == l1) | (y == l2), :], y[(y == l1) | (y == l2)])
                    ptilde1 = p1 / (p1 + p2)
                    ptilde2 = p2 / (p1 + p2)

                    self.bayes_lower += (p1 + p2) * (0.5 - 0.5 * np.sqrt(max(1 - 4 * (hp_est.gfr_ints[0,1]), 0)))
                    self.bayes_upper += (p1 + p2) * 2 * hp_est.gfr_ints[0,1]
            else:
                for (l1, p1, f1), (l2, p2, f2) in itertools.combinations(zip(self.label_types, self.priors, self.conditionals), 2):
                    isl1l2 = (y == l1) | (y == l2)
                    ptilde1 = p1 / (p1 + p2)
                    ptilde2 = p2 / (p1 + p2)

                    hp_est = GHPEstimator(priors=[ptilde1, ptilde2],
                                          distributions=[f1, f2])
                    hp_est = hp_est.fit(X[isl1l2, :], y[isl1l2])

                    self.bayes_lower += (p1 + p2) * (1 - np.sqrt(1 - 4 * ( hp_est.gfr_ints[0,1])))
                    self.bayes_upper += (p1 + p2) * 2 * hp_est.gfr_ints[0,1]

            self.bayes_lower = self.bayes_lower / self.num_labels
            self.bayes_est = None

        elif self.method == 'PW-R':
            raise NotImplementedError(
                "Recursive form of the pairwise estimate is not yet implemented.")

        else:
            raise NotImplementedError(
                "Non HP-type, JS-type, and direct MCMC estimates are not implemented at this time.")

        return self


    def get_bayes_bounds(self):
        """ Get Bayes error bounds.

        Parameters
        ----------


        Returns
        -------
        bayes_lower_bound: float
            Lower bound for the Bayes error.
        bayes_upper_bound: float
            Upper bound for Bayes error.
        """

        return self.bayes_lower, self.bayes_upper


    def get_bayes_estimate(self):
        """ Get Bayes error estimate.

        Parameters
        ----------


        Returns
        -------
        bayes_estimate: float
            Estimate for Bayes error.
        """

        return self.bayes_est
