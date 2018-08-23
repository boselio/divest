"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


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
            MST = sps.csgraph.minimum_spanning_tree(A).toarray()

            x1, x2 = MST.nonzero()
            for l1, l2 in itertools.product(labels, repeat=2):
                self.gfr_ints[l1, l2] += sum((y[x1] == l1) & (y[x2] == l2))
                self.gfr_ints[l1, l2] += sum((y[x1] == l2) & (y[x2] == l1))
                self.gfr_ints[l1, l2] /= 2 * n_samples
                self.gfr_ints[l2, l1] = gfr_ints[l1, l2]
        else:
            for (p1, f1, l1), (p2, f2, l2) in itertools.combinations(zip(self.priors,
                                                                         self.distributions,
                                                                         self.labels)):
                est1 = (f1(X[labels == l2]) /
                        sum([p * f(X[labels == l2]) for (p, f) in zip(priors, fs)])).sum() / np.sum(labels == l2)

                est2 = (f2(X[labels == l1]) /
                        sum([p * f(X[labels == l1]) for (p, f) in zip(priors, fs)])).sum() / np.sum(labels == l1)

                self.gfr_ints[l1,l2] = p1 * p2 * (est1 + est2) / 2
                self.gfr_ints[l2,l1] = p1 * p2 * (est1 + est2) / 2
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


class HPEstimator(BaseEstimator):
    """ A MST-based estimator for Henze-Penrose divergence.

    Parameters
    ----------
    """
    def __init__(self):
       return

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
        n = X.shape[0]

        label_set = set(np.unique(y))
        self.ordered_labels = sorted(label_set)
        assert len(ordered_labels) == 2

        A = dist.squareform(dist.pdist(X, 'euclidean'))
        MST = sps.csgraph.minimum_spanning_tree(A).toarray()
        x1, x2 = MST.nonzero()

        self.fr_stat = sum(y[x1] != y[x2])


        # Return the estimator
        return self

    def get_bayes_bounds(self):
        """ Estimates the Bayes error from the estimated GHP integrals.

        Parameters
        ----------
        None

        Returns
        -------
        bayes_estimate : float
            Estimate Bayes error estimate for the multi-class classification problem.
        """
        ghp_sum = np.sum(np.triu(self.gfr_ints, 1))
        return (self.bayes_lower, self.bayes_upper)


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

        X, y = check_(X, y)
        self.n_samples, self.n_features = X.shape
        self.num_labels = np.unique(y).shape[0]
        self.label_types = np.sort(np.unique(y))

        if method == 'GHP':
            ghp_est = GHPEstimator()
            ghp_est.fit(X, y)

            ghp_sum = np.sum(np.triu(ghp_est.gfr_ints, 1))

            self.bayes_upper = 2 * ghp_sum
            self.bayes_lower = (self.num_labels - 1) / self.num_labels * (
                           1 - (1 - 2 * self.num_labels / (self.num_labels - 1) * ghp_sum)**(0.5))
            self.bayes_est = None

        elif method == 'MCMC':
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

        elif method == 'JS':
            js_est = JSEstimator()
            js_est.fit(X, y)

            self.bayes_upper = 0.5 * (entropy(self.priors, base=2) - js_est.js_estimate)
            self.bayes_lower = 1 / (4 * (self.num_labels - 1)) * (entropy(self.priors, base=2) - js_est.js_estimate)**2
            self.bayes_est = None

        elif method == 'PW':
            hp_est = GHPEstimator()
            self.bayes_upper = 0
            self.bayes_lower = 0

            for (l1, l2) in itertools.product(self.label_types, 2):

                isl1l2 = (y == l1) | (y == l2)
                self.pw_lower, self.pw_upper = self.ghp_bound(X[isl1l2, :],
                                                              y[isl1l2],
                                                              priors=self.priors[isl1l2],
                                                              conditionals=self.conditionals[isl1l2])
                self.bayes_lower += self.priors[isl1l2].sum() * self.pw_lower
                self.bayes_upper += self.priors[isl1l2].sum() * self.pw_upper

            self.bayes_lower = self.bayes_lower / self.num_labels
            self.bayes_est = None

        elif method == 'PW-R':
            raise NotImplementedError(
                "Recursive form of the pairwise estimate is not yet implemented.")

        else:
            raise NotImplementedError(
                "Non HP-type, JS-type, and direct MCMC estimates are not implemented at this time.")

        return self


    def get_bounds(self):
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
