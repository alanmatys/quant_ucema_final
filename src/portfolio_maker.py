import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class PortfolioStrategy:
    def __init__(self, returns):
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
        self.weights = None

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class HRP(PortfolioStrategy): # Hierarchical Risk Parity
    @staticmethod
    def get_cluster_var(cov, c_items):
        """Compute variance for a cluster."""
        cov_ = cov.loc[c_items, c_items]
        w_ = 1.0 / np.diag(cov_)
        w_ /= w_.sum()
        w_ = w_.reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def get_quasi_diag(link):
        """Sort clustered items by distance."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def get_rec_bipart(cov, sort_ix):
        """Recursive bisection for weights."""
        w = pd.Series(1.0, index=sort_ix, dtype="float64")
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = HRP.get_cluster_var(cov, c_items0)
                c_var1 = HRP.get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    @staticmethod
    def correl_dist(corr):
        """Compute a proper distance matrix from correlations."""
        dist = ((1 - corr) / 2.0) ** 0.5
        dist[~np.isfinite(dist)] = 0  
        return dist

    def get_weights(self):
        dist = self.correl_dist(self.corr)
        dist = pd.DataFrame(dist, index=self.corr.index, columns=self.corr.index)
        dist = dist.fillna(0)
        dist = (dist + dist.T) / 2

        condensed_dist = squareform(dist.values)
        link = linkage(condensed_dist, "single")

        sort_ix = self.get_quasi_diag(link)
        sort_ix = self.corr.index[sort_ix].tolist()
        self.weights = self.get_rec_bipart(self.cov, sort_ix)
        return self.weights
    
class ASRP(PortfolioStrategy): # Adaptive Spearman Rank Portfolio
    def __init__(self, returns, target_volatility=0.10): # shouldn´t this be automatic based on the risk-adjusted return?
        super().__init__(returns)
        self.target_volatility = target_volatility  # Target portfolio volatility
        # Replace covariance matrix with Spearman correlation
        self.corr = returns.corr(method="spearman")

    @staticmethod
    def correl_dist(corr):
        """Compute a proper distance matrix from Spearman correlations."""
        dist = ((1 - corr) / 2.0) ** 0.5
        dist[~np.isfinite(dist)] = 0  # Handle invalid values
        return dist

    @staticmethod
    def dynamic_linkage_selection(dist):
        """Select the best linkage method adaptively."""
        linkage_methods = ["single", "complete", "average", "ward"]
        best_method = None
        best_score = float("-inf")

        for method in linkage_methods:
            link = linkage(squareform(dist.values), method)
            score = ASRP.evaluate_dendrogram_quality(link)
            if score > best_score:
                best_score = score
                best_method = method

        return best_method

    @staticmethod
    def evaluate_dendrogram_quality(link):
        """Evaluate the quality of a dendrogram (placeholder function)."""
        return -np.var(link[:, 2])  # Example: minimize variance of cluster distances

    def get_weights(self):
        # Compute distance matrix from Spearman correlation and enforce symmetry
        dist = self.correl_dist(self.corr)
        dist = pd.DataFrame(dist, index=self.corr.index, columns=self.corr.index)
        dist = (dist + dist.T) / 2

        # Select best linkage adaptively
        best_linkage = self.dynamic_linkage_selection(dist)
        link = linkage(squareform(dist.values), method=best_linkage)

        # Perform quasi-diagonalization and compute weights
        sort_ix = self.get_quasi_diag(link)
        sort_ix = self.corr.index[sort_ix].tolist()
        raw_weights = self.get_rec_bipart(self.cov, sort_ix)

        # Adjust weights to achieve target volatility
        portfolio_volatility = np.sqrt(np.dot(raw_weights.T, np.dot(self.cov, raw_weights)))
        leverage = self.target_volatility / portfolio_volatility
        self.weights = raw_weights * leverage
        return self.weights

    @staticmethod
    def get_quasi_diag(link):
        """Sort clustered items by distance."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def get_rec_bipart(cov, sort_ix):
        """Recursive bisection for weights."""
        w = pd.Series(1.0, index=sort_ix, dtype="float64")
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = ASRP.get_cluster_var(cov, c_items0)
                c_var1 = ASRP.get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    @staticmethod
    def get_cluster_var(cov, c_items):
        """Compute variance for a cluster."""
        cov_ = cov.loc[c_items, c_items]
        w_ = 1.0 / np.diag(cov_)
        w_ /= w_.sum()
        w_ = w_.reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

class IVP(PortfolioStrategy): # Inverse Variance Portfolio
    def get_weights(self):
        ivp = 1.0 / np.diag(self.cov)
        ivp /= ivp.sum()
        self.weights = pd.Series(ivp, index=self.cov.index)
        return self.weights

class MVP(PortfolioStrategy): # Minimum Variance Portfolio
    def get_weights(self):
        n = len(self.cov)
        initial_weights = np.ones(n) / n
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(n)]

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov, weights))

        result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        self.weights = pd.Series(result.x, index=self.cov.index)
        return self.weights