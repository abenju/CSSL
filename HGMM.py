import numpy as np
from scipy.stats import multivariate_normal
from fit_latent_model import *


def normal_multivar_pdf(x, mu, cov):
    vals, vecs = np.linalg.eigh(cov)
    logdet = np.sum(np.log(vals))
    valsinv = 1. / vals
    U = vecs * np.sqrt(valsinv)
    dim = len(vals)
    dev = x - mu
    maha = np.square(np.dot(dev, U)).sum()
    log2pi = np.log(2 * np.pi)
    res = -0.5 * (dim * log2pi + maha + logdet)
    return np.exp(res)


class ConvergenceMonitor:
    def __init__(self, tol=1e-2, iter=10):
        self.tol = tol
        self.n_iter = iter
        self.iter = 0
        self.history = []

    def reset(self):
        self.iter = 0
        self.history = []

    def report(self, log_prob):
        self.history.append(log_prob)
        self.iter += 1

    @property
    def converged(self):
        """Whether the EM algorithm converged."""
        # XXX we might want to check that ``log_prob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and self.history[-1] - self.history[-2] < self.tol))


class HGMM:
    def __init__(self, y, x, d_threshold=0.000001):
        """
        Latent model for CSSL regression using a Gaussian HMM.
        Args:
            y(int): Vector size of regression output
            x(int): Vector size of feature observation input
        """
        def rand_symmetric(size):
            # r = np.random.standard_normal(size)
            # sym = (r + r.T)
            # eigval, eigvec = np.linalg.eig(sym)
            # eigval[eigval < 0] = 0
            #
            # result = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
            # # for i in range(size[0]):
            # #     sym[i, i] = 2 * abs(sym[i, i])
            # return result

            # i = np.eye(size[0])
            # r = np.random.rand(size[0], size[0])
            # ri = r + i
            # sym = ri.dot(ri.T)
            # return sym

            r = np.random.normal(0, 1, size=(1000, size[0]))
            if r.shape[1] == 1:
                sym = np.zeros(size)
                sym[0, 0] = np.std(r)
            else:
                sym = np.cov(r, rowvar=False)
            return sym

        self.divergence_threshold = d_threshold
        self.current_divergence = 1

        self.y = y
        self.x = x

        self.pi_mu = np.random.rand(y, 1)             # Static means mu0
        self.pi_p = rand_symmetric((y, y))            # Static covariances P0
        self.a = np.random.rand(y, y)          # Transitions means matrix At
        self.b = np.random.rand(x, y)           # Emission means matrix Bt

        self.q = rand_symmetric((y, y))          # Transitions covariances matrix Qt
        self.r = rand_symmetric((x, x))

        self.c_monitor = ConvergenceMonitor()

    def initialize(self):
        self.pi_mu = np.random.rand(self.y)
        self.pi_p = np.cov(np.random.rand(self.y))
        self.a = np.random.rand(self.y, self.y)
        self.b = np.random.rand(self.x, self.y)

        self.q = np.zeros((self.y, self.y))
        self.r = np.zeros((self.x, self.x))
        self.q = np.cov(np.random.rand(self.y))  # Transitions covariances matrix Qt
        self.r = np.cov(np.random.rand(self.x))

    def supervised_seq(self, seq, labs):
        # self.pi_mu = np.average(labs, axis=0, keepdims=True).T
        # if labs.shape[1] == 1:
        #     self.pi_p[0, 0] = np.std(labs)
        # else:
        #     self.pi_p = np.cov(labs, rowvar=False)

        self.pi_mu, self.pi_p = gauss_fit(labs)
        self.a, self.q = auto_reg1_fit(labs)
        self.b, self.r = linear_reg_fit(seq, labs)

        self.baum_welch(seq)

    def bw_forward(self, seq: np.ndarray):
        """
        Baum-Welch forward phase, what is the probability of being in a state given a sequence of previous observations
        Args:
            seq: sequence of observations

        Returns:

        """

        # TODO: store useful values over time and adjust docstrings

        T = seq.shape[0]                                        # T length of observation sequence
        mu_t = self.pi_mu                                       # Initialize mu 0|0
        p_t = self.pi_p                                         # Initialize P 0|0

        mus = np.zeros((T, self.y, 1))
        ps = np.zeros((T, self.y, self.y))
        hs = np.zeros((T, self.y, self.y))

        likelihoods = np.zeros(T)

        for t, x in enumerate(seq):
            prev_mu = self.a @ mu_t                             # mu t|t-1 = At * mu t-1|t-1
            prev_p = self.q + self.a @ p_t @ self.a.T           # P t|t-1 = Qt - At * P t-1|t-1 * AtT
            h = p_t @ self.a.T @ np.linalg.inv(prev_p)          # Ht = P t-1|t-1 * AtT * P t|t-1 ^-1
            hs[t, :] = h

            v = x - self.b @ prev_mu                            # vt = xt - Bt * mu t|t-1
            sigma = self.r + self.b @ prev_p @ self.b.T         # Sigma t = Rt + Bt * P t|t-1 * BtT
            g = prev_p @ self.b.T @ np.linalg.inv(sigma)        # Gt = P t|t-1 * BtT * Sigma t ^-1

            igb = np.identity(self.y) - g @ self.b
            mu_t = igb @ prev_mu + g @ x                        # mu t|t = (I - Gt * Bt) * mu t|t-1 + Gt * xt
            mus[t, :] = mu_t
            p_t = igb @ prev_p                                  # (I - Gt * Bt) * Pt|t-1
            ps[t, :] = p_t
            likelihoods[t] = multivariate_normal(mean=None, cov=sigma, allow_singular=True).pdf(v.squeeze())  # need log pdf?
            #likelihoods[t] = normal_multivar_pdf(v.T, np.zeros((sigma.shape[0], 1)), sigma)

        seq_likelihood = np.prod(likelihoods)  # p(Z_T) = prod[t=1, T](N(vt: 0, Sigma t))

        return seq_likelihood, mus, ps, hs

    def bw_backward(self, seq, mus, ps, hs):
        """
        Baum-Welch backwards phase, what is the probability of being in a state given a sequence of future observations
        Returns:

        """
        T = seq.shape[0]

        p_Ts = np.zeros((T, self.y, self.y))
        mu_Ts = np.zeros((T, self.y, 1))
        p_prev_Ts = np.zeros((T, self.y, self.y))

        xi_next = np.zeros((self.y, 1))  # xi T|T+1 = 0
        gamma_next = np.zeros((self.y, self.y))  # Gamma T|T+1 = 0

        for t in range(T-1, -1, -1):
            x = seq[t, :]

            xi = xi_next + self.b.T @ np.linalg.inv(self.r) @ x  # xi t|t = xi t|t+1 + BtT * Rt^-1 * xt
            gamma = gamma_next @ self.b.T @ np.linalg.inv(self.r) @ self.b  # Gamma t|t = Gamma t|t+t * BtT * Rt^-1 * Bt
            gq_inv = np.linalg.inv(gamma + np.linalg.inv(self.q))
            xi_prev = self.a.T @ np.linalg.inv(self.q) @ gq_inv @ xi  # AtT * Qt^-1 * (Gamma t|t + Qt^-1)^-1 * xi t|t
            gamma_prev = self.a.T @ (np.linalg.inv(self.q) - np.linalg.inv(self.q) @ gq_inv @ np.linalg.inv(self.q)) @ self.a  # AtT * [Qt^-1 - Qt^-1 * (Gamma t|t + Qt^-1)^-1 * Qt^-1] * At

            p_T = np.linalg.inv(np.linalg.inv(ps[t, :]) + gamma_next)  # Pt|T = (Pt|t^-1 + Gamma t|t+1)^-1
            p_Ts[t] = p_T
            mu_T = np.linalg.inv(p_T) @ (np.linalg.inv(ps[t, :]) @ mus[t, :] + xi_next)  # mu t|T = Pt|T^-1 * (P t|t^-1 * mut|t + xi t|t+1)
            mu_Ts[t] = mu_T

            p_prev_T = p_T @ hs[t, :].T  # P t,t|T = P t|T * HtT
            p_prev_Ts[t] = p_prev_T

            xi_next = xi_prev
            gamma_next = gamma_prev
        return p_Ts, mu_Ts, p_prev_Ts

# From this point, K is assumed to be 1

    def corr_y_y(self, p_t_T, mu_t_T):
        return p_t_T + mu_t_T @ mu_t_T.T

    def corr_y_ymin(self, p_prev_T, mu_t_T, mu_prev_t):
        return p_prev_T + mu_t_T @ mu_prev_t.T

    def corr_x_x(self, x_t):
        return x_t @ x_t.T

    def corr_x_y(self, x_t, mu_t_T):
        return x_t @ mu_t_T.T

    def em_train(self, t, mu_Ts, p_prev_Ts, p_Ts, xs, epochs=1):
        n = mu_Ts.shape[0]
        c_y_ymin = np.mean([self.corr_y_ymin(p_prev_Ts[t], mu_Ts[t], mu_Ts[t-1]) for t in range(1,n)], axis=0)
        c_ymin_ymin = np.mean([self.corr_y_y(p_Ts[t-1], mu_Ts[t-1]) for t in range(1,n)], axis=0)
        c_x_y = np.mean([self.corr_x_y(xs[t], mu_Ts[t]) for t in range(n)], axis=0)
        c_y_y = np.mean([self.corr_y_y(p_Ts[t], mu_Ts[t]) for t in range(n)], axis=0)
        c_x_x = np.mean([self.corr_x_x(xs[t]) for t in range(n)], axis=0)

        for i in range(epochs):
            self.a = c_y_ymin @ np.linalg.inv(c_ymin_ymin) #+ 1e-5 * np.identity(self.a[t].shape[1])
            self.b = c_x_y @ np.linalg.inv(c_y_y) #+ 1e-5 * np.identity(self.b[t].shape[1])
            self.q = c_y_y - c_y_ymin @ np.linalg.inv(c_ymin_ymin) @ c_y_ymin.T
            self.r = c_x_x - c_x_y @ np.linalg.inv(c_y_y) @ c_x_y.T
            self.pi_mu = mu_Ts[0]
            e = mu_Ts[0] - self.pi_mu
            self.pi_p = p_Ts[0] + e @ e.T

    def baum_welch(self, seq, t_theta=0):
        while True:
            seq_likelihood, mus, ps, hs = self.bw_forward(seq)
            #print(seq_likelihood)
            p_Ts, mu_Ts, p_prev_Ts = self.bw_backward(seq, mus, ps, hs)
            self.c_monitor.report(seq_likelihood)
            if self.c_monitor.converged:
                break
            self.em_train(t_theta, mu_Ts, p_prev_Ts, p_Ts, seq)
        return seq_likelihood, mu_Ts, p_prev_Ts
