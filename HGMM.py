import numpy as np
from scipy.stats import multivariate_normal


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
    def __init__(self, y, x, t, d_threshold=0.000001):
        """
        Latent model for CSSL regression using a Gaussian HMM.
        Args:
            y(int): Vector size of regression output
            x(int): Vector size of feature observation input
        """
        self.divergence_threshold = d_threshold
        self.current_divergence = 1

        self.y = y
        self.x = x
        self.T = t

        self.pi_mu = np.random.rand(y, 1)             # Static means mu0
        self.pi_p = np.cov(np.random.rand(y, y))   # Static covariances P0
        self.a = np.random.rand(t, y, y)           # Transitions means matrix At
        self.b = np.random.rand(t, x, y)           # Emission means matrix Bt

        #self.a = np.zeros((t, y, y))
        #self.b = np.zeros((t, x, y))
        self.q = np.zeros((t, y, y))
        self.r = np.zeros((t, x, x))
        for i in range(t):
            #self.a[i] = np.cov(np.random.rand(x, x))
            #self.b[i] = np.cov(np.random.rand(x, x))
            self.q[i] = np.cov(np.random.rand(y, y))  # Transitions covariances matrix Qt
            self.r[i] = np.cov(np.random.rand(x, x))  # Emission covariances matrix Rt

        self.c_monitor = ConvergenceMonitor()

    def initialize(self):
        self.pi_mu = np.random.rand(self.y)
        self.pi_p = np.cov(np.random.rand(self.y))
        self.a = np.random.rand(self.T, self.y, self.y)
        self.b = np.random.rand(self.T, self.x, self.y)

        self.q = np.zeros((self.T, self.y, self.y))
        self.r = np.zeros((self.T, self.x, self.x))
        for i in range(self.T):
            self.q[i] = np.cov(np.random.rand(self.y))  # Transitions covariances matrix Qt
            self.r[i] = np.cov(np.random.rand(self.x))

    def bw_forward(self, seq: np.ndarray, t_theta):
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
            prev_mu = self.a[t_theta] @ mu_t                             # mu t|t-1 = At * mu t-1|t-1
            prev_p = self.q[t_theta] + self.a[t_theta] @ p_t @ self.a[t_theta].T           # P t|t-1 = Qt - At * P t-1|t-1 * AtT
            h = p_t @ self.a[t_theta].T @ np.linalg.inv(prev_p)          # Ht = P t-1|t-1 * AtT * P t|t-1 ^-1
            hs[t, :] = h

            v = x - self.b[t_theta] @ prev_mu                            # vt = xt - Bt * mu t|t-1
            sigma = self.r[t_theta] + self.b[t_theta] @ prev_p @ self.b[t_theta].T         # Sigma t = Rt + Bt * P t|t-1 * BtT
            g = prev_p @ self.b[t_theta].T @ np.linalg.inv(sigma)        # Gt = P t|t-1 * BtT * Sigma t ^-1

            igb = np.identity(self.y) - g @ self.b[t_theta]
            mu_t = igb @ prev_mu + g @ x                        # mu t|t = (I - Gt * Bt) * mu t|t-1 + Gt * xt
            mus[t, :] = mu_t
            p_t = igb @ prev_p                                  # (I - Gt * Bt) * Pt|t-1
            ps[t, :] = p_t
            norm = multivariate_normal(mean=None, cov=sigma)
            likelihoods[t] = norm.pdf(v.T)

        seq_likelihood = np.prod(likelihoods)  # p(Z_T) = prod[t=1, T](N(vt: 0, Sigma t))

        return seq_likelihood, mus, ps, hs

    def bw_backward(self, seq, mus, ps, hs, t_theta):
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

            xi = xi_next + self.b[t_theta].T @ np.linalg.inv(self.r[t_theta]) @ x  # xi t|t = xi t|t+1 + BtT * Rt^-1 * xt
            gamma = gamma_next @ self.b[t_theta].T @ np.linalg.inv(self.r[t_theta]) @ self.b[t_theta]  # Gamma t|t = Gamma t|t+t * BtT * Rt^-1 * Bt
            gq_inv = np.linalg.inv(gamma + np.linalg.inv(self.q[t_theta]))
            xi_prev = self.a[t_theta].T @ np.linalg.inv(self.q[t_theta]) @ gq_inv @ xi  # AtT * Qt^-1 * (Gamma t|t + Qt^-1)^-1 * xi t|t
            gamma_prev = self.a[t_theta].T @ (np.linalg.inv(self.q[t_theta]) - np.linalg.inv(self.q[t_theta]) @ gq_inv @ np.linalg.inv(self.q[t_theta])) @ self.a[t_theta]  # AtT * [Qt^-1 - Qt^-1 * (Gamma t|t + Qt^-1)^-1 * Qt^-1] * At

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
        c_y_ymin = self.corr_y_ymin(p_prev_Ts[t], mu_Ts[t], mu_Ts[t-1])
        c_ymin_ymin = self.corr_y_y(p_Ts[t-1], mu_Ts[t-1])
        c_x_y = self.corr_x_y(xs[t], mu_Ts[t])
        c_y_y = self.corr_y_y(p_Ts[t], mu_Ts[t])
        c_x_x = self.corr_x_x(xs[t])

        for i in range(epochs):
            self.a[t] = c_y_ymin @ np.linalg.inv(c_ymin_ymin) #+ 1e-5 * np.identity(self.a[t].shape[1])
            self.b[t] = c_x_y @ np.linalg.inv(c_y_y) #+ 1e-5 * np.identity(self.b[t].shape[1])
            self.q[t] = c_y_y - c_y_ymin @ np.linalg.inv(c_ymin_ymin) @ c_y_ymin.T
            self.r[t] = c_x_x - c_x_y @ np.linalg.inv(c_y_y) @ c_x_y.T
            self.pi_mu = mu_Ts[0]
            e = mu_Ts[0] - self.pi_mu
            self.pi_p = p_Ts[0] + e @ e.T

    def baum_welch(self, seq, t_theta=0):
        while True:
            seq_likelihood, mus, ps, hs = self.bw_forward(seq, t_theta)
            #print(seq_likelihood)
            p_Ts, mu_Ts, p_prev_Ts = self.bw_backward(seq, mus, ps, hs, t_theta)
            self.c_monitor.report(seq_likelihood)
            if self.c_monitor.converged:
                break
            self.em_train(t_theta, mu_Ts, p_prev_Ts, p_Ts, seq)
        return seq_likelihood, mu_Ts, p_prev_Ts
