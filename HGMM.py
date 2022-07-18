import numpy as np
from scipy.stats import multivariate_normal


class HGMM:
    def __init__(self, y, x, t):
        """
        Latent model for CSSL regression using a Gaussian HMM.
        Args:
            y(int): Vector size of regression output
            x(int): Vector size of feature observation input
        """
        self.y = y
        self.x = x
        self.T = t

        self.pi_mu = np.random.rand(y, 1)             # Static means mu0
        self.pi_p = np.cov(np.random.rand(y, y))   # Static covariances P0
        self.a = np.random.rand(t, y, y)           # Transitions means matrix At
        self.q = np.cov(np.random.rand(t, y, y))   # Transitions covariances matrix Qt
        self.b = np.random.rand(t, x, y)           # Emission means matrix Bt
        self.r = np.cov(np.random.rand(t, x, x))   # Emission covariances matrix Rt

    def initilize(self):
        self.pi_mu = np.random.rand(self.y)
        self.pi_p = np.cov(np.random.rand(self.y, self.y))
        self.a = np.random.rand(self.T, self.y, self.y)
        self.q = np.cov(np.random.rand(self.T, self.y, self.y))
        self.b = np.random.rand(self.T, self.x, self.y)
        self.r = np.cov(np.random.rand(self.T, self.x, self.x))

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
            prev_mu = self.a[t] @ mu_t                             # mu t|t-1 = At * mu t-1|t-1
            prev_p = self.q[t] + self.a[t] @ p_t @ self.a[t].T           # P t|t-1 = Qt - At * P t-1|t-1 * AtT
            h = p_t @ self.a[t].T @ np.linalg.inv(prev_p)          # Ht = P t-1|t-1 * AtT * P t|t-1 ^-1
            hs[t, :] = h

            v = x - self.b[t] @ prev_mu                            # vt = xt - Bt * mu t|t-1
            sigma = self.r[t] + self.b[t] @ prev_p @ self.b[t].T         # Sigma t = Rt + Bt * P t|t-1 * BtT
            g = prev_p @ self.b[t].T @ np.linalg.inv(sigma)        # Gt = P t|t-1 * BtT * Sigma t ^-1

            igb = np.identity(self.y) - g @ self.b[t]
            mu_t = igb @ prev_mu + g @ x                        # mu t|t = (I - Gt * Bt) * mu t|t-1 + Gt * xt
            mus[t, :] = mu_t
            p_t = igb @ prev_p                                  # (I - Gt * Bt) * Pt|t-1
            ps[t, :] = p_t
            norm = multivariate_normal(mean=None, cov=sigma)
            likelihoods[t] = norm.pdf(v.T)

        seq_likelihood = np.prod(likelihoods)

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

        xi_next = 0
        gamma_next = 0

        for t in range(T-1, -1, -1):
            x = seq[t, :]

            xi = xi_next + self.b[t].T @ np.linalg.inv(self.r[t]) @ x
            gamma = gamma_next @ self.b[t].T @ np.linalg.inv(self.r[t]) @ self.b[t]  # fix size
            gq_inv = np.linalg.inv(gamma + np.linalg.inv(self.q[t]))
            xi_prev = self.a[t].T @ np.linalg.inv(self.q[t]) @ gq_inv @ xi
            gamma_prev = self.a[t].T @ (np.linalg.inv(self.q[t]) - np.linalg.inv(self.q[t]) @ gq_inv @ np.linalg.inv(self.q[t])) @ self.a[t]

            p_T = np.linalg.inv(np.linalg.inv(ps[t, :]) + gamma_next)
            p_Ts[t] = p_T
            mu_T = np.linalg.inv(p_T) @ (np.linalg.inv(ps[t, :]) @ mus[t, :] + xi_next)
            mu_Ts[t] = mu_T

            p_prev_T = p_T @ hs[t, :].T
            p_prev_Ts[t] = p_prev_T

            xi_next = xi_prev
            gamma_next = gamma_prev
        return p_Ts, mu_Ts, p_prev_Ts

        return None  # TODO: sort out returns, page 9

    def corr_y_y(self):
        pass

    def corr_y_ymin(self):
        pass

    def corr_x_x(self):
        pass

    def corr_x_y(self):
        pass

    def em_train(self, t):
        for i in range(100):
            self.a[t] = self.corr_y_ymin() @ np.linalg.inv(self.corr_y_y())
            self.b[t] = self.corr_x_y() @ np.linalg.inv(self.corr_y_y())
            self.q[t] = self.corr_y_y() - self.corr_y_ymin() @ np.linalg.inv(self.corr_y_y()) @ self.corr_y_ymin().T

