# ================================
# Regime detection with joint (X,Y)
# Sticky HDP-HMM + NIW emissions
# PCA on X and Y (user-configurable)
# ================================

import numpy as np
import scipy.stats as ss
import scipy.special as ssp
from numpy.linalg import svd, cholesky, solve
from numpy.random import dirichlet, beta as rbeta, gamma as rgamma, multinomial, binomial, uniform
import pandas as pd


EPS = 1e-9

# -----------------
# PCA (no sklearn)
# -----------------

def pca_fit(X, n_components):
    """MODIFICATION: PCA on input or output block.
    Fit PCA via SVD on centered data; return mean, components, explained variance."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = svd(Xc, full_matrices=False)
    comps = Vt[:n_components].T
    n = X.shape[0]
    ev = (S**2) / max(n - 1, 1)
    return mu, comps, ev[:n_components]

def pca_transform(X, mu, comps):
    """Project data onto previously-fitted PCA components."""
    X = np.asarray(X, dtype=float)
    return (X - mu) @ comps

# -------------------------------------------
# NIW posterior and predictive (full cov)
# -------------------------------------------

def niw_posterior(m0, kappa0, nu0, S0, n, x_sum, xxT_sum):
    """Compute NIW posterior parameters from prior + sufficient stats."""
    d = m0.shape[0]
    if n == 0:
        return m0.copy(), kappa0, nu0, S0.copy()
    xbar = x_sum / n
    S = xxT_sum - n * np.outer(xbar, xbar)
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    m_n = (kappa0 * m0 + n * xbar) / (kappa0 + n)
    md = (xbar - m0).reshape(-1, 1)
    S_n = S0 + S + (kappa0 * n / (kappa0 + n)) * (md @ md.T)
    return m_n, kappa_n, nu_n, S_n

def _logdet_psd(A):
    jitter = 1e-9
    for _ in range(5):
        try:
            L = cholesky(A + jitter * np.eye(A.shape[0]))
            return 2.0 * np.sum(np.log(np.diag(L))), L
        except np.linalg.LinAlgError:
            jitter *= 10.0
    sign, ld = np.linalg.slogdet(A + jitter * np.eye(A.shape[0]))
    return ld, None

def multivariate_student_t_logpdf(x, m, S, df):
    """Log pdf of multivariate Student-t with location m, scale S, df."""
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.shape[0]
    S = 0.5 * (S + S.T)
    ldS, L = _logdet_psd(S)
    if L is None:
        Si = np.linalg.pinv(S)
        quad = (x - m).T @ Si @ (x - m)
    else:
        v = solve(L, (x - m))
        quad = v.T @ v
    lg = ssp.gammaln(0.5 * (df + d)) - ssp.gammaln(0.5 * df)
    const = lg - 0.5 * (d * np.log(df * np.pi) + ldS)
    return const - 0.5 * (df + d) * np.log1p(quad / max(df, 1e-12))

def niw_posterior_predictive_logpdf(x, m0, kappa0, nu0, S0, n, x_sum, xxT_sum):
    """NIW posterior predictive (multivariate Student-t) log pdf."""
    d = m0.shape[0]
    m_n, kappa_n, nu_n, S_n = niw_posterior(m0, kappa0, nu0, S0, n, x_sum, xxT_sum)
    df = max(nu_n - d + 1, 1.0)
    scale = ((kappa_n + 1.0) / (kappa_n * df)) * S_n
    return multivariate_student_t_logpdf(x, m=m_n, S=scale, df=df)

# --------------------------------------
# Sticky HDP-HMM (HCRP components)
# --------------------------------------

def transform_var_poly(v0, v1, p):
    """Same parameterization as your code for Beta stickiness prior."""
    if p == 'inf':
        rho0 = -v0 * np.log(v1)
    else:
        rho0 = v0 / (np.power(v1, p))
    rho1 = (1 - v0) * rho0 / v0
    return rho0, rho1

def sample_kappa(zt, wt, rho0, rho1, K):
    kappa_vec = np.zeros(K)
    num_1_vec = np.zeros(K)
    num_0_vec = np.zeros(K)
    for j in range(K):
        ind = np.where(zt[:-1] == j)[0] + 1
        num_1 = wt[ind].sum() if len(ind) else 0.0
        num_0 = len(ind) - num_1
        num_1_vec[j] = num_1
        num_0_vec[j] = num_0
        kappa_vec[j] = rbeta(rho0 + num_1, rho1 + num_0)
    kappa_new = rbeta(rho0, rho1)
    return kappa_vec, kappa_new, num_1_vec, num_0_vec

def sample_m(n_mat, beta_vec, alpha0, K):
    m_mat = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            n_jk = int(n_mat[j, k])
            if n_jk > 0:
                probs = alpha0 * beta_vec[k] / (np.arange(n_jk) + alpha0 * beta_vec[k])
                x_vec = binomial(1, probs)
                m_mat[j, k] = x_vec.sum()
    if K > 0:
        m_mat[0, 0] += 1
    return m_mat

def sample_beta(m_mat, gamma0):
    #print(m_mat, gamma0)
    counts = m_mat.sum(axis=0)
    if np.any(counts == 0) == True:
        print("Zero Found in alpha vector for Dirichlet Distribution: Replaced by int(mean)")
        counts[np.where(counts == 0)] = int(counts.mean())
    beta_full = dirichlet(np.hstack([counts, gamma0]))
    return beta_full[:-1], beta_full[-1]

def sample_alpha(m_mat, n_mat, alpha0, a_pri, b_pri):
    r_list = []
    n_rows = n_mat.sum(axis=1)
    for val in n_rows:
        if val > 0:
            r_list.append(rbeta(alpha0 + 1.0, val))
    r_vec = np.array(r_list) if len(r_list) else np.array([rbeta(alpha0 + 1.0, 1.0)])
    s_vec = binomial(1, n_rows / (n_rows + alpha0 + EPS))
    return rgamma(a_pri + m_mat.sum() - 1 - s_vec.sum(), 1.0 / (b_pri - np.log(r_vec + EPS).sum()))

def sample_gamma(K, m_mat, gamma0, a_pri, b_pri):
    eta = rbeta(gamma0 + 1.0, m_mat.sum())
    numer = a_pri + K - 1.0
    denom = b_pri - np.log(eta + EPS)
    pi_m = numer / (numer + m_mat.sum() * denom)
    indicator = binomial(1, pi_m)
    if indicator:
        return rgamma(a_pri + K, 1.0 / denom)
    return rgamma(a_pri + K - 1.0, 1.0 / denom)

def compute_rho_posterior(rho0, rho1, K, num_1_vec, num_0_vec):
    val = K * (ssp.gammaln(rho0 + rho1) - ssp.gammaln(rho0) - ssp.gammaln(rho1))
    val += (ssp.gammaln(rho0 + num_1_vec)).sum() + (ssp.gammaln(rho1 + num_0_vec)).sum()
    val -= (ssp.gammaln(rho0 + rho1 + num_1_vec + num_0_vec)).sum()
    return np.real(val)

def sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, K, num_1_vec, num_0_vec, p):
    v0_grid = np.linspace(v0_range[0], v0_range[1], v0_num_grid)
    v1_grid = np.linspace(v1_range[0], v1_range[1], v1_num_grid)
    posterior_grid = np.zeros((v0_num_grid, v1_num_grid))
    for i, v0 in enumerate(v0_grid):
        for j, v1 in enumerate(v1_grid):
            rho0, rho1 = transform_var_poly(v0, v1, p)
            posterior_grid[i, j] = compute_rho_posterior(rho0, rho1, K, num_1_vec, num_0_vec)
    pg = np.exp(posterior_grid - np.max(posterior_grid))
    pg = pg / (pg.sum() + EPS)
    idx = np.where(multinomial(1, pg.reshape(-1)))[0][0]
    i = idx // v1_num_grid
    j = idx % v1_num_grid
    v0 = v0_grid[int(i)]
    v1 = v1_grid[int(j)]
    return transform_var_poly(v0, v1, p) + (pg,)

def sample_pi_sticky(K, alpha0, beta_vec, beta_new, n_mat, kappa_vec, kappa_new):
    pi_mat = np.zeros((K + 1, K + 1))
    for j in range(K):
        base = alpha0 * beta_vec + n_mat[j]
        prob_vec = np.hstack([base, alpha0 * beta_new])
        prob_vec = np.clip(prob_vec, 1e-3, None)
        pi_mat[j] = dirichlet(prob_vec)
    prob_vec = np.hstack([alpha0 * beta_vec, alpha0 * beta_new])
    prob_vec = np.clip(prob_vec, 1e-3, None)
    pi_mat[-1] = dirichlet(prob_vec)
    kappa_all = np.hstack([kappa_vec, kappa_new])
    return pi_mat * (1.0 - kappa_all).reshape(-1, 1) + np.diag(kappa_all)

# -------------------------------
# Emission sufficient statistics
# -------------------------------

def init_emission_stats(K, d):
    return np.zeros((K, d)), np.zeros((K, d, d))

def add_point_to_state(k, o, n_vec, O_sum, O_xxT):
    n_vec[k] += 1
    O_sum[k] += o
    O_xxT[k] += np.outer(o, o)

def remove_point_from_state(k, o, n_vec, O_sum, O_xxT):
    n_vec[k] -= 1
    O_sum[k] -= o
    O_xxT[k] -= np.outer(o, o)

def emission_logpred_existing(k, o, n_vec, O_sum, O_xxT, m0, kappa0, nu0, S0):
    n = int(n_vec[k])
    x_sum = O_sum[k]
    xxT_sum = O_xxT[k]
    return niw_posterior_predictive_logpdf(o, m0, kappa0, nu0, S0, n, x_sum, xxT_sum)

def emission_logpred_new(o, m0, kappa0, nu0, S0):
    zeros = np.zeros_like(m0)
    return niw_posterior_predictive_logpdf(o, m0, kappa0, nu0, S0, 0, zeros, np.zeros((m0.shape[0], m0.shape[0])))

# -------------------------
# Initialization (vector)
# -------------------------

def init_gibbs_vector(T, o_seq, m0, kappa0, nu0, S0, alpha0, gamma0, rho0, rho1):
    d = o_seq.shape[1]
    K = 1
    zt = np.zeros(T, dtype=int)
    b = dirichlet(np.array([1.0, gamma0]))
    beta_vec, beta_new = b[:-1], b[-1]
    kappa_vec = np.clip(np.array([rbeta(rho0, rho1)]), 0, 0.95)
    kappa_new = rbeta(rho0, rho1)
    wt = binomial(1, kappa_vec[0], size=T)
    wt[0] = 0
    n_mat = np.zeros((1, 1))
    n_vec = np.zeros(1, dtype=int)
    O_sum, O_xxT = init_emission_stats(1, d)
    add_point_to_state(0, o_seq[0], n_vec, O_sum, O_xxT)
    for t in range(1, T):
        j = zt[t - 1]
        loglik_exist = emission_logpred_existing(j, o_seq[t], n_vec, O_sum, O_xxT, m0, kappa0, nu0, S0)
        loglik_new = emission_logpred_new(o_seq[t], m0, kappa0, nu0, S0)
        zt_dist = (alpha0 * beta_vec + n_mat[j]) / max(alpha0 + n_mat[j].sum(), EPS)
        knew = alpha0 * beta_new / max(alpha0 + n_mat[j].sum(), EPS)
        p_stay = np.clip(kappa_vec[j], 0, 1)
        cands = np.array([p_stay * np.exp(loglik_exist), (1 - p_stay) * knew * np.exp(loglik_new)])
        if cands.sum() <= 0:
            cands = np.array([0.5, 0.5])
        probs = cands / cands.sum()
        r = np.where(multinomial(1, probs))[0][0]
        if r == 0:
            zt[t] = j; wt[t] = 1
        else:
            K += 1; zt[t] = K - 1; wt[t] = 0
            n_mat = np.pad(n_mat, ((0, 1), (0, 1)), constant_values=0)
            n_vec = np.pad(n_vec, (0, 1), constant_values=0)
            O_sum = np.pad(O_sum, ((0, 1), (0, 0)), constant_values=0.0)
            O_xxT = np.pad(O_xxT, ((0, 1), (0, 0), (0, 0)), constant_values=0.0)
            bdraw = rbeta(1, gamma0)
            beta_vec = np.hstack([beta_vec, bdraw * beta_new])
            beta_new = (1 - bdraw) * beta_new
            kappa_vec = np.hstack([kappa_vec, rbeta(rho0, rho1)])
        if wt[t] == 0:
            n_mat[zt[t - 1], zt[t]] += 1
        add_point_to_state(zt[t], o_seq[t], n_vec, O_sum, O_xxT)
    return (K, zt, wt, n_mat, n_vec, O_sum, O_xxT, beta_vec, beta_new, kappa_vec, kappa_new)

def decre_K(zt, n_mat, n_vec, O_sum, O_xxT, beta_vec):
    keep = np.unique(zt)
    mapping = {old: new for new, old in enumerate(keep)}
    zt = np.array([mapping[z] for z in zt], dtype=int)
    n_mat = n_mat[keep][:, keep]
    n_vec = n_vec[keep]
    O_sum = O_sum[keep]
    O_xxT = O_xxT[keep]
    beta_vec = beta_vec[keep]
    return zt, n_mat, n_vec, O_sum, O_xxT, beta_vec, len(keep)

# ------------------------
# Resampling z,w (vector)
# ------------------------

def sample_zw_vector(zt, wt, o_seq, n_mat, n_vec, O_sum, O_xxT,
                     beta_vec, beta_new, kappa_vec, kappa_new,
                     alpha0, gamma0, m0, kappa0, nu0, S0, rho0, rho1, K):
    T = len(zt)
    for t in range(1, T):
        k_old = zt[t]
        remove_point_from_state(k_old, o_seq[t], n_vec, O_sum, O_xxT)
        j = zt[t - 1]
        if wt[t] == 0:
            n_mat[j, k_old] = max(n_mat[j, k_old] - 1, 0)

        loglikes = np.zeros(K + 1)
        for k in range(K):
            loglikes[k] = emission_logpred_existing(k, o_seq[t], n_vec, O_sum, O_xxT, m0, kappa0, nu0, S0)
        loglikes[K] = emission_logpred_new(o_seq[t], m0, kappa0, nu0, S0)

        base = alpha0 * beta_vec + n_mat[j]
        zt_dist = base / max(alpha0 + n_mat[j].sum(), EPS)
        knew = alpha0 * beta_new / max(alpha0 + n_mat[j].sum(), EPS)
        p_stay = kappa_vec[j]

        probs = np.zeros(K + 1)
        for k in range(K):
            val = np.exp(loglikes[k])
            if k == j:
                probs[k] = p_stay * val + (1 - p_stay) * zt_dist[k] * val
            else:
                probs[k] = (1 - p_stay) * zt_dist[k] * val
        probs[K] = (1 - p_stay) * knew * np.exp(loglikes[K])

        s = probs.sum()
        probs = np.ones(K + 1) / (K + 1) if (s <= 0 or not np.isfinite(s)) else probs / s
        choice_idx = np.where(multinomial(1, probs))[0][0]

        if choice_idx == K:
            K += 1
            z_new = K - 1
            n_mat = np.pad(n_mat, ((0, 1), (0, 1)), constant_values=0)
            n_vec = np.pad(n_vec, (0, 1), constant_values=0)
            O_sum = np.pad(O_sum, ((0, 1), (0, 0)), constant_values=0.0)
            O_xxT = np.pad(O_xxT, ((0, 1), (0, 0), (0, 0)), constant_values=0.0)
            bdraw = rbeta(1, gamma0)
            beta_vec = np.hstack([beta_vec, bdraw * beta_new])
            beta_new = (1 - bdraw) * beta_new
            kappa_vec = np.hstack([kappa_vec, rbeta(rho0, rho1)])
            zt[t] = z_new
            wt[t] = 0
        else:
            zt[t] = choice_idx
            wt[t] = 1 if (choice_idx == j) else 0

        if wt[t] == 0:
            n_mat[j, zt[t]] += 1
        add_point_to_state(zt[t], o_seq[t], n_vec, O_sum, O_xxT)

    zt, n_mat, n_vec, O_sum, O_xxT, beta_vec, K = decre_K(zt, n_mat, n_vec, O_sum, O_xxT, beta_vec)
    return zt, wt, n_mat, n_vec, O_sum, O_xxT, beta_vec, beta_new, kappa_vec, K

# -------------------------
# Main sampler driver
# -------------------------

def run_ds_hdp_hmm_joint(o_seq, n_iter=200, 
                         m0=None, kappa0=0.2, nu0=None, S0=None,
                         alpha0_a_pri=2.0, alpha0_b_pri=1,
                         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
                         v0_range=(0.78, 0.93), v1_range=(0.20, 0.70), p=3, 
                         v0_num_grid=30, v1_num_grid=30,
                         seed=123):
    #np.random.seed(seed)
    T, d = o_seq.shape
    if m0 is None:
        m0 = o_seq.mean(axis=0)
    if nu0 is None:
        nu0 = d + 10.0
    if S0 is None:
        S0 = np.cov(o_seq.T) * (nu0 - d - 1)*1 if T > d else np.eye(d)

    alpha0 = rgamma(alpha0_a_pri, 1.0 / alpha0_b_pri)
    gamma0 = rgamma(gamma0_a_pri, 1.0 / gamma0_b_pri)
    v0 = uniform(v0_range[0], v0_range[1])
    v1 = uniform(v1_range[0], v1_range[1])
    rho0, rho1 = transform_var_poly(v0, v1, p)

    (K, zt, wt, n_mat, n_vec, O_sum, O_xxT, 
     beta_vec, beta_new, kappa_vec, kappa_new) = init_gibbs_vector(
        T, o_seq, m0, kappa0, nu0, S0, alpha0, gamma0, rho0, rho1
    )

    K_trace = np.zeros(n_iter, dtype=int)
    for it in range(n_iter):
        #print(gamma0)
        zt, wt, n_mat, n_vec, O_sum, O_xxT, beta_vec, beta_new, kappa_vec, K = sample_zw_vector(
            zt, wt, o_seq, n_mat, n_vec, O_sum, O_xxT,
            beta_vec, beta_new, kappa_vec, kappa_new,
            alpha0, gamma0, m0, kappa0, nu0, S0, rho0, rho1, K
        )
        kappa_vec, kappa_new, num_1_vec, num_0_vec = sample_kappa(zt, wt, rho0, rho1, K)
        m_mat = sample_m(n_mat, beta_vec, alpha0, K)
        beta_vec, beta_new = sample_beta(m_mat, gamma0)
        alpha0 = sample_alpha(m_mat, n_mat, alpha0, alpha0_a_pri, alpha0_b_pri)
        gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri)
        rho0, rho1, _ = sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, K, num_1_vec, num_0_vec, p)
        K_trace[it] = K
        #print(gamma0, m_mat)

    P = sample_pi_sticky(K, alpha0, beta_vec, beta_new, n_mat, kappa_vec, kappa_new)

    posterior_params = []
    for k in range(K):
        n = int(n_vec[k]); x_sum = O_sum[k]; xxT_sum = O_xxT[k]
        m_n, kappa_n, nu_n, S_n = niw_posterior(m0, kappa0, nu0, S0, n, x_sum, xxT_sum)
        posterior_params.append(dict(m=m_n, kappa=kappa_n, nu=nu_n, S=S_n, n=n))
    return dict(K=K, zt=zt, wt=wt, P=P, 
                n_vec=n_vec, O_sum=O_sum, O_xxT=O_xxT,
                beta_vec=beta_vec, beta_new=beta_new,
                kappa_vec=kappa_vec, kappa_new=kappa_new,
                alpha0=alpha0, gamma0=gamma0, rho0=rho0, rho1=rho1,
                m0=m0, kappa0=kappa0, nu0=nu0, S0=S0,
                posterior_params=posterior_params,
                K_trace=K_trace)

# --------------------------------------------
# Input-only regime weights (static/dynamic)
# --------------------------------------------

def regime_weights_from_input(x_new_raw, Ux, Xmean, posterior_params, px, py, method='static', w_prev=None, P=None):
    """MODIFICATION: mathematically consistent P(Z=k|X) via marginal t over X from joint NIW posterior."""
    K = len(posterior_params)
    x_tilde = pca_transform(x_new_raw.reshape(1, -1), Xmean, Ux).reshape(-1)
    loglik = np.zeros(K); d_x = px
    for k in range(K):
        m = posterior_params[k]['m']
        kappa = posterior_params[k]['kappa']
        nu = posterior_params[k]['nu']
        S = posterior_params[k]['S']
        d = m.shape[0]
        mx = m[:d_x]; Sxx = S[:d_x, :d_x]
        df = max(nu - d + 1, 1.0)
        scale = ((kappa + 1.0) / (kappa * df)) * Sxx
        loglik[k] = multivariate_student_t_logpdf(x_tilde, mx, scale, df)

    n_vec = np.array([pp['n'] for pp in posterior_params], dtype=float)
    prior = n_vec / max(n_vec.sum(), EPS)
    if method == 'dynamic':
        if (w_prev is None) or (P is None):
            raise ValueError("dynamic method requires w_prev and P")
        Kmat = P[:K, :K]
        prior = w_prev @ Kmat

    logpost = np.log(prior + EPS) + loglik
    a = logpost.max()
    w = np.exp(logpost - a)
    return w / (w.sum() + EPS)

# -----------------------------
# Data: local or synthetic
# -----------------------------

def load_or_synthesize_data():
    """Use your file paths; fallback to synthetic if missing."""
    try:
        price = pd.read_excel('PowerAll_updated.xlsx', engine='openpyxl')
        fore_gen = pd.read_excel('Forecasted_generation_Day-Ahead_2015_2024.xlsx', engine='openpyxl')
        fore_res = pd.read_excel('Forecasted_residual_load.xlsx', engine='openpyxl')
        price_vals = price.iloc[:, 1:25].to_numpy(dtype=float)
        res_vals = fore_res.iloc[:, 1:25].to_numpy(dtype=float)
        gen_vals = fore_gen.iloc[:, 1:25].to_numpy(dtype=float)
        T = min(len(price_vals), len(res_vals), len(gen_vals))
        Y_daily = price_vals[:T]
        X_daily = np.hstack([res_vals[:T], gen_vals[:T]])  # Dx=48
        return X_daily, Y_daily
    except Exception:
        # Synthetic fallback
        rng = np.random.default_rng(0)
        T = 1456; Dx = 248; Dy = 24; K_true = 3
        A = np.array([[0.90, 0.08, 0.02],
                      [0.06, 0.90, 0.04],
                      [0.05, 0.05, 0.90]])
        z = np.zeros(T, dtype=int)
        for t in range(1, T): z[t] = rng.choice(K_true, p=A[z[t-1]])
        X_daily = np.zeros((T, Dx)); Y_daily = np.zeros((T, Dy))
        for k in range(K_true):
            mx = rng.normal(0, 1, size=Dx)
            my = rng.normal(0, 1, size=Dy) + (k - 1) * 0.5
            Cx = 0.3 * np.eye(Dx) + 0.7 * rng.standard_normal((Dx, Dx)); Cx = Cx @ Cx.T / Dx
            Cy = 0.3 * np.eye(Dy) + 0.7 * rng.standard_normal((Dy, Dy)); Cy = Cy @ Cy.T / Dy
            idx = np.where(z == k)[0]
            X_daily[idx] = rng.multivariate_normal(mx, Cx + 1e-6*np.eye(Dx), size=len(idx))
            Y_daily[idx] = rng.multivariate_normal(my, Cy + 1e-6*np.eye(Dy), size=len(idx))
        return X_daily, Y_daily

# ------------------------------------------
# Build joint and run sampler (small demo)
# ------------------------------------------

def build_joint_observations(X_daily, Y_daily, px=6, py=4):
    mu_x, Ux, _ = pca_fit(X_daily, n_components=min(px, X_daily.shape[1]))
    mu_y, Uy, _ = pca_fit(Y_daily, n_components=min(py, Y_daily.shape[1]))
    X_tilde = pca_transform(X_daily, mu_x, Ux)
    Y_tilde = pca_transform(Y_daily, mu_y, Uy)
    return np.hstack([X_tilde, Y_tilde]), (Ux, mu_x), (Uy, mu_y)

def demo_train_and_weights(px=6, py=4, n_iter=10, seed=0):
    X_daily, Y_daily = load_or_synthesize_data()
    print(X_daily.shape, Y_daily.shape)
    O_seq, (Ux, Xmean), (Uy, Ymean) = build_joint_observations(X_daily, Y_daily, px=px, py=py)
    d = O_seq.shape[1]
    m0 = O_seq.mean(axis=0)
    S_emp = np.cov(O_seq.T) + 1e-3*np.eye(d)
    nu0 = d + 5.0; kappa0 = 0.1; S0 = S_emp * (nu0 - d - 1)
    state = run_ds_hdp_hmm_joint(O_seq, n_iter=n_iter, m0=m0, kappa0=kappa0, nu0=nu0, S0=S0, seed=seed,
                                 v0_num_grid=10, v1_num_grid=10)  # smaller grids => faster
    x_new = X_daily[-1]
    w_static = regime_weights_from_input(x_new, Ux, Xmean, state['posterior_params'], px, py, method='static')
    K = state['K']; w_prev = np.ones(K)/K
    w_dynamic = regime_weights_from_input(x_new, Ux, Xmean, state['posterior_params'], px, py, method='dynamic',
                                          w_prev=w_prev, P=state['P'])
    return dict(K=state['K'], zt=state['zt'], K_trace=state['K_trace'],
                w_static=w_static, w_dynamic=w_dynamic)
"""
if __name__ == "__main__":
    out = demo_train_and_weights(px=5, py=3, n_iter=800, seed=42)
    print("Discovered K:", out['K'])
    print("First 10 zt:", out['zt'][:10])
    print("K trace:", out['K_trace'])
    print("Static weights for a new input:", out['w_static'])
    print("Dynamic weights (one-step) for a new input:", out['w_dynamic'])
"""
