"""
bl_model.py — Black-Litterman 核心数学实现
严格按照文档公式实现：
  - 隐含均衡收益: Π = λ * Σ * w_mkt
  - 信心矩阵 Ω (Idzorek 方法)
  - 后验均值与协方差
  - 带约束均值-方差优化
"""

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# 1. 隐含均衡收益
# ─────────────────────────────────────────────

def compute_implied_returns(lambda_: float, cov_matrix: np.ndarray, w_mkt: np.ndarray) -> np.ndarray:
    """
    Π = λ * Σ * w_mkt
    Parameters
    ----------
    lambda_ : 风险厌恶系数
    cov_matrix : 年化协方差矩阵 (n x n)
    w_mkt : 市场初始权重向量 (n,)
    Returns
    -------
    Pi : 隐含均衡收益向量 (n,)
    """
    return lambda_ * cov_matrix @ w_mkt


# ─────────────────────────────────────────────
# 2. 观点不确定性矩阵 Ω
# ─────────────────────────────────────────────

def compute_omega(
    P: np.ndarray,
    cov_matrix: np.ndarray,
    tau: float,
    confidences: np.ndarray,
) -> np.ndarray:
    """
    Idzorek 简化方法:
        Ω_ii = (1/c_i - 1) * [τ * P Σ P^T]_ii

    Parameters
    ----------
    P           : 观点矩阵 (k x n)
    cov_matrix  : 年化协方差矩阵 (n x n)
    tau         : 标量 τ
    confidences : 信心水平数组 (k,)，每项 ∈ (0, 1]
    Returns
    -------
    Omega : 对角矩阵 (k x k)
    """
    tau_psp = tau * P @ cov_matrix @ P.T  # (k x k)
    omega_diag = []
    for i, c in enumerate(confidences):
        c = float(np.clip(c, 1e-6, 1 - 1e-6))
        omega_diag.append((1.0 / c - 1.0) * tau_psp[i, i])
    return np.diag(omega_diag)


# ─────────────────────────────────────────────
# 3. BL 后验均值与协方差
# ─────────────────────────────────────────────

def bl_posterior(
    Pi: np.ndarray,
    cov_matrix: np.ndarray,
    tau: float,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    E[R] = [(τΣ)⁻¹ + P^T Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + P^T Ω⁻¹ q]
    Σ_post = Σ + [(τΣ)⁻¹ + P^T Ω⁻¹ P]⁻¹

    当观点为空时（P shape (0, n)），后验收益 = Π，后验协方差 = Σ + τΣ

    Returns
    -------
    mu_post    : 后验均值 (n,)
    sigma_post : 后验协方差 (n x n)
    """
    n = len(Pi)

    if P.shape[0] == 0:
        # 无观点：后验 = 先验
        mu_post = Pi.copy()
        sigma_post = cov_matrix + tau * cov_matrix
        return mu_post, sigma_post

    tau_sigma = tau * cov_matrix
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)

    M = tau_sigma_inv + P.T @ omega_inv @ P          # (n x n)
    M_inv = np.linalg.inv(M)

    mu_post = M_inv @ (tau_sigma_inv @ Pi + P.T @ omega_inv @ q)
    sigma_post = cov_matrix + M_inv

    return mu_post, sigma_post


# ─────────────────────────────────────────────
# 4. 带约束均值-方差优化
# ─────────────────────────────────────────────

def optimize_portfolio(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    lambda_: float,
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    max_vol: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """
    最大化 μ^T w - 0.5 λ w^T Σ w
    约束:
        Σ w_i = 1
        w_lower_i ≤ w_i ≤ w_upper_i
        (可选) sqrt(w^T Σ w) ≤ max_vol

    Returns
    -------
    w      : 最优权重 (n,)
    success: 是否收敛
    msg    : 状态信息
    """
    n = len(mu)

    def neg_utility(w):
        return -(mu @ w - 0.5 * lambda_ * w @ cov_matrix @ w)

    def grad(w):
        return -(mu - lambda_ * cov_matrix @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if max_vol is not None and max_vol > 0:
        constraints.append(
            {"type": "ineq", "fun": lambda w: max_vol**2 - w @ cov_matrix @ w}
        )

    bounds = [(float(w_lower[i]), float(w_upper[i])) for i in range(n)]

    # 可行初始点：等权并裁剪到边界
    w0 = np.clip(np.ones(n) / n, w_lower, w_upper)
    s = w0.sum()
    if s > 0:
        w0 = w0 / s
    else:
        w0 = np.ones(n) / n

    opts = {"ftol": 1e-10, "maxiter": 2000}

    result = minimize(neg_utility, w0, jac=grad, method="SLSQP",
                      bounds=bounds, constraints=constraints, options=opts)

    if not result.success:
        # 再试一次，以均匀初值
        w0b = np.array([(w_lower[i] + w_upper[i]) / 2.0 for i in range(n)])
        s = w0b.sum()
        if s > 0:
            w0b = w0b / s
        result = minimize(neg_utility, w0b, method="SLSQP",
                          bounds=bounds, constraints=constraints, options=opts)

    return result.x, result.success, result.message


# ─────────────────────────────────────────────
# 5. 夏普比率最大化（备用方法）
# ─────────────────────────────────────────────

def max_sharpe(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    rf: float = 0.03,
    max_vol: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """最大化夏普比率（可选，作为替代优化目标）"""
    n = len(mu)

    def neg_sharpe(w):
        port_ret = mu @ w
        port_vol = np.sqrt(w @ cov_matrix @ w + 1e-12)
        return -(port_ret - rf) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if max_vol is not None and max_vol > 0:
        constraints.append(
            {"type": "ineq", "fun": lambda w: max_vol**2 - w @ cov_matrix @ w}
        )

    bounds = [(float(w_lower[i]), float(w_upper[i])) for i in range(n)]
    w0 = np.clip(np.ones(n) / n, w_lower, w_upper)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-10, "maxiter": 2000})
    return result.x, result.success, result.message
