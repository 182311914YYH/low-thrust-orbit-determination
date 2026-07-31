"""RTN 坐标系模块

提供批量 RTN 基底计算、RTN ↔ ECI 方向余弦矩阵。
"""
from __future__ import annotations

import numpy as np


def compute_rtn_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单点 RTN 正交基底。

    Args:
        r: (3,) ECI 位置
        v: (3,) ECI 速度

    Returns:
        (R_hat, T_hat, N_hat): (3,), (3,), (3,) 单位正交基
    """
    r_norm = np.linalg.norm(r)
    R_hat = r / r_norm

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    N_hat = h / h_norm

    T_hat = np.cross(N_hat, R_hat)
    return R_hat, T_hat, N_hat


def rtn_to_eci_transform(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算 RTN → ECI 方向余弦矩阵。

    v_eci = C @ v_rtn

    Args:
        r: (3,) ECI 位置
        v: (3,) ECI 速度

    Returns:
        (3, 3) C_RTN^ECI
    """
    R_hat, T_hat, N_hat = compute_rtn_basis(r, v)
    return np.column_stack([R_hat, T_hat, N_hat])


def eci_to_rtn_transform(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算 ECI → RTN 方向余弦矩阵。

    v_rtn = C @ v_eci, where C = (C_RTN^ECI)^T
    """
    return rtn_to_eci_transform(r, v).T


def compute_rtn_basis_batch(
    r_array: np.ndarray, v_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量计算 RTN 基底。

    Args:
        r_array: (N, 3) ECI 位置
        v_array: (N, 3) ECI 速度

    Returns:
        R_hat, T_hat, N_hat: (N, 3) each
    """
    N = r_array.shape[0]
    r_norm = np.linalg.norm(r_array, axis=1, keepdims=True)  # (N, 1)
    R_hat = r_array / r_norm

    h = np.cross(r_array, v_array)  # (N, 3)
    h_norm = np.linalg.norm(h, axis=1, keepdims=True)
    N_hat = h / h_norm

    T_hat = np.cross(N_hat, R_hat)
    return R_hat, T_hat, N_hat
