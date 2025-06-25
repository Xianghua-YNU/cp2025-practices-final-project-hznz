
import numpy as np
from scipy.optimize import brentq
from utils import U, dU, Js, mu0

def solve_j(h, j_prev, chi):
    """
    求解给定磁场强度H和前一状态Jp的磁极化强度J
    使用能量最小化模型(标量形式)
    """
    # 尝试J=Jp是否满足条件
    if abs(dU(j_prev) - h) <= chi:
        return j_prev
    
    # 定义两种情况下的方程
    def eq1(j):  # J > Jp 情况
        return dU(j) - (h - chi)
    
    def eq2(j):  # J < Jp 情况
        return dU(j) - (h + chi)
    
    # 求解两种情况的可能解
    solutions = []
    
    # 情况1: J > Jp (在[Jp, Js]区间搜索)
    try:
        j1 = brentq(eq1, j_prev, Js * 0.99)
        solutions.append((j1, U(j1) - h*j1 + chi*abs(j1 - j_prev)))
    except ValueError:
        pass
    
    # 情况2: J < Jp (在[-Js, Jp]区间搜索)
    try:
        j2 = brentq(eq2, -Js * 0.99, j_prev)
        solutions.append((j2, U(j2) - h*j2 + chi*abs(j2 - j_prev)))
    except ValueError:
        pass
    
    # 选择能量最小的解
    if not solutions:
        return j_prev  # 无解时返回Jp
    
    solutions.sort(key=lambda x: x[1])
    return solutions[0][0]

def initial_magnetization_curve(h_max=150.0, n_points=100, chi=40.0):
    """模拟起始磁化曲线 (从退磁状态到饱和)"""
    H_inc = np.linspace(0, h_max, n_points)  # 单调增加的H
    J_vals = []
    B_vals = []
    j_prev = 0.0  # 初始状态 (退磁)
    
    for h in H_inc:
        j = solve_j(h, j_prev, chi)
        b = mu0 * h + j  # B = μ0*H + J
        
        J_vals.append(j)
        B_vals.append(b)
        j_prev = j  # 更新前一状态
    
    return H_inc, np.array(B_vals)

def dynamic_hysteresis_loop(h_max=150.0, n_points=200, chi=40.0):
    """模拟动态磁滞回线 (饱和->负饱和->饱和)"""
    # H变化路径: 0->Hmax->-Hmax->Hmax
    H_cycle = np.concatenate([
        np.linspace(0, h_max, n_points//4),       # 0 -> Hmax
        np.linspace(h_max, -h_max, n_points//2),   # Hmax -> -Hmax
        np.linspace(-h_max, h_max, n_points//2),   # -Hmax -> Hmax
        np.linspace(h_max, 0, n_points//4)         # Hmax -> 0
    ])
    
    J_vals = []
    B_vals = []
    j_prev = 0.0  # 初始状态 (退磁)
    
    for h in H_cycle:
        j = solve_j(h, j_prev, chi)
        b = mu0 * h + j  # B = μ0*H + J
        
        J_vals.append(j)
        B_vals.append(b)
        j_prev = j  # 更新前一状态
    
    return H_cycle, np.array(B_vals)
