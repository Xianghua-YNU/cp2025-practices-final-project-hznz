import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 材料参数 (参考附件二表1)
A = 65.0          # 场强参数 [A/m]
Js = 0.44         # 饱和磁极化强度 [T]
chi = 40.0        # 钉扎力参数 [A/m]
mu0 = 4e-7 * np.pi  # 真空磁导率 [H/m]

# 样品参数 (参考附件一)
L = 0.130         # 平均磁路长度 [m]
S = 1.24e-4       # 截面积 [m²]
N1 = 100          # 初级线圈匝数
N2 = 100          # 次级线圈匝数

# 内能函数及其导数 (Bergqvist模型，附件二公式)
def U(J):
    """磁极化强度的内能函数"""
    # 处理接近饱和点的情况，避免数值问题
    cos_arg = np.pi * J / (2 * Js)
    # 当J接近Js时，cos_arg接近π/2，cos值接近0，取最大值避免log(0)
    cos_val = np.cos(np.clip(cos_arg, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))
    return -(2 * A * Js / np.pi) * np.log(np.maximum(cos_val, 1e-6))

def dU(J):
    """内能函数的一阶导数 = dU/dJ"""
    tan_arg = np.pi * J / (2 * Js)
    # 当J接近Js时，tan_arg接近π/2，tan值会很大，需要限制范围
    tan_val = np.tan(np.clip(tan_arg, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))
    return A * tan_val

# 磁滞模型求解器 (标量简化版)
def solve_j(h, j_prev):
    """
    求解给定磁场强度H和前一状态Jp的磁极化强度J
    使用附件二中的能量最小化模型(标量形式)
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

# 1. 起始磁化曲线 (单调增加磁场)
def initial_magnetization_curve(h_max=150.0, n_points=100):
    """模拟起始磁化曲线 (从退磁状态到饱和)"""
    H_inc = np.linspace(0, h_max, n_points)  # 单调增加的H
    J_vals = []
    B_vals = []
    j_prev = 0.0  # 初始状态 (退磁)
    
    for h in H_inc:
        j = solve_j(h, j_prev)
        b = mu0 * h + j  # B = μ0*H + J
        
        J_vals.append(j)
        B_vals.append(b)
        j_prev = j  # 更新前一状态
    
    return H_inc, np.array(B_vals)

# 2. 动态磁滞回线 (完整周期)
def dynamic_hysteresis_loop(h_max=150.0, n_points=200):
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
        j = solve_j(h, j_prev)
        b = mu0 * h + j  # B = μ0*H + J
        
        J_vals.append(j)
        B_vals.append(b)
        j_prev = j  # 更新前一状态
    
    return H_cycle, np.array(B_vals)

# 3. 计算磁特征参数
def calculate_parameters(H, B):
    """从磁滞回线计算Bs, Br, Hc"""
    # 饱和磁感应强度 (正负峰值)
    Bs_pos = np.max(B)
    Bs_neg = np.min(B)
    
    # 剩磁 (H=0时的B值)
    # 找到H从正到负穿越0的点
    zero_crossings = np.where(np.diff(np.sign(H)))[0]
    if len(zero_crossings) >= 2:
        idx1, idx2 = zero_crossings[0], zero_crossings[1]
        Br = np.interp(0, [H[idx1], H[idx2]], [B[idx1], B[idx2]])
    else:
        Br = B[np.argmin(np.abs(H))]  # 备用方法
    
    # 矫顽力 (B=0时的H值)
    # 找到B从正到负穿越0的点
    zero_crossings_b = np.where(np.diff(np.sign(B)))[0]
    if len(zero_crossings_b) >= 2:
        idx1, idx2 = zero_crossings_b[0], zero_crossings_b[1]
        Hc = np.interp(0, [B[idx1], B[idx2]], [H[idx1], H[idx2]])
    else:
        Hc = H[np.argmin(np.abs(B))]  # 备用方法
    
    return {
        'Bs+': Bs_pos,
        'Bs-': Bs_neg,
        'Br+': Br,
        'Br-': -Br,  # 对称值
        'Hc+': abs(Hc),
        'Hc-': -abs(Hc)
    }

# 主程序
if __name__ == "__main__":
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =============================================
    # 1. 起始磁化曲线 (基本磁化曲线)
    # =============================================
    H_initial, B_initial = initial_magnetization_curve(h_max=150.0, n_points=100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(H_initial, B_initial * 1000, 'b-', linewidth=2)  # B转换为mT
    plt.title('起始磁化曲线 (基本磁化曲线)', fontsize=15)
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (mT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # =============================================
    # 2. 动态磁滞回线
    # =============================================
    H_cycle, B_cycle = dynamic_hysteresis_loop(h_max=150.0, n_points=400)
    
    # 计算特征参数
    params = calculate_parameters(H_cycle, B_cycle)
    print("磁特征参数:")
    print(f"饱和磁感应强度 Bs+ = {params['Bs+']*1000:.2f} mT")
    print(f"剩磁 Br+ = {params['Br+']*1000:.2f} mT")
    print(f"矫顽力 Hc+ = {params['Hc+']:.2f} A/m")
    
    plt.figure(figsize=(10, 8))
    plt.plot(H_cycle, B_cycle * 1000, 'r-', linewidth=1.5)
    plt.title('动态磁滞回线', fontsize=15)
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (mT)', fontsize=12)
    
    # 标记特征点
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.scatter(0, params['Br+']*1000, s=100, c='blue', zorder=5)
    plt.scatter(params['Hc+'], 0, s=100, c='green', zorder=5)
    plt.scatter(150, params['Bs+']*1000, s=100, c='red', zorder=5)
    
    plt.annotate(f'$B_s$ = {params["Bs+"]*1000:.1f} mT', 
                (150, params['Bs+']*1000), 
                (130, params['Bs+']*1000 - 10),
                arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'$B_r$ = {params["Br+"]*1000:.1f} mT', 
                (0, params['Br+']*1000), 
                (-20, params['Br+']*1000),
                arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'$H_c$ = {params["Hc+"]:.1f} A/m', 
                (params['Hc+'], 0), 
                (params['Hc+'] + 10, -20),
                arrowprops=dict(arrowstyle='->'))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # =============================================
    # 3. 不同材料比较 (改变钉扎力参数chi)
    # =============================================
    plt.figure(figsize=(10, 8))
    
    # 不同钉扎力参数 (模拟硬磁和软磁材料)
    for chi_val, label in zip([20, 40, 60], ['软磁 (χ=20)', '中硬磁 (χ=40)', '硬磁 (χ=60)']):
        # 定义局部求解函数，使用当前的chi_val
        def solve_j_local(h, j_prev):
            """局部求解函数，使用当前的钉扎力参数"""
            # 尝试J=Jp是否满足条件
            if abs(dU(j_prev) - h) <= chi_val:
                return j_prev
            
            # 定义两种情况下的方程
            def eq1(j):  # J > Jp 情况
                return dU(j) - (h - chi_val)
            
            def eq2(j):  # J < Jp 情况
                return dU(j) - (h + chi_val)
            
            solutions = []
            
            # 情况1: J > Jp (在[Jp, Js]区间搜索)
            try:
                j1 = brentq(eq1, j_prev, Js * 0.99)
                solutions.append((j1, U(j1) - h*j1 + chi_val*abs(j1 - j_prev)))
            except ValueError:
                pass
            
            # 情况2: J < Jp (在[-Js, Jp]区间搜索)
            try:
                j2 = brentq(eq2, -Js * 0.99, j_prev)
                solutions.append((j2, U(j2) - h*j2 + chi_val*abs(j2 - j_prev)))
            except ValueError:
                pass
            
            # 选择能量最小的解
            if not solutions:
                return j_prev  # 无解时返回Jp
            
            solutions.sort(key=lambda x: x[1])
            return solutions[0][0]
        
        # 模拟磁滞回线
        H_temp = np.concatenate([
            np.linspace(0, 150, 100),
            np.linspace(150, -150, 200),
            np.linspace(-150, 150, 200)
        ])
        B_temp = []
        j_prev_temp = 0.0  # 初始状态 (退磁)
        
        for h in H_temp:
            j = solve_j_local(h, j_prev_temp)
            b = mu0 * h + j  # B = μ0*H + J
            B_temp.append(b)
            j_prev_temp = j  # 更新前一状态
        
        plt.plot(H_temp, np.array(B_temp)*1000, label=label, linewidth=1.5)
    
    plt.title('不同磁性材料的磁滞回线比较', fontsize=15)
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (mT)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()