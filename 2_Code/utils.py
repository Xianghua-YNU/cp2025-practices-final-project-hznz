
import numpy as np

# 材料参数 )
A = 65.0          # 场强参数 [A/m]
Js = 0.44         # 饱和磁极化强度 [T]
chi = 40.0        # 钉扎力参数 [A/m]
mu0 = 4e-7 * np.pi  # 真空磁导率 [H/m]

# 样品参数 
L = 0.130         # 平均磁路长度 [m]
S = 1.24e-4       # 截面积 [m²]
N1 = 100          # 初级线圈匝数
N2 = 100          # 次级线圈匝数

# 内能函数及其导数 (Bergqvist模型)
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
