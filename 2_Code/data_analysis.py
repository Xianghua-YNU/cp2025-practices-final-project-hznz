
import numpy as np

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
