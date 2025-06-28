
import matplotlib.pyplot as plt

def plot_initial_magnetization_curve(H, B):
    """绘制起始磁化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(H, B * 1000, 'b-', linewidth=2)  # B转换为mT
    plt.title('起始磁化曲线 (基本磁化曲线)', fontsize=15)
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (mT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_dynamic_hysteresis_loop(H, B, params):
    """绘制动态磁滞回线"""
    plt.figure(figsize=(10, 8))
    plt.plot(H, B * 1000, 'r-', linewidth=1.5)
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

def plot_comparison(chi_values, labels, h_max=150.0, n_points=500):
    """绘制不同钉扎力参数下的磁滞回线比较"""
    plt.figure(figsize=(10, 8))
    
    for chi_val, label in zip(chi_values, labels):
        # 模拟磁滞回线
        H_cycle, B_cycle = dynamic_hysteresis_loop(h_max=h_max, n_points=n_points, chi=chi_val)
        plt.plot(H_cycle, B_cycle * 1000, label=label, linewidth=1.5)
    
    plt.title('不同磁性材料的磁滞回线比较', fontsize=15)
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (mT)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
