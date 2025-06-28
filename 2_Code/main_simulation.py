
import matplotlib.pyplot as plt
from numerical_methods import initial_magnetization_curve, dynamic_hysteresis_loop
from data_analysis import calculate_parameters
from visualization import plot_initial_magnetization_curve, plot_dynamic_hysteresis_loop, plot_comparison
from utils import chi

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    # 1. 起始磁化曲线 (基本磁化曲线)
    H_initial, B_initial = initial_magnetization_curve(h_max=150.0, n_points=100, chi=chi)
    plot_initial_magnetization_curve(H_initial, B_initial)
    
    # 2. 动态磁滞回线
    H_cycle, B_cycle = dynamic_hysteresis_loop(h_max=150.0, n_points=400, chi=chi)
    params = calculate_parameters(H_cycle, B_cycle)
    print("磁特征参数:")
    print(f"饱和磁感应强度 Bs+ = {params['Bs+']*1000:.2f} mT")
    print(f"剩磁 Br+ = {params['Br+']*1000:.2f} mT")
    print(f"矫顽力 Hc+ = {params['Hc+']:.2f} A/m")
    plot_dynamic_hysteresis_loop(H_cycle, B_cycle, params)
    
    # 3. 不同材料比较 (改变钉扎力参数chi)
    chi_values = [20, 40, 60]
    labels = ['软磁 (χ=20)', '中硬磁 (χ=40)', '硬磁 (χ=60)']
    plot_comparison(chi_values, labels, h_max=150.0, n_points=500)

if __name__ == "__main__":
    main()
