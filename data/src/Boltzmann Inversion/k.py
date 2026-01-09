import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图

from matplotlib import rcParams
# 设置中文字体，确保使用系统中已安装的中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 这里使用 SimHei（黑体）
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据的函数
def read_data(file_name):
    """
    读取文件并返回数据列表
    """
    data = np.loadtxt(file_name)
    return data[:, 0], data[:, 1]  # 返回键长和对应的值


# 定义误差函数Q
def error_function(k_bond, r_values, U_bond_true, r0=0.7):
    """
    计算误差函数Q，U_bond_true是真实势能数据，r_values是键长数据，k_bond是拟合参数
    """
    U_bond_fit = k_bond * (r_values - r0) ** 2  # 拟合势能公式
    Q = np.sum((U_bond_true - U_bond_fit) ** 2)  # 最小二乘误差
    return Q


# 主函数
def main():
    # 读取键长数据和真实势能数据

    # 完整的键长数据
    # r_values, _ = read_data('B_S_values.txt')  # 读取键长数据
    # U_bond_true = np.loadtxt('U_bond.txt')  # 读取真实的势能数据
    # 0.65开始的键长数据
    r_values, _ = read_data('B_S_valuesfrom0.6.txt')  # 读取键长数据
    U_bond_true = np.loadtxt('U_bondfrom0.6.txt')  # 读取真实的势能数据

    # 初始k_bond值，可以设置为一个初始猜测值
    k_bond_initial = 1.474275e-17 #1e-17
    # k_bond_initial = 1.6496180670873464e-17

    # 使用最小二乘法拟合k_bond
    result = minimize(error_function, k_bond_initial, args=(r_values, U_bond_true))

    # 输出拟合结果
    k_bond_optimal = result.x[0]
    print(f"最优的 k_bond 值为: {6.022e20 * k_bond_optimal:.6e}")

    # 使用拟合的k_bond计算势能
    U_bond_fit = k_bond_optimal * (r_values - 0.711) ** 2  # r0=0.7


    # 输出拟合结果到文件
    np.savetxt('B_S_U_bond_fitted.txt', U_bond_fit, fmt='%.6e')
    print("拟合结果已保存到 'B_S_U_bond_fitted.txt'")

    # 绘制真实势能与拟合势能的对比图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(r_values, U_bond_true, label='真实势能 (True Energy)', color='blue', linestyle='-', linewidth=2)
    plt.plot(r_values, U_bond_fit, label='拟合势能 (Fitted Energy)', color='red', linestyle='--', linewidth=2)
    plt.xlabel('键长 (Bond Length) [nm]')
    plt.ylabel('势能 (Energy) [单位]')
    plt.title('真实势能与拟合势能对比图')
    plt.legend()
    plt.grid(True)
    plt.show()  # 显示图像


if __name__ == "__main__":
    main()
