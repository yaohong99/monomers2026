import numpy as np

# 玻尔兹曼常数 (J/K)
k_B = 1.380649e-23  # J/K

# 温度 (假设温度为300K)
T = 300  # K


# 读取数据的函数
def read_data(file_name):
    """
    读取文件并返回键长和概率密度值的列表
    """
    data = np.loadtxt(file_name)
    bond_lengths = data[:, 0]  # 第一列是键长
    probabilities = data[:, 1]  # 第二列是对应的概率密度
    return bond_lengths, probabilities


# 归一化键长数据
def normalize_bond_lengths(bond_lengths):
    """
    归一化键长数据到 [0, 1] 区间
    """
    bond_length_min = np.min(bond_lengths)
    bond_length_max = np.max(bond_lengths)
    normalized_bond_lengths = (bond_lengths - bond_length_min) / (bond_length_max - bond_length_min)
    return normalized_bond_lengths


# 计算玻尔兹曼势能
def calculate_potential(probabilities, k_B, T):
    """
    使用玻尔兹曼反演计算势能
    U = -k_B * T * ln(p)
    """
    # 避免对零取对数，防止概率值为零
    probabilities[probabilities == 0] = 1e-10
    # 计算势能
    potential = (-1) * k_B * T * np.log(probabilities)
    return potential


# 主函数
def main():
    # 读取数据
    bond_lengths, probabilities = read_data('B_S_values.txt')

    # 归一化键长数据
    normalized_bond_lengths = normalize_bond_lengths(probabilities)

    # 计算玻尔兹曼势能
    # 使用归一化后的键长数据作为概率密度
    bond_potential = calculate_potential(normalized_bond_lengths, k_B, T)

    # 输出势能到U_bond.txt
    np.savetxt('U_bond.txt', bond_potential, fmt='%.6e')

    # 打印完成信息
    print("键长势能计算完成并保存至 'U_bond.txt'")


if __name__ == "__main__":
    main()
