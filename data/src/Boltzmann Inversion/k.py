import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def read_data(file_name):

    data = np.loadtxt(file_name)
    return data[:, 0], data[:, 1]



def error_function(k_bond, r_values, U_bond_true, r0=0.7):


    U_bond_fit = k_bond * (r_values - r0) ** 2
    Q = np.sum((U_bond_true - U_bond_fit) ** 2)
    return Q



def main():
    
    r_values, _ = read_data('B_S_values.txt')
    U_bond_true = np.loadtxt('U_bond.txt')


    k_bond_initial = 1.474275e-17


    result = minimize(error_function, k_bond_initial, args=(r_values, U_bond_true))

    k_bond_optimal = result.x[0]
    print(f"Best k_bond: {6.022e20 * k_bond_optimal:.6e}")

    U_bond_fit = k_bond_optimal * (r_values - 0.711) ** 2  # r0=0.7

    np.savetxt('B_S_U_bond_fitted.txt', U_bond_fit, fmt='%.6e')
    print("The fitted results have been saved to 'B_S_U_bond_fitted.txt'.")


    plt.figure(figsize=(10, 6))
    plt.plot(r_values, U_bond_true, label='(True Energy)', color='blue', linestyle='-', linewidth=2)
    plt.plot(r_values, U_bond_fit, label='(Fitted Energy)', color='red', linestyle='--', linewidth=2)
    plt.xlabel('(Bond Length) [nm]')
    plt.ylabel('(Energy) ')
    plt.title('Comparison of the True and Fitted Potential Energies')
    plt.legend()
    plt.grid(True)
    plt.show()  


if __name__ == "__main__":
    main()
