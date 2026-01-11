import numpy as np

k_B = 1.380649e-23  # J/K

T = 300  # K

def read_data(file_name):

    data = np.loadtxt(file_name)
    bond_lengths = data[:, 0]
    probabilities = data[:, 1]
    return bond_lengths, probabilities

def normalize_bond_lengths(bond_lengths):

    bond_length_min = np.min(bond_lengths)
    bond_length_max = np.max(bond_lengths)
    normalized_bond_lengths = (bond_lengths - bond_length_min) / (bond_length_max - bond_length_min)
    return normalized_bond_lengths


def calculate_potential(probabilities, k_B, T):

    probabilities[probabilities == 0] = 1e-10

    potential = (-1) * k_B * T * np.log(probabilities)
    return potential



def main():

    bond_lengths, probabilities = read_data('B_S_values.txt')

    normalized_bond_lengths = normalize_bond_lengths(probabilities)

    bond_potential = calculate_potential(normalized_bond_lengths, k_B, T)

    np.savetxt('U_bond.txt', bond_potential, fmt='%.6e')
    print("The bond length potential energy calculation has been completed and saved to 'U_bond.txt'.")



if __name__ == "__main__":
    main()
