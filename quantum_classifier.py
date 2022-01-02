from scipy.optimize import minimize
from time import time
from functools import partial
import cirq
import sympy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import random

## Ansatz hyperparameters
n_qubits = 4  # needs to be even.
depth = 2
n_params = 2 * depth * n_qubits

data_points = 100


def gaussian_pdf(num_bit, mu, sigma):
    '''get gaussian distribution function'''
    x = np.arange(2**num_bit)
    pl = 1. / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    return pl/pl.sum()


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


x = make_moons(n_samples=data_points, shuffle=False, noise=None, random_state=42)
data = normalize_data(x[0])
labels = x[1]
plt.scatter(data[:, 0], data[:, 1])
plt.show()


def aprox_to_binary(n, bits):
    if n > 1 or n < 0:
        raise ValueError("number must be between 0 or 1 both included")
    result = [0]*bits

    for k in range(1, bits + 1):
        if n - 0.5**k < 0:
            continue
        else:
            n -= 0.5**k
            result[k - 1] = 1
    return result


def u_encode(n_qubits, x, y):
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even so we can encode the x and y value in the same way.")
    bits_per_float = int(n_qubits / 2)

    x_binary = aprox_to_binary(x, bits_per_float)
    y_binary = aprox_to_binary(y, bits_per_float)
    total_binary = x_binary + y_binary

    for i in range(n_qubits):
        if total_binary[i] == 1:
            yield cirq.X(cirq.GridQubit(i, 0))


# Layer of single qubit z rotations
def rot_z_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many or few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.rz(parameters[i])(cirq.GridQubit(i, 0))


# Layer of single qubit y rotations
def rot_y_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many of few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.ry(parameters[i])(cirq.GridQubit(i, 0))


# Layer of entangling CZ(i,i+1 % n_qubits) gates.
def entangling_layer(n_qubits):
    if n_qubits == 2:
        yield cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))
        return
    for i in range(n_qubits):
        yield cirq.CZ(cirq.GridQubit(i, 0), cirq.GridQubit((i + 1) % n_qubits, 0))


def all_z_layer(n_qubits):
    for i in range(n_qubits):
        yield cirq.Z(cirq.GridQubit(i, 0))


# Variational circuit, i.e., the ansatz.
def variational_circuit(n_qubits, depth, theta, x, y):
    if len(theta) != (2 * depth * n_qubits):
        raise ValueError("Theta of incorrect dimension, must equal 2*depth*n_qubits")

    # Initializing qubits and circuit
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()

    # encode our data points
    circuit.append(u_encode(n_qubits, x, y))

    # Adding layers of rotation gates and entangling gates.
    for d in range(depth):
        # Adding single qubit rotations
        circuit.append(rot_z_layer(n_qubits, theta[d * 2 * n_qubits: (d + 1) * 2 * n_qubits: 2]))
        circuit.append(rot_y_layer(n_qubits, theta[d * 2 * n_qubits + 1: (d + 1) * 2 * n_qubits + 1: 2]))
        # Adding entangling layer
        circuit.append(entangling_layer(n_qubits))

    return circuit


# Initial theta
np.random.seed(0)
theta0 = np.random.random(n_params)*2*np.pi

# theta_entry_symbols = [sympy.Symbol('theta_' + str(i)) for i in range(2 * n_qubits * depth)]
# theta_symbol = sympy.Matrix(theta_entry_symbols)
# ansatz = variational_circuit(n_qubits, depth, theta_symbol, x=0.5, y=1)
# print(ansatz.to_text_diagram(transpose=False))

ansatzes = []
for i in range(data_points):
    ansatzes.append(variational_circuit(n_qubits, depth, theta0, data[i, 0], data[i, 1]))

observable_z = cirq.unitary(cirq.Z)
for i in range(n_qubits - 1):
    observable_z = np.kron(observable_z, cirq.unitary(cirq.Z))


def predict(ansatz, observable):
    s = cirq.Simulator()
    simulated = s.simulate(ansatz)
    psi_x_theta = np.array(simulated.state_vector())

    # this is formula (1) from the Project description
    step1 = np.matmul(observable, psi_x_theta)
    f_theta = np.matmul(psi_x_theta.conjugate().transpose(), step1)

    # turning above function into a binary classifier
    if f_theta.real < 0:
        return 0
    else:
        return 1


random_index = random.randrange(data_points)
random_ansatz = ansatzes[random_index]
print("input: " + str(data[random_index]))
print("random predict: " + str(predict(random_ansatz, observable_z)))
print("actual label: " + str(labels[random_index]))

