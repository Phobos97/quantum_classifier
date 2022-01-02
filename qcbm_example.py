from scipy.optimize import minimize
from time import time
from functools import partial
import cirq
import sympy
import matplotlib.pyplot as plt
import numpy as np


## Ansatz hyperparameters
n_qubits = 3
depth = 2
n_params = 2 * depth * n_qubits

# Begin with statevector simulator
shots = 0

def gaussian_pdf(num_bit, mu, sigma):
    '''get gaussian distribution function'''
    x = np.arange(2**num_bit)
    pl = 1. / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    return pl/pl.sum()

pg = gaussian_pdf(n_qubits, mu=2**(n_qubits-1)-0.5, sigma=2**(n_qubits-2))
plt.plot(pg, 'ro')
plt.show()


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


# Variational circuit, i.e., the ansatz.
def variational_circuit(n_qubits, depth, theta):
    if len(theta) != (2 * depth * n_qubits):
        raise ValueError("Theta of incorrect dimension, must equal 2*depth*n_qubits")

    # Initializing qubits and circuit
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()

    # Adding layers of rotation gates and entangling gates.
    for d in range(depth):
        # Adding single qubit rotations
        circuit.append(rot_z_layer(n_qubits, theta[d * 2 * n_qubits: (d + 1) * 2 * n_qubits: 2]))
        circuit.append(rot_y_layer(n_qubits, theta[d * 2 * n_qubits + 1: (d + 1) * 2 * n_qubits + 1: 2]))
        # Adding entangling layer
        circuit.append(entangling_layer(n_qubits))

    return circuit


theta_entry_symbols = [sympy.Symbol('theta_' + str(i)) for i in range(2 * n_qubits * depth)]
theta_symbol = sympy.Matrix(theta_entry_symbols)
ansatz = variational_circuit(n_qubits, depth, theta_symbol)
print(ansatz.to_text_diagram(transpose=True))


# Estimate all probabilities of the PQCs distribution.
def estimate_probs(circuit, theta, n_shots=shots):
    # Creating parameter resolve dict by adding state and theta.
    try:
        theta_mapping = [('theta_' + str(i), theta[i]) for i in range(len(theta))]
    except IndexError as error:
        print("Could not resolve theta symbol, array of wrong size.")
    resolve_dict = dict(theta_mapping)
    resolver = cirq.ParamResolver(resolve_dict)
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    # Use statevector simulator
    if n_shots == 0:
        final_state = cirq.final_state_vector(resolved_circuit)
        probs = np.array([abs(final_state[i]) ** 2 for i in range(len(final_state))])
    # Run the circuit.
    else:
        # Adding measurement at the end.
        resolved_circuit.append(cirq.measure(*resolved_circuit.all_qubits(), key='m'))
        results = cirq.sample(resolved_circuit, repetitions=n_shots)
        frequencies = results.histogram(key='m')
        probs = np.zeros(2 ** n_qubits)
        for key, value in frequencies.items():
            probs[key] = value / n_shots

    return probs


# Function that computes the kernel for the MMD loss
def multi_rbf_kernel(x, y, sigma_list):
    '''
    multi-RBF kernel.

    Args:
        x (1darray|2darray): the collection of samples A.
        x (1darray|2darray): the collection of samples B.
        sigma_list (list): a list of bandwidths.

    Returns:
        2darray: kernel matrix.
    '''
    ndim = x.ndim
    if ndim == 1:
        exponent = np.abs(x[:, None] - y[None, :]) ** 2
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * exponent)
    return K


# Function that computes expectation of kernel in MMD loss
def kernel_expectation(px, py, kernel_matrix):
    return px.dot(kernel_matrix).dot(py)


# Function that computes the squared MMD loss related to the given kernel_matrix.
def squared_MMD_loss(probs, target, kernel_matrix):
    dif_probs = probs - target
    return kernel_expectation(dif_probs, dif_probs, kernel_matrix)


# The loss function that we aim to minimize.
def loss(theta, circuit, target, kernel_matrix, n_shots=shots):
    probs = estimate_probs(circuit, theta, n_shots=n_shots)
    return squared_MMD_loss(probs, target, kernel_matrix)


# Cheat and get gradient.
def gradient(theta, target, kernel_matrix, n_shots=shots):
    prob = estimate_probs(ansatz, theta, n_shots=shots)
    grad = []
    for i in range(len(theta)):
        # pi/2 phase
        theta[i] += np.pi / 2.
        prob_pos = estimate_probs(ansatz, theta, n_shots=shots)
        # -pi/2 phase
        theta[i] -= np.pi
        prob_neg = estimate_probs(ansatz, theta, n_shots=shots)
        # recover
        theta[i] += np.pi / 2.
        grad_pos = kernel_expectation(prob, prob_pos, kernel_matrix) - kernel_expectation(prob, prob_neg, kernel_matrix)
        grad_neg = kernel_expectation(target, prob_pos, kernel_matrix) - kernel_expectation(target, prob_neg,
                                                                                            kernel_matrix)
        grad.append(grad_pos - grad_neg)
    return np.array(grad)


# MMD kernel
basis = np.arange(2**n_qubits)
sigma_list = [0.25,4]
kernel_matrix = multi_rbf_kernel(basis, basis, sigma_list)

# Initial theta
np.random.seed(0)
theta0 = np.random.random(n_params)*2*np.pi

# Initializing loss function with our ansatz, target and kernel matrix
loss_ansatz = partial(loss, circuit=ansatz, target=pg, kernel_matrix=kernel_matrix)

# Callback function to track status
step = [0]
tracking_cost = []


def callback(x, *args, **kwargs):
    step[0] += 1
    tracking_cost.append(loss_ansatz(x))
    print('step = %d, loss = %s'%(step[0], loss_ansatz(x)))

# Training the QCBM.
start_time = time()
final_params = minimize(loss_ansatz,
                        theta0,
                        method="L-BFGS-B",
                        jac=partial(gradient, target=pg, kernel_matrix=kernel_matrix),
                        tol=10**-5,
                        options={'maxiter':50, 'disp': 0, 'gtol':1e-10, 'ftol':0},
                        callback=callback)
end_time = time()
print(end_time-start_time)


print(final_params)

plt.plot(list(range(len(tracking_cost))), tracking_cost)
plt.show()
plt.plot(estimate_probs(ansatz, final_params.x), 'ro')
plt.show()

