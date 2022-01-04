import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow import keras

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import random

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

import sys

# Ansatz hyperparameters
n_qubits = 8  # needs to be even.
depth = 2
n_params = 2 * depth * n_qubits


data_points = 1000

qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


data = make_moons(n_samples=data_points, shuffle=False, noise=None, random_state=42)
x = normalize_data(data[0])
y = data[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train_bin = np.array(x_train, dtype=np.float32)
x_test_bin = np.array(x_test, dtype=np.float32)


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


def u_encode(n_qubits, x):
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even so we can encode the x and y value in the same way.")
    bits_per_float = int(n_qubits / 2)

    x_binary = aprox_to_binary(x[0], bits_per_float)
    y_binary = aprox_to_binary(x[1], bits_per_float)
    total_binary = x_binary + y_binary

    circuit = cirq.Circuit()

    for i in range(n_qubits):
        if total_binary[i] == 1:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [u_encode(n_qubits, x) for x in x_train_bin]
x_test_circ = [u_encode(n_qubits, x) for x in x_test_bin]

x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


class CircuitLayerBuilder():
    def __init__(self, data_qubits):
        self.data_qubits = data_qubits

    # Layer of single qubit rotations
    def rot_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(symbol)(qubit))

    # Layer of entangling CZ(i,i+1 % n_qubits) gates.
    def entangling_layer(self, circuit, n_qubits):
        for i, qubit in enumerate(self.data_qubits):
            circuit.append(cirq.CZ(cirq.GridQubit(i, 0), cirq.GridQubit((i + 1) % n_qubits, 0)))


def create_quantum_model(n_qubits):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(n_qubits, 1)  # a 4x4 grid.
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits
    )
    # Then add layers (experiment by adding more).

    for i in range(depth):
        builder.rot_layer(circuit, cirq.rz, "rz1"+str([i]))
        builder.rot_layer(circuit, cirq.ry, "ry1"+str([i]))
        builder.entangling_layer(circuit, n_qubits)


    return circuit

model_circuit = create_quantum_model(n_qubits)

print(model_circuit.to_text_diagram(transpose=False))


def predict(circuit, observable):
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


def binary_classifier(n):
    if n >= 0:
        return 1
    else:
        return -1


measurements = []
for i, qubit in enumerate(qubits):
    measurements.append(cirq.Z(qubit))


output_layer = tfq.layers.PQC(model_circuit, measurements)

# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    output_layer
])


y_train_hinge = 2.0*y_train-1.0
y_test_hinge = 2.0*y_test-1.0


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])

print(model.summary())

NUM_EXAMPLES = len(x_train_tfcirc)


qnn_history = model.fit(
      x_train_tfcirc, y_train_hinge,
      batch_size=64,
      epochs=10,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = model.evaluate(x_test_tfcirc, y_test)



