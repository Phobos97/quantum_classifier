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
from functools import reduce
import math

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

import sys


def filter_mnist(x, y, true_digit, false_digit):
    keep = (y == true_digit) | (y == false_digit)
    x, y = x[keep], y[keep]
    y = y == true_digit
    return x, y


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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


def u_encode_binary(n_qubits, x):
    if n_qubits % 2 != 0:
        raise ValueError(
            "n_qubits must be even so we can encode the x and y value in the same way.")
    bits_per_float = int(n_qubits / 2)

    x_binary = aprox_to_binary(x[0], bits_per_float)
    y_binary = aprox_to_binary(x[1], bits_per_float)
    total_binary = x_binary + y_binary

    circuit = cirq.Circuit()

    for i in range(n_qubits):
        if total_binary[i] == 1:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def u_encode_rot(n_qubits, qubits, x):
    if n_qubits % 2 != 0:
        raise ValueError(
            "n_qubits must be even so we can encode the x and y value in the same way.")
    circuit = cirq.Circuit()

    for i in range(len(x)):
        for k in range(int(n_qubits / 2)):
            if k % 2 == 0:
                circuit.append(cirq.ry(x[i] * 2 * math.pi)
                               (qubits[i*int(n_qubits / 2) + k]))
            else:
                circuit.append(cirq.rz(x[i] * 2 * math.pi)
                               (qubits[i * int(n_qubits / 2) + k]))
    return circuit


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
        if n_qubits == 2:
            circuit.append(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)))
            return
        for i, qubit in enumerate(self.data_qubits):
            circuit.append(cirq.CZ(cirq.GridQubit(i, 0),
                           cirq.GridQubit((i + 1) % n_qubits, 0)))


def create_quantum_model(n_qubits, depth):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(n_qubits, 1)  # a 4x4 grid.
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits
    )
    # Then add layers (experiment by adding more).

    for i in range(depth):
        builder.rot_layer(circuit, cirq.rz, "rz" + str(i))
        builder.rot_layer(circuit, cirq.ry, "ry" + str(i))
        builder.entangling_layer(circuit, n_qubits)

    return circuit


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def create_moons_dataset(data_points):
    data = make_moons(n_samples=data_points, shuffle=False,
                      noise=None, random_state=42)
    x = normalize_data(data[0])
    y = data[1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    return 2, x_train, y_train, x_test, y_test


def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass
    
    return np.array(new_x), np.array(new_y)


def create_mnist_dataset(pixels):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / \
        255.0, x_test[..., np.newaxis]/255.0
    x_train, y_train = filter_mnist(x_train, y_train, 2, 4)
    x_test, y_test = filter_mnist(x_test, y_test, 2, 4)
    x_train_small = tf.image.resize(x_train, (pixels, pixels)).numpy()
    x_test_small = tf.image.resize(x_test, (pixels, pixels)).numpy()
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    return pixels**2, x_train_nocon, y_train_nocon, x_test_small, y_test




def prepare_dataset(choice):
    if (choice == 'moons'):
        return create_moons_dataset(20000)
    elif (choice == 'mnist'):
        return create_mnist_dataset(4)
    else:
        print('invalid dataset choice')
        sys.exit()

def select_encoding_method(choice):
    if (choice == 'rot'):
        return u_encode_rot
    elif (choice == 'bin'):
        return u_encode_binary
    else:
        print('invalid encoder choice')
        sys.exit()

def main():
    args = sys.argv[1:]
    if (len(args) != 4):
        print('invalid arguments: [dataset] [encoder] [n_qubits] [depth]')
        return

    dataset = args[0]
    encoder = args[1]
    n_qubits = int(args[2])
    depth = args[3]


    dim, x_train, y_train, x_test, y_test = prepare_dataset(dataset)

    x_train_bin = np.array(x_train, dtype=np.float32)
    x_test_bin = np.array(x_test, dtype=np.float32)

    print(x_train_bin)
    print(x_train_bin.shape)
    print(x_train_bin.reshape(10409, 4, 4))

    # choose method of encoding data for use in quantum classifier
    encode_function = select_encoding_method(encoder)

    if (dim % n_qubits != 0):
        print('error: dimension not divisible by qubits')
        return
    

    n_qubits *= dim/n_qubits
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]

    x_train_circ = [encode_function(n_qubits, x) for x in x_train_bin]
    x_test_circ = [encode_function(n_qubits, x) for x in x_test_bin]

    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    model_circuit = create_quantum_model(n_qubits, qubits, depth)

    print(model_circuit.to_text_diagram(transpose=False))

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]

    output_layer = tfq.layers.PQC(model_circuit, observables)

    # Build the Keras model.
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        output_layer
    ])

    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    print(model.summary())

    NUM_EXAMPLES = len(x_train_tfcirc)

    qnn_history = model.fit(
        x_train_tfcirc, y_train_hinge,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_hinge))

    qnn_results = model.evaluate(x_test_tfcirc, y_test)

    print(model.predict(x_test_tfcirc))
    print(y_test)


if __name__ == "__main__":
    main()
