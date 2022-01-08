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
from keras.layers import Concatenate

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


def u_encode_rot(dim, n_qubits, qubits, x):
    encoded_data = []
    for i in range(np.size(x, 1)):
        circuit = cirq.Circuit()
        for k in range(np.size(x, 0)):
            circuit.append(cirq.ry(x[k, i] * 2 * math.pi)(qubits[i*n_qubits + k]))

        encoded_data.append(circuit)
    return encoded_data


class CircuitLayerBuilder():
    def __init__(self, data_qubits):
        self.data_qubits = data_qubits

    # Layer of single qubit rotations
    def rot_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(symbol)(qubit))

    # Layer of entangling CZ(i,i+1 % n_qubits) gates.
    def entangling_layer(self, circuit):
        if len(self.data_qubits) == 2:
            circuit.append(cirq.CZ(self.data_qubits[0], self.data_qubits[1]))
            return
        for i in range(len(self.data_qubits)):
            circuit.append(cirq.CZ(self.data_qubits[i], self.data_qubits[(i + 1) % len(self.data_qubits)]))


def create_quantum_model(n_qubits, l, depth):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(n_qubits * l, 1)  # a 4x4 grid.

    circuits = []
    for k in range(l):
        circuit = cirq.Circuit()

        builder = CircuitLayerBuilder(
            data_qubits=data_qubits[k*n_qubits:(k+1)*n_qubits]
        )
        # Then add layers (experiment by adding more).

        for i in range(depth):
            builder.rot_layer(circuit, cirq.rz, "rz" + str(k) + "-" + str(i))
            builder.rot_layer(circuit, cirq.ry, "ry" + str(k) + "-" + str(i))
            builder.entangling_layer(circuit)

        circuits.append(circuit)

    return circuits


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


# def remove_contradicting(xs, ys):
#     mapping = collections.defaultdict(set)
#     orig_x = {}
#     # Determine the set of labels for each unique image:
#     for x, y in zip(xs, ys):
#         orig_x[tuple(x.flatten())] = x
#         mapping[tuple(x.flatten())].add(y)
#
#     new_x = []
#     new_y = []
#     for flatten_x in mapping:
#         x = orig_x[flatten_x]
#         labels = mapping[flatten_x]
#         if len(labels) == 1:
#             new_x.append(x)
#             new_y.append(next(iter(labels)))
#         else:
#             # Throw out images that match more than one label.
#             pass
#
#     return np.array(new_x), np.array(new_y)


def create_mnist_dataset(pixels):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / \
                      255.0, x_test[..., np.newaxis] / 255.0
    x_train, y_train = filter_mnist(x_train, y_train, 0, 1)
    x_test, y_test = filter_mnist(x_test, y_test, 0, 1)
    x_train_small = tf.image.resize(x_train, (pixels, pixels)).numpy()
    x_test_small = tf.image.resize(x_test, (pixels, pixels)).numpy()
    # x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    return pixels ** 2, x_train_small, y_train, x_test_small, y_test


def prepare_dataset(choice):
    if (choice == 'moons'):
        return create_moons_dataset(20000)
    elif (choice == 'mnist'):
        return create_mnist_dataset(8)
    else:
        print('invalid dataset choice')
        sys.exit()


def select_encoding_method(choice):
    if (choice == 'rot'):
        return u_encode_rot
    # elif (choice == 'bin'):
    #     return u_encode_binary
    else:
        print('invalid encoder choice')
        sys.exit()


def main():
    args = sys.argv[1:]
    if (len(args) != 4):
        print('invalid arguments: [dataset] [encoder] [n_qubits] [depth]')
        dataset = "mnist"
        encoder = "rot"
        n_qubits = 4
        depth = 2
    else:
        dataset = args[0]
        encoder = args[1]
        n_qubits = int(args[2])
        depth = int(args[3])

    datapoints = 1000

    dim, x_train, y_train, x_test, y_test = prepare_dataset(dataset)
    y_train = y_train[0:datapoints]
    y_test = y_test[0:datapoints]

    if dim % n_qubits != 0:
        raise ValueError('Error: dim % n_qubits != 0')

    l = int(dim / n_qubits)

    x_train_bin = np.array(x_train, dtype=np.float32).reshape((-1, int(dim/l), l))[0:datapoints]
    x_test_bin = np.array(x_test, dtype=np.float32).reshape((-1, int(dim/l), l))[0:datapoints]

    # choose method of encoding data for use in quantum classifier
    # currently only rotational encoding is supported
    encode_function = select_encoding_method(encoder)

    qubits = [cirq.GridQubit(i, 0) for i in range(dim)]

    x_train_circ = [encode_function(dim, n_qubits, qubits, x) for x in x_train_bin]
    x_test_circ = [encode_function(dim, n_qubits, qubits, x) for x in x_test_bin]

    train_data = []
    test_data = []
    for i in range(l):
        train_data.append([x[i] for x in x_train_circ])
        test_data.append([x[i] for x in x_test_circ])

    x_train_tfcirc = []
    x_test_tfcirc = []
    for i in range(l):
        x_train_tfcirc.append(tfq.convert_to_tensor(train_data[i]))
        x_test_tfcirc.append(tfq.convert_to_tensor(test_data[i]))

    model_circuits = create_quantum_model(n_qubits, l, depth)

    for i, circuit in enumerate(model_circuits):
        print("circuit number " + str(i) + " :")
        print(circuit.to_text_diagram(transpose=False))

    ops = [cirq.Z(q) for q in qubits]
    observables_l = [ops[x * n_qubits:x * n_qubits + n_qubits] for x in range(l)]
    observables = []

    for i, observable in enumerate(observables_l):
        observables.append(reduce((lambda x, y: x * y), observable))

    # create seperate layers for each quantum "block"
    quantum_layers = []
    for i, circuit in enumerate(model_circuits):
        model1 = tf.keras.Sequential()
        model1.add(tf.keras.Input(shape=(), dtype=tf.string))
        model1.add(tfq.layers.PQC(circuit, observables[i]))
        quantum_layers.append(model1)

    model_concat = keras.layers.concatenate([layer.output for layer in quantum_layers], axis=-1)

    # add layer for simple addition with equal weight and no trainable parameters
    model_concat = tf.keras.layers.Dense(l, activation='relu')(model_concat)
    model_concat = tf.keras.layers.Dense(1, activation='tanh')(model_concat)


    # give all inputs of all l "blocks" to the model
    model = keras.Model(inputs=[layer.input for layer in quantum_layers], outputs=model_concat)

    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[hinge_accuracy])

    print(model.summary())

    qnn_history = model.fit(
        x_train_tfcirc, y_train_hinge,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_hinge))

    qnn_score = model.evaluate(x_test_tfcirc, y_test)

    # print(model.predict(x_test_tfcirc))
    # print(y_test)

    print("accuracy and val_accuracy:")
    print(qnn_history.history['hinge_accuracy'])
    print(qnn_history.history['val_hinge_accuracy'])

    f = open('accuracy_NN_8x8_4qubits_mnist_' + str(datapoints) + 'datapoints.txt', 'a+')
    f.write(str(qnn_history.history['hinge_accuracy']) + "\n")
    f.write(str(qnn_history.history['val_hinge_accuracy']) + "\n")
    f.close()


if __name__ == "__main__":
    main()
