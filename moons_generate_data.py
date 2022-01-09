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


def u_encode_rot(dim, n_qubits, qubits, x, scale, rotation_gates):
    encoded_data = []
    repeats = int(n_qubits/2)
    for i in range(np.size(x, 1)):
        circuit = cirq.Circuit()
        for k in range(np.size(x, 0)):
            if n_qubits == 2:
                circuit.append(rotation_gates[0](x[k, i] * scale[0] * math.pi)(qubits[i * n_qubits + k]))
            else:

                if k >= repeats:
                    circuit.append(rotation_gates[k - repeats](x[k, i] * scale[k - repeats] * math.pi)(qubits[i*n_qubits + k]))
                else:
                    circuit.append(rotation_gates[k](x[k, i] * scale[k] * math.pi)(qubits[i * n_qubits + k]))

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


def create_moons_dataset(data_points, repeats):
    data = make_moons(n_samples=data_points, shuffle=True,
                      noise=0.1, random_state=42)
    x = normalize_data(data[0])
    y = data[1]

    x = np.repeat(x, repeats, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    return 2*repeats, x_train, y_train, x_test, y_test


def create_square_data_points(data_points, repeats):
    x = []
    for i in range(data_points):
        for k in range(data_points):
            x.append(np.repeat([k/data_points, i/data_points], repeats))

    x = normalize_data(x)
    return x


def main():
    # settings [n_qubits, depth, scale, rotation gate]
    # settings = [
    #     [2, 1, 2, [cirq.ry]],
    #     [2, 1, 2, [cirq.rz]],
    #     [2, 1, 2, [cirq.rx]],
    #     [2, 2, 2, [cirq.ry]],
    #     [2, 2, 2, [cirq.rz]],
    #     [2, 2, 2, [cirq.rx]],
    #     [2, 3, 2, [cirq.ry]],
    #     [2, 3, 2, [cirq.rz]],
    #     [2, 3, 2, [cirq.rx]],
    #     [4, 2, 2, [cirq.ry, cirq.ry]],
    #     [4, 2, 2, [cirq.rz, cirq.rz]],
    #     [4, 2, 2, [cirq.rx, cirq.rx]],
    #     [4, 2, 2, [cirq.ry, cirq.rz]],
    #     [4, 2, 2, [cirq.rx, cirq.rz]],
    #     [4, 2, 2, [cirq.ry, cirq.rx]],
    #     ]
    #

    # settings = [
    #     [2, 2, 1, [cirq.ry]],
    #     [2, 2, 4, [cirq.ry]]
    #     ]

    # settings = [
    #     [4, 2, [1, 2], [cirq.ry, cirq.ry]],
    #     [4, 2, [2, 4], [cirq.ry, cirq.ry]],
    #     [4, 2, [1, 4], [cirq.ry, cirq.ry]],
    #     ]

    # settings = [
    #     [4, 2, [1, 2], [cirq.ry, cirq.rz]],
    #     [4, 2, [1, 2], [cirq.ry, cirq.rx]],
    #     [4, 2, [1, 2], [cirq.rz, cirq.rx]],
    #     ]

    # settings = [
    #     [8, 2, [1, 2, 1, 2], [cirq.ry, cirq.ry, cirq.rx, cirq.rx]],
    #     ]

    settings = [
        [2, 2, [2], [cirq.ry]],
        ]

    for run_number in range(len(settings)):
        print("current run: " + str(run_number))
        n_qubits = settings[run_number][0]
        depth = settings[run_number][1]
        scales = settings[run_number][2]
        rotation_gates = settings[run_number][3]

        datapoints = -1
        repeats = int(n_qubits/2)

        gate_names = ""
        for gate in rotation_gates:
            gate_names += str(gate.__name__)

        scale_names = ""
        for scale in scales:
            scale_names += "_" + str(scale)

        print(scales)
        name = 'moons_predictions/' + str(n_qubits) + "qubits_" + str(depth) + "depth_" + gate_names + "_scale" + str(scale_names) + "noisy"

        plot_data = create_square_data_points(100, repeats)
        print(len(plot_data))
        dim, x_train, y_train, x_test, y_test = create_moons_dataset(10000, repeats=repeats)
        y_train = y_train[0:datapoints]
        y_test = y_test[0:datapoints]

        if dim % n_qubits != 0:
            raise ValueError('Error: dim % n_qubits != 0')

        l = int(dim / n_qubits)

        plot_bin = np.array(plot_data, dtype=np.float32).reshape((-1, int(dim/l), l))
        print(len(plot_bin))
        x_train_bin = np.array(x_train, dtype=np.float32).reshape((-1, int(dim/l), l))[0:datapoints]
        x_test_bin = np.array(x_test, dtype=np.float32).reshape((-1, int(dim/l), l))[0:datapoints]


        qubits = [cirq.GridQubit(i, 0) for i in range(dim)]

        plot_circ = [u_encode_rot(dim, n_qubits, qubits, x, scale=scales, rotation_gates=rotation_gates) for x in plot_bin]
        x_train_circ = [u_encode_rot(dim, n_qubits, qubits, x, scale=scales, rotation_gates=rotation_gates) for x in x_train_bin]
        x_test_circ = [u_encode_rot(dim, n_qubits, qubits, x, scale=scales, rotation_gates=rotation_gates) for x in x_test_bin]

        plot_data = []
        train_data = []
        test_data = []
        for i in range(l):
            plot_data.append([x[i] for x in plot_circ])
            train_data.append([x[i] for x in x_train_circ])
            test_data.append([x[i] for x in x_test_circ])

        plot_tfcirc = []
        x_train_tfcirc = []
        x_test_tfcirc = []
        for i in range(l):
            plot_tfcirc.append(tfq.convert_to_tensor(plot_data[i]))
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

        if len(quantum_layers) > 1:
            model_concat = keras.layers.concatenate([layer.output for layer in quantum_layers], axis=-1)
        else:
            model_concat = quantum_layers[0].output

        # add layer for simple addition with equal weight and no trainable parameters
        model_concat = tf.keras.layers.Flatten()(model_concat)
        model_concat = tf.reduce_sum(model_concat, axis=1)

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

        qnn_score = model.evaluate(x_test_tfcirc, y_test_hinge)

        predict = np.array(model.predict(plot_tfcirc))
        np.save(name, predict)

        f = open('moons_predictions/moons_accs.txt', 'a+')
        f.write(name + " : " + str(qnn_history.history['val_hinge_accuracy']) + "\n")
        f.close()

        print(predict)

        print("accuracy and val_accuracy:")
        print(qnn_history.history['hinge_accuracy'])
        print(qnn_history.history['val_hinge_accuracy'])


if __name__ == "__main__":
    main()
