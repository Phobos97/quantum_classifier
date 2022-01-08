import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    f = open(path, 'r')
    data = f.readlines()
    f.close()

    data = [line.replace("[", "").replace("]", "") for line in data]
    data = np.array([line.split(",") for line in data], dtype=float)

    epochs = len(data[0])
    runs = int(len(data)/2)

    averaged_test_acc = np.zeros(data[0].shape)
    averaged_train_acc = np.zeros(data[0].shape)
    for i in range(runs):
        averaged_test_acc += data[i * 2]
        averaged_train_acc += data[i * 2 + 1]

    averaged_test_acc /= runs
    averaged_train_acc /= runs

    return averaged_test_acc, averaged_train_acc, epochs, runs


path_simple_summation = 'accuracy_simple_summation_8x8_4qubits_mnist_1000datapoints.txt'
path_trainable_summation = 'accuracy_trainable_summation_8x8_4qubits_mnist_1000datapoints.txt'
path_nn = 'accuracy_NN_8x8_4qubits_mnist_1000datapoints.txt'

averaged_test_acc1, averaged_train_acc1, epochs1, runs1 = load_data(path_simple_summation)
averaged_test_acc2, averaged_train_acc2, epochs2, runs2 = load_data(path_trainable_summation)
averaged_test_acc3, averaged_train_acc3, epochs3, runs3 = load_data(path_nn)

# x = range(epochs)
# plt.plot(x, averaged_train_acc)
# plt.plot(x, averaged_test_acc)
# plt.legend(["Train Accuracy", "Test Accuracy"])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy (%)")
# plt.show()

plt.plot(range(epochs1), averaged_test_acc1)
plt.plot(range(epochs2), averaged_test_acc2)
plt.plot(range(epochs3), averaged_test_acc3)
plt.legend(["simple_summation", "trainable_summation", "nn"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.show()

print(averaged_test_acc1[-1])
print(averaged_test_acc2[-1])
print(averaged_test_acc3[-1])