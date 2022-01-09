import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def create_moons_dataset(data_points):
    data = make_moons(n_samples=data_points, shuffle=False,
                      noise=0.0, random_state=42)
    x = normalize_data(data[0])*99
    y = data[1]

    lower_half = []
    top_half = []

    for i, point in enumerate(x):
        if y[i]:
            top_half.append(point)
        else:
            lower_half.append(point)

    top_half = np.array(top_half)
    lower_half = np.array(lower_half)

    return lower_half, top_half


def load_prediction(name):
    data = np.load('moons_predictions/' + name + ".npy")
    data1 = np.array([x > 0 for x in data])
    data1 = np.clip(data1, 0.5, 1)
    data2 = data * data1
    return data2.reshape((100, 100))


plot1 = False
plot2 = False
plot3 = False
plot4 = True

# lower_half, top_half = create_moons_dataset(100)
# data = load_prediction('2qubits_2depth_ry_scale_2noisy')
# plt.imshow(data, cmap="RdYlBu")
# plt.plot(lower_half[:, 0], lower_half[:, 1], "r.")
# plt.plot(top_half[:, 0], top_half[:, 1], "b.")
# plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
# plt.show()


if plot1:
    names_to_plot = [['2qubits_1depth_rx_scale2', '2qubits_1depth_ry_scale2', '2qubits_1depth_rz_scale2'],
                     ['2qubits_2depth_rx_scale2', '2qubits_2depth_ry_scale2', '2qubits_2depth_rz_scale2'],
                     ['2qubits_3depth_rx_scale2', '2qubits_3depth_ry_scale2', '2qubits_3depth_rz_scale2']]

    accuracies = [[0.58513623, 0.58513623, 0.4954928],
                     [0.86478364, 0.8671875, 0.4954928],
                     [0.86538464, 0.86688703, 0.4954928]]

    lower_half, top_half = create_moons_dataset(100)

    x_plots = 3
    y_plots = 3
    fig, axs = plt.subplots(x_plots, y_plots)

    for i in range(x_plots):
        for k in range(y_plots):
            data = load_prediction(names_to_plot[i][k])
            axs[i, k].imshow(data, cmap="RdYlBu")
            axs[i, k].plot(lower_half[:, 0], lower_half[:, 1], "r")
            axs[i, k].plot(top_half[:, 0], top_half[:, 1], "b")
            axs[i, k].set_xlabel("x")
            axs[i, k].set_ylabel("y")
            axs[i, k].set_title("acc: " + str(round(accuracies[i][k], 2)))
            axs[i, k].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            axs[i, k].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, labelright=False)


    fig.text(0.25, 0.05, 'gate: rx', va='center', rotation='horizontal', size=15)
    fig.text(0.5, 0.05, 'gate: ry', va='center', rotation='horizontal', size=15)
    fig.text(0.75, 0.05, 'gate: rz', va='center', rotation='horizontal', size=15)

    fig.text(0.05, 0.75, 'depth: 1', va='center', rotation='horizontal', size=15)
    fig.text(0.05, 0.5, 'depth: 2', va='center', rotation='horizontal', size=15)
    fig.text(0.05, 0.25, 'depth: 3', va='center', rotation='horizontal', size=15)
    plt.show()

if plot2:
    names_to_plot = [['4qubits_2depth_ryry_scale2', '4qubits_2depth_rzrz_scale2', '4qubits_2depth_rxrx_scale2'],
                     ['4qubits_2depth_ryrz_scale2', '4qubits_2depth_rxrz_scale2', '4qubits_2depth_ryrx_scale2']]

    accuracies = [[0.7090345, 0.4954928, 0.7406851],
                  [0.8659856, 0.8635817, 0.73938304]]

    lower_half, top_half = create_moons_dataset(100)

    x_plots = 2
    y_plots = 3
    fig, axs = plt.subplots(x_plots, y_plots)

    for i in range(x_plots):
        for k in range(y_plots):
            data = load_prediction(names_to_plot[i][k])
            axs[i, k].imshow(data, cmap="RdYlBu")
            axs[i, k].plot(lower_half[:, 0], lower_half[:, 1], "r")
            axs[i, k].plot(top_half[:, 0], top_half[:, 1], "b")
            axs[i, k].set_xlabel("x")
            axs[i, k].set_ylabel("y")
            axs[i, k].set_title("acc: " + str(round(accuracies[i][k], 2)))
            axs[i, k].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            axs[i, k].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, labelright=False)

    fig.text(0.25, 0.05, 'gate: rx', va='center', rotation='horizontal', size=15)
    fig.text(0.5, 0.05, 'gate: ry', va='center', rotation='horizontal', size=15)
    fig.text(0.75, 0.05, 'gate: rz', va='center', rotation='horizontal', size=15)


    # fig.text(0.25, 0.05, 'gate: rx', va='center', rotation='horizontal', size=15)
    # fig.text(0.5, 0.05, 'gate: ry', va='center', rotation='horizontal', size=15)
    # fig.text(0.75, 0.05, 'gate: rz', va='center', rotation='horizontal', size=15)
    #
    # fig.text(0.05, 0.75, 'depth: 1', va='center', rotation='horizontal', size=15)
    # fig.text(0.05, 0.5, 'depth: 2', va='center', rotation='horizontal', size=15)
    # fig.text(0.05, 0.25, 'depth: 3', va='center', rotation='horizontal', size=15)
    plt.show()


if plot3:
    data = load_prediction('4qubits_2depth_ryry_scale_1_2')
    names_to_plot = ['4qubits_2depth_ryry_scale_1_2', '4qubits_2depth_ryry_scale_2_4', '4qubits_2depth_ryry_scale_1_4']

    accuracies = [0.9359976, 0.8347356, 0.9353966]

    lower_half, top_half = create_moons_dataset(100)

    x_plots = 1
    y_plots = 3
    fig, axs = plt.subplots(x_plots, y_plots)

    for k in range(y_plots):
        data = load_prediction(names_to_plot[k])
        axs[k].imshow(data, cmap="RdYlBu")
        axs[k].plot(lower_half[:, 0], lower_half[:, 1], "r")
        axs[k].plot(top_half[:, 0], top_half[:, 1], "b")
        axs[k].set_xlabel("x")
        axs[k].set_ylabel("y")
        axs[k].set_title("acc: " + str(round(accuracies[k], 2)))
        axs[k].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
        axs[k].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, labelright=False)

    fig.text(0.25, 0.05, 'scale: 1, 2', va='center', rotation='horizontal', size=15)
    fig.text(0.5, 0.05, 'scale: 2, 4', va='center', rotation='horizontal', size=15)
    fig.text(0.75, 0.05, 'scale: 1, 4', va='center', rotation='horizontal', size=15)

    plt.show()


if plot4:
    data = load_prediction('4qubits_2depth_ryry_scale_1_2')
    names_to_plot = ['4qubits_2depth_ryrz_scale_1_2', '4qubits_2depth_ryrx_scale_1_2', '4qubits_2depth_rzrx_scale_1_2', '8qubits_2depth_ryryrxrx_scale_1_2_1_2']

    accuracies = [0.82902646, 0.96063703, 0.86658657, 1]

    lower_half, top_half = create_moons_dataset(100)

    x_plots = 1
    y_plots = 4
    fig, axs = plt.subplots(x_plots, y_plots)

    for k in range(y_plots):
        data = load_prediction(names_to_plot[k])
        axs[k].imshow(data, cmap="RdYlBu")
        axs[k].plot(lower_half[:, 0], lower_half[:, 1], "r")
        axs[k].plot(top_half[:, 0], top_half[:, 1], "b")
        axs[k].set_xlabel("x")
        axs[k].set_ylabel("y")
        axs[k].set_title("acc: " + str(round(accuracies[k], 2)))
        axs[k].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
        axs[k].tick_params(axis='y', which='both', bottom=False, top=False, labelleft=False, labelright=False)

    fig.text(0.25, 0.05, 'scale: 1, 2, 1, 2', va='center', rotation='horizontal', size=15)
    fig.text(0.5, 0.05, 'gate: ryryrxrx', va='center', rotation='horizontal', size=15)

    plt.show()
