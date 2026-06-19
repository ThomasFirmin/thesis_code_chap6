import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib


def learn_outputs(path, n_class, plot=True):

    files = [
        name
        for name in os.listdir(path)
        if (os.path.isfile(os.path.join(path, name)) and name[:3] != "tar")
    ]
    N_FILES = len(files)

    outputs = torch.load(os.path.join(path, files[0]))
    targets = torch.load(os.path.join(path, "tar_" + files[0]))
    batch_size, outputs_size = outputs.shape

    cell_target = {i: np.zeros(n_class, dtype=int) for i in range(outputs_size)}
    learned_cell = np.full(outputs_size, -1, dtype=int)

    for name in files:
        outputs = torch.load(os.path.join(path, name))
        targets = torch.load(os.path.join(path, "tar_" + name))

        max_idx = torch.argmax(outputs, dim=1)

        for t, idx in enumerate(max_idx):
            if outputs[t][int(idx)] != 0:
                cell_target[int(idx)][int(targets[t])] += 1

    for i in range(outputs_size):
        try:
            learned_cell[i] = np.nanargmax(
                np.where(cell_target[i] != 0, cell_target[i], np.nan)
            )
        except ValueError:
            pass

    if plot:
        map = learned_cell.reshape(
            (int(np.sqrt(outputs_size)), int(np.sqrt(outputs_size)))
        )

        fig, ax = plt.subplots(1)
        sns.heatmap(map, ax=ax, linewidth=0.5, annot=True)
        ax.set_aspect("equal", adjustable="box")

        return learned_cell, fig
    else:
        return learned_cell, None


def predict(path, learned_cell):
    files = [
        name
        for name in os.listdir(path)
        if (os.path.isfile(os.path.join(path, name)) and name[:3] != "tar")
    ]
    N_FILES = len(files)

    predictions = np.array([], dtype=int)
    success = 0

    for name in files:
        outputs = torch.load(os.path.join(path, name))
        targets = torch.load(os.path.join(path, "tar_" + name)).numpy()

        max_idx = torch.argmax(outputs, dim=1)
        max_idx = max_idx.tolist()
        preds = learned_cell[max_idx]

        for t, idx in enumerate(max_idx):
            if outputs[t][int(idx)] == 0:
                preds[t] = -1

        success += np.count_nonzero(preds == targets)

        predictions = np.append(predictions, preds[:])

    accuracy = success / len(predictions)
    no_spikes = np.count_nonzero(predictions == -1)

    return predictions, accuracy, success, no_spikes


def plot_weights(path, weights):

    outputs_size, inputs_size = weights.shape
    map_size = int(np.ceil(np.sqrt(outputs_size)))
    total = map_size ** 2
    fig, axes = plt.subplots(
        map_size,
        map_size,
        constrained_layout=True,
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )

    for i, w in enumerate(weights):
        print(f"{i}/{len(weights)}")
        ax = axes[i // map_size, i % map_size]
        map = w.reshape((28, 28))
        ax.axis("off")
        ax.set(adjustable="box", aspect="equal")
        my_cmap = matplotlib.cm.get_cmap("copper")
        my_cmap.set_under("white")
        ax.imshow(
            map,
            interpolation="none",
            cmap=my_cmap,
            vmin=0.0000001,
            vmax=1.0,
            aspect="auto",
        )

    for i in range(len(weights), total):
        ax = axes[i // map_size, i % map_size]
        ax.axis("off")
        ax.set(adjustable="box", aspect="equal")

    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0.0, wspace=0.0)

    return fig


def plot_map(path, learned_cell):
    outputs_size = len(learned_cell)

    map = learned_cell.reshape(
        (
            int(np.ceil(np.sqrt(outputs_size))),
            int(np.ceil(np.sqrt(outputs_size))),
        )
    )

    fig, ax = plt.subplots(1)
    sns.heatmap(map, ax=ax, linewidth=0.5, annot=True)
    ax.set_aspect("equal", adjustable="box")

    return fig
