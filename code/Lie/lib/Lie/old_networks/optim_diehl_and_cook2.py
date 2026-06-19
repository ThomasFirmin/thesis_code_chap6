import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import (
    all_activity,
    assign_labels,
    proportion_weighting,
)
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights


from typing import Iterable, List, Optional, Sequence, Tuple, Union
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # STDP
        exc: float = 22.5,
        inh: float = 17.5,
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        # Excit
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        e_rest=-65.0,
        e_reset=-60.0,
        e_thresh=-52.0,
        e_refrac=5,
        e_tc_decay=100.0,
        e_tc_trace=20.0,
        # Inhib
        i_rest=-60.0,
        i_reset=-45.0,
        i_thresh=-40.0,
        i_tc_decay=10.0,
        i_refrac=2,
        i_tc_trace=20.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.
        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_decay=e_tc_decay,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=i_rest,
            reset=i_reset,
            thresh=i_thresh,
            tc_decay=i_tc_decay,
            refrac=i_refrac,
            tc_trace=i_tc_trace,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            weight_decay=weight_decay,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")

        self.assignments = -torch.ones(self.n_neurons)

    def to(self, device=None, *args, **kwargs):
        self.assignments = self.assignments.to(device=device)
        return super().to(device=device, *args, **kwargs)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        super().save(os.path.join(filepath, "network.pt"))
        torch.save(self, open(os.path.join(filepath, "assignments.pt"), "wb"))


class Objective(object):
    def __init__(
        self,
        split=0.80,
        dt=1,
        input_features=784,
        input_shape=(1, 28, 28),
        progress_interval=10,
        intensity=128,
        gpu=True,
    ):
        self.intensity = intensity
        self.progress_interval = progress_interval

        # Sets up Gpu use
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if gpu:
            self.gpu = False

        self.dt = dt
        self.input_features = input_features
        self.input_shape = input_shape

        # Load MNIST data.
        large_dataset = MNIST(
            None,
            None,
            root=os.path.join("data", "MNIST"),
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * self.intensity),
                ]
            ),
        )
        size = 1000
        (
            self.train_dataset,
            self.valid_dataset,
            _,
        ) = torch.utils.data.random_split(
            large_dataset,
            (
                int(split * size),
                size - int(split * size),
                len(large_dataset) - size,
            ),
        )
        self.n_train = len(self.train_dataset)
        self.n_valid = len(self.valid_dataset)

        # Neuron assignments and spike proportions.
        self.n_classes = 10

    def __call__(self, **kwargs):
        n_epochs = kwargs.get("epochs", 1)
        # Map size
        n_neurons = kwargs.get("map_size", 100)
        # Encoder
        time = kwargs.get("encoding_window", 250)
        # STDP
        exc = kwargs.get("strength_exc", 22.5)
        inh = kwargs.get("strength_inh", 17.5)
        theta_plus = kwargs.get("theta_plus", 0.05)
        nu_pre = kwargs.get("nu_pre", 1e-4)
        nu_post = kwargs.get("nu_post", 1e-2)
        # Excit
        e_rest = kwargs.get("e_rest", -65.0)
        e_reset = kwargs.get("e_reset", -60.0)
        e_thresh = kwargs.get("e_thresh", -52.0)
        e_refrac = kwargs.get("e_refrac", 5)
        e_tc_decay = kwargs.get("e_tc_decay", 100.0)
        e_tc_trace = kwargs.get("e_tc_trace", 20.0)
        # Inhib
        i_rest = kwargs.get("i_rest", -60.0)
        i_reset = kwargs.get("i_reset", -45.0)
        i_thresh = kwargs.get("i_thresh", -40.0)
        i_tc_decay = kwargs.get("i_tc_decay", 10.0)
        i_refrac = kwargs.get("i_refrac", 2)
        i_tc_trace = kwargs.get("i_tc_trace", 20.0)

        network = DiehlAndCook2015(
            n_inpt=self.input_features,
            inpt_shape=self.input_shape,
            n_neurons=n_neurons,
            dt=self.dt,
            # STDP
            exc=exc,
            inh=inh,
            theta_plus=theta_plus,
            nu=(nu_pre, nu_post),
            norm=78.4,
            # Excit
            e_rest=e_rest,
            e_reset=e_reset,
            e_thresh=e_thresh,
            e_refrac=e_refrac,
            e_tc_decay=e_tc_decay,
            e_tc_trace=e_tc_trace,
            # Inhib
            i_rest=i_rest,
            i_reset=i_reset,
            i_thresh=i_thresh,
            i_tc_decay=i_tc_decay,
            i_refrac=i_refrac,
            i_tc_trace=i_tc_trace,
        )

        update_interval = 30

        # Directs network to GPU
        if self.gpu:
            network.to("cuda")

        self.train_dataset.dataset.image_encoder = PoissonEncoder(
            time=time, dt=self.dt
        )
        self.valid_dataset.dataset.image_encoder = PoissonEncoder(
            time=time, dt=self.dt
        )

        # Record spikes during the simulation.
        spike_record = torch.zeros(
            (update_interval, int(time / self.dt), n_neurons),
            device=self.device,
        )

        proportions = torch.zeros(
            (n_neurons, self.n_classes), device=self.device
        )
        rates = torch.zeros((n_neurons, self.n_classes), device=self.device)

        # Sequence of accuracy estimates.
        accuracy_train = {"all": [], "proportion": []}

        ############## MEMORY COST  ##############
        # Voltage recording for excitatory and inhibitory layers.
        # exc_voltage_monitor = Monitor(
        #     network.layers["Ae"], ["v"], time=int(time / dt), device=device
        # )
        # inh_voltage_monitor = Monitor(
        #     network.layers["Ai"], ["v"], time=int(time / dt), device=device
        # )
        # network.add_monitor(exc_voltage_monitor, name="exc_voltage")
        # network.add_monitor(inh_voltage_monitor, name="inh_voltage")
        ############## MEMORY COST  ##############

        # Set up monitors for spikes and voltages
        spikes = {}
        layer = "Ae"
        spikes[layer] = Monitor(
            network.layers[layer],
            state_vars=["s"],
            time=int(time / self.dt),
            device=self.device,
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        ############## MEMORY COST  ##############
        # voltages = {}
        # for layer in set(network.layers) - {"X"}:
        #     voltages[layer] = Monitor(
        #         network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        #     )
        #     network.add_monitor(voltages[layer], name="%s_voltages" % layer)
        ############## MEMORY COST  ##############

        # Train the network.
        print("\nBegin training.\n")
        start = t()
        for epoch in range(n_epochs):
            labels = []

            if epoch % self.progress_interval == 0:
                print(
                    "Progress: %d / %d (%.4f seconds)"
                    % (epoch, n_epochs, t() - start)
                )
                start = t()

            # Create a dataloader to iterate and batch data
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                pin_memory=self.gpu,
            )

            for step, batch in enumerate(tqdm(dataloader)):
                if step > self.n_train:
                    break
                # Get next input sample.
                inputs = {
                    "X": batch["encoded_image"].view(
                        int(time / self.dt), 1, 1, 28, 28
                    )
                }

                if self.gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                if step % update_interval == 0 and step > 0:
                    # Convert the array of labels into a tensor
                    label_tensor = torch.tensor(labels, device=self.device)
                    print("ICIIIIIIIIII")
                    print(len(label_tensor))
                    print(network.assignments)
                    # Get network predictions.
                    all_activity_pred = all_activity(
                        spikes=spike_record,
                        assignments=network.assignments,
                        n_labels=self.n_classes,
                    )
                    proportion_pred = proportion_weighting(
                        spikes=spike_record,
                        assignments=network.assignments,
                        proportions=proportions,
                        n_labels=self.n_classes,
                    )

                    # Compute network accuracy according to available classification strategies.
                    accuracy_train["all"].append(
                        torch.sum(
                            label_tensor.long() == all_activity_pred
                        ).item()
                        / len(label_tensor)
                    )
                    accuracy_train["proportion"].append(
                        torch.sum(label_tensor.long() == proportion_pred).item()
                        / len(label_tensor)
                    )

                    print(
                        "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                        % (
                            accuracy_train["all"][-1],
                            np.mean(accuracy_train["all"]),
                            np.max(accuracy_train["all"]),
                        )
                    )
                    print(
                        "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                        " (best)\n"
                        % (
                            accuracy_train["proportion"][-1],
                            np.mean(accuracy_train["proportion"]),
                            np.max(accuracy_train["proportion"]),
                        )
                    )

                    # Assign labels to excitatory layer neurons.
                    network.assignments, proportions, rates = assign_labels(
                        spikes=spike_record,
                        labels=label_tensor,
                        n_labels=self.n_classes,
                        rates=rates,
                    )

                    labels = []

                labels.append(batch["label"])

                # Run the network on the input.
                network.run(inputs=inputs, time=time, input_time_dim=1)
                # # Get voltage recording.
                # exc_voltages = exc_voltage_monitor.get("v")
                # inh_voltages = inh_voltage_monitor.get("v")

                # Add to spikes recording.
                spike_record[step % update_interval] = (
                    spikes["Ae"].get("s").squeeze()
                )

                network.reset_state_variables()  # Reset state variables.

        print(
            "Progress: %d / %d (%.4f seconds)"
            % (epoch + 1, n_epochs, t() - start)
        )
        print("Training complete.\n")

        # Sequence of accuracy estimates.
        accuracy_test = {"all": 0, "proportion": 0}

        # Record spikes during the simulation.
        spike_record = torch.zeros(
            (1, int(time / self.dt), n_neurons), device=self.device
        )

        # Train the network.
        print("\nBegin testing\n")
        network.train(mode=False)
        start = t()

        pbar = tqdm(total=self.n_valid)
        for step, batch in enumerate(self.valid_dataset):
            if step >= self.n_valid:
                break
            # Get next input sample.
            inputs = {
                "X": batch["encoded_image"].view(
                    int(time / self.dt), 1, 1, 28, 28
                )
            }
            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1)

            # Add to spikes recording.
            spike_record[0] = spikes["Ae"].get("s").squeeze()

            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(batch["label"], device=self.device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=network.assignments,
                n_labels=self.n_classes,
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=network.assignments,
                proportions=proportions,
                n_labels=self.n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy_test["all"] += float(
                torch.sum(label_tensor.long() == all_activity_pred).item()
            )
            accuracy_test["proportion"] += float(
                torch.sum(label_tensor.long() == proportion_pred).item()
            )

            network.reset_state_variables()  # Reset state variables.
            pbar.set_description_str("Test progress: ")
            pbar.update()

        print(
            "\nAll activity accuracy: %.2f"
            % (accuracy_test["all"] / self.n_valid)
        )
        print(
            "Proportion weighting accuracy: %.2f \n"
            % (accuracy_test["proportion"] / self.n_valid)
        )

        print(
            "Progress: %d / %d (%.4f seconds)"
            % (epoch + 1, n_epochs, t() - start)
        )
        print("Testing complete.\n")

        return {
            "train": np.mean(accuracy_train["all"]),
            "valid": accuracy_test["all"] / self.n_valid,
        }, network
