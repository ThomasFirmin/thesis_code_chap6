# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-27T14:02:40+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-11T12:24:48+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import os
from time import time as t
import numpy as np
import torch

from typing import Iterable, List, Optional, Sequence, Tuple, Union
from scipy.spatial.distance import euclidean

import bindsnet
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection
import collections.abc as container_abcs

from Lie.objective import Objective


class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture
    from `(Hazan et al. 2018) <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_inpt: int,
        n_classes=10,
        n_neurons: int = 100,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # Inhib
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        # STDP
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
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
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

        self.n_classes = n_classes
        self.n_input = n_inpt
        self.n_neurons = n_neurons
        self.n_out = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt
        self.inpt_shape = inpt_shape

        input_layer = Input(
            n=self.n_input,
            shape=self.inpt_shape,
            traces=True,
            tc_trace=e_tc_trace,
        )
        self.add_layer(input_layer, name="inpt")

        output_layer = DiehlAndCookNodes(
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
        self.add_layer(output_layer, name="outpt")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["inpt"],
            target=self.layers["outpt"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            weight_decay=weight_decay,
        )
        self.add_connection(input_output_conn, source="inpt", target="outpt")

        # add internal inhibitory connections
        w = torch.ones(self.n_neurons, self.n_neurons) - torch.diag(
            torch.ones(self.n_neurons)
        )
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    w[i, j] = np.sqrt(euclidean([x1, y1], [x2, y2]))
        w = w / w.max()
        w = (w * self.max_inhib) + self.start_inhib
        recurrent_output_conn = Connection(
            source=self.layers["outpt"], target=self.layers["outpt"], w=w
        )
        self.add_connection(
            recurrent_output_conn, source="outpt", target="outpt"
        )

        self.assignments = -torch.ones(self.n_out)
        self.proportions = torch.zeros((self.n_out, self.n_classes))
        self.rates = torch.zeros((self.n_out, self.n_classes))

    def to(self, device=None, *args, **kwargs):
        self.assignments = self.assignments.to(device=device)
        self.proportions = self.proportions.to(device=device)
        self.rates = self.rates.to(device=device)
        return super().to(device=device, *args, **kwargs)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        super().save(os.path.join(filepath, "network.pt"))
        torch.save(
            self.assignments,
            open(os.path.join(filepath, "assignments.pt"), "wb"),
        )
        torch.save(
            self.proportions,
            open(os.path.join(filepath, "proportion.pt"), "wb"),
        )
        torch.save(self.rates, open(os.path.join(filepath, "rates.pt"), "wb"))


class Model(Objective):
    def __call__(self, *args, load=False, train=True, **kwargs):
        super().__call__(*args, **kwargs)

        # Map size
        n_neurons = kwargs.get("map_size", 100)
        # STDP
        nu_pre = kwargs.get("nu_pre", 1e-4)
        nu_post = kwargs.get("nu_post", 1e-2)
        # Excit
        theta_plus = kwargs.get("theta_plus", 0.05)
        tc_theta_decay = kwargs.get("tc_theta_decay", 1e7)
        e_rest = kwargs.get("e_rest", -65.0)
        e_reset = kwargs.get("e_reset", -60.0)
        e_thresh = kwargs.get("e_thresh", -52.0)
        e_refrac = kwargs.get("e_refrac", 5)
        e_tc_decay = kwargs.get("e_tc_decay", 100.0)
        e_tc_trace = kwargs.get("e_tc_trace", 20.0)

        if load:
            network = bindsnet.network.load(os.path.join(load, "network.pt"))
            network.assignments = torch.load(
                os.path.join(load, "assignments.pt")
            )
            network.proportions = torch.load(
                os.path.join(load, "proportions.pt")
            )
            network.rates = torch.load(os.path.join(load, "rates.pt"))
        else:
            network = IncreasingInhibitionNetwork(
                n_inpt=self.input_features,
                inpt_shape=self.input_shape,
                n_neurons=n_neurons,
                dt=self.dt,
                # STDP
                theta_plus=theta_plus,
                tc_theta_decay=tc_theta_decay,
                nu=(nu_pre, nu_post),
                norm=78.4,
                # Excit
                e_rest=e_rest,
                e_reset=e_reset,
                e_thresh=e_thresh,
                e_refrac=e_refrac,
                e_tc_decay=e_tc_decay,
                e_tc_trace=e_tc_trace,
            )

        # Directs network to GPU
        if self.gpu:
            network.to(self.device)

        start = t()
        if train:
            self.train(network, decoder=self.decoder)

        train_time = t() - start

        start = t()
        self.test(network, decoder=self.decoder)
        test_time = t() - start
        total_time = train_time + test_time

        results = self.accuracy()
        results["train_time"] = train_time
        results["test_time"] = test_time
        results["total_time"] = total_time
        return results, network
