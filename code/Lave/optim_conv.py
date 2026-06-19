# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-20T15:12:59+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

# import slayer from lava-dl
from geleeepicee.abstract_network import AbstractNetwork
import numpy as np


class ConvMNIST(AbstractNetwork):
    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        threshold=1.25,
        threshold_decay=0,
        dropout=0.05,
        c1_filters=5,
        c1_k=12,
        c1_p=0,
        c1_d=1,
        c1_s=1,
        c2_filters=64,
        c2_k=5,
        c2_p=0,
        c2_d=1,
        c2_s=1,
        a1_k=2,
        a1_p=0,
        a1_s=1,
        a2_k=2,
        a2_p=0,
        a2_s=1,
        alpha=4,
    ):
        super(ConvMNIST, self).__init__(n_inpt, n_classes, inpt_shape)

        c1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1
        )

        a1o_shape = np.floor((c1o_shape + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1)
        c2o_shape = np.floor((a1o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a2o_shape = np.floor((c2o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)

        print(f"SHAPE: ,{c1o_shape}, {a1o_shape}, {c2o_shape}, {a2o_shape}")

        c1 = layer.Conv2d(
            in_channels=int(self.inpt_shape[0]),
            out_channels=int(c1_filters),
            kernel_size=c1_k,
            stride=c1_s,
            padding=c1_p,
            dilation=c1_d,
            bias=False,
            step_mode="m",
        )

        norm1 = layer.BatchNorm2d(int(c1_filters), step_mode="m")

        spike1 = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        a1 = layer.MaxPool2d(
            kernel_size=a1_k,
            stride=a1_s,
            padding=a1_p,
            step_mode="m",
        )

        c2 = layer.Conv2d(
            in_channels=int(c1_filters),
            out_channels=int(c2_filters),
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            dilation=c2_d,
            bias=False,
            step_mode="m",
        )

        norm2 = layer.BatchNorm2d(int(c2_filters), step_mode="m")

        spike2 = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        a2 = layer.MaxPool2d(
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
            step_mode="m",
        )

        flat1 = layer.Flatten(step_mode="m")
        drop1 = layer.Dropout(dropout, step_mode="m")
        layero = layer.Linear(
            in_features=int(a2o_shape**2 * c2_filters),
            out_features=int(10 * self.n_classes),
            step_mode="m",
        )
        spikeo = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )
        self.vote = layer.VotingLayer(step_mode="m")

        self.seq = nn.Sequential(
            c1, norm1, spike1, a1, c2, norm2, spike2, a2, flat1, drop1, layero, spikeo
        )

    def forward(self, spikes):
        self._update_computed_images(spikes)
        self._update_recorders(spikes, "inpt")
        out = self.seq(spikes)
        self._update_recorders(out, "outpt")
        return self.vote(out), out


class ConvDVS(AbstractNetwork):
    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        threshold=1.25,
        threshold_decay=0,
        dropout=0.05,
        c1_filters=16,
        c1_k=5,
        c1_p=2,
        c1_d=1,
        c1_s=1,
        c2_filters=32,
        c2_k=3,
        c2_p=1,
        c2_d=1,
        c2_s=1,
        a1_k=4,
        a1_p=0,
        a1_s=1,
        a2_k=2,
        a2_p=0,
        a2_s=1,
        a3_k=2,
        a3_p=0,
        a3_s=1,
        d1_n=512,
        alpha=4,
    ):
        super(ConvDVS, self).__init__(n_inpt, n_classes, inpt_shape)

        a1o_shape = np.floor(
            (self.inpt_shape[1] + 2 * a1_p - (a1_k - 1) - 1) / a1_s + 1
        )
        c1o_shape = np.floor((a1o_shape + 2 * c1_p - c1_d * (c1_k - 1) - 1) / c1_s + 1)
        a2o_shape = np.floor((c1o_shape + 2 * a2_p - (a2_k - 1) - 1) / a2_s + 1)
        c2o_shape = np.floor((a2o_shape + 2 * c2_p - c2_d * (c2_k - 1) - 1) / c2_s + 1)
        a3o_shape = np.floor((c2o_shape + 2 * a3_p - (a3_k - 1) - 1) / a3_s + 1)

        a1 = layer.MaxPool2d(
            kernel_size=a1_k,
            stride=a1_s,
            padding=a1_p,
            step_mode="m",
        )

        c1 = layer.Conv2d(
            in_channels=int(self.inpt_shape[0]),
            out_channels=int(c1_filters),
            kernel_size=c1_k,
            stride=c1_s,
            padding=c1_p,
            dilation=c1_d,
            bias=False,
            step_mode="m",
        )

        norm1 = layer.BatchNorm2d(int(c1_filters), step_mode="m")

        spike1 = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        a2 = layer.MaxPool2d(
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
            step_mode="m",
        )

        c2 = layer.Conv2d(
            in_channels=int(c1_filters),
            out_channels=int(c2_filters),
            kernel_size=c2_k,
            stride=c2_s,
            padding=c2_p,
            dilation=c2_d,
            bias=False,
            step_mode="m",
        )

        norm2 = layer.BatchNorm2d(int(c2_filters), step_mode="m")

        spike2 = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        a3 = layer.MaxPool2d(
            kernel_size=a2_k,
            stride=a2_s,
            padding=a2_p,
            step_mode="m",
        )

        flat1 = layer.Flatten(step_mode="m")
        drop1 = layer.Dropout(dropout, step_mode="m")
        linear1 = layer.Linear(
            in_features=int(a3o_shape**2 * c2_filters),
            out_features=d1_n,
            step_mode="m",
        )
        spike3 = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        drop2 = layer.Dropout(dropout, step_mode="m")
        layero = layer.Linear(
            in_features=d1_n,
            out_features=int(10 * self.n_classes),
            step_mode="m",
        )
        spikeo = neuron.ParametricLIFNode(
            init_tau=threshold_decay,
            v_threshold=threshold,
            step_mode="m",
            surrogate_function=surrogate.Sigmoid(alpha=alpha),
        )

        self.vote = layer.VotingLayer(step_mode="m")

        self.seq = nn.Sequential(
            a1,
            c1,
            norm1,
            spike1,
            a2,
            c2,
            norm2,
            spike2,
            a3,
            flat1,
            drop1,
            linear1,
            spike3,
            drop2,
            layero,
            spikeo,
        )

    def forward(self, spikes):
        self._update_computed_images(spikes)
        self._update_recorders(spikes, "inpt")
        out = self.seq(spikes)
        self._update_recorders(out, "outpt")
        return self.vote(out), out
