# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-27T16:04:01+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-19T17:09:52+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import os
from time import time as t
import numpy as np
import torch

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import bindsnet
from bindsnet.learning import PostPre
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection
import collections.abc as container_abcs

from Lie.abstract_network import AbstractNetwork
from Lie.objective import Objective


class Conv2D(AbstractNetwork):
    def __init__(
        self,
        n_inpt: int,
        n_classes=10,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        n_filters=25,
        kernel_size=16,
        padding=0,
        stride=1,
        dilatation=1,
        # STDP
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
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
        super().__init__(dt=dt)
        self.n_classes = n_classes
        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        # CONV
        self.n_filters = n_filters  # 25
        self.kernel_size = kernel_size  # 16
        self.padding = padding  # 0
        self.stride = stride  # 4
        self.dilatation = dilatation  # 1

        self.conv_size = (
            int((28 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        )
        assert (
            self.conv_size > 1
        ), f"ERROR on self.conv_size got {self.conv_size}"

        self.n_out = self.n_filters * self.conv_size**2

        # Build network.
        input_layer = Input(n=self.n_inpt, shape=self.inpt_shape, traces=True)

        conv_layer = DiehlAndCookNodes(
            n=self.n_out,
            shape=(self.n_filters, self.conv_size, self.conv_size),
            traces=True,
            rest=e_rest,  # -65
            reset=e_reset,  # -65
            thresh=e_thresh,  # -52
            refrac=e_refrac,  # 5
            tc_decay=e_tc_decay,  # 100
            tc_trace=e_tc_trace,  # 20
            theta_plus=theta_plus,  # 0.05
            tc_theta_decay=tc_theta_decay,  # 1e7
        )

        conv_conn = Conv2dConnection(
            input_layer,
            conv_layer,
            kernel_size=self.kernel_size,
            stride=self.stride,
            update_rule=PostPre,
            norm=0.4 * kernel_size**2,
            nu=nu,  # [1e-4, 1e-2]
            reduction=reduction,  # None
            wmin=wmin,
            wmax=wmax,  # 1
            weight_decay=weight_decay,  # 0.0
        )

        w = torch.zeros(
            self.n_filters,
            self.conv_size,
            self.conv_size,
            self.n_filters,
            self.conv_size,
            self.conv_size,
        )
        for fltr1 in range(self.n_filters):
            for fltr2 in range(self.n_filters):
                if fltr1 != fltr2:
                    for i in range(self.conv_size):
                        for j in range(self.conv_size):
                            w[fltr1, i, j, fltr2, i, j] = -100.0

        w = w.view(
            self.n_out,
            self.n_out,
        )
        recurrent_conn = Connection(conv_layer, conv_layer, w=w)

        self.add_layer(input_layer, name="inpt")
        self.add_layer(conv_layer, name="outpt")
        self.add_connection(conv_conn, source="intpt", target="outpt")
        self.add_connection(recurrent_conn, source="outpt", target="outpt")

        self.assignments = -torch.ones(self.n_out)
        self.proportions = torch.zeros((self.n_out, self.n_classes))
        self.rates = torch.zeros((self.n_out, self.n_classes))
        self.ngram_scores = {}

    def to(self, device=None, *args, **kwargs):
        self.assignments = self.assignments.to(device=device)
        self.proportions = self.proportions.to(device=device)
        self.rates = self.rates.to(device=device)
        self.ngram_scores = {
            k: v.to(device=device, non_blocking=True)
            for k, v in self.ngram_scores.items()
        }
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
        torch.save(
            self.ngram_scores, open(os.path.join(filepath, "ngram.pt"), "wb")
        )


class Model(Objective):
    def __call__(self, *args, load=False, train=True, **kwargs):
        super().__call__(**kwargs)
        # Convolution
        n_filters = kwargs.get("filters", 25)
        kernel_size = kwargs.get("kernel_size", 16)
        padding = kwargs.get("padding", 0)
        stride = kwargs.get("stride", 4)
        dilatation = kwargs.get("dilatation", 1)
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
            network = Conv2D(
                n_inpt=self.input_features,
                inpt_shape=self.input_shape,
                dt=self.dt,
                # Convolution
                n_filters=n_filters,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilatation=dilatation,
                # STDP
                theta_plus=theta_plus,
                nu=(nu_pre, nu_post),
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
