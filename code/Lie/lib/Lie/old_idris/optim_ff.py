# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-28T18:05:22+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-11T12:24:58+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import os
from time import time as t
import numpy as np
import torch

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import bindsnet
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes
from bindsnet.network.topology import Connection
import collections.abc as container_abcs

from Lie.objective import Objective


class FeedForward(Network):
    # language=rst

    def __init__(
        self,
        n_inpt: int,
        n_classes=10,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # Local
        n_in=100,
        n_hidden=25,
        n_out=50,
        # STDP
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        inh: float = 25.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        # In
        i_theta_plus: float = 0.05,
        i_tc_theta_decay: float = 1e7,
        i_rest=-65.0,
        i_reset=-60.0,
        i_thresh=-52.0,
        i_refrac=5,
        i_tc_decay=100.0,
        i_tc_trace=20.0,
        # hidden
        h_theta_plus: float = 0.05,
        h_tc_theta_decay: float = 1e7,
        h_rest=-65.0,
        h_reset=-60.0,
        h_thresh=-52.0,
        h_refrac=5,
        h_tc_decay=100.0,
        h_tc_trace=20.0,
        # out
        o_theta_plus: float = 0.05,
        o_tc_theta_decay: float = 1e7,
        o_rest=-65.0,
        o_reset=-60.0,
        o_thresh=-52.0,
        o_refrac=5,
        o_tc_decay=100.0,
        o_tc_trace=20.0,
    ) -> None:
        # language=rst
        super().__init__(dt=dt)
        self.n_classes = n_classes
        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_out = n_out

        input_layer = Input(
            shape=self.inpt_shape,
            traces=True,
            tc_trace=20,
        )

        in_layer = AdaptiveLIFNodes(
            n=n_in,
            traces=True,
            rest=i_rest,
            reset=i_reset,
            thresh=i_thresh,
            refrac=i_refrac,
            tc_trace=i_tc_trace,
            theta_plus=i_theta_plus,
            tc_theta_decay=i_tc_theta_decay,
        )

        hidden_layer = AdaptiveLIFNodes(
            n=n_hidden,
            traces=True,
            rest=h_rest,
            reset=h_reset,
            thresh=h_thresh,
            refrac=h_refrac,
            tc_trace=h_tc_trace,
            theta_plus=h_theta_plus,
            tc_theta_decay=h_tc_theta_decay,
        )

        output_layer = AdaptiveLIFNodes(
            n=n_out,
            traces=True,
            rest=o_rest,
            reset=o_reset,
            thresh=o_thresh,
            refrac=o_refrac,
            tc_trace=o_tc_trace,
            theta_plus=o_theta_plus,
            tc_theta_decay=o_tc_theta_decay,
        )

        input_in_conn = Connection(
            source=input_layer,
            target=in_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=0.5 * torch.rand(input_layer.n, in_layer.n),
        )

        in_hidden_conn = Connection(
            source=in_layer,
            target=hidden_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=0.5 * torch.rand(in_layer.n, hidden_layer.n),
        )

        hidden_out_layer = Connection(
            source=hidden_layer,
            target=output_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=0.5 * torch.rand(hidden_layer.n, output_layer.n),
        )

        self.add_layer(input_layer, name="inpt")
        self.add_layer(in_layer, name="in")
        self.add_layer(hidden_layer, name="hiddn")
        self.add_layer(output_layer, name="outpt")
        self.add_connection(input_in_conn, source="inpt", target="in")
        self.add_connection(in_hidden_conn, source="in", target="hiddn")
        self.add_connection(hidden_out_layer, source="hiddn", target="outpt")

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
        super().__call__(**kwargs)

        # Convolution
        n_in = kwargs.get("in", 100)
        n_hidden = kwargs.get("hidden", 50)
        n_out = kwargs.get("out", 25)
        # STDP
        nu_pre = kwargs.get("nu_pre", 1e-4)
        nu_post = kwargs.get("nu_post", 1e-2)
        # In
        i_theta_plus = kwargs.get("i_theta_plus", 0.05)
        i_tc_theta_decay = kwargs.get("i_tc_theta_decay", 1e7)
        i_rest = kwargs.get("i_rest", -65.0)
        i_reset = kwargs.get("i_reset", -60.0)
        i_thresh = kwargs.get("i_thresh", -52.0)
        i_refrac = kwargs.get("i_refrac", 5)
        i_tc_decay = kwargs.get("i_tc_decay", 100.0)
        i_tc_trace = kwargs.get("i_tc_trace", 20.0)
        # Hidden
        h_theta_plus = kwargs.get("h_theta_plus", 0.05)
        h_tc_theta_decay = kwargs.get("h_tc_theta_decay", 1e7)
        h_rest = kwargs.get("h_rest", -65.0)
        h_reset = kwargs.get("h_reset", -60.0)
        h_thresh = kwargs.get("h_thresh", -52.0)
        h_refrac = kwargs.get("h_refrac", 5)
        h_tc_decay = kwargs.get("h_tc_decay", 100.0)
        h_tc_trace = kwargs.get("h_tc_trace", 20.0)
        # Out
        o_theta_plus = kwargs.get("o_theta_plus", 0.05)
        o_tc_theta_decay = kwargs.get("o_tc_theta_decay", 1e7)
        o_rest = kwargs.get("o_rest", -65.0)
        o_reset = kwargs.get("o_reset", -60.0)
        o_thresh = kwargs.get("o_thresh", -52.0)
        o_refrac = kwargs.get("o_refrac", 5)
        o_tc_decay = kwargs.get("o_tc_decay", 100.0)
        o_tc_trace = kwargs.get("o_tc_trace", 20.0)

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
            network = FeedForward(
                n_inpt=self.input_features,
                inpt_shape=self.input_shape,
                dt=self.dt,
                n_in=n_in,
                n_hidden=n_hidden,
                n_out=n_out,
                # STDP
                nu=(nu_pre, nu_post),
                # In
                i_theta_plus=i_theta_plus,
                i_rest=i_rest,
                i_reset=i_reset,
                i_thresh=i_thresh,
                i_refrac=i_refrac,
                i_tc_decay=i_tc_decay,
                i_tc_trace=i_tc_trace,
                # Hidden
                h_theta_plus=h_theta_plus,
                h_rest=h_rest,
                h_reset=h_reset,
                h_thresh=h_thresh,
                h_refrac=h_refrac,
                h_tc_decay=h_tc_decay,
                h_tc_trace=h_tc_trace,
                # Excit
                o_theta_plus=o_theta_plus,
                o_rest=o_rest,
                o_reset=o_reset,
                o_thresh=o_thresh,
                o_refrac=o_refrac,
                o_tc_decay=o_tc_decay,
                o_tc_trace=o_tc_trace,
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
