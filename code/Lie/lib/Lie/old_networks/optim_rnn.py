# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-28T15:34:01+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-11T12:25:05+02:00
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


class RNN(Network):
    # language=rst

    def __init__(
        self,
        n_inpt: int,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # Local
        n_exc=100,
        n_inh=25,
        n_out=50,
        # STDP
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        inh: float = 25.0,
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
        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_out = n_out

        input_layer = Input(
            shape=self.inpt_shape,
            traces=True,
            tc_trace=20,
        )

        exc_layer = AdaptiveLIFNodes(
            n=n_exc,
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        inh_layer = AdaptiveLIFNodes(
            n=n_inh,
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        output_layer = AdaptiveLIFNodes(
            n=n_out,
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=0.5 * torch.rand(input_layer.n, exc_layer.n),
        )

        w = 0.5 * torch.rand(exc_layer.n, exc_layer.n)
        w = w.fill_diagonal_(0.0)
        exc_exc_conn = Connection(
            source=exc_layer,
            target=exc_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=w,
        )

        exc_inh_conn = Connection(
            source=exc_layer,
            target=inh_layer,
            w=0.5 * torch.rand(exc_layer.n, inh_layer.n),
        )

        inh_exc_conn = Connection(
            source=inh_layer,
            target=exc_layer,
            w=0.5 * torch.rand(inh_layer.n, exc_layer.n),
        )

        w = 0.5 * torch.rand(inh_layer.n, inh_layer.n)
        w = w.fill_diagonal_(0.0)
        inh_inh_conn = Connection(
            source=inh_layer,
            target=inh_layer,
            w=0.5 * torch.rand(inh_layer.n, inh_layer.n),
        )

        exc_output_layer = Connection(
            source=exc_layer,
            target=output_layer,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin,
            wmax=wmax,
            w=0.5 * torch.rand(exc_layer.n, output_layer.n),
        )

        self.add_layer(input_layer, name="inpt")
        self.add_layer(exc_layer, name="exc")
        self.add_layer(inh_layer, name="inh")
        self.add_layer(output_layer, name="outpt")
        self.add_connection(input_exc_conn, source="inpt", target="exc")
        self.add_connection(exc_exc_conn, source="exc", target="exc")
        self.add_connection(exc_inh_conn, source="exc", target="inh")
        self.add_connection(inh_exc_conn, source="inh", target="exc")
        self.add_connection(inh_inh_conn, source="inh", target="inh")
        self.add_connection(exc_output_layer, source="exc", target="outpt")

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
        torch.save(self, open(os.path.join(filepath, "assignments.pt"), "wb"))


class Model(Objective):
    def __call__(self, *args, load=False, train=True, **kwargs):

        super().__call__(**kwargs)
        # Convolution
        n_exc = kwargs.get("exc", 100)
        n_inh = kwargs.get("inh", 25)
        n_out = kwargs.get("out", 50)
        # STDP
        nu_pre = kwargs.get("nu_pre", 1e-4)
        nu_post = kwargs.get("nu_post", 1e-2)
        # Excit
        theta_plus = kwargs.get("theta_plus", 0.05)
        e_rest = kwargs.get("e_rest", -65.0)
        e_reset = kwargs.get("e_reset", -60.0)
        e_thresh = kwargs.get("e_thresh", -52.0)
        e_refrac = kwargs.get("e_refrac", 5)
        e_tc_decay = kwargs.get("e_tc_decay", 100.0)
        e_tc_trace = kwargs.get("e_tc_trace", 20.0)

        network = RNN(
            n_inpt=self.input_features,
            inpt_shape=self.input_shape,
            dt=self.dt,
            n_exc=n_exc,
            n_inh=n_inh,
            n_out=n_out,
            # STDP
            nu=(nu_pre, nu_post),
            # Excit
            theta_plus=theta_plus,
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
