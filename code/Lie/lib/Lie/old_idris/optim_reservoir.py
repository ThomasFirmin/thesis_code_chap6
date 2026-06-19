# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-03-17T12:28:52+01:00
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


class Reservoir(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_classes,
        n_exc: int = 100,
        n_inh: int = 100,
        n_out: int = 100,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # STDP
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        # Excit
        e_theta_plus: float = 0.05,
        e_tc_theta_decay: float = 1e7,
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
        # Outpt
        o_theta_plus: float = 0.05,
        o_tc_theta_decay: float = 1e7,
        o_rest=-65.0,
        o_reset=-60.0,
        o_thresh=-52.0,
        o_refrac=5,
        o_tc_decay=100.0,
        o_tc_trace=20.0,
    ) -> None:
        print(
            f"n_inpt:{n_inpt}\nn_classes:{n_classes}\nn_neurons:{n_neurons}\ninpt_shape:{inpt_shape}\ndt:{dt}\nexc:{exc}\ninh:{inh}\nweight_decay:{weight_decay}\nnu:{nu}\nreduction:{reduction}\nwmin:{wmin}\nwmax:{wmax}\nnorm:{norm}\ntheta_plus:{theta_plus}\ntc_theta_decay:{tc_theta_decay}\ne_rest:{e_rest}\ne_reset:{e_reset}\ne_thresh:{e_thresh}\ne_refrac:{e_refrac}\ne_tc_decay:{e_tc_decay}\ne_tc_trace:{e_tc_trace}\ni_rest:{i_rest}\ni_reset:{i_reset}\ni_thresh:{i_thresh}\ni_tc_decay:{i_tc_decay}\ni_refrac:{i_refrac}\ni_tc_trace"
        )
        """
        Constructor for class ``Reservoir``.
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

        self.n_classes = n_classes
        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_out = n_out
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(n=self.n_inpt, shape=self.inpt_shape, traces=False)
        exc_layer = AdaptiveLIFNodes(
            n=self.n_exc,
            traces=False,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_decay=e_tc_decay,
            tc_trace=e_tc_trace,
            theta_plus=e_theta_plus,
            tc_theta_decay=e_tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_inh,
            traces=False,
            rest=i_rest,
            reset=i_reset,
            thresh=i_thresh,
            tc_decay=i_tc_decay,
            refrac=i_refrac,
            tc_trace=i_tc_trace,
        )
        outpt_layer = AdaptiveLIFNodes(
            n=self.n_out,
            traces=True,
            rest=o_rest,
            reset=o_reset,
            thresh=o_thresh,
            refrac=o_refrac,
            tc_decay=o_tc_decay,
            tc_trace=o_tc_trace,
            theta_plus=o_theta_plus,
            tc_theta_decay=o_tc_theta_decay,
        )

        # Inputs -> Excitatory
        w = torch.rand(self.n_inpt, self.n_exc)
        input_exc_conn = Connection(
            source=input_layer, target=exc_layer, w=w, wmin=0.0, wmax=1.0
        )
        # Excitatory -> Inhibitory
        w = torch.rand(self.n_exc, self.n_inh)
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=1.0
        )
        # Inhibitory -> Excitatory
        w = -torch.rand(self.n_inh, self.n_exc)
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=0, wmax=1.0
        )
        # Excitatory -> Excitatory
        w = torch.rand(self.n_exc, self.n_exc)
        exc_exc_conn = Connection(
            source=exc_layer, target=exc_layer, w=w, wmin=0, wmax=1.0
        )
        # Excitatory -> Output
        w = torch.rand(self.n_exc, self.n_inh)
        exc_out_conn = Connection(
            source=exc_layer,
            target=outpt_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            weight_decay=weight_decay,
        )

        # Add to network
        self.add_layer(input_layer, name="inpt")
        self.add_layer(exc_layer, name="exc")
        self.add_layer(inh_layer, name="inh")
        self.add_layer(outpt_layer, name="outpt")
        self.add_connection(input_exc_conn, source="inpt", target="exc")
        self.add_connection(exc_inh_conn, source="exc", target="inh")
        self.add_connection(inh_exc_conn, source="inh", target="exc")
        self.add_connection(exc_exc_conn, source="exc", target="exc")
        self.add_connection(exc_out_conn, source="exc", target="outpt")

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
            open(os.path.join(filepath, "proportions.pt"), "wb"),
        )
        torch.save(self.rates, open(os.path.join(filepath, "rates.pt"), "wb"))


class Model(Objective):
    def __call__(self, *args, load=False, train=True, **kwargs):
        print(args, kwargs)

        # epochs, encoding_window, decoder, encoder
        super().__call__(*args, **kwargs)

        n_exc = kwargs.get("n_exc", 100)
        n_inh = kwargs.get("n_inh", 100)
        n_out = kwargs.get("n_out", 100)
        # STDP
        weight_decay = kwargs.get("weight_decay", 0.0)
        nu = kwargs.get("nu", (1e-4, 1e-2))
        wmin = kwargs.get("wmin", 0.0)
        wmax = kwargs.get("wmax", 1.0)
        norm = kwargs.get("norm", 78.4)
        # Excit
        e_theta_plus = kwargs.get("e_theta_plus", 0.05)
        e_tc_theta_decay = kwargs.get("e_tc_theta_decay", 1e7)
        e_rest = kwargs.get("e_rest", 65.0)
        e_reset = kwargs.get("e_reset", 60.0)
        e_thresh = kwargs.get("e_thresh", 52.0)
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
        # Outpt
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
            network = Reservoir(
                n_inpt=self.input_features,
                inpt_shape=self.input_shape,
                dt=self.dt,
                # STDP
                weight_decay=weight_decay,
                nu=nu,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
                # Excit
                e_theta_plus=e_theta_plus,
                e_tc_theta_decay=e_tc_theta_decay,
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
                # Outpt
                o_theta_plus=o_theta_plus,
                o_tc_theta_decay=o_tc_theta_decay,
                o_rest=o_rest,
                o_reset=o_reset,
                o_thresh=o_thresh,
                o_refrac=o_refrac,
                o_tc_decay=o_tc_decay,
                o_tc_trace=o_tc_trace,
            )

        init_w = torch.flatten(
            torch.clone(network.connections["exc", "outpt"].w)
        ).to("cpu")

        # Directs network to GPU
        if self.gpu:
            print("Device:", self.device)
            network.to(self.device)

        start = t()
        if train:
            network = self.train(network)

        end_w = torch.flatten(network.connections["exc", "outpt"].w).to("cpu")

        # init
        init_amean = torch.mean(init_w).item()
        init_astd = torch.std(init_w).item()
        init_amedian = torch.median(init_w).item()
        init_afqrt = torch.quantile(init_w, 0.25).item()
        init_atqrt = torch.quantile(init_w, 0.75).item()
        init_amin = torch.min(init_w).item()
        init_amax = torch.max(init_w).item()
        init_nonzeros = torch.count_nonzero(init_w).item()
        init_zeros = len(init_w) - init_nonzeros
        # init
        end_amean = torch.mean(end_w).item()
        end_astd = torch.std(end_w).item()
        end_amedian = torch.median(end_w).item()
        end_afqrt = torch.quantile(end_w, 0.25).item()
        end_atqrt = torch.quantile(end_w, 0.75).item()
        end_amin = torch.min(end_w).item()
        end_amax = torch.max(end_w).item()
        end_nonzeros = torch.count_nonzero(end_w).item()
        end_zeros = len(end_w) - end_nonzeros
        # end
        variations = end_w - init_w
        # All
        amean = torch.mean(variations).item()
        astd = torch.std(variations).item()
        amedian = torch.median(variations).item()
        afqrt = torch.quantile(variations, 0.25).item()
        atqrt = torch.quantile(variations, 0.75).item()
        amin = torch.min(variations).item()
        amax = torch.max(variations).item()
        nonzeros = torch.count_nonzero(variations).item()
        zeros = len(variations) - nonzeros
        # end
        variations = variations[variations != 0]
        # Non zeros
        nzmean = torch.mean(variations).item()
        nzstd = torch.std(variations).item()
        nzmedian = torch.median(variations).item()
        nzfqrt = torch.quantile(variations, 0.25).item()
        nztqrt = torch.quantile(variations, 0.75).item()
        nzmin = torch.min(variations).item()
        nzmax = torch.max(variations).item()

        init_w = None
        end_w = None
        variations = None

        if not self.stopped:
            network = self.test(network)

        results = self.accuracy()

        results["init_amean"] = init_amean
        results["init_astd"] = init_astd
        results["init_amedian"] = init_amedian
        results["init_afqrt"] = init_afqrt
        results["init_atqrt"] = init_atqrt
        results["init_amin"] = init_amin
        results["init_amax"] = init_amax
        results["init_nonzeros"] = init_nonzeros
        results["init_zeros"] = init_zeros

        results["end_amean"] = end_amean
        results["end_astd"] = end_astd
        results["end_amedian"] = end_amedian
        results["end_afqrt"] = end_afqrt
        results["end_atqrt"] = end_atqrt
        results["end_amin"] = end_amin
        results["end_amax"] = end_amax
        results["end_nonzeros"] = end_nonzeros
        results["end_zeros"] = end_zeros

        results["w_amean"] = amean
        results["w_astd"] = astd
        results["w_amedian"] = amedian
        results["w_afqrt"] = afqrt
        results["w_atqrt"] = atqrt
        results["w_amin"] = amin
        results["w_amax"] = amax
        results["w_nonzeros"] = nonzeros
        results["w_zeros"] = zeros

        results["w_nzmean"] = nzmean
        results["w_nzstd"] = nzstd
        results["w_nzmedian"] = nzmedian
        results["w_nzfqrt"] = nzfqrt
        results["w_nztqrt"] = nztqrt
        results["w_nzmin"] = nzmin
        results["w_nzmax"] = nzmax

        return results, network
