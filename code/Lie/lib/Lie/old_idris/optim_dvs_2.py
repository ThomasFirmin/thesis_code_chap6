# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-16T15:18:15+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import os
from time import time as t
import numpy as np
import torch
from functools import reduce
from operator import mul

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import bindsnet
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection2D
import collections.abc as container_abcs

from Lie.abstract_network import AbstractNetwork


class LocalReservoir(AbstractNetwork):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_classes,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # Local
        n_filters=6,
        kernel_size=8,
        stride=2,
        # Reservoir
        n_reservoir: int = 100,
        inh_ratio: float = 0.20,
        inh=25,
        # Out layer
        n_out: int = 50,
        # Local Connection
        c_theta_plus: float = 0.05,
        c_tc_theta_decay: float = 1e7,
        c_rest=-65.0,
        c_reset=-60.0,
        c_thresh=-52.0,
        c_refrac=5,
        c_tc_decay=100.0,
        c_tc_trace=20.0,
        # Excit
        e_theta_plus: float = 0.05,
        e_tc_theta_decay: float = 1e7,
        e_rest=-65.0,
        e_reset=-60.0,
        e_thresh=-52.0,
        e_refrac=5,
        e_tc_decay=100.0,
        e_tc_trace=20.0,
        # Outpt
        o_theta_plus: float = 0.05,
        o_tc_theta_decay: float = 1e7,
        o_rest=-65.0,
        o_reset=-60.0,
        o_thresh=-52.0,
        o_refrac=5,
        o_tc_decay=100.0,
        o_tc_trace=20.0,
        out_inh=1,
        # STDP
        weight_decay=0.0,
        nu_pre=1e-4,
        nu_post=1e-2,
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = None,
        lateral_inhibition=True,
    ) -> None:
        super().__init__(n_inpt, n_classes, inpt_shape, dt=dt)

        # Network
        # Local connection
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.c_rest = c_rest
        self.c_reset = c_reset
        self.c_thresh = c_thresh
        self.c_refrac = c_refrac
        self.c_tc_decay = c_tc_decay
        self.c_tc_trace = c_tc_trace
        self.c_theta_plus = c_theta_plus
        self.c_tc_theta_decay = c_tc_theta_decay
        self.inh = inh
        # Excitatory reservoir
        self.n_reservoir = n_reservoir
        self.inh_ratio = inh_ratio

        self.e_rest = e_rest
        self.e_reset = e_reset
        self.e_thresh = e_thresh
        self.e_refrac = e_refrac
        self.e_tc_decay = e_tc_decay
        self.e_tc_trace = e_tc_trace
        self.e_theta_plus = e_theta_plus
        self.e_tc_theta_decay = e_tc_theta_decay

        # Output
        self.n_out = n_out

        self.o_rest = o_rest
        self.o_reset = o_reset
        self.o_thresh = o_thresh
        self.o_refrac = o_refrac
        self.o_tc_decay = o_tc_decay
        self.o_tc_trace = o_tc_trace
        self.o_theta_plus = o_theta_plus
        self.o_tc_theta_decay = o_tc_theta_decay
        self.out_inh = out_inh

        # STDP
        self.weight_decay = weight_decay
        self.nu = (nu_pre, nu_post)
        self.reduction = reduction
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        self.lateral_inhibition = lateral_inhibition

        # Input
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="inpt")

        # Input to Reservoir
        res_layer, name = self._reservoir(0, input_layer, "inpt")

        # Local
        conv_out, name = self._local(0, res_layer, name)

        output_layer, name = self.ff_layer(0, conv_out, name, 500)
        output_layer, name = self.outpt_layer(output_layer, name, self.n_out)

    def _reservoir(self, id, inputs, in_name):
        # Reservoir
        exc_layer = AdaptiveLIFNodes(
            shape=[1, self.n_reservoir, self.n_reservoir],
            traces=True,
            rest=self.e_rest,
            reset=self.e_reset,
            thresh=self.e_thresh,
            refrac=self.e_refrac,
            tc_decay=self.e_tc_decay,
            tc_trace=self.e_tc_trace,
            theta_plus=self.e_theta_plus,
            tc_theta_decay=self.e_tc_theta_decay,
        )
        out_name = f"res_{id}"

        # Inputs -> Excitatory
        w = torch.rand(reduce(mul, inputs.shape), self.n_reservoir**2)
        # 70% of synapses are =0
        rand_inhib = int(0.70 * self.n_reservoir**2)
        selected = torch.randperm(self.n_reservoir**2)[:rand_inhib]
        w[:, selected] = float(0.0)
        print("Reservoir inputs", w.shape, inputs.shape, exc_layer.shape)
        input_exc_conn = Connection(source=inputs, target=exc_layer, w=w.float())

        # Excitatory -> Excitatory
        rrange = torch.arange(self.n_reservoir)
        mesh = torch.meshgrid(rrange, rrange)  # 3D
        points = torch.vstack(list(map(torch.ravel, mesh))).T
        w = torch.cdist(points.float(), points.float())
        w = w.reshape(
            1,
            self.n_reservoir,
            self.n_reservoir,
            1,
            self.n_reservoir,
            self.n_reservoir,
        )

        wmin, wmax = w.min(), w.max()
        w -= wmin
        w /= wmax - wmin
        w[w != 0.0] = 1.0 - w[w != 0.0]

        rand_inhib = int(self.inh_ratio * self.n_reservoir**2)
        selected = points[torch.randperm(self.n_reservoir**2)[:rand_inhib]]
        w[0, selected[:, 0], selected[:, 1], 0, :, :] = -w[
            0, selected[:, 0], selected[:, 1], 0, :, :
        ]
        w = w.reshape(self.n_reservoir**2, self.n_reservoir**2)

        print("Reservoir rec ", w.shape)
        exc_exc_conn = Connection(
            source=exc_layer, target=exc_layer, w=w.float(), wmin=0, wmax=1.0
        )

        # Add layer
        self.add_layer(exc_layer, name=out_name)
        # Add topology
        self.add_connection(input_exc_conn, source=in_name, target=out_name)
        self.add_connection(exc_exc_conn, source=out_name, target=out_name)

        return (exc_layer, out_name)

    def _local(self, id, inputs, in_name):
        # Conv size
        compute_conv_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
        # Dimension 0 = channels
        # Dimension 1 = width
        conv_size_d1 = compute_conv_size(inputs.shape[1], self.kernel_size, self.stride)
        # Dimension 2 = length
        conv_size_d2 = compute_conv_size(inputs.shape[2], self.kernel_size, self.stride)
        print(
            "Local filters:",
            self.n_filters,
            conv_size_d1,
            conv_size_d2,
        )
        assert (
            self.kernel_size <= conv_size_d1 and self.kernel_size <= conv_size_d2
        ), f"Kernel size and convolution size does not match, got for lolcal_{id}, inputs {in_name} of shape {inputs.shape}, d1:{self.kernel_size}<={conv_size_d1}, d2:{self.kernel_size}<={conv_size_d2}"

        output_layer = AdaptiveLIFNodes(
            shape=(self.n_filters, conv_size_d1, conv_size_d2),
            traces=True,
            rest=self.c_rest,
            reset=self.c_reset,
            thresh=self.c_thresh,
            refrac=self.c_refrac,
            tc_trace=self.c_tc_trace,
            theta_plus=self.c_theta_plus,
            tc_theta_decay=self.c_tc_theta_decay,
        )
        out_name = f"local_outpt_{id}"
        input_output_conn = LocalConnection2D(
            inputs,
            output_layer,
            kernel_size=self.kernel_size,
            stride=self.stride,
            n_filters=self.n_filters,
            nu=self.nu,
            update_rule=PostPre,
            wmin=self.wmin,
            wmax=self.wmax,
            norm=self.norm,
        )
        print(f"Local shape: {inputs.shape},{output_layer.shape}")

        w_inh_LC = torch.zeros(
            self.n_filters,
            conv_size_d1,
            conv_size_d2,
            self.n_filters,
            conv_size_d1,
            conv_size_d2,
        )
        for c in range(self.n_filters):
            for w1 in range(conv_size_d1):
                for w2 in range(conv_size_d2):
                    w_inh_LC[c, w1, w2, :, w1, w2] = -self.inh
                    w_inh_LC[c, w1, w2, c, w1, w2] = 0

        w_inh_LC = w_inh_LC.reshape(output_layer.n, output_layer.n)
        recurrent_conn = Connection(output_layer, output_layer, w=w_inh_LC.float())

        # Add layer
        self.add_layer(output_layer, name=out_name)
        # Add topology
        self.add_connection(input_output_conn, source=in_name, target=out_name)
        self.add_connection(recurrent_conn, source=out_name, target=out_name)

        return output_layer, out_name

    def ff_layer(self, id, inputs, in_name, nneurons):
        outpt_layer = AdaptiveLIFNodes(
            n=nneurons,
            traces=True,
            rest=self.o_rest,
            reset=self.o_reset,
            thresh=self.o_thresh,
            refrac=self.o_refrac,
            tc_trace=self.o_tc_trace,
            theta_plus=self.o_theta_plus,
            tc_theta_decay=self.o_tc_theta_decay,
        )
        out_name = f"ff_{id}"

        # Output
        w = torch.rand(reduce(mul, inputs.shape), nneurons)
        exc_out_conn = Connection(
            source=inputs,
            target=outpt_layer,
            w=w.float(),
            update_rule=PostPre,
            nu=self.nu,
            reduction=self.reduction,
            wmin=self.wmin,
            wmax=self.wmax,
            norm=self.norm,
            weight_decay=self.weight_decay,
        )

        # Add layer
        self.add_layer(outpt_layer, name=out_name)
        # Add topology
        self.add_connection(exc_out_conn, source=in_name, target=out_name)

        return outpt_layer, out_name

    def outpt_layer(self, inputs, in_name, nneurons):
        outpt_layer = AdaptiveLIFNodes(
            n=nneurons,
            traces=True,
            rest=self.o_rest,
            reset=self.o_reset,
            thresh=self.o_thresh,
            refrac=self.o_refrac,
            tc_trace=self.o_tc_trace,
            theta_plus=self.o_theta_plus,
            tc_theta_decay=self.o_tc_theta_decay,
        )

        # Output
        w = torch.rand(reduce(mul, inputs.shape), nneurons)
        exc_out_conn = Connection(
            source=inputs,
            target=outpt_layer,
            w=w.float(),
            update_rule=PostPre,
            nu=self.nu,
            reduction=self.reduction,
            wmin=self.wmin,
            wmax=self.wmax,
            norm=self.norm,
            weight_decay=self.weight_decay,
        )
        out_name = "outpt"

        # Add layer
        self.add_layer(outpt_layer, name=out_name)
        # Add topology
        self.add_connection(exc_out_conn, source=in_name, target=out_name)

        if self.lateral_inhibition:
            rrange = torch.arange(nneurons)
            mesh = torch.meshgrid(rrange, rrange)  # 3D
            points = torch.vstack(list(map(torch.ravel, mesh))).T
            w = torch.cdist(points.float(), points.float())
            w /= w.max()
            w *= self.out_inh
            w.fill_diagonal_(0.0)

            # Lateral inhibition
            wta = Connection(
                source=outpt_layer,
                target=outpt_layer,
                w=w.float(),
            )
            self.add_connection(wta, source=out_name, target=out_name)

        return outpt_layer, out_name
