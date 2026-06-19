# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-27T09:51:41+02:00
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
        n_layers=1,
        # Local
        n_filters=11,
        kernel_size=20,
        stride=9,
        # Reservoir
        n_exc: int = 800,
        n_inh: int = 200,
        # Out layer
        n_out: int = 100,
        # Local Connection
        c_theta_plus: float = 0.05,
        c_tc_theta_decay: float = 1e7,
        c_rest=-65.0,
        c_reset=-60.0,
        c_thresh=-52.0,
        c_refrac=5,
        c_tc_decay=100.0,
        c_tc_trace=20.0,
        inh=25,
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
        # STDP
        weight_decay=0.0,
        nu_pre=1e-4,
        nu_post=1e-2,
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = None,
    ) -> None:

        super().__init__(n_inpt, n_classes, inpt_shape, dt=dt)

        # Network
        self.n_layers = n_layers
        # Local conenction
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.c_rest = e_rest
        self.c_reset = e_reset
        self.c_thresh = e_thresh
        self.c_refrac = e_refrac
        self.c_tc_decay = e_tc_decay
        self.c_tc_trace = e_tc_trace
        self.c_theta_plus = e_theta_plus
        self.c_tc_theta_decay = e_tc_theta_decay
        self.inh = inh
        # Excitatory reservoir
        self.n_exc = n_exc

        self.e_rest = e_rest
        self.e_reset = e_reset
        self.e_thresh = e_thresh
        self.e_refrac = e_refrac
        self.e_tc_decay = e_tc_decay
        self.e_tc_trace = e_tc_trace
        self.e_theta_plus = e_theta_plus
        self.e_tc_theta_decay = e_tc_theta_decay
        # Inhibitory reservoir
        self.n_inh = n_inh

        self.i_rest = i_rest
        self.i_reset = i_reset
        self.i_thresh = i_thresh
        self.i_tc_decay = i_tc_decay
        self.i_refrac = i_refrac
        self.i_tc_trace = i_tc_trace

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

        self.weight_decay = weight_decay
        self.nu = (nu_pre, nu_post)
        self.reduction = reduction
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        # Out list
        out_list = []
        connections_list = []

        # Input
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="inpt")

        # Input to conv
        conv_out = self._local(0, input_layer)

        # Reservoir
        inh_layer, exc_layer = self._reservoir(0, conv_out)
        out_list.append(exc_layer)

        # additionnal layers
        for l in range(1, self.n_layers):

            # Local part
            conv_out = self._local(l, conv_out)

            # Reservoir
            inh_layer, exc_layer = self._reservoir(0, conv_out)
            out_list.append(exc_layer)

        outpt_layer = AdaptiveLIFNodes(
            n=self.n_out,
            traces=True,
            rest=self.o_rest,
            reset=self.o_reset,
            thresh=self.o_thresh,
            refrac=self.o_refrac,
            tc_trace=self.o_tc_trace,
            theta_plus=self.o_theta_plus,
            tc_theta_decay=self.o_tc_theta_decay,
        )
        self.add_layer(outpt_layer, name=f"outpt")
        # Link Reservoir outputs to learning output
        for layer in out_list:

            # Output
            w = torch.rand(*layer.shape, self.n_out)
            exc_out_conn = Connection(
                source=layer,
                target=outpt_layer,
                w=w,
                update_rule=PostPre,
                nu=self.nu,
                reduction=self.reduction,
                wmin=self.wmin,
                wmax=self.wmax,
                norm=self.norm,
                weight_decay=self.weight_decay,
            )

        self.add_connection(exc_out_conn, source="exc", target="outpt")

    def _reservoir(self, id, inputs):
        # Reservoir
        exc_layer = AdaptiveLIFNodes(
            n=self.n_exc,
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
        inh_layer = LIFNodes(
            n=self.n_inh,
            traces=False,
            rest=self.i_rest,
            reset=self.i_reset,
            thresh=self.i_thresh,
            tc_decay=self.i_tc_decay,
            refrac=self.i_refrac,
            tc_trace=self.i_tc_trace,
        )

        # Inputs -> Excitatory
        w = torch.ones(reduce(mul, inputs.shape), self.n_exc)
        input_exc_conn = Connection(source=inputs, target=exc_layer, w=w)

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
        w = torch.zeros(self.n_exc, self.n_exc)
        upper_triangular = torch.rand(int(self.n_exc * (self.n_exc + 1) / 2))
        i, j = torch.triu_indices(self.n_exc, self.n_exc)
        w[i, j] = upper_triangular
        w[j, i] = upper_triangular
        w.fill_diagonal_(0)

        exc_exc_conn = Connection(
            source=exc_layer, target=exc_layer, w=w, wmin=0, wmax=1.0
        )

        # Add layer
        self.add_layer(exc_layer, name=f"exc_{id}")
        self.add_layer(inh_layer, name=f"inh_{id}")
        # Add topology
        self.add_connection(
            input_exc_conn, source=f"inpt_{id}", target=f"exc_{id}"
        )
        self.add_connection(
            exc_inh_conn, source=f"exc_{id}", target=f"inh_{id}"
        )
        self.add_connection(
            inh_exc_conn, source=f"inh_{id}", target=f"exc_{id}"
        )
        self.add_connection(
            exc_exc_conn, source=f"exc_{id}", target=f"exc_{id}"
        )

        return inh_layer, exc_layer

    def _local(self, id, inputs):
        # Conv size
        compute_conv_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
        # Dimension 0 = channels
        # Dimension 1 = width
        conv_size_d1 = compute_conv_size(
            inputs.shape[1], self.kernel_size, self.stride
        )
        # Dimension 2 = length
        conv_size_d2 = compute_conv_size(
            inputs.shape[2], self.kernel_size, self.stride
        )
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
        recurrent_conn = Connection(output_layer, output_layer, w=w_inh_LC)

        self.add_layer(output_layer, name=f"local_outpt_{id}")
        self.add_connection(
            input_output_conn,
            source=f"local_in_{id}",
            target=f"local_outpt_{id}",
        )
        self.add_connection(
            recurrent_conn,
            source=f"local_outpt_{id}",
            target=f"local_outpt_{id}",
        )

        return output_layer
