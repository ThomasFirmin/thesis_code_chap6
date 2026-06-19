import os
from time import time as t
import numpy as np
import torch

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection2D

from Lie.objective import Objective


class Local2D(Network):
    # language=rst

    def __init__(
        self,
        n_inpt: int,
        inpt_shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        # Local
        n_filters=50,
        kernel_size=12,
        stride=4,
        # STDP
        weight_decay=0.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        inh: float = 25.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        # Excit
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e6,
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
        # Hyperparameters
        self.n_filters = n_filters  # 50
        self.inpt_shape = inpt_shape  # [20, 20]
        self.kernel_size = kernel_size  # _pair(12)
        self.stride = stride  # _pair(4)
        self.norm = 0.2 * kernel_size ** 2

        input_layer = Input(
            shape=self.inpt_shape,
            traces=True,
            tc_trace=20,
        )

        compute_conv_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
        self.conv_size = compute_conv_size(
            self.inpt_shape[1], self.kernel_size, self.stride
        )

        output_layer = AdaptiveLIFNodes(
            shape=(self.n_filters, self.conv_size, self.conv_size),
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        input_output_conn = LocalConnection2D(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=self.norm,
        )

        w_inh_LC = torch.zeros(
            self.n_filters,
            self.conv_size,
            self.conv_size,
            self.n_filters,
            self.conv_size,
            self.conv_size,
        )
        for c in range(n_filters):
            for w1 in range(self.conv_size):
                for w2 in range(self.conv_size):
                    w_inh_LC[c, w1, w2, :, w1, w2] = -inh
                    w_inh_LC[c, w1, w2, c, w1, w2] = 0

        w_inh_LC = w_inh_LC.reshape(output_layer.n, output_layer.n)
        recurrent_conn = Connection(output_layer, output_layer, w=w_inh_LC)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")

        self.assignments = -torch.ones(self.n_filters * self.conv_size ** 2)

    def to(self, device=None, *args, **kwargs):
        self.assignments = self.assignments.to(device=device)
        return super().to(device=device, *args, **kwargs)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        super().save(os.path.join(filepath, "network.pt"))
        torch.save(self, open(os.path.join(filepath, "assignments.pt"), "wb"))


class Model(Objective):
    def __call__(self, **kwargs):
        super().__call__(**kwargs)

        # Convolution
        n_filters = kwargs.get("filters", 50)
        kernel_size = kwargs.get("kernel_size", 12)
        stride = kwargs.get("stride", 4)
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

        network = Local2D(
            n_inpt=self.input_features,
            inpt_shape=self.input_shape,
            dt=self.dt,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
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
            network.to("cuda")

        self.train(network, decoder=self.decoder)
        self.test(network, decoder=self.decoder)

        return self.accuracy, network
