from typing import Tuple
import numpy as np

import torch

from norse.torch.functional.heaviside import heaviside
from norse.torch.functional.stdp import (
    stdp_step_linear,
    STDPParameters,
    STDPState,
)


class NewSTDPState:
    """State of spike-timing-dependent plasticity (STDP).
    Parameters:
        t_pre (torch.Tensor): presynaptic spike trace
        t_post (torch.Tensor): postsynaptic spike trace
    """

    def __init__(self, t_pre: torch.Tensor, t_post: torch.Tensor):
        self.t_pre = t_pre
        self.t_post = t_post

    def decay(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
        tau_pre_inv: torch.Tensor,
        tau_post_inv: torch.Tensor,
        a_pre: torch.Tensor,
        a_post: torch.Tensor,
        dt: float = 0.001,
    ):
        """Decay function for STDP traces.
        Parameters:
            z_pre (torch.Tensor): presynaptic spikes
            z_post (torch.Tensor): postsynaptic spikes
            tau_pre_inv (torch.Tensor): inverse time-constant for the presynaptic trace
            tau_post (torch.Tensor): inverse time-constant for the postsynaptic trace
            a_pre (torch.Tensor): presynaptic trace
            a_post (torch.Tensor): postsynaptic trace
            dt (float): time-resolution
        """
        self.t_pre -= self.t_pre / tau_pre_inv
        self.t_post -= self.t_post / tau_post_inv
