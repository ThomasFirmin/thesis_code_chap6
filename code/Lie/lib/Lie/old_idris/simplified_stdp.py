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


def sstdp_step_linear(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    w: torch.Tensor,
    state_stdp: STDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """STDP step for a FF LIF layer.
    Input:
        z_pre (torch.Tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        z_post (torch.Tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update STDP traces
    state_stdp.decay(
        z_pre,
        z_post,
        p_stdp.tau_pre_inv,
        p_stdp.tau_post_inv,
        p_stdp.a_pre,
        p_stdp.a_post,
        dt,
    )
    z_post_np = z_post.numpy()
    z_pre_np = z_pre.numpy()
    t_post_np = state_stdp.t_post.numpy()

    delta = np.zeros_like(w.numpy())

    condp1 = (t_post_np > 0)[:, None]
    condp2 = z_post_np.transpose(1, 0) - z_pre_np <= 0
    condp2bis = (t_post_np > 0)[:, None] & (
        z_post_np.transpose(1, 0) - z_pre_np > 0
    )

    np.putmask(
        delta,
        (
            (t_post_np > 0)[:, None]
            & (z_post_np.transpose(1, 0) - z_pre_np <= 0)
        ),
        p_stdp.A_plus(w)
        * np.exp(-1.5 * ((w - p_stdp.w_min) / (p_stdp.w_max - p_stdp.w_min))),
    )

    np.putmask(
        delta,
        ((t_post_np > 0)[:, None] & (z_post_np.transpose(1, 0) - z_pre_np > 0)),
        -p_stdp.A_minus(w)
        * np.exp(-2.5 * ((p_stdp.w_max - w) / (p_stdp.w_max - p_stdp.w_min))),
    )

    w += delta
    # Bound checking
    w = p_stdp.bounding_func(w)

    return (w, state_stdp)


def bstdp_step_linear(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    w: torch.Tensor,
    state_stdp: NewSTDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, NewSTDPState]:
    """STDP step for a FF LIF layer.
    Input:
        z_pre (torch.Tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        z_post (torch.Tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    mask_pre = z_pre.squeeze() == 1
    mask_post = z_post.squeeze() == 1

    if torch.any(mask_pre):
        state_stdp.t_pre[mask_pre] += p_stdp.eta_plus
        w[:, mask_pre] = torch.clamp(
            w[:, mask_pre] + state_stdp.t_post[:, None],
            p_stdp.w_min,
            p_stdp.w_max,
        )

    if torch.any(mask_post):
        state_stdp.t_post[mask_post] += p_stdp.eta_minus
        w[mask_post, :] = torch.clamp(
            w[mask_post, :] + state_stdp.t_pre,
            p_stdp.w_min,
            p_stdp.w_max,
        )

    # Update STDP traces
    state_stdp.decay(
        mask_pre,
        mask_post,
        p_stdp.tau_pre_inv,
        p_stdp.tau_post_inv,
        p_stdp.a_pre,
        p_stdp.a_post,
        dt,
    )

    return (w, state_stdp)


def new_stdp_step_linear(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    w: torch.Tensor,
    state_stdp: NewSTDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, NewSTDPState]:
    """STDP step for a FF LIF layer.
    Input:
        z_pre (torch.Tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        z_post (torch.Tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update STDP traces
    state_stdp.decay(
        z_pre,
        z_post,
        p_stdp.tau_pre_inv,
        p_stdp.tau_post_inv,
        p_stdp.a_pre,
        p_stdp.a_post,
        dt,
    )

    mask_pre = z_pre.squeeze() == 1
    mask_post = z_post.squeeze() == 1

    if torch.any(mask_pre):
        state_stdp.t_pre[mask_pre] += p_stdp.eta_plus
        w[:, mask_pre] = torch.clamp(
            torch.where(
                state_stdp.t_post[:, None] < 0,
                w[:, mask_pre] + state_stdp.t_post[:, None],
                w[:, mask_pre],
            ),
            p_stdp.w_min,
            p_stdp.w_max,
        )

    if torch.any(mask_post):
        state_stdp.t_post[mask_post] += p_stdp.eta_minus
        w[mask_post, :] = torch.clamp(
            torch.where(
                state_stdp.t_pre > 0,
                w[mask_post, :] + state_stdp.t_pre,
                w[mask_post, :] + state_stdp.t_post[mask_post, None],
            ),
            p_stdp.w_min,
            p_stdp.w_max,
        )

    return (w, state_stdp)
