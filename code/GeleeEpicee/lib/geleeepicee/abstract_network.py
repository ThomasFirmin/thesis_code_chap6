# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-19T17:08:25+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:24:03+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
from typing import Any
import torch
import h5py
import os


class AbstractNetwork(torch.nn.Module):
    def __init__(self, n_inpt, n_classes, inpt_shape):
        super(AbstractNetwork, self).__init__()
        self.n_inpt = n_inpt
        self.n_classes = n_classes
        self.inpt_shape = inpt_shape

        self.recorders_train = {"inpt": 0, "outpt": 0}
        self.recorders_test = {"inpt": 0, "outpt": 0}

        self.computed_images_train = 0
        self.computed_images_test = 0

        self._train_mode = True

    def _update_recorders(self, spikes, layer):
        numpspikes = int(torch.sum(spikes).item())
        if self._train_mode:
            self.recorders_train[layer] += numpspikes
        else:
            self.recorders_test[layer] += numpspikes

    def _update_computed_images(self, spikes):
        if self._train_mode:
            self.computed_images_train += int(spikes.shape[1])
        else:
            self.computed_images_test += int(spikes.shape[1])
