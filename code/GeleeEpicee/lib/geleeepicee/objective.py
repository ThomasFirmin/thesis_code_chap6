# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-28T18:48:58+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:52:09+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin

import traceback
from time import time as t

import numpy as np
import random

import torch
import torch.nn.functional as F
import sys

from spikingjelly.activation_based import functional


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Objective(object):

    def __init__(
        self,
        network,
        datagetter,
        alpha_test,
        beta_test,
        early_stopping=None,
        gpu=True,
        optimizer="SGD",
        max_epochs=20,
        start_valid=0.90,
        force_nostop=False,
    ):
        # maximum number of epochs
        self.max_epoch = max_epochs
        self.start_valid = start_valid

        self.network_type = network

        self.datagetter = datagetter
        self.alpha_test = alpha_test
        self.beta_test = beta_test

        # Sets up Gpu use
        if gpu:
            if isinstance(gpu, str):
                self.gpu = gpu
                self.device = gpu
            else:
                self.gpu = gpu
                self.device = "cuda"
        else:
            self.gpu = gpu
            self.device = "cpu"

        self.test_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        # early stopping
        self.stopped = False
        self.early_stopping = early_stopping

        self.img_spikes = {"outpt": []}

        self.optimizer = optimizer

        self.nonspiking_count = 0

        # FORCE NON STOPPING DESPITE EARLY STOPPING
        self.force_nostop = force_nostop
        self.fake_stop = False

    def data_split(self, frames):
        (
            self.train_dataset,
            self.test_dataset,
            self.valid_dataset,
            encode,
            input_features,
            input_shape,
            classes,
            time_step,
        ) = self.datagetter.get_data(frames)

        self.input_features = input_features
        self.input_shape = input_shape
        self.time = frames

        if self.train_size < 0.99:
            train_elem = range(int(len(self.train_dataset) * self.train_size))
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_elem)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size),
            pin_memory=self.gpu,
            num_workers=2,
            worker_init_fn=seed_worker,
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=int(self.batch_size),
            pin_memory=self.gpu,
            num_workers=2,
            worker_init_fn=seed_worker,
        )
        if self.valid_dataset:
            valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=int(self.batch_size),
                pin_memory=self.gpu,
                num_workers=2,
                worker_init_fn=seed_worker,
            )
        else:
            valid_dataloader = None
            self.start_valid = float("inf")

        # Neuron assignments
        self.n_classes = classes

        self.train_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")
        self.test_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")
        self.valid_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")

        return train_dataloader, test_dataloader, valid_dataloader

    def accuracy(self, network, train_ll, train_la, test_ll, test_la, valid_l, valid_a):
        if self.error:
            d = {
                "train": 0.0,
                "test": 0.0,
                "valid": 0.0,
                "train_loss": 0.0,
                "test_loss": 0.0,
                "train_time": 1e6,
                "test_time": 1e6,
                "total_time": 1e6,
                "approximate_cost": 1e6,
                "stopped": True,
                "fakestopped": True,
                "processed_images_train": 0,
                "processed_images_test": 0,
                "nonspikingcount": 0,
                "nonspikingpercent": 100,
                "error": 1,
            }
        else:
            # Approximate total training time
            if self.stopped:
                time_per_image = self.train_time / network.computed_images_train
                train_numel = int(len(self.train_dataset) * self.train_size)
                train_time = time_per_image * train_numel * self.n_epochs

                time_per_image = self.test_time / network.computed_images_test
                test_numel = len(self.test_dataset)
                test_time = time_per_image * test_numel * self.n_epochs

                total_time = train_time + test_time
            # Use real time
            else:
                total_time = self.train_time + self.test_time

            d = {
                "train": train_la[-1],
                "test": test_la[-1],
                "valid": 0.0,
                "train_loss": train_ll[-1],
                "test_loss": test_ll[-1],
                "train_time": self.train_time,
                "test_time": self.test_time,
                "total_time": self.train_time + self.test_time,
                "approximate_cost": total_time,
                "stopped": self.stopped,
                "fakestopped": self.fake_stop,
                "processed_images_train": int(network.computed_images_train),
                "processed_images_test": int(network.computed_images_test),
                "nonspikingcount": int(self.nonspiking_count),
                "nonspikingpercent": float(
                    self.nonspiking_count / len(self.test_dataset) - self.beta_test
                ),
                "error": 0,
            }

            if self.error:
                d["valid"] = 0.0
            elif not self.stopped and test_la[-1] > self.start_valid:
                d["valid"] = valid_a

        if d["test"] is None:
            d["test"] = 0.0

        for layer in network.recorders_train:
            try:
                d[f"train_{layer}_otspikes"] = int(network.recorders_train[layer])
            except:
                d[f"train_{layer}_otspikes"] = -1

            try:
                d[f"test_{layer}_otspikes"] = int(network.recorders_test[layer])
            except:
                d[f"test_{layer}_otspikes"] = -1

        for e in range(self.max_epoch):
            if e < len(train_la):
                d[f"train_accuracy_e{e}"] = train_la[e]
            else:
                d[f"train_accuracy_e{e}"] = None

            if e < len(test_la):
                d[f"test_accuracy_e{e}"] = test_la[e]
            else:
                d[f"test_accuracy_e{e}"] = None

            if e < len(train_ll):
                d[f"train_loss_e{e}"] = train_ll[e]
            else:
                d[f"train_loss_e{e}"] = None

            if e < len(test_ll):
                d[f"test_loss_e{e}"] = test_ll[e]
            else:
                d[f"test_loss_e{e}"] = None

        for c in range(self.n_classes):
            d[f"train_ot_spikes_class{c}"] = (
                self.train_outpt_spike_per_class[c].cpu().item()
            )
            d[f"test_ot_spikes_class{c}"] = (
                self.test_outpt_spike_per_class[c].cpu().item()
            )
            if self.error:
                d[f"valid_ot_spikes_class{c}"] = 0
            else:
                if not self.stopped and test_la[-1] > self.start_valid:
                    d[f"valid_ot_spikes_class{c}"] = (
                        self.valid_outpt_spike_per_class[c].cpu().item()
                    )
                else:
                    d[f"valid_ot_spikes_class{c}"] = 0

        if self.early_stopping:
            constraints = self.early_stopping.to_zeroinequality()
            # print(f"CONSTRAINTS:  {constraints}")
            if isinstance(constraints, float):
                if self.error:
                    d["constraint_0"] = 100
                else:
                    d["constraint_0"] = constraints
            else:
                for i in range(len(constraints)):
                    if self.error:
                        d[f"constraint_{i}"] = 100
                    else:
                        d[f"constraint_{i}"] = constraints[i]

        return d

    def reset(self):
        self.processed_images = 0
        self.test_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        self.train_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )
        self.test_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )
        self.valid_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )

        # early stopping
        self.stopped = False
        self.error = False
        self.fake_stop = False

        self.img_spikes = {"outpt": []}

        if self.early_stopping:
            self.early_stopping.reset()

    def __call__(self, *args, **kwargs):
        torch.cuda.empty_cache()

        train_ll = []
        train_la = []
        test_ll = []
        test_la = []

        print(f"Evaluating: {args}, {kwargs}")

        self.n_epochs = kwargs.pop("epochs", 1)
        assert (
            self.n_epochs <= self.max_epoch
        ), f"Too high number of epochs, {self.n_epochs}<={self.max_epoch}"

        self.time = kwargs.pop("frames", self.datagetter.frames)
        self.batch_size = kwargs.pop("batch_size", 16)
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.train_size = kwargs.pop("train_size", 1.0)
        self.gamma = kwargs.pop("gamma", None)
        if self.optimizer == "SGD":
            self.mu = kwargs.pop("mu", 0.9)
        else:
            self.beta_1 = kwargs.pop("beta_1", 0.9)
            self.beta_2 = kwargs.pop("beta_2", 0.999)

        self.weight_decay = kwargs.pop("weight_decay", 0)

        if self.weight_decay < 5e-8:
            self.weight_decay = 0

        # LOAD DATA
        train_dataloader, test_dataloader, valid_dataloader = self.data_split(self.time)

        self.reset()

        print(
            "TRAINING : ",
            len(self.train_dataset),
            self.train_size,
            self.n_epochs,
            self.batch_size,
            self.time,
        )
        print(f"TRAINING on {self.device}")

        self.train_time = 0
        self.test_time = 0

        network = self.network_type(
            n_inpt=self.input_features,
            n_classes=self.n_classes,
            inpt_shape=self.input_shape,
            **kwargs,
        )

        functional.set_step_mode(network, "m")

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                network.parameters(),
                lr=self.learning_rate,
                momentum=self.mu,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.weight_decay,
            )

        if self.gamma:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.gamma
            )

        # Network to GPU
        if self.gpu:
            print("Device:", self.device)
            network.to(self.device)

        try:
            print("STARTING TRAINING PHASE")
            for epoch in range(self.n_epochs):
                if self.early_stopping:
                    self.early_stopping.reset()

                start = t()
                network._train_mode = True
                network.train()
                train_samples = 0
                train_loss = 0
                train_acc = 0
                for i, batch in enumerate(train_dataloader):  # training loop
                    optimizer.zero_grad()
                    inputs = torch.movedim(batch[0].float().to(self.device), 0, 1)
                    label = batch[1].to(device=self.device)
                    label_onehot = F.one_hot(label, self.n_classes).float()

                    votes, output = network(inputs)
                    votes = votes.mean(0)

                    loss = F.mse_loss(votes, label_onehot)
                    loss.backward()
                    optimizer.step()

                    train_samples += label.numel()
                    train_loss += loss.item() * label.numel()
                    train_acc += (votes.argmax(1) == label).float().sum().item()

                    dims = (0, *range(2, len(output.shape)))
                    ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")

                    self.train_outpt_spike_per_class[label.cpu()] += ssum
                    self.img_spikes["outpt"].extend(list(ssum))

                    functional.reset_net(network)

                    if i % 50 == 0:
                        print(
                            f"""
                            [Epoch {epoch:2d}/{self.n_epochs}: batch {i}/{len(train_dataloader)}]\n
                            Loss {train_loss/train_samples}\n
                            Accu {train_acc/train_samples}\n
                            Computed images: {network.computed_images_train}\n
                            Spikes train: {network.recorders_train}\n
                            Spikes test: {network.recorders_test}\n
                            IMG SPIKES: {len(self.img_spikes['outpt'])}\n
                            """
                        )

                    if self.early_stopping and self.early_stopping(self, network):
                        if self.force_nostop:
                            self.stopped = False
                            self.fake_stop = True
                        else:
                            self.stopped = True
                            break

                    self.img_spikes = {"outpt": []}

                train_loss /= train_samples
                train_acc /= train_samples
                train_ll.append(train_loss)
                train_la.append(train_acc)

                self.train_time += t() - start

                if self.gamma:
                    scheduler.step()

                self.img_spikes = {"outpt": []}

                network._train_mode = False
                network.eval()

                start = t()
                self.nonspiking_count = 0
                test_loss = 0
                test_acc = 0
                test_samples = 0
                with torch.no_grad():
                    for i, batch in enumerate(test_dataloader):  # training loop
                        inputs = torch.movedim(batch[0].float().to(self.device), 0, 1)
                        label = batch[1].to(device=self.device)
                        label_onehot = F.one_hot(label, self.n_classes).float()

                        votes, output = network(inputs)
                        votes = votes.mean(0)

                        loss = F.mse_loss(votes, label_onehot)

                        test_samples += label.numel()
                        test_loss += loss.item() * label.numel()
                        test_acc += (votes.argmax(1) == label).float().sum().item()

                        dims = (0, *range(2, len(output.shape)))
                        ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")

                        self.test_outpt_spike_per_class[label.cpu()] += ssum
                        self.nonspiking_count += (ssum < self.alpha_test).sum()

                        functional.reset_net(network)

                self.test_time += t() - start
                test_loss /= test_samples
                test_acc /= test_samples
                test_ll.append(test_loss)
                test_la.append(test_acc)

                if self.stopped:
                    if self.force_nostop:
                        self.stopped = False
                        self.fake_stop = True
                    else:
                        self.stopped = True
                        break

            print(f"END TRAINING on {self.device}")
        except Exception as e:
            print("ERROR OCCURED")
            print(traceback.format_exc(), file=sys.stderr)
            self.error = True

        valid_loss = 0
        valid_acc = 0
        valid_samples = 0
        if (
            not self.error
            and not self.stopped
            and test_la[-1] > self.start_valid
            and valid_dataloader
        ):
            print("DOING VALIDATION")
            # Network to GPU
            if self.gpu:
                print("Device:", self.device)
                network.to(self.device)

            network._train_mode = False
            network.eval()

            start = t()
            with torch.no_grad():
                for i, batch in enumerate(valid_dataloader):  # valid loop
                    inputs = torch.movedim(batch[0].float().to(self.device), 0, 1)
                    label = batch[1].to(device=self.device)
                    label_onehot = F.one_hot(label, self.n_classes).float()

                    votes, output = network(inputs)
                    votes = votes.mean(0)

                    loss = F.mse_loss(votes, label_onehot)

                    valid_samples += label.numel()
                    valid_loss += loss.item() * label.numel()
                    valid_acc += (votes.argmax(1) == label).float().sum().item()

                    dims = (0, *range(2, len(output.shape)))
                    ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")
                    self.valid_outpt_spike_per_class[label.cpu()] += ssum

                    functional.reset_net(network)

                    print(f"""[VALIDATION: batch {i}/{len(valid_dataloader)}]\n""")

            self.valid_time += t() - start
            valid_loss /= valid_samples
            valid_acc /= valid_samples

            print("ENDING VALIDATION")

        results = self.accuracy(
            network, train_ll, train_la, test_ll, test_la, valid_loss, valid_acc
        )
        nparameters = 0
        train_nparameters = 0
        train_zeronparameters = 0
        train_onenparameters = 0
        train_95nparameters = 0
        train_05nparameters = 0

        try:
            for p in network.parameters():
                numel = p.numel()
                nparameters += numel
                if p.requires_grad:
                    train_nparameters += numel
                    train_zeronparameters += (
                        (p == 1.0).sum().to(dtype=int, device="cpu").item()
                    )
                    train_onenparameters += (
                        (p == 0.0).sum().to(dtype=int, device="cpu").item()
                    )
                    train_95nparameters += (
                        (p > 0.95).sum().to(dtype=int, device="cpu").item()
                    )
                    train_05nparameters += (
                        (p < 0.05).sum().to(dtype=int, device="cpu").item()
                    )
        except:
            pass

        results["parameters"] = nparameters
        results["trainable"] = train_nparameters
        results["trainable_zeros"] = train_zeronparameters
        results["trainable_ones"] = train_onenparameters
        results["trainable_95"] = train_95nparameters
        results["trainable_05"] = train_05nparameters

        if self.gpu:
            mem_stat = torch.cuda.memory_stats()

            results["active_bytes.all.allocated"] = mem_stat[
                "active_bytes.all.allocated"
            ]
            results["active_bytes.all.peak"] = mem_stat["active_bytes.all.peak"]
            results["allocated_bytes.all.allocated"] = mem_stat[
                "allocated_bytes.all.allocated"
            ]
            results["allocated_bytes.all.peak"] = mem_stat["allocated_bytes.all.peak"]

        print(f"RETURNING")
        return results
