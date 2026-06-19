# Parallel hyperparameter optimization of spiking neural networks

> [Back to main](https://github.com/ThomasFirmin/mythesis)

This repository contains hyperlinks to redirect the reader to the source code of each chapter from the thesis [Parallel hyperparameter optimization of spiking neural networks](https://theses.fr/s327519).

The thesis is accessible at (_available once published_):
* [HAL]()

## Chapter 6 - Accelerating Hyperparameter Optimization with Multi-Fidelity

Chapter 6 is the enhancement of chapter 5. It leverages silent networks, early stopping, and blackbox constraints, combined with a generalization of multi-fidelity ayesian optimization known as cost-aware optimization. We empirically illustrates that we can reduce by 7 the budget from chapter 5, necessary to convergence. We optimized up-to 46 hyperparameters on the MNIST, NMNIST, SHD datasets, using [LAVA-DL](https://lava-nc.org/) and [SpikingJellyy](https://spikingjelly.readthedocs.io/zh-cn/latest/#).

## Content

* The file `search_spaces.py` contains all the search spaces for chapters 5 and 6.
* The file `load_dataset.py` contains dataloaders and encoders for different datasets.
* The folder `scbo` contains scripts of the experiments with the SCBO algorithm.
* The folder `turbo` contains scripts of the experiments with the TuRBO algorithm.
* The folder `sensitivity` contains scripts for the early stopping sensitivity analysis.

## Zellij

First install the frozen thesis version of **Zellij**.

Zellij is the main Python package made for this thesis, including both FBD and HPO algorithms.
The actual version of the thesis was frozen. The documentation is not up-to-date.

> [Github to Zellij](https://github.com/ThomasFirmin/zellij/tree/thesis_freeze)

## Simulators

Install local packages to instantiates the networks, and early stopping with [BindsNET](https://bindsnet-docs.readthedocs.io/), [LAVA-DL](https://lava-nc.org/), and [SpikingJellyy](https://spikingjelly.readthedocs.io/zh-cn/latest/#).

> BindsNET
```
$ python3 -m pip install -e ./code/Lie
```

> LAVA-DL
```
$ python3 -m pip install -e ./code/Lave
```

> SpikingJelly
```
$ python3 -m pip install -e ./code/SpikingJelly
```

## Run 

The experiments were designed for the [Grid'5000](http://www.idris.fr/jean-zay/jean-zay-presentation.html) using [OAR](https://oar.readthedocs.io/en/2.5/) and [OpenMPI](https://www.open-mpi.org/), on a distributed GPU partition.
The `HOSTFILE` contains the addresses of the nodes. The `RANKFILE` contains the mapping between processes and nodes, the processes 0 is dedicated to the master process running the optimization algorithm on a dedicated GPU.

```
$ mpiexec --mca plm_rsh_agent "oarsh" --mca pml ^ucx --mca mtl ^psm2,ofi --mca btl ^ofi,openib -machinefile <HOSTFILE> --rankfile <RANKFILE> --mca mpi_yield_when_idle 1 -np <NUMBER OF PROCESSES> python3 <PATH TO SCRIPT> --dataset <DATASET> --mpi flexible --record_time --gpu --gpu_per_node 4 --time <TIME BUDGET IN SECONDS> 1> log.out 2> log.err
```

> Example
```
$ mpiexec --mca plm_rsh_agent "oarsh" --mca pml ^ucx --mca mtl ^psm2,ofi --mca btl ^ofi,openib -machinefile ../host --rankfile ../rankfile --mca mpi_yield_when_idle 1 -np 16 python3 f_launch_random_fidelity_nmnist_ext_sj.py --dataset NMNIST --mpi flexible --record_time --gpu --gpu_per_node 4 --time 144000 1> log.out 2> log.err
```

## Authors and acknowledgment
* Author: Thomas Firmin
* Supervisor: El-Ghazali Talbi
* Co-Supervisor: Pierre Boulet

Experiments presented in this work were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations (see \url{https://www.grid5000.fr}).


This work was granted access to the HPC resources of IDRIS under the allocation 2023-AD011014347 made by GENCI.


This work has been supported by the University of Lille, the ANR-20-THIA-0014 program AI\_PhD$@$Lille and the ANR PEPR AI and Numpex. It was also supported by IRCICA(CNRS and Univ. Lille USR-3380).

## License
CeCILL-C