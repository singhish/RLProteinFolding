# RLProteinFolding

Protein folding driven by reinforcement learning. Powered by PyTorch and BioPython.

## Introduction
The goal of this project is to successfully fold a protein using reinforcement learning techniques. For
now, this project's scope lies within folding a
[small protein consisting of 20 amino acid residues](https://www.rcsb.org/structure/2jof). Due to its
continuous action space, the Deep Deterministic Policy Gradient (DDPG)
algorithm as described by Lillicrap et al. in the paper 
[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) is being
implemented in this project atop PyTorch to take on the "protein folding problem." For now, much of the
code here is based off of Vikas Yadav's implementation of this paper (see his repo
[here](https://github.com/vy007vikas/PyTorch-ActorCriticRL)).

This Readme will be updated as this projected progresses. More to come soon!

## Getting Started
Prerequisites:
- Anaconda/Miniconda 4.8.4
- A modified installation of [REDCRAFT](https://redcraft.readthedocs.io/en/latest/) must be present on
your system. Instructions to do this will be added in the near future, but they involve the
[molan.cpp](molan.cpp) file included in this repo.

Instructions:
1. Install REDCRAFT (instructions coming soon).
2. Clone this repository.
3. Within this repository's directory, run
```bash
conda env create -f environment.yml
```
