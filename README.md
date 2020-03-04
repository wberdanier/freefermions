# freefermions
TLDR: Python and Julia code for computing entanglement and Loschmidt echo dynamics in free-fermionic models. 

## Introduction 

In general, quantum systems are difficult to simulate classically (i.e. on your laptop or even your local supercomputer). For an arbitrary quantum system consisting of ![N](https://render.githubusercontent.com/render/math?math=N) qubits, the state space is ![2^N](https://render.githubusercontent.com/render/math?math=2%5EN)-dimensional, so the Hamiltonian is a ![2^N \times 2^N](https://render.githubusercontent.com/render/math?math=2%5EN%20%5Ctimes%202%5EN) matrix with complex entries. Even if this can be stored in memory, exactly diagonalizing such a system is slow, generally scaling as ![O((2^N)^3)](https://render.githubusercontent.com/render/math?math=O((2%5EN)%5E3)). This limits most laptop ED computations to around 20 qubits, and the state-of-the-art is around 40-50 qubits. This is quite a long way from the thermodynamic limit of ![N \sim 10^{23}](https://render.githubusercontent.com/render/math?math=N%20%5Csim%2010%5E%7B23%7D). 

There are some tricks to alleviate this exponential wall in certain cases, for instance, using the locality of the Hamiltonian (and its resulting sparsity) to run sparse matrix methods, allowing for slightly higher system sizes. If the quantum state you wish to target has relatively small entanglement (for instance, the ground state of a gapped Hamiltonian), then DMRG and associated tensor network methods work well -- but are still limited to a few hundred sites. 

It turns out, however, that free-fermionic systems can be simulated classically with system sizes of order a few *thousand* sites. 

## Free-fermionic systems 

Some systems enjoy a convenient *free fermion* representation. That is, the Hamiltonian is of the form 

![H = \sum_{ij} H_{ij} c_i^\dagger c_j](https://render.githubusercontent.com/render/math?math=H%20%3D%20%5Csum_%7Bij%7D%20H_%7Bij%7D%20c_i%5E%5Cdagger%20c_j)


involving only fermionic bilinears. If this is the case, then we can use the power of Wick's theorem to represent a system of ![N](https://render.githubusercontent.com/render/math?math=N) fermions in an ![N \times N](https://render.githubusercontent.com/render/math?math=N%20%5Ctimes%20N) matrix -- exponentially smaller in size. This was explicitly shown in the works of Peschel and Eisler:

I. Peschel, J. Phys. A: Math. Gen. 36, L205 (2003), https://arxiv.org/abs/cond-mat/0212631,

V. Eisler and I. Peschel, J. Stat. Mech. 2007, P06005 (2007), https://arxiv.org/abs/cond-mat/0703379.

The technical details of this procedure can be found in above original references, or summarized in the supplemental material of the following paper: https://arxiv.org/abs/1701.05899, among other sources.

## This repo

The purpose of this repo is to host a simple implementation of free fermion numerics. It has been implemented in both python and julia. For strongly disordered systems, one must take great care with numerical precision issues (see https://arxiv.org/abs/1803.00019); also included is an arbitrary-precision implementation in C++. 

Please enjoy the power of simulating quantum systems of several thousand sites, and let me know of any bugs you encounter.

Thanks to my collaborators, in particular Michael Kolodrubetz, for their help in developing this codebase.
