#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:31:46 2018

@author: will
"""

import numpy as np
import scipy.sparse as sp
import numpy.linalg as la
import os
import numpy.random as ra

def get_H(hb,J,L_TFI):
    # Returns an nparray Hamiltonian for the XX chain with a given bulk hopping J
    # and boundary hopping hb (units of J). The length of the chain is L=2*L_TFI+1;
    # this uses the mapping of XX = 2 decoupled copies of TFI, and 1 extra site for the boundary field.

    L = 2*L_TFI + 1

    H = sp.dok_matrix((L,L), dtype='d')
    for i in range(0,L-1):
        H[i,i+1] = -J / 2.
        H[i+1,i] = H[i,i+1]
    H[0,1] = -hb * J / 2.
    H[1,0] = H[0,1]
    return H.todense()

def diag_H(H):
    # Returns sorted eigenvalues/vectors of a Hermitian nparray H.
    (wa,va) = np.linalg.eigh(H)
    idx = wa.argsort()
    wa = wa[idx]
    va = va[:,idx]
    return (wa,va)

def get_U(H,dt):
    # Returns time evolution operator for evolving under hamiltonian H for a time dt.
    (w,v) = diag_H(H)
    U = v.dot(np.diag(np.exp(-1j * dt * w))).dot(v.T)
    return U

def get_echo(P0,Pt):
    # Returns the Loschmidt echo (fidelity) between a time-evolved state Pt and initial state P0.
    return np.abs(la.det(P0.conj().T.dot(Pt)))**2

def get_diagonal(M,j=0):
    # Gets diagonal slice of a matrix M along diagonal j.
    M = np.array(M)
    L = M.shape[0]
    return [M[i,i+j] for i in range(L-np.abs(j))]

def run_code_markovian(L,J,hb,prob,dt,type="fixed_fixed",print_step=0):
    ###############################################################################
    # Full method to calculate the Loschmidt echo of a critical Ising model with boundary
    # field hb that stochastically flips according to a Poisson process (Markovian).
    # Boundary field flips between (-hb,hb) (type="fixed_fixed") or (0,hb) (type="free_fixed")
    # with probability prob in time dt.
    # TFI model parameters: L=system size, J=bulk hopping (critical so g=1).
    # See PRL 123, 230604 (2019) for physics details.
    ###############################################################################

    ## testing parameters:
    #L = 500
    #J = 2. # sets speed of sound
    #hb = 0.5 # in units of J
    #prob = 0. # prob of a flip
    #dt = 0.1

    t_max = L/J
    assert t_max <= L/J

    ts = np.arange(0,t_max,dt)

    # boolean array -- true = minus, false = plus. Markovian, no memory
    hbs = [ra.choice([True,False],p=[prob,1-prob]) for time in ts]

    # # boolean array -- true = flip, false = don't flip. Non-Markovian, has memory of 1 previous time step
    # flips = [ra.choice([True,False],p=[prob,1-prob]) for time in ts]
    # if_minus = True
    ## sum(flips)/float(len(ts)) ##sanity check

    Hp = get_H(hb,J,L)
    Hm = get_H(-hb,J,L)
    Hf = get_H(0.,J,L)

    Up = get_U(Hp,dt)
    Um = get_U(Hm,dt)
    Uf = get_U(Hf,dt)

    (wp,vp) = diag_H(Hp) # start in GS of plus
    Nf = sum(1 for n in wp if n < 0)
    Pt = vp[:,0:Nf] # fill the Fermi Sea

    P0 = Pt.copy()


    echoes = [] # initialize
    for j in range(len(ts)):
        if j in [len(ts)/print_step * t for t in range(print_step)]:
            print("t = " + str(j * dt) + " of " + str(t_max))

        # if flips[j] == True:
        #     if_minus = not if_minus #do the flip

        # ## evolve with the field
        # if if_minus == True:
        #     Pt = Um.dot(Pt)
        # else:
        #     Pt = Up.dot(Pt)

        ## evolve with the field
        if hbs[j] == True:
            if type=="free_fixed":
                Pt = Uf.dot(Pt)
            elif type=="fixed_fixed":
                Pt = Um.dot(Pt)
        else:
            Pt = Up.dot(Pt)

        echo = get_echo(P0,Pt)
        echoes.append(echo)
    #print "Done!"
    echoes = np.sqrt(np.array(echoes)) # TFI echo

    return echoes

def run_code_nonmarkovian(L,J,hb,prob,dt,type="fixed_fixed",print_step=0):
    # Full method to calculate the Loschmidt echo of a critical Ising model with boundary
    # field hb that stochastically flips according to a non-Markovian process (has memory).
    # Boundary field flips between (-hb,hb) (type="fixed_fixed") or (0,hb) (type="free_fixed")
    # with probability prob in time dt.
    # TFI model parameters: L=system size, J=bulk hopping (critical so g=1).

    ### testing parameters
    #L = 500
    #J = 2. # sets speed of sound
    #hb = 0.5 # in units of J
    #prob = 0. # prob of a flip
    #dt = 0.1

    t_max = L/J
    assert t_max <= L/J

    ts = np.arange(0,t_max,dt)

    ### boolean array -- true = minus, false = plus. Markovian, no memory
    # hbs = [ra.choice([True,False],p=[prob,1-prob]) for time in ts]

    ### boolean array -- true = flip, false = don't flip. Non-Markovian, has memory of 1 previous time step
    flips = [ra.choice([True,False],p=[prob,1-prob]) for time in ts]
    if_minus = True # start with evolution under minus -- this starts with a flip
    # if_minus = False # start with evolution under plus -- does not start with a flip necessarily

    Hp = get_H(hb,J,L)
    Hm = get_H(-hb,J,L)
    Hf = get_H(0.,J,L)

    Up = get_U(Hp,dt)
    Um = get_U(Hm,dt)
    Uf = get_U(Hf,dt)

    (wp,vp) = diag_H(Hp) # start in GS of plus
    Nf = sum(1 for n in wp if n < 0)
    Pt = vp[:,0:Nf] # fill the Fermi Sea

    P0 = Pt.copy()


    echoes = [] # initialize
    for j in range(len(ts)):
        if j in [len(ts)/print_step * t for t in range(print_step)]:
            print("t = " + str(j * dt) + " of " + str(t_max))

        if flips[j] == True:
            if_minus = not if_minus #do the flip

        ## evolve with the field
        if if_minus == True:
            if type=="free_fixed":
                Pt = Uf.dot(Pt)
            elif type=="fixed_fixed":
                Pt = Um.dot(Pt)
        else:
            Pt = Up.dot(Pt)

        # ## evolve with the field
        # if hbs[j] == True:
        #     Pt = Um.dot(Pt)
        # else:
        #     Pt = Up.dot(Pt)

        echo = get_echo(P0,Pt)
        echoes.append(echo)
    #print "Done!"
    echoes = np.sqrt(np.array(echoes)) # TFI echo

    return echoes
