#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:49:46 2017

@author: William Berdanier
"""

import numpy as np

import scipy.sparse as sp
import numpy.linalg as la
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.linalg as sa
import numpy.random as ra
import random
from scipy.linalg import expm,logm
import sys
import math
import itertools
import mk_qm as mk

###############################################################################
# This code contains helper methods for diagonalizing the periodically driven XX model in the fermionic language.
###############################################################################

def get_onebody_eigs(F,l1='',l2='',L=''): # calculates many body eigenvalues from single body Floquet operator
    assert np.abs(np.max((F.dot(F.conj().T) - np.identity(F.shape[0])).flatten())) < 1e-5 # assert unitarity
    F = np.array(F)
#    (w,v) = la.eig(F)
#    (w,v) = sa.eig(F)
    tol = 1e-14
    (w,v) = mk.eigu(F,tol)
    Lpi = w.shape[0]
    QEs_raw = np.real(-1j * np.log(w))
    QEs = np.sort(QEs_raw)

    plt.figure()
    plt.plot(QEs/np.pi,'ko')
    plt.ylabel('Quasienergy / $\pi$')
    plt.xlabel('Index (meaningless)')
    plt.ylim([-1.1,1.1])
    plt.title('Eigenvalues of the single-body Floquet operator for pi chain with $L=$'+str(Lpi)+'\n $(l_1,l_2) = $('+str(l1)+','+str(l2)+')')
    return QEs_raw,v

def get_entanglement_entropy(P,l):
    if P.shape[0] == 0 or l == 0:
        return 0.
    C = np.tensordot(P[:l,:],P.conj()[:l,:],axes=[1,1])
#    wC = la.eigvalsh(C) # only computes eigenvalues
    wC = sa.eigvalsh(C)
    half = wC.shape[0] / 2

#    # method 1: sum over all eigenvalues
#    S = 0
#    for l in wC:
#        if (l>0.0000000001) and (l<1-0.0000000001):
#            S -= (1.-l) * np.log(1.-l) + l * np.log(l)
#    S = np.real(S)

    # method 2: sum only over half
    Sa = 0
    for l in wC[:half]:
        if (l>0.0000000001) and (l<1-0.0000000001):
            Sa -= (1.-l) * np.log(1.-l) + l * np.log(l)
    Sa = 2 * np.real(Sa)
    Sb = 0
    for l in wC[half:]:
        if (l>0.0000000001) and (l<1-0.0000000001):
            Sb -= (1.-l) * np.log(1.-l) + l * np.log(l)
    Sb = 2 * np.real(Sb)
    S = max(Sa,Sb)

    return S

def get_rand_couplings(w,L):
    # returns exponential distribution with strength w on length L, interval [0,1]. w=1 is a box
    # the logs have mean -w, the couplings have mean exp(-w).
    log_couplings = np.array([-random.expovariate(1./w) for i in range(0,L)])
    couplings = np.exp(log_couplings)

    # now give them a random sign
    couplings = np.array([couplings[i] * (-1) ** random.sample([0,1],1)[0] for i in range(couplings.shape[0])])
    return couplings

def get_HF(F):
    # numpy routine, diagonalize and do log by hand
#    (wF,vF) = sa.eig(F) # F is not hermitian
    tol = 1e-14
    (wF,vF) = mk.eigu(F,tol)
    w = np.real(-1j * np.log(wF))
    v = vF
    HF = v.dot(np.diag(w)).dot(v.conj().T)

#    w,v = sa.eigh(HF) # this can be turned off

    idx = w.argsort() # sort the eigenvalues to fill the fermi sea
    w = w[idx]
    v = v[:,idx]

#    assert np.abs(np.max(HF - HF.conj().T).flatten()) < 1e-5 # ensure HF is hermitian

#    # debug
#    HF = F
#    (w,v) = la.eigh(HF)

    return (wF,w,v,HF)

def symm_sample(L,N):
    assert L%2==0
    inds = range(L)
    out = []

    for j in range(N):
        r = random.sample(inds,1)[0]
        inds.remove(r)
        inds.remove(L-1-r)
        out.append(r)
    return out

def sprinkle(couplings,N):
    # sprinkles in N pi couplings to the chain. Replaces x by 1-x
    L = couplings.shape[0]
    for j in random.sample(range(L),N):
        couplings[j] = 1. - couplings[j]
    return couplings

def get_F(hs,Js):
    L = 2 * hs.shape[0]
    assert L >= 2 # requirement of model

    # NOTE: this code uses PBCs.

    ## debug
#    epsilon_max = 0.5 * 0
#    delta_max = epsilon_max
#    w_epsilon = 1.
#    w_delta = w_epsilon
#    n_pi_delta = 1/2. * 0
#    n_pi_epsilon = n_pi_delta
#
##    (l1,l2) = (1/np.pi, 1/np.pi)
#
#    (l1, l2) = (0.2,0.2)
#
#    L = 4
#    ###
#

#
#    deltas = delta_max * get_rand_couplings(w_delta,L / 2)
#    epsilons = epsilon_max * get_rand_couplings(w_epsilon,L / 2)
#
##    # See Khemani PRL for the phase diagram for pi modes
##    # coordinates in the phase diagram, user input (l1,l2)
##    # l1, l2 in [0,1], 0 identified with 1
#    hs = l1 * np.ones(L / 2) + deltas
##    print hs
##    Js = l2 * np.ones(L-1) + epsilons
#    Js = l2 * np.ones(L / 2) + epsilons
##    print Js
#
##    hsp = hs
##    Jsp = Js
#
#    hs = sprinkle(hs,int(n_pi_delta * deltas.shape[0]))
#    Js = sprinkle(Js,int(n_pi_epsilon * epsilons.shape[0]))

#    print max(hsp-hs)
#    print max(Jsp-Js)

#    print 'c'

########################################
#    hs_mat = np.loadtxt("2hs.out")
#    Js_mat = np.loadtxt("2Js.out")
#
#    index = DELETE_ME
#    hs = hs_mat[index,:]
#    Js = Js_mat[index,:]
########################################

    H1 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(0,L-1,2):
        H1[i,i+1] = -np.pi * hs[i/2]
        H1[i+1,i] = -H1[i,i+1]
    H1 = H1.todense()
    (w1,v1) = sa.eigh(H1)
#    print H1

#    print 'e'

    H2 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(1,L-1,2):
        H2[i,i+1] = -np.pi * Js[(i-1)/2]
        H2[i+1,i] = -H2[i,i+1]

    H2[0,L-1] = -np.pi * Js[-1] # PBCs
    H2[L-1,0] = H2[0,L-1]


    H2 = H2.todense()
    (w2,v2) = sa.eigh(H2)
#    print H2

###     debug
#    H = H1 + H2
#    (w,v) = la.eigh(H)

### debug
#    H2 = H1 + H2
#    (w2,v2) = la.eigh(H2)

#    print 'f'

    U1 = v1.dot(np.diag(np.exp(-1j * w1))).dot(v1.T)
    U2 = v2.dot(np.diag(np.exp(-1j * w2))).dot(v2.T)
    F = U2.dot(U1)

#    print 'g'

####    debug
#    return H2
#    return U2

    return F

def get_roll(w):
    ell = w.shape[0]
    devs = []
    rs = range(-20,21)
    for r in rs:
        w_r = np.roll(w,r)
        dev = np.mean(np.abs(w_r - w_r[::-1].conj()))
        devs.append(dev)
    r_best_ind = np.argmin(devs)
    r_best = rs[r_best_ind]
    return r_best,devs[r_best_ind]

def get_F_openBCs(hs,Js):
    L = 2 * hs.shape[0]
    assert L >= 2 # requirement of model

    # NOTE: this code uses open BCs.

    ## debug
#    epsilon_max = 0.5 * 0
#    delta_max = epsilon_max
#    w_epsilon = 1.
#    w_delta = w_epsilon
#    n_pi_delta = 1/2. * 0
#    n_pi_epsilon = n_pi_delta
#
##    (l1,l2) = (1/np.pi, 1/np.pi)
#
#    (l1, l2) = (0.2,0.2)
#
#    L = 4
#    ###
#

#
#    deltas = delta_max * get_rand_couplings(w_delta,L)
##    epsilons = epsilon_max * get_rand_couplings(w_epsilon,L - 1)
#    epsilons = epsilon_max * get_rand_couplings(w_epsilon,L)
#
##    # See Khemani PRL for the phase diagram for pi modes
##    # coordinates in the phase diagram, user input (l1,l2)
##    # l1, l2 in [0,1], 0 identified with 1
#    hs = l1 * np.ones(L) + deltas
##    print hs
##    Js = l2 * np.ones(L-1) + epsilons
#    Js = l2 * np.ones(L) + epsilons
##    print Js
#
##    hsp = hs
##    Jsp = Js
#
#    hs = sprinkle(hs,int(n_pi_delta * deltas.shape[0]))
#    Js = sprinkle(Js,int(n_pi_epsilon * epsilons.shape[0]))
#
#    print max(hsp-hs)
#    print max(Jsp-Js)

#    print 'c'

    H1 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(0,L-1,2):
        H1[i,i+1] = -np.pi * hs[i/2]
        H1[i+1,i] = -H1[i,i+1]
    H1 = H1.todense()
    (w1,v1) = sa.eigh(H1)
#    print H1

#    print 'e'

    H2 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(1,L-1,2):
        H2[i,i+1] = -np.pi * Js[i/2]
        H2[i+1,i] = -H2[i,i+1]


    H2 = H2.todense()
    (w2,v2) = sa.eigh(H2)
#    print H2

###     debug
#    H = H1 + H2
#    (w,v) = la.eigh(H)

### debug
#    H2 = H1 + H2
#    (w2,v2) = la.eigh(H2)

#    print 'f'

    U1 = v1.dot(np.diag(np.exp(-1j * w1))).dot(v1.T)
    U2 = v2.dot(np.diag(np.exp(-1j * w2))).dot(v2.T)
    F = U2.dot(U1)

#    print 'g'

####    debug
#    return H2
#    return U2

    return F

def lay_dws_regularly(couplings,l_domains):
    if_pi = False # are we laying 0 bricks or pi bricks?
    if l_domains == 0:
        return couplings

    for i in range(couplings.shape[0]):
        if (i + 1) % l_domains == 0:
            if_pi = not if_pi

        if if_pi:
            couplings[i] = 1. - couplings[i]
    return couplings

def lay_dws_asymmetrically(couplings,l_pi_domains,l_0_domains):
    if_pi = False # are we laying 0 bricks or pi bricks?
    if l_domains == 0:
        return couplings

    for i in range(couplings.shape[0]):
        if (i + 1) % l_domains == 0:
            if_pi = not if_pi

        if if_pi:
            couplings[i] = 1. - couplings[i]
    return couplings

def lay_dws_randomly(couplings,n_dws):
    if_pi = False # are we laying 0 bricks or pi bricks?alrea
    if n_dws == 0:
        return couplings

    Lc = couplings.shape[0]
    N_dws = int(n_dws * Lc)
    dws = random.sample(range(Lc),N_dws)

    for i in range(Lc):
        if i in dws:
            if_pi = not if_pi

        if if_pi:
            couplings[i] = 1. - couplings[i]
    return couplings

def get_F_dws(l1,l2,L,epsilon_max=0.1,delta_max=0.1,w_epsilon=1,w_delta=1,n_dws=0):
    assert L >= 2 # requirement of model

    # NOTE: this code uses PBCs.

##    ## debug
#    epsilon_max = 0.5 * 0
#    delta_max = epsilon_max
#    w_epsilon = 1.
#    w_delta = w_epsilon
#    n_dws = 0 # dw density
#
##    (l1,l2) = (1/np.pi, 1/np.pi)
#
#    (l1, l2) = (0.2,0.2)
#
#    L = 4
##    ###

    ### non-user paramters
    T1 = 1.
    T2 = 1.
    T = 2.
    ###
#
    ### L = 2, clean
    #epsilons = np.array([-0.2])
    #deltas = np.array([-0.2,-0.2])

##    ## L = 2, clean, this works
#    epsilons = np.array([-0.2])
#    deltas = np.array([1.2,1.2])

#    ### L = 2, clean, this is broken
#    epsilons = np.array([0.])
#    deltas = np.array([0.,1.2])

    ## L = 2, dirty
#    epsilons = np.array([-0.2])
#    deltas = np.array([-0.82,-0.75])

#    ### L = 2, broken
#    epsilons = np.array([-0.07206952])
#    deltas = np.array([1+0.16086343,-0.13013709])

    ### L = 3
    #epsilons = np.array([-0.08466438,  0.02838463])
    #deltas = np.array([ 0.07229095,  0.1442309 , -0.02887752])

    ## L = 4
#    epsilons = np.array([ 0.16088299, -0.03123118, -0.15213832])
#    deltas = np.array([ 0.17320622, -0.148111  ,  0.18086554, -0.16409297])

#    # L = 4, pis, broken
#    epsilons = np.array([ 1. - 0.16088299, -0.03123118, - 0.15213832])
#    deltas = np.array([ 1 - 0.17320622, 1. + 0.148111  ,  1. - 0.18086554, -0.16409297])

#    ## L = 6
#    epsilons = np.array([-0.1871197 ,  0.07842424,  0.13041357,  0.14470927, -0.04501188])
#    deltas = np.array([ 0.09934838, -0.09175597, -0.12866234, -0.1026317 , -0.17337427, 0.00892681])

    ## L = 6, pis
#    epsilons = 1. - np.array([-0.1871197 ,  0.07842424,  0.13041357,  0.14470927, -0.04501188])
#    deltas = 1. - np.array([ 0.09934838, -0.09175597, -0.12866234, -0.1026317 , -0.17337427, 0.00892681])


#    ### w = 3., L = 6, pis, broken
#    epsilons = np.array([ 1 - 0.00540518, -0.19833121,  0.00994274,  1 - 0.0022542 , -0.10338123])
#    deltas = np.array([1 + 0.00011637, -0.02354735, 1 + 0.04832932,  1 - 0.0005081 ,  1 - 0.0282347 ,-0.00367225])


#    ## w = 3., L = 6, pis - broken
#    epsilons = 1. - np.array([ 1 - 0.00540518, -0.19833121,  0.00994274,  1 - 0.0022542 , -0.10338123, 0.1])
#    deltas = 1. - np.array([1 + 0.00011637, -0.02354735, 1 + 0.04832932,  1 - 0.0005081 ,  1 - 0.0282347 ,-0.00367225])




#    # See Khemani PRL for the phase diagram for pi modes
#    # coordinates in the phase diagram, user input (l1,l2)
#    # l1, l2 in [0,1], 0 identified with 1


    hs = l1 * np.ones(L/2) + deltas
#    Js = l2 * np.ones(L/2 - 1) + epsilons
    Js = l2 * np.ones(L/2) + epsilons

#    hsp = hs
#    Jsp = Js
    if n_dws != 0:
        l_dom = 1./n_dws
        hs = lay_dws_regularly(hs,l_dom)
        Js = lay_dws_regularly(Js,l_dom)

#        hs = lay_dws_randomly(hs,n_dws)
#        Js = lay_dws_randomly(Js,n_dws)

#    print max(hsp-hs)
#    print max(Jsp-Js)

#    print 'c'

    H1 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(0,L-1,2):
#        H1[i,i+1] = -np.pi * hs[i] * 1j # pure imaginary. This isn't working?
#        H1[i+1,i] = -H1[i,i+1]

        H1[i,i+1] = -np.pi * hs[i/2]
        H1[i+1,i] = H1[i,i+1]
    H1 = H1.todense()
    (w1,v1) = sa.eigh(H1)
#    print H1

#    print 'e'

    H2 = sp.dok_matrix((L,L), dtype='complex')
    for i in range(1,L-1,2):
#        H2[i,i+1] = -np.pi * Js[i] * 1j # pure imaginary. This isn't working?
#        H2[i+1,i] = -H2[i,i+1]

        H2[i,i+1] = -np.pi * Js[i/2]
        H2[i+1,i] = H2[i,i+1]

    H2[0,L-1] = -np.pi * Js[-1] # PBCs
    H2[L-1,0] = H2[0,L-1]


    H2 = H2.todense()
    (w2,v2) = sa.eigh(H2)
#    print H2

###     debug
#    H = H1 + H2
#    (w,v) = la.eigh(H)

### debug
#    H2 = H1 + H2
#    (w2,v2) = la.eigh(H2)

#    print 'f'

    U1 = v1.dot(np.diag(np.exp(-1j * T1 * w1))).dot(v1.T)
    U2 = v2.dot(np.diag(np.exp(-1j * T2 * w2))).dot(v2.T)
    F = U2.dot(U1)

#    print 'g'

####    debug
#    return H2
#    return U2

    return F

def single_to_many_quasi(spectrum1):
    cmb = []
    spectrum = spectrum1
    for x in range(1,spectrum.shape[0] + 1):
        cmb += itertools.combinations(spectrum, x) # calculate all fillings
    mbspectrum = [(sum(i) + np.pi) % (2 * np.pi) - np.pi for i in cmb] # sum the fillings to get E. Ensuring in the window [-pi,pi)
    mbspectrum.append(0.) # all empty, needs to be manually added
    cmb.append(())
    mbspectrum = np.array(mbspectrum)
    return np.array(mbspectrum),cmb

def print_arr(array):
    for j in range(array.shape[0]):
        print array[j]

def print_arrs(array1,array2):
    for j in range(array1.shape[0]):
        print array1[j], array2[j]

def mod(arr,m): #performs mod into window [-m/2, m/2)
    for j in range(arr.shape[0]):
        arr[j] = ((arr[j] + m/2.) % m) - m/2.
    return arr
