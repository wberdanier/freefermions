#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:37:32 2017

@author: will
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

def print_mat(M):
    for j in xrange(M.shape[0]):
        st=''
        for k in xrange(M.shape[1]):
            st += ('%.3f ' % M[j,k])
        print st

def get_diag(M,k):
    M = np.array(M)
    return np.array([M[i,i+k] for i in range(M.shape[0] - k)])

def get_onebody_eigs(F,l1=0,l2=0,L=0): # calculates many body eigenvalues from single body Floquet operator
    # assert unitarity here
    F = np.array(F)
    w,v = la.eig(F)
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

def eig_to_diag_matrix(lam):
    M = np.zeros((2*len(lam),2*len(lam)))
    lam = np.array(lam)
    iA = range(0,M.shape[0],2)
    iB = range(1,M.shape[0],2)
    M[iA,iB] = lam+0.
    M[iB,iA] = -lam+0.
    return M

def diag_antisymm_matr(A,check_form=True):
    '''Diagonalize real antisymmetric matrix to form Q^T \Sigma Q, where
    Q is orthogonal and \Sigma is block diagonal of J-W form as on Wikipedia with
    positive real numbers in upper triangle'''
    assert A.shape[0] % 2 == 0
#    assert max(np.abs(A+A.T).flatten()) < 1e-9

    L = A.shape[0]/2

    A_sq = A.dot(A)
    (ilam_sq,Q_1_T) = np.linalg.eigh(A_sq)
    lam = (np.abs(ilam_sq[::2])) ** 0.5 # more stable than previously (would give NaN)
    Q_1 = Q_1_T.T

    Sig_1 = Q_1.dot(A).dot(Q_1.T) # Should almost be Sigma, but possibly some J-W blocks inverted
    R_1 = 0. * Sig_1
    Sig = 0. * Sig_1

    for j in xrange(L):
        if Sig_1[2*j,2*j+1] > 0.: # If above diagonal is positive, this block is correct
            R_1[2*j,2*j] = 1.
            R_1[2*j+1,2*j+1] = 1.
        else: # Otherwise need to flip it with a sigma_x
            R_1[2*j,2*j+1] = 1.
            R_1[2*j+1,2*j] = 1.
        # Sig[2*j,2*j+1]=lam[j]+0.
        # Sig[2*j+1,2*j]=-lam[j]+0.

    Sig = eig_to_diag_matrix(lam)

    Q = R_1.dot(Q_1)
    if check_form and max(np.abs((Sig-Q.dot(A).dot(Q.T)).flatten())) > 1e-5:
        print 'Difference between Sigma and Q.A.QT'
        print 'Sig_1='
        print_mat(Sig_1)
        print 'Sig='
        print_mat(Sig)
        print 'Q.A.QT='
        print_mat(Q.dot(A).dot(Q.T))
        sys.exit()

    return (Q,lam)

def trim(arr,eps):
    out = []
    for j in range(arr.shape[0]):
        elt = arr[j]
        if not (np.isnan(elt) or elt <= eps or 1 - elt <= eps):
            out.append(elt)
    out = np.array(out)
    return out

def get_entanglement_entropy(alpha):
    '''Get the entanglement entropy from the 2Lx2L matrix C_{ij}=\delta_{ij}+i\alpha_{ij}. See
    notes EEM for details'''

#    (q,mu) = diag_antisymm_matr(alpha)
#    mu = np.sqrt(np.abs(la.eigh(-1 * alpha.dot(alpha))[0]))

    mu = la.eigvalsh(1j * alpha)
    mu[mu > 1.] = 1.
    mu[mu < -1.] = -1.

    S = 0
    p = (1. - mu) / 2.

#    p = (1. - mu[:ell]) / 2.
    with np.errstate(divide='ignore',invalid='ignore'):
        temp = p*np.log(p)+(1.-p)*np.log(1.-p)
    temp[np.isnan(temp)] = 0.
    S = -np.sum(temp) / 2. ## We have double counted the eigenvalues by this method. (only need the positive one for each k) Hence /2.
#    S = -2 * np.sum(temp)
    return S

#    C = alpha
#    wC = sa.eigvalsh(C)
#    half = wC.shape[0] / 2
#
##    # method 1: sum over all eigenvalues
##    S = 0
##    for l in wC:
##        if (l>0.0000000001) and (l<1-0.0000000001):
##            S -= (1.-l) * np.log(1.-l) + l * np.log(l)
##    S = np.real(S)
#
#    # method 2: sum only over half
#    Sa = 0
#    for l in wC[:half]:
#        if (l>0.0000000001) and (l<1-0.0000000001):
#            Sa -= (1.-l) * np.log(1.-l) + l * np.log(l)
#    Sa = 2 * np.real(Sa)
#    Sb = 0
#    for l in wC[half:]:
#        if (l>0.0000000001) and (l<1-0.0000000001):
#            Sb -= (1.-l) * np.log(1.-l) + l * np.log(l)
#    Sb = 2 * np.real(Sb)
#    S = max(Sa,Sb)


    ### old way
#    (q,mu) = diag_antisymm_matr(alpha)
#    eps = []
#    for m in range(mu.shape[0]):
#        if mu[m] != 1.:
#            eps.append(np.arctanh(mu[m]))
#    eps = np.array(eps)
#
#    p_0 = np.exp(-eps) / (np.exp(-eps) + np.exp(eps))
#    p_0 = trim(p_0,cutoff)

#    print q # debug
#    print mu
#    print eps
#    print p_0
#    return 0. - np.sum(p_0 * np.log(p_0) + (1. - p_0) * np.log(1. - p_0))

def get_rand_couplings(w,L):
    # returns exponential distribution with strength w on length L, interval [0,1]. w=1 is a box
    # the logs have mean -w, the couplings have mean exp(-w).
    log_couplings = np.array([-random.expovariate(1./w) for i in range(0,L)])
    couplings = np.exp(log_couplings)

    # now give them a random sign
    couplings = np.array([couplings[i] * (-1) ** random.sample([0,1],1)[0] for i in range(couplings.shape[0])])
    return couplings

def sprinkle(couplings,N):
    # sprinkles in N pi couplings to the chain. Replaces x by 1-x if pos, -1 + x if neg
    L = couplings.shape[0]
    for j in random.sample(range(L),N):
        if couplings[j] >= 0.:
            couplings[j] = 1. - couplings[j]
        elif couplings[j] < 0.:
            couplings[j] = -1. - couplings[j]
    return couplings

def get_F(phi1s,phi2s): # constructs Majorana F operator
    
    L = phi1s.shape[0]
    
    A_inds=range(0,2*L,2)
    B_inds=range(1,2*L,2)

    U1 = np.zeros((2*L,2*L))
    phi1 = phi1s
    U1[A_inds,A_inds] = np.cos(phi1)
    U1[A_inds,B_inds] = np.sin(phi1)
    U1[B_inds,A_inds] = -np.sin(phi1)
    U1[B_inds,B_inds] = np.cos(phi1)

    U2 = np.zeros((2*L,2*L))
    phi2 = phi2s[:-1]
    A_inds_2 = [A_inds[i] + 1 for i in range(len(A_inds) - 1)]
    B_inds_2 = [B_inds[i] + 1 for i in range(len(B_inds) - 1)]

    U2[A_inds_2,A_inds_2] = np.cos(phi2)
    U2[A_inds_2,B_inds_2] = np.sin(phi2)
    U2[B_inds_2,A_inds_2] = -np.sin(phi2)
    U2[B_inds_2,B_inds_2] = np.cos(phi2)
    
    phiL =  phi2s[-1]
    U2[0,0] = np.cos(phiL)
    U2[-1,-1] = np.cos(phiL)
    U2[0,-1] = -np.sin(phiL)
    U2[-1,0] = np.sin(phiL)

    F = U2.dot(U1)
    return F