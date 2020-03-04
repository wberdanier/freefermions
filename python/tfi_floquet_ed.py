#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:33:00 2017

@author: will
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.linalg import sqrtm
import random

X = np.array([[0,1],[1,0]],dtype = 'complex')
Y = np.array([[0,1j],[-1j,0]],dtype = 'complex')
Z = np.array([[1,0],[0,-1]],dtype = 'complex')
Id = np.array([[1,0],[0,1]],dtype = 'complex')
ZZ = np.kron(Z,Z)

def get_similarity(A,B): # returns percentage of eigenvalues that are the same up to epsilon tolerance
    G = np.dot(A, la.inv(B))
    lams, P = la.eig(G)
    c = 0
    eps = 10**-2
#    print G, lams
    for lam in lams:
        if np.abs(lam - 1) < eps:
            c += 1
    return float(c) / lams.shape[0]
    
def trace_overlap(A,B):
    return np.abs(np.trace(np.dot(A,B)))

def get_X(i,N): # constructs X_i spin matrix
    assert i <= N-1
    Xi = np.array([[1]],dtype = 'complex')
    for j in range(N):
        if j == i:
            Xi = np.kron(Xi, X)
        else:
            Xi = np.kron(Xi, Id)
    return Xi

def get_ZZ(i,N): # constructs Z_i Z_i+1 spin matrix
    if i <= N-2:
        ZZi = np.array([[1]],dtype = 'complex')
        for j in range(N-1):
            if j == i:
                ZZi = np.kron(ZZi, ZZ)
            else:
                ZZi = np.kron(ZZi, Id)
        return ZZi
    if i == N-1:
        ZZi = Z + 0.
        for j in range(N-2):
            ZZi = np.kron(ZZi,Id)
        ZZi = np.kron(ZZi,Z)
        return ZZi

def get_Z(i,N): # constructs X_i spin matrix
    assert i <= N-1
    Zi = np.array([[1]],dtype = 'complex')
    for j in range(N):
        if j == i:
            Zi = np.kron(Zi, Z)
        else:
            Zi = np.kron(Zi, Id)
    return Zi

def get_F(H1,H2): 
    F = np.dot(expm(-1j * H2), expm(-1j * H1))
    return F

def comm(A,B):
    c = np.dot(A,B) - np.dot(B,A)  
    return c

def sq(A):
    return np.dot(A,A)

def get_EE(rho):
    lams = la.eigvalsh(rho)
    S = 0.
    for lam in lams:
        if lam > 1e-14:
            S -= np.real(lam * np.log(lam))
    return np.abs(S)

def get_rho(psi,l): # cut from edge of system to middle
    mat = np.tensordot(psi,psi.conj().T,0)
    L = mat.shape[0]
    chunk = L / 2**l
    rho = 0. * mat[:chunk,:chunk]
    
    for j in range(2**l):
        rho += mat[chunk * j : chunk * (j+1),chunk * j : chunk * (j+1)]
        
    return rho

#l1 = 0.8
#l2 = 0.8
L = 4
ell = L/2

L_TFI = L

epsilon_max = 0.2
delta_max = 0.2
w_epsilon = 1.
w_delta = 1.

#deltas = l1 * np.ones(L) 
#epsilons = l2 * np.ones(L-1) 

#w_epsilon = 3.
#w_delta = w_epsilon
#epsilons = epsilon_max * get_rand_couplings(w_epsilon,L - 1)
#deltas = delta_max * get_rand_couplings(w_delta,L)

### L = 2, clean
#epsilons = np.array([-0.2]) 
#deltas = np.array([-0.2,-0.2])

### L = 2, clean, this works
#epsilons = np.array([-0.2]) 
#deltas = np.array([1.2,1.2])

### L = 2, clean, this is broken 
#epsilons = np.array([0.]) 
#deltas = np.array([0.,1.2])

## L = 2, dirty
#epsilons = np.array([-0.2]) 
#deltas = np.array([-0.82,-0.75])

### L = 2, broken
#epsilons = np.array([-0.07206952])
#deltas = np.array([1+0.16086343,-0.13013709])

### L = 3
#epsilons = np.array([-0.08466438,  0.02838463])
#deltas = np.array([ 0.07229095,  0.1442309 , -0.02887752])

### L = 4
#epsilons = np.array([ 0.16088299, -0.03123118, -0.15213832])
#deltas = np.array([ 0.17320622, -0.148111  ,  0.18086554, -0.16409297])

#### L = 4, pis, broken
#epsilons = np.array([ 1. - 0.16088299, -0.03123118, - 0.15213832])
#deltas = np.array([ 1 - 0.17320622, 1. + 0.148111  ,  1. - 0.18086554, -0.16409297])

### L = 6
#epsilons = np.array([-0.1871197 ,  0.07842424,  0.13041357,  0.14470927, -0.04501188])
#deltas = np.array([ 0.09934838, -0.09175597, -0.12866234, -0.1026317 , -0.17337427, 0.00892681])

### L = 6, pis
#epsilons = 1. - np.array([-0.1871197 ,  0.07842424,  0.13041357,  0.14470927, -0.04501188])
#deltas = 1. - np.array([ 0.09934838, -0.09175597, -0.12866234, -0.1026317 , -0.17337427, 0.00892681])

#### w = 3., L = 6, pis - broken
#epsilons = np.array([ 1 - 0.00540518, -0.19833121,  0.00994274,  1 - 0.0022542 , -0.10338123])
#deltas = np.array([1 + 0.00011637, -0.02354735, 1 + 0.04832932,  1 - 0.0005081 ,  1 - 0.0282347 ,-0.00367225])
## w = 3., L = 6, pis - broken
#epsilons = 1. - np.array([ 1 - 0.00540518, -0.19833121,  0.00994274,  1 - 0.0022542 , -0.10338123, -0.1]) 
#deltas = 1. - np.array([1 + 0.00011637, -0.02354735, 1 + 0.04832932,  1 - 0.0005081 ,  1 - 0.0282347 ,-0.00367225])

#hs = np.array([0.18,0.15,0.9,0.3])
#Js = np.array([0.8,0.2,0.7,0.]) 

#### debug on 4 sites
hs_mat = np.loadtxt('4hs.out')
Js_mat = np.loadtxt('4Js.out')
z = 150
hs = hs_mat[z,:]
Js = Js_mat[z,:]
#Js[-1] *= -1
#Js[-1] = 0. ## OBCs
####

sites = range(L)

H1 = np.zeros((2 ** L,2 ** L),dtype = 'complex')
H2 = np.zeros((2 ** L,2 ** L),dtype = 'complex')
for j in range(L):
    H1 += hs[j] * np.pi / 2. * get_X(j,L)
for j in range(L-1):
    H2 += Js[j] * np.pi / 2. * get_ZZ(j,L)
    
H2 += Js[L-1] * np.pi / 2. * get_ZZ(L-1,L) # PBCs
    
    
F = get_F(H1,H2)


wF, vF = la.eig(F)
wF = np.real(1j * np.log(wF))

idx = wF.argsort()
wF = wF[idx]
vF = vF[:,idx]

wF_ed = wF
vF_ed = vF

## entanglement in all states
Sss = []
for i in range(2**L):
    psi = vF[:,i]
    E = wF[i]
    Ss = []
    
#    ells = range(L+1)
    ells = [L/2]
    
    for ell in ells:
        rho = get_rho(psi,ell)
        S = get_EE(rho)
        Ss.append(S)
    Sss.append(Ss)
Sss = np.array(Sss)

for j in range(2**L):
    print Sss[j]
#plt.plot(range(L+1),Sss.T)
#gs_ind = np.argmin(Sss[:(2**L)/2,L/2])

###### ground state entanglement
#Ss = []
#psi = vF[:,gs_ind]
#E = wF[gs_ind]
#for ell in range(L+1):
#    rho = get_rho(psi,ell)
#    S = get_EE(rho)
#    Ss.append(S)
#plt.plot(range(L+1),Ss)