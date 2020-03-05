#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:47:15 2016

@author: William Berdanier
"""

###############################################################################
# Do a quench from free (h_0=0) to fixed (h_0 \neq 0) b.c. in the TFI chain at criticality.
# The entanglement entropy is calculated as a function of time, in the Majorana representation.
# This code calculates the entanglement entropy of a single slice of the chain only,
# which enables much larger system sizes to be computed in the same running time.
# See PRL 118, 260602 (2017) for physics details.
###############################################################################

import numpy as np
import numpy.linalg as la
import sys
import os
import matplotlib
matplotlib.use('Agg') #fixes figure making error on remote machine
import matplotlib.pyplot as plt
import scipy.optimize as op
#import time

## Uncomment this block to set variables via command line arguments
#if len(sys.argv) < 5:
#    print 'Usage: get_entropy_majorana.py $L $h_0 $t_max $dt [$h=1] [$J=1]'
#    sys.exit()
#L=int(sys.argv[1])
#h_0=float(sys.argv[2])
#t_max=float(sys.argv[3])
#dt=float(sys.argv[4])
#h=1.
#if len(sys.argv) > 5 and sys.argv[5] != 'show':
#    h=float(sys.argv[5])
#J=1.
#if len(sys.argv) > 6 and sys.argv[6] != 'show':
#    J=float(sys.argv[6])

def print_mat(M):
    for j in xrange(M.shape[0]):
        st=''
        for k in xrange(M.shape[1]):
            st += ('%.3f ' % M[j,k])
        print st

#def load_run(path): #loads all the output files from a run
#    entropy_occ = np.loadtxt('entropy_occ.out')
#    entropy_up = np.loadtxt('entropy_up.out')
#    entropy_vac = np.loadtxt('entropy_vac.out')
#    entropy_diff = np.loadtxt('entropy_diff.out')
#    entropy = np.loadtxt('entropy.out')
#    t_vals = np.loadtxt('t.out')
#    if ifEcho:
#        echo = np.loadtxt('echo.out')
#        log_echo = np.loadtxt('log_echo.out')

def eig_to_diag_matrix(lam):
    M = np.zeros((2*len(lam),2*len(lam)))
    iA = range(0,M.shape[0],2)
    iB = range(1,M.shape[0],2)
    M[iA,iB] = lam+0.
    M[iB,iA] = -lam+0.
    return M

def diag_antisymm_matr(A,check_form=False):
    '''Diagonalize real antisymmetric matrix to form Q^T \Sigma Q, where
    Q is orthogonal and \Sigma is block diagonal of J-W form as on Wikipedia with
    positive real numbers in upper triangle'''
    assert A.shape[0] % 2 == 0
    assert max(np.abs(A+A.T).flatten()) < 1e-9

    L = A.shape[0]/2

    A_sq = A.dot(A)
    (ilam_sq,Q_1_T) = la.eigh(A_sq)
    lam = (-ilam_sq[::2]) ** 0.5
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

def get_entanglement_entropy(alpha):
    '''Get the entanglement entropy from the 2Lx2L matrix C_{ij}=\delta_{ij}+i\alpha_{ij}. See
    notes EEM for details'''

    (q,mu) = diag_antisymm_matr(alpha)
    #    print 'mu='+str(mu)
    eps = np.arctanh((mu+1e-9)/(1.+2e-9)) # Round (0,1) to (1e-9,1-1e-9)
    p_0 = np.exp(-eps)/(np.exp(-eps) + np.exp(eps))
    return -np.sum(p_0*np.log(p_0) + (1.-p_0)*np.log(1.-p_0))

def maj_to_ferm(A):
    #inputs an even-dim antisymmetric hamiltonian A in the Majorana basis and outputs BdG fermionic hamiltonian
    assert A.shape[0] %2 == 0
    L = A.shape[0]/2
    BdG = np.zeros(A.shape,dtype=complex)
    for i in range(L):
        for j in range(L):
            BdG[i,j] = 1j * A[2*i,2*j] - A[2*i+1,2*j] + A[2*i,2*j+1] + 1j * A[2*i+1,2*j+1]
            BdG[i+L,j] = 1j * A[2*i,2*j] + A[2*i+1,2*j] + A[2*i,2*j+1] - 1j * A[2*i+1,2*j+1]
            BdG[i,j+L] = 1j * A[2*i,2*j] - A[2*i+1,2*j] - A[2*i,2*j+1] - 1j * A[2*i+1,2*j+1]
            BdG[i+L,j+L] = 1j * A[2*i,2*j] + A[2*i+1,2*j] - A[2*i,2*j+1] + 1j * A[2*i+1,2*j+1]
    return BdG

#this works for a single quench only.
def get_echo(before,after,t): #get loschmidt echo from determinants as in Vasseur PRX 2014 appendix A
    assert before.shape[0] % 2 == 0
    assert after.shape[0] % 2 == 0
    L = before.shape[0]/2 #should be system size
    #We need to convert the input Hamiltonians from Majorana basis to BdG fermionic basis
    BdG0 = maj_to_ferm(before)
    BdG1 = maj_to_ferm(after)
    (lambda_d, V_d) = la.eigh(BdG0) #diagonalize the two BdG hamiltonians before and after
    (lambda_q, V_q) = la.eigh(BdG1)
#    print 'lambda_d:'
#    print lambda_d
#    print 'lambda_q:'
#    print lambda_q
    M = V_d.dot(V_q.conjugate().T) # this is the A B matrix
    A = M[:L,:L]
    B = M[:L,L:]
    Dtq = np.zeros((L,L),dtype=complex)
    E = 2. * lambda_q
    for i in range(L):
        Dtq[i,i] = np.exp(1.j * E[i] * t)
    echo = abs( la.det(A.T + Dtq.dot( B.conj().T )) )**2 / abs( la.det( A.T + B.conj().T ) )**2
    return echo

def comm(A,B):
    return A.dot(B) - B.dot(A)

def loadrun(path):
    entropy_diff = np.loadtxt(path+'entropy_diff.out')
    entropy_occ = np.loadtxt(path+'entropy_occ.out')
    entropy_up = np.loadtxt(path+'entropy_up.out')
    entropy_vac = np.loadtxt(path+'entropy_vac.out')
    entropy = np.loadtxt(path+'entropy.out')
    t_vals = np.loadtxt(path+'t.out')
    entropy_cut = np.loadtxt(path+'entropy_cut.out')
    return entropy_diff,entropy_occ,entropy_up,entropy_vac,entropy,t_vals,entropy_cut

def loadrun_cut(path):
    entropy_cut = np.loadtxt(path+'entropy_cut.out')
    t_vals = np.loadtxt(path+'t.out')
    return t_vals,entropy_cut

def code(L,T,h_0i,h_0f,t_max,m,u,ifHigh,cut):
    T /= u
    omega = 1. / T
    dt = float(T) / m #timestep
#    if dt > 0.2:
#        dt = 0.1
#        N = float(T) / dt
    h = 1.
    J = 1.

    t_vals=np.arange(0.,t_max+dt/2.,dt)
    A_inds=range(0,2*(L+1),2)
    B_inds=range(1,2*(L+1),2)

    # Define the hopping matrix for our model, say the TFI chain with L sites, obc, at field h (J=1).
    H_0 = np.zeros((2*(L+1),2*(L+1))) # Initial Hamiltonian, i \eta^T H \eta, where \eta^T=(A_0,B_0,A_1,...)
    H_0[A_inds[1:],B_inds[1:]] = h/2+0.
    H_0[B_inds[1:],A_inds[1:]] = -h/2+0.
    H_0[B_inds[1:-1],A_inds[2:]] = J/2+0.
    H_0[A_inds[2:],B_inds[1:-1]] = -J/2+0.
    H_1 = H_0+0.
    H_0[1,2] = h_0i/2+0.
    H_0[2,1] = -h_0i/2+0.
    H_1[1,2] = h_0f/2+0.
    H_1[2,1] = -h_0f/2+0.
    H_high = 0.5 * (H_0 + H_1) + T * comm(H_0,H_1) - T**2 * 16/ 24. * ( comm(H_0,comm(H_0,H_1)) + comm(H_1,comm(H_1,H_0)) ) - T**3 * 64/ 48. * comm(H_1,comm(H_0,comm(H_0,H_1))) + T**4 * 256/ 1440. * ( comm(H_1,comm(H_1,comm(H_1,comm(H_1,H_0)))) + comm(H_0,comm(H_0,comm(H_0,comm(H_0,H_1)))) - 2 * ( comm(H_0,comm(H_1,comm(H_1,comm(H_1,H_0)))) + comm(H_1,comm(H_0,comm(H_0,comm(H_0,H_1)))) ) - 6 * ( comm(H_1,comm(H_0,comm(H_1,comm(H_0,H_1)))) + comm(H_0,comm(H_1,comm(H_0,comm(H_1,H_0)))) ) )

    # Diagonalize H_0 and get the correlation matrix in the ground state, neglecting A_0, B_0 (fake field fermions)
    (Q_0,lam_0)=diag_antisymm_matr(H_0[2:,2:],True)
    corr_gs = 0.*Q_0 # 1.j times the correlation matrix in the g.s. in the rotated basis C'=i*corr_gs
    corr_gs[A_inds[:-1],B_inds[:-1]] = 1.
    corr_gs[B_inds[:-1],A_inds[:-1]] = -1.
    A_0_up = 0.*H_0
    A_0_up[2:,2:] = Q_0.T.dot(corr_gs).dot(Q_0) # Correlations in the g.s. in the original basis with extra spin=up
    A_0_vac = A_0_up+0. # Correlations in the g.s. in the original basis with extra fermion=vac
    # For spin up state, corr are zero in the A_0, B_0 block.
    A_0_vac[0,1] = 1.
    A_0_vac[1,0] = -1.
    A_0_occ=A_0_up+0. # Correlations in the g.s. in the original basis with extra fermion=occ
    A_0_occ[0,1] = -1.
    A_0_occ[1,0] = 1.

    S_0_vac = 0.*A_0_vac
    S_0_vac[:(L+1),(L+1):]=-np.identity(L+1)
    S_0_vac[(L+1):,:(L+1)]=np.identity(L+1)
    S_0_occ=S_0_vac+0.
    S_0_occ[0,L+1] = 1.
    S_0_occ[L+1,0] = -1. # Flip the occupation of "site" 0
    Q_0_full = 0.*H_0
    Q_0_full[2:,2:] = Q_0+0.
    Q_0_full[:2,:2] = np.identity(2)

    # Diagonalize H_1 for time ev.
    (Q_1,lam_1) = diag_antisymm_matr(H_1,True)
    E_1 = 2.*lam_1
    (Q_2,lam_2) = diag_antisymm_matr(H_0,True)
    E_2 = 2.*lam_2

    # Then time evolve and calculate entropy
    # Figure out the entropy from the correlation matrix
    entropy_up = np.zeros((L+1,len(t_vals)))
    entropy_vac = np.zeros((L+1,len(t_vals)))
    entropy_occ = np.zeros((L+1,len(t_vals)))

    D1 = 0. * Q_1
    D2 = 0. * Q_2

    A_t_up = A_0_up
    A_t_vac = A_0_vac
    A_t_occ = A_0_occ

    # Time evolution
    phi1 = 2. * E_1 * dt
    D1[A_inds,A_inds] = np.cos(phi1)
    D1[B_inds,B_inds] = np.cos(phi1)
    D1[A_inds,B_inds] = np.sin(phi1)
    D1[B_inds,A_inds] = -np.sin(phi1)

    phi2 = 2. * E_2 * dt
    D2[A_inds,A_inds] = np.cos(phi2)
    D2[B_inds,B_inds] = np.cos(phi2)
    D2[A_inds,B_inds] = np.sin(phi2)
    D2[B_inds,A_inds] = -np.sin(phi2)

    for t_ind in xrange(len(t_vals)):
    #    init = time.clock()
        t = t_vals[t_ind]
        if t_ind % (2 * m) < m:
            print 't='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T)+' omega= '+str(omega)+' H1'
            U = Q_1.T.dot(D1).dot(Q_1)
            j = cut #calculate entropy cut
            entropy_up[j,t_ind] = get_entanglement_entropy(A_t_up[:(2*j+2),:(2*j+2)])
            entropy_vac[j,t_ind] = get_entanglement_entropy(A_t_vac[:(2*j+2),:(2*j+2)])
            entropy_occ[j,t_ind] = get_entanglement_entropy(A_t_occ[:(2*j+2),:(2*j+2)])
            # Current correlation functions in the three states
            A_t_up = U.dot(A_t_up).dot(U.T)
            A_t_vac = U.dot(A_t_vac).dot(U.T)
            A_t_occ = U.dot(A_t_occ).dot(U.T)
        else:
            print 't='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T)+' omega= '+str(omega)+' H0'
            U=Q_2.T.dot(D2).dot(Q_2)
            j = cut
            entropy_up[j,t_ind] = get_entanglement_entropy(A_t_up[:(2*j+2),:(2*j+2)])
            entropy_vac[j,t_ind] = get_entanglement_entropy(A_t_vac[:(2*j+2),:(2*j+2)])
            entropy_occ[j,t_ind] = get_entanglement_entropy(A_t_occ[:(2*j+2),:(2*j+2)])
            # Current correlation functions in the three states
            A_t_up = U.dot(A_t_up).dot(U.T)
            A_t_vac = U.dot(A_t_vac).dot(U.T)
            A_t_occ = U.dot(A_t_occ).dot(U.T)

    entropy = entropy_up - np.log(2)
    entropy[0] = 0
    entropy[-1] = 0

    entropy = np.nan_to_num(entropy) #Avoids numerical garbage

    # Subtract off entropy of ground state
    entropy_diff = 0. * entropy
    for t in range(entropy.shape[1]):
        entropy_diff[:,t] = entropy[:,t] - entropy[:,0]

    entropy_cut = entropy_diff[cut,:]

    # Save output
    path='build/h0i='+str(h_0i)+'_to_h0f='+str(h_0f)+'/L='+str(L)+'/T='+str(T)+'/t='+str(t_max)+'/cut='+str(cut)+'/'

    if not os.path.exists(path):
        os.makedirs(path)


    np.savetxt(path+'t.out',t_vals,)
    np.savetxt(path+'entropy_up.out',entropy_up)
    np.savetxt(path+'entropy_vac.out',entropy_vac)
    np.savetxt(path+'entropy_occ.out',entropy_occ)
    np.savetxt(path+'entropy.out',entropy)
    np.savetxt(path+'entropy_diff.out',entropy_diff)
    np.savetxt(path+'entropy_cut.out',entropy_cut)

    entropy_diff_temp = entropy_diff

    if ifHigh:
        H_1 = H_high
            # Diagonalize H_0 and get the correlation matrix in the ground state, neglecting A_0, B_0 (fake field fermions)
        (Q_0,lam_0)=diag_antisymm_matr(H_0[2:,2:],True)
        corr_gs = 0.*Q_0 # 1.j times the correlation matrix in the g.s. in the rotated basis C'=i*corr_gs
        corr_gs[A_inds[:-1],B_inds[:-1]] = 1.
        corr_gs[B_inds[:-1],A_inds[:-1]] = -1.
        A_0_up = 0.*H_0
        A_0_up[2:,2:] = Q_0.T.dot(corr_gs).dot(Q_0) # Correlations in the g.s. in the original basis with extra spin=up
        A_0_vac = A_0_up+0. # Correlations in the g.s. in the original basis with extra fermion=vac
        # For spin up state, corr are zero in the A_0, B_0 block.
        A_0_vac[0,1] = 1.
        A_0_vac[1,0] = -1.
        A_0_occ=A_0_up+0. # Correlations in the g.s. in the original basis with extra fermion=occ
        A_0_occ[0,1] = -1.
        A_0_occ[1,0] = 1.

        S_0_vac = 0.*A_0_vac
        S_0_vac[:(L+1),(L+1):]=-np.identity(L+1)
        S_0_vac[(L+1):,:(L+1)]=np.identity(L+1)
        S_0_occ=S_0_vac+0.
        S_0_occ[0,L+1] = 1.
        S_0_occ[L+1,0] = -1. # Flip the occupation of "site" 0
        Q_0_full = 0.*H_0
        Q_0_full[2:,2:] = Q_0+0.
        Q_0_full[:2,:2] = np.identity(2)

        # Diagonalize H_1 for time ev.
        (Q_1,lam_1) = diag_antisymm_matr(H_1,False)
        E_1 = 2.*lam_1
        (Q_2,lam_2) = diag_antisymm_matr(H_0,True)
        E_2 = 2.*lam_2

        # Then time evolve and calculate entropy
        # Figure out the entropy from the correlation matrix
        entropy_up = np.zeros((L+1,len(t_vals)))
        entropy_vac = np.zeros((L+1,len(t_vals)))
        entropy_occ = np.zeros((L+1,len(t_vals)))

        D1 = 0. * Q_1
        D2 = 0. * Q_2

        A_t_up = A_0_up
        A_t_vac = A_0_vac
        A_t_occ = A_0_occ

        # Time evolution
        phi1 = 2. * E_1 * dt
        D1[A_inds,A_inds] = np.cos(phi1)
        D1[B_inds,B_inds] = np.cos(phi1)
        D1[A_inds,B_inds] = np.sin(phi1)
        D1[B_inds,A_inds] = -np.sin(phi1)

        phi2 = 2. * E_2 * dt
        D2[A_inds,A_inds] = np.cos(phi2)
        D2[B_inds,B_inds] = np.cos(phi2)
        D2[A_inds,B_inds] = np.sin(phi2)
        D2[B_inds,A_inds] = -np.sin(phi2)

        for t_ind in xrange(len(t_vals)):
        #    init = time.clock()
            t = t_vals[t_ind]
            print 't='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T), 'H1'
            U = Q_1.T.dot(D1).dot(Q_1)
            j = cut #calculate entropy cut
            entropy_up[j,t_ind] = get_entanglement_entropy(A_t_up[:(2*j+2),:(2*j+2)])
            entropy_vac[j,t_ind] = get_entanglement_entropy(A_t_vac[:(2*j+2),:(2*j+2)])
            entropy_occ[j,t_ind] = get_entanglement_entropy(A_t_occ[:(2*j+2),:(2*j+2)])
            # Current correlation functions in the three states
            A_t_up = U.dot(A_t_up).dot(U.T)
            A_t_vac = U.dot(A_t_vac).dot(U.T)
            A_t_occ = U.dot(A_t_occ).dot(U.T)


        entropy = entropy_up - np.log(2)
        entropy[0] = 0
        entropy[-1] = 0

        entropy = np.nan_to_num(entropy) #Avoids numerical garbage

        # Subtract off entropy of ground state
        entropy_diff = 0. * entropy
        for t in range(entropy.shape[1]):
            entropy_diff[:,t] = entropy[:,t] - entropy[:,0]

        entropy_diff_high = entropy_diff
        entropy_diff = entropy_diff_temp
        entropy_highvsdrive = np.abs(entropy_diff_high[cut,:] - entropy_diff[cut,:])

        # Save output
        path='build/h0i='+str(h_0i)+'_to_h0f='+str(h_0f)+'/L='+str(L)+'/T='+str(T)+'/t='+str(t_max)+'/cut='+str(cut)+'/'

        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(path+'t.out',t_vals,)
        np.savetxt(path+'entropy_up_high.out',entropy_up)
        np.savetxt(path+'entropy_vac_high.out',entropy_vac)
        np.savetxt(path+'entropy_occ_high.out',entropy_occ)
        np.savetxt(path+'entropy_high.out',entropy)
        np.savetxt(path+'entropy_diff_high.out',entropy_diff)
        np.savetxt(path+'entropy_highvsdrive.out',entropy_highvsdrive)
