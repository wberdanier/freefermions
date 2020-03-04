import calc_entropy as ce
import majorana_floquet as mf
import os
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'plasma'
plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rc('text', usetex=True)

import sys
import scipy.optimize as op

def cft_EE(ls,c,s0):
#    entropy = (float(c) / 3.) * np.log(ls) + s0 # bulk cft formula
    entropy = (float(c) / 6.) * np.log(ls) + s0 # boundary cft formula
    return entropy

r_max = 1

digits = 8

delta_max = 0.5
epsilon_max = delta_max

w = 1.
w_epsilon = w
w_delta = w

n_pi = 0.
n_pi_epsilon = n_pi
n_pi_delta = n_pi

l1 = 0.
l2 = 0.

rs = range(r_max)
Ls = np.array([4,8,12,16,24,36,52,76,108,156,224,324,468,672]) / 2
# Ls = np.array([4])
#Ls = np.array([4])

# Ls = np.array([2])

EEs_avg = 0. * Ls
#,672

s0 = 0. * Ls # to calculate std dev on the fly, see wikipedia
s1 = 0. * Ls
s2 = 0. * Ls
error = 0. * Ls

EEs_total = []
phi1_total = []
phi2_total = []
flips_total = []

goods = 0
for r in rs:
    print "realization: " + str(r+1) + " of " + str(r_max) + " n_pi=" + str(n_pi) + " w="+str(w) + " delta_max = "+str(delta_max)
    sys.stdout.flush() ## make it print

    EEs = []
    for L in Ls:
#        print "L = "+str(L)
        deltas = delta_max * mf.get_rand_couplings(w_delta,L)
        epsilons = epsilon_max * mf.get_rand_couplings(w_epsilon,L)
        hs = l1 * np.ones(L) + deltas
        Js = l2 * np.ones(L) + epsilons
        hs = mf.sprinkle(hs,int(n_pi_delta * deltas.shape[0]))
        Js = mf.sprinkle(Js,int(n_pi_epsilon * epsilons.shape[0]))
        Js[-1] = 0. ## OBCs

        ### debug
#        hs_mat = np.loadtxt('4hs.out')
#        Js_mat = np.loadtxt('4Js.out')
#        z = 150
#        hs = hs_mat[z,:]
#        Js = Js_mat[z,:]
#        Js[-1] = 0. ## OBCs
        ###


        phi1 = np.pi * hs
        phi2 = np.pi * Js

        assert phi1.shape[0] == L
        assert phi2.shape[0] == L ## prevent memory leaks in C++

#        flips = [1. for j in range(L/2)]+[0. for j in range(L/2)] # excite exactly L/2 modes
#        random.shuffle(flips)
#        flips = numpy.array(flips)



#        f = open('f.in', 'w')
#        f.write(str(L)+'\n')
#        for i in range(phi1.shape[0]):
#            f.write(str(phi1[i]))
#        f.write('\n')
#        for i in range(phi2.shape[0]):
#            f.write(str(phi2[i]))
#        f.write('\n')
#        for i in range(flips.shape[0]):
#            f.write(str(flips[i]))
#        f.close()

#        for flips in [[0.,0.,0.,0.],[1.,1.,0.,1.],[1.,0.,0.,0.]]:
#            print flips
#            S = ce.entropy(phi1,phi2,flips)
#            EEs.append(S)

        flips = [random.randint(0,1) for j in range(L)] # flip or not randomly
#
        S = ce.entropy(phi1,phi2,flips,digits)

        # # check for convergence
        # inc = 5
        # S2 = ce.entropy(phi1,phi2,flips,digits+inc)
        # while(S - S2 > 1e-2 or S2 > 10.):
        #     inc += 5
        #     S2 = ce.entropy(phi1,phi2,flips,digits+inc)
        # S = S2


        EEs.append(S)
        print "L: "+str(L)
        print "S: "+str(S)

        if S > 10:
            S = np.array([S])
            path = 'data/bad/'+str(goods)+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            np.savetxt(path+'phi1.out',phi1)
            np.savetxt(path+'phi2.out',phi2)
            np.savetxt(path+'flips.out',flips)
            np.savetxt(path+'S.out',S)
            break

    EEs_total.append(EEs) ## raw data
    phi1_total.append(list(phi1))
    phi2_total.append(list(phi2))
    flips_total.append(flips)

    ## do some averaging
    EEs = np.array(EEs)

    s0 += 1
    s1 += EEs
    s2 += EEs * EEs
    if s0[0] > 1:
        error = np.sqrt((s0 * s2 - s1 * s1)/(s0 * (s0 - 1))) # standard deviation so far
        error /= np.sqrt(s0[0]-1)
    EEs_avg += EEs

    ### checkpointing
    if (r+1) % 50 == 0:
        path = 'data/C++/n_pi='+str(n_pi)+'/w='+str(w)+'/r='+str(r+1)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(path+'EEs_avg.out',EEs_avg / r)
        np.savetxt(path+'error.out',error)
        np.savetxt(path+'EEs_total.out',EEs_total)
        np.savetxt(path+'phi1_total.out',phi1_total)
        np.savetxt(path+'phi2_total.out',phi2_total)
        np.savetxt(path+'flips_total.out',flips_total)

    goods += 1

    if EEs[-1] > 10.:
        break
        print "bad realization"
        sys.exit()

EEs_avg /= goods

print EEs_avg
print error


##### save
path = 'data/C++/n_pi='+str(n_pi)+'/w='+str(w)+'/r='+str(r_max)+'/'
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path+'EEs_avg.out',EEs_avg)
np.savetxt(path+'error.out',error)
np.savetxt(path+'EEs_total.out',EEs_total)
np.savetxt(path+'phi1_total.out',phi1_total)
np.savetxt(path+'phi2_total.out',phi2_total)
np.savetxt(path+'flips_total.out',flips_total)
####

#EEs_avg = np.loadtxt('EEs_avg.out')
#error = np.loadtxt('error.out')
#
#interval = 7
#params, errors = op.curve_fit(cft_EE,Ls[-interval:],EEs_avg[-interval:])
#c_fit = params[0]
#EE0 = params[1]
#
#
#plt.figure()
#plt.errorbar(Ls,EEs_avg,yerr=error,label='C++')
#plt.plot(Ls,cft_EE(Ls,c_fit,EE0),label='fit $c=$'+str(np.around(c_fit / np.log(2),5))+' $\ln 2$')
##plt.errorbar(Ls,EEs_xx[:5] / 2.,yerr=error_xx[:5] / 2.,label='xx')
#plt.xscale('log')
#plt.xlabel('$L$')
#plt.ylabel('$S(L)$')
#plt.legend(loc='best')
#plt.title('w = '+str(w)+', r = '+str(r_max))

#plt.savefig('test.png')
