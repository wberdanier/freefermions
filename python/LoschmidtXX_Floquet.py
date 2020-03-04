import numpy as np

import scipy.sparse as sp
import numpy.linalg as la
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as op

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

L_TFI = 1000
L = 2 * L_TFI + 1
J = 1.
u = 1.
h_0i = 0.
h_0f = 0.3 * np.sqrt(u)
n = 20 #number of time steps in a period
#N_tot = 9 #total number of periods
omega = 0.02 * u
T = 1 / omega
#t_max = N_tot * (2 * T)
t_max = L_TFI / 2
N_tot = int(np.floor(t_max / (2*T)))
assert t_max <= 0.5 * L_TFI # finite size effects
dt = T / float(n)
#if dt > 1:
#    dt = t_max / 50.
# dt = T
ifHigh = False

J *= 2

def comm(A,B):
    return A.dot(B) - B.dot(A)

def loadrun(path):
    Gt2_list_tfi = np.loadtxt(path+'Gt2_tfi.out')
    Gt2_list = np.loadtxt(path+'Gt2_xx.out')
    log_Gt2_tfi = np.loadtxt(path+'log_Gt2_tfi.out')
    t_list = np.loadtxt(path+'t.out')
    return Gt2_list_tfi, Gt2_list, log_Gt2_tfi, t_list
#Jt=[random() for i in range(0,L-1)]

Jt=J*np.ones(L-1);

# after the quench
Jt[0] = J * h_0f

Hb = sp.dok_matrix((L,L), dtype='d')
for i in range(0,L-1):
    Hb[i,i+1] = -Jt[i]
    Hb[i+1,i] = Hb[i,i+1]
    Hb[i,i] = 0.0

# Before the quench
Jt[0] = 0.0

Ha= sp.dok_matrix((L,L), dtype='d')
for i in range(0,L-1):
    Ha[i,i+1] = -Jt[i]
    Ha[i+1,i] = Ha[i,i+1]
    Ha[i,i] = 0.0

# Compute spectrum of the initial state,
# it is important to sort the eigenvalues! (Fill the Fermi Sea)
Ha = Ha.todense()
Hb = Hb.todense()

(wa,va) = np.linalg.eig(Ha)
idx = wa.argsort()
wa = wa[idx]
va = va[:,idx]
(wb,vb) = np.linalg.eig(Hb)

# Much simpler method (paper Dante)
Nf = sum(1 for n in wa if n < 0)
P = va[:,0:Nf] # L*N_f, fill the Fermi Sea
t = 0
Gt2_list = [] # Loschmidt echo |G(t)|^2
t_list = [] # real time
Gt2_local_list = []

P0 = np.conj(np.transpose(P))

UTb = np.dot(vb,np.dot(np.diag(np.exp(-1j*T*wb)),np.transpose(vb)))
UTa = np.dot(va,np.dot(np.diag(np.exp(-1j*T*wa)),np.transpose(va)))
UT = np.dot(UTa,UTb)
Pt = P0.conj().T

while (t <= t_max + 0.01):
    t_list.append(t)

    if (int(np.floor_divide(t,T))%2==0):
        print('t='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T), 'H1')
        Ut = np.dot(vb,np.dot(np.diag(np.exp(-1j*np.mod(t,T)*wb)),np.transpose(vb)))
        N = int(np.floor_divide(t,T))/2
        Ut = np.dot(Ut,la.matrix_power(UT,N))
    else:
        print('t='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T), 'H0')
        Ut = UTb
        N = int(np.floor_divide(t,T)-1)/2
        Ut = np.dot(Ut,la.matrix_power(UT,N))

    Pt = np.dot(Ut,P)
    Gt2 = (np.abs(np.linalg.det( np.dot(P0,Pt))))**2

    #    # Entanglement entropy (from correlation matrix): much slower
#    Sl=[]
#    for l in range(1,L):
#        C=np.dot(Pt,np.conj(np.transpose(Pt)))[:l,:l]
#        (wC,vC) = np.linalg.eig(C)
#        S=0
#        for l in wC:
#            if (l>0.0000000001) and (l<1-0.0000000001):
#                S-=(1.0-l)*np.log(1.0-l)+(l)*np.log(l)
#        Sl.append(np.real(S))
#    Sl.append(0.0)
#    S_list.append(Sl)

#    # Calculate Local Loschmidt echo (much slower)
#    Gt2_local = []
#    for l in range(1,L):
#        Gt2l = (np.abs(np.linalg.det( np.dot(P0[:l,:l],Pt[:l,:l]))))**2
#        Gt2_local.append(Gt2l)
#    Gt2_local_list.append(Gt2_local)


#    print G
    Gt2_list.append(Gt2)
    t += dt

Gt2_list_tfi = np.array(Gt2_list) ** 0.5

Gt2_list_tfi_temp = Gt2_list_tfi
T_temp = T

if ifHigh:
    # Compute spectrum of the initial state,
    # it is important to sort the eigenvalues! (Fill the Fermi Sea)

    # High frequency expansion for floquet hamiltonian to 4th order
    HF = 0.5 * (Ha + Hb) - 1j * T / 4. * comm(Ha,Hb) - T**2 / 24. * ( comm(Ha,comm(Ha,Hb)) + comm(Hb,comm(Hb,Ha)) ) - 1j * T**3 / 48. * comm(Hb,comm(Ha,comm(Ha,Hb))) - T**4 / 1440. * ( comm(Hb,comm(Hb,comm(Hb,comm(Hb,Ha)))) + comm(Ha,comm(Ha,comm(Ha,comm(Ha,Hb)))) - 2. * ( comm(Ha,comm(Hb,comm(Hb,comm(Hb,Ha)))) + comm(Hb,comm(Ha,comm(Ha,comm(Ha,Hb)))) ) - 6. * ( comm(Hb,comm(Ha,comm(Hb,comm(Ha,Hb)))) + comm(Ha,comm(Hb,comm(Ha,comm(Hb,Ha)))) ) )

    Hb = HF # I am lazy
    T = 10000000

    (wa,va) = np.linalg.eig(Ha)
    idx = wa.argsort()
    wa = wa[idx]
    va = va[:,idx]
    (wb,vb) = np.linalg.eig(Hb)

    # Much simpler method (paper Dante)
    Nf = sum(1 for n in wa if n < 0)
    P = va[:,0:Nf] # L*N_f, fill the Fermi Sea
    t = 0
    Gt2_list = [] # Loschmidt echo |G(t)|^2
    t_list = [] # real time
    Gt2_local_list = []

    P0 = np.conj(np.transpose(P))

    UTb = np.dot(vb,np.dot(np.diag(np.exp(-1j*T*wb)),np.transpose(vb)))
    UTa = np.dot(va,np.dot(np.diag(np.exp(-1j*T*wa)),np.transpose(va)))
    UT = np.dot(UTa,UTb)

    Pt = P0.conj().T

    while (t <= t_max + 0.01):
        t_list.append(t)

        # Calculate full Loschmidt echo
        Gt2 = (np.abs(np.linalg.det( np.dot(P0,Pt))))**2

        # Compute time evolution operator: Ut

        if (int(np.floor_divide(t,T))%2==0):
            print('t='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T), 'H1')
            Ut = np.dot(vb,np.dot(np.diag(np.exp(-1j*np.mod(t,T)*wb)),np.transpose(vb)))
            N = int(np.floor_divide(t,T))/2
            Ut = np.dot(Ut,la.matrix_power(UT,N))
        else:
            print('t='+str(t)+' out of '+str(t_max)+' h_0i= '+str(h_0i)+' h0f= '+str(h_0f)+' L= '+str(L)+' T= '+str(T), 'H0')
            Ut = UTb
            N = int(np.floor_divide(t,T)-1)/2
            Ut = np.dot(Ut,la.matrix_power(UT,N))

        Pt = np.dot(Ut,P)

        #    # Entanglement entropy (from correlation matrix): much slower
    #    Sl=[]
    #    for l in range(1,L):
    #        C=np.dot(Pt,np.conj(np.transpose(Pt)))[:l,:l]
    #        (wC,vC) = np.linalg.eig(C)
    #        S=0
    #        for l in wC:
    #            if (l>0.0000000001) and (l<1-0.0000000001):
    #                S-=(1.0-l)*np.log(1.0-l)+(l)*np.log(l)
    #        Sl.append(np.real(S))
    #    Sl.append(0.0)
    #    S_list.append(Sl)

    #    # Calculate Local Loschmidt echo (much slower)
    #    Gt2_local = []
    #    for l in range(1,L):
    #        Gt2l = (np.abs(np.linalg.det( np.dot(P0[:l,:l],Pt[:l,:l]))))**2
    #        Gt2_local.append(Gt2l)
    #    Gt2_local_list.append(Gt2_local)
    #    print G
        Gt2_list.append(Gt2)
        t += dt

    Gt2_list_tfi = np.array(Gt2_list) ** 0.5
    Gt2_list_tfi_high = Gt2_list_tfi

Gt2_list_tfi = Gt2_list_tfi_temp
T = T_temp

t_list = np.array(t_list)
Gt2_list = np.array(Gt2_list)

log_Gt2_tfi = np.log10(Gt2_list)
log_t_universal = np.log10(t_list * h_0f**2)

# Save output
path='build/h0i='+str(h_0i)+'_to_h0f='+str(h_0f)+'/L='+str(L_TFI)+'/T='+str(T)+'/t='+str(t_max)+'/'
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path+'t.out',t_list)
np.savetxt(path+'Gt2_xx.out',Gt2_list)
np.savetxt(path+'Gt2_tfi.out',Gt2_list_tfi)
np.savetxt(path+'log_Gt2_tfi.out',log_Gt2_tfi)

# Fitting
def line(x,m,b):
    x = np.array(x)
    return m * x + b

def log(x,m,b):
    x = np.array(x)
    return m * np.log(x) + b

def log10(x,m,b):
    x = np.array(x)
    return m * np.log10(x) + b
#
## Load a run
#omega = 0.02
#T = 1/omega
#
#path='build/h0i='+str(h_0i)+'_to_h0f='+str(h_0f)+'/L='+str(L_TFI)+'/T='+str(T)+'/t='+str(t_max)+'/'
#run = loadrun(path)
#Gt2_list_tfi = run[0]
#Gt2_list = run[1]
#log_Gt2_tfi = run[2]
#t_list = run[3]

# Plotting

## Loschmidt echo
#plt.figure()
#plt.plot(t_list,Gt2_list_tfi,label='Drive')
#if ifHigh:
#    plt.plot(t_list,Gt2_list_tfi_high,label='High frequency')
#    plt.legend(loc='best')
#plt.xlabel('t')
#plt.ylabel('|G(t)|^2')
#plt.title('Loschmidt echo TFI')
#plt.legend(loc='best')
#

#transients = 4 # time index after which transients have dissipated
#line_params, line_cov = op.curve_fit(line,log_t_universal[transients:],log_Gt2_tfi[transients:])

indep = 0 * Gt2_list_tfi
for N in range(int(np.floor(t_max / (2 * T)))):
    indep[N * 2*n : (N + 1) * 2*n] = Gt2_list_tfi[:2*n] * Gt2_list_tfi[2*n-1]**N

## Log Loschmidt echo
plt.figure()
plt.plot(t_list,Gt2_list_tfi,label='data')
plt.plot(t_list,indep,label='independent quenches')
plt.title('Loschmidt echo, L = '+ str(L_TFI)+' \n hb = '+str(h_0f)+', omega = '+str(omega))
plt.ylabel('|G(t)|^2')
plt.xlabel('t')
plt.legend()

#plt.figure()
#plt.plot(log_t_universal,log_Gt2_tfi,label='Drive')
#plt.xlabel('Log_10 (N * hb^2 / omega)')
#plt.ylabel('Log_10 |G(N * hb^2 / omega)|^2')
#plt.title('Loschmidt echo TFI')
#plt.legend(loc='best')

#plt.figure()
#plt.plot(log_t_0_1,log_Gt2_0_1,'b--',label='hb = 0.1')
#plt.plot(log_t_0_1,log_Gt2_0_1,'bo')
#plt.plot(log_t_0_2,log_Gt2_0_2,'k--',label='hb = 0.2')
#plt.plot(log_t_0_2,log_Gt2_0_2,'ko')
#plt.plot(log_t_0_3,log_Gt2_0_3,'r--',label='hb = 0.3')
#plt.plot(log_t_0_3,log_Gt2_0_3,'ro')
#plt.plot(log_t_0_4,log_Gt2_0_4,'g--',label='hb = 0.4')
#plt.plot(log_t_0_4,log_Gt2_0_4,'go')
#plt.xlabel('Log_10 hb^2 t')
#plt.ylabel('Log_10 |G(hb^2 t)|^2')
#plt.title('Loschmidt echo TFI single quench, L = '+str(L_TFI))
#plt.legend(loc='best')

#log_t_universal_0_02 = log_t_universal
#log_Gt2_tfi_0_02 = log_Gt2_tfi
##
#plt.figure()
#plt.plot(log_t_0_5,log_Gt2_0_5,'b',label='u = 0.5')
#plt.plot(log_t_0_5,log_Gt2_0_5,'bo')
#plt.plot(log_t_1,log_Gt2_1,'k--',label='u = 1')
#plt.plot(log_t_1,log_Gt2_1,'ko')
#plt.plot(log_t_1_5,log_Gt2_1_5,'r--',label='u = 1.5')
#plt.plot(log_t_1_5,log_Gt2_1_5,'ro')
#plt.xlabel('Log_10 (N hb^2 / omega)')
#plt.ylabel('Log_10 |N G(hb^2 / omega)|^2')
#plt.title('Loschmidt echo TFI intermediate regime, L = '+ str(L_TFI)+' \n hb = sqrt(u) * '+str(h_0f)+', omega = u * '+str(omega))
#plt.legend(loc='best')

#plt.figure()
#plt.plot(log_t_1,np.abs(( log_Gt2_1_5[:log_Gt2_1.shape[0]] - log_Gt2_1 ) / log_Gt2_1),label='abs((u=1.5 - u=1)/u=1)')
#plt.plot(log_t_0_5,np.abs(( log_Gt2_1_5[:log_Gt2_0_5.shape[0]] - log_Gt2_0_5 ) / log_Gt2_0_5),label='abs((u=1.5 - u=0.5)/u=0.5)')
#plt.xlabel('Log_10 (N hb^2 / omega)')
#plt.ylabel('Log_10 L(t) percent difference')
#plt.title('Loschmidt echo percentage difference, L = '+ str(L_TFI)+' \n hb = sqrt(u) * '+str(h_0f)+', omega = u * '+str(omega))
#plt.legend(loc='best')
#
#plt.figure()
#plt.plot(log_t_1,np.abs(log_Gt2_1_5[:log_Gt2_1.shape[0]] - log_Gt2_1),label='abs(u=1.5 - u=1')
#plt.plot(log_t_0_5,np.abs(log_Gt2_1_5[:log_Gt2_0_5.shape[0]] - log_Gt2_0_5),label='abs(u=1.5 - u=0.5')
#plt.xlabel('Log_10 (N hb^2 / omega)')
#plt.ylabel('Log_10 L(t) absolute difference')
#plt.title('Loschmidt echo absolute difference, L = '+ str(L_TFI)+' \n hb = sqrt(u) * '+str(h_0f)+', omega = u * '+str(omega))
#plt.legend(loc='best')

#periods = range(0,N_tot+1)
#transients = 40 # number of periods after which transients have dissipated
#line_params, line_cov = op.curve_fit(line,periods[transients:],log_Gt2_tfi[2 * transients::2*n])
#log_params, log_cov = op.curve_fit(log,periods[transients:],log_Gt2_tfi[2 * transients::2*n])

#plt.figure()
#plt.plot(np.log(periods),log_Gt2_tfi[::2*n],label='data')
##plt.plot(periods,line(periods,line_params[0],line_params[1]),'k--',label='log(L(NT)) = '+str(line_params[0] / (2*T))+' * 2T * N')
##plt.plot(periods,line(periods,log_params[0],log_params[1]),'k--',label='log(L(NT)) = '+str(line_params[0])+' * log(N) + const')
##plt.plot(periods,line(periods,-0.25 * np.log(T),line_params[1]),'r--',label='- alpha * log(T) * N')
#plt.xlabel('Log N')
#plt.ylabel('Log |G(t)|^2')
#plt.title('L(NT) scaling with N, h_0f=' + str(h_0f)+', omega='+str(omega)+', L='+str(L_TFI))
#plt.legend(loc='lower left')


#plt.figure()
#plt.plot(np.log(t_list),log_Gt2_tfi_0_05,label='omega = 0.05')
#plt.plot(np.log(t_list),log_Gt2_tfi_0_3,label='omega = 0.3')
#plt.plot(np.log(t_list),log_Gt2_tfi_10,label='omega = 10')
#plt.xlabel('Log t')
#plt.ylabel('Log |G(t)|^2')
#plt.title('Loschmidt echo TFI')
#plt.legend(loc='best')

## Local Loschmidt echo
#cut = 10
#plt.figure()
#plt.plot(np.array(t_list),np.array(Gt2_local_list)[:,cut])

#skip = 15
#plt.figure()
#plt.plot(t_list,Gt2_0_02,label='I: omega = 0.02')
#plt.plot(np.array(t_list)[::skip],Gt2_0_02_indep[::skip],'mx',label='I: omega = 0.02, L(T)^N')
##plt.plot(t_list,Gt2_0_05,label='I: omega = 0.05')
#plt.plot(t_list,Gt2_0_3,label='II: omega = 0.3')
#plt.plot(t_list,Gt2_10,'k',label='III: omega = 10')
#plt.plot(np.array(t_list)[::skip],np.array(Gt2_control)[::skip],'rx',label='hb = 0.15')
#plt.legend(loc='best')
#plt.ylim([0,1])
#plt.xlabel('Time')
#plt.ylabel('Loschmidt echo, |G(t)|^2')
#plt.title('Loschmidt echo, hb = '+str(h_0f)+', L = '+str(L_TFI))
