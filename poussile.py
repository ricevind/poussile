# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:51:51 2016

@author: ricevind
"""

import numba as nb
import timeit as ti
from numpy import *; from numpy.linalg import *; import numpy as np
import matplotlib.pyplot as plt; from matplotlib import cm
import symarch; import flow; import profile
import os

##############Archiwizacja#####################################################
notatki = " liniowa zależność na wylocie 2xf-2 - f-3 + uy[:,-1]=0"
r = symarch.create_proba('poussile0')
proba, zdjecia, obliczenia, run = symarch.create_run(r)# rewrite=1, test='run002')#, rewrite=1, test='run001')

#### New Re and omega calculations ############################################
H = 0.05 # wielkość geometryczna harakterystyczna
LdH = 10 # stosunek dlugosci do wielkosci harakterystycznej
L=H*LdH
U = 0.01 # predkość harakterystyczna przeplywu
ni = ni = 0.035*1e-4 # lepkość kinematyczna wody


Re = H*U/ni # Realne Re


# Przejście bezwymiarowe
t0p = H/U # czas odniesienia
ud = U*t0p/H # bezwymiarowa prędkość harakterystyczna

ny = 40 # wybieram


dy = 1/ny #jednostka siatki geometryczna w kierunku harakterystycznym
dx = dy #jednostka siatki geometryczna taka sam dy (kwadratowa siatka)
nx = np.int((LdH)/dx)

uLB = 0.1# wybieram prędkość siatki jaka odpowiada U
dt = uLB * dy /ud # wyliczam czas siatki

niLB = (dt/dy**2)*1/Re
omega = 1.0 / (3.*niLB+0.5)
tau = 1/omega
aomega = np.ones((ny,nx))*omega


print('tau = {}'.format(1/omega))
#### Profil kwadratowy wlot
a = uLB/((ny**2)/4 - (ny**2)/2)
x = np.linspace(0,ny,ny)
profiles = lambda x: a*x*(x-ny)
puLB = uLB*np.ones(x.shape)#profiles(x)


parametry = ['Re', 'ny', 'nx', 'uLB', 'tau', 'notatki']
parametryd = {key:eval(key) for key in parametry}
symarch.arch_param(proba, parametryd, parametry)
##### Lattice Constants ######################################################
c = array([[0,0], [1,0],[0,1],[-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1, -1]]) # Lattice velocities.
q = 9

noslip = np.array([c.tolist().index((-c[i]).tolist()) for i in range(q)] )
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.




############################## Funkcje ########################################

sumpop = lambda f: sum(f,axis=0)

@nb.jit(nopython=True)  
def rownowaga(rho, velo, tau=0, g=0):
    feq = zeros((9,ny,nx))
    f1 = 3. # 1 róznica dla k = 1 
    f2 = 9/2.
    f3 = 3./2
    u = velo[0,:,:]
    v = velo[1,:,:]
    for i in range(0, nx):
        for j in range(0, ny):
            rt0 = (4./9)*rho[j,i]
            rt1 = (1./9)*rho[j,i]
            rt2 = (1./36)*rho[j,i]  
            
            ueq = u[j,i] + tau*g
            veq = v[j,i]
            usq = ueq*ueq
            vsq = veq*veq
            
            uv5 = ueq + veq
            uv6 = -ueq + veq
            uv7 = -ueq - veq
            uv8 = ueq - veq
            uvsq = usq + vsq
            
            feq[0, j,i] = rt0*(1                        - f3*uvsq)
            feq[1, j,i] = rt1*(1. + f1*ueq + f2*usq     - f3*uvsq)
            feq[2, j,i] = rt1*(1. + f1*veq + f2*vsq     - f3*uvsq)
            feq[3, j,i] = rt1*(1. - f1*ueq + f2*usq     - f3*uvsq)
            feq[4, j,i] = rt1*(1. - f1*veq + f2*vsq     - f3*uvsq)
            feq[5, j,i] = rt2*(1. + f1*uv5 + f2*uv5*uv5 - f3*uvsq)
            feq[6, j,i] = rt2*(1. + f1*uv6 + f2*uv6*uv6 - f3*uvsq)
            feq[7, j,i] = rt2*(1. + f1*uv7 + f2*uv7*uv7 - f3*uvsq)
            feq[8, j,i] = rt2*(1. + f1*uv8 + f2*uv8*uv8 - f3*uvsq)        
    return feq
    
@nb.jit(nopython=True)      
def stream(fin):
    f = fin.copy()
    for j in range(ny):
        for i in range(nx-1, 0, -1):
            f[1, j, i] = f[1, j, i-1]
        for i in range(0, nx-1):
            f[3, j, i] = f[3, j, i+1]
            
    for j in range(ny-1):
        for i in range(nx):
            f[2, j, i] = f[2, j+1, i]
        for i in range(nx-1,0, -1):
            f[5, j, i] = f[5, j+1, i-1]
        for i in range(0, nx-1):
            f[6, j, i] = f[6, j+1, i+1]
    
    for j in range(ny-1,0,-1):
        for i in range(nx):
            f[4, j, i] = f[4, j-1, i]
        for i in range(0, nx-1):
            f[7, j, i] = f[7, j-1, i+1]
        for i in range(nx-1,0, -1):
            f[8, j, i] = f[8, j-1, i-1]
    return f
    

@nb.jit(nopython=True)
def obstacling(f, obstacle):
    fout = f.copy()
    for i in range(nx):
        for j in range(ny):
            if obstacle[j,i] == 1: 
                temp   = fout[1,j,i]; fout[1,j,i] = fout[3,j,i]; fout[3,j,i] = temp;
                temp   = fout[2,j,i]; fout[2,j,i] = fout[4,j,i]; fout[4,j,i] = temp;
                temp   = fout[5,j,i]; fout[5,j,i] = fout[7,j,i]; fout[7,j,i] = temp;
                temp   = fout[6,j,i]; fout[6,j,i] = fout[8,j,i]; fout[8,j,i] = temp;
    return fout
 
 
 
@nb.jit(nopython=True)     
def ulocity(fin, rho):
    u = np.zeros((2, ny, nx))
    for i in range(nx):
        for j in range(ny):
            if  not obstacle[j,i]:
                usum = 0.
                vsum = 0.
                for k in range(9):
                    usum += fin[k,j,i]*c[k,0]
                    vsum += fin[k,j,i]*c[k,1]
                u[0,j,i] = usum/rho[j,i]
                u[1,j,i] = vsum/rho[j,i]
            else:
                u[0,j,i] = 0
                u[1,j,i] = 0
    return u
    
@nb.jit(nopython=True)    
def collision(fin, feq, omega):
    fout = fin.copy()
    for i in range(nx):
        for j in range(ny):
            if not obstacle[j,i]:
                for k in range(q):
                    fout[k,j,i] = fin[k,j,i] - omega[j,i]*(fin[k,j,i]-feq[k,j,i])
    return fout
   
@nb.jit(nopython=True)    
def shear(fin, feq, omega):
    tau = 1/omega
    om = -3/(2*tau)
    srate = np.ones((ny,nx))
    fneq = fin - feq
    for i in range(nx):
        for j in range(ny):
            S00 = 0
            S01 = 0
            S10 = 0
            S11 = 0
            for k in range(q):
                S00 += fneq[k,j,i]*c[k,0]*c[k,0]
                S01 += fneq[k,j,i]*c[k,0]*c[k,1]
                S10 += fneq[k,j,i]*c[k,1]*c[k,0]
                S11 += fneq[k,j,i]*c[k,1]*c[k,1]
            srate[j,i] = np.sqrt(2*((om[j,i]*S00)**2 + (om[j,i]*S11)**2 + (om[j,i]*S01)**2 + (om[j,i]*S10)**2))
    return srate
    
def power_law(s,n=1,m=niLB,li=100):
    ni = m*s**(n-1)
    a = ni > ni*li
    b = ni < ni/li
    ni[a] =ni[a]*li
    ni[b] =ni[b]/li
    omega =  1.0 / (3.*ni+0.5)
    return omega
    
####################################### Inicjalizacja #########################
    
vel = np.ones((2,ny,nx))*uLB*0 # pole prędkości w t = 0
rho = np.ones((ny, nx)) # pole gęstości w t = 0
feq = rownowaga(rho, vel) # równowagowa funkcja rozkładu
f = feq.copy() # tablica rozkładu dla t-1
f_history = np.ones((9,ny,nx,100))

image = 0
count=0
######################### Pętla################################################
maxIter =15000*5 # liczba iteracji
u = vel
u1 = u.copy()
u1[0,1:-1,0] = puLB[1:-1]

t0 = ti.default_timer()

rho_history = []
th = 0
############################## Geometria ######################################
obstacle = fromfunction(lambda x,y: x>10000000 , (ny,nx))       
#obstacle = fromfunction(lambda y,x: (x-100)**2+(y-20)**2<10**2, (ny,nx))
obstacle[0,:] = 1; obstacle[-1,:] = 1
for time in range(maxIter):
    rho = sumpop(f)
    rho_history.append(abs(average(rho)))
    ##Implementacja warunków brzegowych
    # Wlot    
    rho[1:-1,0] = 1./(1.-u1[0,1:-1,0]) * (sumpop(f[i2,1:-1,0])+2.*sumpop(f[i1,1:-1,0]))

    f[1,1:-1,0] = f[3,1:-1,0] + (2/3)*rho[1:-1,0]*u1[0,1:-1,0]
    f[5,1:-1,0] = f[7,1:-1,0] - (1/2)*(f[2,1:-1,0] - f[4,1:-1,0]) + (1/6)*rho[1:-1,0]*u1[0,1:-1,0] 
    f[8,1:-1,0] = f[6,1:-1,0] + (1/2)*(f[2,1:-1,0] - f[4,1:-1,0]) + (1/6)*rho[1:-1,0]*u1[0,1:-1,0] 
    # Wylot
    f[i1,1:-1,-1] = 2*f[i1,1:-1,-2] - f[i1,1:-1,-3] 
    
    ## Pole prędkości
    u = ulocity(f, rho)
    u[1,:,-1] = 0
    
    ## Lattice Boltzmann
    feq = rownowaga(rho,u)
    ## Uwzględnienie lepkości ######
#    s = shear(f,feq,aomega)     ###
#    aomega = power_law(s,1)     ###
    ################################
    f = collision(f,feq,aomega)
    f = obstacling(f, obstacle)
    f = stream(f)
    ## Zapis f
    if time > int(maxIter - 1/dt):
        f_history[:,:,:,th] = f
        th = th + 1
        if th == 100:
            np.save(os.path.join(obliczenia, 'f{0:06d}'.format(image)), f_history)
            th = 0
    ##Wizualizacja
    if (time%100==0): # 
        t1 = ti.default_timer()
        deltaT = t1 - t0
        print('100 petli wykonuje sie ', deltaT, ' sekund', count)
        print('###########################')
        print( 1/omega,average(rho))
        t0 = ti.default_timer()
        
#        plt.subplot(4, 1, 1)
#        plt.imshow(u[1,:,0:50],vmin=-uLB*.15, vmax=uLB*.15, interpolation='none')#,cmap=cm.seismic
#        plt.colorbar()

        plt.subplot(3, 1, 1)
        plt.imshow(sqrt(u[0]**2+u[1]**2),vmin=0, vmax=uLB*1.5)#,cmap=cm.seismic
        plt.colorbar()
        plt.title('tau = {:f}'.format(1/omega))        
        
        plt.subplot(3, 1, 2)
        plt.imshow(rho,vmin=0, vmax=1.5 )#,cmap=cm.seismic
        plt.title('rho')   
        
        plt.subplot(3, 1,3)
        plt.title(' history rho')
        plt.plot(linspace(0,len(rho_history),len(rho_history)),rho_history)
        plt.xlim([0,maxIter])        
        
        plt.savefig(os.path.join(zdjecia,'f{0:06d}.png'.format(image)))
        plt.clf();
        image += 1
        count += 1

np.save(os.path.join(obliczenia, 'f{0:06d}'.format(image)), f_history)
#########CURVE FITTING #######################################################

flow.animate(zdjecia, run, proba)
profile.save_profile(proba, run, ny,nx,  u, uLB,rho)

x = linspace(0,ny,len(u[1,:,0]))
y = sqrt(u[0]**2+u[1]**2)[:,-1]/uLB
#y = u[1][:,-1]/uLB

w = np.ones(len(x))
w[[1,-2]] = 1000
z = np.polyfit(x[1:-1],y[1:-1],2,)
ff = np.poly1d(z)

xx = linspace(0,ny,25)
yy = ff(xx)

plt.plot(x,y)
plt.plot(xx, yy, 'o')