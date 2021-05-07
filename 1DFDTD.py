# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:03:08 2019

@author: Ehsan
"""
import math 
import numpy as np
import matplotlib.pyplot as plt

'''FDTD Engine'''
'''Units'''
meters=1;
centimeters=meters*1e-2;
milimeters=meters*1e-3;
nanometers= meters*1e-9;
feet=0.3048*meters;
seconds=1;
hertz = 1/seconds;
megahertz=1e6*hertz;
gigahertz=1e9*hertz;

'''constants'''
c0=299792458 * meters/seconds;
e0=8.8541878176e-12*1/meters;
u0=1.2566370614*1/meters;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Dashboard'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Source Parameters
fmax = 5*gigahertz;
NFREQ = 1000;
FREQ = np.linspace(0,fmax,NFREQ);

#Slab Properties
dslab= 0.25*feet;
erair=1.0;
erslab=12;

#Grid Paramaters
ermax=max(erair,erslab);
nmax=math.sqrt(ermax);
NLAM=10;
NDIM=1;
NBUFZ=[100,100];
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Compute Optimized Grid'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Nominal Resolution
lam0 = c0/fmax;
dz1=lam0/nmax/NLAM;
dz2=dslab/NDIM;
dz=min(dz1,dz2);

#Snap grid critical dimension
nz=math.ceil(dslab/dz);
dz=dslab/nz;

#Compute Grid Size
Nz = round(dslab/dz)+sum(NBUFZ)+3;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Build Device on Grid'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Initialize materials to free space
Er = np.ones(Nz);
Ur = np.ones(Nz);

#compute position indecies
nz1=2+NBUFZ[0]+1;
nz2=nz1+round(dslab/dz)-1;

#ADD slab
Er[nz1-1:nz2]=erslab;

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'BComputing the Source'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Compute the time step
nbc = math.sqrt(Ur[0]*Er[0]);
dt = nbc*dz/(2*c0);

#Compute source Parameters
tau = 0.5/fmax;
t0 = 5*tau;

#Compute the number of time steps
tprop = nmax*Nz*dz/c0;
t = 2*t0 + 3*tprop;
#steps=100;
steps = math.ceil(t/dt);

#compute the source
t=np.arange(0,steps-1)*dt;
s=dz/(2*c0)+dt/2;
nz_src = 2;
Esrc =np.exp(-(np.power((t-t0)/tau,2)));
A = -math.sqrt(Er[nz_src])/(Ur[nz_src]);
Hsrc =A*np.exp(-(np.power((t-t0)/tau,2)));

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Initialize FDTD Parameters'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Initialize Fourier Transform
K = np.exp(-1j*2*math.pi*dt*FREQ);  #Kernel update
REF = np.zeros(NFREQ);
TRN = np.zeros(NFREQ);
SRC = np.zeros(NFREQ);
RE = np.zeros(NFREQ);
TR = np.zeros(NFREQ);
Con=np.zeros(NFREQ);

#Update Coefficients
mEy = (c0*dt)/Er;
mHx = (c0*dt)/Ur;

#Initialize Fields
Ey = np.zeros(Nz);
Hx = np.zeros(Nz);

#Initilize Boundry Condition
H1=0; H2=0; H3=0;
E1=0; E2=0; E3=0;

#Plot Initialization
plot=range(steps+12)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'Perform FDTD'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Main Loop for Time

for T in range(steps+1):
    
    #update H from E
    for nz in range(Nz-1):
        Hx[nz] = Hx[nz]+mHx[nz]*((Ey[nz+1]-Ey[nz])/dz);
        
    Hx[Nz-1] = Hx[Nz-1]+mHx[Nz-1]*((E3-Ey[Nz-1])/dz);
    
    #H Source
    Hx[nz_src-1] = Hx[nz_src-1] - mHx[nz_src-1]*Esrc[T]/dz;
    
    #Record H-Field at the Boundry
    H3=H2;
    H2=H1;
    H1=Hx[0];
    
    #update E from H
    Ey[0] = Ey[0] + mEy[0]*((Hx[0]-H3)/dz);
    for nz in range(1,Nz):
        Ey[nz] = Ey[nz] + mEy[nz]*((Hx[nz]-Hx[nz-1])/dz);
        

    #E Source
    Ey[nz_src] = Ey[nz_src]-mEy[nz_src]*Hsrc[T]/dz;
    
    #Record E-Field at the Boundry
    E3=E2;
    E2=E1;
    E1=Ey[Nz-1];
    
    
    #Update Fourier Transform    
    REF=REF+((np.power(K,T))*Ey[0])
    TRN=TRN+((np.power(K,T))*Ey[Nz-1])
    SRC=SRC+((np.power(K,T))*Esrc[T])
        
    RE=np.power(np.abs(np.divide(REF,SRC)),2);
    TR=np.power(np.abs(np.divide(TRN,SRC)),2);
    Con=np.add(RE,TR);
    
        
    
    if np.mod(T,20)==0:
        p1=plt.subplot(3,1,1);
        plt.plot(range(Nz),Ey,zorder=10);
        plt.plot(range(Nz),Hx,zorder=10);
        plt.axvspan(nz1, nz2, alpha=0.5, color='Orange',zorder=1)
        plt.xlabel('Z')
        plt.ylabel('A.U')
        plt.title('E & H Intensity at the time '+str(T)+'of '+str(steps))
        plt.xlim(0,Nz);
        plt.ylim(-1,1);
        plt.legend('EH');
        
        
        p2=plt.subplot(3,1,3);
        plt.plot(FREQ,TR);
        plt.plot(FREQ,RE);
        plt.plot(FREQ,Con);
        plt.xlim(0,fmax);
        plt.ylim(0,2);
        plt.legend('TRC');
        plt.gcf()
        plt.pause(0.01)
        plt.clf()
        
        
        
    
        
        
        
        
      
        
    
   
    


