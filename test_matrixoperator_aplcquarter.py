#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:33:38 2025

@author: mndiaye
"""

#%%
"""
### initialization
"""
import numpy as np

try:
    import gurobipy as gb
except ImportError:
    gb = False
    print('gb is False')   
    
import time

import matplotlib.pyplot as plt
    
#%%
"""
### Parameters
"""
# Gurobi parameters
slvLogToConsole = 1
slvCrossover    = 0
slvMethod       = 2
slvSparse       = 0
allLogToConsole = 1

# pupil sampling
nPup= 100
# secondary mirror size in pupil diameter
ID = 0.14

# mask radius in lam0/D units
#rMask = 1.766 # ALC1 at 1.593um (145mas) 
rMask = 2.252 # ALC2 at 1.593um (185mas)

# focal plane mask sampling
nFPM = 50

# Image smapling
Fmax2d = 45#22.5
nImg2d = 90#45

# parameter for the slow fourier transform
mB = rMask*1.
mD = Fmax2d*1.

# dark zone bounds (inner and outer edges) in lam0/D unit
rho0 =  0.0
rho1 = 20.0

# position of the center in the pupil and in the image
CtrBtwnPix  = True
CtrBtwnPix2 = True

# Optimization problem type
problem_name = 'MaxContrastL1'

# tau (integrated Pupil transmission)
tau   = 0.5

# Plot parameters
vmin0 = -8
vmax0 = 0

dtype0 = 'float64'
ImPart = False
if ImPart is True:
    dtype0 = 'complex128'


#%%
def sft_qrt_gb(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    if A2.ndim == 2:
        NA    = 2*np.shape(A2)[0]
    elif A2.ndim == 1:
        NA    = 2*int(np.round(np.sqrt(np.shape(A2)[0])))
    else:
        raise ValueError('incorrect number of matrix dimensions')
    coeff = (m)/(NA*NB)
    
    U = np.zeros((1,NB//2))
    X = np.zeros((1,NA//2))
    
    X[0,:] = (1./NA)*(np.arange(NA//2)+val)
    U[0,:] = (m/NB)*(np.arange(NB//2)+val)
       
    XU = 2.*np.pi* X.T.dot(U)
    A3 = np.cos(XU)
    A1 = A3.T
    
    #B  = A1.dot(A2.dot(A3))
    if A2.ndim == 2:
        B = 4*coeff*np.kron(A1, A3.T) @ A2.reshape(-1)
    else:
        B = 4*coeff*np.kron(A1, A3.T) @ A2

    return B

#%%
def isft_qrt_gb(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft_qrt_gb(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)

#%%
def compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, corono=None):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    field_A_qrt    = Pupil2d_qrt.reshape(-1)*Apod2d_qrt.reshape(-1)
                                      
    field_B_qrt    = mask2d_qrt.reshape(-1)*sft_qrt_gb(field_A_qrt, nFPM, mB, 
                                    CtrBtwnPix=True)
    field_C_qrt    = field_A_qrt - isft_qrt_gb(field_B_qrt, nPup, mB, 
                                   CtrBtwnPix=True)
    field_L_qrt    = LyotStop2d_qrt.reshape(-1)*field_C_qrt
    field_Dtmp_qrt = sft_qrt_gb(field_L_qrt, nImg2d, mD, 
              CtrBtwnPix=True)
                     
    return field_Dtmp_qrt

#%%
def uniform_disk(n, radius, CtrBtwnPix=False):
    """
    Generates a uniform disk in a 2D array.
    
    Parameters
    ----------
    n : integer
        size of the array
    
    radius : float
        radius of the disk
    
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ----------
    res : array_like
        (ys x xs) array with a uniform disk of radius "radius".
        
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2 
    xx,yy  = np.meshgrid(np.arange(n)-n/2+val, np.arange(n)-n/2+val)
    mydist = np.hypot(yy,xx)
    res    = np.zeros_like(mydist)
    # res[mydist <= radius] = 1.0
    res[mydist < radius] = 1.0
    return res

#%%
def sft(A2, NB, m, inv=False, CtrBtwnPix=False):
    """
    Slow Fourier Transform, using the theory described in [1]_. 
    Assumes the original array is square. 

    Parameters
    ----------
    A2 : array_like
        the 2D original array
    
    NB : int
        the linear size of the resulting array (integer)
    
    m : float
        m/2 = maximum spatial frequency to be computed (in lam/D)
    
    inv : boolean (default=False)
        boolean (direct or inverse) see the definition of isft()
        
    CtrBtwnPix : boolean (default=False)
        type of centering for the disk. If True, the disk is centered between
        four pixels.
    
    Returns
    ---------
    res : array_like
        Fourier transform of the array A2 within array of dimensions NBxNB
    
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
    
    """
    val    = 0
    if CtrBtwnPix is True:
        val = 1/2
    NA    = np.shape(A2)[0]
    coeff = m/(NA*NB)
    
    sign = -1.0
    if inv:
        sign = 1.0

    U = np.zeros((1,NB))
    X = np.zeros((1,NA))
    
    X[0,:] = (1./NA)*(np.arange(NA)-NA/2.+val)
    U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.+val)
       
    XU = 2.*np.pi* X.T.dot(U)
    A3 = sign*1j*np.sin(XU)  +np.cos(XU)
    A1 = A3.T
    
    B  = A1.dot(A2.dot(A3))

    return coeff*B

#%%
def isft(A2, NB, m, CtrBtwnPix=False):
    """
    Explicit inverse Slow Fourier Transform, using the theory described in [1].

    See Also
    --------
    sft() : Slow Fourier Transform
        
    References
    ---------
    
    .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
        of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
        p. 15935 (2007).
        https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
    """
    return sft(A2, NB, m, inv=True, CtrBtwnPix=CtrBtwnPix)


#%%
"""
### Entrance pupil
"""
Pupil2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)-uniform_disk(nPup, ID*nPup/2., CtrBtwnPix=CtrBtwnPix)
Pupil2d_qrt = Pupil2d[nPup//2:,nPup//2:]

#%%
"""
### Focal plane mask
"""
mask2d = uniform_disk(nFPM, nFPM/2., CtrBtwnPix=CtrBtwnPix)
mask2d_qrt = mask2d[nFPM//2:,nFPM//2:]

#%%
"""
### Lyot Stop
"""
LyotStop2d = uniform_disk(nPup, nPup/2., CtrBtwnPix=CtrBtwnPix)-uniform_disk(nPup, ID*nPup/2., CtrBtwnPix=CtrBtwnPix)
LyotStop2d_qrt = LyotStop2d[nPup//2:,nPup//2:]

#%%
"""
### Selection of the points in the pupil plane
"""
Pupil1d = np.reshape(Pupil2d, (nPup)**2)
Pupil1d_qrt = np.reshape(Pupil2d_qrt, (nPup//2)**2)

bbb  = np.arange(nPup**2)
bbb_qrt  = np.arange((nPup//2)**2)

idx_pup = list(bbb_qrt[Pupil1d_qrt == 1])
npp     = len(idx_pup)
idx_out2d = Pupil2d_qrt == 0
TR = np.sum(Pupil1d_qrt)

#%%
"""
### Selection of the points in the image plane
"""
xx,yy  = np.meshgrid(np.arange(nImg2d)-nImg2d//2+1/2, 
                     np.arange(nImg2d)-nImg2d//2+1/2)
dist2d = (Fmax2d/nImg2d)*np.hypot(yy,xx)
dz2d = (dist2d <= rho1)*(dist2d >= rho0)
dz1d = np.reshape(dz2d, (nImg2d)**2)
aaa  = np.arange(nImg2d**2)

dist2d_qrt = dist2d[nImg2d//2:, nImg2d//2:]
dz2d_qrt = dz2d[nImg2d//2:, nImg2d//2:]
dz1d_qrt = np.reshape(dz2d_qrt, (nImg2d//2)**2)
aaa_qrt = np.arange((nImg2d//2)**2)

idx_dz = list(aaa_qrt[dz1d_qrt == True])
ndz    = len(idx_dz) 

#%%
"""
### Epsilon variable for the optimization problem
"""
neps = 0
if problem_name == 'MaxContrastLinf':
    neps = 1
elif problem_name == 'MaxContrastL1':
    neps = ndz*1

#%%
"""
### Gurobi model
"""
print('generating gurobi model')
t00=time.time()
# Create a new model  
model = gb.Model("LP max C new")

Apod2d_qrtvar = model.addMVar((nPup//2, nPup//2), lb=0., ub=1., name="Apo")

#%%
t0 = time.time()
operator_Psi_D = sft_qrt_gb(Apod2d_qrtvar, nImg2d, mD, CtrBtwnPix=CtrBtwnPix2)
operator_Psi_D = compute_corono_field_2d_qrt(Apod2d_qrtvar, Pupil2d_qrt, LyotStop2d_qrt)
t1 = time.time()
print(f'computation time: {t1-t0:.2f}s')

#%%
# Create variables
Eps = model.addMVar(neps, lb=0.0, name="Eps")

# Set objective
model.setObjective(Eps.sum(), gb.GRB.MINIMIZE)

# Add constraint:
model.addConstr( operator_Psi_D[idx_dz] - Eps <= 0)
model.addConstr(-operator_Psi_D[idx_dz] - Eps <= 0)  
model.addConstr(Apod2d_qrtvar[idx_out2d] == 0)

model.addConstr(-Apod2d_qrtvar.sum()  <= -tau*TR)  

# Update model
model.update()
t11=time.time()
print(f'gurobi model computation time: {t11-t00:.3f}s\n')

#%%
"""
### Solving of the model
"""
print('solving problem with gurobipy package')


# solve problem with gurobipy package
t0 = time.time()
try:               
    model.Params.Method       = slvMethod
    model.Params.LogToConsole = slvLogToConsole
    model.Params.Crossover    = slvCrossover
        
    print('preparing to save optimization problem')
    print('ok')
    
    model.optimize()
    
except gb.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

t1 = time.time()
print(f'optimization time             : {t1-t0:.2f}s\n')

#%% Display of the apodizer
"""
### Generation of full apodizer for quarter pupil optimization
"""
print('generation of the final apodizer')
t0 = time.time()

Apod2d = np.zeros((nPup, nPup))

Apod2dtmp =  Apod2d_qrtvar.x
Apod2d[nPup//2:, nPup//2:] = Apod2dtmp
Apod2d[:nPup//2, nPup//2:] = np.flip(Apod2dtmp, axis=0)
Apod2d[:, :nPup//2]        = np.flip(Apod2d[:, nPup//2:], axis=1)
        
t1 = time.time()
print(f'apodizer generation time             : {t1-t0:.3f}s\n')

#%%
"""
### Compute final images
"""
Psi_B = sft(Pupil2d*Apod2d, nFPM, mB, CtrBtwnPix=CtrBtwnPix2)
Psi_B *= mask2d
Psi_C = Pupil2d*Apod2d - isft(Psi_B, nPup, mB, CtrBtwnPix=CtrBtwnPix2)
Psi_C *= LyotStop2d
Psi_D = sft(Psi_C, nImg2d, mD, CtrBtwnPix=CtrBtwnPix2)

Psi_D0 = sft(Pupil2d*Apod2d*LyotStop2d, nImg2d, mD, CtrBtwnPix=CtrBtwnPix2)

Int_D0 = np.abs(Psi_D0)**2
Int_D = np.abs(Psi_D)**2

norm_Int_D0 = 1./np.max(Int_D0)

Int_D0 *= norm_Int_D0
Int_D *=  norm_Int_D0

#%%
"""
### Display result
"""
plt.figure(0)
plt.clf()
plt.subplot(121)
plt.imshow(Pupil2d)
plt.subplot(122)
plt.imshow(Apod2d)

#%%
plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(np.log10(Int_D0), vmin=vmin0, vmax=vmax0)
plt.subplot(122)
plt.imshow(np.log10(Int_D), vmin=vmin0, vmax=vmax0)

