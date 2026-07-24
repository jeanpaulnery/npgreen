#!/usr/bin/env python
import netCDF4 as nc
import numpy as np
import xarray as xr
from time import time

from overlaps import *
from green_function import *
from constants import *
from inputs import get_input_params

# TODO: Would have to check running the test like this works

# Get shared parameters from inputs
calc_overlaps, Ns, nbuf, Ncfgs, Ncfgs_GSigma, ind, Temp, ik, nband, delta_eV, delta, lenw, istwfk, root, ext_PC, ext_SC = get_input_params()

def calc_alternative_spectral_function():
    print("Running alternative spectral function calculation...")
    t0 = time()
    
    # Load energy data to setup W and GSigma exactly like in nonperturbative.py
    ds = nc.Dataset(f"{root}/{ext_PC}/runo_WFK.nc",'r') 
    eps0 = ds["eigenvalues"][0]
    w = eps0[ik,3] + np.linspace(-5,14,lenw,endpoint=False)/Ha_to_eV 
    
    gsigma = GSigma(w,eps0,nband,delta=delta)
    
    # Run the alternative calculation
    coeffs_squared = build_coeffs_squared(Ns,nband,root,ext_PC,ext_SC,istwfk,nbuf)
    Aalt = build_Aalt(gsigma, coeffs_squared, root, ext_SC)
    
    # Save output
    da = xr.Dataset( {"Aalt": (("windex"), Aalt), "w": w} )
    save_to_netcdf(da,f"{root}/{ext_SC}/Aalt.nc")
    
    print('Total time elapsed: ', time() - t0, ' s')

def build_coeffs_squared(Ns,nband,root,ext_PC,ext_SC,istwfk,nbuf):
    """Copied from overlaps"""

    # Define Gs_valid 
    ds = nc.Dataset(f"{root}/{ext_PC}/runo_WFK.nc",'r')
    Gs = ds['reduced_coordinates_of_plane_waves'][:] # Gs
    #coeffs = ds["coefficients_of_wavefunctions"][0,:,:,0,:,0] + 1j*ds["coefficients_of_wavefunctions"][0,:,:,0,:,1]
    mask_G = Gs[0,:,0].mask # mask_G = coeffs[0,0,:].mask
    Gs_valid = Gs[0,~mask_G,:]
    #coeffs_valid = coeffs[0,:,:][:,~mask_G]
    
    # Define the qs_valid
    ds_SC = nc.Dataset(f"{root}/{ext_SC}/config1/runo_WFK.nc", 'r')
    qs = ds_SC['reduced_coordinates_of_plane_waves'][:]
    nJ = len(ds_SC["coefficients_of_wavefunctions"][0,0,:,0,0,0]); nJ_buf = nJ - nbuf # Number of states/J's - nbdbuf
    mask_q = qs[0,:,0].mask
    qs_valid = qs[0,~mask_q,:]
    if qs_valid.shape[0]==1:qs_valid = qs_valid[0]

    # Keep only the adequate qs
    q0 = [0,0,0]
    mask = np.all( (qs_valid-q0) % Ns == 0, axis=1)
    qs_valid =qs_valid[mask]

    # Get indices corresponding to Gs_valid
    Gs_valid = Ns*Gs_valid
    lookup = {tuple(row): j for j, row in enumerate(np.array(qs_valid))}
    indices = np.array([lookup[tuple(row)] for row in np.array(Gs_valid)])
    
    # Get coeffs and keep only the adequate ones
    ds_SC = nc.Dataset(f"{root}/{ext_SC}/config1/runo_WFK.nc", 'r')
    coeffs_SC = ds_SC["coefficients_of_wavefunctions"][0,:,:,0,:,0] + 1j*ds_SC["coefficients_of_wavefunctions"][0,:,:,0,:\
,1]
    coeffs_SC_valid = coeffs_SC[0, :, :] if not np.any(mask_q) else coeffs_SC[0, :, :][:, ~mask_q]
    coeffs_SC_valid = coeffs_SC_valid[:,mask]

    # Calculate the squared coeffcients
    coeffs_squared = np.zeros(nJ_buf,dtype=complex)

    if istwfk == 1:
        for j in range(nJ_buf):
            coeffs_squared[j] = sum(np.conj(coeffs_SC_valid[j,indices[:]])*coeffs_SC_valid[j,indices[:]])

    else:
        for j in range(nJ_buf):
            coeffs_squared[j] = np.conj(coeffs_SC_valid[j,0])*coeffs_SC_valid[j,0] + \
                                2*sum(np.conj(coeffs_SC_valid[j,indices[1:]])*coeffs_SC_valid[j,indices[1:]])

    return coeffs_squared

def build_Aalt(gsigma, coeffs_squared, root, ext_SC):
    ds_SC = nc.Dataset(f"{root}/{ext_SC}/config1/runo_EIG.nc", 'r')
    epsJ = ds_SC["Eigenvalues"][0] 
    
    Aalt = 0
    for j in range(coeffs_squared.shape[0]):
        invden = 1 / (gsigma.w + 1j*gsigma.delta - epsJ[0,j])
        Aalt = Aalt + np.imag(invden) * coeffs_squared[j]
        
    Aalt = -1/np.pi * Aalt
    return Aalt.astype(np.float64) 

if __name__ == "__main__":
    calc_alternative_spectral_function()
