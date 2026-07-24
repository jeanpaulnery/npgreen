#!/usr/bin/env python
"""
Main execution script for NPGreen.
Reads input parameters, calculates the overlaps,
and computes both the non-perturbative and pseudo-AHC Green's functions

To run, basically do:
conda activate python_netcdf
python nonperturbative.py  # --Ns 8 --Ncfgs 1 --nband 5 --calc_overlaps

"""
import netCDF4 as nc
import numpy as np
import xarray as xr
import argparse

from overlaps import *
from green_function import *
from constants import *
from inputs import get_input_params

import sys
from time import time

t0 = time()

# GET PARAMETERS AND PATHS
calc_overlaps, Ns, nbuf, Ncfgs, Ncfgs_GSigma, ind, Temp, ik, nband, delta_eV, delta, lenw, root, ext_PC, ext_SC, calc_type, all_k = get_input_params()

# READ PRIMITIVE CELL STUFF
ds = nc.Dataset(f"{root}/{ext_PC}/runo_WFK.nc",'r') 
eps0 = ds["eigenvalues"][0]

if calc_type != "standard":
    if eps0.shape[0] not in (8**3,6**3):
        import sys
        print(f"Error: Number of k-points in Primitive Cell is not 512!")
        sys.exit(1)

# CALCULATE OR GET OVERLAPS
if calc_overlaps:
    overlaps = build_overlaps(Ns,Ncfgs,nband,root,ext_PC,ext_SC,nbuf,ind,ik,all_k=all_k)  # TODO: Remove coeffs_squared
else:
     fname = f"overlaps{ind}.nc" if calc_type=="standard" else "overlaps_all.nc"
     overlaps = xr.load_dataset(f"{root}/{ext_SC}/{fname}")["overlaps"][:].values
     overlaps = overlaps[0] + 1j*overlaps[1]


# CALCULATE G AND SIGMA
# Initialize base physics objects
w = eps0[ik,3] + np.linspace(-5,14,lenw,endpoint=False)/Ha_to_eV 
Sigma = np.zeros((len(w),nband,nband),dtype=complex)
gsigma = GSigma(w,eps0,nband,delta=delta)

#######################          
##### STANDARD ########
if calc_type == "standard":
    print("Doing standard NP calculation")
    if overlaps.ndim == 4 and overlaps.shape[2] > 1:
        print("Error: overlaps has multiple k-points. Standard NP expects single-k overlaps.")
        print("Load overlaps.nc (single k) or run with --no-calc_overlaps and the correct file.")
        sys.exit(1)
    
    tmpG = 0
    for icfg in range(Ncfgs_GSigma):
        t1 = time()
        ds_SC = nc.Dataset(f"{root}/{ext_SC}/config{icfg+1}/runo_EIG.nc", 'r') # Energy file. This we store...
        epsJ = ds_SC["Eigenvalues"][0]  # Energies of dist config, Eigenvalues(nsppol, nkpt, mband)
        gsigma.build_Ginit(epsJ,overlaps[icfg])
        tmpG += gsigma.Ginit
        print('Time elapsed: ', time() - t1)

    gsigma.Ginit = tmpG/Ncfgs_GSigma # Now Ginit stores the final averaged for this T...
    gsigma.build_Sigma()
    gsigma.build_G()
    Ginit = gsigma.Ginit
    G = gsigma.G
    Sigma = gsigma.Sigma

    # SAVE TO NETCDF
    da = xr.Dataset({ "Ginit": (("complex", "windex", "band1", "band2"), np.array([Ginit.real, Ginit.imag])),
                    "G": (("complex", "windex", "band1", "band2"), np.array([G.real, G.imag])),
                    "Sigma": (("complex", "windex", "band1", "band2"), np.array([Sigma.real, Sigma.imag])),
                    "w": w })

    delta_str = f"_d{delta_eV:f}".rstrip('0').replace('.', '')        
    save_to_netcdf(da,f"{root}/{ext_SC}/GSigma_Ncfgs{Ncfgs_GSigma}{delta_str}{ind}.nc") # TODO: Implement diagonal approximation

#######################

#######################
##### NO OFF-DIAG #####
# Test to check if removing the degenerate off-diagonal elements are important in the width
elif calc_type == "no_offd":
    print("Doing no_offd Green's function")
    gsigma.build_G_no_offd(Ncfgs_GSigma, overlaps, Ns, root, ext_PC, ext_SC)

    gsigma.build_Sigma()
    gsigma.build_G()
    Ginit = gsigma.Ginit
    G = gsigma.G
    Sigma = gsigma.Sigma

    # SAVE TO NETCDF
    da = xr.Dataset({ "Ginit": (("complex", "windex", "band1", "band2"), np.array([Ginit.real, Ginit.imag])),
                    "G": (("complex", "windex", "band1", "band2"), np.array([G.real, G.imag])),
                    "Sigma": (("complex", "windex", "band1", "band2"), np.array([Sigma.real, Sigma.imag])),
                    "w": w })
  
    delta_str = f"_d{delta_eV:f}".rstrip('0').replace('.', '')        
    save_to_netcdf(da,f"{root}/{ext_SC}/GSigma_Ncfgs{Ncfgs_GSigma}{delta_str}_removeoffd.nc")


#######################
##### PSEUDO-AHC ######
else:
    print("Doing pseudoAHC Green's function, not the standard NP procedure")
    kweights = ds['kpoint_weights'][:]*Ns**3 # Because weights are divided by N_mesh = Ns^3, so we have to multiply by it

    gsigma.build_Sigma_pseudoAHC(Ncfgs,overlaps,kweights,root,ext_SC)
    gsigma.build_G()
    G = gsigma.G
    Sigma = gsigma.Sigma

    # There is no Ginit in this case; the first thing we calculate is Sigma
    # TODO: Potential issue - When doing this approach, q=0 was missing....
    da = xr.Dataset({ "G": (("complex", "windex", "band1", "band2"), np.array([G.real, G.imag])),
                      "Sigma": (("complex", "windex", "band1", "band2"), np.array([Sigma.real, Sigma.imag])),
                      "w": w })

    save_to_netcdf(da,f"{root}/{ext_SC}/GSigma_Ncfgs{Ncfgs_GSigma}_pseudoAHC.nc") # TODO: Implement diagonal approximation
#else:
#    print("Skipping pseudo-AHC calculation (requires --all_k flag)")
#######################


print('Total time elapsed: ', time() - t0, ' s')

















