#!/usr/bin/env python
"""
Overlaps Module.
Handles the reciprocal-space mapping between Primitive Cell (PC) and Supercell (SC) G-vectors,
and computes wave-function overlaps <mk|J>^I between all configurations, and bands and k-points.
"""
import netCDF4 as nc
import numpy as np
import xarray as xr
import os
from time import time

from constants import * # Our module


def build_overlaps(Ns,Ncfgs,nband,root,ext_PC,ext_SC,nbuf,ind,ik0=0,all_k=False):

    # READ ONE SC WAVEFUNCTION: get qs (all iK blocks), nJ, and SC k-points
    print('Start building overlap stuff')
    ds_SC = nc.Dataset(f"{root}/{ext_SC}/config1/runo_WFK.nc", 'r')
    qs = ds_SC['reduced_coordinates_of_plane_waves'][:]  # (nK_SC, nq_max, 3)
    kpts_SC = ds_SC['reduced_coordinates_of_kpoints'][:]  # SC k-points in SC reduced units [0,1)
    nJ = len(ds_SC["coefficients_of_wavefunctions"][0,0,:,0,0,0]); nJ_buf = nJ - nbuf

    # READ PC WAVEFUNCTION (all k at once, outside ik loop)
    print('Load PC stuff')
    ds = nc.Dataset(f"{root}/{ext_PC}/runo_WFK.nc",'r')
    Gs = ds['reduced_coordinates_of_plane_waves'][:]  # (nk_PC, nG_max, 3)
    coeffs = ds["coefficients_of_wavefunctions"][0,:,:,0,:,0].data + 1j*ds["coefficients_of_wavefunctions"][0,:,:,0,:,1].data  # (nk_PC, nband, nG_max)

    # READ PC k POINTS
    ds = nc.Dataset(f"{root}/{ext_PC}/runo_EIG.nc",'r')
    kpts = ds['Kptns'][:]*Ns  # SC units: k_PC_frac * Ns; commensurate k-points are integers
    k_indices = [ik0] if not all_k else list(range(len(kpts)))
    nkpt = len(k_indices)

    iK_list = find_iK(kpts, kpts_SC)


    indices = []
    conj_mask = []
    coeffs_PC_matched = []
    for i_k, ik in enumerate(k_indices):

        iK = iK_list[ik] # Must use 'ik' to fetch the correct mapped SC K-point

        # APPLY SC q MASK for this iK block
        mask_q_iK = qs[iK, :, 0].mask  # same mask for any Cartesian component
        qs_iK = qs[iK, :, :] if not np.any(mask_q_iK) else qs[iK, ~mask_q_iK, :]  # guard for Gamma-only (no mask)

        # APPLY PC G MASK for this ik
        mask_G = Gs[ik,:,0].mask
        Gs_valid = Gs[ik, ~mask_G, :]
        coeffs_valid = coeffs[ik, :, :][:, ~mask_G]

        # FIND MATCHING INDICES: q0 + Ns*G in qs_iK, where q0 = k0 - K0
        k0 = kpts[ik]
        K0 = kpts_SC[iK]
        k0_minus_K0 = k0 - K0
        t0 = time()
        indices_ik, conj_mask_ik, coeffs_PC_matched_ik = get_kpoint_mapping(k0_minus_K0, Ns, Gs_valid, qs_iK, coeffs_valid)
        print(f'Time elapsed mapping {ik}: ', np.round(time() - t0,3), ' s')

        indices.append(indices_ik)
        conj_mask.append(conj_mask_ik)
        coeffs_PC_matched.append(coeffs_PC_matched_ik)

             
    # CALCULATE OVERLAPS
    overlaps = np.zeros((Ncfgs,nJ_buf,nkpt,nband),dtype=complex)

    for icfg in range(Ncfgs):

        t0 = time()
        # Load SC WF coefficients for this config (all SC k-blocks)
        ds_SC = nc.Dataset(f"{root}/{ext_SC}/config{icfg+1}/runo_WFK.nc", 'r')
        coeffs_SC = ds_SC["coefficients_of_wavefunctions"][0,:,:,0,:,0].data + 1j*ds_SC["coefficients_of_wavefunctions"][0,:,:,0,:,1].data  # (nK_SC, nJ, nq_max)

        # ik dependent part: each ik uses its own iK block of the SC coefficients
        for i_k, ik in enumerate(k_indices):
            #print('ik: ', ik)
            iK = iK_list[ik]
            mask_q_iK = qs[iK, :, 0].mask
            coeffs_SC_iK = coeffs_SC[iK, :, :] if not np.any(mask_q_iK) else coeffs_SC[iK][ :, ~mask_q_iK]  # guard for Gamma-only

            #print(not np.any(mask_q_iK))
            #print(ik, iK, len(indices[i_k]))
            #print(coeffs_SC_iK.shape)
            coeffs_SC_matched = coeffs_SC_iK[:, indices[i_k]]
            coeffs_SC_matched[:, conj_mask[i_k]] = np.conj(coeffs_SC_matched[:, conj_mask[i_k]])

            overlaps[icfg, :nJ_buf, i_k, :nband] = np.conjugate(coeffs_SC_matched[:nJ_buf, :]) @ coeffs_PC_matched[i_k][:nband, :].T

        # Check overlap normalization for all computed k-points at the first config
        if icfg == 0:
            for i_k, ik in enumerate(k_indices):
                norm = np.sum(np.abs(overlaps[0, :nJ_buf, i_k, :nband])**2, axis=0)
                if np.any(norm[:7] < 0.9):
                    raise ValueError(f"Overlap normalization too low for bands 0-6 at ik={ik}: {norm}. "
                                     f"Need more SC bands (nJ_buf={nJ_buf}).")
                elif len(norm) > 7 and np.any(norm[7:] < 0.9):
                    print(f"WARNING: Overlap normalization low for higher bands (>6) at ik={ik}: {norm}. "
                          f"You may need more SC bands (nJ_buf={nJ_buf}).")
            

        
        print(f'Time elapsed to build overlap{icfg}: ', time() - t0, 's')
        print('')

    # SAVE OVERLAPS
    #da = xr.Dataset({"overlaps": (("complex", "config", "kpts", "Jstate", "band"), np.array([overlaps.real, overlaps.imag]))})
    if not all_k:
        overlaps = overlaps[:, :nJ_buf, 0, :nband]
        da = xr.Dataset({"overlaps": (("complex", "config", "Jstate", "band"), np.array([overlaps.real, overlaps.imag]))})
        save_to_netcdf(da,f"{root}/{ext_SC}/overlaps{ind}.nc")
    # Saving overlaps for many k points in this case
    else:
        da = xr.Dataset({"overlaps": (("complex", "config", "Jstate", "kpts", "band"), np.array([overlaps.real, overlaps.imag]))})
        save_to_netcdf(da,f"{root}/{ext_SC}/overlaps_all.nc")

    return overlaps


# TODO: Consider renaming to get_G_q_mapping or get_planewaves_mapping
def get_kpoint_mapping(k0_minus_K0, Ns, Gs_valid, qs_iK, coeffs_PC):
    """
    Builds the index mapping from Primitive Cell G-vectors to Supercell q-vectors for a specific k-point.
    k0_minus_K0: residual q-vector after subtracting the SC k-point K0 from the PC k-point k0.
    qs_iK: valid SC q-vectors for the SC k-block iK that k0 folds into.
    Returns:
        indices: Array mapping directly into the valid SC qs_iK grid.
        conj_mask: Boolean array indicating if the coefficient must be conjugated.
        coeffs_PC_matched: The correctly filtered Primitive Cell coefficients.
    """
    q0_plus_G_set = np.round(k0_minus_K0 + Ns*Gs_valid).astype(int)
    
    qs_index = {tuple(row): i for i, row in enumerate(qs_iK)}

    indices = []
    conj_mask = []
    found_mask = []

    for q0_plus_G in q0_plus_G_set:
        if tuple(q0_plus_G) in qs_index:
            # Exact match found
            indices.append(qs_index[tuple(q0_plus_G)])
            conj_mask.append(False)
            found_mask.append(True)

        elif tuple(-q0_plus_G) in qs_index:
            # Symmetry match found (-q)
            indices.append(qs_index[tuple(-q0_plus_G)])
            conj_mask.append(True)
            found_mask.append(True)

        else:
            # Not found
            found_mask.append(False) 
            #dists = np.linalg.norm(qs_iK - q0_plus_G, axis=1)
            #closest_idx = np.argmin(dists)
            #if len(indices) < 3:  # print just a few
            #    print(f"  UNMATCHED: q={q0_plus_G}, closest stored q={qs_iK[closest_idx]}, dist={dists[closest_idx]:.6e}")
            #    print(f"  -q={-q0_plus_G}, closest to -q: dist={np.min(np.linalg.norm(qs_iK + q0_plus_G, axis=1)):.6e}")
            
    # Convert lists to NumPy arrays for speed
    indices = np.array(indices, dtype=int)
    conj_mask = np.array(conj_mask, dtype=bool)
    found_mask = np.array(found_mask, dtype=bool)
    
    # Filter PC coefficients to match the ones actually found
    coeffs_PC_matched = coeffs_PC[:, found_mask] # Ok recalled that coeffs_PC and Gs_valid go together; coeffs_PC = c_G

    # Validation
    n_found = np.sum(found_mask)
    n_total = len(found_mask)
    if n_found < n_total:
        raise ValueError( f"Only matched {n_found}/{n_total} G-vectors for k0_minus_K0={k0_minus_K0}. "
                          f"Likely floating-point issue in q-vector matching.")
    
    
    return indices, conj_mask, coeffs_PC_matched


def find_iK(kpts, kpts_SC):
    """
    For each PC k-point, find which SC k-point it folds into.
    kpts: PC k-points in SC units (PC fractional * Ns).
    kpts_SC: SC k-points in SC fractional units.
    Returns iK_list: list of SC k-point indices, one per PC k-point.
    """
    iK_list = []
    for ik in range(len(kpts)):
        k_folded = kpts[ik] % 1.0
        k_folded[k_folded > 0.5 + tol8] -= 1.0
        dists = np.linalg.norm(kpts_SC - k_folded, axis=1)
        iK = int(np.argmin(dists))
        if dists[iK] > 1e-6:
            print('Issue for: ', kpts[ik])
            print(f"ERROR: No SC k-point matches folded k={k_folded} (ik={ik}), min dist={dists[iK]:.2e}")
            print(f"kpts_SC = {kpts_SC}")
            raise ValueError("k-point folding mismatch")
        iK_list.append(iK)
    return iK_list


def save_to_netcdf(ds,filename):

    # Check if file exists
    if os.path.exists(filename):
        print(f"Warning: {filename} already exists!")
        
        i = 1
        while True:
            new_filename = f"{filename[:-3]}_{i:02d}.nc"
            if not os.path.exists(new_filename):
                filename = new_filename
                print(f"Saving to {filename} instead.")
                break
            i += 1
    else:
        print(f"Saving to {filename}")

    ds.to_netcdf(filename)
