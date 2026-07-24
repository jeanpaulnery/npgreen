"""
Contains the GSigma class from which we build the initial G,
the self-energy, and the final G (without broadening of G0).
It also calculates pseudoAHC, to check if off-diagonal elements (not included) are important or not...
"""
import numpy as np
import netCDF4 as nc
import os
from constants import *
from overlaps import find_iK


class GSigma:
    def __init__(self, w, eps0, nband, delta=0.1/Ha_to_eV):
        self.w = w          # frequencies (array), in Hartree
        self.eps0 = eps0    # matrix (nkpt, nband)
        self.delta = delta
        self.nband = nband # TODO: Consider getting nband from inserting only the range of eps0 we want
        self.Ginit = np.zeros((len(w),self.nband,self.nband),dtype=complex)
        self.Sigma = np.zeros((len(w),self.nband,self.nband),dtype=complex)
        self.G = np.zeros((len(w),self.nband,self.nband),dtype=complex)

        
    def build_Ginit(self,epsJ,overlaps,nJ_test=0):
                                                
        if nJ_test > 0:
            nJ_buf = nJ_test
        else:
            nJ_buf = overlaps.shape[0]

        invden = 1.0 / (self.w[:, None] + 1j*self.delta - epsJ[0, :nJ_buf][None, :])
        self.Ginit[:, :self.nband, :self.nband] = np.einsum(
            'wj,jn,jm->wnm',
            invden,
            np.conjugate(overlaps[:nJ_buf, :self.nband]),
            overlaps[:nJ_buf, :self.nband],
            optimize=True )


    def build_Sigma(self):
        "Build Sigma from Ginit"

        Ginv = np.linalg.inv(self.Ginit)
        I = np.eye(self.nband) # self.eps0.shape[0])
        wI = (self.w + 1j*self.delta)[:, None, None] * I[None, :, :]
        eps_diag = np.diag(self.eps0[0, :self.nband])[None, :, :] # TODO: Verify if this is a genuine issue or just a centering artifact
        # ik is first index. But like this we get all spectral functions in the same range...
        
        self.Sigma = wI - eps_diag - Ginv


    def build_G(self):
        "Build the final G from Sigma"

        I = np.eye(self.nband)
        wI = self.w[:, None, None] * I[None, :, :]
        eps_diag = np.diag(self.eps0[0, :self.nband])[None, :, :]
        
        self.G = np.linalg.inv(wI - eps_diag - self.Sigma)


    def build_G_no_offd(self, Ncfgs, overlaps, Ns, root, ext_PC, ext_SC):
        nJ_buf = overlaps.shape[1]
        nkpt = overlaps.shape[2]


        eps_ref = 11.4/Ha_to_eV
        
        # Read k-points internally
        tmp = nc.Dataset(f"{root}/{ext_PC}/runo_EIG.nc", 'r')
        kpts = tmp['Kptns'][:] * Ns  # SC units
        tmp = nc.Dataset(f"{root}/{ext_SC}/config1/runo_WFK.nc", 'r')
        Kpts_SC = tmp['reduced_coordinates_of_kpoints'][:]

        # Find iK for each k-point and build K subgroups (config-independent)
        iK_list = find_iK(kpts, Kpts_SC)
        nK_SC = len(Kpts_SC)
        
        K_subgroups = []  # K_subgroups[iK] = list of k-indices that fold into iK
        total_mapped_k = 0
        expected_k_per_K = nkpt // nK_SC
        for iK in range(nK_SC):
            k_in_group = [ik for ik in range(nkpt) if iK_list[ik] == iK]
            K_subgroups.append(k_in_group)
            if len(k_in_group) != expected_k_per_K:
                raise ValueError(f"SC k-point iK={iK} has {len(k_in_group)} folded PC k-points, but expected exactly {expected_k_per_K} (nkpt={nkpt} / nK_SC={nK_SC}). Check k-grids.")
            print(f"  iK={iK}: mapped {len(k_in_group)} PC k-points")
            total_mapped_k += len(k_in_group)
            
        if total_mapped_k != nkpt:
            raise ValueError(f"Mapping error! Total mapped k-points ({total_mapped_k}) does not equal total PC k-points ({nkpt}).")

        self.Ginit.fill(0)
        tmpG = np.zeros_like(self.Ginit)

        for icfg in range(Ncfgs):
            print('Computing config: ', icfg)
            ds_SC = nc.Dataset(f"{root}/{ext_SC}/config{icfg+1}/runo_EIG.nc", 'r') 
            epsJ_all = ds_SC["Eigenvalues"][0]  # (nK_SC, mband)
            
            ov = overlaps[icfg]  # (nJ, nkpt, nband)
            
            # Build H block by block: H is block-diagonal in K
            H_full = np.zeros((nkpt * self.nband, nkpt * self.nband), dtype=complex)
            
            for iK in range(nK_SC):
                k_in_group = K_subgroups[iK]
                
                eps_K = epsJ_all[iK, :nJ_buf] - eps_ref
                ov_block = ov[:, k_in_group, :self.nband]  # (nJ, len(k_in_group), nband)
                
                H_block = np.einsum('jac,j,jbd->acbd', np.conjugate(ov_block), eps_K, ov_block, optimize=True)
                
                # Place into full H 
                for i, ik1 in enumerate(k_in_group):
                    for j, ik2 in enumerate(k_in_group):
                        H_full[ik1*self.nband:(ik1+1)*self.nband, ik2*self.nband:(ik2+1)*self.nband] = H_block[i,:,j,:]
            
            # TEST
            #S = np.zeros_like(H_full)
            #for iK in range(nK_SC):
            #    k_in_group = K_subgroups[iK]
            #    ov_block = ov[:, k_in_group, :self.nband]
            #    S_block = np.einsum('jac,jbd->acbd', np.conjugate(ov_block), ov_block)
            #    for i, ik1 in enumerate(k_in_group):
            #        for j, ik2 in enumerate(k_in_group):
            #            S[ik1*self.nband:(ik1+1)*self.nband,  ik2*self.nband:(ik2+1)*self.nband] = S_block[i,:,j,:]
            #print("Max deviation from identity:", np.max(np.abs(S - np.eye(S.shape[0]))))
            #print("S shape:", S.shape)
            #print("S diagonal (first 10):", np.diag(S)[:10].real)
            #print("S max abs value:", np.max(np.abs(S)))
            #print("nJ_buf:", nJ_buf)
            #print("needed for completeness:", expected_k_per_K * self.nband)
            #
            ## Where is the max deviation?
            #diff = np.abs(S - np.eye(S.shape[0]))
            #idx = np.unravel_index(np.argmax(diff), diff.shape)
            #print("Max deviation at index:", idx)
            #print("S value there:", S[idx])
            #print("Max off-diagonal abs:", np.max(np.abs(S - np.diag(np.diag(S)))))
            #
            ## Check if different k-points have nearly identical overlap rows
            ## (would indicate an indexing problem in the overlap computation)
            #ik1, ik2 = 0, 1  # two different k-points
            #block = S[ik1*self.nband:(ik1+1)*self.nband, ik2*self.nband:(ik2+1)*self.nband]
            #print("Off-diagonal block (k0,k1):\n", np.abs(block))
            #
            #diag = np.real(np.diag(S))
            #broken = np.where(diag < 0.5)[0]
            #print(f"Number of near-zero diagonal elements: {len(broken)}")
            #print("Indices:", broken)
            #print("Corresponding (k-point, band):", [(idx // self.nband, idx % self.nband) for idx in broken])
            #
            ## Check if it's specific bands or specific k-points
            #broken_kpts = set(idx // self.nband for idx in broken)
            #broken_bands = set(idx % self.nband for idx in broken)
            #print("Affected k-points:", broken_kpts)
            #print("Affected bands:", broken_bands)
            #
            #good_kpts = sorted(set(range(216)) - broken_kpts)
            #print(f"Good k-points ({len(good_kpts)}):", good_kpts)
            
            # Set to 0 the elements 23, 24 and 34 at Gamma (ik=0 for both, so indices are just the band numbers)
            H_full[1, 2] = 0.0
            H_full[1, 3] = 0.0
            H_full[2, 1] = 0.0
            H_full[2, 3] = 0.0
            H_full[3, 1] = 0.0
            H_full[3, 2] = 0.0
            
            evals, evecs = np.linalg.eigh(H_full)


            evals = evals + eps_ref
            
            # Construct G at Gamma for all band components
            U_gamma = evecs[:self.nband, :]  # shape: (nband, M)
            invden = 1.0 / (self.w[:, None] + 1j * self.delta - evals[None, :])
            tmpG += np.einsum('wm, nm, km -> wnk', invden, U_gamma, np.conjugate(U_gamma), optimize=True)

        self.Ginit = tmpG / Ncfgs

    # Should I put some comment here, or directly with quotations when defining the function?
    # Maybe with quotations right...describe what Ginit_approx does...
    def build_Ginit_approx(self,epsJ,overlaps):
        """
        Simple diagonal approximation @ Just call if diag instead of approx?
        """

        nJ_buf = overlaps.shape[0]
        H_diag = np.zeros(self.nband,dtype=complex)
        for n in range(self.nband):
            H_diag[n] = np.sum(np.conjugate(overlaps[:,n])*epsJ[0,:nJ_buf]*overlaps[:,n]) # TODO: Consider renaming to H_PC
        
        G_approx_diag = 1.0 / (self.w[:, None] + 1j*self.delta - H_diag[None, :])
        
        self.Ginit.fill(0)
        for n in range(self.nband):
            self.Ginit[:, n, n] = G_approx_diag[:, n]


    def build_Sigma_pseudoAHC(self,Ncfgs,overlaps,kweights,root,ext_SC):
        """
        So for this casea we do everything here, since we now have to average the self-energy
        This was kind of tricky because q=0 was not present when averaging the sel-energy...
        So check, but maybe it works in this way?
        """

        nJ_buf = overlaps.shape[1] # Ncfgs, nJ, nkpt, nband
        nkpt = overlaps.shape[2]

        Sigma = np.zeros((len(self.w),self.nband),dtype=complex)
        
        DeltaV_av = np.zeros((self.nband,nkpt,self.nband),dtype=complex) # Average of DeltaV. Only has even terms; not the first derivative
        FAN = np.zeros((len(self.w),self.nband),dtype=complex)
        DW = np.zeros(self.nband,dtype=complex) # Try also real...
    
        bands = [0,3,]

        for icfg in range(Ncfgs):
            ds_SC = nc.Dataset(f"{root}/{ext_SC}/config{icfg+1}/runo_EIG.nc", 'r') 
            epsJ = ds_SC["Eigenvalues"][0]  # Eigenvalues(nsppol, nkpt, mband)

            DeltaV = np.zeros((self.nband,nkpt,self.nband),dtype=complex)

            # Build V_{mk,nGamma}
            for n in range(self.nband):
                for m in range(self.nband):
                    for ik in range(nkpt):
                        DeltaV[m,ik,n] = np.sum(np.conjugate(overlaps[icfg,:,ik,m])*epsJ[0,:nJ_buf]*overlaps[icfg,:,0,n]) \
                                - self.eps0[0,n]*int(m==n)*int(ik==0) #np.eye(self.nband)[n,m] 

            DeltaV_av += DeltaV
        
        DeltaV_av = DeltaV_av/Ncfgs
        #DeltaV = DeltaV - DeltaV_av

        for icfg in range(Ncfgs):

            ds_SC = nc.Dataset(f"{root}/{ext_SC}/config{icfg+1}/runo_EIG.nc", 'r') 
            epsJ = ds_SC["Eigenvalues"][0]  # Eigenvalues(nsppol, nkpt, mband)

            DeltaV = np.zeros((self.nband,nkpt,self.nband),dtype=complex)

            # Build V_{mk,nGamma}
            for n in range(self.nband):
                for m in range(self.nband):
                    for ik in range(nkpt):
                        DeltaV[m,ik,n] = np.sum(np.conjugate(overlaps[icfg,:,ik,m])*epsJ[0,:nJ_buf]*overlaps[icfg,:,0,n]) \
                                - self.eps0[0,n]*int(m==n)*int(ik==0) #np.eye(self.nband)[n,m] 

            DeltaV = DeltaV - DeltaV_av*0
            # Term 2: Fan
            #FAN = np.zeros((len(self.w),self.nband),dtype=complex)
            for n in range(self.nband):
                for m in range(self.nband):
                    for ik in range(nkpt): # TODO: Move self.kpts initialization to __init__
                        #if ik == 0 and abs(self.eps0[0, m] - self.eps0[0, n]) < 1e-4:
                        #    continue
                        # Maybe more bands are needed relative to the NP approach, but here we just want to check the width
                        # I guess with the NP approach we proved we don't need so many bands...
                        num = np.conjugate(DeltaV[m,ik,n])*DeltaV[m,ik,n]
                        den = self.w[:] + 1j*self.delta - self.eps0[ik,m] # TODO: Verify this denominator logic
                        FAN[:,n] += kweights[ik]*num/den # Fan is also just calculated at Gamma
            
        FAN = FAN/Ncfgs

        # Term 3: Subtract, now that we have DW
        reduc = np.zeros((len(self.w),self.nband),dtype=complex)
        for n in range(self.nband):
            for ik in range(nkpt):
                for m in range(self.nband):
                    den = self.w[:] + 1j*self.delta - self.eps0[ik,m]
                    num = np.conjugate(DeltaV_av[m,ik,n]) * DeltaV_av[m,ik,n]
                    reduc[:,n] += kweights[ik] * num/den

        # The rigid shift is exactly the <DeltaV> term evaluated purely at m=n and ik=0 (Gamma)
        for n in range(self.nband):
            DW[n] = DeltaV_av[n,0,n]
            print(DeltaV_av[3,0,n]*Ha_to_eV) # Hm no, off-diagonal is very small

        Sigma = DW + FAN #- reduc

        self.Sigma.fill(0)
        for n in range(self.nband):
            self.Sigma[:,n,n] = Sigma[:,n]       

