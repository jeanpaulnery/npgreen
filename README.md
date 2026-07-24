# npgreen code

## What does this code do?
It computes the adiabatic nonperturbative Green function including electron-phonon interactions (i.e., distorted phononic configurations). From it, one can determine the spectral function. It is currently implemented for use with ABINIT.

## TLDR

In the folder `ensemble_Ns{Ns}_Ncfgs{Ncfgs}_T{Temp}`, you should have subfolders named `config{i}`, and inside each of them, the files `runo_EIG.nc` and `runo_WFK.nc`. Run the following command inside the code folder:

```bash
python nonperturbative.py --delta_eV {delta_eV} --nbuf {nbuf} --calc_overlaps --Ncfgs {Ncfgs}
```

This will generate the overlaps and compute the Green function. If you have already calculated the overlaps, you can omit the `--calc_overlaps` flag, and it will use the existing `overlaps.nc` file. This file should be located in the folder `ensemble_Ns{Ns}_Ncfgs{Ncfgs}_T{Temp}`.

## Important parameters

The main execution script is `nonperturbative.py`. Here are some of the key parameters you can pass:
- `--Ns`: Supercell size (e.g., `8`).
- `--Ncfgs`: Number of distorted configurations (default: `120`).
- `--Ncfgs_GSigma`: Number of configs to use for the Green's function calculation (default: `120`).
- `--nbuf`: Buffer size. Essential parameter, e.g., Ns2:8, Ns4:40, Ns6:24, Ns8:100.
- `--delta_eV`: Broadening smearing in eV (default: `0.1`).
- `--calc_overlaps`: Flag to compute overlaps. If omitted, it will try to load existing ones.
- `--nband`: Number of bands to consider (default: `7`).
- `--Temp`: Temperature associated with the configurations (default: `0`).
- `--root`: The base directory path where the script looks for outputs (e.g. `/path/to/your/output`).
- `--calc_type`: Type of calculation (`standard`, `no_offd`, `pseudoAHC`).

## Requirements

To run this code, you need a Python environment with the following packages installed:
- `numpy`
- `xarray`
- `netCDF4`

## Structure of folders

### Input
- `runo_EIG.nc` for each distorted config
- `runo_WFK.nc` for each distorted config

### Output
- `overlaps.nc` or `overlaps_all.nc`: Contains the overlaps, either for a specific k-point or for all k-points.
- `GSigma_Ncfgs{Ncfgs}_{delta_eV}{ind}`: G and Sigma vs w, obtained using the trick to remove the broadening of G0.

## Theoretical background

The method is based on JP Nery and F Mauri, PRB 105, 245120 (2022). All technical details can be found there. The basic idea is that one calculates the energies and wavefunctions (WFs) of the ground state of distorted configurations, plugs that into an expression for the Green's function, and averaging yields the final result.

In addition, the method improves the numerical accuracy by defining a self-energy, which is then used to define an improved Green's function that does not have artificial broadening in the G0 term.

The distorted configurations are computed in a supercell (SC). However, the Green function is defined in the basis of the undistorted primitive cell (PC). Therefore, this procedure requires what is commonly referred to as "unfolding," expressing the WFs in the SC in terms of the basis `|kn>` of the PC. In practice, this just means computing overlaps and being careful about the units of the plane-waves (which differ if expressed in reduced coordinates). The SC WFs have more plane-wave coefficients, since the reciprocal lattice vectors g of the SC are smaller than the G vectors of the PC.

## 1. Calculate overlaps
Generally, we compute it at one k-point at a time, resulting in `overlaps.nc`.
Most of the time we just compute Gamma, but sometimes also 3/4X in the Ns=8 SC, which approximately corresponds to the conduction band minimum (CBM). We store this as `overlaps_ind.nc`.
If we want the full band structure, we need the overlaps for many k-points along a high-symmetry path.
Ideally, we perform the calculation just at Gamma in a large enough SC, and from that, we gain access to many k-points—those commensurate with the SC.
We also calculated the overlaps for all k-points and saved them as `overlaps_all.nc`.

Since we eventually obtain the overlaps for all k-points, we could theoretically just use this comprehensive overlap file. However, for Ns=8 and Ncfgs=120, this file size can reach 40 GB, which is quite large. Therefore, we might store that file in a university cluster and primarily work locally with the overlap file for a single k-point (ik).

* We also must store the energies `eps_KJ`. These are found in the standard `runo_EIG.nc` files.

**Crucial Note on Ground State Calculations (`istwfk`)**: For the Primitive Cell (PC), **always use `istwfk=1`**. The code no longer supports the `istwfk=2` case (which previously required a more complex `1 + overlap(G) + conf(G)` formulation).

**IMPORTANT**: Make sure the energies are calculated in a SC K-grid that doesn't use symmetries! Otherwise, we don't get access to all the `<KJ|kn>` overlaps (at least not easily). In some cases, it's not possible if the WFs take up too much space and get deleted.
For Ns=4, we were using a K-grid of 2x2x2, but then all the K points are in the "positive (or 0) quadrant" and no time reversal symmetry can be used anyway.
For Ns=8, which uses a 4x4x4 K-grid, we must make sure to use `kptopt=3` (and perhaps `nsym=1` just to be safe).

Additionally: for Ns=6, we have to modify the ground state to a 6x6x6 or 12x12x12 grid, though 6x6x6 seems sufficient (it's just a rigid shift of the grid).

## Other functionalities

The code includes other experimental or secondary calculation modes, controlled via `--calc_type`:
- `no_offd`: Used to test the effect of removing degenerate off-diagonal elements in the width computation.
- `pseudoAHC`: Implements a pseudo-Allen-Heine-Cardona approach. It was largely experimental and unused in the final paper (it might lack some q=0 terms or have convergence issues).

## To run, basically do:
```bash
conda activate python_netcdf
python ~/project_diamond/scripts/NPGreen/nonperturbative.py  # --Ns 8 --Ncfgs 1 --nband 5 --calc_overlaps
```

## Notes

- **Band Structure Interpolation:** A planned future feature is extracting a real-space self-energy to interpolate to arbitrary k-points for band structure plotting.
- **Band Selection:** In `overlaps_all.nc`, 10 bands are saved by default to provide flexibility. However, typically only 7 bands are used for the main calculation, as some act as buffer bands. Additional bands are primarily required for convergence testing; for such tests, you should compute and use overlaps with a higher number of bands.
