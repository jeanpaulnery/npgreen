"""
Input Parsing Module.
Sets arguments and paths (local or cluster) for 
Primitive Cell and Supercell ABINIT NetCDF outputs.
"""
import argparse
from constants import Ha_to_eV

def get_input_params():
    parser = argparse.ArgumentParser(description="NPGreen nonperturbative calculations")
    
    # Main variables test
    parser.add_argument("--calc_overlaps", action="store_true", help="Calculate overlaps instead of loading them")
    parser.add_argument("--Ns", type=int, default=8, help="Ns parameter")
    parser.add_argument("--nbuf", type=int, required=True, help="Buffer size") # Ns2:8, Ns4:40, Ns6:24, Ns8:100
    parser.add_argument("--Ncfgs", type=int, default=120, help="Number of configs")
    parser.add_argument("--Ncfgs_GSigma", type=int, default=120, help="Ncfgs to use for G and Sigma calculation")
    parser.add_argument("--ind", action="store_const", const="_ind", default="", help="Suffix for k=3/4 X, the ind gap, '_ind')")
    parser.add_argument("--Temp", type=int, default=0, help="Temperature")
    
    # Other Variables
    parser.add_argument("--nband", type=int, default=7, help="Number of bands")
    parser.add_argument("--delta_eV", type=float, default=0.1, help="Delta smearing in eV")
    parser.add_argument("--lenw", type=int, default=16000, help="Energy grid size")
    
    # Paths
    parser.add_argument("--root", type=str, default="/Users/jeanpaul/Solid_State/project_diamond/output", help="Root of output path")
    parser.add_argument("--ext_SC", type=str, default=None, help="Supercell output directory path (relative to root). Defaults to standard ensemble path.")
    parser.add_argument("--cluster", action="store_true", help="Use cluster paths instead of local paths")
    parser.add_argument("--calc_type", type=str, default="standard", choices=["standard", "no_offd", "pseudoAHC"], help="Type of calculation: standard, no_offd (all ks), or pseudoAHC")
    parser.add_argument("--all_k", action="store_true", help="Type of calculation: standard, no_offd (all ks), or pseudoAHC")

    args = parser.parse_args()

    # Derived parameters and logic
    ik = 17 if args.ind == "_ind" else 0
    if ik == 17: print("WARNING: make sure the 888 ground state is the one with symmetries, which gives index ik=17")
    delta = args.delta_eV / Ha_to_eV


    # Setup paths based on whether on cluster or local
    ext_SC = args.ext_SC if args.ext_SC else f"nonperturbative/ensemble_Ns{args.Ns}_Ncfgs{args.Ncfgs}_T{args.Temp}"

    if args.cluster:
        root = "/gpfs/scratch/acad/thermo/jpnery/project_diamond"
        ext_PC = f"ground_state/k888_istwfk1" # Ah yes, here later we don't have to do the formula 1 + overalp(G) + conf(G) stuff
        #ext_PC = f"ground_state/k888_nosym" # Needed for all_k calculation (needed for no_offd). Otherwise doesn't matter
        if args.Ns == 6:
            ext_PC = f"ground_state/k666_nosym"
    else:
        root = args.root
        #ext_SC = f"nonperturbative/ensemble_Ns{args.Ns}_Ncfgs{args.Ncfgs}_T{args.Temp}_nogamma"
        #ext_SC = f"nonperturbative/conv_tests/Ns2_nogamma"
        ext_PC = f"ground_state/k888_istwfk1" # ORIGINAL
        #ext_PC = "ground_state/k888_nosym" # TEST OFF-DIAG
        #ext_PC = f"ground_state/k888{ist}" # @Remove istwfk=2 in PC doesn't work anymore

    return args.calc_overlaps, args.Ns, args.nbuf, args.Ncfgs, args.Ncfgs_GSigma, args.ind, args.Temp, ik, args.nband, args.delta_eV, delta, args.lenw, root, ext_PC, ext_SC, args.calc_type, args.all_k
