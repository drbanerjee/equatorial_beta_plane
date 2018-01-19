#!/usr/bin/bash
#SBATCH --nodes=2

source $HOME/build/dedalus_intel_mpi/bin/activate

date
mpirun -np 32 python3 alfven_nonequatorial_beta_plane_A_omega.py --nx=1024 --ny=1024 --beta=0.
date
