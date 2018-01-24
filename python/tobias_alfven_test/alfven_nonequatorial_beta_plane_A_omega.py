"""alfven_nonequatorial_beta_plane_A_omega

Run the driven non-equatorial beta plane equations from Tobias, Diamond, & Hughes (2007).

Usage:
    alfven_nonequatorial_beta_plane_A_omega.py [--Lx=<Lx> --Ly=<Ly> --nx=<nx> --ny=<ny> --mhd --B0=<B0> --beta=<beta> --eta=<eta> --nu=<nu> --dt=<dt> --G0=<G0> --n_modes=<n_modes>]

Options:
    --Lx=<Lx>                                length in latitude [default: 6.283185307179586]
    --Ly=<Ly>                                length in longitude [default: 6.283185307179586]
    --nx=<nx>                                grid points in x [default: 128]
    --ny=<ny>                                grid points in y [default: 128]
    --B0=<B0>                                Zonal (x) background field strength [default: 0]
    --mhd                                    Use MHD
    --beta=<beta>                            beta parameter [default: 5]
    --nu=<nu>                                viscosity [default: 1e-4]
    --eta=<eta>                              diffusivity [default: 1e-4]
    --G0=<G0>                                forcing amplitude [default: 2.0]
    --n_modes=<n_modes>                      number of modes to force with [default: 3]
"""
import sys
import os
import numpy as np
from mpi4py import MPI
import time
from dedalus.tools.config import config
from dedalus.extras import flow_tools

from docopt import docopt

# parse arguments for parameters
args = docopt(__doc__)

Lx = float(args['--Lx'])
Ly = float(args['--Ly'])
nx = int(args['--nx'])
ny = int(args['--ny'])
beta = float(args['--beta'])
nu = float(args['--nu'])
B0 = float(args['--B0'])
eta = float(args['--eta'])
G0 = float(args['--G0'])
n_modes = int(args['--n_modes'])

if args['--mhd']:
    mhd=True
else:
    mhd=False

basename = sys.argv[0].split('.py')[0]

data_dir = "scratch/" + basename
data_dir += "_Lx{0:f}_Ly{1:f}_nx{2:d}_ny_{3:d}_beta{4:5.02e}_B0{5:5.02e}_nut{6:5.02e}".format(Lx,Ly,nx,ny,beta,B0,nu)

# if cheb:
#     data_dir += "_chebyshev"

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import logging
logger = logging.getLogger(__name__)

# Create bases and domain
# bases
x_basis = de.Fourier('x', nx, interval=(0,Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)

domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)

# 2D Boussinesq magnetohydrodynamics on the beta plane
variables = ['omega','psi']
if mhd:
    variables.append('A')
bp = de.IVP(domain,variables=variables)
bp.parameters['beta'] = beta
bp.parameters['B0'] = B0
bp.parameters['Lx'] = Lx
bp.parameters['Ly'] = Ly
bp.parameters['pi'] = np.pi
bp.parameters['eta'] = eta
bp.parameters['nu'] = nu
bp.parameters['G0'] = G0
bp.parameters['n_modes'] = n_modes

bp.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Ly)'
bp.substitutions['J(A,B)'] = 'dx(A)*dy(B) - dx(B)*dx(A)'
bp.substitutions['Lap(A)'] = 'dx(dx(A)) + dy(dy(A))'
bp.substitutions['u'] = 'dy(psi)'
bp.substitutions['v'] = '-dx(psi)'

def forcing(*args):
    x = args[0].data
    y = args[1].data
    ampl = args[2].value 
    n_modes = args[3].value

    kx = [14, 12, 12]
    ky = [0, 15, -15]
    phase_x = [0, 0, 0]
    phase_y = [0, 0, 0]

    spatial_pattern = 0
    for i in range(n_modes):
        spatial_pattern += np.cos(kx[i]*x + phase_x[i])*np.cos(ky[i]*y + phase_y[i])/n_modes

    return ampl*spatial_pattern

def GF(*args, domain=domain, F=forcing):
    return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

de.operators.parseables['G'] = GF

if mhd:
    bp.add_equation("dt(omega) + B0*Lap(dx(A)) - nu*Lap(omega) = J(psi,omega) + beta*dx(psi) + J(A,Lap(A)) + G(x, y, G0, n_modes)")
    bp.add_equation("dt(A) - B0*dx(psi) - eta*Lap(A) = J(psi, A)")
else:
    bp.add_equation("dt(omega) - nu*Lap(omega) = J(psi,omega) + beta*dx(psi) + G(x, y, G0, n_modes)")
bp.add_equation("omega + Lap(psi) = 0", condition='nx != 0 or ny != 0')
bp.add_equation("psi = 0", condition='nx == 0 and ny == 0')

# Build solver
dt = 0.01
ts = de.timesteppers.RK443
IVP = bp.build_solver(ts)
IVP.stop_sim_time = 100.
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 10000000
logger.info('Solver built')


CFL = flow_tools.CFL(IVP, initial_dt=dt, cadence=10, safety=0.8, max_change=2.0)
CFL.add_velocities(('u','v'))

if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Analysis
snapshots = IVP.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1., max_writes=50)
snapshots.add_system(IVP.state)

slices = IVP.evaluator.add_file_handler(os.path.join(data_dir,'slices'), sim_dt=0.5, max_writes=50)
slices.add_task("u",name="u")
slices.add_task("v",name="v")

timeseries = IVP.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'),iter=10, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(v*v))",name='vrms')
timeseries.add_task("vol_avg(0.5*(u*u+v*v))",name='Ekin')
timeseries.add_task("vol_avg((u*u)/(u*u+v*v))",name='Ekin_x_ratio')

analysis_tasks = [snapshots, slices, timeseries]

flow = flow_tools.GlobalFlowProperty(IVP, cadence=10)
flow.add_property("0.5*(u*u + v*v)", name="Ekin")
flow.add_property("u", name="u")
flow.add_property("v", name="v")
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while IVP.ok:
        dt = CFL.compute_dt()
        dt = IVP.step(dt)
        if (IVP.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e, Kinetic Energy: %e, u max: %e, v max: %e' %(IVP.iteration, IVP.sim_time, dt, flow.volume_average('Ekin'), flow.max('u'), flow.max('v')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %IVP.iteration)
    logger.info('Sim end time: %f' %IVP .sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)
