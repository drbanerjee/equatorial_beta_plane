"""Kelvin wave test script. Implements Matsuno (1966) solution for
Kelvin wave on an equatorial beta plane, using linear, inviscid
shallow water equations.

IC:

eta = u = exp(-y^2/2)*sin(2*pi/Lx * x)

Can be run doubly periodic with sponge layer or Chebyshev in y with no
sponge layer.

Usage:
    kelvin_wave.py [--Lx=<Lx> --Ly=<Ly> --nx=<nx> --ny=<ny> --chebyshev --zeta=<zeta> --gamma=<gamma> --dt=<dt>]

Options:
    --Lx=<Lx>                                length in latitude [default: 4]
    --Ly=<Ly>                                length in longitude [default: 4]
    --nx=<nx>                                grid points in x [default: 128]
    --ny=<ny>                                grid points in y [default: 128]
    --chebyshev                              Use chebyshev in y
    --zeta=<zeta>                            Control parameter for sponge layer [default: 0.9]
    --gamma=<gamma>                          damping for sponge layer [default: 1]
    --dt=<dt>                                timestep in units of wave advection time [default: 0.01]

"""
import sys
import os
import numpy as np
from mpi4py import MPI
import time
from dedalus.tools.config import config


from docopt import docopt

# parse arguments for parameters
args = docopt(__doc__)

Lx = float(args['--Lx'])
Ly = float(args['--Ly'])
nx = int(args['--nx'])
ny = int(args['--ny'])
gamma = float(args['--gamma'])
zeta = float(args['--zeta'])
dt_frac = float(args['--dt'])
cheb = args['--chebyshev']

beta = 1.

basename = sys.argv[0].split('.py')[0]

data_dir = "scratch/" + basename
data_dir += "_Lx{0:f}_Ly{1:f}_nx{2:d}_ny_{3:d}_gamma{4:5.02e}_zeta{5:5.02e}_dt{6:5.02e}".format(Lx,Ly,nx,ny,gamma,zeta,dt_frac)

if cheb:
    data_dir += "_chebyshev"

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import logging
logger = logging.getLogger(__name__)

c = 1.
tau_advect = Lx/c

# Create bases and domain
# bases
x_basis = de.Fourier('x', 128, interval=(0,Lx), dealias=3/2)
if cheb:
    y_basis = de.Chebyshev('y', 128, interval=(-Ly, Ly), dealias=3/2)
else:
    y_basis = de.Fourier('y', 128, interval=(-Ly, Ly), dealias=3/2)

domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics on the beta plane
bp = de.IVP(domain,variables=['u','v','eta'])
bp.parameters['beta'] = beta
bp.parameters['gamma'] = gamma
bp.parameters['zeta'] = zeta
bp.parameters['Lx'] = Lx
bp.parameters['Ly'] = Ly
bp.parameters['pi'] = np.pi
bp.substitutions['theta'] = "pi*y/Ly"
bp.substitutions['Y(y)'] = "Ly/pi *((1+zeta)/zeta * arctan(zeta*sin(theta)/(1+zeta*cos(theta))))"
bp.substitutions['Z(y)'] = "(1-zeta)**2/2 * (1-cos(theta))/(1+zeta**2+2*zeta*cos(theta))"

bp.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Ly)'

bp.add_equation("dt(eta) + dx(u) + dy(v) = 0")
if cheb:
    bp.add_equation("dt(u) + dx(eta) =  beta*y*v")
    bp.add_equation("dt(v) + dy(eta) = -beta*y*u")
    bp.add_bc("left(v) = 0")
    bp.add_bc("right(v) = 0")
else:
    bp.add_equation("dt(u) + dx(eta) =  beta*Y(y)*v - gamma*Z(y)*u")
    bp.add_equation("dt(v) + dy(eta) = -beta*Y(y)*u - gamma*Z(y)*v")

# Build solver

dt = dt_frac*tau_advect

ts = de.timesteppers.RK443
IVP = bp.build_solver(ts)
IVP.stop_sim_time = 10.* tau_advect
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 10000000
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
eta = IVP.state['eta']
u = IVP.state['u']

# perturbations 
eta['g'] = np.exp(-y**2/2)*np.sin(2*np.pi*x/Lx)
u['g'] = eta['g']

if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Analysis
snapshots = IVP.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1., max_writes=50)
snapshots.add_system(IVP.state)

timeseries = IVP.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'),iter=10, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(v*v))",name='vrms')
timeseries.add_task("vol_avg((v*v)/(u*u+v*v))",name='E_ky_ratio')

analysis_tasks = [snapshots, timeseries]

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while IVP.ok:
        dt = IVP.step(dt)
        if (IVP.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(IVP.iteration, IVP.sim_time, dt))
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
