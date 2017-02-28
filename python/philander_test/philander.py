"""Philander Experiment script. Implements Philander (1984) solution for
Rossby and Kelvin wave separation on an equatorial beta plane, using linear, inviscid
shallow water equations.

IC:

Can be run doubly periodic with sponge layer or Chebyshev in y with no
sponge layer.

Usage:
    philander_experiment.py [--Lx=<Lx> --Ly=<Ly> --nx=<nx> --ny=<ny> --chebyshev --zeta=<zeta> --startup]

Options:
    --Lx=<Lx>                                length in latitude in meters  [default: 10000000]
    --Ly=<Ly>                                length in longitude in meters [default: 5000000]
    --nx=<nx>                                grid points in x [default: 128]
    --ny=<ny>                                grid points in y [default: 128]
    --chebyshev                              Use chebyshev in y
    --zeta=<zeta>                            Control parameter for sponge layer [default: 0.9]
    --startup                                Run without dt(h) to get initial conditions

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
zeta = float(args['--zeta'])
D = 98. #depth in  meters
a = 1.157e-7 #time constant in 1/seconds
b = 1.157e-7 #time constant in 1/seconds 
g = 0.02 #reduced gravity in meters per seconds squared
Co = 1.4 #speed in meters per second
beta = 2.28e-11 #beta plane constant in 1/m*s
cheb = args['--chebyshev']
startup = args['--startup']

basename = sys.argv[0].split('.py')[0]

data_dir = "scratch/" + basename
#data_dir += "_Lx{0:f}_Ly{1:f}_nx{2:d}_ny_{3:d}_gamma{4:5.02e}_zeta{5:5.02e}".format(Lx,Ly,nx,ny,gamma,zeta)

if cheb:
    data_dir += "_chebyshev"

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
if cheb:
    y_basis = de.Chebyshev('y', ny, interval=(-Ly, Ly), dealias=3/2)
else:
    y_basis = de.Fourier('y', ny, interval=(-Ly, Ly), dealias=3/2)

domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics on the beta plane                                                                                              
bp = de.IVP(domain,variables=['u','v','h'])
bp.parameters['beta'] = beta
#bp.parameters['gamma'] = gamma
bp.parameters['zeta'] = zeta
bp.parameters['Lx'] = Lx
bp.parameters['Ly'] = Ly
bp.parameters['pi'] = np.pi
bp.parameters['a'] = a
bp.parameters['b'] = b
bp.parameters['D'] = D
bp.parameters['g'] = g
bp.parameters['Co'] = Co
bp.substitutions['theta'] = "pi*y/Ly"
bp.substitutions['Y(y)'] = "Ly/pi *((1+zeta)/zeta * arctan(zeta*sin(theta)/(1+zeta*cos(theta))))"
bp.substitutions['Z(y)'] = "(1-zeta)**2/2 * (1-cos(theta))/(1+zeta**2+2*zeta*cos(theta))"
bp.substitutions['f'] = "beta*y"
bp.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Ly)'

if startup:
    bp.add_equation("dt(h) = 0")
else:
    bp.add_equation("dt(h) + D*(dx(u)+dy(v)) + b*h = 0")

bp.add_equation("dt(u) + a*u - g*dx(h) = f*v")
bp.add_equation("dt(v) + a*v - g*dy(h) = -f*u")
bp.add_bc("left(v) = 0")
if not startup:
    bp.add_bc("right(v) = 0")

# Build solver                                                                                                                               
ts = de.timesteppers.RK443
IVP = bp.build_solver(ts)
IVP.stop_sim_time = 864000
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 10000000
logger.info('Solver built')

dx = np.min(IVP.domain.grid_spacing(0))
safety = 0.2
dt = safety*dx/Co

# Initial conditions                                                                                                                         
x = domain.grid(0)
y = domain.grid(1)
h = IVP.state['h']
u = IVP.state['u']

# perturbations                      
d = 500000.                                                                                                        
h['g'] = 0.2 * np.exp(-((x-5000000.)**2 + y**2)/d**2)

if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Analysis                                                                                                                                   
snapshots = IVP.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1., max_writes=50)
snapshots.add_system(IVP.state)

timeseries = IVP.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'),iter=10, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(v*v))",name='vrms')

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
