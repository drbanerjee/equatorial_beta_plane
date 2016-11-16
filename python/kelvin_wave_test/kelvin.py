"""Kelvin wave test script



"""
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

from filter_field import filter_field

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx = 4.
Ly = 2.
beta = 1.
gamma = 0. #0.1
nu = 0.1

# Create bases and domain
# bases
x_basis = de.Fourier('x', 128, interval=(0,Lx), dealias=3/2)
y_basis = de.Fourier('y', 128, interval=(-Ly, Ly), dealias=3/2)
domain = de.Domain([y_basis,x_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics on the beta plane
bp = de.IVP(domain,variables=['u','v','eta'])
bp.parameters['beta'] = beta
bp.parameters['gamma'] = gamma
bp.parameters['zeta'] = 0.9
bp.parameters['Lx'] = Lx
bp.parameters['Ly'] = Ly
bp.parameters['pi'] = np.pi
bp.parameters['nu'] = nu
bp.substitutions['theta'] = "pi*y/Ly"
bp.substitutions['Y(y)'] = "Ly/pi *((1+zeta)/zeta * arctan(zeta*sin(theta)/(1+zeta*cos(theta))))"
bp.substitutions['Z(y)'] = "(1-zeta)**2/2 * (1-cos(theta))/(1+zeta**2+2*zeta*cos(theta))"

bp.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Ly)'

bp.add_equation("dt(u) + dx(eta) - nu*(dx(dx(u)) + dy(dy(u))) = beta*Y(y)*v - gamma*Z(y)*u")
bp.add_equation("dt(v) + dy(eta) - nu*(dx(dx(v)) + dy(dy(v)))= -beta*Y(y)*u - gamma*Z(y)*v")
bp.add_equation("dt(eta) + dx(u) + dy(v) = 0")

# Build solver

dt = 0.01

ts = de.timesteppers.RK443
IVP = bp.build_solver(ts)
IVP.stop_sim_time = 150.
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 10000000
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
eta = IVP.state['eta']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# perturbations 
pert =  1e-3 * noise
eta['g'] = pert
filter_field(eta)

# Analysis
snapshots = IVP.evaluator.add_file_handler('snapshots', sim_dt=1., max_writes=50)
snapshots.add_system(IVP.state)

timeseries = IVP.evaluator.add_file_handler('timeseries',iter=100, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(v*v))",name='vrms')

# CFL
CFL = flow_tools.CFL(IVP, initial_dt=dt, cadence=10, safety=0.7,
                     max_change=1.5, min_change=0.5, max_dt=.35, threshold=0.05)
CFL.add_velocities(('u', 'v'))

# Flow properties
#flow = flow_tools.GlobalFlowProperty(IVP, cadence=1)
#flow.add_property("sqrt(u*u)", name='urms')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while IVP.ok:
        dt = CFL.compute_dt()
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
