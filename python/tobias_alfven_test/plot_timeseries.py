import matplotlib
matplotlib.use('Agg')
import sys
import h5py
import matplotlib.pyplot as plt


fname = sys.argv[1]
data = h5py.File(fname,'r')

plt.plot(data['/scales/sim_time'],data['/tasks/urms'][:,0,0],label=r'$\left< u \right>$')
plt.plot(data['/scales/sim_time'],data['/tasks/vrms'][:,0,0],label=r'$\left< v \right>$')
plt.legend(loc='upper left').draw_frame(False)
plt.xlabel("time")
plt.ylabel("velocity")

plt.savefig('rms_vel_vs_t.png',dpi=300)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.semilogy(data['/scales/sim_time'],data['/tasks/Ekin'][:,0,0])
ax1.set_xlabel("time")
ax1.set_ylabel("Kinetic Energy")

ax2 = fig.add_subplot(122)
ax2.plot(data['/scales/sim_time'],data['/tasks/Ekin_x_ratio'][:,0,0])
ax1.set_xlabel("time")
ax2.set_ylabel(r"$u^2/(u^2 + v^2)$")

fig.savefig('energy_xfrac_vs_t.png',dpi=300)
