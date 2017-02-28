import sys
import h5py
import matplotlib.pyplot as plt
plt.style.use('ggplot')


filename = sys.argv[-1]
ts = h5py.File(filename, 'r')

days_in_sec = 24*60*60.

plt.semilogy(ts['scales/sim_time'][:]/days_in_sec,ts['tasks/urms'][:,0],label='u')
plt.semilogy(ts['scales/sim_time'][:]/days_in_sec,ts['tasks/vrms'][:,0],label='v')
plt.xlabel('time (days)')
plt.ylabel('rms velocity (m/s)')

plt.legend(loc='upper left').draw_frame(False)
plt.savefig('rms_velocity.png')
