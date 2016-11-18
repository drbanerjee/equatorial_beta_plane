import os
import sys
import h5py
import matplotlib.pyplot as plt
plt.style.use("ggplot")

datadir = sys.argv[-1]

data =h5py.File(os.path.join(datadir,"timeseries/timeseries_s1.h5"),"r")

plt.plot(data['/scales/sim_time'][:],data['/tasks/urms'][:,0],linewidth=2,label='u')
plt.plot(data['/scales/sim_time'][:],data['/tasks/vrms'][:,0],linewidth=2,label='v')
plt.legend(loc='upper right')

plt.savefig('rms_vel.png')
