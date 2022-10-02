import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams["image.cmap"] = 'YlOrRd' #'viridis'
matplotlib.rcParams["font.size"]  = 12

rescale_on = True
fname='../OUTPUT/calbuco.src.nc'
fname_w='weights.dat'

ds = xr.open_dataset(fname)
w  = np.loadtxt(fname_w)
ds['w'] = ('ens',w)

X = ds.time.values / 3600.0 # time in h
Z = ds.lev.values  / 1000.0 # hight asl in km
DT = (X[1]-X[0])*3600.      # time interval in sec
rho=800 # density in kg/m3

if rescale_on:
    fname_out='src-rescaled.png'
    C = ds.src.dot(ds.w).values / 1000.0
    M = ds.mfr.dot(ds.w).values
    my_loc = 'upper left'
else:
    fname_out='src-background.png'
    C = ds.src.mean(dim='ens').values / 1000.0
    M = ds.mfr.mean(dim='ens').values
    my_loc = 'center right'

fig, ax = plt.subplots(figsize=(12.5,9))

im = ax.pcolormesh(X,Z,C.T,shading='gouraud')
ax.set_ylabel('Altitude [km asl]')
ax.set_xlabel('Simulation time [hours since 22 April 2015 at 00:00 UTC]')
ax.set_xlim([15,40])

cbar = fig.colorbar(im,
        orientation="horizontal", 
        shrink=.75,
        ax=ax)
cbar.set_label(r'Linear source emission strength ($\times 10^3$) [$kg~s^{-1}~m^{-1}$]')

ax2 = ax.twinx()
l2, = ax2.plot(X,1E-6*M,
        label     = "Emission rate", 
        color     = "blue",
        linestyle = "dashdot" )
ax2.set_ylabel(r'Emission rate ($\times 10^6$) [kg/s]')
ax2.set_ylim([0,20])

ax3 = ax.twinx()
l3, = ax3.plot(X,DT*1E-9*M.cumsum()/rho,
        label="Erupted volume",
        color="black",
        linestyle="solid" )
ax3.set_ylabel(r'Erupted volume [$km^3$]')

ax2.spines['left'].set_position(('outward', 50))
ax2.yaxis.set_label_position("left")
ax2.yaxis.set_ticks_position('left')

#print(DT*1E-9*M.cumsum()/rho)

ax.text(19.0,23.0,
        "1st eruptive\nphase",
        horizontalalignment='center',
        bbox={'facecolor': 'white', 'alpha': 0.5},
       )
ax.text(36.0,15,
        "2nd eruptive\nphase",
        horizontalalignment='center',
        bbox={'facecolor': 'white', 'alpha': 0.5},
       )
plt.legend(handles=[l2,l3],loc='upper center')
plt.savefig(fname_out,bbox_inches='tight')
