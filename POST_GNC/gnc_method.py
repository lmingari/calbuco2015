import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

###
### Parameters
###
remove_logbias = False 
plot_cost      = True
out_table      = True
out_factors    = True
out_netcdf     = True

###
### Input files
###
fname_obs='../POST_DATA/deposit.csv'
fname_ens='../OUTPUT/output_full.nc'

###
### Read model data
###
ds = xr.open_dataset(fname_ens)
#Convert mass loading to thickness (cm)
#Bulk deposit density 800 kg/m3
fu = 100.0/800.0
x = fu*ds.isel(time=-1)['tephra_grn_load']

###
### Read obs data
###
df = pd.read_csv(fname_obs)
df = df[df.dataset=='assimila']
nobs = len(df)
nens = len(ds.ens)
print("Number of observations: {}".format(nobs))

hx  = np.zeros((nobs,nens))
hxp = np.zeros((nobs,nens))
y   = np.zeros(nobs)
ye  = np.zeros(nobs)

for i in range(nobs):
    lat   = df.iloc[i].latitude
    lon   = df.iloc[i].longitude
    y[i]  = df.iloc[i].thickness #Observation in cm
    ye[i] = df.iloc[i].error
    ###
    ### Interpolate
    ###
    hx[i,:] = x.interp(lat=lat,lon=lon).values

hxm = np.mean(hx,axis=1)
for i in range(nens):
    hxp[:,i] = hx[:,i] - hxm

Ri = np.diag(1.0/ye**2)
P  = hxp@hxp.T
P /= nens-1
#
Pi,rank = linalg.pinvh(P,return_rank=True)
aa = np.allclose(P, np.dot(P, np.dot(Pi,P)))
print("Checking succesful inversion:")
print(aa)

###
### Solve iterative procedure
###
Q  = hx.T@(Ri+Pi)@hx
b  = -1*hx.T@(Pi@hxm+Ri@y)
Ap = np.abs(Q)+Q
An = np.abs(Q)-Q

NT=40000

w  = np.ones(nens) / (1.0*nens)
J1 = []
J2 = []
for i in range(NT):
    a = Ap@w
    c = An@w
    f = (np.sqrt(b**2+a*c)-b)/a
    if(np.allclose(w*f,w)):
        print("Finishing at iter: {}".format(i))
        break
    elif(i==NT-1):
        error = np.linalg.norm(w-w*f)
        print("WARN: Final error: {}".format(error))
    w *= f
    J1.append( (hx@w-hxm).T@Pi@(hx@w-hxm) )
    J2.append( (hx@w-y).T@Ri@(hx@w-y) )

# Remove log bias
if remove_logbias:
    y_tmp = y/(hx@w)
    k=np.mean(np.log(y_tmp[y_tmp>0]))
    k=np.exp(k)
    print("Factor k: ",k)
    w *= k

# Write factor weigths
if out_factors:
    with open('weights.dat','w') as f:
        for item in w:
            f.write("{:.9E}\n".format(item))

if out_table:
    df['forecast'] = hxm
    df['analysis'] = hx@w
    #
    df.to_csv("analysis.csv",
              columns = ["latitude",
                         "longitude",
                         "thickness",
                         "error",
                         "forecast",
                         "analysis"],
              index = False)
    
if plot_cost:
    fig, ax = plt.subplots()
    J1 = np.array(J1)
    J2 = np.array(J2)
    Jt = np.sqrt((J1+J2)/nobs)
    it = np.arange(len(J1))+1
    ax.plot(it,Jt,'bo',ms=4,mfc="None",alpha=0.75)
    ax.set_ylabel(r'Normalised cost function $\sqrt{J/p}$')
    ax.set_xlabel('Iteration step')
    ax.set_xscale('log')
    ax.grid()
    plt.savefig("cost.pdf",bbox_inches='tight')

if out_netcdf:
    t = xr.DataArray(w, coords=[x.ens],dims=["ens"])
    ds_out = xr.Dataset()
    ds_out['weights']  = t
    ds_out['analysis'] = x.dot(t)
    ds_out['forecast'] = x.mean(dim='ens')
    ds_out.to_netcdf("output.nc")
