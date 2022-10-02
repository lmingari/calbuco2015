import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import gamma

###
### Parameters
###
corr_th         = 0.0
update_forecast = False
random_sort     = False
plot_histo      = True
plot_skew       = False
plot_obs        = False
out_table       = False
out_netcdf      = False

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
xf = x.mean(dim='ens')

###
### Read obs data
###
df = pd.read_csv(fname_obs)
df = df[df.dataset=='assimila']
nobs = len(df)
print("Number of observations: {}".format(nobs))

if random_sort: 
    df = df.sample(frac=1) # Random sorting

if plot_skew:
    fig, ax = plt.subplots()
    data_stat = []

if plot_histo:
    plt.rc('legend',fontsize  = 'x-large')
    plt.rc('lines' ,linewidth = 4.4)
    fig, ax = plt.subplots()


if update_forecast:
    if corr_th>0:
        print("Localization: ON with rho_o={}".format(corr_th))
    else:
        print("Localization: OFF")

def compute_analysis_mean(ym,yo,p2,r2):
    y_inv = 1.0/ym
    y = y_inv + p2/(p2+r2)*(1./yo - (r2+1)*y_inv)
    y = 1.0 / y
    return y

def compute_analysis_pert(yf,yf_mean,yo,p2,r2):
    yg1 = r2/(1+2*r2)
    yg_mean = (1+2*r2)*yo
    yg_var  = yg1*yg_mean**2
    k = 1.0/yg1
    theta = yg_mean*yg1
    yg = np.random.gamma(k,theta,size=yf.sizes['ens'])
    #
    a_coeff = np.sqrt(1-p2)/yf_mean 
    b_coeff = p2/(p2+r2)
    c_coeff = 1.0/np.sqrt(yg_mean**2-2*yg_var)
    output = (yf-yf_mean)*a_coeff + b_coeff*(c_coeff*(yg-yg_mean) - a_coeff*(yf-yf_mean))
    #
    return output

for index,row in df.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    yo  = row["thickness"]   #Observation in cm
    r1  = row["error_r"]**2  #type 1 relative observation error variance
    r2  = 1./(1./r1+1)       #type 2 relative observation error variance
    #
    if yo==0:
        yo = 0.1*np.random.uniform()
        print("A zero observation was perturbed: ",yo)
    #
    yf      = x.interp(lat=lat,lon=lon) #Model interpolated
    yf_mean = yf.mean().item()
    yf_var  = yf.var().item()
    yf_skew = scipy.stats.skew(yf)
    p1      = yf_var/yf_mean**2          #type 1 relative forecast error variance
    p2      = yf_var/(yf_var+yf_mean**2) #type 2 relative forecast error variance
    #
    ya_mean = compute_analysis_mean(yf_mean,yo,p2,r2)
    ya_var  = ya_mean**2 / (1.0/r2+1.0/p2)
    ya_mode = max(yf_mean*(1-p1),0.0)
    pert    = compute_analysis_pert(yf,yf_mean,yo,p2,r2) #compute relative perturbation
    ya      = ya_mean * (1 + pert)
    #
    x_var   = x.var(dim='ens')
    dx      = xr.cov(x,yf,dim='ens') 
    corr    = np.sqrt(yf_var * x_var)
    corr    = corr.where(corr>0)
    corr    = (dx/corr).fillna(0)
    #
    if update_forecast:
        if corr_th>0:
            delta = np.abs(corr)/corr_th
            delta = delta.where(delta<1,1)
            delta = delta**2
            dx = dx * delta * (1.0/yf_var)
        else:
            dx = dx * (1.0/yf_var)
        x = x + dx*(ya-yf)
        x = x.where(x>0,0)
    #
    if plot_skew:
        data_stat.append( [yf_mean,yf_var,yf_skew] )

    if plot_histo:
        y_histo,bins,_=yf.plot.hist(bins=20,density=True,ax=ax)
        #
        p1_inv = 1./p2-1
        x_gamma = np.linspace(0.25*bins[1],bins[-1],num=100)
        y_gamma = x_gamma**(p1_inv-1) * np.exp(-p1_inv/yf_mean*x_gamma)
        y_gamma *= 1.0/(gamma(p1_inv)*(yf_mean/p1_inv)**p1_inv) #y_histo.max()/y_gamma.max()
        ax.plot(x_gamma,y_gamma,alpha=0.65,label="Gamma distribution")
        ax.legend()
        ax.set_title("Sampled prior distribution at lat={:.2f} lon={:.2f}".format(lat,lon))
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Deposit mass loading [cm]")
        plt.savefig("histo{}.pdf".format(index),bbox_inches='tight')
        ax.clear()
    
# Output table
if out_table:
    ya_mean = []
    ya_std  = []
    for index,row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        #
        ya = x.interp(lat=lat,lon=lon)
        ya_mean.append( ya.mean().item() )
        ya_std.append( ya.std().item() )
    #
    df['analysis'] = ya_mean
    df['error_an'] = ya_std
    #
    df.to_csv("analysis.csv",
              columns = ["latitude",
                         "longitude",
                         "thickness",
                         "error",
                         "analysis",
                         "error_an"],
              index = False)

# Skewness according prior at observation sites
if plot_skew:
    dfStat = pd.DataFrame(data_stat,columns=['mean', 'var', 'skew'])
    dfStat['delta'] = np.sqrt(dfStat['var'])/dfStat['mean']

    # Theoretical prediction for popular PDF's
    dfStatTheory = pd.DataFrame(columns=['delta', 'skew'])
    dfStatTheory['delta'] = np.linspace(0.1,3.2)
    dfStatTheory['gamma'] = 2*dfStatTheory['delta']
    dfStatTheory['gauss'] = 0.0
    dfStatTheory['lnorm'] = (dfStatTheory['delta']**2 + 3)*dfStatTheory['delta']
    dfStatTheory.plot(x='delta',y='gauss', label="Gaussian",    style='k:',  ax=ax)
    dfStatTheory.plot(x='delta',y='lnorm', label="log-normal",  style='r-.', ax=ax)
    dfStatTheory.plot(x='delta',y='gamma', label="Gamma",       style='k-',  ax=ax)
    dfStat.plot.scatter(x='delta',y='skew',ax=ax)

    ax.set_ylim([-1,10])
    ax.set_ylabel("Skewness")
    ax.set_xlabel("Standard deviation-to-mean ratio")
    ax.set_title("Prior distribution skewness")
    ax.legend(title="Theoretical distributions",loc=0)
    plt.savefig("skewness.pdf",bbox_inches='tight')

if out_netcdf:
    ds_out = xr.Dataset()
    ds_out['forecast'] = xf
    ds_out['analysis'] = x.mean(dim='ens')
    ds_out['variance'] = x.var(dim='ens')
    ds_out.to_netcdf("output.nc")
