import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering as ClusteringMethod
#from sklearn.cluster import KMeans as ClusteringMethod
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams.update({'font.size': 7})

################# Inputs ################# 
plot_map      = True
plot_boxplot  = True
out_table     = True
er            = 0.35                      # Default relative error
thickness_min = 1E-1                      # Minumum thickness (cm)
nclusters     = 8                         # Number of clusters
transform     = 'mean'                    # mean or median
extension     = [-74,-67.5,-42,-36]       # [lon1,lon2,lat1,lat2]
########################################## 

# i. Assimilation dataset:
# ------------------------
# Original dataset. Reported by Van Eaton et al. 2016
# https://doi.org/10.1002/2016GL068076
#
# ii. Validation dataset:
# -----------------------
# Reckziegel (personal communication)
fname_in = {"assimila": 'grl54177.csv',
            "valida"  : 'reckziegel.csv'
            }

# Processed Dataset
fname_out = 'deposit.csv'

### Read observations
df_list=[]
for key in fname_in:
    df = pd.read_csv(fname_in[key], index_col=0)
    df["dataset"] = key
    df_list.append(df)
df = pd.concat(df_list)
print(df)
df['thickness'] = (df.thickness_max + df.thickness_min)*0.5

if key=="assimila":
    #Remove ambiguous data
    df.drop(['Cz.18','CAL002'],inplace=True)

### Clustering
#model = ClusteringMethod(NC)
model = ClusteringMethod(n_clusters=nclusters,
                         assign_labels='discretize',
                         random_state=0)
data = df[['latitude','longitude','thickness']].copy()

data['thickness'].where(df.thickness>thickness_min,thickness_min,inplace=True)
data['thickness'] = np.log10(data.thickness)
model.fit(data)
df['cluster'] = model.fit_predict(data)
df['true']    = df.groupby("cluster")["thickness"].transform(transform)

if False:
    df['error']   = er*df['true']
    df['error'].where(df.error>er*df.thickness,er*thickness_min, inplace=True)
    df['error_r'] = er
else:
    df['error']   = df.groupby("cluster")["thickness"].transform('std')
    df['error'].where(df.error>0.0,er*df.true, inplace=True)
    df['error_r'] = df.error/df.true

fig = plt.figure(figsize=(5,12))
gs = fig.add_gridspec(nrows=2)
gs.update(hspace=0.1)

### Plot map
if plot_map:
    ax1 = fig.add_subplot(gs[0], projection=crs.PlateCarree())
    countries = cfeature.NaturalEarthFeature(scale='10m',
                                             category='cultural',
                                             name='admin_0_boundary_lines_land',
                                             facecolor='none')
    ax1.coastlines(linewidth=0.5)
    ax1.add_feature(cfeature.LAND, color="lightgrey", alpha=0.4)
    ax1.add_feature(countries,linewidth=0.5)
    ax1.set_extent(extension, crs=crs.PlateCarree())
    scatter = ax1.scatter(x          = df.longitude, 
                          y          = df.latitude,
                          c          = df.cluster,
                          s          = 15,
                          cmap       = 'Paired',
                          edgecolors = 'k',
                          linewidths = 0.2,
                          alpha      = 0.8,
                          transform=crs.PlateCarree())
    gl = ax1.gridlines(crs=crs.PlateCarree(),
                      draw_labels = True,
                      linewidth   = 0.5, 
                      color       = 'gray', 
                      alpha       = 0.5, 
                      linestyle   = '--')
    legend = ax1.legend(*scatter.legend_elements(),
                       loc="lower right", 
                       title="Clusters")
    ax1.set(title = "(a) Sampling sites")
    ax1.add_artist(legend)
    gl.top_labels   = False
    gl.right_labels = False
    gl.ylabel_style = {'rotation': 90}
#    plt.savefig("clusters.png",dpi=200,bbox_inches='tight')

### Plot boxes
if plot_boxplot:
    ax2 = fig.add_subplot(gs[1])
    df.boxplot(positions = np.arange(nclusters),
               column    = ['thickness'],
               by        = 'cluster',
               whis      = (0, 100),
               ax        = ax2)
    df.plot.scatter(x     = 'cluster',
                    y     = 'thickness',
                    color = 'red',
                    alpha = 0.5,
                    label = 'Measurements',
                    ax    = ax2)
    ax2.set(xlabel = 'Cluster label',
           ylabel = 'Deposit thickness [cm]',
           title  = '(b) Clustered box plot diagram',
           yscale = 'log',
           )
    ax2.grid(axis='y', color = 'gray', linestyle='--', linewidth=0.4)
    #plt.savefig("boxplot.png",dpi=200,bbox_inches='tight')

fig.suptitle("")
fig.savefig("clusters.png",dpi=200,bbox_inches='tight')

### Output table
if out_table:
    df.to_csv(fname_out,
              columns=["latitude",
                       "longitude",
                       "thickness",
                       "error",
                       "error_r",
                       "dataset",
                       "cluster"],
              index=False)
