# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
from joblib import Parallel, delayed
from skimage import io
import numpy as np
import pandas as pd
import csv
import os
import glob
from scipy import fftpack
from scipy.ndimage.interpolation import geometric_transform,rotate
from scipy.ndimage.measurements import center_of_mass
from sklearn import preprocessing

from skimage.exposure import rescale_intensity,equalize_adapthist,equalize_hist,adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean,warp_polar
from skimage import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk,square
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns
from skimage.filters import gaussian

import scipy.optimize as opt
from scipy.stats import norm



%matplotlib inline

# Dimension reduction and clustering libraries
import umap

import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA





def open_file(file):
    img=downscale_local_mean(io.imread(file), (1, 1,1))
    return img
                
def img_shape(file):
 #   img=open_file(file)
    shape_img=open_file(file).shape
    return shape_img
                
def tospherical(img, shape, order=1):
                    D = shape[1] - 1
                    R = D/2.
                    def transform(coords):
                        [z,x,y] =  coords
                        phi = np.pi*z / R
                        theta = np.pi*y / R
                        rho =  x / 2.

                        x = rho*np.sin(theta)*np.cos(phi) + R
                        y = rho*np.sin(theta)*np.sin(phi) + R
                        z = rho*np.cos(theta) + R
                        return z,x,y

                    spherical = geometric_transform(img, transform, order=order, mode='nearest', prefilter=False)
                    return spherical

#generic Fourier transformation
def GFT(file):
    print(file)
    return abs(fftpack.fftshift(fftpack.fftn(tospherical(open_file(file),img_shape(file))))).flatten()




def GFT2(file):
    
    image1=io.imread(file)
    image=image1[1,:,:,:]
    coordinates=np.where(image>50)
    z_t=coordinates[0]
    image=rescale_intensity(image,in_range=(20,image.max()))
    z=[0.6,0.7]
    
    GI=[]
  
    for i in range(len(z)):
        z_interest=round((z_t.max()-z_t.min())*z[i])+z_t.min()
        im=image[z_interest,:,:]
        im=rescale_intensity(im,in_range=(im.max()*0.2,im.max()))
        im=gaussian(im,sigma=1)
        polar=warp_polar(im,radius=58)
        fft=abs(fftpack.fftshift(fftpack.fftn(polar))).flatten()
        GI.append(fft)
    GI=np.array(GI)
    print(file)
    return GI

#GFT data to compute dimensionality reduction
path='./'

file_location = os.path.join(path,'*segmented.tif')
filenames = glob.glob(file_location)
filenames.sort()

filenames[0:6]





start_time = time.time()
df_cl1=np.array([GFT2(file) for file in filenames])
elapsed_time = time.time() - start_time

df_cl2=df_cl1.reshape(df_cl1.shape[0],(df_cl1.shape[1]*df_cl1.shape[2]))



file='./11_10_21_WT__Lmb1_BIC__gb_06-19_original_segmented.tif'


def barcode(file):
    barcode=file[2:]
    barcode=barcode.replace('__','_')
    barcode=barcode.replace('-','_')
    barcode=barcode.split('_')
    #what items in the barcode do you want to delete?
    items=[4,6,9,10]
    for index in sorted(items, reverse=True):
        del barcode[index]
    barcode.append(barcode[4]+' '+barcode[5]+'-'+barcode[6])
    columns=['day','month','year','genotype','treatment','image','nucleus','location']
    data_frame=pd.DataFrame([barcode],columns=columns)
    return data_frame 

data=[barcode(file) for file in filenames]
data_frame=pd.concat(data).reset_index()



#####################################BIC vs NBQX

#################################PCA
from sklearn.decomposition import PCA,KernelPCA
from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance_matrix
import scipy.optimize as opt
from scipy.stats import norm
from scipy.cluster import hierarchy
from matplotlib.colors import  LinearSegmentedColormap

####pca####

from sklearn.decomposition import PCA
flx_pca=PCA(n_components=50,whiten=True)

data_FLX1=flx_pca.fit_transform(df_cl2)

#pca analysis
flx_variance = np.cumsum(flx_pca.explained_variance_ratio_)

component_number = np.arange(len(flx_variance)) + 1

# Plot variance

fig = plt.figure(figsize=(8,5.71))
sns.set(font='MS Reference Sans Serif',context='notebook')

sns.lineplot(x=component_number,y=flx_variance,legend=False,linewidth=3,color='#1E434C')
plt.axvline(6, color='#9B4F0F',linestyle='--',linewidth=2)
plt.ylabel('Explained variance',fontsize=15,labelpad=15,fontweight=15)

plt.xlabel('N-components',fontsize=15,labelpad=15,fontweight=15)

plt.title('Explained Variance',fontsize=20,pad=20,fontweight=15)
labels=[0,6,10,20,30,40,50]
plt.xticks(labels)

plt.tight_layout()

plt.savefig('pca_plot_wt_gfft_.png',dpi=800)

########
########
flx_pca=PCA(n_components=6,whiten=True)


data_FLX=flx_pca.fit_transform(df_cl2)
summe=np.cumsum(flx_pca.explained_variance_ratio_)

palette=['#A4CABC','#B2473E']

##plot pca 1 and 2
fig = plt.figure(figsize=(8,5.71))
sns.set(font='MS Reference Sans Serif',context='notebook')


f=sns.scatterplot(x=data_FLX[:,0],y=data_FLX[:,1],hue=data_frame.iloc[:]['treatment'],palette=palette )
handles, labels = f.get_legend_handles_labels()
f.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)

plt.xlabel('Principal component 1 ('+str(round(summe[0],2)*100)+'%)',fontsize=15,labelpad=20,fontweight=15)
plt.ylabel('Principal component 2 ('+str(round((summe[1]-summe[0])*100,2))+'%)',fontsize=15,labelpad=20,fontweight=15)
plt.title('PCA 1 and PCA 2 WT generic Fourier transformation',fontsize=20,pad=30,fontweight=15)
plt.tight_layout()
plt.ylim(-3,2)
plt.xlim(-2,2.5)
plt.savefig('pca_1_2_expl_var_gff_wt.png',dpi=800)


#################
##################### merge two dataset
flx_pca=PCA(n_components=6,whiten=True)


data_FLX=flx_pca.fit_transform(df_cl2)

data_gft=pd.DataFrame(data_FLX)
data_gft=pd.concat([data_frame,data_gft],axis=1)

#delete flatten bic shapes

data_gft.drop(columns='index', inplace=True)


######UMAP
# standard embbeding
# UMAP settings
n_neighbors=30
min_dist=0.3
n_epochs=10000
learning_rate=0.0001
target_weight=0.00001
dens_lambda=10
metric='euclidean'
plot_code_FLX='FLX_pca4_2_gfft_minmax_scaled_'+str(output_metric)+'_'+str(n_neighbors)+'_'+str(min_dist)+'_'+str(n_epochs)+'_'+str(learning_rate)+'_'+str(dens_lambda)+'.png'

tot_nuc_embedding_1 = umap.umap_.UMAP(metric=metric,n_neighbors=n_neighbors,min_dist=min_dist,n_components=2,random_state=42,n_epochs=n_epochs,learning_rate=learning_rate,densmap=False,dens_lambda=dens_lambda).fit_transform(data_gft.iloc[:,8:17])
total_nuc_embedding_x_1,total_nuc_embedding_y_1=tot_nuc_embedding_1[:,0],tot_nuc_embedding_1[:,1]


###############generate plot

import seaborn as sns

palette=['#A4CABC','#B2473E','#EAB364','#ACBD78']
palette=['#A4CABC','#B2473E']
palette_bins=['#669BBC','#003049','#F2CC8F','#C1121F','#780000']
sns.set_palette(sns.color_palette(palette_bins))
fig = plt.figure(figsize=(8,5.71))
sns.set(font='MS Reference Sans Serif',context='notebook')

f=sns.scatterplot(x=total_nuc_embedding_x_1,y=total_nuc_embedding_y_1, hue=data_gft['treatment'],alpha=0.8)#palette=df_E['GI'])
#f.legend([],[],frameon=False)
#handles, labels = f.get_legend_handles_labels()
#f.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)

plt.title('Treatment dependent distribution \n of shapes',fontsize=20,pad=20,fontweight=15)



plt.tight_layout()

plt.savefig('shape_analysis_wt_gft_pca.png',dpi=800)



#########################




n_components=6

flx_pca=PCA(n_components=n_components,whiten=True)


data_FLX1=flx_pca.fit_transform(df_cl2)

data_gft=pd.DataFrame(df_cl2)
data_gft=pd.concat([data_frame,data_gft],axis=1)

palette=['#A4CABC','#B2473E']


treatment=data_gft.treatment.map({'BIC':palette[0],'NBQX':palette[1]})

R1=squareform(pdist(data_gft.iloc[:,9:], metric= 'cityblock'))
R1=pd.DataFrame(R1,columns=data_gft['location']).set_index(data_gft['location'])


gsns=sns.clustermap(R1.reset_index(drop=True) ,method='average',metric='cityblock',row_colors=treatment,cmap='RdBu_r')
plt.title('PCA=5',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('clustermap_pca5.png',dpi=500)

plt.tight_layout()

g = hierarchy.linkage(R1[0:72,0:72],method='average')
                                         



#R2= np.corrcoef(data_FLX)
#R1=R1[0:1318+359,0:1318+359]
#R1=pd.DataFrame(data=R1)
#corr = pd.melt(R1.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
#corr.columns = ['x', 'y', 'value']



fig, ax = plt.subplots(figsize=(17,18))

plt.axis('equal')
plt.axis('off')

#"CMRmap"
cmap2=['#283655','#617BA8']#,'#662E1C','#AF4425']

sns.set(font='MS Reference Sans Serif',context='notebook')
ax = sns.heatmap(R1,cmap='rocket_r',cbar_kws=dict(shrink=0.90),cbar=True)
plt.title('PCA=5',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('heatmap_pca5.png')



#UMAP
import umap

# standard embbeding
# UMAP settings
n_neighbors=30
min_dist=0.3
n_epochs=10000
learning_rate=0.0001
target_weight=0.00001
dens_lambda=10
output_metric='euclidean'
plot_code_FLX='FLX_pca4_2_gfft_minmax_scaled_'+str(output_metric)+'_'+str(n_neighbors)+'_'+str(min_dist)+'_'+str(n_epochs)+'_'+str(learning_rate)+'_'+str(dens_lambda)+'.png'

tot_nuc_embedding_1 = umap.umap_.UMAP(metric=output_metric,output_metric=output_metric,n_neighbors=n_neighbors,min_dist=min_dist,n_components=6,random_state=42,n_epochs=n_epochs,learning_rate=learning_rate,densmap=False,dens_lambda=dens_lambda).fit_transform(df_cl2)


data_gft2=pd.DataFrame(tot_nuc_embedding_1)
data_gft2=pd.concat([data_frame,data_gft2],axis=1)



R2=squareform(pdist(data_gft2.iloc[:,9:], metric='cityblock'))
R2=pd.DataFrame(R2,columns=data_gft2['location']).set_index(data_gft2['location'])
gsns=sns.clustermap(R2.reset_index(drop=True) ,method='average',row_colors=treatment,cmap='RdBu_r')
plt.title('UMAP=6',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('clustermap_UMAP10.png',dpi=500)



x=gsns.dendrogram_col.reordered_ind
def get_img(x,y,z):
    vector=x[x.index(y):x.index(z)]
    img=[]
    for n in vector:
       img.append(filenames[n])
    return img

A=get_img(x,4,0)


fig, ax = plt.subplots(figsize=(17,18))

plt.axis('equal')
plt.axis('off')

#"CMRmap"
cmap2=['#283655','#617BA8']#,'#662E1C','#AF4425']

sns.set(font='MS Reference Sans Serif',context='notebook')
ax = sns.heatmap(R2,cmap='rocket_r',cbar_kws=dict(shrink=0.90),cbar=True)
plt.title('UMAP 10',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('umap_10.png')



gsns_2.

#######################################bic

#UMAP
import umap

# standard embbeding
# UMAP settings
n_neighbors=50
min_dist=0.3
n_epochs=10000
learning_rate=0.0001
target_weight=0.00001
dens_lambda=10
output_metric='manhattan'
plot_code_FLX='FLX_pca4_2_gfft_minmax_scaled_'+str(output_metric)+'_'+str(n_neighbors)+'_'+str(min_dist)+'_'+str(n_epochs)+'_'+str(learning_rate)+'_'+str(dens_lambda)+'.png'

tot_nuc_embedding_1 = umap.umap_.UMAP(metric=output_metric,n_neighbors=n_neighbors,min_dist=min_dist,n_components=6,random_state=42,n_epochs=n_epochs,learning_rate=learning_rate,densmap=False,dens_lambda=dens_lambda).fit_transform(df_cl2[:])


data_gft2=pd.DataFrame(tot_nuc_embedding_1)
data_gft2=pd.concat([data_frame,data_gft2],axis=1)

bic=data_gft2
images=list(bic['image'])
colors=['#OD1321','#1D2D44','#3E5C76','#748CAB','#F0EBD8']
custom_palette = sns.color_palette('CMRmap', 24)
d={}
images=['01','02','03','04','05','06','07','08','09','10','11',
        '12','13','14','15','16','17','18','19','20','21','22','23']
for n in range(len(images)):
    d[images[n]]=custom_palette[n]
treatment=data_gft2[0:370].image.map(d)


R2=squareform(pdist(data_gft2.iloc[:,9:], metric= 'cityblock'))
R2=pd.DataFrame(R2,columns=data_gft2['location']).set_index(data_gft2['location'])
gsns=sns.clustermap(R2.reset_index(drop=True) ,method='centroid',metric='euclidean',row_colors=treatment,cmap='RdBu_r')
plt.title('UMAP 10 bic',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('umap_10_bic.png')
s


g = hierarchy.linkage(R1[0:72,0:72],method='average')
x=gsns.dendrogram_col.reordered_ind

def get_img(x,y,z):
    vector=x[x.index(y):x.index(z)]
    img=[]
    for n in vector:
       img.append(filenames[n])
    return img

A=get_img(x,194,37)

x=gsns.dendrogram_col.reordered_ind

fig, ax = plt.subplots(figsize=(17,18))

plt.axis('equal')
plt.axis('off')

#"CMRmap"
cmap2=['#283655','#617BA8']#,'#662E1C','#AF4425']

sns.set(font='MS Reference Sans Serif',context='notebook')
ax = sns.heatmap(R2,cmap='rocket_r',cbar_kws=dict(shrink=0.90),cbar=False)

##################################pca5
n_components=6

flx_pca=PCA(n_components=n_components,whiten=True)


data_FLX1=flx_pca.fit_transform(df_cl2)

data_gft=pd.DataFrame(df_cl2)
data_gft=pd.concat([data_frame,data_gft],axis=1)

data_gft=data_gft[0:370]


R1=squareform(pdist(data_gft.iloc[:,9  :], metric= 'cityblock'))
R1=pd.DataFrame(R1,columns=data_gft['location']).set_index(data_gft['location'])
gsns=sns.clustermap(R1.reset_index(drop=True) ,method='centroid',row_colors=treatment,cmap='RdBu_r',metric='euclidean')

######


n_components=6

flx_pca=PCA(n_components=n_components,whiten=True)


data_FLX1=flx_pca.fit_transform(df_cl2[0:370])

data_gft=pd.DataFrame(data_FLX1)
data_gft=pd.concat([data_frame,data_gft],axis=1)

palette=['#A4CABC','#B2473E']


treatment=data_gft.treatment.map({'BIC':palette[0],'NBQX':palette[1]})

R1=squareform(pdist(data_gft.iloc[:,9:17], metric= 'correlation'))
R1=pd.DataFrame(R1,columns=data_gft['location']).set_index(data_gft['location'])


gsns=sns.clustermap(R1.reset_index(drop=True) ,method='average',row_colors=treatment,cmap='RdBu_r')
plt.title('PCA=5',fontsize=20,pad=60,fontweight=15,loc='center')
plt.savefig('clustermap_pca5.png',dpi=500)

plt.tight_layout()



