import requests
import os

#La Manno et al. 2020, Developing Mouse Brain data
from tqdm import tnrange, tqdm_notebook
def download_file(doi,ext):
	url = 'https://api.datacite.org/dois/'+doi+'/media'
	r = requests.get(url).json()
	netcdf_url = r['data'][0]['attributes']['url']
	r = requests.get(netcdf_url,stream=True)
	#Set file name
	fname = doi.split('/')[-1]+ext
	#Download file with progress bar
	if r.status_code == 403:
		print("File Unavailable")
	if 'content-length' not in r.headers:
		print("Did not get file")
	else:
		with open(fname, 'wb') as f:
			total_length = int(r.headers.get('content-length'))
			pbar = tnrange(int(total_length/1024), unit="B")
			for chunk in r.iter_content(chunk_size=1024):
				if chunk:
					pbar.update()
					f.write(chunk)
		return fname


#Merfish count matrices, and cell metadata

#Metadata
#MERFISH data

#metadata.csv
download_file('10.22002/D1.2063','.gz')

#counts.h5ad
download_file('10.22002/D1.2064','.gz')



#10x VMH data
#metadata.csv
download_file('10.22002/D1.2065','.gz')

#tenx.mtx (log counts)
download_file('10.22002/D1.2072','.gz')

#10X raw Count Matrix
download_file('10.22002/D1.2073','.gz')




os.system("gunzip *.gz")


os.system("mv D1.2063 metadata.csv")
os.system("mv D1.2064 counts.h5ad")

os.system("mv D1.2065 tenxmetadata.csv")
os.system("mv D1.2072 tenx.mtx")
os.system("mv D1.2073 tenxCount.mtx")





os.system("pip3 install --quiet torch --no-cache-dir")
os.system("pip3 install --quiet anndata --no-cache-dir")
os.system("pip3 install --quiet matplotlib --no-cache-dir")
os.system("pip3 install --quiet scikit-learn --no-cache-dir")
os.system("pip3 install --quiet torchsummary --no-cache-dir")
os.system("pip install --quiet scanpy==1.6.0 --no-cache-dir")
#pip3 install --quiet umap-learn --no-cache-dir
os.system("pip3 install --quiet scvi-tools --no-cache-dir")




os.system("git clone --single-branch --branch taraDev https://tarachari3:marsianID2.@github.com/pachterlab/spacetime.git")

os.system("cd /content/spacetime/nnNCApy")

import anndata 
import pandas as pd
import numpy as np
from MCML import NN_NCA #Now has continuous label addition
import random

from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis, NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import torch
import time
import scanpy as sc
import seaborn as sns
import umap
from scipy import stats
import scipy.io as sio
import matplotlib


#Read in 10X VMH data
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.linewidth'] = 0.1

ncaLossDF = pd.DataFrame()

loss = []
method = []
alltime = []
dataset = []




state = 42
ndims = 2

data_path = '../..'

pcs = 50
n_latent = 50

count_mat = sio.mmread(data_path+'/tenx.mtx')
count_mat.shape


#Center and scale log-normalized data
scaled_mat = scale(count_mat)
meta = pd.read_csv(data_path+'/tenxmetadata.csv',index_col = 0)

clusters = np.unique(meta['cluster'].values)
map_dict = {}
for i, c in enumerate(clusters):
	map_dict[c] = i
new_labs = [map_dict[c] for c in meta['cluster'].values]

adata = anndata.AnnData(count_mat, obs = meta)
adata.X = np.nan_to_num(adata.X)

lab1 = list(meta.cluster)
lab2 = list(meta.sex_label)
# lab3 = list(meta.medical_cond_label)


allLabs = np.array([lab1])
allLabs2 = np.array([lab1,lab2])

nanLabs = np.array([[np.nan]*len(lab1)])


device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Compare loss from sklearn NCA fit to loss from nn_NCA latent output -------------------- 

for i in range(3):
	# Latent space from All NCA loss
	tic = time.perf_counter()
	nca = NN_NCA(n_latent = n_latent, epochs = 50)
	latent_allNCA = nca.fit(scaled_mat,allLabs, fracNCA = 1, silent = True)
	toc = time.perf_counter()
	print(f"nnNCA fit in {toc - tic:0.4f} seconds")

	alltime += [toc-tic]




	# Latent space from sklearn (duplicate rows) (Runs out of RAM on Colab)


	X = scaled_mat #logMat.concatenate(logMat, join = "outer")
	y = lab1 #+ lab2

	tic = time.perf_counter()
	ncaSk = NeighborhoodComponentsAnalysis(n_components = n_latent)

	
	ncaSk.fit(X, y)
	toc = time.perf_counter()
	print(f"sklearn NCA in {toc - tic:0.4f} seconds")

	X_trans = ncaSk.transform(X)

	# Make NCA object and use functions to get masks + loss
	newNCA = NN_NCA(n_latent = n_latent, epochs = 50)

	logX = torch.from_numpy(scaled_mat).float().to("cpu")

	masks, weights = newNCA.multiLabelMask(allLabs, None, None, False)



	allNCAloss = newNCA.lossFunc(logX ,logX , torch.from_numpy(latent_allNCA).float().to("cpu"), masks,weights, False, None, 1)

	skNCAloss = newNCA.lossFunc(logX ,logX , torch.from_numpy(X_trans).float().to("cpu"), masks,weights, False, None, 1)

	loss += [allNCAloss[0].item(),skNCAloss[0].item()]
	method += ['MCML 50D','NCA 50D']
	alltime += [toc-tic]

	dataset += ['10x VMH','10x VMH']


print('10x Data Done')




#Merfish data

counts = anndata.read(data_path+'/counts.h5ad')
print(counts)

cellMeta = pd.read_csv(data_path+'/metadata.csv')
print(cellMeta.head())

choice = np.unique(cellMeta.slice_id)[7]

counts.obs['slice'] = pd.Categorical(cellMeta.slice_id)
counts.obs['type'] = pd.Categorical(cellMeta.subclass)
counts.obs['x'] = list(cellMeta.center_x)
counts.obs['y'] = list(cellMeta.center_y)


sub = counts[counts.obs['slice'].isin([choice])]
print(sub)

colors = np.random.rand(len(sub.obs['type']),3)
nanLabs = np.array([[np.nan]*len(sub.obs['type'])])

labs = np.array([list(sub.obs['type'])])
labs_cont = np.array([list(sub.obs['x']),list(sub.obs['y'])])


orig_mat = sub.X
log_mat = np.log1p(sub.X)

sc.pp.log1p(sub)

#Center scale
sc.pp.scale(sub, max_value=10)

scaled_mat = sub.X




for i in range(3):
	# Latent space from All NCA loss
	tic = time.perf_counter()
	nca = NN_NCA(n_latent = n_latent, epochs = 50)
	latent_allNCA = nca.fit(scaled_mat,labs, fracNCA = 1, silent = True)
	toc = time.perf_counter()
	print(f"nnNCA fit in {toc - tic:0.4f} seconds")

	alltime += [toc-tic]




	# Latent space from sklearn (duplicate rows) (Runs out of RAM on Colab)


	X = scaled_mat #logMat.concatenate(logMat, join = "outer")
	y = labs[0] #+ lab2
	
	tic = time.perf_counter()
	ncaSk = NeighborhoodComponentsAnalysis(n_components = n_latent)

	
	ncaSk.fit(X, y)
	toc = time.perf_counter()
	print(f"sklearn NCA in {toc - tic:0.4f} seconds")

	X_trans = ncaSk.transform(X)

	# Make NCA object and use functions to get masks + loss
	newNCA = NN_NCA(n_latent = n_latent, epochs = 50)

	logX = torch.from_numpy(scaled_mat).float().to("cpu")

	masks, weights = newNCA.multiLabelMask(labs, None, None, False)



	allNCAloss = newNCA.lossFunc(logX ,logX , torch.from_numpy(latent_allNCA).float().to("cpu"), masks,weights, False, None, 1)

	skNCAloss = newNCA.lossFunc(logX ,logX , torch.from_numpy(X_trans).float().to("cpu"), masks,weights, False, None, 1)

	loss += [allNCAloss[0].item(),skNCAloss[0].item()]
	method += ['MCML 50D','NCA 50D']
	alltime += [toc-tic]

	dataset += ['MERFISH','MERFISH']

ncaLossDF['Loss'] = loss
ncaLossDF['Method'] = method
ncaLossDF['Time'] = alltime
ncaLossDF['Dataset'] = dataset


ncaLossDF.to_csv('sklearnNCAComps.csv')


# All NCA Loss: (tensor(0.6602), tensor(0), tensor(0), tensor(7983.1934), tensor(-0.6602))
# Sklearn NCA Loss:  (tensor(0.0698), tensor(0), tensor(0), tensor(7983.1934), tensor(-0.0698))

