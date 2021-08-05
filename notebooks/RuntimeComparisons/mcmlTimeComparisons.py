import requests
import os

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


#Pseudotime Packer & Zhu C. elegans data
#counts.mtx
download_file('10.22002/D1.2060','.gz')

#cells.csv
download_file('10.22002/D1.2061','.gz')

#genes.csv
download_file('10.22002/D1.2062','.gz')



#SMART-seq VMH data
#metadata.csv
download_file('10.22002/D1.2067','.gz')

#smartseq.mtx (log counts)
download_file('10.22002/D1.2071','.gz')

#smartseq.mtx (raw counts)
download_file('10.22002/D1.2070','.gz')



#10x VMH data
#metadata.csv
download_file('10.22002/D1.2065','.gz')

#tenx.mtx (log counts)
download_file('10.22002/D1.2072','.gz')

#10X raw Count Matrix
download_file('10.22002/D1.2073','.gz')




os.system("gunzip *.gz")

os.system("mv D1.2060 counts.mtx")
os.system("mv D1.2061 cells.csv")
os.system("mv D1.2062 genes.csv")

os.system("mv D1.2067 smartseqmetadata.csv")
os.system("mv D1.2071 smartseq.mtx")

os.system("mv D1.2070 smartseqCount.mtx")

os.system("mv D1.2065 tenxmetadata.csv")
os.system("mv D1.2072 tenx.mtx")
os.system("mv D1.2073 tenxCount.mtx")






#dev_all_hvg.mtx
download_file('10.22002/D1.2043','.gz')

#dev_all_raw.mtx
download_file('10.22002/D1.2044','.gz')

#lamannometadata.csv
download_file('10.22002/D1.2045','.gz')

os.system("gunzip *.gz")

os.system("mv D1.2043 dev_all_hvg.mtx")
os.system("mv D1.2044 dev_all_raw.mtx")
os.system("mv D1.2045 lamannometadata.csv")


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
import scvi
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
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
matplotlib.rc('axes',edgecolor='black')
sc.set_figure_params(dpi=125)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.linewidth'] = 0.1

state = 42
ndims = 2

data_path = '../..'

pcs = 50
n_latent = 50

times = pd.DataFrame()

alltime = []
dataset = []
embed = []
numCells = []

#Pseudotime run
#pseudo = sc.read(data_path+'/counts.mtx', cache=True).T
count_mat = sio.mmread(data_path+'/counts.mtx')
count_mat.shape

pseudo = anndata.AnnData(count_mat.todense().T)
pseudo.X = np.nan_to_num(pseudo.X)
print(pseudo)

geneMeta = pd.read_csv(data_path+'/genes.csv')
print(geneMeta.head())
cellMeta = pd.read_csv(data_path+'/cells.csv')
print(cellMeta.head())

pseudo.obs_names = list(cellMeta.cell)
pseudo.var_names = list(geneMeta.id)
pseudo.obs['type'] = pd.Categorical(cellMeta['cell.type'])
pseudo

#Subset for ASE_ASJ_AUA lineage
pseudo_sub = pseudo[pseudo.obs['type'].isin(['ASJ','AUA','ASE_parent','Neuroblast_ASJ_AUA','ASE','ASEL','ASER','Neuroblast_ASE_ASJ_AUA'])] #'ASI_parent','ASI','ASK_parent'

sc.pp.filter_cells(pseudo_sub, min_counts=0)
sc.pp.filter_genes(pseudo_sub, min_counts=0)

pseudo_copy = pseudo_sub.copy()
sc.pp.normalize_per_cell(pseudo_copy, counts_per_cell_after=1e4)
raw = pseudo_copy.X

sc.pp.log1p(pseudo_copy)
pseudo_copy.obsm['log'] = pseudo_copy.X

sc.pp.highly_variable_genes(pseudo_copy,n_top_genes=300)

pseudo_copy = pseudo_copy[:,pseudo_copy.var['highly_variable']]

#Center scale
sc.pp.scale(pseudo_copy, max_value=10)
sc.tl.pca(pseudo_copy, n_comps=50)
sc.pp.neighbors(pseudo_copy,n_neighbors=50, n_pcs=15,method='gauss')

pseudo_copy.uns['iroot'] = np.flatnonzero(pseudo_copy.obs['type']  == 'Neuroblast_ASE_ASJ_AUA')[0]

sc.tl.diffmap(pseudo_copy,n_comps=10)
sc.tl.dpt(pseudo_copy,n_dcs=10) #Creates 'dpt_pseudotime'

pseudo_copy

count_mat = pseudo_copy.X

nanLabs = np.array([[np.nan]*len(pseudo_copy.obs['type'])])

labs = np.array([list(pseudo_copy.obs['type'])])
labs_cont = np.array([list(pseudo_copy.obs['dpt_pseudotime'])])

pseudo_sub = pseudo_sub[:,pseudo_copy.var_names]
print(pseudo_sub)

# LDVAE accuracy scores

copy = pseudo_sub.copy()
scvi.data.setup_anndata(copy, labels_key='type')

for i in range(1):
	tic = time.perf_counter()
	vae = scvi.model.LinearSCVI(copy)
	vae.train()
	latent_ldvae = vae.get_latent_representation()
	lab_idx = vae.train_indices
	unlabeled_idx = []
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['pseudotime']
embed += ['LDVAE']
numCells += [count_mat.shape[0]]

scvi.data.setup_anndata(copy, labels_key='type')

for i in range(1):
	tic = time.perf_counter()
	vae = scvi.model.SCANVI(copy, np.nan)
	vae.train(train_size = 0.7)
	latent_scanvi = vae.get_latent_representation()
	lab_idx = vae.train_indices
	unlabeled_idx = []
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['pseudotime']
embed += ['SCANVI']
numCells += [count_mat.shape[0]]

# Reconstruction loss only
for i in range(1): 


	tic = time.perf_counter()
	ncaR = NN_NCA(n_latent = n_latent, epochs = 100)


	lossesR, latentR = ncaR.fit(count_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True) #labels


	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['pseudotime']
embed += ['Recon MCML 50D']
numCells += [count_mat.shape[0]]

#label mcml

for i in range(1):
	numLabs = int(np.round(len(nanLabs[0])*0.7))
	allPos = range(len(nanLabs[0]))
	labeled_idx = random.sample(allPos,numLabs)
	unlabeled_idx = [i for i in allPos if i not in labeled_idx]


	labeled_lab = labs.T[labeled_idx ,:]

	unlabeled_lab = labs.T[unlabeled_idx ,:]

	newLabs = labs.copy().T

	newLabs[unlabeled_idx,:] = np.nan

	tic = time.perf_counter()
	ncaMiss = NN_NCA(n_latent = n_latent, epochs = 100) #n_latent


	lossesNCAMiss, latentNCAMiss = ncaMiss.fit(count_mat,newLabs.T,fracNCA = 0.9999, silent = True,ret_loss = True)
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['pseudotime']
embed += ['Cell Type MCML 50D']
numCells += [count_mat.shape[0]]





#SmartSeq run
count_mat = sio.mmread(data_path+'/smartseq.mtx')
count_mat.shape
raw_count_mat = sio.mmread(data_path+'/smartseqCount.mtx')
raw_count_mat.shape
#Center and scale data
scaled_mat = scale(count_mat)
meta = pd.read_csv(data_path+'/smartseqmetadata.csv',index_col = 0)

clusters = np.unique(meta['smartseq_cluster'].values)
map_dict = {}
for i, c in enumerate(clusters):
	map_dict[c] = i
new_labs = [map_dict[c] for c in meta['smartseq_cluster'].values]

adata = anndata.AnnData(count_mat, obs = meta)
adata.X = np.nan_to_num(adata.X)

adata2 = anndata.AnnData(raw_count_mat, obs = meta)
adata2.X = np.nan_to_num(adata2.X)

# LDVAE accuracy scores
scvi.data.setup_anndata(adata2, labels_key='smartseq_cluster_id')

for i in range(1):
	tic = time.perf_counter()
	vae = scvi.model.LinearSCVI(adata2)
	vae.train()
	latent_ldvae = vae.get_latent_representation()
	lab_idx = vae.train_indices
	unlabeled_idx = []
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['smartseq']
embed += ['LDVAE']
numCells += [raw_count_mat.shape[0]]

# SCANVI accuracy scores
scvi.data.setup_anndata(adata2, labels_key='smartseq_cluster_id')

for i in range(1):
	tic = time.perf_counter()
	vae = scvi.model.SCANVI(adata2, np.nan)
	vae.train(train_size = 0.7)
	latent_scanvi = vae.get_latent_representation()
	lab_idx = vae.train_indices
	unlabeled_idx = []
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['smartseq']
embed += ['SCANVI']
numCells += [raw_count_mat.shape[0]]

#recon mcml
lab1 = list(meta.smartseq_cluster)
lab2 = list(meta.sex_label)
lab3 = list(meta.medical_cond_label)


allLabs = np.array([lab1])
allLabs2 = np.array([lab1,lab2])

nanLabs = np.array([[np.nan]*len(lab1)])

# Reconstruction loss only
for i in range(1): 
	labels = np.array([lab1]).copy()
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False) #0.7 for training fraction
	#Set 30% to no label (nan)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan

	tic = time.perf_counter()
	ncaR = NN_NCA(n_latent = n_latent, epochs = 100)


	lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True) #labels


	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['smartseq']
embed += ['Recon MCML 50D']
numCells += [raw_count_mat.shape[0]]

# label mcml


for i in range(1): #3
	labels = np.array([lab1]).copy()
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False) #0.7
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan

	tic = time.perf_counter()
	nca = NN_NCA(n_latent = n_latent, epochs = 100)


	losses, latent = nca.fit(scaled_mat,labels,fracNCA = 0.3, silent = True,ret_loss = True)


	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['smartseq']
embed += ['Cell Type MCML 50D']
numCells += [raw_count_mat.shape[0]]


#10X Run
count_mat = sio.mmread(data_path+'/tenx.mtx')
count_mat.shape
rawcount_mat = sio.mmread(data_path+'/tenxCount.mtx')
rawcount_mat.shape

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

adata2 = anndata.AnnData(rawcount_mat, obs = meta)
adata2.X = np.nan_to_num(adata2.X)

# LDVAE accuracy scores
scvi.data.setup_anndata(adata2, labels_key='cluster_id')

for i in range(1): #3
	tic = time.perf_counter()
	vae = scvi.model.LinearSCVI(adata2)
	vae.train()
	latent_ldvae = vae.get_latent_representation()
	lab_idx = vae.train_indices
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['tenx']
embed += ['LDVAE']
numCells += [rawcount_mat.shape[0]]



# SCANVI accuracy scores
scvi.data.setup_anndata(adata2, labels_key='cluster_id')

for i in range(1):
	tic = time.perf_counter()
	vae = scvi.model.SCANVI(adata2, np.nan)
	vae.train(train_size = 0.7)
	latent_scanvi = vae.get_latent_representation()
	lab_idx = vae.train_indices
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['tenx']
embed += ['SCANVI']
numCells += [rawcount_mat.shape[0]]


lab1 = list(meta.cluster)
lab2 = list(meta.sex_label)
# lab3 = list(meta.medical_cond_label)


allLabs = np.array([lab1])
allLabs2 = np.array([lab1,lab2])

nanLabs = np.array([[np.nan]*len(lab1)])

# Reconstruction loss only


for i in range(1):
	tic = time.perf_counter()
	ncaR = NN_NCA(n_latent = n_latent, epochs = 100)
	labels = np.array([lab1])
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan

	lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True)
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['tenx']
embed += ['Recon MCML 50D']
numCells += [rawcount_mat.shape[0]]

#label mcml


for i in range(1): #3
	tic = time.perf_counter()
	nca = NN_NCA(n_latent = n_latent, epochs = 100)

	labels = np.array([lab1]).copy()
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan



	losses, latent = nca.fit(scaled_mat,labels,fracNCA = 0.25, silent = True,ret_loss = True)


	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['tenx']
embed += ['Cell Type MCML 50D']
numCells += [rawcount_mat.shape[0]]





#Mouse atlas run

#scvi
count_mat = sio.mmread(data_path+'/dev_all_hvg.mtx')
count_mat = count_mat.todense()

print(count_mat.shape)

rawcount_mat = sio.mmread(data_path+'/dev_all_raw.mtx')
rawcount_mat  = rawcount_mat.todense()
print(rawcount_mat.shape)

meta = pd.read_csv(data_path+'/lamannometadata.csv',index_col = 0)

meta.head()

#Filter out nan cells from counts

rawcount_mat = rawcount_mat[meta.ClusterName == meta.ClusterName,:]
count_mat = count_mat[meta.ClusterName == meta.ClusterName,:]

print(count_mat.shape)
print(rawcount_mat.shape)

meta = meta[meta.ClusterName == meta.ClusterName]

#Center and scale log-normalized data
scaled_mat = scale(count_mat)

clusters = np.unique(meta['ClusterName'].values)
map_dict = {}
for i, c in enumerate(clusters):
	map_dict[c] = i
new_labs = [map_dict[c] for c in meta['ClusterName'].values]

adata = anndata.AnnData(count_mat, obs = meta)
adata.X = np.nan_to_num(adata.X)

adata2 = anndata.AnnData(rawcount_mat, obs = meta)
adata2.X = np.nan_to_num(adata2.X)


# # In[20]:
# SCANVI accuracy scores
scvi.data.setup_anndata(adata2, labels_key='ClusterName')

for i in range(1):
	tic = time.perf_counter()

	vae = scvi.model.SCANVI(adata2, np.nan)
	vae.train(train_size = 0.7)
	latent_scanvi = vae.get_latent_representation()
	lab_idx = vae.train_indices

	toc = time.perf_counter()

	print(lab_idx)

alltime += [toc-tic]
dataset += ['lamanno']
embed += ['SCANVI']
numCells += [rawcount_mat.shape[0]]


# # LDVAE accuracy scores
scvi.data.setup_anndata(adata2, labels_key='ClusterName')

for i in range(1):
	tic = time.perf_counter()

	vae = scvi.model.LinearSCVI(adata2)
	vae.train()
	latent_ldvae = vae.get_latent_representation()
	lab_idx = vae.train_indices

	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['lamanno']
embed += ['LDVAE']
numCells += [rawcount_mat.shape[0]]

#recon mcml

lab1 = list(meta.ClusterName)
lab2 = list(meta.Age)
# lab3 = list(meta.medical_cond_label)


allLabs = np.array([lab1])
allLabs2 = np.array([lab1,lab2])

nanLabs = np.array([[np.nan]*len(lab1)])

# # Reconstruction loss only

for i in range(1):
	labels = np.array([lab1])
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan

	tic = time.perf_counter()
	ncaR = NN_NCA(n_latent = n_latent, epochs = 100)

	lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True)
	toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['lamanno']
embed += ['Recon MCML 50D']
numCells += [rawcount_mat.shape[0]]

#label mcml

for b in [0.99]:
# fracNCA = 0.5 

	for i in range(1):
		labels = np.array([lab1])
		train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
		unlab_inds = [i for i in range(len(adata)) if i not in train_inds]

		labels[:, unlab_inds] = np.nan

		tic = time.perf_counter()
		nca = NN_NCA(n_latent = n_latent, epochs = 100)
		#ncaR2 = NN_NCA(n_latent = n_latent, epochs = 100)


		losses, latent = nca.fit(scaled_mat,labels,fracNCA = b, silent = True,ret_loss = True)
		#losses2, latent2 = ncaR2.fit(scaled_mat,labels2,fracNCA = b, silent = True,ret_loss = True)
			
		toc = time.perf_counter()

alltime += [toc-tic]
dataset += ['lamanno']
embed += ['Cell Type MCML 50D']
numCells += [rawcount_mat.shape[0]]

times['Time'] = alltime
times['Dataset'] = dataset
times['Embed'] = embed
times['Cells'] = numCells

times.to_csv('timeComparisonsMCML.csv')







