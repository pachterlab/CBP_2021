#----------------- Download Data -----------------
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


#dev_all_hvg.mtx
download_file('10.22002/D1.2043','.gz')

#dev_all_raw.mtx
download_file('10.22002/D1.2044','.gz')

#lamannometadata.csv
download_file('10.22002/D1.2045','.gz')

os.system("gunzip *.gz")

os.system("mv D1.2043 dev_all_hvg.mtx")
os.system("mv D1.2044 dev_all_raw.mtx")
os.system("mv D1.2045 metadata.csv")


os.system("pip3 install --quiet torch --no-cache-dir")
os.system("pip3 install --quiet anndata --no-cache-dir")
os.system("pip3 install --quiet matplotlib --no-cache-dir")
os.system("pip3 install --quiet scikit-learn --no-cache-dir")
os.system("pip3 install --quiet torchsummary --no-cache-dir")
os.system("pip install --quiet scanpy==1.6.0 --no-cache-dir")
#pip3 install --quiet umap-learn --no-cache-dir
os.system("pip3 install --quiet scvi-tools --no-cache-dir")


os.system("git clone https://github.com/pachterlab/CBP_2021.git")

os.chdir("./CBP_2021/scripts")


import anndata 
import pandas as pd
import numpy as np
from MCML import MCML #Now has continuous label addition
# import visualizations as vis
# import tools as tl
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
#sns.set_style('white')


#----------------- Code for testing Cell Label Prediction on Developing Mouse Brain data -----------------




plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.linewidth'] = 0.1

state = 42
ndims = 2

data_path = '../..'

pcs = 50
n_latent = 50




count_mat = sio.mmread(data_path+'/dev_all_hvg.mtx')
count_mat = count_mat.todense()

print(count_mat.shape)

rawcount_mat = sio.mmread(data_path+'/dev_all_raw.mtx')
rawcount_mat  = rawcount_mat.todense()
print(rawcount_mat.shape)






meta = pd.read_csv(data_path+'/metadata.csv',index_col = 0)

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




def knn_infer(embd_space, labeled_idx, labeled_lab, unlabeled_idx,n_neighbors=50):
	"""
	Predicts the labels of unlabeled data in the embedded space with KNN.
	Parameters
	----------
	embd_space : ndarray (n_samples, embedding_dim)
		Each sample is described by the features in the embedded space.
		Contains all samples, both labeled and unlabeled.
	labeled_idx : list
		Indices of the labeled samples (used for training the classifier).
	labeled_lab : ndarray (n_labeled_samples)
		Labels of the labeled samples.
	unlabeled_idx : list
		Indices of the unlabeled samples.
	Returns
	-------
	pred_lab : ndarray (n_unlabeled_samples)
		Inferred labels of the unlabeled samples.
	"""

	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]

	from sklearn.neighbors import KNeighborsClassifier

	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(labeled_samp, labeled_lab)

	pred_lab = knn.predict(unlabeled_samp)
	return pred_lab



# SCANVI accuracy scores
scvi.data.setup_anndata(adata2, labels_key='ClusterName')
acc_score_scanvi = []
acc_score_scanvi2 = []
for i in range(3):
	vae = scvi.model.SCANVI(adata2, np.nan)
	vae.train(train_size = 0.7)
	latent_scanvi = vae.get_latent_representation()
	lab_idx = vae.train_indices
	print(lab_idx)
	unlabeled_idx = []
	for i in range(len(adata2)):
		if i not in lab_idx:
			unlabeled_idx.append(i)
	preds = knn_infer(np.array(latent_scanvi), list(lab_idx), adata2.obs.ClusterName.values[lab_idx], unlabeled_idx)
	acc = accuracy_score(adata2.obs.ClusterName.values[unlabeled_idx], preds)
	acc_score_scanvi.append(acc)

	# preds2 = knn_infer(np.array(latent_scanvi), list(lab_idx), adata2.obs.Age.values[lab_idx], unlabeled_idx)
	# acc2 = accuracy_score(adata2.obs.Age.values[unlabeled_idx], preds2)
	# acc_score_scanvi2.append(acc2)

print(acc_score_scanvi)
print(acc_score_scanvi2)


# # # LDVAE accuracy scores
scvi.data.setup_anndata(adata2, labels_key='ClusterName')
acc_score = []
acc_score2 = []
for i in range(3):
	vae = scvi.model.LinearSCVI(adata2)
	vae.train() #train_size = 0.7
	latent_ldvae = vae.get_latent_representation()
	lab_idx = vae.train_indices
	unlabeled_idx = []
	for i in range(len(adata2)):
		if i not in lab_idx:
			unlabeled_idx.append(i)
	preds = knn_infer(np.array(latent_ldvae), list(lab_idx), adata2.obs.ClusterName.values[lab_idx], unlabeled_idx)
	acc = accuracy_score(adata2.obs.ClusterName.values[unlabeled_idx], preds)
	acc_score.append(acc)

	# preds2 = knn_infer(np.array(latent_ldvae), list(lab_idx), adata2.obs.Age.values[lab_idx], unlabeled_idx)
	# acc2 = accuracy_score(adata2.obs.Age.values[unlabeled_idx], preds2)
	# acc_score2.append(acc2)

print(acc_score)
print(acc_score2)







lab1 = list(meta.ClusterName)
lab2 = list(meta.Age)
# lab3 = list(meta.medical_cond_label)


allLabs = np.array([lab1])
allLabs2 = np.array([lab1,lab2])

nanLabs = np.array([[np.nan]*len(lab1)])

#Shuffled labels for over-fitting check
shuff_lab1 = random.sample(lab1, len(lab1))  
shuff_lab2 = random.sample(lab2, len(lab2))  
shuff_allLabs = np.array([shuff_lab1,shuff_lab2])






# # Reconstruction loss only
acc_scoreR = []
acc_scoreR2 = []
for i in range(3):
	tic = time.perf_counter()
	ncaR = MCML(n_latent = n_latent, epochs = 100)
	labels = np.array([lab1])
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan
		
	lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True)
	toc = time.perf_counter()
	unlabeled_idx = []
	for i in range(len(adata)):
			if i not in train_inds:
					unlabeled_idx.append(i)
	preds = knn_infer(latentR, train_inds, adata.obs.ClusterName.values[train_inds], unlabeled_idx)
	acc = accuracy_score(adata.obs.ClusterName.values[unlabeled_idx], preds)
	acc_scoreR.append(acc)

	# preds2 = knn_infer(latentR, train_inds, adata.obs.Age.values[train_inds], unlabeled_idx)
	# acc2 = accuracy_score(adata.obs.Age.values[unlabeled_idx], preds2)
	# acc_scoreR2.append(acc2)
	print(f"nnNCA fit in {toc - tic:0.4f} seconds")

print(acc_scoreR)
print(acc_scoreR2)
# # # In[24]:


# NCA loss (MCML)

acc_scoreBoth = []
acc_scoreBoth2 = []
acc_scoreBoth3 = []

for b in [0.99]:

	for i in range(3):
		tic = time.perf_counter()
		nca = MCML(n_latent = n_latent, epochs = 500)
		#ncaR2 = MCML(n_latent = n_latent, epochs = 100)

		labels = np.array([lab1])
		train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
		unlab_inds = [i for i in range(len(adata)) if i not in train_inds]

		labels[:, unlab_inds] = np.nan

		#2 labels
		labels2 = allLabs2.copy()
		labels2[:, unlab_inds] = np.nan

		losses, latent = nca.fit(scaled_mat,labels,fracNCA = b, silent = True,ret_loss = True)
		#losses2, latent2 = ncaR2.fit(scaled_mat,labels2,fracNCA = b, silent = True,ret_loss = True)
			
		toc = time.perf_counter()
		unlabeled_idx = []
		for i in range(len(adata)):
				if i not in train_inds:
						unlabeled_idx.append(i)
		preds = knn_infer(latent, train_inds, adata.obs.ClusterName.values[train_inds], unlabeled_idx)
		acc = accuracy_score(adata.obs.ClusterName.values[unlabeled_idx], preds)
		acc_scoreBoth.append(acc)



		# preds2 = knn_infer(latent2, train_inds, adata.obs.ClusterName.values[train_inds], unlabeled_idx)
		# acc2 = accuracy_score(adata.obs.ClusterName.values[unlabeled_idx], preds2)
		# acc_scoreBoth2.append(acc2)

		# preds2 = knn_infer(latent2, train_inds, adata.obs.Age.values[train_inds], unlabeled_idx)
		# acc2 = accuracy_score(adata.obs.Age.values[unlabeled_idx], preds2)
		# acc_scoreBoth3.append(acc2)

		print(f"nnNCA fit in {toc - tic:0.4f} seconds")

print(acc_scoreBoth)
# # print(acc_scoreBoth2)
# # print(acc_scoreBoth3)

#PCA 50D accuracy
acc_scorePCA = []

for i in range(3):

	tsvd = TruncatedSVD(n_components=pcs)
	x_pca = tsvd.fit_transform(scaled_mat)

	labels = np.array([lab1])
	train_inds = np.random.choice(len(scaled_mat), size = int(0.7*len(scaled_mat)),replace=False)
	unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
	labels[:, unlab_inds] = np.nan

	unlabeled_idx = []
	for i in range(len(adata)):
		if i not in train_inds:
			unlabeled_idx.append(i)

	preds = knn_infer(x_pca, train_inds, adata.obs.ClusterName.values[train_inds], unlabeled_idx)
	acc = accuracy_score(adata.obs.ClusterName.values[unlabeled_idx], preds)
	acc_scorePCA.append(acc)

print(acc_scorePCA)

#---------------- Save knn prediction accuracy scores for cell type labels ----------------
vals = pd.DataFrame()

vals['Accuracy'] = acc_score + acc_score_scanvi + acc_scoreR + acc_scoreBoth + acc_scorePCA #+ acc_score2 + acc_score_scanvi2 + acc_scoreR2 + acc_scoreBoth3 + acc_scoreBoth2 #+ netAE_score + netAE_score2

r = 3
vals['Embed'] = ['LDVAE']*r + ['SCANVI']*r + ['Recon MCML']*r + ['NCA-Recon MCML']*r +['PCA 50D']*r #+['LDVAE']*r + ['SCANVI']*r + ['Recon MCML']*r + ['NCA-Recon MCML']*r + ['NCA-Recon MCML']*r  #+ ['netAE']*2



vals['Label'] = ['CellType1']*15 #+ ['Gender2']*12 + ['CellType2']*1 #+  ['CellType1'] #+  ['Gender2']
vals.to_csv('allLaMannoPreds0712.csv')
print('Made CSV')




#---------------- Test MCML prediction accuracy with lower percentages of labeled data ----------------

acc_scoreBoth = []
percs = [0.7,0.6,0.5,0.4,0.3,0.2,0.1]

print('Starting lower percentages')

for p in percs:
  nca = mcml(n_latent = n_latent, epochs = 100)
  ncaR2 = mcml(n_latent = n_latent, epochs = 100)

  labels = np.array([lab1])
  train_inds = np.random.choice(len(scaled_mat), size = int(p*len(scaled_mat)),replace=False)
  unlab_inds = [i for i in range(len(adata)) if i not in train_inds]
  labels[:, unlab_inds] = np.nan
	

  #2 labels
  labels2 = allLabs2
  labels2[:, unlab_inds] = np.nan

  losses, latent = nca.fit(scaled_mat,labels,fracNCA = 0.99, silent = True,ret_loss = True)



  toc = time.perf_counter()
  unlabeled_idx = []
  for i in range(len(adata)):
	  if i not in train_inds:
		  unlabeled_idx.append(i)
  preds = knn_infer(latent, train_inds, adata.obs.ClusterName.values[train_inds], unlabeled_idx)
  acc = accuracy_score(adata.obs.ClusterName.values[unlabeled_idx], preds)
  acc_scoreBoth.append(acc)





lowPercsSmartSeq = pd.DataFrame()

lowPercsSmartSeq['Accuracy'] = acc_scoreBoth
lowPercsSmartSeq['Percent'] = percs

lowPercsSmartSeq

lowPercsSmartSeq.to_csv('lowPercsLaMannoPreds0715.csv')


