import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
import itertools
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from MCML import MCML
import densne
import pandas as pd
#Centroids of clusters/labels
def getCentroidDists(embed,clusType):
	""" Compute inter-distances for a set of labels
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	clusType : List of labels for a class
	Returns:
	List of pairwise distances between centroids of labels in clusType
	"""
	clusters = np.unique(clusType)

	centroids = np.zeros((len(clusters),embed.shape[1]))

	for i in range(len(clusters)):

		sub_data = embed[clusType == clusters[i],:]
		centroid = sub_data.mean(axis=0)

		centroids[i,:] = list(centroid)

	dists = pairwise_distances(centroids,centroids,metric='l1')

	return dists.flatten().tolist()

#Get distances to centroids of clusters/labels
def getCentroidDists_oneVsAll(embed,clusType,clus):
	""" Compute inter-distances for one label versus the remaining
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	clusType : List of labels for a class
	clus : Specific label from which to calculate inter-distances
	Returns:
	List of distances between centroid of clus label to all other lable centroids in clusType
	"""
	clusters = np.unique(clusType)

	centroids = np.zeros((len(clusters),embed.shape[1]))

	comp = embed[clusType == clus,:]
	comp_centroid = comp.mean(axis=0)

	for i in range(len(clusters)):

		sub_data = embed[clusType == clusters[i],:]
		centroid = sub_data.mean(axis=0)

		centroids[i,:] = list(centroid)

	dists = pairwise_distances(comp_centroid.reshape(1, -1),centroids,metric='l1')

	return dists.flatten().tolist()

#Pairwise distances between binary labels
def getPairwise(embed,clusType):
	""" Compute inter-distances for binary label
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	clusType : List of labels for a class, where only two unique labels possible
	Returns:
	List of pairwise distances between all points in each of the two labels
	"""

	clusters = np.unique(clusType)

	sub1 = embed[clusType == clusters[0],:]
	sub2 = embed[clusType == clusters[1],:]

	dists = pairwise_distances(sub1,sub2,metric='l1')
	dists = dists.flatten().tolist()

	return dists

def getIntraVar(embed, outLab, inLab):
	""" Compute intra-distances for inner label
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	outLab : 1D array for outer label (e.g. cell type)
	inLab : 1D array for inner label (e.g. sex)
	Returns:
	List of average pairwise distances within labels in inLab
	"""
	outs = np.unique(outLab)
	avg_dists = []

	for i in outs:

		sub = embed[outLab == i,:]

		sub_ins = inLab[outLab == i]
		ins = np.unique(sub_ins)

		for j in ins:

			sub_i = sub[sub_ins == j,:]
			if sub_i.shape[0] > 1:
				
				d = pairwise_distances(sub_i,sub_i,metric='l1')
				np.fill_diagonal(d, np.nan)
				d = d[~np.isnan(d)].reshape(d.shape[0], d.shape[1] - 1)

				f_d = d.flatten().tolist()

				avg_dists += [np.mean(f_d)] 

	return avg_dists

def getInterVar(embed, outLab, inLab):
	""" Compute inter-distances for inner label
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	outLab : 1D array for outer label (e.g. cell type)
	inLab : 1D array for inner label (e.g. sex)
	Returns:
	List of average pairwise distances between labels in inLab
	"""


	outs = np.unique(outLab)
	avg_dists = []

	for i in outs:
		sub = embed[outLab == i,:]

		sub_ins = inLab[outLab == i]
		ins = np.unique(sub_ins)

		pairs = list(itertools.combinations(ins, 2))
		for p in pairs:

			sub_1 = sub[sub_ins == p[0],:]
			sub_2 = sub[sub_ins == p[1],:]
			avg_dists += [np.mean(pairwise_distances(sub_1,sub_2,metric='l1').flatten().tolist())]



	return avg_dists


def ecdf(data):
	""" Compute eCDF
	data : List of values
	Returns:
	Tuple of x and y values for eCDF plot
	"""
	x = np.sort(data)
	n = len(x)
	y = np.arange(n) / float(n)


	return(x,y)

def getNeighbors(embed, n_neigh = 15, p=1):
	"""Get indices of nearest neighbors in embedding 
	embed : Numpy array for latent space (n_obs x n_features or n_latent)
	n_neigh : No. of neighbors for each cell
	p : Distance metric (1= Manhattan, 2= Euclidean)
	Returns:
	indices : Matrix of n_obs x n_neigh with indices of nearest neighbors for each obs
	"""
	nbrs = NearestNeighbors(n_neighbors=n_neigh, p=p).fit(embed)
	distances, indices = nbrs.kneighbors(embed)

	return indices

def getIntersect(orig, new):
	"""Get frac of neighbors intersecting in ambient space
	orig : Original/ambient space nearest neighbor indices, from getNeighbors()
	new : Latent/Comparison space nearest neighbor indices, from getNeighbors()
	Returns:
	frac : List of fraction of neighbors shared in new space, for each obs"""
	frac = [0]*new.shape[0]
	for i in range(new.shape[0]):
		inter = set(orig[i,:]).intersection(new[i,:])
		frac[i] = len(inter)/orig.shape[1]

	return frac


def getJaccard(orig, new):
	"""Get jaccard distance of neighbors intersecting with those in ambient space
	orig : Original/ambient space nearest neighbor indices, from getNeighbors()
	new : Latent/Comparison space nearest neighbor indices, from getNeighbors()
	Returns:
	frac : List of Jaccard distances for each obs
	"""
	frac = [0]*new.shape[0]
	for i in range(new.shape[0]):
		inter = set(orig[i,:]).intersection(new[i,:])
		frac[i] = 1 - len(inter)/len(set(orig[i,:]).union(new[i,:]))

	return frac


def frac_unique_neighbors(latent, cluster_label, metric = 1,neighbors = 30):
	""" Calculates the fraction of nearest neighbors from same cell type
	latent : numpy array of latent space (n_obs x n_latent)
	cluster_label : list of labels for all n_obs
	metrics : Distance metric, 1 = manhattan
	neighbors : No. of nearest neighbors to consider
	Returns:
	Dictionary mapping each unique label in the class cluster_label
	to list of fraction of neighbors in the same label for each cell, and Dictionary mapping each unique label in the category cluster_label
	to a list of unique labels of each cell's neighbors
	"""
	cats = pd.Categorical(cluster_label)
	# Get nearest neighbors in each space
	n = neighbors
	neigh = NearestNeighbors(n_neighbors=n, p=metric)
	# Get transformed count matrices
	clusters = np.unique(cluster_label)
	unique_clusters = {}
	frac_neighbors = {}
	X_full  = latent
	neigh.fit(X_full)
	for c in clusters:
		X  = latent[cats == c, :]
		# Find n nearest neighbor cells (L1 distance)
		kNeigh = neigh.kneighbors(X)
		matNeigh = kNeigh[1]
		frac = np.zeros(matNeigh.shape[0])
		#How many of top n neighbors come from same cluster in the labeled data (out of n neighbors)
		unique_clusters[c] = np.unique([cats[matNeigh[i]] for i in range(0, len(frac))])
		frac_neighbors[c] = [cats[matNeigh[i]].value_counts()[c]/n for i in range(0,len(frac))]
	return frac_neighbors, unique_clusters

def knn_infer(embd_space, labeled_idx, labeled_lab, unlabeled_idx,n_neighbors=10):
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



	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(labeled_samp, labeled_lab)

	pred_lab = knn.predict(unlabeled_samp)
	return pred_lab

def knnReg_infer(embd_space, labeled_idx, labeled_lab, unlabeled_idx,n_neighbors=10):
	"""
	Predicts the  continuous labels of unlabeled data in the embedded space with KNN.
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
		Inferred labels (continuous) of the unlabeled samples.
	"""

	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]



	knn = KNeighborsRegressor(n_neighbors=n_neighbors)
	knn.fit(labeled_samp, labeled_lab)

	pred_lab = knn.predict(unlabeled_samp)
	return pred_lab

def visComp(scaled_mat, ndims=2, pcs=50, rounds = 5):
	""" Compute latent space representations usually used for visualization
	scaled_mat : Numpy array of latent space (n_obs x n_latent)
	ndims : No. of dimensions to reduce scaled_mat to
	pcs : No. of PCs to use
	rounds : No. of rounds to replicate over
	Returns:
	latents : List containing each generated latent space
	latentLab : List containing label for each latent space
	latentType : List containing broad category label for each latent space
	"""
	latents = []
	latentLab = []
	latentType = []

	nanLabs = np.array([[np.nan]*scaled_mat.shape[0]])

	for i in range(rounds):
		reducer = umap.UMAP(n_components = ndims) # random_state = state
		densUMAP = umap.UMAP(n_components = ndims,densmap=True)
		tsne = TSNE(n_components = ndims) 

		tsvd = TruncatedSVD(n_components=pcs)
		x_pca = tsvd.fit_transform(scaled_mat)

		tsvd = TruncatedSVD(n_components=2)
		x_pca_2d = tsvd.fit_transform(scaled_mat)

		pcaUMAP = reducer.fit_transform(x_pca)
		pcaDensUMAP = densUMAP.fit_transform(x_pca)

		pcaTSNE = tsne.fit_transform(x_pca)
		pcaDensTSNE, ro ,re = densne.run_densne(x_pca,no_dims = ndims) # randseed = state

	

		latents += [x_pca, x_pca_2d,  pcaTSNE,  pcaUMAP,pcaDensTSNE,  pcaDensUMAP ]
		latentLab += ['PCA 50D','PCA 2D','PCA TSNE','PCA UMAP','PCA densSNE','PCA densMAP']
		latentType += ['50D','2D','2D','2D','dens 2D','dens 2D']

	return latents,latentLab,latentType

def visCompAll(scaled_mat, ndims=2, pcs=50, rounds = 5):
	""" Compute latent space representations usually used for visualization
	scaled_mat : Numpy array of latent space (n_obs x n_latent)
	ndims : No. of dimensions to reduce scaled_mat to
	pcs : No. of PCs to use
	rounds : No. of rounds to replicate over
	Returns:
	latents : List containing each generated latent space
	latentLab : List containing label for each latent space
	latentType : List containing broad category label for each latent space
	"""
	latents = []
	latentLab = []
	latentType = []

	nanLabs = np.array([[np.nan]*scaled_mat.shape[0]])

	for i in range(rounds):
		reducer = umap.UMAP(n_components = ndims) # random_state = state
		densUMAP = umap.UMAP(n_components = ndims,densmap=True)
		tsne = TSNE(n_components = ndims) 

		tsvd = TruncatedSVD(n_components=pcs)
		x_pca = tsvd.fit_transform(scaled_mat)

		tsvd = TruncatedSVD(n_components=2)
		x_pca_2d = tsvd.fit_transform(scaled_mat)

		pcaUMAP = reducer.fit_transform(x_pca)
		pcaDensUMAP = densUMAP.fit_transform(x_pca)

		pcaTSNE = tsne.fit_transform(x_pca)
		pcaDensTSNE, ro ,re = densne.run_densne(x_pca,no_dims = ndims) # randseed = state

		#MCML runs
		ncaR = MCML(n_latent = pcs, epochs = 100)

		lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True)

		latentRUMAP = reducer.fit_transform(latentR)

		latentRTSNE = tsne.fit_transform(latentR)

		latentRDensUMAP = densUMAP.fit_transform(latentR)

		latentRDensTSNE, ro ,re = densne.run_densne(latentR, no_dims = ndims)

		latents += [latentR, x_pca, x_pca_2d, latentRTSNE, pcaTSNE, latentRUMAP, pcaUMAP,latentRDensTSNE, pcaDensTSNE, latentRDensUMAP, pcaDensUMAP ]
		latentLab += ['Recon MCML 50D','PCA 50D','PCA 2D','Recon MCML TSNE','PCA TSNE','Recon MCML UMAP','PCA UMAP','Recon MCML denSNE','PCA densSNE','Recon MCML densMAP','PCA densMAP']
		latentType += ['50D','50D','2D','2D','2D','2D','2D','dens 2D','dens 2D','dens 2D','dens 2D']

	return latents,latentLab,latentType


def reconComp(scaled_mat, ndims=2, pcs=50, rounds = 3):
	""" Compute latent space representations as baseline for reconstruction abilities
	scaled_mat : Numpy array of latent space (n_obs x n_latent)
	ndims : No. of dimensions to reduce scaled_mat to
	pcs : No. of PCs to use
	rounds : No. of rounds to replicate over
	Returns:
	latents : List containing each generated latent space
	latentLab : List containing label for each latent space
	latentType : List containing broad category label for each latent space
	"""

	latents = []
	latentLab = []
	latentType = []

	nanLabs = np.array([[np.nan]*scaled_mat.shape[0]])

	for i in range(rounds):
		tsvd = TruncatedSVD(n_components=pcs)
		x_pca = tsvd.fit_transform(scaled_mat)

		tsvd = TruncatedSVD(n_components=2)
		x_pca_2d = tsvd.fit_transform(scaled_mat)


		#MCML runs
		ncaR = MCML(n_latent = pcs, epochs = 100)

		lossesR, latentR = ncaR.fit(scaled_mat,nanLabs,fracNCA = 0, silent = True,ret_loss = True)

		latents += [latentR, x_pca, x_pca_2d]
		latentLab += ['Recon MCML 50D','PCA 50D','PCA 2D']
		latentType += ['50D','50D','2D']

	return latents,latentLab,latentType




