import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import umap
import pandas as pd
import seaborn as sns

def tSNE(adata, label, latent, ndims = 2, state = 42, metric = "euclidean", perplexity = 30):
	"""
	Parameters:
    adata : AnnData object to perform tSNE on
    label : observation category (string) to use for tSNE
    ndims : (optional) number of dimensions for tSNE, default = 2
    state : (optional) integer for random_state of tSNE, default = 42
    metric : (optional) string of metric to use for tSNE, default = euclidean
    perplexity: (optional) float of perplexity value for tSNE, default = 50
    Returns :
    Modifies adata in place to add tSNE space in the obsm category
    """
	tsne = TSNE(n_components = ndims, metric = metric, random_state = state, perplexity = perplexity) 
	adata.obsm[label + "_tsne"] = tsne.fit_transform(latent)

def UMAP(adata, label, latent, ndims = 2, state = 42, metric = "euclidean", neighbors = 15):
	"""
	Parameters:
    adata : AnnData object to perform UMAP on
    label : observation category (string) to use for UMAP
    ndims : (optional) number of dimensions for UMAP, default = 2
    state : (optional) integer for random_state of UMAP, default = 42
    metric : (optional) string of metric to use for UMAP, default = euclidean
    neighbors: (optional) int of n_neighbors value for UMAP, default = 50
    Returns :
    Modifies adata in place to add UMAP space in the obsm category
    """
	reducer = umap.UMAP(n_neighbors = neighbors, n_components = ndims, metric = metric, random_state = state)
	adata.obsm[label + "_umap"] = reducer.fit_transform(latent)

def obj_plot_embed(latent, cluster_label, fname = None, colors = [], alpha=0.4,figsize=(7,5)):
	""" Plot latent space in 2D and color cells by cluster_label """

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)
	#cm.get_cmap("tab20")
	cluster_labels = pd.unique(cluster_label)

	cmap = np.random.rand(len(cluster_labels),3)

	for i, c in (enumerate(cluster_labels)):

		XX = latent[cluster_label == c,:]
		x = XX[:,0]
		y = XX[:,1]

		if(len(colors) >= len(cluster_labels)):
			color = colors[i]
		else:
			color = cmap[i,:]

		ax.scatter(x, y, s=5, alpha = alpha, label=c, color = color)
		
	ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 8},frameon=False,ncol=2)
	ax.set_axis_off()

	fig.tight_layout()
	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()

def obj_plot_subembed(latent, cluster_label, dimx=0, dimy=1, fname = None, colors = [], alpha=0.4, figsize=(7,5),axisFontSize = 11, tickFontSize = 10):
	""" Plot specified dimensions of latent space in 2D and color cells by cluster_label """

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)
	#cm.get_cmap("tab20")
	cluster_labels = pd.unique(cluster_label)

	cmap = np.random.rand(len(cluster_labels),3)

	for i, c in (enumerate(cluster_labels)):

		XX = latent[cluster_label == c,:]
		x = XX[:,dimx]
		y = XX[:,dimy]

		if(len(colors) >= len(cluster_labels)):
			color = colors[i]
		else:
			color = cmap[i,:]

		ax.scatter(x, y, s=5, alpha = alpha, label=c, color = color)
		
	ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 8},frameon=False,ncol=3)
	#ax.set_axis_off()

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.xlabel(r'$Z_{{{}}}$'.format(dimx),fontsize=axisFontSize)
	#plt.xlabel(r"$Z_{"+str(dimx)+"}$",fontsize=axisFontSize)
	#plt.ylabel("Pearsonr (to Ambient)",fontsize=axisFontSize)
	plt.ylabel(r'$Z_{{{}}}$'.format(dimy),fontsize=axisFontSize)
	plt.xticks(fontsize=tickFontSize)
	plt.yticks(fontsize=tickFontSize)

	plt.grid(False)

	fig.tight_layout()
	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()

def obj_plot_annot(latent, annots, dimx=0, dimy=1, fontsize = 5, fname = None, alpha=0.4, figsize=(7,5),axisFontSize = 11, tickFontSize = 10):
	""" Plot specified dimensions of latent space in 2D and annotate top 3 genes in each dimension """

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)
	#cm.get_cmap("tab20")


	XX = latent
	x = XX[:,dimx]
	y = XX[:,dimy]

	sortx = x.argsort()[::-1][0:3] #Top 3 genes
	sorty = y.argsort()[::-1][0:3]

	allInds = list(sortx) + list(sorty)

	ax.scatter(x, y, s=5, alpha = alpha, color = 'lightgray')

	for i in allInds:
		ax.annotate(annots[i], (x[i], y[i]),fontsize=fontsize)
		ax.scatter(x[i], y[i], s=5, alpha = alpha, color = 'royalblue')

	
		
	#ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 8},frameon=False,ncol=2)
	#ax.set_axis_off()

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.xlabel(r'$W_{{{}}}$'.format(dimx),fontsize=axisFontSize)
	#plt.xlabel(r"$Z_{"+str(dimx)+"}$",fontsize=axisFontSize)
	#plt.ylabel("Pearsonr (to Ambient)",fontsize=axisFontSize)
	plt.ylabel(r'$W_{{{}}}$'.format(dimy),fontsize=axisFontSize)
	plt.xticks(fontsize=tickFontSize)
	plt.yticks(fontsize=tickFontSize)

	plt.grid(False)
	fig.tight_layout()
	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()


def plot_embed(adata, label, cluster_label, fname = None, colors = []):
	""" Plot adata obsm in 2D and color cells by cluster_label """

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,5))
	#cm.get_cmap("tab20")
	cluster_labels = np.unique(adata.obs[cluster_label])

	cmap = np.random.rand(len(cluster_labels),3)

	for i, c in (enumerate(cluster_labels)):

		XX = adata[adata.obs[cluster_label] == c,:].obsm[label]
		x = XX[:,0]
		y = XX[:,1]

		if(len(colors) >= len(cluster_labels)):
			color = colors[i]
		else:
			color = cmap[i,:]

		ax.scatter(x, y, s=5, alpha = 0.4, label=c, color = color)
		
	ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 8},frameon=False,ncol=2)
	ax.set_axis_off()

	fig.tight_layout()
	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()

def plot_embed_3d(adata, label, cluster_label, fname = None, colors = []):
	""" Plot adata obsm in 3D and color cells by cluster_label """

	fig = plt.figure(figsize=(7,5))
	ax = fig.add_subplot(111, projection='3d')
	#cm.get_cmap("tab20")
	cluster_labels = np.unique(adata.obs[cluster_label])

	cmap = np.random.rand(len(cluster_labels),3)

	for i, c in (enumerate(cluster_labels)):

		XX = adata[adata.obs[cluster_label] == c,:].obsm[label]
		x = XX[:,0]
		y = XX[:,1]
		z = XX[:,2]

		if(len(colors) >= len(cluster_labels)):
			color = colors[i]
		else:
			color = cmap[i,:]

		ax.scatter(x, y,z, s=5, alpha = 0.4, label=c, color = color)
		
	ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 8},frameon=False,ncol=2)
	ax.set_axis_off()

	fig.tight_layout()
	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()

def plotLatentStats(allVals, axisFontSize = 11, tickFontSize = 10, errwidth=1, figsize =(8,4), dodge=0.4, fname = None, ymin = 0):
	""" Plot pearsonr correlation metrics for multiple latent spaces """

	plt.figure(figsize=figsize)
	g=sns.pointplot(x='Distance', y='Pearsonr', data=allVals, hue='Latent', err_style='bars',join=False,plot_kws=dict(alpha=0.6),errwidth=errwidth, dodge=dodge)
	plt.setp(g.collections, alpha=.6) #for the markers
	plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop={"size":10})

	plt.ylim(ymin=ymin)

	plt.xlabel("Distance Metric",fontsize=axisFontSize)
	plt.ylabel("Pearsonr (to Ambient)",fontsize=axisFontSize)
	plt.xticks(fontsize=tickFontSize)
	plt.yticks(fontsize=tickFontSize)
	plt.tight_layout()

	if(fname != None):
		plt.savefig(fname)
	else:
		plt.show()




# #https://bsouthga.dev/posts/color-gradients-with-python
# def hex_to_RGB(hex):
# 	''' "#FFFFFF" -> [255,255,255] '''
# 	# Pass 16 to the integer function for change of base
# 	return [int(hex[i:i+2], 16) for i in range(1,6,2)]

# def RGB_to_hex(RGB):
# 	''' [255,255,255] -> "#FFFFFF" '''
# 	# Components need to be integers for hex to make sense
# 	RGB = [int(x) for x in RGB]
# 	return "#"+"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB])

# def color_dict(gradient):
# 	''' Takes in a list of RGB sub-lists and returns dictionary of
# 	colors in RGB and hex form for use in a graphing function
# 	defined later on '''
# 	return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
# 		"r":[RGB[0] for RGB in gradient],
# 		"g":[RGB[1] for RGB in gradient],
# 		"b":[RGB[2] for RGB in gradient]}


# def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
# 	''' returns a gradient list of (n) colors between
# 	two hex colors. start_hex and finish_hex
# 	should be the full six-digit color string,
# 	including the number sign ("#FFFFFF") '''
# 	# Starting and ending colors in RGB form
# 	s = hex_to_RGB(start_hex)
# 	f = hex_to_RGB(finish_hex)
# 	# Initilize a list of the output colors with the starting color
# 	RGB_list = [s]
# 	# Calcuate a color at each evenly spaced value of t from 1 to n
# 	for t in range(1,n):
# 		# Interpolate RGB vector for color at the current value of t
# 		curr_vector = [int(s[j] + (float(t)/(n-1))*(f[j]-s[j])) for j in range(3)]
# 		# Add it to our list of output colors
# 		RGB_list.append(curr_vector)

# 	return color_dict(RGB_list)


