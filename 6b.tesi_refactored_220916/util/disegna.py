from matplotlib import pyplot as plt
import cv2
import random
from descartes import PolygonPatch
from matplotlib.path import Path
import matplotlib.path as mplPath
import matplotlib.patches as patches
import networkx as nx
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import os
from scipy import ndimage

def setup_plot():

	plt.clf()
	plt.cla()
	plt.close()
	fig, ax = plt.subplots()
	plt.axis('equal')
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])
	return fig,ax

def disegna(lista):
	'''
	disegna una lista di Segmenti
	'''
	
	ascisse = []
	ordinate = []	
	for muro in lista:
		ascisse.append(muro.x1)
		ascisse.append(muro.x2)
		ordinate.append(muro.y1)
		ordinate.append(muro.y2)
		plt.plot(ascisse,ordinate, color='k', linewidth=2.0)
		del ascisse[:]
		del ordinate[:]
	plt.title('test')
	plt.savefig('culotroia.png')

def disegna_map(map,savefig = True, format='pdf', filepath = '.', savename = '0_Map',title=False) :
	'''
	Stampa la mappa di ingresso

	'''
	fig,ax = setup_plot()
	ax.imshow(map)

	savename = os.path.join(filepath, savename+'.'+format)

	if title :
		ax.set_title('0.metric map')
	if savefig :
		plt.savefig(savename,bbox_inches = 'tight')
	else:
		plt.show()

def disegna_hough(img, lines,savefig = True, format='pdf', filepath = '.', savename ='2_Hough', title = False):
	'''
	plotta le hough lines

	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)


	img2 = img.copy()
	if(len(img2.shape)==2):
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
	if title :
		ax.set_title('1.hough lines') #1.primo plot

	for x1,y1,x2,y2 in lines:
    		cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
	ax.imshow(img2,cmap='Greys')
	if savefig :
		plt.savefig(savename, bbox_inches='tight')
	else:
		plt.show()
	'''
	ascisse = []
	ordinate = []
	for l in lines[0]:
		ascisse.extend((l[0],l[2]))
		ordinate.extend((l[1],l[3]))
		plt.plot(ascisse,ordinate, color='k', linewidth=2.0)
		ascisse = []
		ordinate = []
	plt.show()
	'''
def disegna_grafici_per_accuracy(stanze, stanze_gt, savefig = True, format='pdf', filepath = '.', savename = '9_accuracy', title = False):

	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	if title:
		ax.set_title('accuracy')
		
	for poly in stanze:
		x,y = poly.exterior.xy
		ax.plot(x,y, color= 'BLACK', alpha = 0.7, linewidth = 3, solid_capstyle= 'round', zorder = 2 )
	for poly in stanze_gt:
		x,y = poly.exterior.xy
		ax.plot(x,y, color= 'RED', alpha = 0.7, linewidth = 3, solid_capstyle= 'round', zorder = 2 )	
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()


def disegna_canny(edges_canny, savefig = True, format='pdf', filepath = '.', savename = '1_Canny', title = False):
	'''
	plotta gli edge individuati da canny
	'''

	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	if title:
		ax.set_title('canny')
	ax.imshow(edges_canny,cmap='Greys')
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()

def disegna_segmenti(lista,savefig = True, format='pdf', filepath = '.', savename = '3_Muri', title = False):
	'''
	disegna i segmenti passati come lista
	'''
	fig,ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	if title:
		ax.set_title('3.Muri') #2.muri, plotta i muri 
	ascisse = []
	ordinate = []
	for s in lista:
		x1 = s.x1
		x2 = s.x2
		y1 = s.y1
		y2 = s.y2
		ascisse.extend((x1,x2))
		ordinate.extend((y1,y2))
		ax.plot(ascisse,ordinate, color='k',linewidth=2)
		del ascisse[:]
		del ordinate[:]
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()


def disegna_dbscan(labels, facce, facce_poligoni, xmin, ymin, xmax, ymax, edges, contours,savefig = True, format='pdf', filepath = '.', savename = '6_DBSCAN', title = False):
	'''
	disegna le facce in base ai cluster ottenuti dal dbscan. Facce dello stesso cluster hanno stesso colore.
	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	colori_assegnati = []
	colors = []
	colors.extend(('#800000','#DC143C','#FF0000','#FF7F50','#F08080','#FF4500','#FF8C00','#FFD700','#B8860B','#EEE8AA','#BDB76B','#F0E68C','#808000','#9ACD32','#7CFC00','#ADFF2F','#006400','#90EE90','#8FBC8F','#00FA9A','#20B2AA','#00FFFF','#4682B4','#1E90FF','#000080','#0000FF','#8A2BE2','#4B0082','#800080','#FF00FF','#DB7093','#FFC0CB','#F5DEB3','#8B4513','#808080'))
	#plt.subplot(224)
	if title :
		ax.set_title('6.dbscan')
	for label in set(labels):
		col = random.choice(colors)
		colori_assegnati.append(col)
		for index,l in enumerate(labels):
			if (l == label):
				f = facce[index]
				f_poly = facce_poligoni[index]
				f_patch = PolygonPatch(f_poly,fc=col,ec='BLACK')
				ax.add_patch(f_patch)
				ax.set_xlim(xmin,xmax)
				ax.set_ylim(ymin,ymax)
				sommax = 0
				sommay = 0
				for b in f.bordi:
					sommax += (b.x1)+(b.x2)
					sommay += (b.y1)+(b.y2)
				xtesto = sommax/(2*len(f.bordi))
				ytesto = sommay/(2*len(f.bordi))
				ax.text(xtesto,ytesto,str(l),fontsize=8)
	ascisse = []
	ordinate = []
	for edge in edges:
		if (edge.weight>=0.3):
			ascisse.append(edge.x1)
			ascisse.append(edge.x2)
			ordinate.append(edge.y1)
			ordinate.append(edge.y2)
			ax.plot(ascisse,ordinate, color='k', linewidth=4.0)
			del ascisse[:]
			del ordinate[:]
	ascisse = []
	ordinate = []
	for c1 in contours:
		for c2 in c1:
			ascisse.append(c2[0][0])
			ordinate.append(c2[0][1])
		ax.plot(ascisse,ordinate,color='0.8',linewidth=3.0)
		del ascisse[:]
		del ordinate[:]
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
	return (colori_assegnati, fig, ax)

def disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax,savefig  =  True, format='pdf', filepath = '.', savename = '8_Stanze', title = False):
	'''
	disegna il layout delle stanze.
	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	#plt.subplot(224)
	if title :
		ax.set_title('7.stanze')
	
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc=colori[index],ec='BLACK') 
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
	return (fig, ax)


def disegna_contorno(vertici,xmin,ymin,xmax,ymax,savefig = True, format='pdf', filepath = '.', savename = '4_Contorno', title = False):
	'''
	disegna il contorno esterno della mappa metrica
	'''

	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)



	bbPath = mplPath.Path(vertici)
	patch = patches.PathPatch(bbPath, facecolor='orange', lw=2)
	ax.add_patch(patch)
	ax.set_xlim(xmin-1,xmax+1)
	ax.set_ylim(ymin-1,ymax+1)
	if title :
		ax.set_title('4.contorno esterno')
	if savefig :
		plt.savefig(savename)#bbox_inches='tight')
	else :
		plt.show()

# def disegna_contorno(vertici,xmin,ymin,xmax,ymax):
# 	print vertici
# 	bbPath = mplPath.Path(vertici)
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	patch = patches.PathPatch(bbPath, facecolor='orange', lw=2)
# 	ax.add_patch(patch)
# 	ax.set_xlim(xmin-1,xmax+1)
# 	ax.set_ylim(ymin-1,ymax+1)
# 	plt.show()


def disegna_cluster_angolari(cluster_centers, lista_muri, cluster_angolari, savefig = True,  format='pdf', filepath = '.',savename = '5b_Contorno', title = False):
	'''
	disegna con lo stesso colore i muri con stesso cluster angolare
	''' 
	ascisse = []
	ordinate = []
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	if title: 
		ax.set_title('cluster angolari')
	#per ogni cluster angolare visualizzo in rosso i segmenti che vi appartengono, in nero quelli che non vi appartengono
	for c in set(cluster_angolari):
		for muro1 in lista_muri:
			if muro1.cluster_angolare == c:
				ascisse.extend((muro1.x1,muro1.x2))
				ordinate.extend((muro1.y1,muro1.y2))
				ax.plot(ascisse,ordinate, color='r', linewidth=2.0)
				del ascisse[:]
				del ordinate[:]
			else:
				ascisse.extend((muro1.x1,muro1.x2))
				ordinate.extend((muro1.y1,muro1.y2))
				ax.plot(ascisse,ordinate, color='g', linewidth=2.0)
				del ascisse[:]
				del ordinate[:]
		
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()

def disegna_cluster_spaziali(cluster_spaziali, lista_muri,savefig = True, format='pdf', filepath = '.', savename = '5_MURA', title = False):
	'''
	disegna con lo stesso colore i muri con stesso cluster spaziale
	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	ascisse = []
	ordinate = []
	numofcolors = len(set(cluster_spaziali))
	cm = plt.get_cmap("nipy_spectral")
	cNorm = colors.Normalize(vmin=0, vmax=numofcolors)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap = cm)
	#plt.subplot(222)
	if title :
		ax.set_title('4.spatial clusters')
	for index,c in enumerate(np.random.permutation(list(set(cluster_spaziali)))):
		for muro in lista_muri:
			if muro.cluster_spaziale == c:
				ascisse.extend((muro.x1,muro.x2))
				ordinate.extend((muro.y1,muro.y2))
				ax.plot(ascisse,ordinate, color=colors.rgb2hex(scalarMap.to_rgba(index)), linewidth=2.0)
				del ascisse[:]
				del ordinate[:]
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
	
	#metodo alternativo
	'''
	#per ogni cluster spaziale visualizzo in rosso i segmenti che vi appartengono, in nero quelli che non vi appartengono
	for c in set(cluster_spaziali):
		for muro1 in lista_muri:
			if muro1.cluster_spaziale == c:
				ascisse.extend((muro1.x1,muro1.x2))
				ordinate.extend((muro1.y1,muro1.y2))
				plt.plot(ascisse,ordinate, color='r', linewidth=4.0)
				del ascisse[:]
				del ordinate[:]
			else:
				ascisse.extend((muro1.x1,muro1.x2))
				ordinate.extend((muro1.y1,muro1.y2))
				plt.plot(ascisse,ordinate, color='#51af42', linewidth=2.0)
				del ascisse[:]
				del ordinate[:]
		plt.show()

	raw_input("Press enter to exit")
	'''


def disegna_extended_segments(extended_segments, lista_muri, savefig = True, format='pdf', filepath = '.', savename = '7_Extended', title = False):
	'''
	disegna in nero i muri di lista_muri, in rosso gli extended_segments
	'''
	ascisse = []
	ordinate = []
	#plt.subplot(223)
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)

	if title :
		plt.ax.set_title('7.extended segments')
	for e in lista_muri:		
		ascisse.append(e.x1)
		ascisse.append(e.x2)
		ordinate.append(e.y1)
		ordinate.append(e.y2)
		ax.plot(ascisse,ordinate, color='k', linewidth=3.0)
		del ascisse[:]
		del ordinate[:]
	
	for e in extended_segments:		
		ascisse.append(e.x1)
		ascisse.append(e.x2)
		ordinate.append(e.y1)
		ordinate.append(e.y2)
		ax.plot(ascisse,ordinate, color='r', linewidth=1.5)
		del ascisse[:]
		del ordinate[:]

	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
		
		
def disegna_distance_transform(im_bw, savefig = True, format='pdf', filepath = '.', savename = '10_distance_transform', title = False):
	'''
	disegna la distance transforme
	'''

	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)
	
	if title :
		plt.title('10_distance_transform')
	
	distanceMap = ndimage.distance_transform_edt(im_bw)
	plt.imshow(distanceMap)
	
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
	
	return distanceMap

def disegna_medial_axis(points,b3, savefig = True, format='png', filepath = '.', savename = '11_medial_axis', title = False):
	'''
	disegna medial axis
	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)
	
	if title :
		plt.title('11_medial_axis')
	
	ax.plot(points[:,0],points[:,1],'.')
	ax.imshow(b3,cmap='Greys')
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
		
def plot_nodi_e_stanze(colori,xmin, xmax, ymin, ymax, G, pos, stanze,stanze_collegate, savefig = True, format='pdf', filepath = '.', savename = '12_grafo_topologico', title = False):
	'''
	disegna le stenze con i nodi corrispondenti
	'''
	fig, ax = setup_plot()
	savename = os.path.join(filepath, savename+'.'+format)
	
	if title :
		plt.title('12_grafo_topologico')
		
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc=colori[index],ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)

	#plotto il grafo
	nx.draw_networkx_nodes(G,pos,node_color='w')
	#nx.draw_networkx_edges(G,pos)
	nx.draw_networkx_labels(G,pos)

	#plotto gli edges come linee da un representative point a un altro, perche' con solo drawedges non li plotta. Forse sono nascosti dai poligoni.
	for coppia in stanze_collegate:
		i1 = stanze[coppia[0]]
		i2 = stanze[coppia[1]]
		p1 = i1.representative_point()
		p2 = i2.representative_point()
		plt.plot([p1.x,p2.x],[p1.y,p2.y],color='k',ls = 'dotted', lw=0.5)	
	
		
	if savefig :
		plt.savefig(savename,bbox_inches='tight')
	else :
		plt.show()
	
	return G, pos