from matplotlib import pyplot as plt
import cv2
import random
from descartes import PolygonPatch
from matplotlib.path import Path
import matplotlib.path as mplPath
import matplotlib.patches as patches

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
	plt.show()


def disegna_hough(img, lines):
	'''
plotta le hough lines
	'''
	#plt.title('hough')
	img2 = img.copy()
	if(len(img2.shape)==2):
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
	#plt.subplot(221)
	plt.title('hough lines')
	for x1,y1,x2,y2 in lines[0]:
    		cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
	plt.imshow(img2,cmap='Greys')
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


def disegna_canny(edges_canny):
	'''
plotta gli edge individuati da canny
	'''
	plt.title('canny')
	plt.imshow(edges_canny,cmap='Greys')
	plt.show()


def disegna_dbscan(labels, facce, facce_poligoni, xmin, ymin, xmax, ymax, edges, contours):
	'''
disegna le facce in base ai cluster ottenuti dal dbscan. Facce dello stesso cluster hanno stesso colore.
	'''
	fig = plt.figure()
	colori_assegnati = []
	colors = []
	colors.extend(('#800000','#DC143C','#FF0000','#FF7F50','#F08080','#FF4500','#FF8C00','#FFD700','#B8860B','#EEE8AA','#BDB76B','#F0E68C','#808000','#9ACD32','#7CFC00','#ADFF2F','#006400','#90EE90','#8FBC8F','#00FA9A','#20B2AA','#00FFFF','#4682B4','#1E90FF','#000080','#0000FF','#8A2BE2','#4B0082','#800080','#FF00FF','#DB7093','#FFC0CB','#F5DEB3','#8B4513','#808080'))
	#plt.subplot(224)
	plt.title('dbscan')
	ax = fig.add_subplot(111)
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
			plt.plot(ascisse,ordinate, color='k', linewidth=4.0)
			del ascisse[:]
			del ordinate[:]
	ascisse = []
	ordinate = []
	for c1 in contours:
		for c2 in c1:
			ascisse.append(c2[0][0])
			ordinate.append(c2[0][1])
		plt.plot(ascisse,ordinate,color='0.8',linewidth=3.0)
		del ascisse[:]
		del ordinate[:]
	#plt.show()
	return (colori_assegnati, fig, ax)

def disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax):
	'''
disegna il layout delle stanze.
	'''
	fig = plt.figure()
	#plt.subplot(224)
	plt.title('stanze')
	ax = fig.add_subplot(111)
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc=colori[index],ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
	#plt.show()
	return (fig, ax)


def disegna_contorno(vertici,xmin,ymin,xmax,ymax):
	'''
disegna il contorno esterno della mappa metrica
	'''
	bbPath = mplPath.Path(vertici)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	patch = patches.PathPatch(bbPath, facecolor='orange', lw=2)
	ax.add_patch(patch)
	ax.set_xlim(xmin-1,xmax+1)
	ax.set_ylim(ymin-1,ymax+1)
	plt.title('contorno esterno')
	plt.show()
