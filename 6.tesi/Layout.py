# -*- coding: utf-8 -*-

from __future__ import division
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import matplotlib.colors as colors
import numpy as np
import math
import matplotlib.pyplot as plt
from igraph import *
import matplotlib.path as mplPath
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.geometry import Point
import random
import networkx as nx

import segmento as sg
import mean_shift as ms
import retta as rt
import extended_segment as ext
import faccia as fc
import matrice as mtx
import image as im
import disegna as dsg

from itertools import cycle

import cv2
import matplotlib.image as mpimg

from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull
from shapely.ops import cascaded_union
import xml.etree.ElementTree as ET


def crea_muri(linee):
	'''
trasforma le linee in oggetti di tipo Segmento, e ne ritorna la lista.
	'''
	walls = []
	for muro in linee:
		x1 = float(muro[0])
		y1 = float(muro[1])
		x2 = float(muro[2])
		y2 = float(muro[3])
		walls.append(sg.Segmento(x1,y1,x2,y2))
	return walls
	

def clustering_dbscan_celle(eps, min_samples, X):
	'''
esegue il dbscan clustering sulle celle, prendendo in ingresso eps (massima distanza tra 2 campioni per essere considerati nello stesso neighborhood), min_samples (numero di campioni in un neighborhood per un punto per essere considerato un core point) e X (1-matrice di affinità locale tra le celle).
	'''
	af = DBSCAN(eps, min_samples, metric="precomputed").fit(X)
	#print("num of clusters = ")
	#print len(set(af.labels_))	
	return af.labels_


def flip_lines(linee, altezza):
	'''
flippa le y delle linee, perche' l'origine dei pixel è in alto a sx, invece la voglio in basso a sx
	'''
	for l in linee:
		l[1] = altezza-l[1]
		l[3] = altezza-l[3]
	return linee


def flip_contorni(contours, altezza):
	'''
flippa le y dei punti del contorno
	'''
	for c1 in contours:
		for c2 in c1:
			c2[0][1] = altezza - c2[0][1]
	return contours


def get_estremi(lines):
	'''
ritorna gli estremi delle linee come lista di punti
	'''
	points = []
	for l in lines:
		p1 = np.array([l[0],l[1]])
		p2 = np.array([l[2],l[3]])
		points.extend((p1,p2))
	return points


def uniq(lst):
	last = object()
	for item in lst:
		if item == last:
			continue
		yield item
		last = item


def sort_and_deduplicate(l):
	'''
elimina i doppioni da una lista di coppie
	'''
	return list(uniq(sorted(l, reverse=True)))


def algo(p):
	'''
riordina i punti in senso orario
	'''
	return (math.atan2(p[0] - centroid[0], p[1] - centroid[1]) + 2 * math.pi) % (2*math.pi)


def flip_vertici(vertici, altezza):
	'''
flippo le y dei vertici
	'''
	for v in vertici:
		v[1] = altezza - v[1]
	return vertici


def unisciCelle(clusters, celle, celle_poligoni, False):
	'''
i poligoni delle celle dello stesso cluster vengono uniti in un unico poligono, che è il poligono della stanza.
	'''
	stanze = []
	for l in set(clusters):
		poligoni = []
		for index,cluster in enumerate(clusters):
			if (l == cluster) and not (celle[index].out):
				poligoni.append(celle_poligoni[index])
		stanza = cascaded_union(poligoni)
		stanze.append(stanza)
	return stanze


def get_layout(metricMap, minVal, maxVal, rho, theta, thresholdHough, minLineLength, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation):
	'''
prende in input la mappa metrica, i parametri di canny, hough, mean-shift, dbscan, distanza per clustering spaziale. Genera il layout delle stanze e ritorna la lista di poligoni shapely che costituiscono le stanze, la lista di cluster corrispondenti, la lista estremi che contiene [minx,maxx,miny,maxy] e la lista di colori.
	'''
	
	img_rgb = cv2.imread(metricMap)
	ret,thresh1 = cv2.threshold(img_rgb,200,255,cv2.THRESH_BINARY)


	#------------------CANNY E HOUGH PER TROVARE MURI----------------------------------
	
	#canny
	cannyEdges = cv2.Canny(thresh1,minVal,maxVal,apertureSize = 5)
	#hough
	walls = cv2.HoughLinesP(cannyEdges,rho,theta,thresholdHough,minLineLength,maxLineGap) 

	dsg.disegna_hough(img_rgb,walls)

	lines = flip_lines(walls[0], img_rgb.shape[0]-1)

	walls = crea_muri(lines)
	
	#disegno i muri
	sg.disegna_segmenti(walls)


	#------------SETTO XMIN YMIN XMAX YMAX DI walls--------------------------------------------

	#tra tutti i punti dei muri trova l'ascissa e l'ordinata minima e massima.
	estremi = sg.trova_estremi(walls)
	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]
	offset = 20
	xmin -= offset
	xmax += offset
	ymin -= offset
	ymax += offset


	#---------------CONTORNO ESTERNO-------------------------------------------------------

	#creo il contorno esterno facendo prima canny sulla mappa metrica.
	cannyEdges = cv2.Canny(img_rgb,minVal,maxVal,apertureSize = 5)
	t=1 #threshold di hough
	m=20 #maxLineGap di hough
	hough_contorni, contours = im.trova_contorno(t,m,cannyEdges, metricMap)

	#dsg.disegna_hough(cannyEdges, hough_contorni)

	contours = flip_contorni(contours, img_rgb.shape[0]-1)

	#disegno contorno esterno
	vertici = []
	for c1 in contours:
		for c2 in c1:
			vertici.append([float(c2[0][0]),float(c2[0][1])])
	dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax)


	#-------------------MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------------------
	
	#creo i cluster centers tramite mean shift
	cluster_centers = ms.mean_shift(h, minOffset, walls)


	#ci sono dei cluster angolari che sono causati da pochi e piccoli line_segments, che sono solamente rumore. Questi cluster li elimino dalla lista cluster_centers ed elimino anche i rispettivi segmenti dalla walls.
	num_min = 3
	lunghezza_min = 3
	indici = ms.indici_da_eliminare(num_min, lunghezza_min, cluster_centers, walls)


	#ora che ho gli indici di clusters angolari e di muri da eliminare, elimino da walls e cluster_centers, partendo dagli indici piu alti
	for i in sorted(indici, reverse=True):
		del walls[i]
		del cluster_centers[i]


	#ci son dei cluster che si somigliano ma non combaciano per una differenza infinitesima, e non ho trovato parametri del mean shift che rendano il clustering piu' accurato di cosi', quindi faccio una media normalissima, tanto la differenza e' insignificante.
	unito = ms.unisci_cluster_simili(cluster_centers)
	while(unito):
		unito = ms.unisci_cluster_simili(cluster_centers)


	#assegno i cluster ai muri di walls
	walls = sg.assegna_cluster_angolare(walls, cluster_centers)


	#creo lista di cluster_angolari
	cluster_angolari = []
	for muro in walls:
		cluster_angolari.append(muro.cluster_angolare)


	#---------------CLUSTER SPAZIALI--------------------------------------------------------------------

	#setto i cluster spaziali a tutti i muri di walls
	walls = sg.spatialClustering(minLateralSeparation, walls)

	#disegno i cluster angolari
	#sg.disegna_cluster_angolari(cluster_centers, walls, cluster_angolari)

	#creo lista di cluster spaziali
	cluster_spaziali = []
	for muro in walls:
		cluster_spaziali.append(muro.cluster_spaziale)

	#disegno cluster spaziali
	sg.disegna_cluster_spaziali(cluster_spaziali, walls)


	#-------------------CREO EXTENDED_LINES---------------------------------------------------------

	extended_lines = rt.crea_extended_lines(cluster_spaziali, walls, xmin, ymin)


	#extended_lines hanno punto, cluster_angolare e cluster_spaziale, per disegnarle pero' mi servono 2 punti. Creo lista di segmenti
	extended_segments = ext.crea_extended_segments(xmin,xmax,ymin,ymax, extended_lines)


	#disegno le extended_lines in rosso e la mappa in nero
	ext.disegna_extended_segments(extended_segments, walls)


	#-------------CREO GLI EDGES TRAMITE INTERSEZIONI TRA EXTENDED_LINES-------------------------------

	edges = sg.crea_edges(extended_segments)

	#sg.disegna_segmenti(edges)


	#----------------------SETTO PESI DEGLI EDGES------------------------------------------------------

	edges = sg.setPeso(edges, walls)

	#sg.disegna_pesanti(edges, peso_min)


	#----------------CREO LE CELLE DAGLI EDGES----------------------------------------------------------

	print("creando le celle")

	celle = fc.crea_celle(edges)

	print("celle create")

	#fc.disegna_celle(celle)


	#----------------CLASSIFICO CELLE----------------------------------------------

	#creo poligono del contorno
	contorno = Polygon(vertici)
	
	celle_poligoni = []
	indici = []
	celle_out = []
	celle_parziali = []
	for index,f in enumerate(celle):
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		#ottengo i vertici della cella senza ripetizioni
		punti = sort_and_deduplicate(punti)
		#ora li ordino in senso orario
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		global centroid
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		#dopo averli ordinati in senso orario, creo il poligono della cella.
		cella = Polygon(punti)
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella è fuori.
		if cella.intersects(contorno)==False:
			#indici.append(index)
			f.set_out(True)
			f.set_parziale(False)
			celle_out.append(f)
		#se il poligono della cella interseca il contorno esterno della mappa
		if (cella.intersects(contorno)):
			#se l'intersezione è più grande di una soglia la cella è interna
			if(cella.intersection(contorno).area >= cella.area/2):
				f.set_out(False)
			#altrimenti è esterna
			else:
				f.set_out(True)
				f.set_parziale(False)
				celle_out.append(f)
	'''
	#le celle che non sono state messe come out, ma che sono adiacenti al bordo dell'immagine (hanno celle adiacenti < len(bordi)) sono per forza parziali
	a=0
	for f in celle:
		for f2 in celle:
			if (f!=f2) and (fc.adiacenti(f,f2)):
				a += 1
		if (a<len(f.bordi)):
			#print("ciao")
			if not (f.out):
				f.set_out(True)
				f.set_parziale(True)
				celle_parziali.append(f)
		a = 0

	#le celle adiacenti ad una cella out tramite un edge che pesa poco, sono parziali.	
	a = 1
	while(a!=0):
		a = 0
		for f in celle:
			for f2 in celle:
				if (f!=f2) and (f.out==False) and (f2.out==True) and (fc.adiacenti(f,f2)):
					if(fc.edge_comune(f,f2)[0].weight < 0.2):
						f.set_out(True)
						f.set_parziale(True)
						celle_parziali.append(f)
						a = 1
	'''
	
	#tolgo dalle celle out le parziali
	celle_out = list(set(celle_out)-set(celle_parziali))
	#tolgo dalle celle quelle out e parziali
	celle = list(set(celle)-set(celle_out))
	celle = list(set(celle)-set(celle_parziali))


	#--------------------------POLIGONI CELLE-------------------------------------------------

	#adesso creo i poligoni delle celle (celle = celle interne) e delle celle esterne e parziali 

	#poligoni celle interne
	celle_poligoni = []
	for f in celle:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		celle_poligoni.append(cella)	

	#poligoni celle esterne
	out_poligoni = []
	for f in celle_out:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		out_poligoni.append(cella)

	#poligoni celle parziali
	parz_poligoni = []
	for f in celle_parziali:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		parz_poligoni.append(cella)



	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L---------------------------------------

	sigma = 0.1
	val = 0
	matrice_l = mtx.crea_matrice_l(celle, sigma, val)

	matrice_d = mtx.crea_matrice_d(matrice_l)

	matrice_d_inv = matrice_d.getI()

	matrice_m = matrice_d_inv.dot(matrice_l)
	matrice_m = mtx.simmetrizza(matrice_m)

	X = 1-matrice_m


	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-----------------------------------------

	clustersCelle = []
	clustersCelle = clustering_dbscan_celle(eps, minPts, X)


	colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours)
	'''
	#plotto le celle esterne
	for f_poly in out_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#ffffff',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("out"),fontsize=8)

	#plotto le celle parziali
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("parz"),fontsize=8)
	'''


	#------------------POLIGONI STANZE-------------------------------------------------------------------

	#creo i poligoni delle stanze (unione dei poligoni delle celle con stesso cluster).
	stanze = []
	stanze = unisciCelle(clustersCelle, celle, celle_poligoni, False)

	#disegno layout stanze.
	dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax)
	plt.show()

	return (stanze, clustersCelle, estremi, colori)







def get_layout_parziale(metricMap, minVal, maxVal, rho, theta, thresholdHough, minLineLength, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation):
	'''
prende in input la mappa metrica, i parametri di canny, hough, mean-shift, dbscan, distanza per clustering spaziale. Genera il layout delle stanze e ritorna la lista di poligoni shapely che costituiscono le stanze, la lista di cluster corrispondenti, la lista estremi che contiene [minx,maxx,miny,maxy] e la lista di colori.
	'''
	
	img_rgb = cv2.imread(metricMap)
	ret,thresh1 = cv2.threshold(img_rgb,127,255,cv2.THRESH_BINARY)


	#------------------CANNY E HOUGH PER TROVARE MURI----------------------------------
	
	#canny
	cannyEdges = cv2.Canny(thresh1,minVal,maxVal,apertureSize = 5)
	#hough
	walls = cv2.HoughLinesP(cannyEdges,rho,theta,thresholdHough,minLineLength,maxLineGap) 

	dsg.disegna_hough(img_rgb,walls)

	lines = flip_lines(walls[0], img_rgb.shape[0]-1)

	walls = crea_muri(lines)
	
	#disegno i muri
	sg.disegna_segmenti(walls)


	#------------SETTO XMIN YMIN XMAX YMAX DI walls--------------------------------------------

	#tra tutti i punti dei muri trova l'ascissa e l'ordinata minima e massima.
	estremi = sg.trova_estremi(walls)
	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]
	offset = 20
	xmin -= offset
	xmax += offset
	ymin -= offset
	ymax += offset


	#---------------CONTORNO ESTERNO-------------------------------------------------------

	#creo il contorno esterno facendo prima canny sulla mappa metrica.
	cannyEdges = cv2.Canny(img_rgb,minVal,maxVal,apertureSize = 5)
	t=1 #threshold di hough
	m=21 #maxLineGap di hough
	hough_contorni, contours = im.trova_contorno(t,m,cannyEdges, metricMap)

	#dsg.disegna_hough(cannyEdges, hough_contorni)

	contours = flip_contorni(contours, img_rgb.shape[0]-1)

	#disegno contorno esterno
	vertici = []
	for c1 in contours:
		for c2 in c1:
			vertici.append([float(c2[0][0]),float(c2[0][1])])
	dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax)


	#-------------------MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------------------
	
	#creo i cluster centers tramite mean shift
	cluster_centers = ms.mean_shift(h, minOffset, walls)


	#ci sono dei cluster angolari che sono causati da pochi e piccoli line_segments, che sono solamente rumore. Questi cluster li elimino dalla lista cluster_centers ed elimino anche i rispettivi segmenti dalla walls.
	num_min = 3
	lunghezza_min = 3
	indici = ms.indici_da_eliminare(num_min, lunghezza_min, cluster_centers, walls)


	#ora che ho gli indici di clusters angolari e di muri da eliminare, elimino da walls e cluster_centers, partendo dagli indici piu alti
	for i in sorted(indici, reverse=True):
		del walls[i]
		del cluster_centers[i]


	#ci son dei cluster che si somigliano ma non combaciano per una differenza infinitesima, e non ho trovato parametri del mean shift che rendano il clustering piu' accurato di cosi', quindi faccio una media normalissima, tanto la differenza e' insignificante.
	unito = ms.unisci_cluster_simili(cluster_centers)
	while(unito):
		unito = ms.unisci_cluster_simili(cluster_centers)


	#assegno i cluster ai muri di walls
	walls = sg.assegna_cluster_angolare(walls, cluster_centers)


	#creo lista di cluster_angolari
	cluster_angolari = []
	for muro in walls:
		cluster_angolari.append(muro.cluster_angolare)


	#---------------CLUSTER SPAZIALI--------------------------------------------------------------------

	#setto i cluster spaziali a tutti i muri di walls
	walls = sg.spatialClustering(minLateralSeparation, walls)

	#disegno i cluster angolari
	#sg.disegna_cluster_angolari(cluster_centers, walls, cluster_angolari)

	#creo lista di cluster spaziali
	cluster_spaziali = []
	for muro in walls:
		cluster_spaziali.append(muro.cluster_spaziale)

	#disegno cluster spaziali
	sg.disegna_cluster_spaziali(cluster_spaziali, walls)


	#-------------------CREO EXTENDED_LINES---------------------------------------------------------

	extended_lines = rt.crea_extended_lines(cluster_spaziali, walls, xmin, ymin)


	#extended_lines hanno punto, cluster_angolare e cluster_spaziale, per disegnarle pero' mi servono 2 punti. Creo lista di segmenti
	extended_segments = ext.crea_extended_segments(xmin,xmax,ymin,ymax, extended_lines)


	#disegno le extended_lines in rosso e la mappa in nero
	ext.disegna_extended_segments(extended_segments, walls)


	#-------------CREO GLI EDGES TRAMITE INTERSEZIONI TRA EXTENDED_LINES-------------------------------

	edges = sg.crea_edges(extended_segments)

	#sg.disegna_segmenti(edges)


	#----------------------SETTO PESI DEGLI EDGES------------------------------------------------------

	edges = sg.setPeso(edges, walls)

	#sg.disegna_pesanti(edges, peso_min)


	#----------------CREO LE CELLE DAGLI EDGES----------------------------------------------------------

	print("creando le celle")

	celle = fc.crea_celle(edges)

	print("celle create")

	#fc.disegna_celle(celle)


	#----------------CLASSIFICO CELLE----------------------------------------------

	#creo poligono del contorno
	contorno = Polygon(vertici)
	
	celle_poligoni = []
	indici = []
	celle_out = []
	celle_parziali = []
	for index,f in enumerate(celle):
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		#ottengo i vertici della cella senza ripetizioni
		punti = sort_and_deduplicate(punti)
		#ora li ordino in senso orario
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		global centroid
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		#dopo averli ordinati in senso orario, creo il poligono della cella.
		cella = Polygon(punti)
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella è fuori.
		if cella.intersects(contorno)==False:
			#indici.append(index)
			f.set_out(True)
			f.set_parziale(False)
			celle_out.append(f)
		#se il poligono della cella interseca il contorno esterno della mappa
		if (cella.intersects(contorno)):
			#se l'intersezione è più grande di una soglia la cella è interna
			if(cella.intersection(contorno).area >= cella.area/2):
				f.set_out(False)
			#altrimenti è esterna
			else:
				f.set_out(True)
				f.set_parziale(False)
				celle_out.append(f)
	
	#le celle che non sono state messe come out, ma che sono adiacenti al bordo dell'immagine (hanno celle adiacenti < len(bordi)) sono per forza parziali
	a=0
	for f in celle:
		for f2 in celle:
			if (f!=f2) and (fc.adiacenti(f,f2)):
				a += 1
		if (a<len(f.bordi)):
			#print("ciao")
			if not (f.out):
				f.set_out(True)
				f.set_parziale(True)
				celle_parziali.append(f)
		a = 0

	#le celle adiacenti ad una cella out tramite un edge che pesa poco, sono parziali.	
	a = 1
	while(a!=0):
		a = 0
		for f in celle:
			for f2 in celle:
				if (f!=f2) and (f.out==False) and (f2.out==True) and (fc.adiacenti(f,f2)):
					if(fc.edge_comune(f,f2)[0].weight < 0.1):
						f.set_out(True)
						f.set_parziale(True)
						celle_parziali.append(f)
						a = 1
	
	
	#tolgo dalle celle out le parziali
	celle_out = list(set(celle_out)-set(celle_parziali))
	#tolgo dalle celle quelle out e parziali
	celle = list(set(celle)-set(celle_out))
	celle = list(set(celle)-set(celle_parziali))


	#--------------------------POLIGONI CELLE-------------------------------------------------

	#adesso creo i poligoni delle celle (celle = celle interne) e delle celle esterne e parziali 

	#poligoni celle interne
	celle_poligoni = []
	for f in celle:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		celle_poligoni.append(cella)	

	#poligoni celle esterne
	out_poligoni = []
	for f in celle_out:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		out_poligoni.append(cella)

	#poligoni celle parziali
	parz_poligoni = []
	for f in celle_parziali:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		parz_poligoni.append(cella)



	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L---------------------------------------

	sigma = 0.1
	val = 0
	matrice_l = mtx.crea_matrice_l(celle, sigma, val)

	matrice_d = mtx.crea_matrice_d(matrice_l)

	matrice_d_inv = matrice_d.getI()

	matrice_m = matrice_d_inv.dot(matrice_l)
	matrice_m = mtx.simmetrizza(matrice_m)

	X = 1-matrice_m


	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-----------------------------------------

	clustersCelle = []
	clustersCelle = clustering_dbscan_celle(eps, minPts, X)


	colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours)
	'''
	#plotto le celle esterne
	for f_poly in out_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#ffffff',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("out"),fontsize=8)
	'''
	#plotto le celle parziali
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("parz"),fontsize=8)
	plt.show()


	#------------------POLIGONI STANZE-------------------------------------------------------------------

	#creo i poligoni delle stanze (unione dei poligoni delle celle con stesso cluster).
	stanze = []
	stanze = unisciCelle(clustersCelle, celle, celle_poligoni, False)

	#disegno layout stanze.
	fig, ax = dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax)
	#plotto le celle parziali
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
	plt.show()

	return (stanze, clustersCelle, estremi, colori)









def get_layout_planimetria(metricMap, mappa_pulita, minVal, maxVal, rho, theta, thresholdHough, minLineLength, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation):
	'''
prende in input la mappa metrica, i parametri di canny, hough, mean-shift, dbscan, distanza per clustering spaziale. Genera il layout delle stanze e ritorna la lista di poligoni shapely che costituiscono le stanze, la lista di cluster corrispondenti, la lista estremi che contiene [minx,maxx,miny,maxy] e la lista di colori.
	'''

	img_rgb = mappa_pulita
	ret,thresh1 = cv2.threshold(img_rgb,127,255,cv2.THRESH_BINARY)


	#------------------CANNY E HOUGH PER TROVARE MURI----------------------------------
	
	#canny
	cannyEdges = cv2.Canny(thresh1,minVal,maxVal,apertureSize = 5)
	#hough
	walls = cv2.HoughLinesP(cannyEdges,rho,theta,thresholdHough,minLineLength,maxLineGap) 

	dsg.disegna_hough(img_rgb,walls)

	lines = flip_lines(walls[0], img_rgb.shape[0]-1)

	walls = crea_muri(lines)
	
	#disegno i muri
	sg.disegna_segmenti(walls)


	#------------SETTO XMIN YMIN XMAX YMAX DI walls--------------------------------------------

	#tra tutti i punti dei muri trova l'ascissa e l'ordinata minima e massima.
	estremi = sg.trova_estremi(walls)
	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]
	offset = 20
	xmin -= offset
	xmax += offset
	ymin -= offset
	ymax += offset


	#---------------CONTORNO ESTERNO-------------------------------------------------------

	#creo il contorno esterno facendo prima canny sulla mappa metrica.
	cannyEdges = cv2.Canny(cv2.imread(metricMap),minVal,maxVal,apertureSize = 5)
	t=1 #threshold di hough
	m=20 #maxLineGap di hough
	hough_contorni, contours = im.trova_contorno(t,m,cannyEdges, metricMap)

	#dsg.disegna_hough(cannyEdges, hough_contorni)

	contours = flip_contorni(contours, img_rgb.shape[0]-1)

	#disegno contorno esterno
	vertici = []
	for c1 in contours:
		for c2 in c1:
			vertici.append([float(c2[0][0]),float(c2[0][1])])
	dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax)


	#-------------------MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------------------

	#creo i cluster centers tramite mean shift
	cluster_centers = ms.mean_shift(h, minOffset, walls)


	#ci sono dei cluster angolari che sono causati da pochi e piccoli line_segments, che sono solamente rumore. Questi cluster li elimino dalla lista cluster_centers ed elimino anche i rispettivi segmenti dalla lista_muri.
	num_min = 3
	lunghezza_min = 3
	indici = ms.indici_da_eliminare(num_min, lunghezza_min, cluster_centers, walls)


	#ora che ho gli indici di clusters angolari e di muri da eliminare, elimino da lista_muri e cluster_centers, partendo dagli indici piu alti
	for i in sorted(indici, reverse=True):
		del walls[i]
		del cluster_centers[i]

	dsg.disegna(walls)


	#ci son dei cluster che si somigliano ma non combaciano per una differenza infinitesima, e non ho trovato parametri del mean shift che rendano il clustering piu' accurato di cosi', quindi faccio una media normalissima, tanto la differenza e' insignificante.
	unito = ms.unisci_cluster_simili(cluster_centers)
	while(unito):
		unito = ms.unisci_cluster_simili(cluster_centers)


	#assegno i cluster ai muri di lista_muri
	walls = sg.assegna_cluster_angolare(walls, cluster_centers)

	#creo lista di cluster_angolari
	cluster_angolari = []
	for muro in walls:
		cluster_angolari.append(muro.cluster_angolare)


	#---------------CLUSTER SPAZIALI--------------------------------------------------------------------

	#setto i cluster spaziali a tutti i muri di lista_muri
	walls = sg.spatialClustering(minLateralSeparation, walls)

	sg.disegna_cluster_angolari(cluster_centers, walls, cluster_angolari)

	#creo lista di cluster spaziali
	cluster_spaziali = []
	for muro in walls:
		cluster_spaziali.append(muro.cluster_spaziale)

	sg.disegna_cluster_spaziali(cluster_spaziali, walls)

	#trovo gli estremi del contorno esterno. Mi servono per settare xmin ymin xmax ymax.
	[x, y, w, h] = cv2.boundingRect(contours[0])

	#-------------------CREO EXTENDED_LINES---------------------------------------------------------

	extended_lines = rt.crea_extended_lines(cluster_spaziali, walls, xmin, ymin)


	#extended_lines hanno punto, cluster_angolare e cluster_spaziale, per disegnarle pero' mi servono 2 punti. Creo lista di segmenti
	extended_segments = ext.crea_extended_segments(xmin,xmax,ymin,ymax, extended_lines)


	#disegno le extended_lines in rosso e la mappa in nero
	ext.disegna_extended_segments(extended_segments, walls)


	#-------------CREO GLI EDGES TRAMITE INTERSEZIONI TRA EXTENDED_LINES-------------------------------

	edges = sg.crea_edges(extended_segments)

	#sg.disegna_segmenti(edges)


	#----------------------SETTO PESI DEGLI EDGES------------------------------------------------------

	edges = sg.setPeso(edges, walls)

	#sg.disegna_pesanti(edges, peso_min)


	#----------------CREO LE CELLE DAGLI EDGES----------------------------------------------------------

	print("creando le celle")

	celle = fc.crea_celle(edges)

	print("celle create")

	#fc.disegna_celle(celle)


	#----------------CLASSIFICO CELLE----------------------------------------------

	#creo poligono del contorno
	contorno = Polygon(vertici)
	
	celle_poligoni = []
	indici = []
	celle_out = []
	celle_parziali = []
	for index,f in enumerate(celle):
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		#ottengo i vertici della cella senza ripetizioni
		punti = sort_and_deduplicate(punti)
		#ora li ordino in senso orario
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		global centroid
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		#dopo averli ordinati in senso orario, creo il poligono della cella.
		cella = Polygon(punti)
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella è fuori.
		if cella.intersects(contorno)==False:
			#indici.append(index)
			f.set_out(True)
			f.set_parziale(False)
			celle_out.append(f)
		#se il poligono della cella interseca il contorno esterno della mappa
		if (cella.intersects(contorno)):
			#se l'intersezione è più grande di una soglia la cella è interna
			if(cella.intersection(contorno).area >= cella.area/2):
				f.set_out(False)
			#altrimenti è esterna
			else:
				f.set_out(True)
				f.set_parziale(False)
				celle_out.append(f)
	'''
	#le celle che non sono state messe come out, ma che sono adiacenti al bordo dell'immagine (hanno celle adiacenti < len(bordi)) sono per forza parziali
	a=0
	for f in celle:
		for f2 in celle:
			if (f!=f2) and (fc.adiacenti(f,f2)):
				a += 1
		if (a<len(f.bordi)):
			#print("ciao")
			if not (f.out):
				f.set_out(True)
				f.set_parziale(True)
				celle_parziali.append(f)
		a = 0

	#le celle adiacenti ad una cella out tramite un edge che pesa poco, sono parziali.	
	a = 1
	while(a!=0):
		a = 0
		for f in celle:
			for f2 in celle:
				if (f!=f2) and (f.out==False) and (f2.out==True) and (fc.adiacenti(f,f2)):
					if(fc.edge_comune(f,f2)[0].weight < 0.2):
						f.set_out(True)
						f.set_parziale(True)
						celle_parziali.append(f)
						a = 1
	'''
	
	#tolgo dalle celle out le parziali
	celle_out = list(set(celle_out)-set(celle_parziali))
	#tolgo dalle celle quelle out e parziali
	celle = list(set(celle)-set(celle_out))
	celle = list(set(celle)-set(celle_parziali))


	#--------------------------POLIGONI CELLE-------------------------------------------------

	#adesso creo i poligoni delle celle (celle = celle interne) e delle celle esterne e parziali 

	#poligoni celle interne
	celle_poligoni = []
	for f in celle:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		celle_poligoni.append(cella)	

	#poligoni celle esterne
	out_poligoni = []
	for f in celle_out:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		out_poligoni.append(cella)

	#poligoni celle parziali
	parz_poligoni = []
	for f in celle_parziali:
		punti = []
		for b in f.bordi:
			punti.append([float(b.x1),float(b.y1)])
			punti.append([float(b.x2),float(b.y2)])
		punti = sort_and_deduplicate(punti)
		x = [p[0] for p in punti]
		y = [p[1] for p in punti]
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		parz_poligoni.append(cella)



	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L---------------------------------------

	sigma = 0.1
	val = 0
	matrice_l = mtx.crea_matrice_l(celle, sigma, val)

	matrice_d = mtx.crea_matrice_d(matrice_l)

	matrice_d_inv = matrice_d.getI()

	matrice_m = matrice_d_inv.dot(matrice_l)
	matrice_m = mtx.simmetrizza(matrice_m)

	X = 1-matrice_m


	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-----------------------------------------

	clustersCelle = []
	clustersCelle = clustering_dbscan_celle(eps, minPts, X)


	colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours)
	'''
	#plotto le celle esterne
	for f_poly in out_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#ffffff',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("out"),fontsize=8)

	#plotto le celle parziali
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("parz"),fontsize=8)
	'''


	#------------------POLIGONI STANZE-------------------------------------------------------------------

	#creo i poligoni delle stanze (unione dei poligoni delle celle con stesso cluster).
	stanze = []
	stanze = unisciCelle(clustersCelle, celle, celle_poligoni, False)

	#disegno layout stanze.
	dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax)
	plt.show()

	return (stanze, clustersCelle, estremi, colori)
