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

def flip_vertici(vertici, altezza):
	'''
flippo le y dei vertici
	'''
	for v in vertici:
		v[1] = altezza - v[1]
	return vertici


def calcola_accuracy(nome_gt,estremi,stanze, file_name):
	#stanze ground truth
	(stanze_gt) = get_stanze_gt(nome_gt, estremi)
	#corrispondenze tra gt e segmentate (backward e forward)
	(indici_corrispondenti_bwd, indici_gt_corrispondenti_fwd) = get_corrispondenze(stanze,stanze_gt,file_name)


def elimina_stanze(stanze,estremi):
	'''
vengono mostrate le stanze segmentate con all'interno la loro posizione nella lista stanze. Da terminale viene chiesto se si vuole eliminare una stanza. Se la risposta è sì (s) viene chiesto il numero della stanza da eliminare. Dopo averla eliminata, viene rimostrato il layout e viene chiesto iterativamente se si vuole eliminare un'altra stanza.
	'''	
	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc='WHITE',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin-20,xmax+20)
		ax.set_ylim(ymin-20,ymax+20)
		ax.text(s.representative_point().x,s.representative_point().y,str(index),fontsize=8)
	plt.show()

	risposta = raw_input('vuoi eliminare stanze? s/n: ')

	while str(risposta) == 's':
		try:
		    da_eliminare=int(raw_input('numero stanza da eliminare:'))
		except ValueError:
		    print "Not a number"
		del stanze[int(da_eliminare)]
		print len(stanze)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for index,s in enumerate(stanze):
			f_patch = PolygonPatch(s,fc='WHITE',ec='BLACK')
			ax.add_patch(f_patch)
			ax.set_xlim(xmin-20,xmax+20)
			ax.set_ylim(ymin-20,ymax+20)
			ax.text(s.representative_point().x,s.representative_point().y,str(index),fontsize=8)
		plt.show()
		risposta = raw_input('vuoi eliminare stanze? s/n: ')

	# plotto le nuove stanze, sono stati eliminati i buchi interni 
	colori = []
	colori.extend(('#800000','#DC143C','#FF0000','#FF7F50','#F08080','#FF4500','#FF8C00','#FFD700','#B8860B','#EEE8AA','#BDB76B','#F0E68C','#808000','#9ACD32','#7CFC00','#ADFF2F','#006400','#90EE90','#8FBC8F','#00FA9A','#20B2AA','#00FFFF','#4682B4','#1E90FF','#000080','#0000FF','#8A2BE2','#4B0082','#800080','#FF00FF','#DB7093','#FFC0CB','#F5DEB3','#8B4513','#808080'))
	dsg.disegna_stanze(stanze, colori, xmin-20, ymin-20, xmax+20, ymax+20)
	plt.show()

	
	return stanze


def get_stanze_gt(nome_gt, estremi):
	'''
Prende in input il nome dell'xml ground, gli estremi delle segmentate, e restituisce la lista di stanze ground truth (scalate e traslate in modo da rendere il layout gt sovrapponibile a quello segmentato) 
	'''

	tree = ET.parse(nome_gt)
	root = tree.getroot()

	xs = []
	ys = []
	for punto in root.findall('*//point'):
		xs.append(int(punto.get('x')))
		ys.append(int(punto.get('y')))	
	
	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]
	xmin_gt = min(xs)
	xmax_gt = max(xs)
	ymin_gt = min(ys)
	ymax_gt = max(ys)
	altezza = ymax + ymin

	stanze_gt = []
	#rendo layout mio e layout gt della stessa dimensione
	fattore_x = (xmax-xmin)/(xmax_gt-xmin_gt)
	fattore_y = (ymax-ymin)/(ymax_gt-ymin_gt)
	xmin_gt_scalato = xmin_gt*fattore_x
	ymin_gt_scalato = ymin_gt*fattore_y
	stanze_gt = []
	spaces = root.findall('.//space')
	for space in spaces:
		pol = space.find('.//bounding_polygon')
		punti = []
		bordi = pol.findall('./point')
		for p in bordi[:-1]:
			x = int(p.get('x'))
			y = int(p.get('y'))
			x_scalato = x*fattore_x
			y_scalato = y*fattore_y
			x_traslato = x_scalato+(xmin-xmin_gt_scalato)
			y_traslato = y_scalato+(ymin-ymin_gt_scalato)
			#da eseguire solo se il layout gt è capovolto rispetto a quello segmentato (con gli xml delle scuole non succede, con quelli del survey si)
			y_traslato = altezza - y_traslato
			punti.append((x_traslato,y_traslato))
		x = int(bordi[0].get('x'))
		y = int(bordi[0].get('y'))
		x_scalato = x*fattore_x
		y_scalato = y*fattore_y
		x_traslato = x_scalato+(xmin-xmin_gt_scalato)
		y_traslato = y_scalato+(ymin-ymin_gt_scalato)
		#come sopra
		y_traslato = altezza - y_traslato
		punti.append((x_traslato,y_traslato))
		poligono = Polygon(punti)
		if not poligono.is_valid:
			poligono = poligono.buffer(0)
		stanze_gt.append(poligono)

	#disegno stanze ground truth
	fig = plt.figure()
	plt.title('stanze_gt')
	ax = fig.add_subplot(111)
	for index,s in enumerate(stanze_gt):
		f_patch = PolygonPatch(s,fc='WHITE',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin-20,xmax+20)
		ax.set_ylim(ymin-20,ymax+20)

	plt.show()

	return stanze_gt



def get_corrispondenze(stanze, stanze_gt, file_name):
	'''
Riceve stanze segmentate, gt, nome della mappa e scrive sul file accuracy.txt accuracy bc ed accuracy fc medie (+- var). 
	''' 
	#calcolo recall, salvando gli indici delle stanze segmentate corrispondenti alle gt.
	somma_recall = 0
	recalls = []
	indici_corrispondenti_bwd = []
	for s in stanze_gt:
		max_overlap_recall = 0
		for index,s2 in enumerate(stanze):
			if (s2.intersects(s)):
				if(s2.intersection(s).area >= max_overlap_recall):
					max_overlap_recall = s2.intersection(s).area
					indice_stanza_corrispondente = index
		recall_stanza = max_overlap_recall/s.area	
		recalls.append(recall_stanza)
		indici_corrispondenti_bwd.append(indice_stanza_corrispondente)
	
	accuracy_bc_medio = np.mean(recalls)
	accuracy_bc_var = np.std(recalls)
	#print recall
	

	#calcolo precision, salvando gli indici delle stanze ground truth corrispondenti a quelle segmentate
	indici_gt_corrispondenti_fwd = [] 
	somma_precision = 0
	precisions = []
	for s in stanze:
		max_overlap_precision = 0
		for index,s2 in enumerate(stanze_gt):
			if (s2.intersects(s)):
				if(s2.intersection(s).area >= max_overlap_precision):
					max_overlap_precision = s2.intersection(s).area
					indice_stanza_gt_corrispondente = index
		precision_stanza = max_overlap_precision/s.area	
		precisions.append(precision_stanza)
		indici_gt_corrispondenti_fwd.append(indice_stanza_gt_corrispondente)
	
	accuracy_fc_medio = np.mean(precisions)
	accuracy_fc_var = np.var(precisions)
	#print precision
	print('accuracy_bc = '+str(accuracy_bc_medio)+' +- '+str(accuracy_bc_var))
	print('accuracy_fc = '+str(accuracy_fc_medio)+' +- '+str(accuracy_fc_var))

	f = open('accuracy.txt','a')
	f.write(file_name+'\n')
	f.write('accuracy_bc = '+str(accuracy_bc_medio)+' +- '+str(accuracy_bc_var)+'\n')
	f.write('accuracy_fc = '+str(accuracy_fc_medio)+' +- '+str(accuracy_fc_var)+'\n\n')
	f.close()

	return indici_corrispondenti_bwd, indici_gt_corrispondenti_fwd







