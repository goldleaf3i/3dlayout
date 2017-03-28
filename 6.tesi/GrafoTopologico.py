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
import medial as mdl
import template as tmpl

from itertools import cycle

import cv2
import matplotlib.image as mpimg

from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull
from shapely.ops import cascaded_union
import xml.etree.ElementTree as ET



def get_grafo(metricMap, stanze, estremi, colori):
	'''
Prende in input la mappa metrica, la lista di stanze, i cluster delle celle, estremi e colori. Trova le connessioni tra le stanze con medial axis. Restituisce il grafo topologico, il dizionario di posizioni dei nodi, le coppie di stanze collegate e le coordinate dei collegamenti. 
	'''
	
	#----------------------------------MEDIAL AXIS---------------------------------------

	im_gray = cv2.imread(metricMap, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	(thresh, im_bw) = cv2.threshold(im_gray, 240, 255, cv2.THRESH_BINARY_INV)
	medialAxis = mdl.medial_points(im_bw)

	stanze_collegate = []
	stanze_collegate, porte_colleganti = mdl.trovaCollegamenti(stanze,stanze_collegate,medialAxis)
	#per ogni punto dello skeleton, vado a vedere se è in prossimità dei bordi di qualche stanza. Se le stanze vicine al punto hanno label diverse, vuol dire che in quel punto c'è una porta, o comunque un passaggio diretto.



	#----------------------------------GRAFO---------------------------------------------------------------

	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]

	#creo grafo, ogni nodo corrisponde a una label
	G=nx.Graph()
	for i,s in enumerate(stanze):
		G.add_node(i)

	#gli edge li prendo da stanze_collegate
	G.add_edges_from(stanze_collegate)
	#creo un dizionario che contiene le posizioni dei nodi. Le posizioni sono i representative point delle stanze corrispondenti. Creo la prima chiave del dizionario, poi aggiungo le restanti con un ciclo for.
	pos = {0: (stanze[0].representative_point().x,stanze[0].representative_point().y)}
	for i,s in enumerate(stanze[1:]):
		pos[i+1] = (s.representative_point().x,s.representative_point().y)

	#aggiungo l'attributo posizione ad ogni nodo
	for n, p in pos.iteritems():
		G.node[n]['pos'] = p

	#plotto le stanze
	fig = plt.figure()
	plt.title('grafo topologico')
	ax = fig.add_subplot(111)
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc=colori[index],ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)

	#plotto il grafo
	nx.draw_networkx_nodes(G,pos,node_color='w')
	#nx.draw_networkx_edges(G,pos)
	nx.draw_networkx_labels(G,pos)

	#plotto gli edges come linee da un representative point a un altro, perchè con solo drawedges non li plotta. Forse sono nascosti dai poligoni.
	for coppia in stanze_collegate:
		i1 = stanze[coppia[0]]
		i2 = stanze[coppia[1]]
		p1 = i1.representative_point()
		p2 = i2.representative_point()
		plt.plot([p1.x,p2.x],[p1.y,p2.y],color='k',ls = 'dotted', lw=0.5)

	plt.show()

	'''
	#plotto il grafo stavolta da solo. Non c'è il problema degli edges.
	nx.draw_networkx_nodes(G,pos,node_color=colori)
	nx.draw_networkx_edges(G,pos)
	nx.draw_networkx_labels(G,pos)
	plt.show()
	'''

	return G, pos, stanze_collegate, porte_colleganti




def get_grafo_planimetria(metricMap, stanze, porte, estremi, colori):
	'''
	#---------------TROVO LE FACCE COLLEGATE DA UNA PORTA--------------------------------
	if(len(metricMap.shape)==3):
		metricMap = cv2.cvtColor(metricMap,cv2.COLOR_RGB2GRAY)
	(thresh, im_bw) = cv2.threshold(metricMap, 240, 255, cv2.THRESH_BINARY_INV)
	medialAxis = mdl.medial_points(im_bw)
	stanze_collegate_medial = []
	stanze_collegate_medial, porte_colleganti_medial = mdl.trovaCollegamenti(stanze,stanze_collegate_medial,medialAxis)
	'''	
	#coppie di labels collegate da una porta.
	(stanze_collegate,porte_colleganti) = tmpl.collegate(porte,stanze)
	#print stanze_collegate
	
	#stanze_collegate = stanze_collegate + stanze_collegate_medial
	#porte_colleganti = porte_colleganti + porte_colleganti_medial


	#----------------------------------GRAFO---------------------------------------------------------------

	xmin = estremi[0]
	xmax = estremi[1]
	ymin = estremi[2]
	ymax = estremi[3]

	#creo grafo, ogni nodo corrisponde a una label
	G=nx.Graph()
	for i,s in enumerate(stanze):
		G.add_node(i)#creo grafo, ogni nodo corrisponde a una label
	G=nx.Graph()
	for i,s in enumerate(stanze):
		G.add_node(i)

	#gli edge li prendo da stanze_collegate
	G.add_edges_from(stanze_collegate)
	#creo un dizionario che contiene le posizioni dei nodi. Le posizioni sono i representative point delle stanze corrispondenti. Creo la prima chiave del dizionario, poi aggiungo le restanti con un ciclo for.
	pos = {0: (stanze[0].representative_point().x,stanze[0].representative_point().y)}
	for i,s in enumerate(stanze[1:]):
		pos[i+1] = (s.representative_point().x,s.representative_point().y)

	#aggiungo l'attributo posizione ad ogni nodo
	for n, p in pos.iteritems():
		G.node[n]['pos'] = p

	#plotto le stanze
	fig = plt.figure()
	plt.title('grafo topologico')
	ax = fig.add_subplot(111)
	for index,s in enumerate(stanze):
		f_patch = PolygonPatch(s,fc=colori[index],ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)

	#plotto il grafo
	nx.draw_networkx_nodes(G,pos,node_color='w')
	#nx.draw_networkx_edges(G,pos)
	nx.draw_networkx_labels(G,pos)

	#plotto gli edges come linee da un representative point a un altro, perchè con solo drawedges non li plotta. Forse sono nascosti dai poligoni.
	for coppia in stanze_collegate:
		i1 = stanze[coppia[0]]
		i2 = stanze[coppia[1]]
		p1 = i1.representative_point()
		p2 = i2.representative_point()
		plt.plot([p1.x,p2.x],[p1.y,p2.y],color='k',ls = 'dotted', lw=0.5)

	plt.show()

	'''
	#plotto il grafo stavolta da solo. Non c'è il problema degli edges.
	nx.draw_networkx_nodes(G,pos,node_color=colori)
	nx.draw_networkx_edges(G,pos)
	nx.draw_networkx_labels(G,pos)
	plt.show()
	'''

	return G, pos, stanze_collegate, porte_colleganti
