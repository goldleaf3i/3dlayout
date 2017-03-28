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
import template as tmpl

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


def crea_poligoni(vertici):
	'''
dati i vertici dei templates trovati, ogni 4 vertici crea un poligono
	'''
	i = 0
	poligoni = []
	while(i < len(vertici)):
		poligono = Polygon(vertici[i:i+4])	
		poligoni.append(poligono)
		i += 4
	return poligoni


def preprocessing(file_name):

	#----------------TROVO PORTE--------------------------------------------------------------

	img_rgb = cv2.imread(file_name)
	templates_porte = tmpl.template_porte()
	templates_muri = tmpl.template_muri()

	#setto i template delle porte da cercare nell'immagine, e i template dei muri con cui sostituire quelli delle porte, per migliorare il clustering (se al posto della porta metto un muro è più facile che le 2 stanze vengano viste come diverse).

	vertici_porte = tmpl.trova_templates(templates_porte, img_rgb, 20, 30, 0.7)
	img_con_porte = tmpl.sostituisci_porte(templates_porte, templates_muri, img_rgb, 20, 30, 0.7)
	#per ogni contorno delle porte creo un poligono
	porte = crea_poligoni(vertici_porte)


	#-------------TROVO ESTINTORI-----------------------------------------------------------

	templates_estintori = tmpl.template_estintori()
	vertici_estintori = tmpl.trova_templates(templates_estintori, img_rgb, 11, 14, 0.9)
	#per ogni contorno delle porte creo un poligono
	estintori = crea_poligoni(vertici_estintori)


	#-------------------TROVO IDRANTI-----------------------------------------------------------

	templates_idranti = tmpl.template_idranti()
	vertici_idranti = tmpl.trova_templates(templates_idranti, img_rgb, 17, 20, 0.8)
	#per ogni contorno delle porte creo un poligono
	idranti = crea_poligoni(vertici_idranti)



	#--------------ELIMINO PORTE E ESTINTORI E IDRANTI---------------------------------------
	#trasformo in bianchi i pixel delle regioni in cui ho trovato i templates

	#elimino gli estintori
	img = tmpl.elimina_templates(img_con_porte,estintori)

	#elimino gli idranti
	img = tmpl.elimina_templates(img,idranti)

	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.show()

	#img ha le porte chiuse perchè son state sostituite dai muri, e la uso per fare segmentazione e trovare le stanze. Invece per medial axis mi serve un'immagine senza le porte (cioè con le porte aperte).
	img_senza_porte = tmpl.elimina_templates(img,porte)
	plt.title('mappa con porte aperte')
	plt.imshow(cv2.cvtColor(img_senza_porte,cv2.COLOR_BGR2RGB))
	plt.show()


	
	'''
	#--------------------ELIMINO TESTO-----------------

	b,g,r = cv2.split(img)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
	#_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # threshold
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	dilated = cv2.dilate(thresh, kernel, iterations = 2)  # dilate
	contours_text, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


	#elimino il testo
	#img = image.copy()
	#c_copy = contours [:]
	#contours_flipped = flip_contorni(c_copy, image.shape[0]-1)
	# for each contour found, draw a rectangle around it on original image
	for contour in contours_text:
		vertici = []
		for c2 in contour:
			vertici.append([float(c2[0][0]),float(c2[0][1])])
		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)
		# discard areas that are too large
		if (h > 40) and (w > 40):
			continue
		# discard areas that are too small
		#if h < 10 or w < 10:
		#	continue
		# draw rectangle around contour on original image
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
		contorno = Polygon(vertici)
		for i in xrange(x-1,x+w+1):
			for j in xrange(y-1,y+h+1):
				p = Point(i,j)
				if p.within(contorno):
					r[j,i]=255
					g[j,i]=255
					b[j,i]=255
	

	img = cv2.merge((b,g,r))
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.show()
	'''


	#------------------ELIMINO I COLORI----------------------------------------------

	#trasformo grigio in nero, poiche' i muri sono grigi. Se non lo faccio, quando trasformo i colori in bianco (frecce verdi, rimasugli di estintori idranti ecc) diventerebbero bianchi anche i muri. Per individuare il colore grigio, per ogni pixel vedo se la differenza tra r g e b è più bassa di una certa soglia. Se è così, trasformo quel pixel in nero.
	soglia=10
	thresh = im.elimino_colori(soglia, img)
	plt.imshow(thresh,cmap='Greys')
	plt.show()


	#-------------------EROSION---------------------------------------------------------

	#assottiglio i muri per facilitare il clustering.

	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(thresh,kernel,iterations = 1)
	#plt.imshow(erosion,cmap='Greys')
	#plt.show()

	return idranti, estintori, porte, erosion, img_senza_porte
