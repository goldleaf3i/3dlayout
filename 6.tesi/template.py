# -*- coding: utf-8 -*-

import cv2
import PIL
from PIL import Image
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt

def template_porte():
	template1 = Image.open('/home/matteo/Desktop/git/templates/door.png')
	template2 = Image.open('/home/matteo/Desktop/git/templates/door2.png')
	template3 = Image.open('/home/matteo/Desktop/git/templates/door3.png')
	template4 = Image.open('/home/matteo/Desktop/git/templates/door4.png')
	template5 = Image.open('/home/matteo/Desktop/git/templates/door_f.png')
	template6 = Image.open('/home/matteo/Desktop/git/templates/door2_f.png')
	template7 = Image.open('/home/matteo/Desktop/git/templates/door3_f.png')
	template8 = Image.open('/home/matteo/Desktop/git/templates/door4_f.png')
	templates_porte = []
	templates_porte.extend((template1,template2,template3,template4,template5,template6,template7,template8))
	return templates_porte


def template_muri():
	template9 = cv2.imread('/home/matteo/Desktop/git/templates/wall.png')
	template10 = cv2.imread('/home/matteo/Desktop/git/templates/wall2.png')
	template11 = cv2.imread('/home/matteo/Desktop/git/templates/wall3.png')
	template12 = cv2.imread('/home/matteo/Desktop/git/templates/wall4.png')
	template13 = cv2.imread('/home/matteo/Desktop/git/templates/wall_f.png')
	template14 = cv2.imread('/home/matteo/Desktop/git/templates/wall2_f.png')
	template15 = cv2.imread('/home/matteo/Desktop/git/templates/wall3_f.png')
	template16 = cv2.imread('/home/matteo/Desktop/git/templates/wall4_f.png')
	templates_muri = []
	templates_muri.extend((template9,template10,template11,template12,template13,template14,template15,template16))
	return templates_muri


def template_estintori():
	template25 = Image.open('/home/matteo/Desktop/git/templates/estintore.png')
	template26 = Image.open('/home/matteo/Desktop/git/templates/estintore2.png')
	template27 = Image.open('/home/matteo/Desktop/git/templates/estintore3.png')
	template28 = Image.open('/home/matteo/Desktop/git/templates/estintore4.png')
	templates_estintori = []
	templates_estintori.extend((template25,template26,template27,template28))
	return templates_estintori


def template_idranti():
	template29 = Image.open('/home/matteo/Desktop/git/templates/idrante.png')
	templates_idranti = []
	templates_idranti.append(template29)
	return templates_idranti


def collegate(porte,stanze):
	stanze_collegate = []
	porte_colleganti = []
	for porta in porte:
		stanze_porta = []
		for index,s in enumerate(stanze):
			if s.intersects(porta):
				if not (s in stanze_porta):
					stanze_porta.append(index)
		if (len(stanze_porta)==2) and not (stanze_porta in stanze_collegate):	
			stanze_collegate.append(stanze_porta)
			porte_colleganti.append(porta.centroid)
		#se ne ho trovate più di 2, le aggiungo a 2 a 2.
		if (len(stanze_porta)>2): #se piu di 2 mi da problemi nell'add_edges_from
			for index,a in enumerate(stanze_porta):
				for b in stanze_porta[index+1:]:
					tmp = []
					tmp.extend((a,b))
					if not (tmp in stanze_collegate):
						stanze_collegate.append(tmp)
						porte_colleganti.append(porta.centroid)
	return stanze_collegate, porte_colleganti


def collegate_vecchio(porte,stanze,labels):
	labels_collegate = []
	porte_colleganti = []
	for porta in porte:
		labels_porta = []
		for index,l in enumerate(set(labels)):
			if stanze[index].intersects(porta):
				if not (l in labels_porta):
					labels_porta.append(l)
		if (len(labels_porta)==2) and not (labels_porta in labels_collegate):	
			labels_collegate.append(labels_porta)
			porte_colleganti.append(porta.centroid)
		#se ne ho trovate più di 2, le aggiungo a 2 a 2.
		if (len(labels_porta)>2): #se piu di 2 mi da problemi nell'add_edges_from
			for index,a in enumerate(labels_porta):
				for b in labels_porta[index+1:]:
					tmp = []
					tmp.extend((a,b))
					if not (tmp in labels_collegate):
						labels_collegate.append(tmp)
						porte_colleganti.append(porta.centroid)
	return labels_collegate, porte_colleganti

def stanze_con_oggetto(oggetti,stanze,labels):
	stanze_con_oggetto = []
	for oggetto in oggetti:
		labels_oggetti = []
		for index,l in enumerate(set(labels)):
			if(stanze[index].intersects(oggetto)):
				if not (l in labels_oggetti):
					labels_oggetti.append(l)
		#se c'è più di una stanza che interseca l'estintore, prendo solo quella che la interseca maggiormente.	
		if (len(labels_oggetti)>1):
			max_intersezione = 0	
			for l in labels_oggetti:
				i = list(set(labels)).index(l)
				intersezione = stanze[i].intersection(oggetto).area
				if intersezione>max_intersezione:
					label_max = l
			labels_oggetti = []
			labels_oggetti.append(label_max)
		#se non è già stata aggiunta all'elenco di stanze con estintore, la aggiungo.
		if not (labels_oggetti in stanze_con_oggetto):
			stanze_con_oggetto.append(labels_oggetti)
	return stanze_con_oggetto


def trova_templates(templates, immagine, dim_min, dim_max, threshold):
	'''
trova i templates dentro all'immagine, eseguendo uno scaling da dim_min a dim_max. Il templates è riconosciuto se c'è una somiglianza > threshold. Ritorna i vertici dei rettangoli che delimitano i template trovati.
	''' 
	img3 = immagine.copy() #usata solo per plottarci sopra i rettangoli che evidenziano i templates trovati
	img_gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY) #usata come immagine in cui cercare i templates
	vertici = []
	for template in templates:
		for basewidth in xrange(dim_min, dim_max):
			#img = img_rgb.copy()
			wpercent = (basewidth / float(template.size[0]))
			hsize = int((float(template.size[1]) * float(wpercent)))
			temp = template.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
			temp = img_as_ubyte(temp)
			if(len(temp.shape)>2):
				temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
			w, h = temp.shape[::-1]
			res = cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED)
			loc = np.where( res >= threshold)
			for pt in zip(*loc[::-1]):
				cv2.rectangle(img3, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
				vertici.append([float(pt[0]),float(pt[1])])
				vertici.append([float(pt[0]),float(pt[1]+h)])
				vertici.append([float(pt[0]+w),float(pt[1]+h)])
				vertici.append([float(pt[0]+w),float(pt[1])])
				#print(pt[0], pt[1], pt[0]+w, pt[1]+h)
	plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
	plt.show()
	vertici = flip_vertici(vertici, immagine.shape[0]-1)
	return vertici


def sostituisci_porte(templates, templates_muri, immagine, dim_min, dim_max, threshold):
	'''
cerca i template delle porte e li sostituisce con i corrispondenti template dei muri, scalati della stessa quantità. Ritorna l'immagine così modificata.
	'''
	img3 = immagine.copy() #usata solo per plottarci sopra i rettangoli che evidenziano i templates trovati
	img_gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY) #usata come immagine in cui cercare i templates
	for index,template in enumerate(templates):
		for basewidth in xrange(dim_min, dim_max):
			#img = img_rgb.copy()
			wpercent = (basewidth / float(template.size[0]))
			hsize = int((float(template.size[1]) * float(wpercent)))
			temp = template.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
			temp = img_as_ubyte(temp)
			if(len(temp.shape)>2):
				temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
			w, h = temp.shape[::-1]
			res = cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED)
			loc = np.where( res >= threshold)
			for pt in zip(*loc[::-1]):
				resized = cv2.resize(templates_muri[index], (w,h))
				img3[pt[1]:pt[1]+resized.shape[0], pt[0]:pt[0]+resized.shape[1]] = resized
	plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
	plt.show()
	return img3


def flip_vertici(vertici_porte, altezza):
	'''
flippa le y dei vertici
	'''
	for v in vertici_porte:
		v[1] = altezza - v[1]
	return vertici_porte


def elimina_templates(immagine, templates):
	b,g,r = cv2.split(immagine)
	for t in templates:
		bounds = t.bounds #(minx,miny,maxx,maxy)
		for i in xrange(int(bounds[0]),int(bounds[2])):
			for j in xrange(int(bounds[1]),int(bounds[3])):
				r[immagine.shape[0]-j,i]=255
				g[immagine.shape[0]-j,i]=255
				b[immagine.shape[0]-j,i]=255
	img = cv2.merge((b,g,r))
	return img

