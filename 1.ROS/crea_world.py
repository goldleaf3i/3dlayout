# -*- coding: utf-8 -*-

#script che prende come argomento il nome della mappa e la risoluzione (fattore per cui vanno moltiplicati i pixel dell'immagine png della mappa) e crea il rispettivo file world, copiando quello di default ed aggiungendoci la risoluzione, il nome, la bitmap, la size e la pose del floorplan e la pose del robot (che e' il centroid della entrance trasformato in coordinate di stage) 

import sys
from shutil import copyfile
from PIL import Image
import xml.etree.ElementTree as ET

path_mappa = "/home/matteo/Desktop/git/Valleceppi_P1_updated.png" #inserire path alla mappa png che si vuole aprire in Stage
risoluzione = 0.1 #inserire risoluzione (io ho usato 0.1)
path_xml = '/home/matteo/Desktop/git/Dataset sistemati/Valleceppi_P1_updated.xml' #inserire path dell'xml da cui è stato ottenuta la mappa png (usando lo script mappa_da_xml.py)
path_world = "/home/matteo/Desktop/git/world_per_stage/Valleceppi_P1_updated.world" #inserire path file world che si vuole creare (deve finire con .world)
 
ascisse = []
ordinate = []
trovato = 0

#salvo larghezza e altezza dell'immagine della mappa
larghezza = float(Image.open(open(path_mappa)).size[0])
altezza = float(Image.open(open(path_mappa)).size[1])

tree = ET.parse(path_xml)
root = tree.getroot()
#mi servono per creare un albero corrispondente all'xml, la radice e' root

#raccolgo nella lista stanze tutte le stanze, ovvero i tag space
stanze = root.findall("./floor/spaces/space")

#prendo le ascisse e le ordinate di tutti i punti presenti nell'xml, estraggo le x e y minime e massime 
punti = root.findall(".//point")
for punto in punti:
	ascisse.append(float(punto.get('x')))
	ordinate.append(float(punto.get('y')))
x_min_xml = min(ascisse)
x_max_xml = max(ascisse)
y_min_xml = min(ordinate)
y_max_xml = max(ordinate)

#estraggo le coordinate del centroid dell'entrance (la prima che trovo va bene)
for stanza in stanze:
	label = stanza.find("./labels/label").text
	if label == 'ENTRANCE':
		trovato = 1
		x_entrance = float(stanza.find("./centroid/point").get('x'))
		y_entrance = float(stanza.find("./centroid/point").get('y'))
		break
#se ho trovato almeno una entrance ho messo trovato=1, se e' ancora a 0 allora cerco stairs
if trovato==0:
	for stanza in stanze:
		label = stanza.find("./labels/label").text
		if label == 'STAIRS':
			trovato = 1
			x_entrance = float(stanza.find("./centroid/point").get('x'))
			y_entrance = float(stanza.find("./centroid/point").get('y'))
			break

#formula che calcola la pose del robot, in pratica e' una proporzione, x_stage : size_stage = x_xml : size_xml
x_pose_robot = ((x_entrance - x_min_xml) / (x_max_xml - x_min_xml)) * (larghezza * risoluzione)
y_pose_robot = ((y_entrance - y_min_xml) / (y_max_xml - y_min_xml)) * (altezza * risoluzione)  

#copio default.world in data e modifico le opportune linee 
with open('default.world','r') as world:
	data = world.readlines()
data[23] = 'resolution                '+str(risoluzione)+'\n'
data[39] = '  name "'+path_mappa+'"\n'
data[40] = '  bitmap "'+path_mappa+'"\n'
data[41] = '  size [ '+str(larghezza*risoluzione)+' '+str(altezza*risoluzione)+' 2.0 ]\n'
data[42] = '  pose [ '+str((larghezza*risoluzione)/2)+' '+str((altezza*risoluzione)/2)+' 0.0 0.0 ]\n'
data[52] = '  pose [ '+str(x_pose_robot)+' '+str(y_pose_robot)+' 0.0 45 ]\n'

#creo world e ci scrivo data 
with open(path_world,"w") as world:
	for line in data:
		world.write(line)



