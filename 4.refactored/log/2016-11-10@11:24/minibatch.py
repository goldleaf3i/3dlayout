from __future__ import division
import datetime as dt
import numpy as np
import util.layout as lay
import util.GrafoTopologico as gtop
import util.transitional_kernels as tk
import util.MappaSemantica as sema
import util.frontiere as fr
from object import Segmento as sg
from util import pickle_util as pk
from util import accuracy as ac
from util import layout as lay
from util import disegna as dsg
from util import predizionePlan_geometriche as pgeom
from object import Superficie as fc
from object import Spazio as sp
from object import Plan as plan
import parameters as par
import pickle
import os 
import glob
import shutil
import time
import cv2
import warnings

warnings.warn("Settare i parametri del lateralLine e cvThresh")

def start_main(parametri_obj, path_obj):	
	#----------------------------1.0_LAYOUT DELLE STANZE----------------------------------
	#------inizio layout
	#leggo l'immagine originale in scala di grigio e la sistemo con il thresholding
	img_rgb = cv2.imread(path_obj.metricMap)
	img_ini = img_rgb.copy() #copio l'immagine
	# 127 per alcuni dati, 255 per altri
	ret,thresh1 = cv2.threshold(img_rgb,parametri_obj.cv2thresh,255,cv2.THRESH_BINARY)#prova
	
	#------------------1.1_CANNY E HOUGH PER TROVARE MURI---------------------------------
	walls , canny = lay.start_canny_ed_hough(thresh1,parametri_obj)
	print len(walls)

	#walls , canny = lay.start_canny_ed_hough(img_rgb,parametri_obj)
	
	if par.DISEGNA:
		#disegna mappa iniziale, canny ed hough
		dsg.disegna_map(img_rgb,filepath = path_obj.filepath, format='png')
		dsg.disegna_canny(canny,filepath = path_obj.filepath, format='png')
		dsg.disegna_hough(img_rgb,walls,filepath = path_obj.filepath, format='png')

	lines = lay.flip_lines(walls, img_rgb.shape[0]-1)
	walls = lay.crea_muri(lines)
	print "lines", len(lines), len(walls)
	if par.DISEGNA:
		#disegno linee
		dsg.disegna_segmenti(walls, format='png')#solo un disegno poi lo elimino
	
	#------------1.2_SETTO XMIN YMIN XMAX YMAX DI walls-----------------------------------
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
	#-------------------------------------------------------------------------------------
	#---------------1.3_CONTORNO ESTERNO--------------------------------------------------
	
	#(contours, vertici) = lay.contorno_esterno(img_rgb, parametri_obj, path_obj)
	(contours, vertici) = lay.contorno_esterno_versione_tre(img_rgb)
	
	if par.DISEGNA:
		dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax,filepath = path_obj.filepath, format='png')
	#-------------------------------------------------------------------------------------
	#---------------1.4_MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------
	(indici, walls, cluster_angolari) = lay.cluster_ang(parametri_obj.h, parametri_obj.minOffset, walls, diagonali= parametri_obj.diagonali)
	if par.DISEGNA:	
		#dsg.disegna_cluster_angolari(walls, cluster_angolari, filepath = path_obj.filepath,savename = '5b_cluster_angolari')
		dsg.disegna_cluster_angolari_corretto(walls, cluster_angolari, filepath = path_obj.filepath,savename = '5b_cluster_angolari',format='png')
	
	#-------------------------------------------------------------------------------------

	#---------------1.5_CLUSTER SPAZIALI--------------------------------------------------
	#questo metodo e' sbagliato, fai quella cosa con il hierarchical clustering per classificarli meglio.e trovare in sostanza un muro
	#cluster_spaziali = lay.cluster_spaz(parametri_obj.minLateralSeparation, walls)
	#inserisci qui il nuovo Cluster_spaz
	nuovo_clustering = 2 #1 metodo di matteo, 2 mio
	#in walls ci sono tutti i segmenti
	if nuovo_clustering == 1:
		cluster_spaziali = lay.cluster_spaz(parametri_obj.minLateralSeparation, walls)#metodo di matteo 
	elif nuovo_clustering ==2:
		cluster_mura = lay.get_cluster_mura(walls, cluster_angolari, parametri_obj)#metodo di valerio

		cluster_mura_senza_outliers = []
		for c in cluster_mura:	
			if c!=-1:
				cluster_mura_senza_outliers.append(c)
		# ottengo gli outliers
# 		outliers = []
# 		for s in walls:
# 			if s.cluster_muro == -1:
# 				outliers.append(s)
# 		dsg.disegna_segmenti(outliers, savename = "outliers")

		
		#ora che ho un insieme di cluster relativi ai muri voglio andare ad unire quelli molto vicini
		#ottengo i rappresentanti dei cluster (tutti tranne gli outliers)
		#segmenti_rappresentanti = lay.get_rappresentanti(walls, cluster_mura)
		segmenti_rappresentanti = lay.get_rappresentanti(walls, cluster_mura_senza_outliers)
	
			
		if par.DISEGNA:
			dsg.disegna_segmenti(segmenti_rappresentanti,filepath = path_obj.filepath, savename = "5c_segmenti_rappresentanti", format='png')

		#classifico i rappresentanti
		#qui va settata la soglia con cui voglio separare i cluster muro
		
		#segmenti_rappresentanti = segmenti_rappresentanti
		segmenti_rappresentanti = sg.spatialClustering(parametri_obj.sogliaLateraleClusterMura, segmenti_rappresentanti)
		#in questo momento ho un insieme di segmenti rappresentanti che hanno il cluster_spaziale settato correttamente, ora setto anche gli altri che hanno lo stesso cluster muro
		cluster_spaziali = lay.new_cluster_spaziale(walls, segmenti_rappresentanti, parametri_obj)
			
		
	if par.DISEGNA:
		dsg.disegna_cluster_spaziali(cluster_spaziali, walls,filepath = path_obj.filepath, format='png')
		dsg.disegna_cluster_mura(cluster_mura, walls,filepath = path_obj.filepath, savename= '5d_cluster_mura', format='png')
	#-------------------------------------------------------------------------------------

	#-------------------1.6_CREO EXTENDED_LINES-------------------------------------------
	(extended_lines, extended_segments) = lay.extend_line(cluster_spaziali, walls, xmin, xmax, ymin, ymax,filepath = path_obj.filepath)
	
	if par.DISEGNA:
		dsg.disegna_extended_segments(extended_segments, walls,filepath = path_obj.filepath, format='png')		
	#-------------------------------------------------------------------------------------

	#-------------1.7_CREO GLI EDGES TRAMITE INTERSEZIONI TRA EXTENDED_LINES--------------
	edges = sg.crea_edges(extended_segments)
	#-------------------------------------------------------------------------------------
	
	#----------------------1.8_SETTO PESI DEGLI EDGES-------------------------------------
	edges = sg.setPeso(edges, walls)
	
	#-------------------------------------------------------------------------------------

	#----------------1.9_CREO LE CELLE DAGLI EDGES----------------------------------------
	celle = fc.crea_celle(edges)
	#-------------------------------------------------------------------------------------

	#----------------CLASSIFICO CELLE-----------------------------------------------------
	global centroid
	#verificare funzioni
	if par.metodo_classificazione_celle ==1:
		print "1.metodo di classificazione ", par.metodo_classificazione_celle
		(celle, celle_out, celle_poligoni, indici, celle_parziali, contorno, centroid, punti) = lay.classificazione_superfici(vertici, celle)
	elif par.metodo_classificazione_celle==2:
		print "2.metodo di classificazione ", par.metodo_classificazione_celle
		#sto classificando le celle con il metodo delle percentuali
		(celle_out, celle, centroid, punti,celle_poligoni, indici, celle_parziali) = lay.classifica_celle_con_percentuale(vertici, celle, img_ini)
	#-------------------------------------------------------------------------------------
		
	#--------------------------POLIGONI CELLE---------------------------------------------	
	(celle_poligoni, out_poligoni, parz_poligoni, centroid) = lay.crea_poligoni_da_celle(celle, celle_out, celle_parziali)
	
	#ora vorrei togliere le celle che non hanno senso, come ad esempio corridoi strettissimi, il problema e' che lo vorrei integrare con la stanza piu' vicina ma per ora le elimino soltanto 
	
	#RICORDA: stai pensando solo a celle_poligoni
	#TODO: questo metodo non funziona benissimo(sbagli ad eliminare le celle)
	#celle_poligoni, celle = lay.elimina_celle_insensate(celle_poligoni,celle, parametri_obj)#elimino tutte le celle che hanno una forma strana e che non ha senso siano stanze
	#-------------------------------------------------------------------------------------
		
	
	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L------------------------
	(matrice_l, matrice_d, matrice_d_inv, X) = lay.crea_matrici(celle, sigma = parametri_obj.sigma)
	#-------------------------------------------------------------------------------------

	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-------------------------
	clustersCelle = lay.DB_scan(parametri_obj.eps, parametri_obj.minPts, X, celle_poligoni)
	#questo va disegnato per forza perche' restituisce la lista dei colori
	if par.DISEGNA:
		colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = path_obj.filepath, format='png')
	else:
		colori = dsg.get_colors(clustersCelle, format='png')
	#-------------------------------------------------------------------------------------

	#------------------POLIGONI STANZE(spazio)--------------------------------------------
	stanze, spazi = lay.crea_spazio(clustersCelle, celle, celle_poligoni, colori, xmin, ymin, xmax, ymax, filepath = path_obj.filepath) 
	if par.DISEGNA:
		dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, format='png')
	#-------------------------------------------------------------------------------------
		
	#cerco le celle parziali
	coordinate_bordi = [xmin, ymin, xmax, ymax]
	celle_parziali, parz_poligoni = lay.get_celle_parziali(celle, celle_out, coordinate_bordi)#TODO: non ho controllato bene ma mi pare che questa cosa possa essere inserita nel metodo 1 che crca le celle parziali 
	#creo i poligoni relativi alle celle_out
	out_poligoni = lay.get_poligoni_out(celle_out)
	
# 	TODO: questo blocco e' da eliminare, mi serviva solo per risolvere un bug
# 	l = []
# 	for i,p in enumerate(out_poligoni):
# 		l.append(i)
# 	col_prova = dsg.get_colors(l)
# 	dsg.disegna_stanze(out_poligoni, col_prova, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename='0a_prova')
# 	exit()
# 		
		
		
	#--------------------------------fine layout------------------------------------------
	
	#------------------------------GRAFO TOPOLOGICO---------------------------------------
	
	#costruisco il grafo 
	(stanze_collegate, doorsVertices, distanceMap, points, b3) = gtop.get_grafo(path_obj.metricMap, stanze, estremi, colori, parametri_obj)
	(G, pos) = gtop.crea_grafo(stanze, stanze_collegate, estremi, colori)
	#ottengo tutte quelle stanze che non sono collegate direttamente ad un'altra, con molta probabilita' quelle non sono stanze reali
	stanze_non_collegate = gtop.get_stanze_non_collegate(stanze, stanze_collegate)
	
	#ottengo le stanze reali, senza tutte quelle non collegate
	stanze_reali, colori_reali = lay.get_stanze_reali(stanze, stanze_non_collegate, colori)
	if par.DISEGNA:
		#sto disegnando usando la lista di colori originale, se voglio la lista della stessa lunghezza sostituire colori con colori_reali
		dsg.disegna_stanze(stanze_reali, colori_reali, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '8_Stanze_reali', format='png')	
	#------------------------------------------------------------------------------------
	if par.DISEGNA:
		dsg.disegna_distance_transform(distanceMap, filepath = path_obj.filepath, format='png')
		dsg.disegna_medial_axis(points, b3, filepath = path_obj.filepath, format='png')
		dsg.plot_nodi_e_stanze(colori,estremi, G, pos, stanze, stanze_collegate, filepath = path_obj.filepath, format='png')
	
	#-----------------------------fine GrafoTopologico------------------------------------
	
	
	#-------------------------------------------------------------------------------------
	#DA QUI PARTE IL NUOVO PEZZO
	#IDEA:
	#1) trovo le celle parziali(uno spazio e' parziali se almeno una delle sue celle e' parziale) e creo l'oggetto Plan
	#2) postprocessing per capire se le celle out sono realmente out
	#3) postprocessing per unire gli spazi che dovrebbero essere uniti 
	
	#creo l'oggetto plan che contiene tutti gli spazi, ogni stanza contiene tutte le sue celle, settate come out, parziali o interne. 	
	
	#setto gli spazi come out se non sono collegati a nulla.	
	spazi = sp.get_spazi_reali(spazi, stanze_reali) #elimino dalla lista di oggetti spazio quegli spazi che non sono collegati a nulla.
		
	#---------------------------trovo le cellette parziali--------------------------------
	#se voglio il metodo che controlla le celle metto 1, 
	#se voglio il confronto di un intera stanza con l'esterno metto 2
	#se volgio il confronto di una stanza con quelli che sono i pixel classificati nella frontiera metto 3
	trova_parziali=3 
	
	if par.mappa_completa ==False and trova_parziali==1:
		#QUESTO METODO OGNI TANTO SBAGLIA PER VIA DELLA COPERTURA DEI SEGMANTI, verifico gli errori con il postprocessing per le stanze parziali.
		#TODO: Questo deve essere fatto solo se sono in presenza di mappe parziali
		sp.set_cellette_parziali(spazi, parz_poligoni)#trovo le cellette di uno spazio che sono parziali 
		spazi = sp.trova_spazi_parziali(spazi)#se c'e' almeno una celletta all'interno di uno spazio che e' parziale, allora lo e' tutto lo spazio.
		
	#creo l'oggetto Plan
	#faccio diventare la lista di out_poligoni delle cellette
	cellette_out = []
	for p,c in zip(out_poligoni, celle_out):	
		celletta = sp.Celletta(p,c)
		cellette_out.append(celletta)
	
	
	plan_o = plan.Plan(spazi, contorno, cellette_out) #spazio = oggetto Spazio. contorno = oggetto Polygon, cellette_out = lista di Cellette
	dsg.disegna_spazi(spazi, colori, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13_spazi', format='png')
	
	if par.mappa_completa ==False and trova_parziali==2:
		#secondo metodo per trovare gli spazi parziali. Fa una media pesata. migliore rispetto al primo ma bisogna fare tuning del parametro
		plan.trova_spazi_parziali_due(plan_o)
		
	if par.mappa_completa == False and trova_parziali==3:
		#terzo metodo per trovare le celle parziali basato sulla ricerca delle frontiere.
		immagine_cluster, frontiere, labels, lista_pixel_frontiere = fr.ottieni_frontire_principali(img_ini)
		
		if len(labels) > 0:
			plan.trova_spazi_parziali_da_frontiere(plan_o, lista_pixel_frontiere, immagine_cluster, labels)
			spazi = sp.trova_spazi_parziali(plan_o.spazi)
		
		if par.DISEGNA:
			dsg.disegna_map(immagine_cluster,filepath = path_obj.filepath, savename = '0a_frontiere', format='png')
	
	#-------------------------------------------------------------------------------------	
	
	#-----------------------------calcolo peso per extended_segments----------------------
	
	#calcolo il peso di un extended segment in base alla copertura sei segmenti. Ovviamente non potra' mai essere 100%.
	extended_segments = sg.setPeso(extended_segments, walls)#TODO:controllare che sia realmente corretto
	#calcolo per ogni extended segment quante sono le stanze che tocca(la copertura)
	lay.calcola_copertura_extended_segment(extended_segments, plan_o.spazi)
	plan_o.set_extended_segments(extended_segments)
	
	#-------------------------------------------------------------------------------------	


	#---------------------------unisco spazi oversegmentati ------------------------------
	
	#unisco le spazi che sono state divisi erroneamente
	#fa schifissimo come metodo(nel caso lo utilizziamo per MCMCs)
	uniciStanzeOversegmentate = 2
	#1) primo controlla cella per cella 
	#2) unisce facendo una media pesata
	#3) non unisce le stanze, non fa assolutamente nulla, usato per mappe parziali se non voglio unire stanze
	if uniciStanzeOversegmentate ==1:
		#fa schifissimo come metodo(nel caso lo utilizziamo per MCMCs)

		#unione stanze
		#provo ad usare la distance transforme
		#dsg.disegna_distance_transform_e_stanze(distanceMap,stanze,colori, filepath = path_obj.filepath, savename = 'distance_and_stanze')
		
		#se esistono due spazi che sono collegati tramite un edge di una cella che ha un peso basso allora unisco quegli spazi
		plan.unisci_stanze_oversegmentate(plan_o)
		#cambio anche i colori
		dsg.disegna_spazi(plan_o.spazi, dsg.get_colors(plan_o.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13b_spazi_nuovo', format='png')	
	elif uniciStanzeOversegmentate == 2:
		#TODO: questo metodo funziona meglio del primo, vedere se vale la pena cancellare il primo
		#metodo molto simile a quello di Mura per il postprocessing		
		plan.postprocessing(plan_o, parametri_obj)
		dsg.disegna_spazi(plan_o.spazi, dsg.get_colors(plan_o.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13b_spazi_nuovo', format='png')	
	else:
		#se non voglio unire le stanze, ad esempio e' utile quando sto guardando le mappe parziali
		dsg.disegna_spazi(plan_o.spazi, dsg.get_colors(plan_o.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13b_spazi_nuovo', format='png')	
	#-------------------------------------------------------------------------------------	

	
	
	
	#------------------------------PREDIZIONE GEOMETRICA----------------------------------	
	#da qui comincia la parte di predizione, io la sposterei in un altro file
	
	#ricavo gli spazi parziali
	cellette_out = plan_o.cellette_esterne
	spazi_parziali = []
	for s in plan_o.spazi:
		if s.parziale == True:
			spazi_parziali.append(s)	
	
	
	
	
	
	# TODO:da eliminare, mi serviva solo per delle immagini e per controllare di aver fatto tutto giusto
	#guardo quali sono gli extended che sto selezionando
	for index,s in enumerate(spazi_parziali):
		celle_di_altre_stanze = []
		for s2 in plan_o.spazi:
			if s2 !=s:
				for c in s2.cells:
					celle_di_altre_stanze.append(c)		 
		a = sp.estrai_extended_da_spazio(s, plan_o.extended_segments, cellette_out)
		dsg.disegna_extended_segments(a, walls,filepath = path_obj.filepath, format='png', savename = '7a_extended'+str(index))		
		b= sp.estrai_solo_extended_visti(s, plan_o.extended_segments, cellette_out)#estraggo solo le extended in comune con l'esterno
		c= sp.estrai_solo_extended_visti(s, plan_o.extended_segments, celle_di_altre_stanze)#estraggo solo le extended in comune con le altre stanze confinanti
		tot_segment = list(set(b+c))
		dsg.disegna_extended_segments(tot_segment, walls,filepath = path_obj.filepath, format='png', savename = '7b_extended'+str(index))		
		print len(a)
		print len(b)
	exit()


	import copy	
	plan_o_2 = copy.deepcopy(plan_o)#copio l'oggetto per poter eseguire le azioni separatamente
	plan_o_3 = copy.deepcopy(plan_o)

	
	#------------------------------AZIONE GEOMETRICA 1)+2)--------------------------------	
	
	#-------------------------------AZIONE GEOMETRICA 1)----------------------------------
	#-----AGGIUNGO CELLE OUT A CELLE PARZIALI SOLO SE QUESTE CELLE OUT SONO STATE TOCCANTE DAL BEAM DEL LASER
		
	celle_candidate = []
	for s in spazi_parziali:
		celle_confinanti = pgeom.estrai_celle_confinanti_alle_parziali(plan_o, s)#estraggo le celle confinanti alle celle interne parziali delle stanze parziali.
		print "le celle confinanti sono: ", len(celle_confinanti)	
		
		#unisco solo se le celle sono state toccate dal beam del laser
		celle_confinanti = plan.trova_celle_toccate_dal_laser_beam(celle_confinanti, immagine_cluster)
		
		#delle celle confinanti non devo unire quelle che farebbero sparire una parete.
		celle_confinanti = pgeom.elimina_celle_con_parete_vista(celle_confinanti, s)
		
		
		#faccio una prova per unire una cella che e' toccata dal beam del laser.
		if len(celle_confinanti)>0:
			#unisco la cella allo spazio
			for cella in celle_confinanti:
				if cella.vedo_frontiera == True:
					sp.aggiungi_cella_a_spazio(s, cella)
				
	dsg.disegna_spazi(plan_o.spazi, dsg.get_colors(plan_o.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13c_azione_geom_1', format='png')	

 	#-------------------------------AZIONE GEOMETRICA 2)-----------------------------------	
	#--UNISCO LE CELLE IN BASE ALLE PARETI CHE CONDIVIDONO CON ALTRE STANZE 
	
	for s in spazi_parziali:
		#estraggo le celle out che confinano con le celle parziali
		celle_confinanti = pgeom.estrai_celle_confinanti_alle_parziali(plan_o, s)#estraggo le celle confinanti alle celle interne parziali delle stanze parziali.
		print "le celle confinanti sono: ", len(celle_confinanti)
		
		#delle celle confinanti appena estratte devo prendere solamente quelle che hanno tutti i lati supportati da una extended line
		celle_confinanti = pgeom.estrai_celle_supportate_da_extended_segmement(celle_confinanti, s, plan_o.extended_segments)
		
		#delle celle confinanti non devo unire quelle che farebbero sparire una parete.
		celle_confinanti = pgeom.elimina_celle_con_parete_vista(celle_confinanti, s)
		
		
		#unisco solo quelle selezionate
		#TODO questa parte e' da cancellare
		if len(celle_confinanti)>0:
			#unisco la cella allo spazio
			for cella in celle_confinanti:
				sp.aggiungi_cella_a_spazio(s, cella)
		
	dsg.disegna_spazi(plan_o.spazi, dsg.get_colors(plan_o.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13e_azione_geom_1_piu_geom_2', format='png')	


	#----------------------------------FINE 1)+2)-----------------------------------------
	
	
	
	#----------------------------FACCIO SOLO AZIONE GEOM 2)-------------------------------
	#questa azione la faccio su una copia di plan
	
	#ricavo gli spazi parziali dalla copia di plan_o che sono esattamente una copia di spazi_parziali precedente.
	cellette_out = plan_o_2.cellette_esterne
	spazi_parziali = []
	for s in plan_o_2.spazi:
		if s.parziale == True:
			spazi_parziali.append(s)
	
	for s in spazi_parziali:
		#estraggo le celle out che confinano con le celle parziali
		celle_confinanti = pgeom.estrai_celle_confinanti_alle_parziali(plan_o_2, s)#estraggo le celle confinanti alle celle interne parziali delle stanze parziali.
		print "le celle confinanti sono: ", len(celle_confinanti)
		
		#delle celle confinanti appena estratte devo prendere solamente quelle che hanno tutti i lati supportati da una extended line
		celle_confinanti = pgeom.estrai_celle_supportate_da_extended_segmement(celle_confinanti, s, plan_o_2.extended_segments)
		print "le celle confinanti sono2: ", len(celle_confinanti)
		#delle celle confinanti non devo unire quelle che farebbero sparire una parete.
		celle_confinanti = pgeom.elimina_celle_con_parete_vista(celle_confinanti, s)
		print "le celle confinanti sono3: ", len(celle_confinanti)
		#unisco solo quelle selezionate
		#TODO questa parte e' da cancellare
		if len(celle_confinanti)>0:
			#unisco la cella allo spazio
			for cella in celle_confinanti:
				sp.aggiungi_cella_a_spazio(s, cella)
		
	dsg.disegna_spazi(plan_o_2.spazi, dsg.get_colors(plan_o_2.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13d_azione_geom_2', format='png')	
	#----------------------------------FINE SOLO AZIONE GEOM 2)--------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	#TODO: da cancellare
	'''
	#----------------------------FACCIO SOLO AZIONE GEOM 3)-------------------------------
	#TODO:questa azione non puo' funzionare se ci sono linee oblique()
	
	
	#questa azione la faccio su una copia di plan
	
	#ricavo gli spazi parziali dalla copia di plan_o che sono esattamente una copia di spazi_parziali precedente.
	cellette_out = plan_o_3.cellette_esterne
	spazi_parziali = []
	for s in plan_o_3.spazi:
		if s.parziale == True:
			spazi_parziali.append(s)
				
	for s in spazi_parziali:
 		celle_da_unire = pgeom.azione_geometrica_3(s, plan_o_3)
		
		#unisco solo quelle selezionate
		#TODO questa parte e' da cancellare
		# if len(celle_confinanti)>0:
# 			#unisco la cella allo spazio
# 			for cella in celle_confinanti:
# 				sp.aggiungi_cella_a_spazio(s, cella)
		
	dsg.disegna_spazi(plan_o_3.spazi, dsg.get_colors(plan_o_3.spazi), xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13d_azione_geom_3', format='png')	
		
		
	
	#----------------------------------FINE SOLO AZIONE GEOM 3)--------------------------
	'''
	
	
				
	
	#------------------------CREO PICKLE--------------------------------------------------
	#creo i file pickle per il layout delle stanze
	print("creo pickle layout")
	pk.crea_pickle((stanze, clustersCelle, estremi, colori, spazi, stanze_reali, colori_reali), path_obj.filepath_pickle_layout)
	print("ho finito di creare i pickle del layout")
	#creo i file pickle per il grafo topologico
	print("creo pickle grafoTopologico")
	pk.crea_pickle((stanze, clustersCelle, estremi, colori), path_obj.filepath_pickle_grafoTopologico)
	print("ho finito di creare i pickle del grafo topologico")
	
	#-----------------------CALCOLO ACCURACY----------------------------------------------
	#L'accuracy e' da controllare, secondo me non e' corretta.
	
	
	if par.mappa_completa:
		#funzione per calcolare accuracy fc e bc
		print "Inizio a calcolare metriche"
		results, stanze_gt = ac.calcola_accuracy(path_obj.nome_gt,estremi,stanze_reali, path_obj.metricMap,path_obj.filepath, parametri_obj.flip_dataset)	
		#results, stanze_gt = ac.calcola_accuracy(path_obj.nome_gt,estremi,stanze, path_obj.metricMap,path_obj.filepath, parametri_obj.flip_dataset)	

		if par.DISEGNA:
			dsg.disegna_grafici_per_accuracy(stanze, stanze_gt, filepath = path_obj.filepath, format='png')
		print "Fine calcolare metriche"
	
	else:
		#setto results a 0, giusto per ricordarmi che non ho risultati per le mappe parziali 
		results = 0
		stanze_gt = ac.get_stanze_gt(path_obj.nome_gt, estremi, flip_dataset = False)
		if par.DISEGNA:
			#raccolgo i poligoni
			stanze_acc = []
			for spazio in plan_o.spazi:
				stanze_acc.append(spazio.spazio)
			dsg.disegna_grafici_per_accuracy(stanze_acc, stanze_gt, filepath = path_obj.filepath, format='png')
	
	#in questa fase il grafo non e' ancora stato classificato con le label da dare ai vai nodi.
	#-------------------------------------------------------------------------------------	
	#creo il file xml dei parametri 
	par.to_XML(parametri_obj, path_obj)
	
	
	
	#-------------------------prova transitional kernels----------------------------------
	
	#splitto una stanza e restituisto la nuova lista delle stanze
	#stanze, colori = tk.split_stanza_verticale(2, stanze, colori,estremi) 
	#stanze, colori = tk.split_stanza_orizzontale(3, stanze, colori,estremi)
	#stanze, colori = tk.slit_all_cell_in_room(spazi, 1, colori, estremi) #questo metodo e' stato fatto usando il concetto di Spazio, dunque fai attenzione perche' non restituisce la cosa giusta.
	#stanze, colori = tk.split_stanza_reverce(2, len(stanze)-1, stanze, colori, estremi) #questo unisce 2 stanze precedentemente splittate, non faccio per ora nessun controllo sul fatto che queste 2 stanze abbiano almeno un muro in comune, se sono lontani succede un casino

	#-----------------------------------------------------------------------------------

	
	#-------------------------MAPPA SEMANTICA-------------------------------------------
	'''
	#in questa fase classifico i nodi del grafo e conseguentemente anche quelli della mappa.
	
	#gli input di questa fase non mi sono ancora molto chiari 
	#per ora non la faccio poi se mi serve la copio/rifaccio, penso proprio sia sbagliata.

	#stanze ground truth
	(stanze_gt, nomi_stanze_gt, RC, RCE, FCES, spaces, collegate_gt) = sema.get_stanze_gt(nome_gt, estremi)

	#corrispondenze tra gt e segmentate (backward e forward)
	(indici_corrispondenti_bwd, indici_gt_corrispondenti_fwd) = sema.get_corrispondenze(stanze,stanze_gt)

	#creo xml delle stanze segmentate
	id_stanze = sema.crea_xml(nomeXML,stanze,doorsVertices,collegate,indici_gt_corrispondenti_fwd,RCE,nomi_stanze_gt)

	#parso xml creato, va dalla cartella input alla cartella output/xmls, con feature aggiunte
	xml_output = sema.parsa(dataset_name, nomeXML)


	#classifico
	predizioniRCY = sema.classif(dataset_name,xml_output,'RC','Y',30)
	predizioniRCN = sema.classif(dataset_name,xml_output,'RC','N',30)
	predizioniFCESY = sema.classif(dataset_name,xml_output,'RCES','Y',30)
	predizioniFCESN = sema.classif(dataset_name,xml_output,'RCES','N',30)

	#creo mappa semantica segmentata e ground truth e le plotto assieme
	
	sema.creaMappaSemantica(predizioniRCY, G, pos, stanze, id_stanze, estremi, colori, clustersCelle, collegate)
	sema.creaMappaSemanticaGt(stanze_gt, collegate_gt, RC, estremi, colori)
	plt.show()
	sema.creaMappaSemantica(predizioniRCN, G, pos, stanze, id_stanze, estremi, colori, clustersCelle, collegate)
	sema.creaMappaSemanticaGt(stanze_gt, collegate_gt, RC, estremi, colori)
	plt.show()
	sema.creaMappaSemantica(predizioniFCESY, G, pos, stanze, id_stanze, estremi, colori, clustersCelle, collegate)
	sema.creaMappaSemanticaGt(stanze_gt, collegate_gt, FCES, estremi, colori)
	plt.show()
	sema.creaMappaSemantica(predizioniFCESN, G, pos, stanze, id_stanze, estremi, colori, clustersCelle, collegate)
	sema.creaMappaSemanticaGt(stanze_gt, collegate_gt, FCES, estremi, colori)
	plt.show()
	'''
	#-----------------------------------------------------------------------------------
	
	
	print "to be continued..."
	return results
	#TODO

def load_main(filepath_pickle_layout, filepath_pickle_grafoTopologico, parXML):
	#carico layout
	pkl_file = open(filepath_pickle_layout, 'rb')
	data1 = pickle.load(pkl_file)
	stanze = data1[0]
	clustersCelle = data1[1]
	estremi = data1[2]
	colori = data1[3]
	spazi = data1[4]
	stanze_reali = data1[5]
	colori_reali= data1[6]
	
	#print "controllo che non ci sia nulla di vuoto", len(stanze), len(clustersCelle), len(estremi), len(spazi), len(colori)
	#carico il grafo topologico
	pkl_file2 = open( filepath_pickle_grafoTopologico, 'rb')
	data2 = pickle.load(pkl_file2)
	G = data2[0]
	pos = data2[1]
	stanze_collegate = data2[2]
	doorsVertices = data2[3]
	
	#creo dei nuovi oggetti parametri caricando i dati dal file xml
	new_parameter_obj, new_path_obj =  par.load_from_XML(parXML)
		
	#continuare il metodo da qui	
	
	
def makeFolders(location,datasetList): 
	for dataset in datasetList:
		if not os.path.exists(location+dataset):
			os.mkdir(location+dataset)
			os.mkdir(location+dataset+"_pickle")			

def main():
	start = time.time()
	print ''' PROBLEMI NOTI \n
	1] LE LINEE OBLIQUE NON VANNO;\n
	2] NON CLASSIFICA LE CELLE ESTERNE CHE STANNO DENTRO IL CONVEX HULL, CHE QUINDI VENGONO CONSIDERATE COME STANZE;\n
	OK 3] ACCURACY NON FUNZIONA;\n
	4] QUANDO VENGONO RAGGRUPPATI TRA DI LORO I CLUSTER COLLINEARI, QUESTO VIENE FATTO A CASCATA. QUESTO FINISCE PER ALLINEARE ASSIEME MURA MOLTO DISTANTI;\n
	5] IL SISTEMA E' MOLTO SENSIBILE ALLA SCALA. BISOGNEREBBE INGRANDIRE TUTTE LE IMMAGINI FACENDO UN RESCALING E RISOLVERE QUESTO PROBLEMA. \n
	[4-5] FANNO SI CHE I CORRIDOI PICCOLI VENGANO CONSIDERATI COME UNA RETTA UNICA\n
	6] BISOGNEREBBE FILTRARE LE SUPERFICI TROPPO PICCOLE CHE VENGONO CREATE TRA DEI CLUSTER;\n
	7] LE IMMAGINI DI STAGE SONO TROPPO PICCOLE; VANNO RIPRESE PIU GRANDI \n
	>> LANCIARE IN BATCH SU ALIENWARE\n
	>> RENDERE CODICE PARALLELO\n
	8] MANCANO 30 DATASET DA FARE CON STAGE\n
	9] OGNI TANTO NON FUNZIONA IL GET CONTORNO PERCHE SBORDA ALL'INTERNO\n
	>> PROVARE CON SCAN BORDO (SU IMMAGINE COPIA)\n
	>> PROVARE A SETTARE IL PARAMETRO O A MODIFICARE IL METODO DI SCAN BORDO\n
	>> CERCARE SOLUZIONI ALTERNATIVE (ES IDENTIFICARE LE CELLE ESTERNE)\n
	OK 10] VANNO TARATI MEGLIO I PARAMETRI PER IL CLUSTERING\n 
	>> I PARAMETRI DE CLUSTERING SONO OK; OGNI TANTO FA OVERSEGMENTAZIONE.\n
	>>> EVENTUALMENTE SE SI VEDE CHE OVERSEGMENTAZIONE SONO UN PROBLEMA CAMBIARE CLUSTERING O MERGE CELLE\n
	11] LE LINEE DELLA CANNY E HOUGH TALVOLTA SONO TROPPO GROSSE \n
	>> IN REALTA SEMBRA ESSERE OK; PROVARE CON MAPPE PIU GRANDI E VEDERE SE CAMBIA.
	12] BISOGNEREBBE AUMENTARE LA SEGMENTAZIONE CON UN VORONOI
	OK 13] STAMPA L'IMMAGINE DELLA MAPPA AD UNA SCALA DIVERSA RISPETTO A QUELLA VERA.\n
	OK 14] RISTAMPARE SCHOOL_GT IN GRANDE CHE PER ORA E' STAMPATO IN PICCOLO (800x600)\n
	OK  VEDI 10] 15] NOI NON CALCOLIAMO LA DIFFUSION DEL METODO DI MURA; PER ALCUNI VERSI E' UN BENE PER ALTRI NO\n
	OK VEDI 4] 16] NON FACCIAMO IL CLUSTERING DEI SEGMENTI IN MANIERA CORRETTA; DOVREMMO SOLO FARE MEANSHIFT\n
	17] LA FASE DEI SEGMENTI VA COMPLETAMENTE RIFATTA; MEANSHIFT NON FUNZIONA COSI';  I SEGMENTI HANNO UN SACCO DI "==" CHE VANNO TOLTI; SPATIAL CLUSTRING VA CAMBIATO;\n
	18] OGNI TANTO IL GRAFO TOPOLOGICO CONNETTE STANZE CHE SONO ADIACENTI MA NON CONNESSE. VA RIVISTA LA PARTE DI MEDIALAXIS;\n
	19] PROVARE A USARE L'IMMAGINE CON IL CONTORNO RICALCATO SOLO PER FARE GETCONTOUR E NON NEGLI ALTRI STEP.\n
	20] TOGLIERE THRESHOLD + CANNY -> USARE SOLO CANNY.\n
	21] TOGLIERE LE CELLE INTERNE CHE SONO BUCHI.\n
	>> USARE VORONOI PER CONTROLLARE LA CONNETTIVITA.\n
	>> USARE THRESHOLD SU SFONDO \n
	>> COMBINARE I DUE METODI\n
	22] RIMUOVERE LE STANZE ERRATE:\n
	>> STANZE "ESTERNE" INTERNE VANNO TOLTE IN BASE ALLE CELLE ESTERNE\n
	>> RIMUOVERE STANZE CON FORME STUPIDE (ES PARETI LUNGHE STRETTE), BISOGNA DECIDERE SE ELIMINARLE O INGLOBARLE IN UN ALTRA STANZA\n
	23] RISOLVERE TUTTI I WARNING.\n
	
	da chiedere: guardare il metodo clustering_dbscan_celle(...) in layout la riga 
	af = DBSCAN(eps, min_samples, metric="precomputed").fit(X) non dovrebbe essere cosi?
	af = DBSCAN(eps= eps, min_samples = min_samples, metric="precomputed").fit(X)
	'''

	print '''
	FUNZIONAMENTO:\n
	SELEZIONARE SU QUALI DATASETs FARE ESPERIMENTI (variabile DATASETs -riga165- da COMMENTARE / DECOMMENTARE)\n
	SPOSTARE LE CARTELLE CON I NOMI DEI DATASET CREATI DALL'ESPERIMENTO PRECEDENTE IN UNA SOTTO-CARTELLA (SE TROVA UNA CARTELLA CON LO STESSO NOME NON CARICA LA MAPPA)\n
	SETTARE I PARAMERI \n
	ESEGUIRE\n
	OGNI TANTO IL METODO CRASHA IN FASE DI VALUTAZIONE DI ACCURATEZZA. NEL CASO, RILANCIARLO\n
	SPOSTARE TUTTI I RISULTATI IN UNA CARTELLA IN RESULTS CON UN NOME SIGNIFICATIVO DEL TEST FATTO\n
	SALVARE IL MAIN DENTRO QUELLA CARTELLA\n
	'''



	#-------------------PARAMETRI-------------------------------------------------------
	#carico parametri di default
	parametri_obj = par.Parameter_obj()
	#carico path di default
	path_obj = par.Path_obj()
	#-----------------------------------------------------------------------------------	
	
	makeFolders(path_obj.OUTFOLDERS,path_obj.DATASETs)
	skip_performed = True
	
	#-----------------------------------------------------------------------------------
	#creo la cartella di log con il time stamp 
	our_time = str(dt.datetime.now())[:-10].replace(' ','@') #get current time
	SAVE_FOLDER = os.path.join('./log', our_time)
	if not os.path.exists(SAVE_FOLDER):
		os.mkdir(SAVE_FOLDER)
	SAVE_LOGFILE = SAVE_FOLDER+'/log.txt'
	#------------------------------------------------------------------------------------

	with open(SAVE_LOGFILE,'w+') as LOGFILE:
		print "AZIONE", par.AZIONE
		print >>LOGFILE, "AZIONE", par.AZIONE
		shutil.copy('./minibatch.py',SAVE_FOLDER+'/minibatch.py') #copio il file del main
		shutil.copy('./parameters.py',SAVE_FOLDER+'/parameters.py') #copio il file dei parametri
		
		if par.AZIONE == "batch":
			if par.LOADMAIN==False:
				print >>LOGFILE, "SONO IN MODALITA' START MAIN"
			else:
				print >>LOGFILE, "SONO IN MODALITA' LOAD MAIN"
			print >>LOGFILE, "-----------------------------------------------------------"
			for DATASET in path_obj.DATASETs :
				print >>LOGFILE, "PARSO IL DATASET", DATASET
				global_results = []
				print 'INIZIO DATASET ' , DATASET
				for metricMap in glob.glob(path_obj.INFOLDERS+'IMGs/'+DATASET+'/*.png') :
				
					print >>LOGFILE, "---parso la mappa: ", metricMap
					print 'INIZIO A PARSARE ', metricMap
					path_obj.metricMap =metricMap 
					map_name = metricMap.split('/')[-1][:-4]
					#print map_name
					SAVE_FOLDER = path_obj.OUTFOLDERS+DATASET+'/'+map_name
					SAVE_PICKLE = path_obj.OUTFOLDERS+DATASET+'_pickle/'+map_name.split('.')[0]
					if par.LOADMAIN==False:
						if not os.path.exists(SAVE_FOLDER):
							os.mkdir(SAVE_FOLDER)
							os.mkdir(SAVE_PICKLE)
						else:
							# evito di rifare test che ho gia fatto
							if skip_performed :
								print 'GIA FATTO; PASSO AL SUCCESSIVO'
								continue

					#print SAVE_FOLDER
					path_obj.filepath = SAVE_FOLDER+'/'
					path_obj.filepath_pickle_layout = SAVE_PICKLE+'/'+'Layout.pkl'
					path_obj.filepath_pickle_grafoTopologico = SAVE_PICKLE+'/'+'GrafoTopologico.pkl'

					
					add_name = '' if DATASET == 'SCHOOL' else ''
					
					if par.mappa_completa == False:
						nome = map_name.split('_updated')[0]
						path_obj.nome_gt = path_obj.INFOLDERS+'XMLs/'+DATASET+'/'+nome+'_updated.xml'
					else:
						path_obj.nome_gt = path_obj.INFOLDERS+'XMLs/'+DATASET+'/'+map_name+add_name+'.xml'				
					
					#--------------------new parametri-----------------------------------
					#setto i parametri differenti(ogni dataset ha parametri differenti)
					parametri_obj.minLateralSeparation = 7 if (DATASET=='SCHOOL' or DATASET=='PARZIALI' or DATASET=='SCHOOL_grandi') else 15	
					#parametri_obj.cv2thresh = 150 if DATASET == 'SCHOOL' else 200
					parametri_obj.cv2thresh = 150 if (DATASET=='SCHOOL' or DATASET=='PARZIALI' or DATASET == 'SCHOOL_grandi') else 200
					parametri_obj.flip_dataset = True if DATASET == 'SURVEY' else False
					#--------------------------------------------------------------------
					#-------------------ESECUZIONE---------------------------------------
					
					if par.LOADMAIN==False:
						print "start main"
						results = start_main(parametri_obj, path_obj)
						global_results.append(results);
			
						#calcolo accuracy finale dell'intero dataset
						if metricMap == glob.glob(path_obj.INFOLDERS+'IMGs/'+DATASET+'/*.png')[-1]:
							accuracy_bc_medio = []
							accuracy_bc_in_pixels = []
							accuracy_fc_medio = []
							accuracy_fc_in_pixels=[]
		
		
							for i in global_results :
								accuracy_bc_medio.append(i[0])
								accuracy_fc_medio.append(i[2]) 
								accuracy_bc_in_pixels.append(i[4])
								accuracy_fc_in_pixels.append(i[5])
		
							filepath= path_obj.OUTFOLDERS+DATASET+'/'
							print filepath
							f = open(filepath+'accuracy.txt','a')
							#f.write(filepath)
							f.write('accuracy_bc = '+str(np.mean(accuracy_bc_medio))+'\n')
							f.write('accuracy_bc_pixels = '+str(np.mean(accuracy_bc_in_pixels))+'\n')
							f.write('accuracy_fc = '+str(np.mean(accuracy_fc_medio))+'\n')
							f.write('accuracy_fc_pixels = '+str(np.mean(accuracy_fc_in_pixels))+'\n\n')
							f.close()
						LOGFILE.flush()
					elif par.LOADMAIN==True:
						print "load main"
						print >>LOGFILE, "---parso la mappa: ", path_obj.metricMap
						load_main(path_obj.filepath_pickle_layout, path_obj.filepath_pickle_grafoTopologico, path_obj.filepath+"parametri.xml")
						LOGFILE.flush()
					
					
				else :
					continue
				break
			LOGFILE.flush()
		
		elif par.AZIONE =='mappa_singola':
			#-------------------ESECUZIONE singola mappa----------------------------------
			if par.LOADMAIN==False:
				print "start main"
				print >>LOGFILE, "SONO IN MODALITA' START MAIN"
				print >>LOGFILE, "---parso la mappa: ", path_obj.metricMap
				start_main(parametri_obj, path_obj)
				LOGFILE.flush()
			else:
				print "load main"
				print >>LOGFILE, "SONO IN MODALITA' LOAD MAIN"
				print >>LOGFILE, "---parso la mappa: ", path_obj.metricMap
				load_main(path_obj.filepath_pickle_layout, path_obj.filepath_pickle_grafoTopologico, path_obj.filepath+"parametri.xml")
				LOGFILE.flush()
			
	#-------------------TEMPO IMPIEGATO-------------------------------------------------
	fine = time.time()
	elapsed = fine-start
	print "la computazione ha impiegato %f secondi" % elapsed	
				
if __name__ == '__main__':
	main()