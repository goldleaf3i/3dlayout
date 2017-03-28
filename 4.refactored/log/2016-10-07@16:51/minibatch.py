from __future__ import division
import datetime as dt
import numpy as np
import util.layout as lay
import util.GrafoTopologico as gtop
import util.transitional_kernels as tk
import util.MappaSemantica as sema
from object import Segmento as sg
from util import pickle_util as pk
from util import accuracy as ac
from util import layout as lay
from util import disegna as dsg
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
		dsg.disegna_map(img_rgb,filepath = path_obj.filepath )
		dsg.disegna_canny(canny,filepath = path_obj.filepath)
		dsg.disegna_hough(img_rgb,walls,filepath = path_obj.filepath)

	lines = lay.flip_lines(walls, img_rgb.shape[0]-1)
	walls = lay.crea_muri(lines)
	print "lines", len(lines), len(walls)
	if par.DISEGNA:
		#disegno linee
		dsg.disegna_segmenti(walls)#solo un disegno poi lo elimino
	
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
	(contours, vertici) = lay.contorno_esterno(img_rgb, parametri_obj, path_obj)
	if par.DISEGNA:
		dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax,filepath = path_obj.filepath)
	#-------------------------------------------------------------------------------------

	
	#---------------1.4_MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------
	(indici, walls, cluster_angolari) = lay.cluster_ang(parametri_obj.h, parametri_obj.minOffset, walls, diagonali= parametri_obj.diagonali)
		
	if par.DISEGNA:	
		#dsg.disegna_cluster_angolari(walls, cluster_angolari, filepath = path_obj.filepath,savename = '5b_cluster_angolari')
		dsg.disegna_cluster_angolari_corretto(walls, cluster_angolari, filepath = path_obj.filepath,savename = '5b_cluster_angolari')
	
	#-------------------------------------------------------------------------------------

	#---------------1.5_CLUSTER SPAZIALI--------------------------------------------------
	#questo metodo e' sbagliato, fai quella cosa con il hierarchical clustering per classificarli meglio.e trovare in sostanza un muro
	#cluster_spaziali = lay.cluster_spaz(parametri_obj.minLateralSeparation, walls)
	#inserisci qui il nuovo Cluster_spaz
	nuovo_clustering = 1 #1 metodo di matteo, 2 mio
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
			dsg.disegna_segmenti(segmenti_rappresentanti,filepath = path_obj.filepath, savename = "5c_segmenti_rappresentanti")

		#classifico i rappresentanti
		#qui va settata la soglia con cui voglio separare i cluster muro
		
		#segmenti_rappresentanti = segmenti_rappresentanti
		segmenti_rappresentanti = sg.spatialClustering(parametri_obj.sogliaLateraleClusterMura, segmenti_rappresentanti)
		#in questo momento ho un insieme di segmenti rappresentanti che hanno il cluster_spaziale settato correttamente, ora setto anche gli altri che hanno lo stesso cluster muro
		cluster_spaziali = lay.new_cluster_spaziale(walls, segmenti_rappresentanti, parametri_obj)
		
		#gestire gli outliers 
		#in sostanza devo unire al cluster piu' vicino ogni segmento di outlier 
		#lay.set_cluster_spaziale_to_outliers(walls, outliers, segmenti_rappresentanti)
		
		# print cluster_spaziali
# 		print len(cluster_spaziali)
# 		print len(set(cluster_spaziali))
# 		
# 		angolari = []
# 		for ang in set(cluster_angolari):
# 			row =[]
# 			for s in walls:
# 				if s.cluster_angolare == ang:
# 					row.append(s)
# 			angolari.append(row)
# 		
# 		cluster=[]
# 		for s in angolari[0]:
# 			cluster.append(s.cluster_spaziale)
# 		print len(set(cluster))
# 		
# 		cluster=[]
# 		for s in angolari[1]:
# 			cluster.append(s.cluster_spaziale)
# 		print len(set(cluster))
		

		
		
	if par.DISEGNA:
		dsg.disegna_cluster_spaziali(cluster_spaziali, walls,filepath = path_obj.filepath)
		#dsg.disegna_cluster_mura(cluster_mura, walls,filepath = path_obj.filepath, savename= '5d_cluster_mura')
	#-------------------------------------------------------------------------------------

	#-------------------1.6_CREO EXTENDED_LINES-------------------------------------------
	(extended_lines, extended_segments) = lay.extend_line(cluster_spaziali, walls, xmin, xmax, ymin, ymax,filepath = path_obj.filepath)
	
	if par.DISEGNA:
		dsg.disegna_extended_segments(extended_segments, walls,filepath = path_obj.filepath)		
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
	(matrice_l, matrice_d, matrice_d_inv, X) = lay.crea_matrici(celle)
	#-------------------------------------------------------------------------------------

	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-------------------------
	clustersCelle = lay.DB_scan(parametri_obj.eps, parametri_obj.minPts, X, celle_poligoni)
	#questo va disegnato per forza perche' restituisce la lista dei colori
	if par.DISEGNA:
		colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = path_obj.filepath)
	else:
		colori = dsg.get_colors(clustersCelle)
	#-------------------------------------------------------------------------------------

	#------------------POLIGONI STANZE(spazio)--------------------------------------------
	stanze, spazi = lay.crea_spazio(clustersCelle, celle, celle_poligoni, colori, xmin, ymin, xmax, ymax, filepath = path_obj.filepath) 
	if par.DISEGNA:
		dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax,filepath = path_obj.filepath)
	#-------------------------------------------------------------------------------------
		
	#cerco le celle parziali
	coordinate_bordi = [xmin, ymin, xmax, ymax]
	celle_parziali, parz_poligoni = lay.get_celle_parziali(celle, celle_out, coordinate_bordi)
	#creo i poligoni relativi alle celle_out
	out_poligoni = lay.get_poligoni_out(celle_out)
	

	#exit()
		
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
		dsg.disegna_stanze(stanze_reali, colori_reali, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '8_Stanze_reali')	
	#------------------------------------------------------------------------------------
	if par.DISEGNA:
		dsg.disegna_distance_transform(distanceMap, filepath = path_obj.filepath)
		dsg.disegna_medial_axis(points, b3, filepath = path_obj.filepath)
		dsg.plot_nodi_e_stanze(colori,estremi, G, pos, stanze, stanze_collegate, filepath = path_obj.filepath)
	
	#-----------------------------fine GrafoTopologico------------------------------------
	
	
	#-------------------------------------------------------------------------------------
	#DA QUI PARTE IL NUOVO PEZZO
	
	#creo l'oggetto plan che contiene tutti gli spazi, ogni stanza contiene tutte le sue celle, settate come out, parziali o interne. 	
	
	#setto gli spazi come out se non sono collegati a nulla.	
	spazi = sp.get_spazi_reali(spazi, stanze_reali) #elimino dalla lista di oggetti spazio quegli spazi che non sono collegati a nulla.
		
	#trovo le cellette parziali
	sp.set_cellette_parziali(spazi, parz_poligoni)#trovo le cellette di uno spazio che sono parziali 
	spazi = sp.trova_spazi_parziali(spazi)#se c'e' almeno una celletta all'interno di uno spazio che e' parziale, allora lo e' tutto lo spazio.
	
	#creo l'oggetto Plan
	plan_o = plan.Plan(spazi, contorno, out_poligoni) #spazio = oggetto Spazio. contorno = oggetto Polygon, out_poligoni = lista di Polygon
	dsg.disegna_spazi(spazi, colori, xmin, ymin, xmax, ymax,filepath = path_obj.filepath, savename = '13_spazi')	

	
		
	
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
			dsg.disegna_grafici_per_accuracy(stanze, stanze_gt, filepath = path_obj.filepath)
		print "Fine calcolare metriche"
	
	else:
		#setto results a 0, giusto per ricordarmi che non ho risultati per le mappe parziali 
		results = 0
	
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