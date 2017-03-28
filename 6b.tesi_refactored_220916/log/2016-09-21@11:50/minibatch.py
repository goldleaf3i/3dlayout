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
	ret,thresh1 = cv2.threshold(img_rgb,parametri_obj.cv2thresh,255,cv2.THRESH_BINARY)
	
	#------------------1.1_CANNY E HOUGH PER TROVARE MURI---------------------------------
	walls , canny = lay.start_canny_ed_hough(thresh1,parametri_obj.minVal,parametri_obj.maxVal,parametri_obj.rho,parametri_obj.theta,parametri_obj.thresholdHough,parametri_obj.minLineLength,parametri_obj.maxLineGap)
	
	if par.DISEGNA:
		#disegna mappa iniziale, canny ed hough
		dsg.disegna_map(img_rgb,filepath = path_obj.filepath )
		dsg.disegna_canny(canny,filepath = path_obj.filepath)
		dsg.disegna_hough(img_rgb,walls,filepath = path_obj.filepath)

	lines = lay.flip_lines(walls, img_rgb.shape[0]-1)
	walls = lay.crea_muri(lines)
	
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
	(contours, vertici) = lay.contorno_esterno(img_rgb, parametri_obj.minVal, parametri_obj.maxVal, xmin, xmax, ymin, ymax, path_obj.metricMap,filepath = path_obj.filepath, m = parametri_obj.m)
	if par.DISEGNA:
		dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax,filepath = path_obj.filepath)
	#-------------------------------------------------------------------------------------
	
	#---------------1.4_MEAN SHIFT PER TROVARE CLUSTER ANGOLARI---------------------------
	(indici, walls, cluster_angolari) = lay.cluster_ang(parametri_obj.h, parametri_obj.minOffset, walls, diagonali= parametri_obj.diagonali)
	#-------------------------------------------------------------------------------------

	#---------------1.5_CLUSTER SPAZIALI--------------------------------------------------
	cluster_spaziali = lay.cluster_spaz(parametri_obj.minLateralSeparation, walls,filepath = path_obj.filepath)
	if par.DISEGNA:
		dsg.disegna_cluster_spaziali(cluster_spaziali, walls,filepath = path_obj.filepath)
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
	if par.metodo_classificazione_celle:
		print "1.metodo di classificazione ", par.metodo_classificazione_celle
		(celle, celle_out, celle_poligoni, indici, celle_parziali, contorno, centroid, punti) = lay.classificazione_superfici(vertici, celle)
	else:
		print "2.metodo di classificazione ", par.metodo_classificazione_celle
		#sto classificando le celle con il metodo delle percentuali
		(celle_out, celle, centroid, punti,celle_poligoni, indici, celle_parziali) = lay.classifica_celle_con_percentuale(vertici, celle, img_ini)

	#-------------------------------------------------------------------------------------
		
	#--------------------------POLIGONI CELLE---------------------------------------------	
	(celle_poligoni, out_poligoni, parz_poligoni, centroid) = lay.crea_poligoni_da_celle(celle, celle_out, celle_parziali)
	#-------------------------------------------------------------------------------------
	
	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L------------------------
	(matrice_l, matrice_d, matrice_d_inv, X) = lay.crea_matrici(celle)
	#-------------------------------------------------------------------------------------

	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-------------------------
	clustersCelle = lay.DB_scan(parametri_obj.eps, parametri_obj.minPts, X, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours, filepath = path_obj.filepath)	
	#questo va disegnato per forza perche' restituisce la lista dei colori
	colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = path_obj.filepath)
	#-------------------------------------------------------------------------------------

	#------------------POLIGONI STANZE(spazio)--------------------------------------------
	stanze, spazi = lay.crea_spazio(clustersCelle, celle, celle_poligoni, colori, xmin, ymin, xmax, ymax, filepath = path_obj.filepath) 
	if par.DISEGNA:
		dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax,filepath = path_obj.filepath)
	#-------------------------------------------------------------------------------------
		
	#------fine layout--------------------------------------------------------------------
	
	#funzione per eliminare stanze che sono dei buchi interni
	print 'PLEASE CAMBIARE QUESTA COSA :|'
	#stanze = ac.elimina_stanze(stanze,estremi)
	#funzione per calcolare accuracy fc e bc
	print "Inizio a calcolare metriche"
	results = ac.calcola_accuracy(path_obj.nome_gt,estremi,stanze, path_obj.metricMap,path_obj.filepath, parametri_obj.flip_dataset)	
	print "Fine calcolare metriche"
	
	#creo i file pickle per il layout delle stanze
	print("creo pickle layout")
	pk.crea_pickle((stanze, clustersCelle, estremi, colori, spazi), path_obj.filepath_pickle_layout)
	#-------------------------------------------------------------------------------------
	
	#------------------------------GRAFO TOPOLOGICO---------------------------------------
	
	#costruisco il grafo 
	(G, pos, collegate, doorsVertices) = gtop.get_grafo(path_obj.metricMap, stanze, estremi, colori, filepath = path_obj.filepath)
	
	#creo i file pickle per il grafo topologico
	print("creo pickle grafoTopologico")
	pk.crea_pickle((stanze, clustersCelle, estremi, colori), path_obj.filepath_pickle_grafoTopologico)
	
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

def load_main(path_obj, parametri_obj):
	#carico layout
	pkl_file = open( path_obj.filepath_pickle_layout, 'rb')
	data1 = pickle.load(pkl_file)
	stanze = data1[0]
	clustersCelle = data1[1]
	estremi = data1[2]
	colori = data1[3]
	spazi = data1[4]
	
	print "controllo che non ci sia nulla di vuoto", len(stanze), len(clustersCelle), len(estremi), len(spazi), len(colori)
	#carico il grafo topologico
	pkl_file2 = open( path_obj.filepath_pickle_grafoTopologico, 'rb')
	data2 = pickle.load(pkl_file2)
	G = data2[0]
	pos = data2[1]
	collegate = data2[2]
	doorsVertices = data2[3]

	#ora carico parametri	
	
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
	3] ACCURACY NON FUNZIONA;\n
	4] QUANDO VENGONO RAGGRUPPATI TRA DI LORO I CLUSTER COLLINEARI, QUESTO VIENE FATTO A CASCATA. QUESTO FINISCE PER ALLINEARE ASSIEME MURA MOLTO DISTANTI;\n
	5] IL SISTEMA E' MOLTO SENSIBILE ALLA SCALA. BISOGNEREBBE INGRANDIRE TUTTE LE IMMAGINI FACENDO UN RESCALING E RISOLVERE QUESTO PROBLEMA. \n
	[4-5] FANNO SI CHE I CORRIDOI PICCOLI VENGANO CONSIDERATI COME UNA RETTA UNICA\n
	6] BISOGNEREBBE FILTRARE LE SUPERFICI TROPPO PICCOLE CHE VENGONO CREATE TRA DEI CLUSTER;\n
	7] LE IMMAGINI DI STAGE SONO TROPPO PICCOLE; VANNO RIPRESE PIU GRANDI \n
	8] MANCANO 30 DATASET DA FARE CON STAGE\n
	9] OGNI TANTO NON FUNZIONA IL GET CONTORNO PERCHE SBORDA ALL'INTERNO\n
	10] VANNO TARATI MEGLIO I PARAMETRI PER IL CLUSTERING\n
	11] LE LINEE DELLA CANNY E HOUGH TALVOLTA SONO TROPPO GROSSE \n
	12] BISOGNEREBBE AUMENTARE LA SEGMENTAZIONE CON UN VORONOI
	13] STAMPA L'IMMAGINE DELLA MAPPA AD UNA SCALA DIVERSA RISPETTO A QUELLA VERA.
	14] RISTAMPARE SCHOOL_GT IN GRANDE CHE PER ORA E' STAMPATO IN PICCOLO (800x600)
	15] NOI NON CALCOLIAMO LA DIFFUSION DEL METODO DI MURA; PER ALCUNI VERSI E' UN BENE PER ALTRI NO
	16] NON FACCIAMO IL CLUSTERING DEI SEGMENTI IN MANIERA CORRETTA; DOVREMMO SOLO FARE MEANSHIFT
	17] LA FASE DEI SEGMENTI VA COMPLETAMENTE RIFATTA; MEANSHIFT NON FUNZIONA COSI';  I SEGMENTI HANNO UN SACCO DI "==" CHE VANNO TOLTI; SPATIAL CLUSTRING VA CAMBIATO;
	18] OGNI TANTO IL GRAFO TOPOLOGICO CONNETTE STANZE CHE SONO ADIACENTI MA NON CONNESSE. VA RIVISTA LA PARTE DI MEDIALAXIS;
	19] PROVARE A USARE L'IMMAGINE CON IL CONTORNO RICALCATO SOLO PER FARE GETCONTOUR E NON NEGLI ALTRI STEP.
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
					parametri_obj.minLateralSeparation = 7 if 'SCHOOL' in DATASET  else 15					
					parametri_obj.cv2thresh = 150 if DATASET == 'SCHOOL' else 200
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
						load_main(path_obj, parametri_obj)
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
				load_main(path_obj, parametri_obj)
				LOGFILE.flush()
			
	#-------------------TEMPO IMPIEGATO-------------------------------------------------
	fine = time.time()
	elapsed = fine-start
	print "la computazione ha impiegato %f secondi" % elapsed	
				
if __name__ == '__main__':
	main()