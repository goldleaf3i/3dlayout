'''
funzioni utili alla costruzione del layout delle stanze partendo da un immagine nota della mappa, dalla quale non si conosce nulla.
'''
import cv2
import warnings
warnings.warn("rinominare object")

from util import disegna as dsg
from util import mean_shift as ms

from object import Segmento as sg
from object import Retta as rt
from object import Extended_segment as ext
from object import Superficie as fc #prima si chiamava cella 
from object import Spazio as sp

import image as im
from shapely.geometry import Polygon
import math
import matrice as mtx
from sklearn.cluster import DBSCAN
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from descartes import PolygonPatch 



#secondo me in questa classe ci vuole un costruttore che contenga il contorno, la lista delle stanze, e tutto quello che mi serve.
def scan_bordo(img_rgb, soglia_minima_grigio = 120, soglia_massima_grigio = 180):	
	#parametri molto buoni che sbagliano poco sono, soglia_minima_grigio = 120, soglia_massima_grigio = 180
	#faccio uno scan della mappa pixel per pixel in scala di grigio e sistemo i buchi per non avere problemi con il convexhull

	altezza = len(img_rgb)
	larghezza = len(img_rgb[0])
	
	#scansione verso il basso		
	for i in xrange( 0, larghezza):
		for j in xrange( 0, altezza-1):
			r = img_rgb[j][i][0]
			g = img_rgb[j][i][1]
			b = img_rgb[j][i][2]
			r1 = img_rgb[j+1][i][0]
			g1 = img_rgb[j+1][i][1]
			b1 = img_rgb[j+1][i][2]
			if soglia_minima_grigio<=r<=soglia_massima_grigio and soglia_minima_grigio<=g<=soglia_massima_grigio and soglia_minima_grigio<=b<=soglia_massima_grigio and j!= altezza-1:
				if (r1>soglia_massima_grigio or g1>soglia_massima_grigio or b1>soglia_massima_grigio) :
					img_rgb[j+1][i][0] = 0
					img_rgb[j+1][i][1] = 0
					img_rgb[j+1][i][2] = 0
				
	#scansione verso l'alto
	for i in xrange( 0, larghezza):
		for j in xrange( altezza-1, -1,-1):
			if soglia_minima_grigio<=img_rgb[j][i][0]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[j][i][1]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[j][i][2]<=soglia_massima_grigio and j!=0:
				if (img_rgb[j-1][i][0]>soglia_massima_grigio or img_rgb[j-1][i][1]>soglia_massima_grigio or img_rgb[j-1][i][2]>soglia_massima_grigio):
					img_rgb[j-1][i][0] = 0
					img_rgb[j-1][i][1] = 0
					img_rgb[j-1][i][2] = 0		
					
		
	#scansione verso destra
	for i in xrange(0, altezza ): 
		for j in xrange(0, larghezza):
			if soglia_minima_grigio<=img_rgb[i][j][0]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[i][j][1]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[i][j][2]<=soglia_massima_grigio and j!= larghezza-1:
				if(img_rgb[i][j+1][0]>soglia_massima_grigio or img_rgb[i][j+1][1]>soglia_massima_grigio or img_rgb[i][j+1][2]>soglia_massima_grigio) :
					img_rgb[i][j+1][0] = 0
					img_rgb[i][j+1][1] = 0
					img_rgb[i][j+1][2] = 0
		
	
	#scansione verso sinistra
	for i in xrange( 0, altezza):
		for j in xrange( larghezza-1, -1,-1):
			if soglia_minima_grigio<=img_rgb[i][j][0]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[i][j][1]<=soglia_massima_grigio and soglia_minima_grigio<=img_rgb[i][j][2]<=soglia_massima_grigio and j!= 0:
				if(img_rgb[i][j-1][0]>soglia_massima_grigio or img_rgb[i][j-1][1]>soglia_massima_grigio or img_rgb[i][j-1][2]>soglia_massima_grigio) :
					img_rgb[i][j-1][0] = 0
					img_rgb[i][j-1][1] = 0
					img_rgb[i][j-1][2] = 0
			
	return img_rgb
					
def get_layout(metricMap, minVal, maxVal, rho, theta, thresholdHough, minLineLength, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation,filepath, cv2thresh=127, diagonali=True, m = 20, metodo_classificazione_celle = True):
	'''
	Genera il layout delle stanze
	INPUT: mappa metrica, i parametri di canny, hough, mean-shift, dbscan, distanza per clustering spaziale.
	OUTPUT: la lista di poligoni shapely che costituiscono le stanze, la lista di cluster corrispondenti, la lista estremi che contiene [minx,maxx,miny,maxy] e la lista di colori.
	'''
	#leggo l'immagine originale in scala di grigio e la sistemo con il thresholding
	img_rgb = cv2.imread(metricMap)
	img_ini = img_rgb.copy() #copio l'immagine
	
	#scansione dei bordi 
	#img_rgb = scan_bordo(img_rgb)
					
	# 127 per alcuni dati, 255 per altri
	ret,thresh1 = cv2.threshold(img_rgb,cv2thresh,255,cv2.THRESH_BINARY)
	
	#img_ini = thresh1.copy()
	
	
	#------------------CANNY E HOUGH PER TROVARE MURI-----------------------------------
	walls , canny = start_canny_ed_hough(thresh1,minVal,maxVal,rho,theta,thresholdHough,minLineLength,maxLineGap)
	

	#questo lo posso anche eliminare alla fine
	#----primo plot-->richiedo il disegno delle hough lines
	dsg.disegna_map(img_rgb,filepath = filepath )
	dsg.disegna_canny(canny,filepath = filepath)
	dsg.disegna_hough(img_rgb,walls,filepath = filepath)


	lines = flip_lines(walls, img_rgb.shape[0]-1)
	walls = crea_muri(lines)
	
	#anche questo lo posso eliminare
	#disegno i muri
	#----secondo plot --> disegna i muri corrispondenti ai segmenti individuati in precedenza
	#dsg.disegna_segmenti(walls)#solo un disegno poi lo elimino
	#-----------------------------------------------------------------------------------

	
	#------------SETTO XMIN YMIN XMAX YMAX DI walls-------------------------------------
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

	#-----------------------------------------------------------------------------------
	
	
	#---------------CONTORNO ESTERNO----------------------------------------------------
	(contours, vertici) = contorno_esterno(img_rgb, minVal, maxVal, xmin, xmax, ymin, ymax, metricMap,filepath = filepath, m = m)
	#-----------------------------------------------------------------------------------
	
	#---------------MEAN SHIFT PER TROVARE CLUSTER ANGOLARI-----------------------------
	(indici, walls, cluster_angolari) = cluster_ang(h, minOffset, walls, diagonali= diagonali) #probabilmente indici non serve veraente
	#-----------------------------------------------------------------------------------
	
	#---------------CLUSTER SPAZIALI----------------------------------------------------
	cluster_spaziali = cluster_spaz(minLateralSeparation, walls,filepath = filepath)
	#-----------------------------------------------------------------------------------
	
	#-------------------CREO EXTENDED_LINES---------------------------------------------
	(extended_lines, extended_segments) = extend_line(cluster_spaziali, walls, xmin, xmax, ymin, ymax,filepath = filepath)
	#-----------------------------------------------------------------------------------

	#-------------CREO GLI EDGES TRAMITE INTERSEZIONI TRA EXTENDED_LINES----------------
	edges = sg.crea_edges(extended_segments)
	#-----------------------------------------------------------------------------------
	
	#----------------------SETTO PESI DEGLI EDGES---------------------------------------
	#non so a cosa serva
	edges = sg.setPeso(edges, walls)
	#-----------------------------------------------------------------------------------
	
	#----------------CREO LE CELLE DAGLI EDGES------------------------------------------
	celle = fc.crea_celle(edges)
	#-----------------------------------------------------------------------------------
	
#----------------------------------problemi---------------------------------------------
	#TODO: questi due pezzi di codice hanno un problema, per ora non ho capito cosa. Ci sono gia' i metodi in fondo
	'''
	#questa non va
		
	#----------------CLASSIFICO CELLE---------------------------------------------------
	(celle, celle_out, celle_poligoni, indici, celle_parziali, contorno, centroid, punti) = classificazione_superfici(vertici, celle)
	#-----------------------------------------------------------------------------------
	
	#questo non va
	#--------------------------POLIGONI CELLE-------------------------------------------
	#TODO: questo medodo non e' di questa classe ma e' della classe Sapzio, in sostanza gli passi le celle e lui crea lo spazio che poi potro classificare come stanza o corridoio
	(celle_poligoni, out_poligoni, parz_poligoni) = crea_poligoni_da_celle(celle, celle_out, celle_parziali)
	#-----------------------------------------------------------------------------------
	'''
	
	#se voglio classifica le celle con il metodo di matteo metti TRUE, altrimenti se vuoi classificare con il metodo di Valerio con le percentuali metti False
	if metodo_classificazione_celle: 
		print "sono entrato qui dentro hahahahahah"
		#----------------CLASSIFICO CELLE---------------------------------------------------
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
			#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella e' fuori.
			if cella.intersects(contorno)==False:
				#indici.append(index)
				f.set_out(True)
				f.set_parziale(False)
				celle_out.append(f)
			#se il poligono della cella interseca il contorno esterno della mappa
			if (cella.intersects(contorno)):
				#se l'intersezione e' piu' grande di una soglia la cella e' interna
				if(cella.intersection(contorno).area >= cella.area/2):
					f.set_out(False)
				#altrimenti e' esterna
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
	else:
		print "uffffa ################################"
		#sto classificando le celle con il metodo delle percentuali
		(celle_out, celle, centroid, punti,celle_poligoni, indici, celle_parziali) = classifica_celle_con_percentuale(vertici, celle, img_ini)

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
#----------------------------fine problemi----------------------------------------------

	#------------------CREO LE MATRICI L, D, D^-1, ED M = D^-1 * L----------------------
	(matrice_l, matrice_d, matrice_d_inv, X) = crea_matrici(celle)
	#-----------------------------------------------------------------------------------
	
	#----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA----------------------
	colori, fig, ax, clustersCelle = DB_scan(eps, minPts, X, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = filepath)
	#-----------------------------------------------------------------------------------

	#------------------POLIGONI STANZE(spazio)------------------------------------------
	stanze, spazi = crea_spazio(clustersCelle, celle, celle_poligoni, colori, xmin, ymin, xmax, ymax, filepath = filepath) 
	#-----------------------------------------------------------------------------------

	return  (stanze, clustersCelle, estremi, colori, spazi)
	
	

def start_canny_ed_hough(thresh1,minVal,maxVal,rho,theta,thresholdHough,minLineLength,maxLineGap):
	'''
	CANNY ED HOUGH per trovare i muri. e' un meccanismo di edge detection.
	OPENCV VERSION PROBLEMS
	'''
	#canny
	cannyEdges = cv2.Canny(thresh1,minVal,maxVal,apertureSize = 5)
	#hough
	walls = cv2.HoughLinesP(cannyEdges,rho,theta,thresholdHough,minLineLength,maxLineGap)
	if cv2.__version__[0] == '3' :
		walls = [i[0]for i in walls]
	elif cv2.__version__[0] == '2' :
		walls = walls[0]
	else :
		raise EnvironmentError('Opencv Version Error. You should have OpenCv 2.* or 3.*')
	return walls ,cannyEdges
	
def flip_lines(linee, altezza):
	'''
	flippa le y delle linee, perche' l'origine dei pixel e' in alto a sx, invece la voglio in basso a sx
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

def crea_muri(linee):
	'''
	trasforma le linee in oggetti di tipo Segmento, e ne restituisce la lista.
	'''
	walls = []
	for muro in linee:
		x1 = float(muro[0])
		y1 = float(muro[1])
		x2 = float(muro[2])
		y2 = float(muro[3])
		walls.append(sg.Segmento(x1,y1,x2,y2))
	return walls

def contorno_esterno(img_rgb,minVal, maxVal, xmin, xmax, ymin, ymax, metricMap, filepath = '.',t=1, m=20):

	#creo il contorno esterno facendo prima canny sulla mappa metrica.
	cannyEdges = cv2.Canny(img_rgb,minVal,maxVal,apertureSize = 5)
	#t=1 #threshold di hough
	#m=20 #maxLineGap di hough
	hough_contorni, contours = im.trova_contorno(t,m,cannyEdges, metricMap)
	
	contours = flip_contorni(contours, img_rgb.shape[0]-1)
	
	#disegno contorno esterno
	#questo e' un grafico, lo posso anche eliminare, se non mi serve

	vertici = []
	for c1 in contours:
		for c2 in c1:
			vertici.append([float(c2[0][0]),float(c2[0][1])])

	#----terzo plot --> disegna il contorno esterno analizzando i vertici dell'immagine e le coordinate massime e minime
	#dsg.disegna_contorno(vertici,xmin,ymin,xmax,ymax,filepath = filepath)
		
	return (contours, vertici)

def cluster_ang(h, minOffset, walls, num_min = 3, lunghezza_min = 3, diagonali= True):
	'''
	crea i cluster angolari
	'''
	
	#creo i cluster centers tramite mean shift
	cluster_centers = ms.mean_shift(h, minOffset, walls)
	
	#ci sono dei cluster angolari che sono causati da pochi e piccoli line_segments, che sono solamente rumore. Questi cluster li elimino dalla lista cluster_centers ed elimino anche i rispettivi segmenti dalla walls.
	indici = ms.indici_da_eliminare(num_min, lunghezza_min, cluster_centers, walls, diagonali)
	
	#ora che ho gli indici di clusters angolari e di muri da eliminare, elimino da walls e cluster_centers, partendo dagli indici piu alti
	for i in sorted(indici, reverse=True):
		del walls[i]
		del cluster_centers[i]
		
	#ci sono dei cluster che si somigliano ma non combaciano per una differenza infinitesima, e non ho trovato parametri del mean shift che rendano il clustering piu' accurato di cosi', quindi faccio una media normalissima, tanto la differenza e' insignificante.
	unito = ms.unisci_cluster_simili(cluster_centers)
	while(unito):
		unito = ms.unisci_cluster_simili(cluster_centers)
		
	#assegno i cluster ai muri di walls
	walls = sg.assegna_cluster_angolare(walls, cluster_centers)
	
	#creo lista di cluster_angolari
	cluster_angolari = []
	for muro in walls:
		cluster_angolari.append(muro.cluster_angolare)

	return (indici, walls, cluster_angolari)

def cluster_spaz(minLateralSeparation, walls,filepath = '.'):
	'''
	crea i cluester spaziali
	'''
	#setto i cluster spaziali a tutti i muri di walls
	walls = sg.spatialClustering(minLateralSeparation, walls)
	
	#creo lista di cluster spaziali
	cluster_spaziali = []
	for muro in walls:
		cluster_spaziali.append(muro.cluster_spaziale)
		
	#disegno cluster spaziali(posso anche eliminarlo, a me non servono tanto i disegni)
	#---- quarto plot --> disegno i cluster spaziali relativi ai segmenti
	#dsg.disegna_cluster_spaziali(cluster_spaziali, walls,filepath = filepath)
	
	return cluster_spaziali

def extend_line(cluster_spaziali, walls, xmin, xmax, ymin, ymax,filepath='.'):
	'''
	crea le extended line
	'''
	extended_lines = rt.crea_extended_lines(cluster_spaziali, walls, xmin, ymin)
	
	#da qui serve per disegnarle
	#extended_lines (credo parli delle rette) hanno un punto, un cluster_angolare ed un cluster_spaziale, per disegnarle pero' mi servono 2 punti. Creo lista di segmenti
	extended_segments = ext.crea_extended_segments(xmin,xmax,ymin,ymax, extended_lines)

	#----quinto plot --> disegno la mappa come era in partenza(con le linee) e ci aggiungo le rette.
	#disegno le extended_lines in rosso e la mappa in nero
	#dsg.disegna_extended_segments(extended_segments, walls,filepath = filepath)
	
	return (extended_lines, extended_segments)

def classificazione_superfici(vertici, celle):
	'''
	classificazione delle celle(superfici) che sono state individuate dai segmenti estesi
	'''
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
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella e' fuori.
		if cella.intersects(contorno)==False:
			#indici.append(index)
			f.set_out(True)
			f.set_parziale(False)
			celle_out.append(f)
		#se il poligono della cella interseca il contorno esterno della mappa
		if (cella.intersects(contorno)):
			#se l'intersezione e' piu' grande di una soglia la cella e' interna
			if(cella.intersection(contorno).area >= cella.area/2):
				f.set_out(False)
			#altrimenti e' esterna
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

	return (celle, celle_out, celle_poligoni, indici, celle_parziali, contorno, centroid, punti)
	
def algo(p):
	'''
	riordina i punti in senso orario
	'''
	return (math.atan2(p[0] - centroid[0], p[1] - centroid[1]) + 2 * math.pi) % (2*math.pi)	
	
def sort_and_deduplicate(l):
	'''
elimina i doppioni da una lista di coppie
	'''
	return list(uniq(sorted(l, reverse=True)))

def uniq(lst):
	last = object()
	for item in lst:
		if item == last:
			continue
		yield item
		last = item
	
def crea_poligoni_da_celle(celle, celle_out, celle_parziali):
	'''
	crea i poligoni a partire dalle celle classificate, e' equivalente all'oggetto Spazio(non ancora creato), infatti appena riesci sposta questo metodo in quella classe.
	'''
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
		global centroid
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
		global centroid
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
		global centroid
		centroid = (sum(x) / len(punti), sum(y) / len(punti))
		punti.sort(key=algo)
		cella = Polygon(punti)
		parz_poligoni.append(cella)

	return (celle_poligoni, out_poligoni, parz_poligoni, centroid)
	
def crea_matrici(celle,sigma=0.1,val=0):
	

	matrice_l = mtx.crea_matrice_l(celle, sigma, val)

	matrice_d = mtx.crea_matrice_d(matrice_l)

	matrice_d_inv = matrice_d.getI()

	matrice_m = matrice_d_inv.dot(matrice_l)
	matrice_m = mtx.simmetrizza(matrice_m)

	X = 1-matrice_m
	
	return (matrice_l, matrice_d, matrice_d_inv, X)
	
def DB_scan(eps, minPts, X, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = '.'):
	clustersCelle = []
	clustersCelle = clustering_dbscan_celle(eps, minPts, X)

	#---- sesto plot --> disegno la mappa con le diverse celle classificate. 
	#colori, fig, ax = dsg.disegna_dbscan(clustersCelle, celle, celle_poligoni, xmin, ymin, xmax, ymax, edges, contours,filepath = filepath)
	'''
	#se lo voglio usare devo andare a commentare in disegna.py in disegna_dbscan lo #show() alla fine del metodo 
	#plotto le celle parziali
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.text(f_poly.representative_point().x,f_poly.representative_point().y,str("parz"),fontsize=8)
	plt.show()
	'''
	
	
	#return colori, fig, ax, clustersCelle
	return clustersCelle

def clustering_dbscan_celle(eps, min_samples, X):
	'''
	esegue il dbscan clustering sulle celle, prendendo in ingresso eps (massima distanza tra 2 campioni per essere considerati nello stesso neighborhood), min_samples (numero di campioni in un neighborhood per un punto per essere considerato un core point) e X (1-matrice di affinita' locale tra le celle).
	'''
	af = DBSCAN(eps, min_samples, metric="precomputed").fit(X)
	#print("num of clusters = ")
	#print len(set(af.labels_))	
	return af.labels_
	
def crea_spazio(clustersCelle, celle, celle_poligoni, colori, xmin, ymin, xmax, ymax,filepath = '.'):
	'''
	questo e' un metodo che andrebbe inserito nella classe Spazio, vedi se ha senso farlo
	'''
	#creo i poligoni delle stanze (unione dei poligoni delle celle con stesso cluster).
	stanze = []
	spazi = []
	stanze, spazi = unisciCelle(clustersCelle, celle, celle_poligoni, False)

	#---- settimo plot --> serve a disegnare il plot della mappa con le stanze separate.
	#disegno layout stanze.
	#dsg.disegna_stanze(stanze, colori, xmin, ymin, xmax, ymax,filepath = filepath)

	return stanze, spazi

def unisciCelle(clusters, celle, celle_poligoni, False):
	'''
	i poligoni delle celle dello stesso cluster vengono uniti in un unico poligono, che e' il poligono della stanza.
	'''
	
	#gli spazi sono la stessa cosa delle stanze ma in realta' ho creato un oggetto spazio
	stanze = []
	spazi = []

	for l in set(clusters):
		poligoni = []
		for index,cluster in enumerate(clusters):
			if (l == cluster) and not (celle[index].out):
				poligoni.append(celle_poligoni[index])
		stanza = cascaded_union(poligoni)
		stanze.append(stanza)
		spazio = sp.Spazio(poligoni, stanza, index)#mio(oggetto Spazio)
		spazi.append(spazio) #mio
		
		
	return stanze, spazi
	
	
	
def get_layout_parziale(metricMap, minVal, maxVal, rho, theta, thresholdHough, minLineLength, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation, diagonali=True):
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

	if cv2.__version__[0] == '3' :
		walls = [i[0]for i in walls]
	elif cv2.__version__[0] == '2' :
		walls = walls[0]
	else :
		raise EnvironmentError('Opencv Version Error. You should have OpenCv 2.* or 3.*')

	dsg.disegna_hough(img_rgb,walls)

	lines = flip_lines(walls, img_rgb.shape[0]-1)

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
	indici = ms.indici_da_eliminare(num_min, lunghezza_min, cluster_centers, walls, diagonali)


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
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella e' fuori.
		if cella.intersects(contorno)==False:
			#indici.append(index)
			f.set_out(True)
			f.set_parziale(False)
			celle_out.append(f)
		#se il poligono della cella interseca il contorno esterno della mappa
		if (cella.intersects(contorno)):
			#se l'intersezione e' piu' grande di una soglia la cella e' interna
			if(cella.intersection(contorno).area >= cella.area/2):
				f.set_out(False)
			#altrimenti e' esterna
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
					if(fc.edge_comune(f,f2)[0].weight < 0.1):  #qua nella funzione normale c'e' 0.2
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
	#plotto le celle parziali, questo per funzionare ha bisogno che si commenti riga 127 di disegna.py (#plt.show()).
	
	for f_poly in parz_poligoni:
		f_patch = PolygonPatch(f_poly,fc='#d3d3d3',ec='BLACK')
		ax.add_patch(f_patch)
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
	plt.show()
	
	return (stanze, clustersCelle, estremi, colori)
	
def classifica_celle_con_percentuale(vertici, celle, img_ini):
	#----------------CLASSIFICO CELLE---------------------------------------------------
	#creo poligono del contorno
	contorno = Polygon(vertici)
	
	#secondo me tutta sta roba la si puo' elimminare dato che tanto dopo si fanno le stesse cose
	#TODO guarda se si puo' unire il pezzo dopo con questo.
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
		#TODO:dopo averli ordinati in senso orario, creo il poligono della cella.#VALERIO: a me sembrano salvato in senso antiorario, ho provato a printarlo ed effetivamente prima erano in senso orario e poi in senso antiorario
		cella = Polygon(punti)
		#se il poligono della cella non interseca quello del contorno esterno della mappa, la cella e' fuori.
		
		#VALERIO:tento di classificare le celle out che pero' stanno all'interno del convexHull, come out.
		#idea: prendo la posizione di tutte le celle rimaste e controllo che all'interno della mappa originaria non ci siano parti grigie.

		bordo = cella.bounds #restituisce una tupla del tipo (minx, miny, maxx, maxy)
		
		#bordo[1]:bordo[3], bordo[0]:bordo[2]
		#il problema principale in questo punto e' che ci sono ancora le righe rosse e non so il perche'
		#plt.close("all")
		
		#TODO: problema risolto, non so perche' ma l'immagine era capovolta, quindi andavo a prendere celle strane
		#-------flippo immagine, penso sia quello il problema principale
		#plt.imshow(img_ini)
		#plt.show()
	
		img_flipt= cv2.flip(img_ini,0)#questo secondo me lo devo solo fare con le mappe del survey
		#plt.imshow(img_flipt)
		#plt.show()
		
		#se sono sul quadrante in alto a destra
		if bordo[1]>=0 and bordo[0]>=0:
			immagine_cella = img_flipt[bordo[1]:bordo[3], bordo[0]:bordo[2]]	
			#plt.imshow(immagine_cella)
			#plt.show()
			altezza = len(immagine_cella)
			larghezza = len(immagine_cella[0])
			total_count_pixel = 0
			pixel_bianco = 0
			pixel_grigio = 0
			pixel_nero = 0
			for i in xrange(0, altezza):
				for j in xrange( 0, larghezza):
					r = immagine_cella[i][j][0]
					g = immagine_cella[i][j][1]
					b = immagine_cella[i][j][2]
					total_count_pixel += 1
					if (r<=210 and g<=210 and b<=210)and(r>=70 and g>=70 and b>=70):
						pixel_grigio += 1
					elif r<70 and g<70 and b<70:
						pixel_nero += 1
					else:
						pixel_bianco += 1
		
			if total_count_pixel != 0:
				percentuale_bianco = (float(pixel_bianco)/float(total_count_pixel))
				percentuale_grigio = (float(pixel_grigio)/float(total_count_pixel))
				percentuale_nero =  (float(pixel_nero)/float(total_count_pixel))
		
				if percentuale_grigio >=0.3:
					f.set_out(True)
					celle_out.append(f)
					
				elif percentuale_bianco <= 0.5:
					f.set_out(True)
					celle_out.append(f)
				else:
					f.set_out(False)
				
			else:
				f.set_out(True)
				celle_out.append(f)
		else:
			#TODO: capire perche'. Non ne conosco il motivo ma ci sono delle parti sotto il primo quadrante
			f.set_out(True)
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
	
	
	return celle_out, celle, centroid, punti,celle_poligoni, indici, celle_parziali