# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt


#--------------------MAIN-------------------------------------------------------------


#-------------------PARAMETRI------------------------------------------------------

#distanza massima in pixel per cui 2 segmenti con stesso cluster angolare sono considerati appartenenti anche allo stesso cluster spaziale 
minLateralSeparation = 8
#parametri di Canny
minVal = 90
maxVal = 100
#parametri di Hough
rho = 1
theta = np.pi/180
thresholdHough = 20
minLineLenght = 7
maxLineGap = 3
#parametri di DBSCAN
eps = 0.85
minPts = 1
#parametri di mean-shift
h = 0.023
minOffset = 0.00001

#xml ground truth corrispondente
nome_gt = '/home/matteo/Desktop/Dataset sistemati/Valleceppi_P1_updated.xml'
#xml creato da me, relativo alle stanze segmentate
nomeXML = "./Input/miei/b.xml"
#dataset usato per le label semantiche
dataset_name = 'school'


#-----------------------------MAPPA METRICA--------------------------------

#mappa metrica
metricMap = '/home/matteo/Desktop/input_mura/g.png'



#----------------------------LAYOUT DELLE STANZE---------------------------

#layout delle stanze
import Layout as lay
(stanze, clustersCelle, estremi, colori) = lay.get_layout_parziale(metricMap, minVal, maxVal, rho, theta, thresholdHough, minLineLenght, maxLineGap, eps, minPts, h, minOffset, minLateralSeparation)


#--------------------------GRAFO TOPOLOGICO-------------------------------

#grafo topologico
import GrafoTopologico as gtop
(G, pos, collegate, doorsVertices) = gtop.get_grafo(metricMap, stanze, estremi, colori)


#-------------------------MAPPA SEMANTICA------------------------------

import MappaSemantica as sema
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

	
