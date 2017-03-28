import numpy as np
import xml.etree.cElementTree as ET

#-------------------------------
#AZIONE = 'batch'
AZIONE = 'mappa_singola'

#-------------------------------
DISEGNA = True

#-------------------------------
#metodo di classificazione per le celle
#TRUE = Metodo di Matteo
#False = Metodo di Valerio
metodo_classificazione_celle = True
#metodo_classificazione_celle = False
#-------------------------------
#se voglio caricare i pickle basta caricare il main e quindi LOADMAIN = True, se voglio rifare tutto basta settare LOADMAIN = False
#LOADMAIN = True
LOADMAIN = False


class Path_obj():
	def __init__(self):
		#path output pickle
		#self.outLayout_pickle_path = './data/OUTPUT/pickle/layout_stanze.pkl'
		#self.outTopologicalGraph_pickle_path = './data/OUTPUT/pickle/grafo_topologico.pkl'

		#self.out_pickle_path = './data/OUTPUT/pickle/'
		#dataset input folder
		self.INFOLDERS = './data/INPUT/'
		self.OUTFOLDERS = './data/OUTPUT/'
		self.DATASETs =['SCHOOL']
 		#dataset output folder
		self.data_out_path = './data/OUTPUT/'
		#-----------------------------MAPPA METRICA--------------------------------
		#mappa metrica di default
		self.metricMap = './data/INPUT/IMGs/SURVEY/Freiburg79_scan.png'	
		#----------------------------NOMI FILE DI INPUT----------------------------
		#xml ground truth corrispondente di default
		#nome_gt = './data/INPUT/XMLs/SCHOOL/cunningham2f_updated.xml'
		self.nome_gt = './data/INPUT/XMLs/SURVEY/Freiburg79_scan.xml'
		#CARTELLA DOVE SALVO
		self.filepath = './'
		self.filepath_pickle_layout = './Layout.pkl'
		self.filepath_pickle_grafoTopologico = './GrafoTopologico.pkl'


class Parameter_obj():
	def __init__(self):
		#distanza massima in pixel per cui 2 segmenti con stesso cluster angolare sono considerati appartenenti anche allo stesso cluster spaziale 
		self.minLateralSeparation = 10
		self.cv2thresh = 220

		#parametri di Canny
		self.minVal = 90
		self.maxVal = 100

		#parametri di Hough
		self.rho = 1
		self.theta = np.pi/180
		self.thresholdHough = 20
		self.minLineLength = 7
		self.maxLineGap = 3

		#parametri di DBSCAN
		self.eps = 0.85#1.5#0.85
		self.minPts = 1

		#parametri di mean-shift
		self.h = 0.023
		self.minOffset = 0.00001
	
		#diagonali
		self.diagonali = True
	
		#maxLineGap di hough
		self.m = 20
		
		#flip_dataset = False #questo lo metti a true se la mappa che stai guardando e' di SURVEY 
		self.flip_dataset=True

def to_XML(parameter_obj):
	
	root = ET.Element("root")
	par = ET.SubElement(root, "parametri")
	
	ET.SubElement(par, "p1", name="minLateralSeparation").text = parameter_obj.minLateralSeparation
	ET.SubElement(par, "p2", name="cv2thresh").text = "some value1"
	ET.SubElement(par, "p3", name="minVal").text = "some value1"
	ET.SubElement(par, "p4", name="maxVal").text = "some value1"
	ET.SubElement(par, "p5", name="theta").text = "some value1"
	ET.SubElement(par, "p6", name="thresholdHough").text = "some value1"
	ET.SubElement(par, "p7", name="maxLineGap").text = "some value1"
	ET.SubElement(par, "p8", name="minPts").text = "some value1"
	ET.SubElement(par, "p9", name="h").text = "some value1"
	ET.SubElement(par, "p10", name="minOffset").text = "some value1"
	ET.SubElement(par, "p11", name="diagonali").text = "some value1"
	ET.SubElement(par, "p12", name="m").text = "some value1"
	ET.SubElement(par, "p13", name="flip_dataset").text = "some value1"

	

	tree = ET.ElementTree(root)
	tree.write("./parametri.xml")