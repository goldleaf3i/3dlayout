from __future__ import division
import numpy as np
import math
from matplotlib import pyplot as plt

class Extended_segment(object):
	def __init__(self, punto1, punto2, cluster_angolare, cluster_spaziale):
		self.x1=punto1[0]
		self.y1=punto1[1]
		self.x2=punto2[0]
		self.y2=punto2[1]
		self.cluster_angolare = cluster_angolare
		self.cluster_spaziale = cluster_spaziale


def crea_extended_segments(xmin, xmax, ymin, ymax, extended_lines):
	'''
	crea un extended_segment per ogni extended_line, come intersezione tra la extended_line e il bounding box
	'''
	extended_seg = []
	for e in extended_lines:
		if e.cluster_angolare == 0: #orizzontale
			point2 = np.array([xmax,e.punto[1]])
			seg = Extended_segment(e.punto,point2,e.cluster_angolare,e.cluster_spaziale)
			extended_seg.append(seg)
		else:		
			if e.cluster_angolare == math.radians(90): #verticale
				point2 = np.array([e.punto[0],ymax])
				seg = Extended_segment(e.punto,point2,e.cluster_angolare,e.cluster_spaziale)
				extended_seg.append(seg)
			else:
				m = math.tan(e.cluster_angolare)
				q = e.punto[1] - (m*e.punto[0])
				y_for_xmin = m*(xmin) + q
				y_for_xmax = m*(xmax) + q
				x_for_ymin = ((ymin)-q)/m
				x_for_ymax = ((ymax)-q)/m
				if e.cluster_angolare < 0 or (e.cluster_angolare)*180/math.pi > 90 :
					if(ymax < y_for_xmin):
						point1 = np.array([x_for_ymax,ymax])
					else:
						point1 = np.array([xmin,y_for_xmin])
					if(ymin > y_for_xmax):
						point2 = np.array([x_for_ymin,ymin])
					else:
						point2 = np.array([xmax,y_for_xmax])
				else:
					if(y_for_xmin > ymin):
						point1 = np.array([xmin,y_for_xmin])
					else:
						point1 = np.array([x_for_ymin,ymin])
					if(y_for_xmax < ymax):
						point2 = np.array([xmax,y_for_xmax])
					else:
						point2 = np.array([x_for_ymax,ymax])
				seg = Extended_segment(point1, point2, e.cluster_angolare, e.cluster_spaziale)
				extended_seg.append(seg)
	point1 = np.array([xmin,ymin])
	point2 = np.array([xmin,ymax])
	point3 = np.array([xmax,ymax])
	point4 = np.array([xmax,ymin])
	seg1 = Extended_segment(point1, point2, None, None)
	seg2 = Extended_segment(point2, point3, None, None)
	seg3 = Extended_segment(point4, point3, None, None)
	seg4 = Extended_segment(point1, point4, None, None)
	extended_seg.extend((seg1,seg2,seg3,seg4))				
	return extended_seg


def disegna_extended_segments(extended_segments, lista_muri):
	'''
	disegna in nero i muri di lista_muri, in rosso gli extended_segments
	'''
	ascisse = []
	ordinate = []
	#plt.subplot(223)
	plt.title('extended segments')
	for e in lista_muri:		
		ascisse.append(e.x1)
		ascisse.append(e.x2)
		ordinate.append(e.y1)
		ordinate.append(e.y2)
		plt.plot(ascisse,ordinate, color='k', linewidth=3.0)
		del ascisse[:]
		del ordinate[:]
	
	for e in extended_segments:		
		ascisse.append(e.x1)
		ascisse.append(e.x2)
		ordinate.append(e.y1)
		ordinate.append(e.y2)
		plt.plot(ascisse,ordinate, color='r', linewidth=1.5)
		del ascisse[:]
		del ordinate[:]

	plt.show()
''' 
per disegnarli uno alla volta
	ascisse = []
	ordinate = []
	#plt.subplot(223)
	plt.title('extended segments')
	for e in lista_muri:
		ascisse.append(e.x1)
		ascisse.append(e.x2)
		ordinate.append(e.y1)
		ordinate.append(e.y2)
		plt.plot(ascisse,ordinate, color='k', linewidth=3.0)
		del ascisse[:]
		del ordinate[:]
		for e2 in lista_muri:
			if e != e2 and e.cluster_spaziale == e2.cluster_spaziale:		
				ascisse.append(e2.x1)
				ascisse.append(e2.x2)
				ordinate.append(e2.y1)
				ordinate.append(e2.y2)
				plt.plot(ascisse,ordinate, color='k', linewidth=3.0)
				del ascisse[:]
				del ordinate[:]
	
		for e3 in extended_segments:
			if e.cluster_spaziale == e3.cluster_spaziale:		
				ascisse.append(e3.x1)
				ascisse.append(e3.x2)
				ordinate.append(e3.y1)
				ordinate.append(e3.y2)
				plt.plot(ascisse,ordinate, color='r', linewidth=1.5)
				del ascisse[:]
				del ordinate[:]

		plt.show()
'''
