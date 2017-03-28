import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.cbook as cbook
import numpy as np
import math
import cmath
import glob
from myDictionaries import *
from xml.dom.minidom import parse
import numpy.random as rnd
from matplotlib.patches import Ellipse
from myDictionaries import labels_java2012toMatlab_Dict


# STUPIDO SCIKIT
import warnings
warnings.filterwarnings("ignore")

def print_matrix(M) :
	[r,c] = M.shape
	for i in xrange(r) :
		line = str()
		for j in M[i]:
			line += "%.3f"%j + ', '
			#line+= str("{0:2f}".format(M[i,j]))+' ,'
		line = line[:-2]
		print line


def label_to_number(label) :
	if label == 'R' :
		return 0
	elif label == 'C' :
		return 1
	elif label == 'E' :
		return 0
	else :
		return -1


def number_to_label(label) :
	if label == 0 :
		return 'R'
	elif label == 1 :
		return 'C'
	else :
		return -1


def building_to_number(building) :
	if building == 'school' :
		return 0
	elif building == 'office' :
		return 1
	elif building == 'fake' :
		return 2
	else :
		return -1


def number_to_building(building) :
	if label == 0 :
		return 'school'
	elif label == 1 :
		return 'office'
	elif label == 2 :
		return 'fake'
	else :
		return -1


def get_label_dict(buildingtype = 'school'):
	labelxml = buildingtype+'.xml'
	xmldoc = parse(labelxml)
	labels = {}
	letters = {}
	nodeLabels = xmldoc.getElementsByTagName("label")
	counter = 1;
	for nodeLabel in nodeLabels:
		name = nodeLabel.getElementsByTagName("name")[0].childNodes[0].nodeValue
		letter = nodeLabel.getElementsByTagName("letter")[0].childNodes[0].nodeValue
		function = nodeLabel.getElementsByTagName("function")[0].childNodes[0].nodeValue
		RC = nodeLabel.getElementsByTagName("type")[0].childNodes[0].nodeValue
		labels[name] = letter
		letters[name] = { 
				'letter' : letter , 
				'color': Java2012_colorDict[letter], 
				'number' : labels_java2012toMatlab_Dict[letter],
				#'RC' : labels_RC_java2012[letter],
				#'ellipse' : Ellipse(xy=[0.7,counter*0.7], width=0.6, height=0.6,angle=0),
				'counter' : counter,
				'RC' : RC if RC != u'E' else u'R',
				'function' : function if function != u'F' else u'R',
				'RCO' : function if function == u'F' or function == u'C' else 'O',
				'namekey' : name 
		}
		counter+=1;
	return letters


def get_features_srl(dataset_name = 'school') :
	counter = 0
	space_labels = {}
	labels = []
	portal_tuple = []
	buildings_dict = dict()
	for xml_file in glob.glob('ClassifierInputs/XMLs/'+dataset_name+'/*.xml'):
		if counter != 0 :
			print "Start parsing files."
			#break
		else :
			counter +=1
		print "#"*50
		print xml_file
		xml_name = xml_file[6:]
		print xml_name
		tree = ET.parse(xml_file)
		root = tree.getroot()
		# assumendo che la root sia sempre <building>
		floor_id = root.attrib.get('id')
		#	buildings_dict[floor_id] = [] 
		floor = root.find('floor')
		spaces = floor.find('spaces')
		pixels = int(root.find('scale').find('represented_distance').find('value').text)
		portals = root.find('portal')
		labels = list(set(labels))
		rooms = dict()
		for space in spaces.iter('space'):
			space_labels[space.get('id')] = space.find('labels').find('label').text
			#	buildings_dict['floor_id'].append(space.get('id'))
			space_dict = dict()
			#	space_dict['floor'] = floor_id
			space_dict['label'] = space.find('labels').find('label').text
			space_dict['connections'] = []
			labels.append(space.find('labels').find('label').text)
			portals = space.find('portals')

			# append features 
			features_xml = space.find('features')
			area = features_xml.find('area').get('value')
			space_dict['area'] = area
			perimeter = features_xml.find('perimeter').get('value')
			space_dict['perimeter'] = perimeter
			aoverp = features_xml.find('aoverp').get('value')
			space_dict['aoverp'] = aoverp
			adcs = features_xml.find('adcs').get('value')
			space_dict['adcs'] = adcs
			ff = features_xml.find('ff').get('value')
			space_dict['ff'] = ff
			circularity = features_xml.find('circularity').get('value')
			space_dict['circularity'] = circularity
			normalcirc = features_xml.find('normalcirc').get('value')
			space_dict['normalcirc'] = normalcirc
			andcs = features_xml.find('andcs').get('value')
			space_dict['andcs'] = andcs
			# Bulding type
			space_dict['building'] = dataset_name

			for portal in portals.iter('portal') :
				tmp = tuple([i.text for i in portal.find('target').findall('id')])
				if tmp[1] != space.get('id') :
					space_dict['connections'].append(tmp[1])
				elif tmp[0] != space.get('id') :
					space_dict['connections'].append(tmp[0])
				else :
					print 'error!'
					exit()

				if not ((tmp[0],tmp[1]) in portal_tuple or (tmp[1],tmp[0]) in portal_tuple) :
					portal_tuple.append(tmp)

			rooms[space.get('id')] = space_dict

		for i in rooms.keys() :
			neigh_labels = []
			for j in rooms[i]['connections'] :
				neigh_labels.append(rooms[j]['label'])
			rooms[i]['neigh'] = neigh_labels

		buildings_dict[floor_id] = rooms
		
	return buildings_dict

def get_labels_reverse_dict(legend, field, value) :
	# mando la legenda di tutti e restituisco solo una slice con chiave
	unique_values = []
	for i in legend.keys() :
		if legend[i][field] == value :
			unique_values.append(i)
	return unique_values
