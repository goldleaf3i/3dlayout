#script che prende come argomento il nome dell'xml, e ne crea un'immagine. Questa immagine andra' salvata senza modificarne la dimensione, e poi andra' tagliata facendo autocrop su pinta.

import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

dataset_name = './Dataset sistemati/Valleceppi_P1_updated.xml'
tree = ET.parse(dataset_name)
root = tree.getroot()

muri = list()
ascisse = list()
ordinate = list()
lines = list()

#raccolgo nella lista stanze tutte le stanze, ovvero i tag space
stanze = root.findall("./floor/spaces/space")

for stanza in stanze:
	#raccolgo nella lista muri tutti i segmenti/muri della stanza corrente
	muri = stanza.findall("./space_representation/linesegment[class='WALL']")	
	for muro in muri:
		for punto in muro.findall("./point"):
			ascisse.append(punto.get('x'))
			ordinate.append(punto.get('y'))
		plt.plot(ascisse,ordinate, color='k', linewidth=2.0)
		del ascisse[:]
		del ordinate[:]

plt.axis('off')
plt.show()


