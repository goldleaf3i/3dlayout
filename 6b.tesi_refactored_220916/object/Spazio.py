'''
spazio si intende un ambiente composto da piu' superfici alla quale e' possibile associare un significato diverso a seconda della forma e della dimensione che possiede.
Uno spazio infatti puo' essere distinto tra corridoio o stanza
'''
from shapely.ops import cascaded_union


class Spazio(object):
	def __init__(self, cells, stanza, id):
		self.cells = cells
		self.spazio = stanza
		self.id = id
	def set_out(self,o):
		self.out = o
	def set_parziale(self,p):
		self.parziale = p
	def set_tipo(self,t):
		self.tipo = t
	def set_id(self, id):
		self.id = id
	def add_cell(self, cell):
		self.cell.append(cell)

def crea_spazio():
	#invece che mettere stanza nel costruttore potrei creare qui il poligono con tutte le celle che ci sono
	a= 0
	return a