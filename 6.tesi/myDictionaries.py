#!/usr/bin/python

# COPIATO DA CARTELLLA DI SVILUPPO 15/9/14

################################################# INIZIO FORMATO salvataggi da file JAVA2012 (MapCREATOR) #################################################
# UFFICI - mappe in Java2012
java_2012_uffici ='''WALLLEFTRIGHT( Colours.WHITE , true, "WallVertical", "_", "_"),
DOORUPDOWN( Colours.WHITE, true, "DoorHorizontal", "_", "_"), 
DOORLEFTRIGHT(Colours.WHITE, true, "DoorVertical", "_", "_"), 
CORRIDOR( Colours.GREEN, false, "Corridor", "C", "C" ), 
SERVICEROOM (Colours.YELLOW, false, "Service", "S", "S"), 
OFFICE( Colours.ORANGE, false, "Office", "M", "M"),
GROUPOFFICE( Colours.RED, false, "GroupOffice", "B", "B"),
OPENSPACE( Colours.RED2, false, "Openspace", "O", "H"), 
MEETINGROOM( Colours.RED3, false, "MeetingRoom", "R", "B"),
CONFERENCEROOM( Colours.ORANGE2, false, "ConferenceRoom", "F", "H"),
HALL( Colours.LIGHTBLUE, false, "Hall", "H", "H"), 
CANTEEN( Colours.LIGHTBLUE2, false, "Canteen", "N", "H"),
DELUXEOFFICE( Colours.PURPLE, false, "DeluxeOffice", "D", "B"),
STAFFROOM( Colours.GREY2, false, "Staff", "K", "M"),'''

# SCUOLE - mappe in Java2012
java_2012_scuole ='''  EMPTY( Colours.WHITE , false, "Empty", "_", "_"), 
  WALLUPDOWN( Colours.WHITE , true, "WallHorizontal", "_", "_"), 
  WALLCORNERUPR( Colours.WHITE , true,"WallCornerUpRight", "_", "_"),
  WALLCORNERUPL( Colours.WHITE , true, "WallCornerUpLeft", "_", "_"),  
  WALLCORNERDOWNR( Colours.WHITE , true, "WallCornerDownRight", "_", "_"), 
  WALLCORNERDOWNL( Colours.WHITE , true, "WallCornerDownLeft", "_", "_"), 
  WALLLEFTRIGHT( Colours.WHITE , true, "WallVertical", "_", "_"),
  DOORUPDOWN( Colours.WHITE, true, "DoorHorizontal", "_", "_"), 
  DOORLEFTRIGHT(Colours.WHITE, true, "DoorVertical", "_", "_"), 
  CORRIDOR( Colours.GREEN, false, "Corridor", "C", "C" ), 
  SERVICEROOM (Colours.YELLOW, false, "Service", "S", "S"), 
  CLASSROOM( Colours.ORANGE, false, "Class", "M", "M"),
  LAB( Colours.RED, false, "Lab", "B", "B"),
  AUDITORIUM( Colours.RED2, false, "Auditorium", "A", "H"), 
  STAFF( Colours.ORANGE2, false, "Staff", "P", "M"),
  HALL( Colours.LIGHTBLUE, false, "Hall", "H", "H"), 
  CANTEEN( Colours.LIGHTBLUE2, false, "Canteen", "N", "H"),
  GYM( Colours.PURPLE, false, "Gym", "G", "H"),
  OFFICE( Colours.GREY2, false, "Office", "K", "S"),
  ENTRANCE (Colours.GREY, false, "Entrance", "E", "E");
'''

# ABITAZIONI - mappe in Java2012
java_2012_abitazioni =''' EMPTY( Colours.WHITE , false, "Empty", "_", "_"), 
  WALLUPDOWN( Colours.WHITE , true, "WallHorizontal", "_", "_"), 
  WALLCORNERUPR( Colours.WHITE , true,"WallCornerUpRight", "_", "_"),
  WALLCORNERUPL( Colours.WHITE , true, "WallCornerUpLeft", "_", "_"),  
  WALLCORNERDOWNR( Colours.WHITE , true, "WallCornerDownRight", "_", "_"), 
  WALLCORNERDOWNL( Colours.WHITE , true, "WallCornerDownLeft", "_", "_"), 
  WALLLEFTRIGHT( Colours.WHITE , true, "WallVertical", "_", "_"),
  DOORUPDOWN( Colours.WHITE, true, "DoorHorizontal", "_", "_"), 
  DOORLEFTRIGHT(Colours.WHITE, true, "DoorVertical", "_", "_"), 
  CORRIDOR( Colours.GREEN, false, "Corridor", "C", "C" ), 
  SQCORRIDOR( Colours.GREEN2, false, "SquareCorridor", "C2", "C"),
  SMALLROOM( Colours.YELLOW, false, "SmallRoom", "S", "S"), 
  HALL( Colours.LIGHTBLUE, false, "Hall", "H", "H"), 
  BIGROOM( Colours.RED, false, "BigRoom", "B", "B"),
  MEDIUMROOM( Colours.ORANGE, false, "MediumRoom", "M", "M"),
  ENTRANCE (Colours.GREY, false, "Entrance", "E", "E");
'''

#C - H - E - S - M - B - K- Y - (R-F-N)

# codice colore per le label di java2012
Java2012_colorDict = {
'C':'#F18705',
'S':'#3FBF04',
'H':'#BF5719',
'B':'#2267F2',
'M':'#2A90F2',
'E':'#268080',
'R':'#3FBF04',#
'F':'#BEB615',
'N':'#B27ABE',
'D':'#716D0D',
'O':'#A81E1E',
'K':'#282828',
"|":"#FF0DFF",
'X':'#BDBDBD',
'P':'#955CA4',#
'Y':'#8C8C8B',
'I':'#A5A5BE',#
'L':'#F24738',
'Z':'#585859',
'G':'#A81E1E',
'Q':'#5BBEBE',
'T':'#FFF87D',
'W':'#E8CAA7',
'A':'#3C3C3C',#
'J':'#BEB94E',
'U':'#5F1978',
'V':'#FF0DFF'
}
# mapping da label a R (room) o C (corridor) come da paper di IAS
# openspace va mappato ad un altra cosa (va contestualizato in quanto e' un caso particolare di stanza di collegamento / funzionale
#  secondo me prevale funzionale)
# e anche Empty ma questo per uan questione di analisi e non semantica (Empty e' una stanza di tipo collegamento)
labels_RC_java2012 = {
'C': 'C',
'C2': 'C',
'S': 'R',
'H':'C',
'B':'R',
'M':'R',
'E':'C',
'R':'R',
'F':'R',
'N':'R',
'D':'R',
'O':'R',
'K':'R',
"|":'R',
'X':'R',
'P':'R',
'Y':'R',
'I':'R',
'L':'C',
'Z':'R',
'G':'R',
'Q':'R',
'T':'R',
'W':'R',
'A':'R',
'J':'R',
'U':'R',
'V':'C'
}
################################################# FINE FORMATO SALVATAGGI JAVA-2012 (MapCreator) #################################################

################################################# INIZIO FORMATO SALVATAGGI DA JAVA2012 -> MATLAB-ADIACENZE (spectralGraph) #################################################
# le matrici di adiacenza hanno queste label sulla diagonale.
# lo stesso formato e' usato anche su gSpan
labels_java2012toMatlab_Dict = {
'C': 100,
'S': 2,
'H':105,
'B':4,
'M':3,
'E':1000,
'R':5,
'F':6,
'N':7,
'D':8,
'O':1,
'K':21 ,
"|":10000,
'X':12,
'P':9,
'Y':10,
'I':11,
'L':110,
'Z':13,
'G':14,
'Q':15,
'T':16,
'W':17,
'A':18,
'J':19,
'U':20,
'V':115
}

 # priorita per le connessioni.
labelsPriority_java2012toMatlab_Dict = {
'C': 1,
'S': 4,
'H': 1,
'B': 2,
'M': 3,
'E': 1,
'R': 3,
'F': 2,
'N': 1,
'D': 2,
'O': 3,
'K': 3 ,
"|": 10000,
'X': 1,
'P': 3,
'Y': 3,
'I': 3,
'L': 1,
'Z': 3,
'G': 3,
'Q': 3,
'T': 3,
'W': 1,
'A': 3,
'J': 3,
'U': 3
}
################################################# INIZIO FORMATO SALVATAGGI DA JAVA2012 -> MATLAB-ADIACENZE (spectralGraph) #################################################


