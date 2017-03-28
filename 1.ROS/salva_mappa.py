#script che chiama un thread ogni tot secondi. Questo thread esegue un comando da terminale, salvando la mappa. Riceve come argomento il nome della mappa. 

import os
import threading
import sys

nome_mappa = str(sys.argv[1]) 
i = 0
step_in_secondi = 5.0

def salva_mappa():
  threading.Timer(step_in_secondi, salva_mappa).start()
  os.system("rosrun map_server map_saver -f ./mappe_salvate/" + nome_mappa + str(i))
  global i
  i+=1
salva_mappa()

