from minisom import MiniSom
import pickle

INFILE = "test1"

with open('./som_files/' + INFILE + '.p', 'rb') as infile:
    som = pickle.load(infile)


## Visu
print(som.win_map())