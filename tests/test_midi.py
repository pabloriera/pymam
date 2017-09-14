from pymam import *
 
notas = [  69,   74,   78,   74,   69]
durs  = [0.5, 0.25, 0.5, 0.25, 0.25]
 
score = playsequence(notas,durs)
midiwrite(score,"prueba.mid")

rtp = midi.midi.realtime.StreamPlayer(score)
