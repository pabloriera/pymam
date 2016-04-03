# -*- coding: utf-8 -*-


def checkmodule(name):
    try:
        __import__(name)
        return True
    except:
        print "No esta instalado "+name
        
        return False

checkmodule("music21")
checkmodule("pygame")
checkmodule("sounddevice")
checkmodule("scipy")
checkmodule("numpy")

if 'np' not in locals():
    print "No se puso el archivo de inicio scientific.py"



if( checkmodule("pymam") and checkmodule("pygame")):

    print "MIDI TEST"
    
    from pymam import *    
     
    notas = [  69,   74,   78,   74,   69]
    durs  = [0.5, 0.25, 0.5, 0.25, 0.25]
        
    score = playsequence(notas,durs)


if( checkmodule("pymam")):    
    print "AUDIO TEST"    
    
    from pymam import * 
    import pylab as pl
    
    sound(pl.randn(10000)*0.05,44100)





