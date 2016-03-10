# -*- coding: utf-8 -*-
"""
IMPORT SOUND MODULES FOR LINUX

sound(x,rate)
play sound from x array at given sampling rate


soundsc(x,rate)
play scaled sound from x array at given sampling rate

"""

import numpy as np
import warnings
#print "MAM SOUND LINUX"

try:
    from _scikits.audiolab import play
    have_audiolab = True
except ImportError:
    have_audiolab = False

if not have_audiolab:
    try:
        from sounddevice import play
        have_sd = True
    except ImportError:
        have_sd = False

def sound(x, rate=44100):
    
    if x.ndim==1:
        ch = 1
    elif x.ndim==2:
        if 1 in x.shape:
            x = x.flatten()
            ch = 1
        elif x.shape[0]==2:
            ch = 2
            x = x.T
        elif x.shape[1]==2:
            ch = 2
             
    elif x.ndim>2 or x.shape[1]>2:
        print 'Error: 1-D array for mono, or 2-D array where rows should be the number of channels, 1 (mono) or 2 (stereo)'
        return
        
    if have_audiolab or have_sd:
        
        if x.ndim==2 and x.shape[1]==2:
            x=x.T
            
        play(x,rate)

    else:
        warnings.warn('Cannot play sound, no sounddevice o audiolab module.')

def soundsc(x, rate=44100):
    sound(x/np.max(np.abs(x))*0.9,rate)
