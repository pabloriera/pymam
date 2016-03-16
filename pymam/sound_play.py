# -*- coding: utf-8 -*-
"""
IMPORT SOUND MODULES FOR LINUX




soundsc(x,rate)
play scaled sound from x array at given sampling rate

"""

import numpy as np
import warnings
from pylab import pause

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

__all__ = ["sound", "soundsc"]

def sound(x, fs=44100, blocking = True):

    """
    Play sound from x array at given sampling rate
    """
    
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
        
        if x.ndim==2 and x.shape[0]==2:
            x=x.T
            
        play(x,fs)

        if blocking:

            pause(float(x.shape[0])/fs)


    else:
        warnings.warn('Cannot play sound, no sounddevice o audiolab module.')

def soundsc(x, fs=44100):
    
    """
    Play normalized sound from x array at given sampling rate
    """

    sound(x/np.max(np.abs(x))*0.9,fs)
