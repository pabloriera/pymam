# -*- coding: utf-8 -*-
#copyright © 1999-2010, Preeti Rao, Maxim Schram, and Dik Hermes
#
#Redistribution and use in source, with or without 
#modification, are permitted provided that the following condition is 
#met:
#
#* Redistributions of source code must retain the above copyright notice, 
#this list of conditions and the following disclaimer.
#
#The authors furnish this item "as is". 
#The authors do not provide any warranty of the item whatsoever, 
#whether express, implied, or statutory, including, but not limited to, 
#any warranty of merchantability or fitness for a particular 
#purpose or any warranty that the contents of the item will be error-free.
#
#In no respect shall the authors incur any liability for any damages, including, 
#but limited to, direct, indirect, special, or consequential damages arising out 
#of, resulting from, or in any way connected to the use of the item, whether or 
#not based upon warranty, contract, tort, or otherwise; whether or not injury 
#was sustained by persons or property or otherwise; and whether or not loss was 
#sustained from, or arose out of, the results of, the item, or any services 
#that may be provided by the authors. 
# 
#
#This software was written by Preeti Rao at the 
#Institute for Perception Research (IPO) in 1999, and 
#adapted by Dik Hermes, Human-Technology Interaction group,
#Eindhoven University of Technology in 2009.

#Adapted to python from matlab source done by the corresponding authors mentioned above.    

from __future__ import division
import numpy as np

import pylab

ascale = 1

def ftoerb(f):
	return 24.7 * (4.37 * f/1000 + 1)

def ftoerbscale(f):
    return 21.4*np.log10(4.37*f/1000+1)

def ftocb(f):
    return 25+75*(1+1.4*(f/1000)**2)**0.69

def rms2db(Arms):
    return 20*np.log10(Arms/20e-6/ascale)

def db2rms(Idb):
    return 20e-6*10**(Idb/20)*ascale

def normalize2db(signal,Idb):
    rms = np.sqrt(np.mean(signal**2));
    signal = db2rms(Idb)*signal/rms;
    return signal

def cosineramp(dur,N,fs=44100):

    ramp = np.ones(N)
    t = np.arange(N)/fs
    ramp[t<=dur] = 0.5 * (1 + np.cos(np.pi/dur * (t[t<=dur] - dur) ))
    return ramp


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
 
	http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

def RoundedExponential_(f_center,F,I_dB):
    min_level_db=20
    fk=f_center / 1000
    I_dB=max(min_level_db,I_dB)
    erb=24.7 * (4.37 * fk + 1)
    erbk=24.7 * (4.37 * 1 + 1)
    pk51=(4 * 1000) / erbk
    p51=4 * f_center / erb
    pl=p51 - 0.35 * (I_dB - 51) * (p51 / pk51)
    pu=p51
    pl=pl.clip(0,max(pl))
    pu=pu.clip(0,max(pu))
    g=(F - f_center) / f_center
    ipos=pylab.find(g >= 0)
    ineg=pylab.find(g < 0)
    w=np.zeros(g.shape)
    w[ipos]=(1 + pu[ipos]*g[ipos])*(np.exp(- pu[ipos]*g[ipos]))
    w[ineg]=(1 - pl[ineg]*g[ineg])*(np.exp(+ pl[ineg]*g[ineg]))
    return w

def MiddleEarFilter_(F,I_in):
    flt=np.array([[  2.00000000e+01,   3.91500000e+01],
       [  2.50000000e+01,   3.14000000e+01],
       [  3.15000000e+01,   2.54000000e+01],
       [  4.00000000e+01,   2.09000000e+01],
       [  5.00000000e+01,   1.80000000e+01],
       [  6.30000000e+01,   1.61000000e+01],
       [  8.00000000e+01,   1.42000000e+01],
       [  1.00000000e+02,   1.25000000e+01],
       [  1.25000000e+02,   1.11300000e+01],
       [  1.60000000e+02,   9.71000000e+00],
       [  2.00000000e+02,   8.42000000e+00],
       [  2.50000000e+02,   7.20000000e+00],
       [  3.15000000e+02,   6.10000000e+00],
       [  4.00000000e+02,   4.70000000e+00],
       [  5.00000000e+02,   3.70000000e+00],
       [  6.30000000e+02,   3.00000000e+00],
       [  7.50000000e+02,   2.70000000e+00],
       [  8.00000000e+02,   2.60000000e+00],
       [  1.00000000e+03,   2.60000000e+00],
       [  1.25000000e+03,   2.70000000e+00],
       [  1.50000000e+03,   3.70000000e+00],
       [  1.60000000e+03,   4.60000000e+00],
       [  2.00000000e+03,   8.50000000e+00],
       [  2.50000000e+03,   1.08000000e+01],
       [  3.00000000e+03,   7.30000000e+00],
       [  3.15000000e+03,   6.70000000e+00],
       [  4.00000000e+03,   5.70000000e+00],
       [  5.00000000e+03,   5.70000000e+00],
       [  6.00000000e+03,   7.60000000e+00],
       [  6.30000000e+03,   8.40000000e+00],
       [  8.00000000e+03,   1.19000000e+01],
       [  9.00000000e+03,   1.06000000e+01],
       [  1.00000000e+04,   9.90000000e+00],
       [  1.12000000e+04,   1.19000000e+01],
       [  1.25000000e+04,   1.39000000e+01],
       [  1.40000000e+04,   1.60000000e+01],
       [  1.50000000e+04,   1.73000000e+01],
       [  1.60000000e+04,   1.78000000e+01]])
    a=np.interp(F,flt[:,0],flt[:,1])
    I_out=I_in / (10.0 ** (a / 10.0))
    return I_out

def InternalExcitation_(F):
    ie=np.array([[  8.00000000e+00,   2.62000000e+01],
       [  5.20000000e+01,   2.62000000e+01],
       [  7.40000000e+01,   2.02000000e+01],
       [  1.08000000e+02,   1.45000000e+01],
       [  1.40000000e+02,   1.20000000e+01],
       [  2.53000000e+02,   6.30000000e+00],
       [  5.00000000e+02,   3.60000000e+00],
       [  2.00000000e+04,   3.60000000e+00]])
    E_thr_Q=np.interp(F,ie[:,0],ie[:,1])
    return E_thr_Q.flatten()

def ExcitationToSpecificLoudness_(E,F):
    I_ref=1e-12
    E_thr_ref_dB=3.6
    C=0.0467338323243
    E_thr_Q_dB=InternalExcitation_(F)
    G_dB=- (E_thr_Q_dB - E_thr_ref_dB)
    G=10.0 ** (G_dB / 10)
    alpha,A=AlphaAFromG_dB_(G_dB)
    E_s= E / I_ref
    specificLoudness=C * (( G * E_s + A) ** alpha - A ** alpha)
    E_thr_Q=np.zeros(E_thr_Q_dB.shape)
    for k in np.arange(len(F)):
        E_thr_Q[k]=10 ** (E_thr_Q_dB[k] / 10)
        if (E_s[0,k] < E_thr_Q[k]).all():
            specificLoudness[0,k]=specificLoudness[0,k] * (2 * E_s[k] / (E_s[k] + E_thr_Q[k])) ** 1.5
        if (E_s[0,k] > 10 ** 10).all():
            specificLoudness[k]=C * (E_s[k] / 1040000.0) ** 0.5
    return specificLoudness

def AlphaAFromG_dB_(G_dB):
    alphaA=np.array([[-25.   ,   0.267,   8.8  ],
       [-20.   ,   0.252,   7.6  ],
       [-15.   ,   0.238,   6.6  ],
       [-10.   ,   0.222,   5.8  ],
       [ -5.   ,   0.21 ,   5.1  ],
       [  0.   ,   0.2  ,   4.62 ]])
    alpha=np.interp(G_dB,alphaA[:,0],alphaA[:,1])
    A=np.interp(G_dB,alphaA[:,0],alphaA[:,2])
    return alpha,A
    
def OuterEarFilter_(F,I_in):
    flt = np.array([[  2.00000000e+01,   0.00000000e+00],
       [  2.50000000e+01,   0.00000000e+00],
       [  3.15000000e+01,   0.00000000e+00],
       [  4.00000000e+01,   0.00000000e+00],
       [  5.00000000e+01,   0.00000000e+00],
       [  6.30000000e+01,   0.00000000e+00],
       [  8.00000000e+01,   0.00000000e+00],
       [  1.00000000e+02,   0.00000000e+00],
       [  1.25000000e+02,   1.00000000e-01],
       [  1.60000000e+02,   3.00000000e-01],
       [  2.00000000e+02,   5.00000000e-01],
       [  2.50000000e+02,   9.00000000e-01],
       [  3.15000000e+02,   1.40000000e+00],
       [  4.00000000e+02,   1.60000000e+00],
       [  5.00000000e+02,   1.70000000e+00],
       [  6.30000000e+02,   2.50000000e+00],
       [  7.50000000e+02,   2.70000000e+00],
       [  8.00000000e+02,   2.60000000e+00],
       [  1.00000000e+03,   2.60000000e+00],
       [  1.25000000e+03,   3.20000000e+00],
       [  1.50000000e+03,   5.20000000e+00],
       [  1.60000000e+03,   6.60000000e+00],
       [  2.00000000e+03,   1.20000000e+01],
       [  2.50000000e+03,   1.68000000e+01],
       [  3.00000000e+03,   1.53000000e+01],
       [  3.15000000e+03,   1.52000000e+01],
       [  4.00000000e+03,   1.42000000e+01],
       [  5.00000000e+03,   1.07000000e+01],
       [  6.00000000e+03,   7.10000000e+00],
       [  6.30000000e+03,   6.40000000e+00],
       [  8.00000000e+03,   1.80000000e+00],
       [  9.00000000e+03,  -1.00000000e+00],
       [  1.00000000e+04,  -1.60000000e+00],
       [  1.12000000e+04,   1.90000000e+00],
       [  1.25000000e+04,   4.90000000e+00],
       [  1.40000000e+04,   2.00000000e+00],
       [  1.50000000e+04,  -2.00000000e+00],
       [  1.60000000e+04,   2.50000000e+00]])
    I_dB=np.interp(F,flt[:,0],flt[:,1])
    I_out=I_in*(10.0 ** (I_dB / 10))
    return I_out

def excitation_pattern(F,I,z=None,freqs=None,I_scale = 'dB',freeField=False,middleEar = True,binaural=False):
    """    
    Calculates the excitation pattern from a list of frecuencies (F) and intensities (I_dB)
    
    freqs = np.logspace(log10(fmin),log(fmax),100)
    
    freqs,ep = excitation_pattern(F, I_dB, freqs=freqs)
    
    Alternative on erbscale z
   
    z = np.arange(1.8,39,0.1)
    
    fErb,ep = excitation_pattern(F, I_dB, z=z)
    
     The models incorporate the descriptions published in:
     
     [1] Moore B.C.J. & Glasberg B.R. (1996)
     A revision of Zwicker's loudness Model.
     Acustica 82, 335-345.
     
     [2] Moore B.C.J., Glasberg B.R. & Baer T. (1997), 
     A model for the prediction of thresholds, loudness, 
     and partial loudness. 
     J. Audio Eng. Soc.45(4), 224-240.
     
     The rounded exponential filter shapes are calculated 
     according to:
     [3] Glasberg B.R. & Moore B.C.J. (1990)
     Derivation of auditory filter shapes from notched-noise data.
     Hearing Research 47, 103-128.

    """
    
    if type(F) is not np.ndarray: F = np.array(F,ndmin=1)
        
    if type(I) is not np.ndarray: I = np.array(I,ndmin=1)
    
    I_ref = 1.0e-12

    if I_scale=='dB':
        I_db = I
        I_lin = I_ref * 10**(I_db/10)        
    else:
        I_lin = I        
        
   
    F = list(F)

   

    min_spl_dB=-500
    min_spl=I_ref * 10 ** (min_spl_dB / 10)
    
    if freeField:
        I_lin = OuterEarFilter_(F,I_lin)

    if middleEar:
        I_lin = MiddleEarFilter_(F,I_lin)
    
    I_lin = I_lin.clip(min_spl,np.max(I_lin))
    
    I_db = 10 * np.log10(I_lin / I_ref)
    
    if z!=None:    
        #erbscale z to hertz
        freqs=(10.0 ** (z / 21.4) - 1) * 1000 / 4.37
            
    E=np.zeros(len(freqs))
    
    for ff, IIdb, II in zip(F,I_db, I_lin ):
        w = RoundedExponential_(freqs , ff, IIdb)
        E = E + w * II
        
    ep=10 * np.log10(E / I_ref)
    
#    specificLoudness=ExcitationToSpecificLoudness_(E,fErb)
#    if binaural:
#        specificLoudness=2 * specificLoudness
#    return fErb,excitationPattern,specificLoudness
    
    return freqs,ep
