# -*- coding: utf-8 -*-

from __future__ import division

import scipy as sp
import numpy as np
import pylab as pl

try:
    from scipy.signal.windows import *
except ImportError:
    print 'No windows functions loaded'
    

from scipy.fftpack import fft, fftshift
from scipy.signal import lfilter, freqz, freqs, tf2zpk,  iirfilter
from itertools import combinations


__all__ = ['time_array', 'envelope', 'dcremove', 'spectrum', 'plotwindowspectrum', 'plotspectrum', 'plotwav', 'filterresponse', 'polosyzeros', 'tukeywin', 'stft', 'butter_filter', 'autocorrelation', 'acf', 'continous_unique', 'cosineramp', 'roughness', 'dissmeasure']


def time_array(start,dur,fs=44100):
    t = np.arange(0 , np.floor(fs*dur) )/fs+start
    return t

def envelope(amps,durs,fs=44100):
    
    t = time_array(0,sum(durs),fs)
    return np.interp(t,np.cumsum(np.array(durs)),amps),t
   
def dcremove(x):
    
    return x-np.mean(x)

def spectrum(x,fs=44100.0,nfft=-1):
    if nfft==-1:
        N = x.size
        F = 2*(abs(fft(x))/N)**2
        f = sp.arange(N)*fs/(N-1)
    else:
        F = 2*(abs(fft(x,nfft))/nfft)**2
        f = sp.arange(nfft)*fs/(nfft-1)
        
    return F,f
    
def plotwindowspectrum(window,nfft=2048):
    A = fft(window, nfft) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / np.abs(A).max())))
    pl.plot(freq,response);
    pl.axis([-0.5, 0.5, -120, 0]);
    pl.xlabel("Frecuencia [ciclos por sample]");
    
def plotspectrum(x,fs=44100.0,xscale='linear',units='db',plotformat = '-',nfft=-1,hzunit=1):
    F,f = spectrum(x,fs,nfft = nfft)
    N = float(len(F))
    if units=='db':
        p = pl.plot(f[0:int(N/2)]/hzunit,10*sp.log10(F[0:int(N/2)]),plotformat)
        pl.ylabel('Potencia dB')
    else:
        p = pl.plot(f[0:int(N/2)]/hzunit,F[0:int(N/2)],plotformat)
        pl.ylabel('Potencia')
    
    pl.xlim(0,fs/2/hzunit)
    pl.gca().set_xscale(xscale)
    if hzunit==1000:
        hzunitstr='k'
    elif hzunit==1:
        hzunitstr=''
    else:
        hzunitstr=str(hzunit)
        
    pl.xlabel('Frecuencia ('+hzunitstr+'hz)')
    return p
    
def plotwav(x,fs = 44100):
    from scipy.signal import decimate
    
    l = len(x)
    
    if l > 1.5e6:
        q = int(np.ceil(l/1.5e6))
        x2 = decimate(x,q)
        l2 = len(x2)
        t = arange(l2)/l2*l/fs
        plot(t,x2)       
        xlabel('Segundos')
        title('Signal decimada en un factor '+str(q))
        
    else:
        t = arange(l)/fs
        plot(t,x)
        xlabel('Segundos')
        title('Signal completa')
        
def filterresponse(b,a,fs=44100,scale='log',**kwargs):
    w, h = freqz(b,a)
    pl.subplot(2,1,1)
    pl.title('Digital filter frequency response')
    pl.plot(w/max(w)*fs/2, 20 * np.log10(abs(h)),**kwargs)
    pl.xscale(scale)
#    if scale=='log':
#        pl.semilogx(w/max(w)*fs/2, 20*np.log10(np.abs(h)), 'k')
#    else:
#        pl.plot(w/max(w)*fs/2, 20*np.log10(np.abs(h)), 'k')
        
    pl.ylabel('Gain (dB)')
    pl.xlabel('Frequency (rad/sample)')
    pl.axis('tight')    
    pl.grid()    
    

    pl.subplot(2,1,2)
    angles = np.unwrap(np.angle(h))
    if scale=='log':
        pl.semilogx(w/max(w)*fs/2, angles, **kwargs)
    else:
        pl.plot(w/max(w)*fs/2, angles, **kwargs)
        

    pl.ylabel('Angle (radians)')
    pl.grid()
    pl.axis('tight')
    pl.xlabel('Frequency (rad/sample)')

def polosyzeros(b, a):
    (zeros,poles,gain) = tf2zpk(b, a)
    angle = np.linspace(-np.pi,np.pi,50)
    cirx = np.sin(angle)
    ciry = np.cos(angle)
    l, = pl.plot(poles.real, poles.imag, 'x')
    pl.plot(zeros.real, zeros.imag, 'o',color=l.get_color())
    pl.plot(cirx,ciry, 'k-')
    pl.grid()
    
    pl.xlim((-2, 2))
    pl.xlabel('Real')
    pl.ylim((-1.5, 1.5))
    pl.ylabel('Imag')
    
    return (zeros,poles,gain)

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



def stft(x, fs,winsize,win=sp.signal.hamming, olap=None):
    if olap==None: olap = winsize/2
    olap = int(olap)
    winsize=int(winsize)
    w = win(winsize)
    X = sp.array([sp.fft(w*x[i:i+winsize]) 
                     for i in range(0, len(x)-winsize, olap)])
    return X


def butter_filter(data, cutoff, fs, btype='low', order=5, ab=False):
    from scipy.signal import butter, lfilter, freqz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype)
    if ab:
        return b, a
    else:
        y = lfilter(b, a, data)
        return y

def autocorrelation(x,maxlag=0,norm='true'):
    """
    Compute autocorrelation using FFT
    The idea comes from 
    http://dsp.stackexchange.com/a/1923/4363 (Hilmar)
    """
    x = np.asarray(x)

    s = np.fft.fft(x)
    ac = np.real(np.fft.ifft(s * np.conjugate(s)))
    
    if x.ndim==2:
        N = x.shape[1]
    
        if maxlag==0:
            ac = ac[:,:N]
        else:
            ac = ac[:,:maxlag]
        if norm:
            ac = (ac.T / ac[:,0]).T
            
        return ac
    else:
        N = x.shape[0]

        if maxlag==0:
            ac = ac[:N]
        else:
            ac = ac[:maxlag]
        if norm:
            ac = (ac.T / ac[0]).T
            
        return ac

def acf(x,maxlag=0,axis = -1): 
    
    from scipy.signal import fftconvolve
    
    if x.ndim==1:
        x = x[np.newaxis,:]
        
    else:

        x = np.rollaxis(x, axis)
        
    y = []
    
    for xx in x:
    
        n = xx.size
              
        if not (n & 0x1):
            xx = xx[:-1]
            n = xx.size
            
        if maxlag==0:
            maxlag = n
        
        b = np.zeros( n + maxlag )
    
        b[0:n] = xx # This works for n being even
    
        # Do an array flipped convolution, which is a correlation.
        ac = fftconvolve(b, xx[::-1], mode='valid') 
        y.append( ac/ac.max() )
        
        
    y = np.rollaxis( np.array(y), axis ) 
        
    return y

def acf_old(x,maxlag=10,method='fft'):
    
    n = len(x)

    if method=='fft':
        from scipy.signal import fftconvolve
        if not (n & 0x1):
            x = x[:-1]
            n = len(x)
        
        b = np.zeros(n * 2)
    
        b[n/2:n/2+n] = x # This works for n being even
    
        # Do an array flipped convolution, which is a correlation.
        return fftconvolve(b, x[::-1], mode='valid') 

    else:
        if np.iscomplex(x).any():
            return np.correlate(x,x,'full')[n-maxlag:n+maxlag]
        else:
            return np.correlate(x,x,'full')[n:n+maxlag]

     
def continous_unique(x,th,y=None):
    """x should have shape like (N,1)"""
    
    from scipy.spatial.distance import pdist
    
    pixs = [(i,j) for i in xrange(len(x)) for j in xrange(i+1,len(x))]
    pd = pdist(x)
    aw = np.argwhere(pd<th)
    boo = np.ones(len(x),dtype=np.bool)
    
    if type(y)!=type(None):
        for p in aw:
            
            i = y[list(pixs[p])].argmin()
            boo[pixs[p][i]]*=False
            
        return x[boo],y[boo] 
    else:
        for p in aw:
            i = pixs[p][0]
            boo[i]*=False
        return boo
               

def cosineramp(dur,N,fs=44100):

    ramp = np.ones(N)
    t = np.arange(N)/fs
    ramp[t<=dur] = 0.5 * (1 + np.cos(np.pi/dur * (t[t<=dur] - dur) ))
    return ramp

def roughness_(f1,f2,A1,A2):
    
    fmin = float(min(f1,f2))
    fmax = float(max(f1,f2))
    Amin = float(min(A1,A2))
    Amax = float(max(A1,A2))
    
    s = 0.24/(0.0207*fmin+18.96)
    X = Amin*Amax
    Y = 2*Amin/(Amin+Amax)
    Z = (np.exp(-3.5*s*(fmax-fmin))-np.exp(-5.75*s*(fmax-fmin)))
    
    return 0.5*(X**0.1)*(Y**3.11)*Z

def roughness(f_list,a_list):       
    
    l1 = list(combinations(f_list,2))
    l2 = list(combinations(a_list ,2))

    R = 0
    for f,a in zip(l1,l2):  
        R+=roughness_(f[0],f[1],a[0],a[1])

    return R

def dissmeasure(fvec, amp, model='min'):
    from numpy import exp, asarray, argsort, sort, sum, minimum
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.

    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    fvec = asarray(fvec)
    amp = asarray(amp)
 
    # used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96
 
    C1 = 5
    C2 = -5
 
    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75
 
    ams = amp[argsort(fvec)]
    fvec = sort(fvec)
 
    D = 0
    for i in range(1, len(fvec)):
        Fmin = fvec[:-i]
        S = Dstar / (S1 * Fmin + S2)
        Fdif = fvec[i:] - fvec[:-i]
        if model == 'min':
            a = minimum(ams[:-i], ams[i:])
        elif model == 'product':
            a = ams[i:] * ams[:-i] # Older model
        else:
            raise ValueError('model should be "min" or "product"')
        Dnew = a * (C1 * exp(A1 * S * Fdif) + C2 * exp(A2 * S * Fdif))
        D += sum(Dnew)
 
    return D
