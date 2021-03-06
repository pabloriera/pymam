# -*- coding: utf-8 -*-
"""
Audio module

"""

import numpy as np
import warnings
from pylab import pause

try:
    from sounddevice import play
    audio_ok = True
except ImportError:
    audio_ok = False

if not audio_ok:
    try:
        from _scikits.audiolab import play
        audio_ok = True
    except ImportError:
        audio_ok = False


__all__ = ["sound", "soundsc", 'wavread24', 'wavwrite24', 'wavread', 'wavwrite']

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
        print('Error: 1-D array for mono, or 2-D array where rows should be the number of channels, 1 (mono) or 2 (stereo)')
        return
        
    if audio_ok:
        
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




def _wav2array(nchannels, sampwidth, data):
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
    """data must be the string containing the bytes from the wav file."""

    import numpy as _np

    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = _np.empty((num_samples, nchannels, 4), dtype=_np.uint8)
        raw_bytes = _np.fromstring(data, dtype=_np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = _np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def wavread24(file):
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
    """
    Read a WAV file.

    Parameters
    ----------
    file : string or file object
        Either the name of a file or an open file pointer.

    Return Values
    -------------
    rate : float
        The sampling frequency (i.e. frame rate)
    sampwidth : float
        The sample width, in bytes.  E.g. for a 24 bit WAV file,
        sampwidth is 3.
    data : numpy array
        The array containing the data.  The shape of the array is
        (num_samples, num_channels).  num_channels is the number of
        audio channels (1 for mono, 2 for stereo).

    Notes
    -----
    This function uses the `wave` module of the Python standard libary
    to read the WAV file, so it has the same limitations as that library.
    In particular, the function does not read compressed WAV files.

    """
    import wave as _wave
    

    wav = _wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def wavwrite24(filename, rate, data):
    """
    Create a 24 bit wav file.

    Parameters
    ----------
    filename : string
        Name of the file to create.
    rate : float
        The sampling frequency (i.e. frame rate) of the data.
    data : array-like collection of integer or floating point values
        data must be "array-like", either 1- or 2-dimensional.  If it
        is 2-d, the rows are the frames (i.e. samples) and the columns
        are the channels.

    Notes
    -----
    The data is assumed to be signed, and the values are assumed to be
    within the range of a 24 bit integer.  Floating point values are
    converted to integers.  The data is not rescaled or normalized before
    writing it to the file.

    Example
    -------
    Create a 3 second 440 Hz sine wave.

    >>> rate = 22050  # samples per second
    >>> T = 3         # sample duration (seconds)
    >>> f = 440.0     # sound frequency (Hz)
    >>> t = np.linspace(0, T, T*rate, endpoint=False)
    >>> x = (2**23 - 1) * np.sin(2 * np.pi * f * t)
    >>> writewav24("sine24.wav", rate, x)

    """

    import numpy as _np
    import wave as _wave

    a32 = _np.asarray(data, dtype=_np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> _np.array([0, 8, 16])) & 255
    wavdata = a8.astype(_np.uint8).tostring()

    w = _wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()

   
def wavread(file_name):
    
    from scipy.io.wavfile import read

    fs, y = read(file_name)
    return fs,np.array(y.T,dtype=np.float64)/(2**15-1)

def wavwrite(file_name,x,fs = 44100):

    from scipy.io.wavfile import write

    x = x.T/np.max(np.abs(x))*0.9
    write(file_name,fs,np.array(x*(2**15-1),dtype=np.int16))