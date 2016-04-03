# -*- coding: utf-8 -*-

"""
Math Applyied to Music and extras

"""

from __future__ import division

try:
	from music21 import *
except ImportError:
	"No music21"
	pass

from scipy.signal import lfilter, firwin, tf2zpk, butter, freqz, freqs, ellip 

__all__ = ["math","midi","signal","plot","audio"]


from . import math
from .math import *

from . import midi
from .midi import *

from . import signal
from .signal import *

from . import mfcc
from .mfcc import *

from . import plot
from .plot import *

from . import audio
from .audio import *

__all__ += math.__all__
__all__ += midi.__all__
__all__ += signal.__all__
__all__ += plot.__all__
__all__ += audio.__all__

