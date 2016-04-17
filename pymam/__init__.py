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

# __all__ = ["math","midi","signal","plot","audio"]


from .math import *

from .midi import *

from .signal import *

from .mfcc import *

from .plot import *

from .audio import *

__all__ = math.__all__
__all__ += midi.__all__
__all__ += signal.__all__
__all__ += plot.__all__
__all__ += audio.__all__

