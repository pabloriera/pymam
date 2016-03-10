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

__all__ = ["math","midi","signals","plots","sound_play","wavio"]


from . import math
from .math import *

from . import midi
from .midi import *

from . import signals
from .signals import *

from . import mfcc
from .mfcc import *

from . import plots
from .plots import *

from . import sound_play
from .sound_play import *

from . import wavio
from .wavio import *

# __all__ += math.__all__
# __all__ += midi.__all__
# __all__ += signals.__all__
# __all__ += plots.__all__
# __all__ += sound_play.__all__
# __all__ += wavio.__all__
