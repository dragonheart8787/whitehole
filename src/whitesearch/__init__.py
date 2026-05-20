"""WhiteSearch — White hole candidate signal search and evidence ranking engine.

Entry points
------------
- whitesearch.models      : parametric white hole models
- whitesearch.simulators  : three-channel forward simulators
- whitesearch.dataio      : public data source interfaces
- whitesearch.preprocess  : quality flagging and calibration
- whitesearch.likelihoods : per-channel and joint likelihoods
- whitesearch.inference   : Bilby/dynesty evidence engine
- whitesearch.validation  : SBC, PPC, injection-recovery, sensitivity
- whitesearch.surrogates  : surrogate / emulator acceleration layer
- whitesearch.utils       : physical constants and math helpers
"""

__version__ = "0.1.0"
__author__ = "WhiteSearch Team"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
