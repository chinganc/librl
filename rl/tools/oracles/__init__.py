from .oracle import Oracle
from .meta_oracles import MetaOracle, LazyOracle, AdversarialOracle, DummyOracle

import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_oracles import tfOracle, tfLikelihoodRatioOracle
