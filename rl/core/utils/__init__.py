from . import logz, math_utils, minibatch_utils, misc_utils, mvavg

import tensorflow as tf
if tf.__version__=='2':
    from . import tf2_utils as tf_utils
else:
    from . import tf_utils

