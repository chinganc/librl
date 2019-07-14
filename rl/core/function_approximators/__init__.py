from .function_approximator import FunctionApproximator, online_compatible
import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_function_approximators import KerasFuncApp, RobustKerasFuncApp, RobustKerasMLP
#from .tf_function_approximators import tfFunctionApproximator, tfMLPFunctionApproximator

