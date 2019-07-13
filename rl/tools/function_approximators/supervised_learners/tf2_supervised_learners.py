import tensorflow as tf
from rl.tools import function_approximators as fa
from rl.tools.function_approximators.supervised_learners import SupervisedLearner

def robust_keras_supervised_learner(cls):
    """ A decorator for creating basic supervised learners from KerasRobustFuncApp. """
    assert issubclass(cls, fa.KerasRobustFuncApp)
    class decorated_cls(cls, SupervisedLearner):

        def __init__(self, x_shape, y_shape, name='k_robust_super_learner',
                     lr=0.001, loss='mse', metrics=('mae','mse'), **kwargs):
            super().__init__(x_shape, y_shape, name=name, **kwargs)
            self._lr =lr
            self._kfun.kmodel.compile(optimizer=tf.keras.optimizers.Adam(self._lr),
                                      loss=loss, metrics=list(metrics))

        def update_funcapp(self, clip_y=False, **kwargs):  # for keras.Model.fit
            """
                `clip_y`: whether to clip the target
                `callback`: a function to execute after training. It inputs
                            (self, xs, ys, ws).
            """
            xs = self._x_nor(self._dataset['xs'])
            ys = self._dataset['ys']
            ws = self._dataset['ws']
            if clip_y:
                ys = self._y_nor(ys)
            return self._kfun.kmodel.fit(xs, ys, sample_weight=ws, **kwargs)


    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls

@robust_keras_supervised_learner
class superKerasRobustFuncApp(fa.KerasRobustFuncApp):
    pass

@robust_keras_supervised_learner
class superKerasRobustMLP(fa.KerasRobustMLP):
    pass


