# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from rl.core import function_approximators as fa
from rl.core.function_approximators.supervised_learners import SupervisedLearner

def robust_keras_supervised_learner(cls):
    """ A decorator for creating basic supervised learners from RobustKerasFuncApp. """
    assert issubclass(cls, fa.KerasFuncApp)
    class decorated_cls(cls, SupervisedLearner):

        def __init__(self, x_shape, y_shape, name='k_robust_super_learner',
                     lr=0.001, loss='mse', metrics=('mae','mse'), **kwargs):
            super().__init__(x_shape, y_shape, name=name, **kwargs)
            self._lr =lr
            self._loss = loss
            self._metrics = metrics
            self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(lr),
                                loss=loss, metrics=list(metrics))

        def update_funcapp(self, clip_y=False,
                            batch_size=128, n_steps=500,
                            epochs=None, **kwargs):  # for keras.Model.fit
            """ `clip_y`: whether to clip the targe """

            if isinstance(self, fa.RobustKerasFuncApp):
                xs = self._x_nor(self._dataset['xs'])
                ys = self._dataset['ys']
                if clip_y:
                    ys = self._y_nor(ys)
            else:
                xs, ys = self._dataset['xs'], self._dataset['ys']

            if epochs is None:
                epochs = int(n_steps/max(1,len(ys)/batch_size))

            return self.kmodel.fit(xs, ys, sample_weight=self._dataset['ws'], verbose=0,
                                   batch_size=batch_size, epochs=epochs,**kwargs)

        def __setstate__(self, d):
            super().__setstate__(d)
            self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(self._lr),
                                loss=self._loss, metrics=list(self._metrics))

        def predict(self, *args, **kwargs):
            # prevent memory-leak
            return self.k_predict(*args, **kwargs)



    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@robust_keras_supervised_learner
class SuperRobustKerasFuncApp(fa.RobustKerasFuncApp):
    pass

@robust_keras_supervised_learner
class SuperRobustKerasMLP(fa.RobustKerasMLP):
    pass

@robust_keras_supervised_learner
class SuperKerasMLP(fa.KerasMLP):
    pass

