# Copyright (c) 2016 rllab contributors
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class OnlineLearner(ABC):
    """ An abstract interface of iterative algorithms. """

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Update the state given feedback. """

    @property
    @abstractmethod
    def decision(self):
        """ Return the (stochastic) decision. """
