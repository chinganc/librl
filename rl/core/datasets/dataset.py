import numpy as np
import copy
from collections import defaultdict, namedtuple


def data_namedtuple(*args, **kwargs):
    CLS = namedtuple(*args, **kwargs)
    def newlen(self):
        x = [len(f) for f in self._fields]
        assert x.count(x[0]) == len(x)
        return x[0]
    CLS.__len__ = newlen
    return CLS


class Dataset:
    """ A collection of batch data where each batch contains a set of samples.

        The batch data class needs to implement `__len__` which returns the
        number of samples of that batch.  If the batch data instance has an
        `attr` attribute or key, `some_dataset[attr]` will return all the
        matched samples across all batch data.

    """

    def __init__(self, data=None, max_n_batches=None, max_n_samples=None,
                       single_batch=False):
        """
            Args:
                `data`: initial batch(es) of data
                `max_n_batches`: maximal number of batches to keep
                `max_n_samples`: maximal number of samples to keep
                `single_batch`: when the batch data class is a list and `data`
                                is one batch data instance, this option needs
                                to be set `True`.

            It keeps most recent data when, it is full. But note that because
            the data are kept in terms of batches, the actual number of samples
            might be slighly larger than `max_n_samples`, when the batch with
            the highest weight has more than `max_n_samples` of samples.

            Given this, if both `max_n_batches` and `max_n_batches` are 0, it
            only retains the most recent data.

        """
        if data is None:
            self._batches = []
        else:
            if type(data)==list and not single_batch:
                self._batches = data
            else:
                self._batches = [data]

        self.max_n_batches = max_n_batches
        self.max_n_samples = max_n_samples
        self._cache ={}

    def append(self, data):
        """ Add an new data into the dataset. """
        self._assert_data_type(data)
        self._make_room_for_batches(1)  # check if we need to throw away some old batches
        self._batches.append(data)  # always keep all the new batch
        self._limit_n_samples()  # check if some samples need removal
        self._cache = {}

    def extend(self, other):
        self._assert_other_type(other)
        self._make_room_for_batches(len(other))  # check if we need to throw away some old batches
        self._batches+=other._batches  # always keep all the new batch
        self._limit_n_samples()  # check if some samples need removal
        self._cache = {}

    def join(self, other):
        """ Combine two datasets """
        self._assert_other_type(other)
        def sum(a, b):
            return None if (a is None or b is None) else a+b
        batches  = copy.deepcopy(self._batches+other._batches)
        max_n_rollouts = sum(self.max_n_rollouts, other.max_n_rollouts)
        max_n_samples = sum(self.max_n_samples, other.max_n_samples)
        return type(self)(batches, max_n_rollouts=max_n_rollouts, max_n_samples=max_n_samples)

    def __add__(self, other):  # alias
        return self.join(other)

    def __getitem__(self, key):
        """ Retrieve a dataset containing only samples with keys, ordered from old to new. """
        if self._cache.get(key, None) is None:
            if key is None:  # try to concatenate all of them
                self._cache[key] = np.concatenate([b for b in self._batches])
            else:
                if type(key) is str:
                    try:
                        self._cache[key] = np.concatenate([getattr(b, key) for b in self._batches])
                    except AttributeError:
                        self._cache[key] = np.concatenate([b[key] for b in self._batches])
                else:
                    self._cache[key] = np.concatenate([b[key] for b in self._batches])


        return self._cache[key]

    def __iter__(self):
        for batch in self._batches:
            yield batch

    @property
    def n_samples(self):
        """ Number of samples across all the batches. """
        return sum([len(b) for b in self._batches])

    def __len__(self):
        """ Return the number of data. """
        return len(self._batches)

    # utils
    def _make_room_for_batches(self, n_batches):
        """ Delete old batches so that it can fit n_batches of new batches. """
        if self.max_n_batches is not None:
            extra_n_batches = len(self) + n_batches - self.max_n_batches
            if extra_n_batches>0:
                del self._batches[:extra_n_batches]

    def _limit_n_samples(self):
        """ Delete old batches so that the number of samples are within limit.

            When it's full, it stays full after deletion.
        """
        if self.max_n_samples is not None:
            ns = np.array([len(b) for b in self._batches])
            n_samples = np.sum(ns)
            extra_n_samples = n_samples - self.max_n_samples
            if extra_n_samples > 0:
                ind = np.argwhere(np.flipud(ns).cumsum()>=self.max_n_samples)
                if len(ind)>1: # let's allow having more than less
                    ind = ind[1][0]
                    del self._batches[:-ind]

    def _assert_data_type(self, data):
        if len(self)>0:
            assert type(data)==type(self._batches[0])

    def _assert_other_type(self, other):
        assert type(self)==type(other)
        if len(self)>0 and len(other)>0:
            self._assert_data_type(self._batches[0], other._batches[0])


