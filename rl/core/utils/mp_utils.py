# Copyright (c) 2018 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import copy
import multiprocessing as mp

ctx = mp.get_context('spawn')
Queue = ctx.Queue
Process = ctx.Process

class Worker(Process):
    """ A persistent process for avoiding loading modules. """

    def __init__(self, method=None, init_fun=None):
        self._init_fun = init_fun
        self._method = method
        super().__init__()

    def start(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        super().start()

    def run(self):
        if self._init_fun is not None:
            self._init_fun()

        while True:
            item = self.in_queue.get()
            if item is None:
                print('Terminating Process.')
                return
            if self._method is None:
                method, args, kwargs = item
                output = method(*args, **kwargs)
            else:
                args, kwargs = item
                output = self._method(*args, **kwargs)
            self.out_queue.put(output)


class JobRunner:

    def __init__(self, workers):
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.workers = workers
        [worker.start(self.in_queue, self.out_queue) for worker in workers]

    def stop(self):
        [self.put(None) for _ in self.workers]
        [worker.join() for worker in self.workers]

    def put(self, job):
        self.in_queue.put(job)

    def aggregate(self, size):
        return [self.out_queue.get() for _ in range(size)]

    def __len__(self):
        return len(self.workers)


