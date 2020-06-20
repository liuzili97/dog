from collections import OrderedDict

import numpy as np


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False
        self.max_record = 1e4

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def update_output(self, vars):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            self.output[key] = var

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg

            if len(self.val_history[key]) > self.max_record:
                self.val_history[key] = self.val_history[key][-self.max_record:]
                self.n_history[key] = self.n_history[key][-self.max_record:]
        self.ready = True
