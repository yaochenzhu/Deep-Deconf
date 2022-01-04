import logging

import numpy  as np
import pandas as pd
from scipy.interpolate import interp1d

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def Init_logging():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule():
    def __init__(self, 
                 endpoints, 
                 interpolation=linear_interpolation, 
                 outside_value=None):
        """
            Piecewise Linear learning schedule.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        ### t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

    def __call__(self, t):
        '''
            for compatibility with keras callbacks
        '''
        return self.value(t)