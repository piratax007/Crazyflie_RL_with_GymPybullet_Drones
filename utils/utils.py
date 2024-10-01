"""General use functions.
"""
import time
import argparse
from collections import deque

from scipy import signal
import numpy as np


################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


class FIRFilter:
    def __init__(
            self,
            sample_frequency: int = 30,
            cutoff_frequency: int = 10,
            buffer_size: int = 15
    ):
        self.sample_frequency = sample_frequency
        self.cutoff_frequency = cutoff_frequency
        self.nyquist_frequency = 0.5 * self.sample_frequency
        self.cutoff_norm = self.cutoff_frequency / self.nyquist_frequency
        self.num_taps = 60
        self.fir_coeffs = signal.firwin(self.num_taps, cutoff=self.cutoff_norm)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.filtered_actions = np.zeros(4).astype('float32')

    def filter_actions(self, actions):
        flatted_buffer = np.concatenate(self.buffer).ravel()
        temp_buffer = np.concatenate((flatted_buffer[-56:], actions[0])).ravel()
        for i in range(len(self.filtered_actions)):
            self.filtered_actions[i] = np.dot(temp_buffer, self.fir_coeffs)
        return np.array([self.filtered_actions])
