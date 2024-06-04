"""
 # Created on 29.10.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Defines logger class
 #
"""

from foundations import paths
from foundations.step import Step
from platforms.platform import get_platform

from torch import tensor

row_separator = '//\n'
field_separator = '$'

class MetricLogger:
    def __init__(self):
        self.log = {}

    def add(self, name: str, step: Step, value: float):
        self.log[(name, step.iteration)] = value

    def __str__(self):
        return f"{row_separator}".join(
            [f'{k[0]}{field_separator}{k[1]}{field_separator}{v}' for k, v in self.log.items()])

    @staticmethod
    def create_from_string(as_str):
        logger = MetricLogger()
        if len(as_str.strip()) == 0:
            return logger

        rows = [row.split(field_separator) for row in as_str.strip().split(row_separator)]
        logger.log = {(name, int(iteration)): eval(value) for name, iteration, value in rows}
        return logger

    @staticmethod
    def create_from_file(filename):
        with get_platform().open(paths.logger(filename)) as fp:
            as_str = fp.read()
        return MetricLogger.create_from_string(as_str)

    def save(self, location):
        if not get_platform().is_primary_process: return
        if not get_platform().exists(location):
            get_platform().makedirs(location)
        with get_platform().open(paths.logger(location), 'w') as fp:
            fp.write(str(self))

    def get_data(self, desired_name):
        d = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, d[k]) for k in sorted(d.keys())]
