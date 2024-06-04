"""
 # Created on 20.02.24
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Foundation runner.
 #
"""

import abc
import argparse


class Runner(abc.ABC):
    """An instance of a training run of some kind."""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this runner."""

        pass

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this runner."""

        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> 'Runner':
        """Create a runner from command line arguments."""

        pass

    @abc.abstractmethod
    def display_output_location(self) -> None:
        """Print the output location for the job."""

        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass
