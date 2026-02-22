"""Import ReMoDe class and statistical tests."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ['ReMoDe', 'perform_fisher_test', 'perform_binomial_test', '__version__']

from .remode import ReMoDe, perform_fisher_test, perform_binomial_test

try:
    __version__ = version('ReMoDe')
except PackageNotFoundError:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = '0+unknown'
