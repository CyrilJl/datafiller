"""Exception classes raised by datafiller.

All validation errors raised by the package derive from :class:`DataFillerError`,
so callers can catch every datafiller-specific error with a single handler::

    from datafiller import DataFillerError, MultivariateImputer

    try:
        MultivariateImputer()(x)
    except DataFillerError:
        ...

The concrete classes also inherit from the matching builtin (:class:`ValueError`
or :class:`TypeError`), so existing ``except ValueError`` handlers keep working.
"""


class DataFillerError(Exception):
    """Base class for all errors raised by datafiller."""


class DataFillerValueError(DataFillerError, ValueError):
    """A datafiller error raised for invalid argument values."""


class DataFillerTypeError(DataFillerError, TypeError):
    """A datafiller error raised for arguments of an unsupported type."""
