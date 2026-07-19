:notoc: true

API Reference
#############

This page provides a detailed API reference for the classes and functions in the `datafiller` library.

Imputer Classes
***************

.. admonition:: MultivariateImputer
   :class: dropdown

   .. autoclass:: datafiller.MultivariateImputer
      :members: __call__
      :undoc-members:
      :show-inheritance:

.. admonition:: TimeSeriesImputer
   :class: dropdown

   .. autoclass:: datafiller.TimeSeriesImputer
      :members: __call__
      :undoc-members:
      :show-inheritance:

Models
******

.. admonition:: FastRidge
   :class: dropdown

   .. autoclass:: datafiller.FastRidge
      :members:
      :undoc-members:
      :show-inheritance:

.. admonition:: ExtremeLearningMachine
   :class: dropdown

   .. autoclass:: datafiller.ExtremeLearningMachine
      :members:
      :undoc-members:
      :show-inheritance:

Exceptions
**********

All validation errors raised by the library derive from
:class:`~datafiller.DataFillerError`, so a single ``except DataFillerError``
catches every datafiller-specific error. The concrete classes also inherit
from the matching builtin (``ValueError`` or ``TypeError``), so existing
handlers keep working.

.. admonition:: Exception classes
   :class: dropdown

   .. autoclass:: datafiller.DataFillerError
      :show-inheritance:

   .. autoclass:: datafiller.DataFillerValueError
      :show-inheritance:

   .. autoclass:: datafiller.DataFillerTypeError
      :show-inheritance:

***********************
Low-Level Functions
***********************

.. admonition:: optimask
   :class: dropdown

   .. automodule:: datafiller._optimask
      :members: optimask
