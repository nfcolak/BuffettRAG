"""Frozen grounded-engine contracts; import concrete modules explicitly.

This package deliberately imports no engine, settings, provider or ML dependency.
The parallel implementations must not add eager imports here.
"""

__all__: list[str] = []
