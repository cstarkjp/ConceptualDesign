"""
Utilities.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`

---------------------------------------------------------------------
"""

import sympy as sy
from sympy import Eq
from sympy.physics.units import convert_to

# e2d = lambda fn: sy.solve(fn, fn.lhs, dict=True)[0]
def e2d(fn, do_flip: bool = False):
    """Convert equation into dictionary."""
    d = sy.solve(fn, fn.lhs, dict=True)[0]
    return (
        {list(d.values())[0]: list(d.keys())[0]} if do_flip
        else d
    )

def conv(expr, units, n):
    """
    Change units in SymPy equation with numerical RHS and round the value.
    """
    conv_x = int if n<=0 else lambda x: x
    return Eq(expr.lhs, conv_x((convert_to(expr.rhs,units)/units).round(n))*units)
