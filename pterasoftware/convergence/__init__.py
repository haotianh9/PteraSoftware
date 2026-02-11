"""Contains functions for analyzing the convergence of SteadyProblems and
UnsteadyProblems.

**Contains the following subpackages:**

None

**Contains the following directories:**

None

**Contains the following modules:**

steady.py: Contains the analyze_steady_convergence function.

unsteady.py: Contains the analyze_unsteady_convergence function.

unsteady_non_trapezoidal.py: Contains the analyze_unsteady_convergence_non_trapezoidal
and analyze_unsteady_convergence_non_trapezoidal_optimized_dt functions.
"""

import pterasoftware.convergence._functions
import pterasoftware.convergence.steady
import pterasoftware.convergence.unsteady
import pterasoftware.convergence.unsteady_non_trapezoidal
from pterasoftware.convergence.steady import analyze_steady_convergence
from pterasoftware.convergence.unsteady import analyze_unsteady_convergence
from pterasoftware.convergence.unsteady_non_trapezoidal import (
    analyze_unsteady_convergence_non_trapezoidal,
    analyze_unsteady_convergence_non_trapezoidal_optimized_dt,
)
