import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint
from numpy.polynomial.polynomial import Polynomial

DEGREE : int = 4
EPS: float = 1e-7

class compute_roots:

    def __init__(self, index:int):
        self.index = index

    def __call__(self,x):
        p = x[:DEGREE + 1]
        roots = [r for r in Polynomial(p).roots()]
        roots.sort()
        if len(roots) < self.index + 1 or type(roots[self.index]) == np.complex128:
            return EPS
        else:
            return np.abs(roots[self.index] - 0.5) - 0.5


root_constraints = [NonlinearConstraint(
    fun=compute_roots(index=i) ,
    lb=EPS,
    ub=np.inf,
    keep_feasible=False
 ) for i in range(DEGREE) ]




## INTEGRAL CONSTRAINTS
integral_constraint = LinearConstraint(A=[1 / (i + 1) for i in range(DEGREE + 1)], lb=1, ub=1, keep_feasible=False)

# Constraint on polynomial coefficient when also one dynamic threshold are optimized
integral_constraint_thr1 = LinearConstraint(A=[1 / (i + 1) for i in range(DEGREE + 1)]  + [0] * 1, lb=1, ub=1, keep_feasible=False)

# Constraint on polynomial coefficient when also two dynamic threshold are optimized
integral_constraint_thr2 = LinearConstraint(A=[1 / (i + 1) for i in range(DEGREE + 1)] + [0] * 2, lb=1, ub=1, keep_feasible=False)



poly_constraints = (
    integral_constraint,
    *root_constraints

               )

poly_constraints_thr = {1:(
     integral_constraint_thr1,
*root_constraints

                ),
     2:(
     integral_constraint_thr2,
*root_constraints

                ),}
