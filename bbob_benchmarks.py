import cocoex
import numpy as np

# --------------------------------------
# Configuration
# --------------------------------------
FUNCTIONS   = [2, 4, 6, 50, 52, 54]
INSTANCES   = [0, 1, 2]
REPETITIONS = 5
DIMENSIONS  = [2, 10]

# --------------------------------------
# COCO Suite Initialization
# --------------------------------------
suite = cocoex.Suite(
    suite_name="bbob-constrained",
    suite_instance="0-2",
    suite_options=""
)

observer = cocoex.Observer("bbob-constrained", "")

# --------------------------------------
# Helper: fetch specific problem
# --------------------------------------
def get_problem(fid, dim, inst):
    return suite.get_problem_by_function_dimension_instance(
        function=fid, dimension=dim, instance=inst
    )

import numpy as np

def make_coco_wrappers(problem):
    """
    Wrap a COCO constrained problem so it can be used with your BayesianOptimizer,
    which assumes X in [0,1]^dim.

    Returns:
        func(X)         -> objective values (batch or single)
        constraints     -> list of funcs [c1(X), c2(X), ...]
    """

    lb = np.asarray(problem.lower_bounds)  # e.g. [-5, -5, ...]
    ub = np.asarray(problem.upper_bounds)  # e.g. [ 5,  5, ...]
    dim = problem.dimension

    assert lb.shape[0] == dim and ub.shape[0] == dim

    def _to_coco(X_unit):
        """
        Map X in [0,1]^d to COCO domain [lb, ub]^d.

        Handles:
          - X_unit shape (dim,)
          - X_unit shape (N, dim)
        """
        X_unit = np.asarray(X_unit, dtype=float)
        X2 = np.atleast_2d(X_unit)
        X_real = lb + X2 * (ub - lb)
        return X_real

    # --- objective wrapper ---
    def func(X):
        X_real = _to_coco(X)
        vals = [problem(xr) for xr in X_real]   # COCO eval
        
        vals = np.asarray(vals)
        # Preserve behavior: your BO uses func(X_train) and func(x_next)
        if np.asarray(X).ndim == 1:
            return vals[0]   # scalar
        return vals          # vector

    # --- constraint wrappers ---
    n_con = problem.number_of_constraints
    constraints = []

    for j in range(n_con):
        def make_con(idx):
            def con(X):
                X_real = _to_coco(X)
                vals = [problem.constraints(xr)[idx] for xr in X_real]
                vals = np.asarray(vals)
                if np.asarray(X).ndim == 1:
                    return vals[0]  # scalar
                return vals        # vector
            return con

        constraints.append(make_con(j))

    return func, constraints
