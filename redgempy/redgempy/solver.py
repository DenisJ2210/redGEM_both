# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solver Configuration Module for RedGEM.

This module provides utilities for configuring the solver for the GEM model
based on the user's preferences specified in redgem_opts.

Functions:
    - set_solver: Configure the solver for the GEM model.
"""


def set_solver(model, redgem_opts):
    """
    Configure solver settings for the GEM model.

    This function sets the solver and its configurations (e.g., tolerances,
    verbosity) for the given GEM model, using the solver specified in the
    `redgem_opts` dictionary.

    Args:
        model: The GEM model to configure.
        redgem_opts (dict): Dictionary of options, including:
            - cplex_parameters (str): Solver to use (e.g., "CplexDefault", "LCSBDefault", "custom").
            - time_limit_for_solver (str): Whether to impose a time limit ("yes" or "no").

    Returns:
        The configured GEM model.

    Raises:
        ValueError: If the specified solver is not supported.
    """
    # Extract solver preference from redgem_opts
    solver_preference = redgem_opts.get("cplex_parameters", "CplexDefault").lower()

    # Determine the solver to use
    if solver_preference in ["cplexdefault", "lcbsdefault", "custom"]:
        solver = "cplex"  # Default to cplex for these options
    elif solver_preference == "gurobi":
        solver = "gurobi"
    else:
        raise ValueError(f"Unsupported solver option: {solver_preference}")

    # Apply solver settings to the model
    model.solver = solver
    model.solver.configuration.tolerances.feasibility = 1e-9
    model.solver.configuration.tolerances.optimality = 1e-9
    model.solver.configuration.tolerances.integrality = 1e-9
    model.solver.configuration.presolve = True
    model.solver.configuration.verbosity = 3

    # Set a time limit if specified
    if redgem_opts.get("time_limit_for_solver", "no").lower() == "yes":
        model.solver.configuration.timeout = 3600  # Default to 1 hour
    else:
        model.solver.configuration.timeout = None

    return model
