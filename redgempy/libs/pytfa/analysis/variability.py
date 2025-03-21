# -*- coding: utf-8 -*-
"""
.. module:: pytfa
   :platform: Unix, Windows
   :synopsis: Thermodynamics-based Flux Analysis

.. moduleauthor:: pyTFA team

Variability analysis

"""

from copy import deepcopy
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import cobra
from cobra.core import Reaction
from optlang.exceptions import SolverError
from optlang.interface import INFEASIBLE
from tqdm import tqdm

from pytfa.optim import DeltaG
from pytfa.optim.constraints import ForbiddenProfile
from pytfa.optim.utils import get_direction_use_variables
from pytfa.optim.variables import ForwardUseVariable

CPU_COUNT = cpu_count()
BEST_THREAD_RATIO = int(CPU_COUNT / (4 * 2))  # Four proc per MILP instance,
# times two threads.


def find_bidirectional_reactions(va, tolerance=1e-8):
    """
    Returns the ids of reactions that can both carry net flux in the forward or
    backward direction.

    :param va: A variability analysis, pandas Dataframe like so:
                                          maximum       minimum
                6PGLter             -8.330667e-04 -8.330667e-04
                ABUTt2r              0.000000e+00  0.000000e+00
                ACALDt               0.000000e+00  0.000000e+00

    :return:
    """

    return va[va["minimum"] * va["maximum"] < -tolerance]


def find_directionality_profiles(
    tmodel, bidirectional, max_iter=1e4, solver="optlang-glpk"
):
    """
    Takes a ThermoModel and performs enumeration of the directionality profiles

    :param tmodel:
    :param max_iter:
    :return:
    """

    raise (NotImplementedError)

    this_tmodel = deepcopy(tmodel)
    this_tmodel.solver = solver
    profiles = dict()

    iter_count = 0

    bidirectional_reactions = this_tmodel.reactions.get_by_any(bidirectional)

    while this_tmodel.solver.status != INFEASIBLE and iter_count < max_iter:

        try:
            solution = this_tmodel.optimize()
        except SolverError:
            break

        profiles[iter_count] = solution.raw
        if iter_count > 0:
            sse = sum((profiles[iter_count - 1] - profiles[iter_count]) ** 2)
        else:
            sse = 0

        tmodel.logger.debug(str(iter_count) + " - " + str(sse))

        # active_use_variables = get_active_use_variables(this_tmodel,solution)
        active_use_variables = get_direction_use_variables(this_tmodel, solution)
        bidirectional_use_variables = [
            x for x in active_use_variables if x.reaction in bidirectional_reactions
        ]
        bool_id = "".join(
            "1" if isinstance(x, ForwardUseVariable) else "0"
            for x in bidirectional_use_variables
        )

        # Make the expression to forbid this expression profile to happen again
        # FP_1101: FU_rxn1 + FU_rxn2 + BU_rxn3 + FU_rxn4 <= 4-1 = 3
        expr = sum(bidirectional_use_variables)
        this_tmodel.add_constraint(
            ForbiddenProfile,
            hook=this_tmodel,
            expr=expr,
            id=str(iter_count) + "_" + bool_id,
            lb=0,
            ub=len(bidirectional_use_variables) - 1,
        )

        iter_count += 1

    return profiles, this_tmodel


def _bool2str(bool_list):
    """
    turns a list of booleans into a string

    :param bool_list: ex: '[False  True False False  True]'
    :return: '01001'
    """
    return "".join(["1" if x else "0" for x in bool_list])


def _variability_analysis_element(tmodel, var, sense):
    tmodel.objective = var
    tmodel.objective.direction = sense
    sol = tmodel.slim_optimize()
    return sol


def variability_analysis(tmodel, kind="reactions", proc_num=BEST_THREAD_RATIO):
    """
    Performs variability analysis, gicven a variable type

    :param tmodel:
    :param kind:
    :param proc_num:
    :return:
    """

    objective = tmodel.objective

    # If the kind variable is iterable, we perform variability analysis on each,
    # one at a time
    if hasattr(kind, "__iter__") and not isinstance(kind, str):
        va = {}
        for k in kind:
            va[k] = variability_analysis(tmodel, kind=k, proc_num=proc_num)
        df = pd.concat(va.values())
        return df
    elif kind == Reaction or (
        isinstance(kind, str) and kind.lower() in ["reaction", "reactions"]
    ):
        these_vars = {r.id: r for r in tmodel.reactions}
    else:
        these_vars = tmodel.get_variables_of_type(kind)
        these_vars = {x.name: x.variable for x in these_vars}

    tmodel.logger.info(
        "Beginning variability analysis for variable of type {}".format(kind)
    )

    results = {"min": {}, "max": {}}
    for sense in ["min", "max"]:
        for k, var in tqdm(these_vars.items(), desc=sense + "imizing"):
            tmodel.logger.debug(sense + "-" + k)
            results[sense][k] = _variability_analysis_element(tmodel, var, sense)

    tmodel.objective = objective
    df = pd.DataFrame(results)
    df.rename(columns={"min": "minimum", "max": "maximum"}, inplace=True)
    return df


def calculate_dissipation(tmodel, solution=None):
    if solution is None:
        solution = tmodel.solution

    reaction_id = [x.id for x in tmodel.reactions]
    fluxes = solution.fluxes[reaction_id]

    deltag_var = tmodel.get_variables_of_type(DeltaG)
    deltag = pd.Series({x.id: solution.raw[x.name] for x in deltag_var})
    dissipation = fluxes * deltag

    return dissipation


def fast_variability_analysis(tmodel, kind="reactions", proc_num=BEST_THREAD_RATIO):
    """
    Performs fast variability analysis using optimized solver calls and parallelization.

    Args:
        tmodel: ThermoModel object.
        kind (str): Type of variables for variability analysis (e.g., 'reactions').
        proc_num (int): Number of processes for parallel execution.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with 'minimum' and 'maximum' values for each variable.
            - list: List of problematic variables for which no conclusive results were obtained.
    """

    # Preserve original objective
    original_objective = tmodel.objective

    # Identify variables of interest
    if kind == "reactions":
        variables = {r.id: r for r in tmodel.reactions}
    else:
        variables = {x.name: x.variable for x in tmodel.get_variables_of_type(kind)}

    tmodel.logger.info(f"Starting variability analysis for {len(variables)} variables.")

    # Initialize results and problematic variable tracker
    results = []  # Store tuples of (variable name, min_bound, max_bound)
    problematic_variables = []  # List to track problematic variables

    def solve_for_bound(variable, sense):
        """
        Solve for the bound of a single variable using slim_optimize.
        If slim_optimize fails, fallback to optimize. Handle OptimizationError explicitly.
        """
        tmodel.objective = variable
        tmodel.objective.direction = sense
        try:
            bound = tmodel.slim_optimize()
            if np.isnan(bound):
                raise ValueError("slim_optimize returned NaN")
            return bound
        except (SolverError, ValueError):
            tmodel.logger.warning(
                f"Falling back to optimize for variable: {variable}, sense: {sense}"
            )
            try:
                solution = tmodel.optimize()
                if solution.status == "optimal":
                    return solution.objective_value
                else:
                    tmodel.logger.error(
                        f"Optimize failed to find a solution for variable: {variable}, sense: {sense}"
                    )
                    return np.nan
            except (SolverError, cobra.exceptions.OptimizationError) as e:
                tmodel.logger.error(
                    f"SolverError or OptimizationError for variable: {variable}, sense: {sense}. Error: {str(e)}"
                )
                return np.nan

    def compute_variable_bounds(var_name, variable):
        """
        Compute both bounds (min and max) for a single variable.
        Tracks problematic variables if neither slim_optimize nor optimize succeed.
        """
        tmodel.logger.debug(f"Processing {var_name}")
        min_bound = solve_for_bound(variable, "min")
        max_bound = solve_for_bound(variable, "max")

        if np.isnan(min_bound) or np.isnan(max_bound):
            tmodel.logger.warning(f"Adding {var_name} to problematic variables list.")
            problematic_variables.append(var_name)

        # Apply default bounds if necessary
        min_bound = min_bound if not np.isnan(min_bound) else -100
        max_bound = max_bound if not np.isnan(max_bound) else 100

        return var_name, min_bound, max_bound

    # Parallel execution using joblib
    results_list = Parallel(n_jobs=proc_num)(
        delayed(compute_variable_bounds)(var_name, variable)
        for var_name, variable in tqdm(variables.items(), desc="Processing variables")
    )

    # Collect results
    results.extend(results_list)

    # Restore original objective
    tmodel.objective = original_objective

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results, columns=["variable", "minimum", "maximum"])
    result_df.set_index("variable", inplace=True)

    tmodel.logger.info("Variability analysis completed.")
    tmodel.logger.info(f"Total problematic variables: {len(problematic_variables)}")

    return result_df, problematic_variables
