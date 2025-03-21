import math
import pandas as pd

from libs.pytfa.optim.variables import (
    ForwardBackwardUseVariable,
    ForwardUseVariable,
    BackwardUseVariable,
)
from libs.pytfa.optim.constraints import ModelConstraint
from libs.pytfa.optim.utils import symbol_sum
from optlang.interface import INFEASIBLE

# =============================================================================
# MAIN FUNCTION: enumerate_minimal_networks_for_BBB
# =============================================================================


def enumerate_minimal_networks_for_BBB(
    model,
    bbb_metnames,
    bbb_not_to_lump,
    stoich_bbb,
    mu_max,
    num_alt_lumped,
):
    """
    Perform the lumping procedure for each biomass building block (BBB) and store results
    directly in a DataFrame.

    For each BBB (excluding those in bbb_not_to_lump), the function:
      - Restores the original reaction bounds.
      - Adjusts the demand reaction bounds: sets the DM_<BBB> reaction lower bound to
        floor(-mu_max * stoich_bbb[bbb]) (rounded to 5 decimals) and its upper bound to 10.
      - Solves the model.
      - Inspects the solution’s raw values for the BFUSE variables (assumed to be stored under
        keys of the form "BFUSE_<reaction_id>"). Reactions with a BFUSE value below 1e-9 are
        considered active in the lump.
          - If no reaction is active, the lump is labeled "ProducedByCore".
          - If the solution is empty or contains NaN, the lump is flagged as "NA" or "Crashed".
          - Otherwise, if alternatives are requested (num_alt_lumped > 1), an alternative
            generation function is called.
      - Appends a row to the results DataFrame with the BBB ID, the lumping solution (DPs),
        the list of active reaction IDs, and the corresponding reaction formulas.

    Returns:
        results_all_bbbs (pd.DataFrame): A DataFrame summarizing the lumped solutions for each BBB.
    """

    # Initialize a list to store rows (one per BBB).
    results_all_bbbs_list = []

    # Retrieve the forward-backward use variables.
    fb_use_variables = model._var_kinds[ForwardBackwardUseVariable.__name__]

    # Save original reaction bounds so each run starts from the same state.
    initial_bounds = {
        rxn.id: (rxn.lower_bound, rxn.upper_bound) for rxn in model.reactions
    }

    # Loop over each biomass building block.
    for bbb in bbb_metnames:
        print(f"Processing BBB: {bbb}")
        if bbb in bbb_not_to_lump:
            print(f"Skipping excluded BBB: {bbb}")
            continue

        # Restore reaction bounds.
        for rxn in model.reactions:
            rxn.lower_bound, rxn.upper_bound = initial_bounds[rxn.id]

        # Initialize a DataFrame to collect alternative results for this BBB and a list for cut constraints.
        results_per_bbb = pd.DataFrame()
        cut_constraints = []

        # Adjust the demand reaction for this BBB.
        dm_rxn_name = f"DM_{bbb.replace('-', '_')}"
        try:
            dm_rxn = model.reactions.get_by_id(dm_rxn_name)
        except KeyError:
            print(f"Warning: Demand reaction {dm_rxn_name} not found; skipping {bbb}.")
            continue

        # Set demand reaction bounds.
        coeff = stoich_bbb.get(bbb, 0)
        new_lb = math.floor(-mu_max * coeff * 1e5) / 1e5
        new_ub = 10
        dm_rxn.bounds = (new_lb, new_ub)

        # Solve the model.
        sol = model.optimize()
        current_DPs = None
        current_rxn_names = ["NA"]
        current_rxn_formulas = ["NA"]

        if (
            sol is None
            or not hasattr(sol, "raw")
            or sol.raw is None
            or len(sol.raw) == 0
        ):
            print(f"No solution for BBB: {bbb}")
        elif any(math.isnan(val) for val in sol.raw):
            print(f"Solution crashed for BBB: {bbb}")
            current_rxn_names = ["Crashed"]
            current_rxn_formulas = ["Crashed"]
        else:
            # Determine active reaction IDs based on BFUSE variable values.
            active_rxn_ids = []
            for rxn in model.reactions:
                bfuse_val = sol.raw.get(f"BFUSE_{rxn.id}", 1)
                if bfuse_val < 1e-9:
                    active_rxn_ids.append(rxn.id)

            if not active_rxn_ids:
                current_rxn_names = ["ProducedByCore"]
                current_rxn_formulas = ["ProducedByCore"]
            else:
                if num_alt_lumped > 1:
                    # Generate alternative solutions.
                    results_per_bbb, cut_constraints = get_alternatives(
                        model,
                        fb_use_variables,
                        sol,
                        pd.DataFrame(),
                        cut_constraints,
                        max_alternatives=num_alt_lumped,
                    )
                    current_DPs = results_per_bbb  # Store alternatives directly.
                else:
                    current_DPs = sol.x
                current_rxn_names = active_rxn_ids
                # Retrieve reaction formulas directly via the 'reaction' attribute.
                current_rxn_formulas = [
                    model.reactions.get_by_id(rxn_id).reaction
                    for rxn_id in current_rxn_names
                ]

        # Remove any cut constraints added during alternative generation.
        for constraint in cut_constraints:
            model = model.remove_cons(constraint)

        # Build a row for this BBB.
        row = {
            "BBB": bbb,
            "DPs": current_DPs,
            "Reaction_Names": current_rxn_names,
            "Reaction_Formulas": current_rxn_formulas,
        }
        results_all_bbbs_list.append(row)

    # Convert the list of rows into a DataFrame.
    results_all_bbbs = pd.DataFrame(results_all_bbbs_list)

    # Print summary for troubleshooting.
    print("Summary of lumped reactions:")
    print(results_all_bbbs)
    BBBsProducedByTheCore = results_all_bbbs.loc[
        results_all_bbbs["Reaction_Names"] == ["ProducedByCore"], "BBB"
    ].tolist()
    BBBsNotProduced = results_all_bbbs.loc[
        results_all_bbbs["Reaction_Names"] == ["NA"], "BBB"
    ].tolist()
    print("BBB produced by core (no lump):", BBBsProducedByTheCore)
    print("BBB with no lump produced:", BBBsNotProduced)

    return results_all_bbbs


# =============================================================================
# ALTERNATIVE GENERATION FUNCTION
# =============================================================================


def get_alternatives(
    model,
    fb_use_variables,
    initial_solution,
    results_df,
    cut_constraints,
    max_alternatives=10,
):
    """
    Generate alternative lumped solutions by adding a cut constraint that forces at least one
    BFUSE variable (associated with the currently active reactions) to change.

    The size of the lumped network is taken as the objective value of the initial solution,
    which serves as the target_size in subsequent optimizations.

    Parameters
    ----------
    model : cobra.Model or pytfa.ThermoModel
        The model to generate alternatives from.
    fb_use_variables : list
        List of BFUSE variables.
    initial_solution : optlang.interface.Solution
        The initial solution obtained from optimization.
    results_df : pd.DataFrame
        DataFrame to accumulate results.
    cut_constraints : list
        List to accumulate added cut constraints.
    max_alternatives : int, optional
        Maximum number of alternatives to generate.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame summarizing the alternatives generated.
    """
    alternative_number = 1
    # Start with the initial solution.
    solution = initial_solution
    target_size = solution.objective_value  # Use initial objective value as target.
    previous_raw_sets = set()
    previous_processed_sets = set()

    try:
        while (
            model.solver.status != INFEASIBLE and alternative_number <= max_alternatives
        ):
            print(f"\n--- Alternative {alternative_number} ---")

            # Get the raw active BFUSE variable names.
            raw_active_rxns_before = frozenset(
                v.name for v in fb_use_variables if solution.raw.get(v.name, 1) == 0
            )
            if raw_active_rxns_before in previous_raw_sets:
                # Try again: re-optimize with target_size.
                solution = model.optimize_custom_gurobi(target_size=target_size)
                raw_active_rxns_before = frozenset(
                    v.name for v in fb_use_variables if solution.raw.get(v.name, 1) == 0
                )
                if raw_active_rxns_before in previous_raw_sets:
                    raise RuntimeError(
                        f"Duplicate RAW alternative detected at alternative {alternative_number}."
                    )
            previous_raw_sets.add(raw_active_rxns_before)

            active_vars, active_rxn_ids_new, solution = get_active_vars_and_rxn_ids(
                model, solution, fb_use_variables
            )
            processed_active_rxns_after = frozenset(active_rxn_ids_new)
            if processed_active_rxns_after in previous_processed_sets:
                raise RuntimeError(
                    f"Duplicate alternative detected at alternative {alternative_number} after processing."
                )
            previous_processed_sets.add(processed_active_rxns_after)

            num_active_reactions = len(active_rxn_ids_new)
            num_total_internal_rxns = len(fb_use_variables)
            num_maximized_inactive = num_total_internal_rxns - num_active_reactions
            print(f"Minimal network size (active reactions): {num_active_reactions}")
            print(
                f"Number of inactive reactions maximized by solver: {num_maximized_inactive}"
            )
            if alternative_number == 1:
                min_size = num_active_reactions

            if num_active_reactions - min_size > 0:
                print(f"Stopping criteria reached at alternative {alternative_number}.")
                break

            current_df = generate_dataframe(model, solution, alternative_number)
            results_df = pd.concat([results_df, current_df])

            # Add the cut constraints
            model, cut_constraints = create_cut_constraint(
                model, active_vars, cut_constraints, alternative_number
            )
            solution = model.optimize_custom_gurobi(target_size=target_size)
            alternative_number += 1

    except Exception as main_error:
        print(f"Error during generation of alternatives: {main_error}")

    finally:
        print("\nExiting generate_alternatives.")
        print(f"Total alternatives generated: {alternative_number}")
        print(f"Final comprehensive DataFrame shape: {results_df.shape}")

    return results_df


# =============================================================================
# DATAFRAME GENERATION FUNCTION
# =============================================================================


def generate_dataframe(model, solution, alternative_number):
    """
    Generate a DataFrame summarizing the results for the model's reactions.

    Parameters:
        model (cobra.Model): The metabolic model.
        solution (cobra.Solution): The optimization solution.
        alternative_number (int): The current alternative number.

    Returns:
        pd.DataFrame: A DataFrame with fluxes, variable values, and other metrics for each reaction.
    """
    solution_raw = solution.raw
    solution_fluxes = solution.fluxes

    rxn_ids = [rxn.id for rxn in model.reactions]
    solution_status = solution.status

    data = {
        "Metabolite Flux": [solution_fluxes.get(rxn_id, None) for rxn_id in rxn_ids],
        "Forward Use": [
            solution_raw.get(f"{ForwardUseVariable.prefix}{rxn_id}", -1)
            for rxn_id in rxn_ids
        ],
        "Backward Use": [
            solution_raw.get(f"{BackwardUseVariable.prefix}{rxn_id}", -1)
            for rxn_id in rxn_ids
        ],
        "Forward-Backward Use": [
            solution_raw.get(f"{ForwardBackwardUseVariable.prefix}{rxn_id}", -1)
            for rxn_id in rxn_ids
        ],
        "Status": [solution_status] * len(rxn_ids),
        "Alternative": [alternative_number] * len(rxn_ids),
    }
    current_df = pd.DataFrame(data, index=rxn_ids)
    return current_df


# =============================================================================
# CUT CONSTRAINT FUNCTION
# =============================================================================


def create_cut_constraint(model, active_vars, cut_constraints, alternative_number):
    """
    Add a cut constraint to block the current solution from recurring.

    The constraint forces the sum of the active variables (typically the BFUSE variables)
    to be at most len(active_vars) - 1.

    Parameters:
        model (cobra.Model): The metabolic model.
        active_vars (list): List of active BFUSE variables.
        cut_constraints (list): List to store the added constraints.
        alternative_number (int): Identifier for the current alternative.

    Returns:
        model: The updated model.
        cut_constraints: The updated list of constraints.
    """
    new_constraint_expr = symbol_sum(active_vars)
    cut_constraint = model.add_constraint(
        kind=ModelConstraint,
        hook=model,
        expr=new_constraint_expr,
        lb=1,
        ub=len(active_vars),
        id_=f"cut_{alternative_number}",
    )
    cut_constraints.append(cut_constraint)
    model.repair()
    print(f"Cut constraint for alternative {alternative_number} added.")
    return model, cut_constraints


# =============================================================================
# GET ACTIVE VARIABLES FUNCTION
# =============================================================================


def get_active_vars_and_rxn_ids(model, solution, fb_use_variables, csv_filepath=None):
    """
    Extract active BFUSE variables and their associated reaction IDs from the solution.

    This function ensures that variables considered inactive (i.e. not near zero)
    are marked with value 1, and that any discrepancies with reaction fluxes are corrected.

    Returns:
        tuple: (active_vars, active_rxn_ids, solution)
            - active_vars: List of active BFUSE variables.
            - active_rxn_ids: List of reaction IDs (prefix removed).
            - solution: The updated solution object.
    """
    solution_raw = solution.raw
    solution_fluxes = solution.fluxes

    # Build a DataFrame for BFUSE variable values.
    fb_names = [v.name for v in fb_use_variables]
    fb_use_df = solution_raw.loc[fb_names]

    # Identify active indices: those with absolute value < 0.5.
    active_indices = set(fb_use_df[(fb_use_df.abs() < 0.5)].index)
    # Also, consider those strictly below 1.
    extended_active_indices = set(fb_use_df[(fb_use_df < 1)].index)

    if active_indices != extended_active_indices:
        print("Discrepancy detected in active indices due to solver precision.")
        diff_indices = extended_active_indices.difference(active_indices)
        valid_diff_indices = [
            idx
            for idx in diff_indices
            if abs(
                solution_fluxes.get(idx[len(ForwardBackwardUseVariable.prefix) :], 0)
            )
            > 1e-9
        ]
        active_indices = active_indices.union(valid_diff_indices)
        for idx in valid_diff_indices:
            solution.raw[idx] = 0  # Mark as active

    all_indices = set(fb_use_df.index)
    inactive_indices = all_indices.difference(active_indices)
    for idx in inactive_indices:
        solution.raw[idx] = 1  # Mark as inactive

    error_messages = []
    for idx in list(inactive_indices):
        rxn_id = idx[len(ForwardBackwardUseVariable.prefix) :]
        f_var = model.forward_use_variable.get_by_id(rxn_id)
        b_var = model.backward_use_variable.get_by_id(rxn_id)
        f_val = solution.raw.get(f_var.name, None)
        b_val = solution.raw.get(b_var.name, None)
        if f_val is None or b_val is None:
            msg = f"Missing forward/backward use value for reaction {rxn_id}."
            print(msg)
            error_messages.append(msg)
            continue
        if abs(f_val) > 1e-9 or abs(b_val) > 1e-9:
            msg = f"Error: Reaction {rxn_id} is inactive, yet forward use = {f_val} or backward use = {b_val} exceeds 1e-9."
            print(msg)
            error_messages.append(msg)
        flux_val = solution_fluxes.get(rxn_id, 0)
        if abs(flux_val) >= 1e-9:
            msg = f"Error: Reaction {rxn_id} is inactive but has flux {flux_val} (>= 1e-9). Updating to active."
            print(msg)
            error_messages.append(msg)
            solution.raw[idx] = 0
            active_indices.add(idx)

    inactive_indices = all_indices.difference(active_indices)

    if error_messages and csv_filepath is not None:
        final_values = solution.raw.loc[fb_use_df.index]
        final_values.to_csv(csv_filepath)
        print(
            f"Forward-backward use variable values saved to {csv_filepath} due to errors."
        )

    active_vars = [model.variables[idx] for idx in active_indices]
    active_rxn_ids = [
        var.name[len(ForwardBackwardUseVariable.prefix) :]
        for var in active_vars
        if var.name.startswith(ForwardBackwardUseVariable.prefix)
    ]

    return active_vars, active_rxn_ids, solution
