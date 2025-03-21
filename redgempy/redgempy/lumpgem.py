# lumpgem.py

import os
from os.path import join

import numpy as np
import pandas as pd
from cobra import Reaction

from pytfa.io import (
    apply_compartment_data,
    load_thermoDB,
    read_compartment_data,
)

from redgempy.gem_optimization import find_max_biomass_growth_rate
from redgempy.gem_processing import (
    identify_transport_reactions,
    remove_futile_reactions,
)
from redgempy.minimization_network import enumerate_minimal_networks_for_BBB

from libs.pytfa.thermo.tmodel import ThermoModel
from libs.pytfa.optim.variables import ForwardBackwardUseVariable
from libs.pytfa.optim.constraints import ReactionConstraint
from libs.pytfa.optim.utils import symbol_sum


def add_biomass(
    model,
    non_core_rxn_ids,
    thermo_dir_path,
    biomass_building_blocks_to_exclude,
    oxygen_metabolite_id,
    aerobic_anaerobic,
    organism,
    num_of_lumped,
    gem_name,
    rxn_names_prev_therm_relax,
    biomass_rxn_names,
    atp_synth_rxn_names,
    add_gam,
    percent_mu_max_lumping,
    impose_thermodynamics,
    output_path,
):
    """
    Main driver function for adding biomass lumped reactions.
    """
    # Step 1: Initialize parameters
    mu_max, model = initialize_parameters(model, percent_mu_max_lumping)

    # Step 2: Identify biomass building blocks
    biomass_building_blocks = identify_biomass_building_blocks(model)

    # Step 3: Preprocess the model
    lump_model, gam_metabolite, gam_stoich_coeff = preprocess_model(
        model,
        biomass_rxn_names,
        biomass_building_blocks,
        biomass_building_blocks_to_exclude,
        add_gam,
        gem_name,
        impose_thermodynamics,
        thermo_dir_path,
        align_transports_method="yesautomatic",
        non_core_rxn_ids=non_core_rxn_ids,
        oxygen_metabolite_id=oxygen_metabolite_id,
        aerobic_anaerobic=aerobic_anaerobic,
        atp_synth_rxn_names=atp_synth_rxn_names,
    )

    # Step 4: Exclude unwanted BBBs and set their bounds
    model, active_dm_ids = adjust_dm_drain_bounds(
        lump_model,
        biomass_building_blocks,
        biomass_building_blocks_to_exclude,
        mu_max,
        gam_metabolite,
        gam_stoich_coeff,
    )

    # Assume lump_model, biomass_building_blocks, gam_metabolite, gam_stoich_coeff are defined.
    stoich_bbb = compute_stoich_bbb(
        lump_model, biomass_building_blocks, gam_metabolite, gam_stoich_coeff
    )

    # Change the objecive fu

    # Step 5: Perform minimal networks identification (
    results_all_bbbs = enumerate_minimal_networks_for_BBB(
        lump_model,
        biomass_building_blocks,
        biomass_building_blocks_to_exclude,
        stoich_bbb,
        mu_max,
        10,
    )

    # process_minimal_networks_results(())

    # Step 6: Lump the minimal networks

    # lumped_results = lump_minimal_networks(results_all_bbbs, num_of_lumped)
    # processed_lumped_results = process_lumped_networks_results(results_all_bbbs)

    processed_lumped_results = []

    return processed_lumped_results


# Subfunctions
def initialize_parameters(model, percent_mu_max):
    """Initialize key parameters and variables."""
    model, max_growth_rate = find_max_biomass_growth_rate(model)
    max_growth_rate = max(0.1, np.floor(percent_mu_max / 100 * max_growth_rate))
    if max_growth_rate == 0:
        max_growth_rate = 0.1
    return max_growth_rate, model


def identify_biomass_building_blocks(model):

    biomass_rxn_metabolites = model.biomass_reaction.metabolites
    biomass_building_blocks = [
        metabolite.id
        for metabolite, coefficient in biomass_rxn_metabolites.items()
        if coefficient < 0
    ]
    return biomass_building_blocks


def preprocess_model(
    model,
    biomass_rxn_names,
    biomass_building_blocks,
    biomass_building_blocks_to_exclude,
    add_gam,
    gem_name,
    impose_thermodynamics,
    thermo_dir_path,
    align_transports_method,
    non_core_rxn_ids,
    oxygen_metabolite_id,
    aerobic_anaerobic,
    atp_synth_rxn_names,
):
    """
    Preprocess the model by adding demand reactions, thermodynamic constraints, and filtering.
    """
    # Add drains for biomass building blocks
    model = add_drain_for_biomass_building_blocks(
        model, biomass_building_blocks, biomass_building_blocks_to_exclude
    )

    gam_metabolite = None
    gam_stoich_coeff = None

    # Add GAM if required
    if add_gam:
        model, gam_metabolite, gam_stoich_coeff = add_gam_to_model(model)

    # Convert to MILP or impose thermodynamic constraints
    model = thermo_or_milp_conversion(
        model, gem_name, impose_thermodynamics, thermo_dir_path
    )

    ##################################
    # CHECKPOINT 1 in MATLAB
    ##################################

    # Block biomass production
    model.biomass_reaction.bounds = (0, 0)

    # Align transport reactions
    model = align_transport_reactions(
        model,
        align_transports_method=align_transports_method,
        biomass_reaction_names=biomass_rxn_names,
        atp_synth_reaction_names=atp_synth_rxn_names,
        check_growth=0,
    )

    model = set_aerobic_anaerobic_conditions(
        model, oxygen_metabolite_id, aerobic_anaerobic
    )

    ##################################
    # CHECKPOINT 2 in MATLAB
    ##################################

    # Identify non-core, non-transport, non-exchange reactions
    filtered_rxn_ids = identify_non_core_non_transport_non_boundary_reactions(
        model,
        non_core_rxn_ids,
        biomass_building_blocks,  # Potentially problematic as it mixes the hook of the rxns, i.e. the attribute _model of each reaction object! (see if model.repair() solves this)
    )

    # Add binary variables and set constraints
    model, binary_vars, added_constraints = add_binary_variables_and_constraints(
        model, filtered_rxn_ids
    )

    # Change the objective function to the maximization of the BFUSE variables (i.e., maximization of the number of inactive reactions)
    fb_use_variables = model._var_kinds[ForwardBackwardUseVariable.__name__]
    model.objective = symbol_sum(fb_use_variables)
    model.objective_direction = "max"

    ##################################
    # CHECKPOINT 3 in MATLAB
    ##################################

    return model, gam_metabolite, gam_stoich_coeff


def add_drain_for_biomass_building_blocks(
    model, biomass_building_blocks, biomass_building_blocks_to_exclude
):

    # create outward drains for substrates of biomass rxn (if the outward drain for the substrate of biomass can carry flux, this means that the substrate is indeed available for the biomass reaction to work)
    for metabolite_id in biomass_building_blocks:
        if metabolite_id in biomass_building_blocks_to_exclude:
            continue

        # Access the metabolite object
        metabolite = model.metabolites.get_by_id(metabolite_id)

        # Create a unique demand reaction name
        demand_rxn_name = f"DM_{metabolite_id.replace('-', '_')}"
        stoich_coeff = -1  # Outward for substrates, default behavior for COBRApy

        # Check if a demand reaction already exists
        try:
            existing_rxn = model.reactions.get_by_id(demand_rxn_name)
            # Update bounds of existing demand reaction to [0, 100]
            print(f"Updating bounds for existing reaction: {existing_rxn.id}")
            existing_rxn.bounds = (0, 100)
        except KeyError:
            # Create a new demand reaction
            demand_rxn = Reaction(demand_rxn_name)
            demand_rxn.name = f"Demand for {metabolite.name}"
            demand_rxn.subsystem = "Demand"
            demand_rxn.lower_bound = 0  # Allow only outward flux
            demand_rxn.upper_bound = 60
            demand_rxn.add_metabolites({metabolite: stoich_coeff})

            # Add reaction to the model
            model.add_reactions([demand_rxn])
            print(f"Added new demand reaction: {demand_rxn.id}")

    model.repair()

    return model


def add_gam_to_model(model):
    """
    Add Growth-Associated Maintenance (GAM) reaction to the model.

    Args:
        model (cobra.Model): The genome-scale metabolic model.

    Returns:
        cobra.Model: The model with the GAM reaction added.
    """
    # Extract GAM coefficients from biomass reaction
    gam_equation, ppi_equation = extract_gam_coefficients_from_biomass(model)

    # Add GAM reaction to represent the energy required for cellular maintenance
    gam_reaction = Reaction("DM_GAM_c")
    gam_reaction.name = "Demand for Growth-Associated Maintenance"
    gam_reaction.lower_bound = (
        0  # The reaction can only proceed in one direction (outward flux)
    )
    gam_reaction.upper_bound = 0  # Bounds fixed as in the MATLAB code
    gam_reaction.add_metabolites(
        gam_equation
    )  # Add GAM metabolites with extracted coefficients

    # Add the GAM reaction to the model
    model.add_reactions([gam_reaction])

    # Track GAM-related metadata (optional, for further analysis)
    gam_metabolite_id = "GAM_c"  # Consistent with MATLAB logic
    gam_stoich_coeff = -1  # Placeholder for further constraints if needed

    # Repair
    model.repair()

    return model, gam_metabolite_id, gam_stoich_coeff


def extract_gam_coefficients_from_biomass(model):
    """
    Extract the coefficients for the Growth-Associated Maintenance (GAM) reaction from the biomass reaction.

    Args:
        model (cobra.Model): The genome-scale metabolic model.

    Returns:
        tuple:
            dict: GAM equation as a dictionary of metabolites and their stoichiometric coefficients.
            dict: PPI equation as a dictionary of metabolites and their stoichiometric coefficients.

    Raises:
        ValueError: If any GAM or PPI metabolites are missing in the biomass reaction.
    """
    # Identify the biomass reaction
    biomass_rxn = model.biomass_reaction
    biomass_coefficients = biomass_rxn.metabolites

    # Define the IDs of GAM and PPI metabolites
    gam_mets = ["atp_c", "h2o_c", "adp_c", "h_c", "pi_c"]
    ppi_mets = ["ppi_c"]

    # Extract coefficients for GAM metabolites
    gam_metabolites = {
        met: coeff for met, coeff in biomass_coefficients.items() if met.id in gam_mets
    }
    missing_gam_mets = [
        met for met in gam_mets if met not in [m.id for m in gam_metabolites]
    ]
    if missing_gam_mets:
        raise ValueError(
            f"The following GAM metabolites are missing or have zero coefficients in the biomass reaction: {missing_gam_mets}"
        )

    # Extract coefficients for PPI metabolites
    ppi_metabolites = {
        met: coeff for met, coeff in biomass_coefficients.items() if met.id in ppi_mets
    }
    missing_ppi_mets = [
        met for met in ppi_mets if met not in [m.id for m in ppi_metabolites]
    ]
    if missing_ppi_mets:
        raise ValueError(
            f"The following PPI metabolites are missing or have zero coefficients in the biomass reaction: {missing_ppi_mets}"
        )

    # Convert coefficients to absolute values (matching MATLAB logic)
    gam_equation = {met: abs(coeff) for met, coeff in gam_metabolites.items()}
    ppi_equation = {met: abs(coeff) for met, coeff in ppi_metabolites.items()}

    return gam_equation, ppi_equation


def thermo_or_milp_conversion(
    model, model_name, impose_thermodynamics, thermo_dir_path
):
    """
    Convert the model to include thermodynamic constraints or MILP variables.

    Args:
        model (cobra.Model): The genome-scale metabolic model.
        impose_thermodynamics (bool): Whether to impose thermodynamic constraints.
        thermo_dir_path (str): Path to the directory containing thermodynamic data.

    Returns:
        cobra.Model: The model with thermodynamic constraints or MILP variables.
    """
    if impose_thermodynamics == "yes":
        print("Attempting to impose thermodynamic constraints...")

        try:
            # Load thermodynamic database
            thermo_data_path = join(
                thermo_dir_path, "thermo_data.json"
            )  # Adjust file name as needed
            if not os.path.exists(thermo_data_path):
                raise FileNotFoundError(
                    f"Thermodynamic data file not found at: {thermo_data_path}"
                )
            thermo_data = load_thermoDB(thermo_data_path)

            # Load compartment data
            compartment_data_path = join(
                thermo_dir_path, "compartment_data.json"
            )  # Adjust file name as needed
            if not os.path.exists(compartment_data_path):
                raise FileNotFoundError(
                    f"Compartment data file not found at: {compartment_data_path}"
                )
            compartment_data = read_compartment_data(compartment_data_path)

            # Initialize ThermoModel
            tfa_model = ThermoModel(
                model=model, thermo_data=thermo_data, name=model_name
            )

            # Apply compartment data
            apply_compartment_data(tfa_model, compartment_data)

            # Prepare and convert model to TFA
            tfa_model.prepare()
            tfa_model.convert()

            print("Thermodynamic constraints successfully imposed.")
            model = tfa_model

        except Exception as e:
            print(f"Error imposing thermodynamic constraints: {e}")
            raise
    else:
        print(
            "Skipping thermodynamic constraints. Adding integer variables for MILP optimization."
        )
        # Add MILP structure if thermodynamics are not imposed
        tfa_model = ThermoModel(model=model, thermo_data=[], name=model_name)
        apply_compartment_data(tmodel=tfa_model, compartment_data=[])
        tfa_model.prepare()
        tfa_model.convert()

    return tfa_model


def align_transport_reactions(
    model,
    align_transports_method,
    biomass_reaction_names,
    atp_synth_reaction_names,
    check_growth,
):
    """
    Align transport reactions in the metabolic model by identifying and constraining
    parallel transport reactions to improve model consistency.

    Args:
        model (cobra.Model): The metabolic model to process.
        align_transports_method (str): Method for aligning transport reactions:
            - "yesautomatic": Automatically identify and align parallel transport reactions.
            - "no": Skip transport alignment.
        biomass_reaction_names (list): List of biomass reaction IDs.
        atp_synth_reaction_names (list): List of ATP synthesis reaction IDs.
        check_growth (bool): Whether to verify growth after adding constraints.

    Returns:
        cobra.Model: The updated model with aligned transport reactions.

    Raises:
        ValueError: If an invalid method for aligning transport reactions is provided.
    """
    if align_transports_method == "yesautomatic":
        print("Aligning parallel transports using automated detection...")

        # Identify lumped and biomass reactions
        lumped_reaction_ids = [
            rxn.id for rxn in model.reactions if rxn.id.startswith("LMPD")
        ]
        biomass_reaction_ids = [
            rxn for rxn in biomass_reaction_names if rxn in model.reactions
        ]
        reactions_to_exclude = set(lumped_reaction_ids + biomass_reaction_ids)

        # Remove excluded reactions directly from the model
        model.remove_reactions(list(reactions_to_exclude), remove_orphans=False)

        # Repair
        model.repair()

        # Identify unused metabolites and remove them
        unused_metabolites = [
            met for met in model.metabolites if len(met.reactions) == 0
        ]
        if unused_metabolites:
            model.remove_metabolites(unused_metabolites, destructive=False)

        # Repair
        model.repair()

        # Identify transport reactions
        transport_info = identify_transport_reactions(
            model, biomass_reaction_names, atp_synth_reaction_names
        )

        # Extract relevant transport reaction data.
        # 'important_transports' is expected to be a list of tuples:
        # (reaction_id, metabolite, compartment, direction)
        important_transports = transport_info["important_transports"]

        # Prepare reaction pairs (formatted similarly to the MATLAB version)
        rxn_pairs = [
            (entry[0], entry[1], entry[2], entry[3]) for entry in important_transports
        ]

        # Remove futile reactions based on transport constraints
        model = remove_futile_reactions(model, rxn_pairs, check_growth)

    elif align_transports_method == "no":
        print("Skipping transport alignment...")

    else:
        raise ValueError(
            f"Invalid method for aligning transport reactions: {align_transports_method}"
        )

    return model


def set_aerobic_anaerobic_conditions(model, oxygen_metabolite_id, aerobic_anaerobic):
    """
    Set aerobic or anaerobic conditions for a metabolic model.

    This function identifies the oxygen exchange reaction in the model and sets its
    bounds based on the specified condition.

    Args:
        model (cobra.Model): The genome-scale metabolic model.
        oxygen_metabolite_id (str): The metabolite ID for oxygen (e.g., "o2_e").
        condition (str): The condition to set, either "aerobic" or "anaerobic".

    Returns:
        cobra.Model: The updated model with the oxygen exchange bounds adjusted.

    Raises:
        ValueError: If the specified condition is not "aerobic" or "anaerobic".
    """
    # Identify the oxygen exchange reaction
    oxygen_exchange_reaction = None
    for reaction in model.reactions:
        metabolites = reaction.metabolites
        if (
            len(metabolites) == 1
        ):  # Ensure it's an exchange reaction with only one metabolite
            only_metabolite = next(
                iter(metabolites)
            )  # Get the single metabolite efficiently
            if only_metabolite.id == oxygen_metabolite_id:
                oxygen_exchange_reaction = reaction
                break

    if not oxygen_exchange_reaction:
        raise ValueError(
            f"No exchange reaction found for metabolite: {oxygen_metabolite_id}"
        )

    # Adjust bounds based on the specified condition
    if aerobic_anaerobic == "aerobic":
        oxygen_exchange_reaction.lower_bound = -20  # Allow oxygen uptake
        print(f"Set model to aerobic conditions: {oxygen_exchange_reaction.id}")
    elif aerobic_anaerobic == "anaerobic":
        oxygen_exchange_reaction.lower_bound = 0  # No oxygen uptake
        print(f"Set model to anaerobic conditions: {oxygen_exchange_reaction.id}")
    else:
        raise ValueError(
            f"Invalid condition: {aerobic_anaerobic}. Must be 'aerobic' or 'anaerobic'."
        )

    return model


def identify_non_core_non_transport_non_boundary_reactions(
    model, non_core_rxn_ids, biomass_building_blocks
):
    """
    Identify reactions that are non-core, not transport, and not exchange.

    Args:
        model (cobra.Model): The metabolic model.
        non_core_rxn_ids (list): IDs of non-core reactions.
        biomass_building_blocks (list): IDs of biomass building block metabolites.

    Returns:
        list: IDs of non-core, non-transport, and non-exchange reactions.
    """

    # Identify core reaction IDs by excluding the non-core ones.
    core_rxn_ids = [rxn.id for rxn in model.reactions if rxn.id not in non_core_rxn_ids]

    # Identify the metabolites present in the network composed only of core reactions.
    core_mets = extract_subnetwork_metabolites(model, core_rxn_ids)
    core_mets_ids = [met.id for met in core_mets]

    # Add the biomass building blocks to the core metabolites.
    core_mets_ids += biomass_building_blocks

    # Annotate the model with transport reaction attributes using the updated core metabolites.
    model = annotate_transport_reactions(model, core_mets_ids)

    # Identify transport reactions.
    transport_rxn_ids = [rxn.id for rxn in model.reactions if rxn.is_transport]

    # Identify exchange (boundary) reactions.
    boundary_rxn_ids = [rxn.id for rxn in model.reactions if rxn.is_boundary]

    # Filter non-core reactions by removing those that are transport or exchange.
    non_core_non_transport_non_boundary_rxn_ids = [
        rxn_id
        for rxn_id in non_core_rxn_ids
        if rxn_id not in transport_rxn_ids and rxn_id not in boundary_rxn_ids
    ]

    return non_core_non_transport_non_boundary_rxn_ids


def add_binary_variables_and_constraints(model, reaction_ids):
    """
    Add binary variables for minimization and constraints of the form FUSE + BUSE + BFUSE ≤ 1.

    Args:
        model (pytfa.ThermoModel): The thermodynamic model.
        reaction_ids (list): List of reaction IDs to process.

    Returns:
        dict: Dictionary of added binary variables for each reaction.
    """
    # Initialize storage for added variables and constraints
    added_variables = {}
    added_constraints = []

    # Process each reaction ID
    for rxn_id in reaction_ids:

        # Retrieve forward and backward use variables
        try:
            rxn = model.reactions.get_by_id(rxn_id)

            forward_use = model.forward_use_variable.get_by_id(rxn_id)
            backward_use = model.backward_use_variable.get_by_id(rxn_id)

            # Add the binary variable (BFUSE)
            fb_use_var = model.add_variable(
                kind=ForwardBackwardUseVariable, hook=rxn
            )  # Use the appropriate kind for PyTFA
            added_variables[rxn_id] = fb_use_var

            # Add the constraint FUSE + BUSE + BFUSE ≤ 1
            constraint_expr = forward_use + backward_use + fb_use_var
            constraint = model.add_constraint(
                kind=ReactionConstraint,  # Use the appropriate kind for PyTFA
                hook=rxn,
                expr=constraint_expr,
                lb=0,
                ub=1,
                id_=f"BFMER5_{rxn_id}",
            )
            added_constraints.append(constraint)

        except KeyError:
            print(f"Reaction ID {rxn_id} not found in the model.")

    # Repair the model to integrate changes
    model.repair()

    return model, added_variables, added_constraints


import math


def adjust_dm_drain_bounds(
    model,
    biomass_building_blocks,
    biomass_building_blocks_to_exclude,
    mu_max,
    gam_metabolite,
    gam_stoich_coeff,
):
    """
    Adjust the upper bounds for demand (DM) reactions corresponding to biomass building blocks (BBBs)
    based on the maximum growth rate (mu_max) and the stoichiometric coefficients in the biomass reaction.

    In this Python version, we assume that each biomass building block (and GAM) already has a demand
    reaction in the model named "DM_<metabolite_id>" (with hyphens replaced by underscores).

    For each metabolite:
      - If the metabolite is in the exclusion list, its demand reaction upper bound is set to zero.
      - Otherwise, the upper bound is set to ceil(-mu_max * stoichiometric_coefficient).

    Parameters
    ----------
    model : cobra.Model
        The COBRApy model, which must have a biomass reaction accessible via model.biomass_reaction.
    biomass_building_blocks : list of str
        List of metabolite IDs (e.g., ['met1', 'met2', ...]) considered as biomass building blocks.
    biomass_building_blocks_to_exclude : list of str
        List of metabolite IDs that should be excluded from lumping.
    mu_max : float
        The maximum growth rate (μₘₐₓ) used to scale the flux bounds.
    gam_metabolite : str
        The metabolite ID for Growth-Associated Maintenance (GAM), e.g., "GAM_c".
    gam_stoich_coeff : float
        The stoichiometric coefficient for GAM in the biomass reaction.

    Returns
    -------
    model : cobra.Model
        The updated model with adjusted upper bounds on demand (DM) reactions.
    active_dm_ids : list of str
        List of DM reaction IDs that are allowed to carry flux (i.e., not forced to zero).
    """
    # Define the exclusion set, including water and GAM, plus any additional exclusions.
    excluded_metabolites = set(
        ["h2o_c", gam_metabolite] + biomass_building_blocks_to_exclude
    )

    # Combine the BBBs with GAM into one list.
    all_metabolites = biomass_building_blocks + [gam_metabolite]

    active_dm_ids = []

    # Get the biomass reaction from the model.
    biomass_rxn = model.biomass_reaction

    for met in all_metabolites:
        # Construct the demand reaction name: DM_<met>, replacing hyphens with underscores.
        dm_rxn_name = f"DM_{met.replace('-', '_')}"

        try:
            # Get the corresponding demand reaction.
            dm_rxn = model.reactions.get_by_id(dm_rxn_name)
        except KeyError:
            print(
                f"Warning: Demand reaction {dm_rxn_name} not found in the model; skipping."
            )
            continue

        # Retrieve the stoichiometric coefficient from the biomass reaction.
        try:
            met_obj = model.metabolites.get_by_id(met)
            coeff = biomass_rxn.metabolites.get(met_obj, None)
        except KeyError:
            coeff = None

        # For GAM, if the coefficient is not found in the biomass reaction, use gam_stoich_coeff.
        if met == gam_metabolite and (coeff is None or coeff == 0):
            coeff = gam_stoich_coeff
        elif coeff is None:
            print(
                f"Warning: Metabolite {met} not present in the biomass reaction; using 0."
            )
            coeff = 0

        # If the metabolite is in the exclusion set, set its demand reaction upper bound to zero.
        if met in excluded_metabolites:
            dm_rxn.upper_bound = 0.0
        else:
            # Compute the new bound as ceil(-mu_max * coeff).
            new_bound = math.ceil(-mu_max * coeff)
            dm_rxn.upper_bound = float(new_bound)
            active_dm_ids.append(dm_rxn_name)

    return model, active_dm_ids


def extract_subnetwork_metabolites(model, rxn_ids, met_names=None):
    """
    Extract the list of metabolites involved in the given reactions.

    Parameters
    ----------
    model : cobra.Model
        The COBRApy model.
    rxn_ids : list of str
        A list of reaction IDs defining the subnetwork.
    met_names : list of str, optional
        If provided, only metabolites with these IDs are returned.

    Returns
    -------
    list of cobra.Metabolite
        A list of metabolite objects present in the subnetwork.
    """
    # Filter reactions that are part of the subnetwork
    subnetwork_rxns = [rxn for rxn in model.reactions if rxn.id in rxn_ids]

    # Use a set to avoid duplicates
    mets = set()
    for rxn in subnetwork_rxns:
        mets.update(rxn.metabolites.keys())

    if met_names is not None:
        met_names_set = set(met_names)
        mets = {met for met in mets if met.id in met_names_set}

    return list(mets)


def annotate_transport_reactions(model, core_metabolite_ids):
    """
    Annotate reactions in a COBRApy model with transport-related flags.

    Each reaction is updated with the following attributes:
      - is_transport:         1 if the reaction is considered a transport reaction
                              (i.e. it has duplicate "cleaned" metabolite names or spans multiple compartments), else 0.
      - is_core_transport:    1 if the reaction is a transport AND all its original metabolite IDs
                              are in the expanded core list, else 0.
      - is_exchange_transport:1 if the reaction involves the 'e' compartment, else 0.
      - extra_enzymatic:      1 if the reaction involves >1 metabolite (after cleaning) but only one compartment,
                              which must be 'e', else 0.
      - is_boundary:          1 if the reaction is identified as a boundary reaction, else 0.

    Parameters
    ----------
    model : cobra.Model
        The COBRApy model.
    core_metabolite_ids : list of str
        A list of core metabolite IDs (with potential compartment suffixes).

    Returns
    -------
    cobra.Model
        The model with reactions annotated with the custom flags.
    """
    # Determine the unique compartments present in the model.
    unique_compartments = {met.compartment for met in model.metabolites}

    # Clean the core metabolite IDs by removing any '_<compartment>' suffix.
    cleaned_core = []
    for met_id in core_metabolite_ids:
        base = met_id
        for comp in unique_compartments:
            base = base.replace(f"_{comp}", "")
        cleaned_core.append(base)

    # Expand the core list by appending each compartment to each cleaned core name.
    expanded_core = []
    for comp in unique_compartments:
        for base in cleaned_core:
            expanded_core.append(f"{base}_{comp}")

    # Define suffixes to remove from metabolite IDs when cleaning reaction metabolites.
    suffixes = [
        "_c",
        "_p",
        "_e",
        "_g",
        "_l",
        "_m",
        "_n",
        "_r",
        "_s",
        "_x",
        "_j",
        "_o",
        "_q",
        "_t",
        "_v",
        "_w",
    ]

    # Annotate each reaction.
    for rxn in model.reactions:
        # Get participating metabolites (as a list of metabolite objects).
        rxn_met_objs = list(rxn.metabolites.keys())
        rxn_met_ids = [met.id for met in rxn_met_objs]
        rxn_compartments = [met.compartment for met in rxn_met_objs]

        # Clean metabolite IDs by removing known suffixes.
        rxn_met_ids_clean = []
        for met_id in rxn_met_ids:
            clean_id = met_id
            for suf in suffixes:
                clean_id = clean_id.replace(suf, "")
            rxn_met_ids_clean.append(clean_id)

        # Compute unique cleaned names and unique compartments.
        unique_clean = set(rxn_met_ids_clean)
        unique_comps = set(rxn_compartments)

        # Initialize flags.
        rxn.is_transport = 0
        rxn.is_core_transport = 0
        rxn.is_exchange_transport = 0
        rxn.extra_enzymatic = 0

        # A reaction is considered transport if it has duplicate cleaned names or spans >1 compartment.
        if (len(rxn_met_ids_clean) != len(unique_clean)) or (len(unique_comps) > 1):
            rxn.is_transport = 1

            # Mark as core transport if every original metabolite ID is in the expanded core list.
            if all(met in expanded_core for met in rxn_met_ids):
                rxn.is_core_transport = 1

            # Mark as exchange transport if compartment 'e' is present.
            rxn.is_exchange_transport = 1 if "e" in unique_comps else 0
        else:
            rxn.is_transport = 0
            rxn.is_exchange_transport = 0

        # Flag extra enzymatic: if >1 cleaned metabolite but only one compartment and that compartment is 'e'.
        if (len(rxn_met_ids_clean) > 1) and (len(unique_comps) == 1):
            only_comp = next(iter(unique_comps))
            rxn.extra_enzymatic = 1 if only_comp == "e" else 0
        else:
            rxn.extra_enzymatic = 0

    # Instead of using model.demands, now extract boundary reactions.
    boundary_rxns, _ = extract_boundary_rxns(model)
    for rxn in model.reactions:
        rxn.is_boundary = 1 if rxn.id in boundary_rxns else 0

    return model


def extract_boundary_rxns(model, boundary_filter=None):
    """
    Identify boundary reactions in the model.

    A boundary reaction (e.g. an exchange reaction) is defined as one that involves exactly one metabolite.
    If the stoichiometric coefficient for that metabolite is 1, then the reaction's stoichiometry is flipped
    (the coefficient becomes -1) and the reaction bounds are swapped (with sign inversion).

    Parameters
    ----------
    model : cobra.Model
        The COBRApy model.
    boundary_filter : iterable of str, optional
        If provided, only reactions with IDs in this collection will be returned.

    Returns
    -------
    boundary_rxns : list of str
        A list of reaction IDs for the identified boundary reactions.
    boundary_mets : list of cobra.Metabolite
        The corresponding list of metabolite objects involved in these reactions.
    """
    boundary_rxns = []
    boundary_mets = []

    for rxn in model.reactions:
        # Consider only reactions with exactly one metabolite.
        if len(rxn.metabolites) == 1:
            met = next(iter(rxn.metabolites))
            boundary_rxns.append(rxn.id)
            boundary_mets.append(met)
            coeff = rxn.metabolites[met]
            # If the coefficient is 1, flip the stoichiometry and swap bounds.
            if coeff == 1:
                try:
                    rxn.modify_coefficient(met, -1)
                except AttributeError:
                    # Fallback: remove and re-add with negative coefficient.
                    rxn.add_metabolites({met: -coeff}, combine=False)
                old_lb, old_ub = rxn.lower_bound, rxn.upper_bound
                rxn.lower_bound = -old_ub
                rxn.upper_bound = -old_lb

    # If a boundary_filter is provided, filter the boundary_rxns.
    if boundary_filter is not None:
        boundary_filter = set(boundary_filter)
        filtered = [
            (rxn_id, met)
            for rxn_id, met in zip(boundary_rxns, boundary_mets)
            if rxn_id in boundary_filter
        ]
        if filtered:
            boundary_rxns, boundary_mets = zip(*filtered)
            boundary_rxns = list(boundary_rxns)
            boundary_mets = list(boundary_mets)
        else:
            boundary_rxns, boundary_mets = [], []

    return boundary_rxns, boundary_mets


def compute_stoich_bbb(model, bbb_metnames, gam_metabolite, gam_stoich_coeff):
    """
    Compute the stoichiometric coefficients for the biomass building blocks (BBBs)
    from the biomass reaction in the model.

    Parameters
    ----------
    model : cobra.Model
        The COBRApy model (which must have a biomass reaction accessible via model.biomass_reaction).
    bbb_metnames : list of str
        List of metabolite IDs considered as biomass building blocks.
    gam_metabolite : str
        The metabolite ID for Growth-Associated Maintenance (e.g., "GAM_c").
    gam_stoich_coeff : float
        The stoichiometric coefficient for GAM in the biomass reaction.

    Returns
    -------
    stoich_bbb : dict
        A dictionary mapping each BBB metabolite ID to its stoichiometric coefficient in the biomass reaction.
        The GAM metabolite is also included.
    """
    biomass_rxn = model.biomass_reaction
    stoich_bbb = {}

    # Iterate over the list of BBB metabolite IDs.
    for met_id in bbb_metnames:
        coeff = None
        # Loop over the metabolites in the biomass reaction.
        for met_obj, c in biomass_rxn.metabolites.items():
            if met_obj.id == met_id:
                coeff = c
                break
        if coeff is None:
            print(
                f"Warning: Metabolite {met_id} not found in biomass reaction; setting coefficient to 0."
            )
            stoich_bbb[met_id] = 0
        else:
            stoich_bbb[met_id] = coeff

    # Include GAM: if not found in the biomass reaction, use the provided gam_stoich_coeff.
    stoich_bbb[gam_metabolite] = gam_stoich_coeff
    return stoich_bbb
