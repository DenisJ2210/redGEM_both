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
GEM Processing Utilities for RedGEM.

This module provides functions for loading and processing genome-scale metabolic models (GEMs)
to prepare them for the RedGEM workflow.

Functions:
    - load_gem: Load the genome-scale metabolic model (GEM).
    - prevent_bbb_uptake: Prevent uptake of biomass building blocks in the GEM.
    - add_etc_subsystem: Add electron transport chain (ETC) as a core subsystem.
    - add_extracellular_subsystem: Add extracellular metabolites as a core subsystem.
    - remove_thermo_fields: Remove thermodynamic fields from the GEM.
"""

import os
from itertools import combinations
from cobra.io import load_json_model
from cobra.io.mat import load_matlab_model
from libs.pytfa.optim.constraints import ModelConstraint


def load_gem(redgem_opts):
    """
    Load the genome-scale metabolic model (GEM) from a file.

    This function supports loading from both MATLAB (.mat) and JSON (.json) files.

    Args:
        redgem_opts (dict): Dictionary of options containing:
            - gem_name (str): Name of the GEM file (without extension).
            - file_extension (str, optional): File extension for the GEM file (default: ".json").
            - key (str, optional): Variable name of the model in the .mat file.
            - output_path (str): Path to the directory containing GEM files.

    Returns:
        cobra.Model: The loaded GEM.

    Raises:
        FileNotFoundError: If the specified GEM file does not exist.
        ValueError: If the file format is unsupported.
    """
    # Extract options
    gem_name = redgem_opts.get("gem_name", "")
    file_extension = redgem_opts.get("file_extension", ".json").lower()
    key = redgem_opts.get("key", None)

    # Construct the full path to the GEM file in `redgempy/models`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    gem_path = os.path.join(models_dir, f"{gem_name}{file_extension}")

    if not os.path.exists(gem_path):
        raise FileNotFoundError(f"GEM file not found at path: {gem_path}")

    # Determine the file type and load the model accordingly
    try:
        if file_extension == ".mat":
            print(f"Loading GEM from MATLAB file: {gem_path} with variable name: {key}")
            return load_matlab_model(gem_path, variable_name=key)
        elif file_extension == ".json":
            print(f"Loading GEM from JSON file: {gem_path}")
            return load_json_model(gem_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise RuntimeError(f"Error loading GEM file: {gem_path}") from e


def identify_biomass_reaction(model):
    """
    Identify the biomass reaction(s) in the GEM.

    This function searches for potential biomass reactions based on common naming conventions
    and characteristics, then asks the user to validate which reactions are truly biomass reactions.
    Finally, the user selects one biomass reaction to be used for the rest of the code.

    Args:
        model (cobra.Model): The GEM to search.

    Returns:
        tuple: (selected_biomass_reaction, valid_biomass_reactions_ids)
            - selected_biomass_reaction (cobra.Reaction): The reaction chosen for the rest of the code.
            - valid_biomass_reactions_ids (list): List of validated biomass reaction IDs.
    """
    # Common naming conventions for biomass reactions
    common_biomass_names = ["biomass", "growth"]

    # Find potential biomass reactions (case-insensitive)
    biomass_candidates = [
        r
        for r in model.reactions
        if any(name in r.id.lower() for name in common_biomass_names)
    ]

    # Additional filtering: Biomass reactions typically involve many metabolites
    biomass_candidates = [r for r in biomass_candidates if len(r.metabolites) > 10]

    if not biomass_candidates:
        raise ValueError(
            "No biomass reaction found. Ensure the model contains a properly labeled biomass reaction."
        )

    # Ask the user to validate which reactions are truly biomass reactions
    print("\nPotential biomass reactions found in the model:")
    valid_biomass_reactions = []

    for i, reaction in enumerate(biomass_candidates):
        print(f"\nOption {i + 1}:")
        print(f"ID: {reaction.id}")
        print(f"Name: {reaction.name}")
        print(f"Number of metabolites: {len(reaction.metabolites)}")
        print(f"Reaction formula: {reaction.reaction}")

        while True:
            user_input = (
                input("Is this a valid biomass reaction? (y/n): ").strip().lower()
            )
            if user_input in ["y", "n"]:
                if user_input == "y":
                    valid_biomass_reactions.append(reaction)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    if not valid_biomass_reactions:
        raise ValueError(
            "No valid biomass reaction was selected. Please check the model."
        )

    # Ask the user to select one biomass reaction for the rest of the code
    print("\nSelect the biomass reaction to be used for the rest of the code:")
    for i, reaction in enumerate(valid_biomass_reactions):
        print(f"\nOption {i + 1}:")
        print(f"ID: {reaction.id}")
        print(f"Name: {reaction.name}")
        print(f"Number of metabolites: {len(reaction.metabolites)}")
        print(f"Reaction formula: {reaction.reaction}")

    while True:
        try:
            choice = int(
                input(
                    f"\nEnter the number corresponding to the chosen biomass reaction (1-{len(valid_biomass_reactions)}): "
                )
            )
            if 1 <= choice <= len(valid_biomass_reactions):
                selected_biomass_reaction = valid_biomass_reactions[choice - 1]
                valid_biomass_reactions_ids = [r.id for r in valid_biomass_reactions]
                return selected_biomass_reaction, valid_biomass_reactions_ids, model
            else:
                print("Invalid choice. Please enter a number within the valid range.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def prevent_bbb_uptake(model):
    """
    Prevent the uptake of biomass building blocks (BBBs) in the GEM.

    This function identifies the biomass building blocks (BBBs) from the biomass reaction
    and ensures that drain reactions associated with these metabolites allow only secretion.

    Args:
        model (cobra.Model): The GEM to process.

    Returns:
        cobra.Model: The GEM with uptake reactions of BBBs closed.
    """
    # Identify the biomass reaction
    if not hasattr(model, "biomass_reaction") or model.biomass_reaction is None:
        biomass_rxn, _, model = identify_biomass_reaction(model)
        if not biomass_rxn:
            raise ValueError("Unable to identify a biomass reaction in the model.")
        model.biomass_reaction = (
            biomass_rxn  # Optionally store it in the model for reuse
        )
    else:
        biomass_rxn = model.biomass_reaction

    # Identify biomass building blocks (BBBs)
    bbb_mets = [
        met
        for met in biomass_rxn.metabolites
        if biomass_rxn.metabolites[met] < 0
    ]
    bbb_extracellular = [f"{met.id.split('_')[0]}_e" for met in bbb_mets]

    # Find drain reactions based on stoichiometry
    bio_drains = []
    for reaction in model.reactions:
        metabolites = reaction.metabolites
        if len(metabolites) == 1:  # Reaction involves a single metabolite
            met, coeff = next(iter(metabolites.items()))
            # Check if the metabolite is in the extracellular compartment
            if coeff < 0 and met.id in bbb_extracellular:
                bio_drains.append(reaction)

    # Robust identification of water metabolite
    water_candidates = [
        met
        for met in model.metabolites
        if "water" in met.name.lower() or met.id.lower().startswith("h2o")
    ]

    if not water_candidates:
        raise ValueError("No water metabolite found in the model.")

    # Filter for extracellular metabolites (compartment "e")
    extracellular_candidates = [
        met for met in water_candidates if met.compartment.lower() == "e"
    ]

    if len(extracellular_candidates) == 1:
        water_met = extracellular_candidates[0]
    elif len(extracellular_candidates) > 1:
        print("Multiple extracellular water metabolites found:")
        for idx, met in enumerate(extracellular_candidates, start=1):
            print(
                f" {idx}. ID: {met.id}, Name: {met.name}, Compartment: {met.compartment}"
            )
        choice = (
            int(input("Please select the correct water metabolite by number: ")) - 1
        )
        if 0 <= choice < len(extracellular_candidates):
            water_met = extracellular_candidates[choice]
        else:
            raise ValueError("Invalid choice for water metabolite.")
    else:
        # No extracellular candidates, fallback to all water candidates
        print(
            "No extracellular water metabolites found. Multiple candidates available:"
        )
        for idx, met in enumerate(water_candidates, start=1):
            print(
                f" {idx}. ID: {met.id}, Name: {met.name}, Compartment: {met.compartment}"
            )
        choice = (
            int(input("Please select the correct water metabolite by number: ")) - 1
        )
        if 0 <= choice < len(water_candidates):
            water_met = water_candidates[choice]
        else:
            raise ValueError("Invalid choice for water metabolite.")

    # Identify water exchange reaction based on stoichiometry
    water_exchange_rxn = next(
        (
            rxn
            for rxn in model.reactions
            if len(rxn.metabolites) == 1 and water_met in rxn.metabolites
        ),
        None,
    )
    if not water_exchange_rxn:
        raise ValueError("Water exchange reaction not found in the model.")

    bio_drains = [rxn for rxn in bio_drains if rxn != water_exchange_rxn]

    # Close uptake for BBB-related drain reactions (set lower bound to 0)
    for reaction in bio_drains:
        if reaction.lower_bound < 0:
            reaction.lower_bound = 0
            print(f"Closed uptake for: {reaction.id}")

    return model


def add_etc_subsystem(model, etc_criteria=None, ox_phos_subsystem=None):
    """
    Add the electron transport chain (ETC) as a core subsystem in the GEM.

    This function identifies reactions associated with the ETC based on given criteria,
    ensuring both SEED-based and model-specific metabolite conventions are used.

    Args:
        model (cobra.Model): The GEM to process.
        etc_criteria (dict, optional): A dictionary of criteria for identifying ETC reactions.
            Example:
                {
                    "metabolites": ["h2o_c", "o2_c", ...],  # Model-specific metabolites
                    "metabolites_SEED": ["cpd15560", "cpd15561", ...],  # SEED IDs
                    "subsystems": ["Oxidative Phosphorylation", "ETC"],  # Subsystem names
                }
        ox_phos_subsystem (str, optional): Name of the oxidative phosphorylation subsystem
            in the model. If provided, reactions in this subsystem will be flagged as ETC.

    Returns:
        cobra.Model: The GEM with ETC reactions annotated.
    """
    # Default SEED metabolite pairs
    default_seed_metabolites = [
        "cpd15560",
        "cpd15561",
        "cpd15499",
        "cpd15500",
        "cpd15352",
        "cpd15353",
        "cpd00109",
        "cpd00110",
        "cpd01351",
        "cpd11665",
        "cpd00986",
        "cpd00097",
    ]

    # Ensure etc_criteria includes both SEED and model-specific metabolites
    if etc_criteria is None:
        etc_criteria = {}
    etc_criteria.setdefault("metabolites", [])  # Model-specific metabolites
    etc_criteria.setdefault("metabolites_SEED", default_seed_metabolites)
    etc_criteria.setdefault(
        "subsystems", ["Oxidative Phosphorylation"]
    )  # Default subsystems

    annotated_reactions = set()

    # Annotate reactions based on subsystems
    if "subsystems" in etc_criteria:
        for reaction in model.reactions:
            if reaction.subsystem and any(
                keyword.lower() in reaction.subsystem.lower()
                for keyword in etc_criteria["subsystems"]
            ):
                reaction.notes["ETC"] = True
                annotated_reactions.add(reaction)

    # Annotate reactions based on model-specific metabolites
    if "metabolites" in etc_criteria and etc_criteria["metabolites"]:
        for reaction in model.reactions:
            mets_in_reaction = reaction.metabolites.keys()
            for met in mets_in_reaction:
                if met.id in etc_criteria["metabolites"]:
                    reaction.notes["ETC"] = True
                    annotated_reactions.add(reaction)
                    break  # Stop further checks for this reaction

    # Annotate reactions based on SEED IDs
    if "metabolites_SEED" in etc_criteria and etc_criteria["metabolites_SEED"]:
        for reaction in model.reactions:
            mets_in_reaction = reaction.metabolites.keys()
            for met in mets_in_reaction:
                # Check annotations for SEED IDs
                if "seed" in met.annotation:
                    seed_id = met.annotation["seed"]
                    if seed_id in etc_criteria["metabolites_SEED"]:
                        reaction.notes["ETC"] = True
                        annotated_reactions.add(reaction)
                        break  # Stop further checks for this reaction

    # Handle Oxidative Phosphorylation subsystem specifically
    if ox_phos_subsystem:
        for reaction in model.reactions:
            if (
                reaction.subsystem
                and reaction.subsystem.lower() == ox_phos_subsystem.lower()
            ):
                reaction.notes["ETC"] = True
                annotated_reactions.add(reaction)

    # Report annotated reactions
    if annotated_reactions:
        print(
            f"ETC subsystem successfully annotated for {len(annotated_reactions)} reactions."
        )
        for reaction in annotated_reactions:
            print(f" - Annotated as ETC: {reaction.id} ({reaction.name})")
    else:
        print("No reactions matched the criteria for ETC subsystem.")

    return model


def add_extracellular_subsystem(model, extra_cell_reactions):
    """
    Add extracellular metabolites as a core subsystem in the GEM.

    This function annotates reactions provided in `extra_cell_reactions` as extracellular.

    Args:
        model (cobra.Model): The GEM to process.
        extra_cell_reactions (list): List of reaction IDs to annotate as extracellular.

    Returns:
        cobra.Model: The GEM with extracellular metabolites annotated.
    """
    for reaction_id in extra_cell_reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.subsystem = "ExtraCell"
        print(f"Added to extracellular subsystem: {reaction.id}")
    return model


def remove_thermo_fields(model):
    """
    Remove thermodynamic fields from the GEM.

    This function clears thermodynamic properties (e.g., dG, dG_std) from reactions
    to ensure compatibility with workflows that do not require thermodynamic data.

    Args:
        model (cobra.Model): The GEM to process.

    Returns:
        cobra.Model: The GEM without thermodynamic fields.
    """
    for reaction in model.reactions:
        if hasattr(reaction, "dG"):
            delattr(reaction, "dG")
        if hasattr(reaction, "dG_std"):
            delattr(reaction, "dG_std")
        print(f"Removed thermodynamic fields from: {reaction.id}")
    return model


def identify_active_exchange_reactions(model):
    """
    Identify active exchange reactions in a genome-scale metabolic model (GEM).

    An active exchange reaction:
        - Involves a single metabolite in the extracellular compartment (met_e).
        - Has a negative lower bound, indicating the possibility of uptake.

    Args:
        model (cobra.Model): The GEM to analyze.

    Returns:
        list: A list of active exchange reactions in the model.
    """
    active_exchanges = []

    for reaction in model.reactions:
        # Check if the reaction involves exactly one metabolite (exchange reaction)
        metabolites = list(reaction.metabolites)
        if len(metabolites) == 1:
            metabolite = metabolites[0]

            # Check if the metabolite is in the extracellular compartment
            is_extracellular = (
                metabolite.compartment.lower() == "e"
                or "extracellular" in metabolite.compartment.lower()
            )

            # Check if the reaction has a negative lower bound
            has_negative_lb = reaction.lower_bound < 0

            if is_extracellular and has_negative_lb:
                active_exchanges.append(reaction)

    return active_exchanges


def identify_transport_reactions(model, biomass_rxn_ids, atp_synth_rxn_ids):
    """
    Identify and classify transport reactions in a genome-scale metabolic model.

    Args:
        model (cobra.Model): The metabolic model.
        biomass_rxn_ids (list): List of biomass reaction IDs to exclude.
        atp_synth_rxn_ids (list): List of ATP synthesis reaction IDs to exclude.

    Returns:
        dict: Contains the following keys:
            - "all_transports": List of all transport reactions.
            - "transport_no_couples": Transport reactions without coupled metabolites.
            - "coupled_transports": Transport reactions with coupled metabolites.
            - "important_transports": Transport reactions that require directionality constraints.
            - "directions": List of directions (-1 or 1) for directional transports.
            - "transport_groups": Transport reactions grouped by compartment pairs.
    """

    def remove_compartment_suffix(met_id):
        """Remove the compartment suffix from a metabolite ID (e.g., 'glc__c' -> 'glc')."""
        return met_id.rsplit("_", 1)[0]

    def get_unique_metabolites(reaction):
        """Get unique metabolite IDs without compartment suffixes from a reaction."""
        return {remove_compartment_suffix(met.id) for met in reaction.metabolites}

    def get_compartments_involved(reaction):
        """Get the set of compartments involved in a reaction."""
        return {met.compartment for met in reaction.metabolites}

    # Step 1: Exclude biomass and ATP synthase reactions
    excluded_rxns = set(biomass_rxn_ids + atp_synth_rxn_ids)
    transport_reactions = []

    for rxn in model.reactions:
        if rxn.id in excluded_rxns:
            continue  # Skip excluded reactions

        # Identify transport reactions: metabolites in multiple compartments
        unique_mets = get_unique_metabolites(rxn)
        compartments = get_compartments_involved(rxn)

        if len(compartments) > 1 and len(rxn.metabolites) > len(unique_mets):
            transport_reactions.append(rxn)

    # Step 3: Group transport reactions by compartment pairs
    transport_groups = {}
    compartments = {met.compartment for met in model.metabolites}
    for comp1, comp2 in combinations(compartments, 2):
        transport_groups[f"{comp1}_{comp2}"] = [
            rxn
            for rxn in transport_reactions
            if {comp1, comp2}.issubset(get_compartments_involved(rxn))
        ]

    # Step 4: Classify transport reactions
    directional_transports = []
    directions = []
    coupled_transports = []
    non_coupled_transports = []

    important_transports = []

    for rxn in transport_reactions:
        substrates = {remove_compartment_suffix(met.id) for met in rxn.reactants}
        products = {remove_compartment_suffix(met.id) for met in rxn.products}
        shared_mets = substrates & products

        if shared_mets:
            directional_transports.append(rxn)
            direction = 1 if len(substrates) > len(products) else -1
            directions.append(direction)
            important_transports.append(
                (
                    rxn.id,
                    list(shared_mets)[0],
                    list(get_compartments_involved(rxn))[0],
                    direction,
                )
            )

        if len(rxn.metabolites) > len(get_unique_metabolites(rxn)):
            coupled_transports.append(rxn)
        else:
            non_coupled_transports.append(rxn)

    # Step 5: Prepare the result dictionary
    result = {
        "all_transports": transport_reactions,
        "transport_no_couples": non_coupled_transports,
        "coupled_transports": coupled_transports,
        "important_transports": important_transports,  # Now formatted correctly
        "directions": directions,
        "transport_groups": transport_groups,
    }

    return result


def remove_futile_reactions(model, futile_cycle_pairs, check_growth):
    """
    Add constraints to the model to remove futile cycles in paired transport reactions.

    Args:
        model (cobra.Model): The metabolic model to process.
        futile_cycle_pairs (list of tuple): Each tuple is (rxn_id, metabolite, compartment, direction)
            representing a transport reaction that requires directionality constraints.
        check_growth (bool): Whether to verify biomass growth after adding constraints.

    Returns:
        cobra.Model: The updated model with futile cycle constraints added.
    """
    # Process unique metabolites involved in futile cycles
    unique_metabolites = set(pair[1] for pair in futile_cycle_pairs)

    for metabolite in unique_metabolites:
        # Filter pairs that involve the current metabolite (and the same compartment)
        filtered_pairs = [
            (rxn1, rxn2, dir1, dir2)
            for (rxn1, met1, comp1, dir1) in futile_cycle_pairs
            for (rxn2, met2, comp2, dir2) in futile_cycle_pairs
            if met1 == metabolite
            and met2 == metabolite
            and comp1 == comp2
            and rxn1 != rxn2
        ]

        # Process each reaction pair
        for rxn1, rxn2, dir1, dir2 in filtered_pairs:
            try:
                # Retrieve forward and backward use variables
                forward_use_1 = model.forward_use_variable.get_by_id(rxn1)
                backward_use_1 = model.backward_use_variable.get_by_id(rxn1)
                forward_use_2 = model.forward_use_variable.get_by_id(rxn2)
                backward_use_2 = model.backward_use_variable.get_by_id(rxn2)

                # Determine constraint expressions based on reaction directions
                if dir1 == dir2:
                    forward_backward_expression = forward_use_1 + backward_use_2
                    backward_forward_expression = backward_use_1 + forward_use_2
                else:
                    forward_backward_expression = forward_use_1 + forward_use_2
                    backward_forward_expression = backward_use_1 + backward_use_2

                # Add forward-backward constraint
                model.add_constraint(
                    kind=ModelConstraint,
                    hook=model,
                    expr=forward_backward_expression,
                    lb=0,
                    ub=1,
                    id_=f"Futile_{rxn1}_{rxn2}_forward_backward",
                )

                # Add backward-forward constraint
                model.add_constraint(
                    kind=ModelConstraint,
                    hook=model,
                    expr=backward_forward_expression,
                    lb=0,
                    ub=1,
                    id_=f"Futile_{rxn1}_{rxn2}_backward_forward",
                )
        

                # Check if the model can still grow
                if check_growth:
                    solution = model.optimize()
                    if solution.status != "optimal" or solution.objective_value < 1e-2:
                        print(
                            f"Growth infeasible with futile cycle constraints on {rxn1} and {rxn2}. Skipping these constraints."
                        )
                        continue  # Skip constraints that cause infeasibility

            except Exception as e:
                print(f"Skipping constraint addition for {rxn1} and {rxn2} due to: {e}")
            
        # Repair
        model.repair()                

    return model
