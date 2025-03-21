"""
Case Processing for Organism-Specific GEMs.

This script provides a unified function to process genome-scale metabolic models (GEMs)
for specific organisms while retaining the unique attributes and preprocessing steps for each.

Functions:
    - process_case: Main function to handle organism-specific GEM processing.
"""

import os
import cobra
from redgempy.gem_processing import (
    load_gem,
    identify_active_exchange_reactions,
)  # Import the load_gem function

# Organism-specific constants
ORGANISM_SETTINGS = {
    "ecoli": {
        "filename": "GSmodel_Ecoli.mat",
        "key": "ttmodel",
        "biomass_reactions": [
            "Ec_biomass_iJO1366_WT_53p95M",
            "Ec_biomass_iJO1366_core_53p95M",
        ],
        "core_subsystems_default": [
            "Citric Acid Cycle",
            "Pentose Phosphate Pathway",
            "Glycolysis/Gluconeogenesis",
            "Pyruvate Metabolism",
            "Glyoxylate Metabolism",
        ],
        "extracellular_default": [
            "DM_succ_e",
            "DM_ac_e",
            "DM_etoh_e",
            "DM_glyc_e",
            "DM_lac-D_e",
            "DM_akg_e",
            "DM_for_e",
            "DM_pyr_e",
            "DM_fum_e",
            "DM_co2_e",
            "DM_mal-L_e",
        ],
        "oxphos_subsystem": "Oxidative Phosphorylation",
    },
    "human": {
        "filename": None,  # GEM name provided dynamically
        "key": "model",
        "biomass_reactions": ["biomass"],
        "core_subsystems_default": [
            "Citric Acid Cycle",
            "Pentose Phosphate Pathway",
            "Glycolysis/Gluconeogenesis",
            "Oxidative Phosphorylation",
        ],
        "extracellular_default": [],  # No extracellular subsystem for humans
        "oxphos_subsystem": "Oxidative phosphorylation",
    },
    "putida": {
        "filename": "putida_GEM_20150812.mat",
        "key": "ttmodel",
        "biomass_reactions": ["GROWTH"],
        "core_subsystems_default": [
            "TCA Cycle",
            "Pentose Phosphate Pathway",
            "Glycolysis",
            "Gluconeogenesis",
            "Pyruvate Metabolism",
        ],
        "extracellular_default": [
            "DM_succ_e",
            "DM_ac_e",
            "DM_glyc_e",
            "DM_lac_D_e",
            "DM_akg_e",
            "DM_mal_L_e",
        ],
        "oxphos_subsystem": "Oxidative phosphorylation",
    },
    "yeast": {
        "filename": "iMM904.mat",
        "key": "iMM904",
        "biomass_reactions": ["yeast 5 biomass pseudoreaction"],
        "core_subsystems_default": [
            "Citric Acid Cycle",
            "Pentose Phosphate Pathway",
            "Glycolysis/Gluconeogenesis",
            "Pyruvate Metabolism",
            "Oxidative Phosphorylation",
        ],
        "extracellular_default": [
            "EX_succ(e)",
            "EX_ac(e)",
            "EX_etoh(e)",
            "EX_glyc(e)",
            "EX_lac-D(e)",
            "EX_akg(e)",
            "EX_for(e)",
            "EX_pyr(e)",
            "EX_co2(e)",
            "EX_mal-L(e)",
        ],
        "oxphos_subsystem": "Oxidative Phosphorylation",
    },
}


def handle_zero_bounds(model, option):
    """
    Handle reactions with zero-zero bounds in the model based on user choice.

    Args:
        model (cobra.Model): The GEM model.
        option (str): Option for handling zero-zero bounds. Options are:
            - "Original": Retain the original bounds.
            - "OpenTo100": Set bounds to [-100, 100].

    Returns:
        cobra.Model: Updated GEM model.

    Raises:
        ValueError: If an invalid option is provided.
    """
    zero_bounds = [
        r for r in model.reactions if r.lower_bound == 0 and r.upper_bound == 0
    ]
    if zero_bounds:
        if option == "Original":
            return model
        if option == "OpenTo100":
            for reaction in zero_bounds:
                reaction.lower_bound = -100
                reaction.upper_bound = 100
            return model
        raise ValueError(f"Invalid zero-zero bounds option: {option}")
    return model


def define_core_subsystems(selected_subsystems, organism_defaults):
    """
    Define core subsystems for the model.

    Args:
        selected_subsystems (list or str): List of selected subsystems or "default".
        organism_defaults (list): Default core subsystems for the organism.

    Returns:
        list: Core subsystems.

    Raises:
        ValueError: If selected_subsystems is invalid.
    """
    if isinstance(selected_subsystems, list):
        return selected_subsystems
    if selected_subsystems == "default":
        return organism_defaults
    raise ValueError(f"Invalid subsystem selection: {selected_subsystems}")


def process_case(options):
    """
    Process an organism-specific GEM case.

    Args:
        options (dict): Configuration options for the organism.

    Returns:
        tuple: Processed components including:
            - Original GEM (cobra.Model)
            - Processed GEM (cobra.Model)
            - Core subsystems
            - Biomass reactions
            - Metabolite pairs to remove
            - Inorganic metabolite IDs
            - Biomass building blocks to exclude
            - Extracellular subsystems
            - Oxidative phosphorylation subsystem
    """
    organism = options.get("organism")
    zero_zero_bounds = options.get("zero_zero_bounds", "Original")
    selected_subsystems = options.get("selected_subsystems", "default")
    add_extra_cell = options.get("add_extracellular_subsystem", "default")

    if organism not in ORGANISM_SETTINGS:
        raise ValueError(f"Unsupported organism: {organism}")

    settings = ORGANISM_SETTINGS[organism]
    gem_name = settings.get("filename")
    file_extension = os.path.splitext(gem_name)[1] if gem_name else ".json"

    redgem_opts = {
        "gem_name": os.path.splitext(gem_name)[0] if gem_name else options["gem_name"],
        "output_path": options["output_path"],
        "file_extension": file_extension,
        "key": settings["key"],
    }

    # Load GEM
    gem_model = load_gem(redgem_opts)
    original_gem = gem_model.copy()

    # Handle zero-zero bounds
    gem_model = handle_zero_bounds(gem_model, zero_zero_bounds)

    # Define core subsystems
    core_ss = define_core_subsystems(
        selected_subsystems, settings["core_subsystems_default"]
    )

    # Handle extracellular subsystem
    if add_extra_cell == "default":
        extra_cell_subsystem = settings["extracellular_default"]
    elif add_extra_cell == "no":
        extra_cell_subsystem = []
    elif add_extra_cell == "automatic":
        active_exchanges = identify_active_exchange_reactions(gem_model)
        extra_cell_subsystem = [reaction.id for reaction in active_exchanges]
    elif add_extra_cell == "custom":
        extra_cell_subsystem = []
        raise ValueError("Not yet implemented my friend ...")
    else:
        raise ValueError("Wrong value.")

    # Placeholder logic for specific metabolites and pairs
    met_pairs_to_remove = []
    inorg_met_seed_ids = []
    bbbs_to_exclude = []

    return (
        original_gem,
        gem_model,
        core_ss,
        settings["biomass_reactions"],
        met_pairs_to_remove,
        inorg_met_seed_ids,
        bbbs_to_exclude,
        extra_cell_subsystem,
        settings["oxphos_subsystem"],
    )
