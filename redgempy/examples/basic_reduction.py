#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic example of using RedGEMPy to reduce a genome-scale metabolic model.

This script demonstrates how to use RedGEMPy to reduce the E. coli iJO1366 model
to a core model containing central carbon metabolism pathways.
"""

import os
import tempfile
from redgempy.redgem import run_redgem


def main():
    # Create a temporary directory for outputs
    output_dir = tempfile.mkdtemp(prefix="redgempy_example_")
    print(f"Output directory: {output_dir}")

    # Configuration for E. coli model reduction
    config = {
        # Basic settings
        "organism": "ecoli",
        "gem_name": "iJO1366",
        "red_model_name": "core_ecoli",
        # Core subsystems to include
        "selected_subsystems": [
            "Citric Acid Cycle",
            "Glycolysis/Gluconeogenesis",
            "Pentose Phosphate Pathway",
        ],
        # Add electron transport chain
        "add_etc_as_subsystem": "yes",
        # Add extracellular metabolites (automatic detection)
        "add_extracellular_subsystem": "automatic",
        # Set aerobic conditions
        "aerobic_anaerobic": "aerobic",
        # Pathway parameters
        "l": 8,  # Maximum pathway length to compute
        "d": 3,  # Maximum pathway depth to include
        "start_from_min": "yes",
        "only_connect_exclusive_mets": "yes",
        "connect_intracellular_subsystems": "no",
        # Perform lumping of reactions
        "perform_lumpgem": "yes",
        "percent_of_mu_max_for_lumping": 90,
        "num_of_lumped": "OnePerBBB",  # One lumped reaction per biomass building block
        # Do not impose thermodynamic constraints (for simplicity)
        "impose_thermodynamics": "no",
        # Remove blocked reactions during post-processing
        "perform_post_processing": "pp_removeblockrxns",
        # Output path
        "output_path": output_dir,
    }

    # Run the RedGEM workflow
    print("Starting RedGEM workflow...")
    redgem_result = run_redgem(config_dict=config)

    # Access the reduced model
    reduced_model = redgem_result.red_model

    # Print statistics about the reduction
    print("\nReduction Statistics:")
    print(f"Original model reactions: {len(redgem_result.original_gem.reactions)}")
    print(f"Reduced model reactions: {len(reduced_model.reactions)}")
    print(
        f"Reduction ratio: {len(reduced_model.reactions) / len(redgem_result.original_gem.reactions):.2f}"
    )

    # Print information about lumped reactions
    if (
        hasattr(redgem_result, "lumped_rxn_formulas")
        and redgem_result.lumped_rxn_formulas
    ):
        print(f"\nNumber of lumped reactions: {len(redgem_result.lumped_rxn_formulas)}")

    # Print path to the saved model
    print(
        f"\nReduced model saved to: {os.path.join(output_dir, 'UserOutputs', 'Models', 'ecoli')}"
    )

    return reduced_model


if __name__ == "__main__":
    reduced_model = main()
