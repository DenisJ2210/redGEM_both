import os

"""
RedGEM Parameter Setup Script.

This script provides a console-based questionnaire for setting up parameters
required to configure the redGEM workflow. The user is prompted to answer a
single default question first, simplifying the configuration process.

Usage:
    Run this script directly to start the questionnaire:
    $ python parameters.py

Returns:
    dict: A dictionary of parameter names and their respective values.
"""

# Dynamically set the default output path relative to the script's location
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs")
)

DEFAULT_OPTIONS = {
    "organism": "ecoli",
    "gem_name": "default_gem",
    "red_model_name": "reduced_model",
    "selected_subsystems": "default",
    "add_etc_as_subsystem": "yes",
    "add_extracellular_subsystem": "automatic",
    "aerobic_anaerobic": "aerobic",
    "list_for_inorganic_mets": "automatic",
    "list_for_cofactor_pairs": "automatic",
    "zero_zero_gem_bounds": "Original",
    "l": 10,
    "d": 4,
    "start_from_min": "yes",
    "throw_error_on_d_violation": "error",
    "only_connect_exclusive_mets": "yes",
    "connect_intracellular_subsystems": "no",
    "apply_shortest_distance_of_subsystems": "bothways",
    "perform_redgemx": "no",
    "num_of_connections": "OnePerMetE",
    "perform_lumpgem": "yes",
    "percent_of_mu_max_for_lumping": 90,
    "add_gam": "yes",
    "prevent_bbb_uptake": "yes",
    "num_of_lumped": "OnePerBBB",
    "align_transports_using_mat_file": "no",
    "impose_thermodynamics": "no",
    "perform_post_processing": "no",
    "time_limit_for_solver": "no",
    "cplex_parameters": "CplexDefault",
    "cplex_path": "/path/to/cplex",
    "tfa_path": "/path/to/tfa",
    "thermo_data_path": "/path/to/thermo_data",
    "output_path": DEFAULT_OUTPUT_PATH,  # Dynamically set output path
}


def redgem_questionnaire():
    """
    Console-based questionnaire for setting up parameters for the redGEM workflow.

    The function first asks the user if they want to use default options.
    If yes, all other questions are skipped and default values are used.
    If no, the user is prompted to answer each question manually.

    Returns:
        dict: A dictionary containing the parameter names as keys and the user-provided or default values as values.

    Example:
        >>> redgem_questionnaire()
        === RedGEM Parameter Setup ===
        Would you like to use the default configuration? (yes/no): yes
        === Final Options Set ===
        {'organism': 'ecoli', 'gem_name': 'default_gem', ...}
    """
    print("=== RedGEM Parameter Setup ===")

    # Ask the user if they want to use default options
    use_defaults = (
        input("Would you like to use the default configuration? (yes/no): ")
        .strip()
        .lower()
    )

    if use_defaults == "yes":
        print("\nUsing default configuration.")
        for k, v in DEFAULT_OPTIONS.items():
            print(f"{k}: {v}")
        return DEFAULT_OPTIONS

    print("\nPlease answer the following questions:\n")

    # Define the questions and possible options (None indicates free text input)
    questions = {
        "organism": ["ecoli", "yeast", "human", "plasmodium", "custom"],
        "gem_name": None,  # Text input for custom GEM names
        "red_model_name": None,
        "selected_subsystems": ["default", "custom"],
        "add_etc_as_subsystem": ["yes", "no"],
        "add_extracellular_subsystem": ["no", "automatic", "default", "custom"],
        "aerobic_anaerobic": ["aerobic", "anaerobic"],
        "list_for_inorganic_mets": ["automatic", "curated"],
        "list_for_cofactor_pairs": ["automatic", "curated"],
        "zero_zero_gem_bounds": ["Original", "OpenTo100", "DefineCustom"],
        "l": None,  # Integer input
        "d": None,  # Integer input
        "start_from_min": ["yes", "no"],
        "throw_error_on_d_violation": ["error", "continue"],
        "only_connect_exclusive_mets": ["yes", "no"],
        "connect_intracellular_subsystems": ["yes", "no"],
        "apply_shortest_distance_of_subsystems": ["bothways", "eachdirection"],
        "perform_redgemx": ["yes", "no"],
        "num_of_connections": ["OnePerMetE", "SminMetE", "custom"],
        "perform_lumpgem": ["yes", "no"],
        "percent_of_mu_max_for_lumping": None,  # Integer input
        "add_gam": ["yes", "no"],
        "prevent_bbb_uptake": ["yes", "no"],
        "num_of_lumped": ["OnePerBBB", "Smin", "Sminp1", "custom"],
        "align_transports_using_mat_file": ["yes", "no"],
        "impose_thermodynamics": ["yes", "no"],
        "perform_post_processing": ["no", "PP_forMCA", "custom"],
        "time_limit_for_solver": ["yes", "no"],
        "cplex_parameters": ["CplexDefault", "LCSBDefault", "custom"],
        "cplex_path": None,
        "tfa_path": None,
        "thermo_data_path": None,
        "output_path": None,
    }

    # Dictionary to store the user inputs
    redgem_opts = {}

    # Iterate through questions and get user input
    for key, options in questions.items():
        if options is None:
            # Free text input
            value = input(f"{key.replace('_', ' ').capitalize()}: ").strip()
        else:
            # Display choices for predefined options
            choices = "/".join(options)
            value = input(f"{key.replace('_', ' ').capitalize()} ({choices}): ").strip()

            # Validate input for predefined choices
            while options and value not in options:
                print(f"Invalid choice. Please choose from {choices}.")
                value = input(
                    f"{key.replace('_', ' ').capitalize()} ({choices}): "
                ).strip()

        # Store the response
        redgem_opts[key] = value

    # Add default output path if not provided
    redgem_opts["output_path"] = redgem_opts.get("output_path", DEFAULT_OUTPUT_PATH)

    print("\n=== Final Options Set ===")
    for k, v in redgem_opts.items():
        print(f"{k}: {v}")

    return redgem_opts


if __name__ == "__main__":
    # Run the questionnaire
    final_options = redgem_questionnaire()
