import os
from cobra.io import load_json_model
from redgempy.gem_processing import identify_biomass_reaction
from redgempy.lumpgem import add_biomass

# Import the function (Assuming it's from a module you have)
# from your_module import add_biomass

# Define the necessary input parameters
model_path = "./models/IJO1366.json"
model = load_json_model(model_path)  # Replace with actual model input
non_core_rxns = []  # List of reaction indices
thermo_dir_path = []
biomass_building_blocks_to_exclude = []  # List of building blocks to exclude
oxygen_metabolite_id = "o2_e"
aerobic_anaerobic = "aerobic"  # or "anaerobic"
organism = "E. coli"  # Replace with your organism name
align_transports_matfile = []
num_of_lumped = 5  # Adjust as needed
gem_name = "iJO1366"  # Replace with actual GEM name
rxn_names_prev_therm_relax = []  # List of previous thermodynamically relaxed reactions
atp_synth_rxn_names = ["ATPS4r"]  # Example ATP synthesis reaction
add_gam = True  # Whether to add growth-associated maintenance (GAM)
percent_mu_max_lumping = 0.1  # Adjust based on needs
impose_thermodynamics = False  # Whether to impose thermodynamic constraints
output_path = "./outputs/"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Add the biomass reaction field
model.biomass_reaction, biomass_rxn_names = identify_biomass_reaction(model)

# Run the function
processed_results = add_biomass(
    model,
    non_core_rxns,
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
)

# Print or save results
print("Processed results:", processed_results)
