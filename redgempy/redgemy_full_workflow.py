from redgempy.parameters import redgem_questionnaire
from redgempy.annotation import get_seed_ids_for_rxn
from redgempy.filesystem import setup_directories, save_workspace
from redgempy.case_processing import process_case
from redgempy.checkpoints import save_checkpoint, load_checkpoint
from redgempy.solver import set_solver
from redgempy.utils import remove_duplicates
from redgempy.gem_processing import (
    load_gem,
    prevent_bbb_uptake,
    add_etc_subsystem,
    add_extracellular_subsystem,
    identify_biomass_reaction,
    identify_active_exchange_reactions,
)
from redgempy.connect_subsystems import extract_adj_data

from redgempy.lumpgem import add_biomass
import logging

import os
from cobra.io.json import load_json_model
from cobra.io.mat import load_matlab_model
from redgempy.removeMetPairsFromS import remove_met_pairs_from_S
from redgempy.adjacentfromstoich import adjacent_from_stoich
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
from itertools import combinations

# Create a dedicated checkpoint folder if it doesn't exist
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load the lists from JSON files
with open("./data/met_pairs_to_remove.json", "r") as f:
    met_pairs_to_remove = json.load(f)

with open("./data/inorganic_mets.json", "r") as f:
    inorganic_mets = json.load(f)


# Get user inputs using the redgem_questionnaire
redgem_opts = redgem_questionnaire()

redgem_opts["l"] = 6
redgem_opts["d"] = 1
redgem_opts["add_extracellular_subsystem"] = "default"
redgem_opts["start_from_min"] = "no"
redgem_opts["only_connect_exclusive_mets"] = False
redgem_opts["connect_intracellular_subsystems"] = True
redgem_opts["apply_shortest_distance_of_subsystems"] = "bothways"
redgem_opts["throw_error_on_d_violation"] = True

# Create necessary directories
output_paths = setup_directories(redgem_opts)

# --- Savepoint 1: Process case and load GEM ---
processed_case_file = os.path.join(checkpoint_dir, "processed_case.pkl")
processed_case = load_checkpoint(processed_case_file)
if processed_case is None:
    processed_case = process_case(redgem_opts)
    save_checkpoint(processed_case, processed_case_file)
else:
    print("Loaded processed_case from checkpoint.")

gem_model = processed_case[1]  # Processed GEM
original_gem = processed_case[0]  # Original GEM

# Configure the solver
redgem_opts["cplex_parameters"] = "gurobi"
gem_model = set_solver(gem_model, redgem_opts)

# Define the biomass in the model
biomass_reaction, _, _ = identify_biomass_reaction(gem_model)
gem_model.biomass_reaction = biomass_reaction

# Prevent uptake of biomass building blocks
if redgem_opts.get("prevent_bbb_uptake", "no") == "yes":
    gem_model = prevent_bbb_uptake(gem_model)
if redgem_opts.get("add_etc_as_subsystem", "no") == "yes":
    gem_model = add_etc_subsystem(gem_model)
if redgem_opts.get("add_extracellular_subsystem", []) != []:
    gem_model = add_extracellular_subsystem(gem_model, processed_case[7])

# Needed inputs from before
Biomass_rxns = processed_case[3]
L = redgem_opts["l"]
InorgMetSEEDIDs = []  # or inorganic_mets

# Savepoint 2

# Load the MATLAB model and restore annotations
# (No checkpointing here since these are file loads.)
gem_model = load_matlab_model("./models/GSmodel_Ecoli_curated_for_tests.mat")
gem_model_annot = load_json_model("./models/iJO1366.json")

# Restore the annotation layer of metabolites
for met in gem_model_annot.metabolites:
    try:
        met_id = met.id.replace("__", "-")
        met_orig = gem_model.metabolites.get_by_id(met_id)
        met_orig.annotation = met.annotation
    except Exception:
        pass

# --- Savepoint 2: Build gem_for_adjmat with stoichiometric matrix ---
gem_adjmat_file = os.path.join(checkpoint_dir, "gem_for_adjmat.pkl")
gem_for_adjmat = load_checkpoint(gem_adjmat_file)
if gem_for_adjmat is None:
    gem_for_adjmat = gem_model.copy()

    # Find and remove the biomass reactions
    bm_rxns = [rxn.id for rxn in gem_model.reactions if rxn.id in Biomass_rxns]
    gem_for_adjmat.remove_reactions(bm_rxns)

    # Create the stoichiometric matrix as a DataFrame
    met_ids = [met.id for met in gem_for_adjmat.metabolites]
    rxn_ids = [rxn.id for rxn in gem_for_adjmat.reactions]
    s_matrix = pd.DataFrame(
        np.zeros((len(met_ids), len(rxn_ids))), index=met_ids, columns=rxn_ids
    )
    for met_id in met_ids:
        met = gem_for_adjmat.metabolites.get_by_id(met_id)
        for rxn in met.reactions:
            s_matrix.loc[met_id, rxn.id] = rxn.metabolites[met]
    gem_for_adjmat.s_matrix = s_matrix

    # Remove cofactors and small inorganic metabolites from the S matrix
    s_matrix_new = remove_met_pairs_from_S(
        gem_for_adjmat, met_pairs_to_remove, inorganic_mets
    )
    gem_for_adjmat.s_matrix = s_matrix_new

    save_checkpoint(gem_for_adjmat, gem_adjmat_file)
else:
    print("Loaded gem_for_adjmat from checkpoint.")

# Test the stoichiometric matrix against MATLAB output
test_s_matrix = pd.read_csv("./matrix_from_MATLAB_for_tests/Smatrix.txt", header=None)
s_matrix_new = gem_for_adjmat.s_matrix
for i in range(len(s_matrix_new.columns)):
    assert np.allclose(
        s_matrix_new.iloc[:, i], test_s_matrix.iloc[:, i], atol=1e-10, rtol=1e-10
    )

# Savepoint 3

# --- Savepoint 3: Compute adjacency matrices ---
adjacency_file = os.path.join(checkpoint_dir, "adjacency_matrices.pkl")
adjacency_data = load_checkpoint(adjacency_file)
if adjacency_data is None:
    # Compute the directed adjacency matrices
    _, DirAdjMatWn, DirAdjMatC = adjacent_from_stoich(gem_for_adjmat)
    adjacency_data = (DirAdjMatWn, DirAdjMatC)
    save_checkpoint(adjacency_data, adjacency_file)
else:
    print("Loaded adjacency matrices from checkpoint.")
    DirAdjMatWn, DirAdjMatC = adjacency_data

# Compare the values of the adjacency matrix created from python and from MATLAB per row
test_DirAdjMatWn = pd.read_csv(
    "./matrix_from_MATLAB_for_tests/DirAdjMatWn.txt", header=None
)
python_DirAdjMatWn = pd.DataFrame(DirAdjMatWn)

for met in range(len(python_DirAdjMatWn.index)):
    assert np.allclose(
        python_DirAdjMatWn.iloc[met, :],
        test_DirAdjMatWn.iloc[met, :],
        atol=1e-10,
        rtol=1e-10,
    )

# --- Savepoint 3b: Generate L matrices (pathway data) ---
L_matrices_file = os.path.join(checkpoint_dir, "L_matrices.pkl")
L_matrices = load_checkpoint(L_matrices_file)
if L_matrices is None:
    metabolites = DirAdjMatC.index.tolist()  # List of metabolite IDs
    num_mets = len(metabolites)

    # Initialize outputs
    L_DirAdjMatWn = np.zeros((num_mets, num_mets, L), dtype=int)
    L_DirAdjMatC = {
        (i, j, l): []
        for i in range(num_mets)
        for j in range(num_mets)
        for l in range(L)
    }
    L_DirAdjMatMets = {
        (i, j, l): []
        for i in range(num_mets)
        for j in range(num_mets)
        for l in range(L)
    }

    # Fill L = 1
    L_DirAdjMatWn[:, :, 0] = DirAdjMatWn
    for i, met_i in enumerate(metabolites):
        for j, met_j in enumerate(metabolites):
            if DirAdjMatWn[i, j] > 0:
                L_DirAdjMatMets[(i, j, 0)] = [
                    [i, j] for _ in DirAdjMatC.at[met_i, met_j]
                ]
                L_DirAdjMatC[(i, j, 0)] = DirAdjMatC.at[met_i, met_j]

    # Precompute connections
    r1, c1 = np.nonzero(DirAdjMatWn)
    L1_connections_map = {i: np.unique(c1[r1 == i]) for i in range(num_mets)}

    # Compute paths for L > 1
    for depth in range(1, L):
        if depth == 1:
            rm, cm = r1, c1
        else:
            rm, cm = np.nonzero(L_DirAdjMatWn[:, :, depth - 1])
        for starting_met in tqdm(range(num_mets), desc="Depth {}".format(depth + 1)):
            current_mets = np.unique(cm[rm == starting_met])
            for current_met in current_mets:
                prev_mets = L_DirAdjMatMets[(starting_met, current_met, depth - 1)]
                prev_mets_flat = (
                    np.array(prev_mets).ravel()
                    if isinstance(prev_mets, list)
                    else np.ravel(prev_mets)
                )
                prev_reacs = L_DirAdjMatC[(starting_met, current_met, depth - 1)]
                len_of_prev_rxns = len(prev_reacs)
                next_mets = L1_connections_map[current_met]
                next_mets = np.setdiff1d(
                    next_mets, np.union1d(prev_mets_flat, [starting_met])
                )
                for next_met in next_mets:
                    second_reacs = L_DirAdjMatC[(current_met, next_met, 0)]
                    len_of_new_rxns = len(second_reacs)
                    combined_reactions = []
                    for prev_reac in prev_reacs:
                        if prev_reac is None:
                            raise ValueError(
                                "None at ", starting_met, current_met, next_met, depth
                            )
                        for second_reac in second_reacs:
                            if isinstance(prev_reac, list):
                                combined_reactions.append(prev_reac + [second_reac])
                            else:
                                combined_reactions.append([prev_reac, second_reac])
                    L_DirAdjMatC[(starting_met, next_met, depth)] += combined_reactions

                    combined_metabolites = []
                    for prev_met in prev_mets:
                        if prev_met is None:
                            raise ValueError(
                                "None at ", starting_met, current_met, next_met, depth
                            )
                        for tmp_id in range(len_of_new_rxns):
                            if isinstance(prev_met, list):
                                combined_metabolites.append(prev_met + [next_met])
                            else:
                                combined_metabolites.append([prev_met, next_met])
                    L_DirAdjMatMets[
                        (starting_met, next_met, depth)
                    ] += combined_metabolites

                    replicated_prevMets = np.tile(prev_mets_flat, (1, len_of_new_rxns))
                    num_columns = replicated_prevMets.shape[1]
                    L_DirAdjMatWn[starting_met, next_met, depth] = np.shape(
                        L_DirAdjMatC[(starting_met, next_met, depth)]
                    )[0]
    L_matrices = (L_DirAdjMatWn, L_DirAdjMatC, L_DirAdjMatMets)
    save_checkpoint(L_matrices, L_matrices_file)
else:
    print("Loaded L matrices from checkpoint.")
    L_DirAdjMatWn, L_DirAdjMatC, L_DirAdjMatMets = L_matrices


# Compare the L matrices created from python and from MATLAB per depth
for depth in range(L):
    print(f"Depth {depth+1}")
    test_L_DirAdjMatWn = pd.read_csv(
        f"./matrix_from_MATLAB_for_tests/L_DirAdjMatWn_{depth+1}.txt", header=None
    )
    python_L_DirAdjMatWn = pd.DataFrame(L_DirAdjMatWn[:, :, depth])

    for met in range(len(python_L_DirAdjMatWn.index)):
        assert np.allclose(
            python_L_DirAdjMatWn.iloc[met, :],
            test_L_DirAdjMatWn.iloc[met, :],
            atol=1e-10,
            rtol=1e-10,
        )

# Iterate through the matrices
for depth in tqdm(range(1, L)):  # MATLAB's 2:L corresponds to Python's range(1, L)
    for starting_met in range(DirAdjMatC.shape[0]):  # Iterate over rows
        for ending_met in range(DirAdjMatC.shape[0]):  # Iterate over columns
            rxns = L_DirAdjMatC.get((starting_met, ending_met, depth), [])
            mets = L_DirAdjMatMets.get((starting_met, ending_met, depth), [])

            if rxns:  # Check if rxns is not empty
                rxns_new, removed_indices = remove_duplicates(rxns)
                if removed_indices:
                    mets_new = [
                        mets[i] for i in range(len(mets)) if i not in removed_indices
                    ]

                    # Update matrices
                    L_DirAdjMatC[(starting_met, ending_met, depth)] = rxns_new
                    L_DirAdjMatMets[(starting_met, ending_met, depth)] = mets_new

                    # Set weight to 0 if all columns were removed
                    if len(rxns_new) == 0:
                        L_DirAdjMatWn[starting_met, ending_met, depth] = 0


# Compare the L matrices created from python and from MATLAB per depth
for depth in range(L):
    print(f"Depth {depth+1}")
    test_L_DirAdjMatWn = pd.read_csv(
        f"./matrix_from_MATLAB_for_tests/L_DirAdjMatWn_clean_{depth+1}.txt", header=None
    )
    python_L_DirAdjMatWn = pd.DataFrame(L_DirAdjMatWn[:, :, depth])

    for met in range(len(python_L_DirAdjMatWn.index)):
        assert np.allclose(
            python_L_DirAdjMatWn.iloc[met, :],
            test_L_DirAdjMatWn.iloc[met, :],
            atol=1e-10,
            rtol=1e-10,
        ), f"Metabolite {met} at depth {depth+1} is not equal."

# %% [markdown]
# # Savepoint 4
##### CHECKPOINT LOGIC UNTIL HERE ##### # Check again chat at: # Compare the L matrices created from Python and from MATLAB per depth

# %%

core_ss = list(set(processed_case[2]))
core_ss.append("ETC_Rxns")  # TODO: FIX THAT BASED ON THE USER INPUT
core_ss.append("ExtraCell")  # TODO: FIX THAT BASED ON THE USER INPUT

D = redgem_opts["d"]
start_from_min = redgem_opts["start_from_min"]
only_connect_exclusive_mets = redgem_opts["only_connect_exclusive_mets"]
connect_intra_subsystem = redgem_opts["connect_intracellular_subsystems"]
apply_shortest_distance_of_subsystems = redgem_opts[
    "apply_shortest_distance_of_subsystems"
]
throw_error_on_d_violation = redgem_opts["throw_error_on_d_violation"]

(
    selected_paths,
    selected_paths_external,
    connecting_rxns,
    rxns_ss,
    mets_ss,
    other_reactions,
) = extract_adj_data(
    gem_for_adjmat,
    core_ss,
    L_DirAdjMatWn,
    L_DirAdjMatC,
    L_DirAdjMatMets,
    D,
    start_from_min,
    only_connect_exclusive_mets,
    connect_intra_subsystem,
    apply_shortest_distance_of_subsystems,
    throw_error_on_d_violation,
)

core_ss.append("ExtraCell")  # TODO: FIX THAT BASED ON THE USER INPUT

core_rxns = set(i.id for i in gem_for_adjmat.reactions if i.subsystem in core_ss)
all_core_rxns = list(np.unique(list(set(connecting_rxns)) + list(core_rxns)))

submodel = gem_model.copy()
reactions_to_remove = [rxn for rxn in submodel.reactions if rxn.id not in all_core_rxns]
submodel.remove_reactions(reactions_to_remove)

# Check whether there are reactions that use only core metabolites
# If yes add them to the core reactions and rebuild the submodel
additional_core_rxns = []
for rxn in gem_model.reactions:
    mets = [met.id for met in rxn.metabolites]
    if not set(mets).difference(set(submodel.metabolites)):
        additional_core_rxns.append(rxn.id)
if additional_core_rxns != []:
    # Add the previous core reactions to the list
    all_core_rxns = list(
        np.unique(list(set(connecting_rxns)) + list(core_rxns) + additional_core_rxns)
    )
    submodel = gem_model.copy()
    reactions_to_remove = [
        rxn for rxn in submodel.reactions if rxn.id not in all_core_rxns
    ]

metabolites_to_remove = [met for met in submodel.metabolites if len(met.reactions) == 0]
submodel.remove_metabolites(metabolites_to_remove)

for gene in submodel.genes:
    if len(gene.reactions) == 0:
        submodel.genes.remove(gene)


# 1. Identify non-core (other) reactions.
non_core_rxns = [rxn for rxn in gem_model.reactions if rxn.id not in core_rxns]
# (These are the reactions that are not core; indices in Python are already aligned with gem_model.reactions.)
other_reactions_gsm_idx = [gem_model.reactions.index(rxn) for rxn in non_core_rxns]

# 2a. Identify ATP synthase reactions.
atp_synth_rxn_ids = (
    []
)  # These will be indices (or reaction objects) for ATP synthase reactions.
SEEDIDs_atpSynth = {"cpd00008", "cpd00009", "cpd00067", "cpd00002", "cpd00001"}


# Loop over all reactions (or you could restrict to non_core_rxns if desired)
for i, rxn in enumerate(gem_model.reactions):
    seed_ids_rxn = get_seed_ids_for_rxn(rxn)
    # If the reaction's set of seed IDs exactly equals the ATP synthase set,
    # then its symmetric difference will be empty.
    if seed_ids_rxn.symmetric_difference(SEEDIDs_atpSynth) == set():
        atp_synth_rxn_ids.append(i)

# Retrieve ATP synthase reaction names (or reaction objects) from indices.
atp_synth_rxn_names = [gem_model.reactions[i].id for i in atp_synth_rxn_ids]

# 2b. Identify ETC reactions.
# Here we assume that each reaction has an attribute or annotation for its subsystem.
# For example, if stored in rxn.annotation['subSystem'] or rxn.subsystem.
etc_rxn_names = [
    rxn.id
    for rxn in gem_model.reactions
    if (hasattr(rxn, "subsystem") and rxn.subsystem == "ETC_Rxns")
    or (rxn.annotation.get("subSystem", None) == "ETC_Rxns")
]

# 2c. Combine ETC and ATP synthase reaction names.
rxn_names_prev_therm_relax = etc_rxn_names + atp_synth_rxn_names

# For debugging, you can print out the lists:
print("Non-core reaction indices:", other_reactions_gsm_idx)
print("ATP synthase reaction names:", atp_synth_rxn_names)
print("ETC reaction names:", etc_rxn_names)
print("Reactions with prevented thermodynamic relaxation:", rxn_names_prev_therm_relax)


# %% [markdown]
# # Savepoint 5

# %%
# RedGEMX

# %% [markdown]
# ## Savepoint 5a

# %%
# Comments indicate supposed sources of the input for the add_biomass

model = add_biomass(
    model=gem_model,
    non_core_rxn_ids=[
        rxn.id for rxn in gem_model.reactions if rxn in non_core_rxns
    ],  # RedGEM_X-dependent
    thermo_dir_path=None,  # redgem_opts["thermo_data_path"],  # RedGEM
    biomass_building_blocks_to_exclude=processed_case[6],
    oxygen_metabolite_id="o2_e",
    aerobic_anaerobic=redgem_opts["aerobic_anaerobic"],
    organism=redgem_opts["organism"],
    num_of_lumped=redgem_opts["num_of_lumped"],
    gem_name=redgem_opts["gem_name"],
    rxn_names_prev_therm_relax=rxn_names_prev_therm_relax,
    biomass_rxn_names=processed_case[3],
    atp_synth_rxn_names=atp_synth_rxn_names,
    add_gam=redgem_opts["add_gam"],
    percent_mu_max_lumping=redgem_opts["percent_of_mu_max_for_lumping"],
    impose_thermodynamics=redgem_opts["impose_thermodynamics"],
    output_path=redgem_opts["output_path"],
)

# %% [markdown]
# # Savepoint 6
