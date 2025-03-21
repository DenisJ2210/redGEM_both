
import numpy as np

def remove_met_pairs_from_S(model, met_pairs_to_remove, inorganic_mets):
    """
    Remove specified metabolite pairs and small metabolites from the stoichiometric matrix
    using COBRApy.

    Parameters:
    - model: A COBRApy model object.
    - met_pairs: A list of cofactor pairs in the format 'met1:met2' (using metabolite IDs).
    - inorganic: A list of small metabolites to remove (either IDs or indices).

    Returns:
    - Smatrix: Modified stoichiometric matrix as a NumPy array.
    """
    # Copy the stoichiometric matrix from the model
    Smatrix = model.s_matrix


    # Remove inorganic metabolites from the stoichiometric matrix
    for met in model.metabolites:
        if met.id in inorganic_mets:
            Smatrix.loc[met.id,:] = 0


    # Parse metabolite pairs
    all_mets = [pair.split(':') for pair in met_pairs_to_remove]

    # Find reactions that contain both metabolites in the pair and remove them from the stoichiometric matrix
    for met_pair in all_mets:
        met1 = model.metabolites.get_by_id(met_pair[0])
        met2 = model.metabolites.get_by_id(met_pair[1])
        for rxn in met1.reactions:
            if met2 in rxn.metabolites:
                Smatrix.loc[met1.id, rxn.id] = 0
                Smatrix.loc[met2.id, rxn.id] = 0

    return Smatrix
