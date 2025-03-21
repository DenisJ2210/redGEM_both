import numpy as np
import pandas as pd

def adjacent_from_stoich(model):
    """
    Create directed adjacency matrices from the stoichiometric matrix of a metabolic model.
    
    Parameters:
    - model: A COBRApy model object.
    
    Returns:
    - DirAdjMatW1: Directed adjacency matrix with weights = 1.
    - DirAdjMatWn: Directed adjacency matrix with weights = the number of reactions connecting metabolites.
    - DirAdjMatC: A pandas DataFrame containing the distinct reactions between metabolites.
    """
    num_mets, num_rxns = model.s_matrix.shape

    # Initialize adjacency matrices
    DirAdjMatW1 = np.zeros((num_mets, num_mets))
    DirAdjMatC = pd.DataFrame([[[] for _ in range(num_mets)] for _ in range(num_mets)], 
                              index=[met.id for met in model.metabolites], 
                              columns=[met.id for met in model.metabolites])

    # Fix erroneous and zero-zero bounds
    for rxn in model.reactions:
        if rxn.lower_bound == 0 and rxn.upper_bound == 0:
            print("Warning: Bounds error: lb = ub = 0!!! Automatically adjusted to -100 and +100.")
            rxn.lower_bound = -100
            rxn.upper_bound = 100
        elif rxn.lower_bound > rxn.upper_bound:
            print("Warning: Bounds error: lb > ub!!! Automatically adjusted to -100 and +100.")
            rxn.lower_bound = -100
            rxn.upper_bound = 100

    # Classify reactions as reversible, strictly positive, or strictly negative
    irrev_neg_ids = [rxn.id for rxn in model.reactions if rxn.upper_bound <= 0]  # strictly negative
    irrev_neg = [model.s_matrix.columns.get_loc(name) for name in irrev_neg_ids if name in model.s_matrix.columns]

    irrev_pos_ids = [rxn.id for rxn in model.reactions if rxn.lower_bound >= 0]  # strictly positive
    irrev_pos = [model.s_matrix.columns.get_loc(name) for name in irrev_pos_ids if name in model.s_matrix.columns]

    # Only follow the positive flow in the graph
    pos_s_matrix = model.s_matrix.values
    pos_s_matrix[:, irrev_neg] = 0  # Exclude strictly negative reactions

    for i, met_i in enumerate(model.metabolites):
        reactions = np.where(pos_s_matrix[i, :] < 0)[0]  # Find consuming reactions for metabolite i
        if reactions.size > 0:
            row, col = np.nonzero(pos_s_matrix[:, reactions] > 0)  # Find producing metabolites for these reactions

            if row.size > 0:
                for r, c in zip(row, col):
                    DirAdjMatC.iloc[i, r].append(model.reactions[reactions[c]].id)

                unique_rows, counts = np.unique(row, return_counts=True)
                DirAdjMatW1[i, unique_rows] += counts  # Increment connection count

    # Only follow the negative flow in the graph
    neg_s_matrix = model.s_matrix.values
    neg_s_matrix[:, irrev_pos] = 0  # Exclude strictly positive reactions

    for i, met_i in enumerate(model.metabolites):
        reactions = np.where(neg_s_matrix[i, :] > 0)[0]  # Find producing reactions for metabolite i
        if reactions.size > 0:
            row, col = np.nonzero(neg_s_matrix[:, reactions] < 0)  # Find consuming metabolites for these reactions

            if row.size > 0:
                for r, c in zip(row, col):
                    DirAdjMatC.iloc[i, r].append(model.reactions[reactions[c]].id)

                unique_rows, counts = np.unique(row, return_counts=True)
                DirAdjMatW1[i, unique_rows] += counts  # Add connection counts

    # Convert DirAdjMatW1 to binary for DirAdjMatWn
    DirAdjMatWn = DirAdjMatW1.copy()
    DirAdjMatW1[DirAdjMatW1 > 0] = 1

    return DirAdjMatW1, DirAdjMatWn, DirAdjMatC
