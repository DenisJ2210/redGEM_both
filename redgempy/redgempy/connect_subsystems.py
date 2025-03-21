import numpy as np
import pandas as pd

import numpy as np
from itertools import combinations


def extract_adj_data(
    model,
    core_ss,
    L_DirAdjMatWn,
    L_DirAdjMatC,
    L_DirAdjMatMets,
    D=1,
    start_from_min=True,
    only_connect_exclusive_mets=True,
    connect_intra_subsystem=False,
    apply_shortest_distance_of_subsystems=False,
    throw_error_on_d_violation=False,
):
    """
    Extract adjacency data from the directed adjacency matrices.

    Parameters:
    - model: A COBRApy metabolic model.
    - core_ss: List of selected core subsystems.
    - L_DirAdjMatWn: Directed adjacency matrix with weights for each depth.
    - L_DirAdjMatC: Pandas DataFrame containing distinct reactions for each depth.
    - L_DirAdjMatMets: Dictionary listing participating metabolites for each connection.
    - D: How many lengths to include in the new model (default = 1).
    - start_from_min: If True, start counting levels from the shortest distance (default = True).
    - only_connect_exclusive_mets: If True, only connect exclusive metabolites (default = True).
    - connect_intra_subsystem: If True, connect metabolites within the same subsystem (default = False).
    - apply_shortest_distance_of_subsystems: Apply shortest distance rule (default = False).
    - throw_error_on_d_violation: If True, throw an error on D violation (default = False).

    Returns:
    - selected_paths: Structure containing pathway information.
    - selected_paths_external: Structure containing pathways for extracellular metabolites.
    - connecting_reactions: Unique reactions connecting subsystems.
    - rxns_ss: Reactions for each subsystem.
    - mets_ss: Metabolites for each subsystem.
    - other_reactions: Non-core and non-connecting reactions.
    """

    # Find reactions and metabolites for each subsystem
    rxns_ss = {}
    mets_ss = {}
    for subsystem in core_ss:
        # Find reactions belonging to the subsystem
        rxns = [i for i in model.reactions if subsystem in i.subsystem]
        rxns_ss[subsystem] = [model.reactions.index(rxn) for rxn in rxns]

        # Find metabolites involved in these reactions
        mets = np.unique([met.id for rxn in rxns for met in rxn.metabolites])
        mets_ss[subsystem] = [model.metabolites.index(met) for met in mets]

    # Handle extracellular metabolites
    extra_mets = []
    if "ExtraCell" in core_ss:
        place = core_ss.index("ExtraCell")
        core_ss.pop(place)
        extra_mets = mets_ss["ExtraCell"]
        mets_ss.pop("ExtraCell")

    central_mets = []
    for i in core_ss:
        central_mets.extend(mets_ss[i])

    # Create source and target metabolite groups
    pairs = list(combinations(core_ss, 2))
    source_met = []
    target_met = []
    for pair in pairs:
        if only_connect_exclusive_mets:
            source_met.append(list(set(mets_ss[pair[0]]) - set(mets_ss[pair[1]])))
            target_met.append(list(set(mets_ss[pair[1]]) - set(mets_ss[pair[0]])))
        else:
            source_met.append(mets_ss[pair[0]])
            target_met.append(mets_ss[pair[1]])

    # Connect metabolites within the same subsystem if enabled
    if connect_intra_subsystem:
        for i in core_ss:
            pairs.append((i, i))
            source_met.append(mets_ss[i])
            target_met.append(mets_ss[i])

    # Extract paths from adjacency matrices
    selected_paths = paths_num(
        L_DirAdjMatWn,
        L_DirAdjMatC,
        L_DirAdjMatMets,
        pairs,
        core_ss,
        source_met,
        target_met,
        start_from_min,
        D,
        apply_shortest_distance_of_subsystems,
        throw_error_on_d_violation,
    )

    selected_paths_external = None
    # Here we treat all the core metabolites as one subsystem and connect them to the extracellular metabolites Only D = 1 for the shortest path
    if extra_mets:
        selected_paths_external = paths_num_extra(
            L_DirAdjMatWn,
            L_DirAdjMatC,
            L_DirAdjMatMets,
            extra_mets,
            core_ss,
            1,
            central_mets,
            apply_shortest_distance_of_subsystems,
        )

    # Final selection of connecting reactions
    all_connecting_reactions = []
    for d in range(D):
        for pair in pairs:
            all_connecting_reactions.extend(
                selected_paths["joiningReacs"][pair[0], pair[1], d]
            )

    if selected_paths_external and "ModelMets" in selected_paths_external:
        all_connecting_reactions = np.unique(
            all_connecting_reactions + list(selected_paths_external["ModelReacs"][0])
        )
    else:
        all_connecting_reactions = np.unique(all_connecting_reactions)

    # Find other reactions
    all_rxns = set(i.id for i in model.reactions)
    core_rxns = set(i.id for i in model.reactions if i.subsystem in core_ss)
    connecting_rxns = set(all_connecting_reactions)
    not_connected_rxns = all_rxns - connecting_rxns - core_rxns

    if selected_paths_external and "ModelMets" in selected_paths_external:
        other_reactions = list(
            not_connected_rxns - set(selected_paths_external["ModelReacs"][0])
        )
    else:
        other_reactions = list(not_connected_rxns)

    return (
        selected_paths,
        selected_paths_external,
        connecting_rxns,
        rxns_ss,
        mets_ss,
        other_reactions,
    )


def paths_num(
    L_DirAdjMatWn,
    L_DirAdjMatC,
    L_DirAdjMatMets,
    pairs,
    core_ss,
    source_mets,
    target_mets,
    start_from_min="yes",
    D=1,
    apply_shortest_distance_of_subsystems="bothways",
    throw_error_on_d_violation="warn",
):
    """
    Find the shortest distance from subsystem to subsystem in a directed adjacency matrix.

    Parameters:
    - L_DirAdjMatWn: 3D numpy array (weighted adjacency matrix).
    - L_DirAdjMatC: Dictionary of connections for each depth.
    - L_DirAdjMatMets: Dictionary of metabolites for each connection.
    - pairs: List of subsystem pairs (source, target).
    - source_mets: List of source metabolites for each pair.
    - target_mets: List of target metabolites for each pair.
    - start_from_min: If 'yes', start counting levels from the shortest distance (default = 'yes').
    - D: Maximum pathway length to include (default = 1).
    - apply_shortest_distance_of_subsystems: 'bothways' or 'eachdirection' (default = 'bothways').
    - throw_error_on_d_violation: Raise an error or warning for unconnected subsystems (default = 'warn').

    Returns:
    - selected_paths: Dictionary with pathway and reaction details.
    """
    num_ss = len(set(pair[0] for pair in pairs))
    d_max = L_DirAdjMatWn.shape[2]

    # Initialize shortest distances
    if start_from_min == "yes":
        shortest_L_sub2sub_num_p = np.zeros((num_ss, num_ss), dtype=int)
        if apply_shortest_distance_of_subsystems == "eachdirection":
            for pair_idx, (src, tgt) in enumerate(pairs):
                d = 0
                while d < d_max and not np.any(
                    L_DirAdjMatWn[source_mets[pair_idx], target_mets[pair_idx], d]
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[src, tgt] = d + 1

                d = 0
                while d < d_max and not np.any(
                    L_DirAdjMatWn[target_mets[pair_idx], source_mets[pair_idx], d]
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[tgt, src] = d + 1

        elif apply_shortest_distance_of_subsystems == "bothways":
            for pair_idx, (src, tgt) in enumerate(pairs):
                d = 0
                while d < d_max and not (
                    np.any(
                        L_DirAdjMatWn[source_mets[pair_idx], target_mets[pair_idx], d]
                    )
                    or np.any(
                        L_DirAdjMatWn[target_mets[pair_idx], source_mets[pair_idx], d]
                    )
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[src, tgt] = d + 1
                    shortest_L_sub2sub_num_p[tgt, src] = d + 1
        else:
            raise ValueError("Invalid option for apply_shortest_distance_of_subsystems")
    elif start_from_min == "no":
        shortest_L_sub2sub_num_p = np.ones((num_ss, num_ss), dtype=int)
    else:
        raise ValueError("Invalid option for start_from_min")

    shortest_L_sub2sub_num_p = pd.DataFrame(
        shortest_L_sub2sub_num_p, index=core_ss, columns=core_ss
    )

    # Check for disconnected subsystems
    disconnected_src, disconnected_tgt = np.where(shortest_L_sub2sub_num_p == 0)
    if len(disconnected_src) > 0:
        if throw_error_on_d_violation == "error":
            raise ValueError(
                f"L exceeded for subsystems: {disconnected_src} and {disconnected_tgt}"
            )
        elif throw_error_on_d_violation == "warn":
            print(
                f"Warning: L exceeded for subsystems: {disconnected_src} and {disconnected_tgt}"
            )

    # Initialize outputs
    joining_reacs = {(i, j, d): [] for i in core_ss for j in core_ss for d in range(D)}
    joining_mets = {(i, j, d): [] for i in core_ss for j in core_ss for d in range(D)}
    joining_paths = {(i, j, d): [] for i in core_ss for j in core_ss for d in range(D)}
    sub2sub_num_p = {(i, j, d): 0 for i in core_ss for j in core_ss for d in range(D)}
    sub2sub_num_r = {(i, j, d): 0 for i in core_ss for j in core_ss for d in range(D)}
    model_reacs = {d: set() for d in range(D)}
    model_mets = {d: set() for d in range(D)}

    # Extract reactions, pathways, and metabolites
    for el in range(D):
        for pair_idx, (src, tgt) in enumerate(pairs):
            if shortest_L_sub2sub_num_p.loc[src, tgt] > 0:
                d = el + shortest_L_sub2sub_num_p.loc[src, tgt] - 1
                if d < d_max:

                    total_paths = []
                    unique_reacs = []
                    unique_mets = []
                    for source_met in source_mets[pair_idx]:
                        for target_met in target_mets[pair_idx]:
                            connecting_rxns = L_DirAdjMatC.get(
                                (source_met, target_met, d), []
                            )
                            connecting_mets = L_DirAdjMatMets.get(
                                (source_met, target_met, d), []
                            )
                            if connecting_rxns != []:
                                total_paths.append(connecting_rxns)
                                if isinstance(connecting_rxns[0], list):
                                    connecting_rxns = [
                                        rxn
                                        for sublist in connecting_rxns
                                        for rxn in sublist
                                    ]
                                unique_reacs.extend(
                                    [
                                        rxn
                                        for rxn in connecting_rxns
                                        if rxn not in unique_reacs
                                    ]
                                )
                                unique_mets.extend(
                                    [
                                        met
                                        for met_list in connecting_mets
                                        for met in met_list
                                        if met not in unique_mets
                                    ]
                                )
                    joining_reacs[(src, tgt, el)] = unique_reacs

                    # Try add fix for compatibility with python 3.12.2 version
                    # total_paths = [tuple(path) for path in total_paths]
                    # unique_paths = np.unique(total_paths)
                    unique_paths = list({tuple(path) for path in total_paths})
                    joining_paths[(src, tgt, el)] = unique_paths

                    sub2sub_num_p[(src, tgt, el)] = len(joining_paths[(src, tgt, el)])
                    sub2sub_num_r[(src, tgt, el)] = len(unique_reacs)

                    model_reacs[el].update(unique_reacs)
                    joining_mets[(src, tgt, el)] = unique_mets
                    model_mets[el].update(unique_mets)

    selected_paths = {
        "sub2subNumP": sub2sub_num_p,
        "sub2subNumR": sub2sub_num_r,
        "joiningReacs": joining_reacs,
        "joiningPaths": joining_paths,
        "joiningMets": joining_mets,
        "ModelReacs": model_reacs,
        "ModelMets": model_mets,
    }

    return selected_paths


def paths_num_extra(
    L_DirAdjMatWn,
    L_DirAdjMatC,
    L_DirAdjMatMets,
    extra_cell_mets,
    core_ss,
    D=1,
    central_mets=None,
    apply_shortest_distance_of_subsystems="bothways",
):
    """
    Find the shortest distance from central metabolites to extracellular metabolites.

    Parameters:
    - L_DirAdjMatWn: 3D numpy array (weighted adjacency matrix).
    - L_DirAdjMatC: Dictionary of connections for each depth.
    - L_DirAdjMatMets: Dictionary of metabolites for each connection.
    - extra_cell_mets: List of extracellular metabolites.
    - D: Maximum pathway length to include (default = 1).
    - central_mets: List of central metabolites (default = None).
    - apply_shortest_distance_of_subsystems: 'bothways' or 'eachdirection' (default = 'bothways').

    Returns:
    - selected_paths: Dictionary with pathway and reaction details.
    """
    if central_mets is None:
        raise ValueError("Central metabolites must be provided.")

    # Create source and target metabolite groups
    source_mets = [central_mets for i in extra_cell_mets]
    target_mets = extra_cell_mets
    pairs = [(1, i) for i in range(len(extra_cell_mets))]

    num_ss = len(extra_cell_mets)
    shortest_plus_l = True
    d_max = L_DirAdjMatWn.shape[2]

    # Initialize shortest distances
    if shortest_plus_l:
        shortest_L_sub2sub_num_p = np.zeros((num_ss, num_ss), dtype=int)
        if apply_shortest_distance_of_subsystems == "eachdirection":
            for pair_idx, (src, tgt) in enumerate(pairs):
                d = 0
                while d < d_max and not np.any(
                    L_DirAdjMatWn[source_mets[pair_idx], target_mets[pair_idx], d]
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[src, tgt] = d + 1

                d = 0
                while d < d_max and not np.any(
                    L_DirAdjMatWn[target_mets[pair_idx], source_mets[pair_idx], d]
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[tgt, src] = d + 1

        elif apply_shortest_distance_of_subsystems == "bothways":
            for pair_idx, (src, tgt) in enumerate(pairs):
                d = 0
                while d < d_max and not (
                    np.any(
                        L_DirAdjMatWn[source_mets[pair_idx], target_mets[pair_idx], d]
                    )
                    or np.any(
                        L_DirAdjMatWn[target_mets[pair_idx], source_mets[pair_idx], d]
                    )
                ):
                    d += 1
                if d < d_max:
                    shortest_L_sub2sub_num_p[src, tgt] = d + 1
                    shortest_L_sub2sub_num_p[tgt, src] = d + 1
        else:
            raise ValueError("Invalid option for apply_shortest_distance_of_subsystems")
    else:
        shortest_L_sub2sub_num_p = np.ones((num_ss, num_ss), dtype=int)

    shortest_L_sub2sub_num_p = pd.DataFrame(
        shortest_L_sub2sub_num_p,
        index=[i[0] for i in pairs],
        columns=[i[1] for i in pairs],
    )

    # Initialize outputs
    joining_reacs = {
        (i, j, d): [] for i in range(num_ss) for j in range(num_ss) for d in range(D)
    }
    joining_mets = {
        (i, j, d): [] for i in range(num_ss) for j in range(num_ss) for d in range(D)
    }
    joining_paths = {
        (i, j, d): [] for i in range(num_ss) for j in range(num_ss) for d in range(D)
    }
    sub2sub_num_p = {
        (i, j, d): 0 for i in range(num_ss) for j in range(num_ss) for d in range(D)
    }
    sub2sub_num_r = {
        (i, j, d): 0 for i in range(num_ss) for j in range(num_ss) for d in range(D)
    }
    model_reacs = {d: set() for d in range(D)}
    model_mets = {d: set() for d in range(D)}

    # Extract reactions, pathways, and metabolites
    for el in range(D):
        for pair_idx, (src, tgt) in enumerate(pairs):
            if shortest_L_sub2sub_num_p.iloc[src, tgt] > 0:
                d = el + shortest_L_sub2sub_num_p.iloc[src, tgt] - 1
                if d < d_max:

                    total_paths = []
                    unique_reacs = []
                    unique_mets = []
                    for source_met in source_mets[pair_idx]:
                        target_met = target_mets[pair_idx]

                        connecting_rxns = L_DirAdjMatC.get(
                            (source_met, target_met, d), []
                        )
                        connecting_mets = L_DirAdjMatMets.get(
                            (source_met, target_met, d), []
                        )
                        if connecting_rxns != []:
                            total_paths.append(connecting_rxns)
                            if isinstance(connecting_rxns[0], list):
                                connecting_rxns = [
                                    rxn
                                    for sublist in connecting_rxns
                                    for rxn in sublist
                                ]
                            unique_reacs.extend(
                                [
                                    rxn
                                    for rxn in connecting_rxns
                                    if rxn not in unique_reacs
                                ]
                            )
                            unique_mets.extend(
                                [
                                    met
                                    for met_list in connecting_mets
                                    for met in met_list
                                    if met not in unique_mets
                                ]
                            )

                    joining_reacs[(src, tgt, el)] = unique_reacs
                    joining_paths[(src, tgt, el)] = np.unique(total_paths)

                    sub2sub_num_p[(src, tgt, el)] = len(joining_paths[(src, tgt, el)])
                    sub2sub_num_r[(src, tgt, el)] = len(unique_reacs)

                    model_reacs[el].update(unique_reacs)
                    joining_mets[(src, tgt, el)] = unique_mets
                    model_mets[el].update(unique_mets)

    selected_paths = {
        "sub2subNumP": sub2sub_num_p,
        "sub2subNumR": sub2sub_num_r,
        "joiningReacs": joining_reacs,
        "joiningPaths": joining_paths,
        "joiningMets": joining_mets,
        "ModelReacs": model_reacs,
        "ModelMets": model_mets,
    }

    return selected_paths
