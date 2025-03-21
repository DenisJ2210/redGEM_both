#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RedGEMPy: A Python implementation of RedGEM for reduction of genome-scale metabolic models.

This module serves as the main entry point for the RedGEM workflow, orchestrating the
process of reducing genome-scale metabolic models to create more focused, core models
with lumped reactions representing complex pathways.

References:
    * RedGEM: Ataman, M., et al., "redGEM: Systematic reduction and analysis of genome-scale
      metabolic reconstructions for development of consistent core metabolic models".
      Plos Computational Biology, 2017. 13(7).
    * RedGEMX: Maria Masid, Meriç Ataman and Vassily Hatzimanikatis.
      "redHUMAN: analyzing human metabolism and growth media through systematic reductions
      of thermodynamically curated genome-scale models"
    * LumpGEM: Meric Ataman and Vassily Hatzimanikatis.
      "lumpGEM: Systematic generation of subnetworks and elementally balanced lumped reactions
      for the biosynthesis of target metabolites". Plos Computational Biology, No. 13(7).
"""

import os
import datetime
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Set

# Import core modules
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
    remove_thermo_fields,
    identify_biomass_reaction,
    identify_transport_reactions,
    remove_futile_reactions,
)
from redgempy.removeMetPairsFromS import remove_met_pairs_from_S
from redgempy.adjacentfromstoich import adjacent_from_stoich
from redgempy.connect_subsystems import extract_adj_data
from redgempy.annotation import get_seed_ids_for_rxn

# Import LumpGEM modules
from redgempy.lumpgem import add_biomass
from redgempy.gem_optimization import find_max_biomass_growth_rate

# Run FVA to identify blocked reactions
from cobra.flux_analysis import flux_variability_analysis


class RedGEM:
    """
    Main RedGEM workflow class for reducing genome-scale metabolic models.

    This class orchestrates the entire RedGEM workflow, from setup to final model reduction,
    including the lumping of metabolic pathways for a more compact, focused model.

    Attributes:
        config (Dict): Configuration parameters for the reduction process.
        paths (Dict): Directory paths for output and working files.
        logger (logging.Logger): Logger for tracking workflow progress.
    """

    def __init__(self, config: Dict):
        """
        Initialize the RedGEM workflow with configuration parameters.

        Args:
            config (Dict): Configuration dictionary containing parameters for the reduction.
        """
        self.config = config
        self.logger = self._setup_logger()
        self.paths = setup_directories(config)

        # Initialize workflow state variables
        self.original_gem = None
        self.gem_model = None
        self.gem_for_adjmat = None
        self.core_subsystems = []
        self.biomass_reaction_ids = []
        self.met_pairs_to_remove = []
        self.inorganic_met_ids = []
        self.bbbs_to_exclude = []
        self.extracellular_subsystem = []
        self.oxphos_subsystem = ""

        # Initialize graph analysis variables
        self.dir_adj_mat_wn = None
        self.dir_adj_mat_c = None
        self.l_dir_adj_mat_wn = None
        self.l_dir_adj_mat_c = None
        self.l_dir_adj_mat_mets = None

        # Initialize pathway variables
        self.selected_paths = None
        self.selected_paths_external = None
        self.connecting_reactions = None
        self.rxns_ss = None
        self.mets_ss = None
        self.other_reactions = None

        # Initialize lumping variables
        self.active_rxns = None
        self.lumped_rxn_formulas = None
        self.bbb_names = None
        self.dps_all = None
        self.rxns = None

        # Initialize the final reduced model
        self.red_model = None
        self.checkpoint_dir = Path(self.paths["temp_workspace"]) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the RedGEM workflow."""
        logger = logging.getLogger("RedGEM")
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

        # Create file handler if output_path exists
        if "output_path" in self.config:
            log_file = Path(self.config["output_path"]) / "redgem.log"
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def run(self) -> None:
        """
        Execute the complete RedGEM workflow.

        This method orchestrates the entire RedGEM process in sequence:
        1. Setup and data loading
        2. Model processing
        3. Adjacency matrix computation
        4. Pathway extraction
        5. Lumping of pathways
        6. Final model creation
        """
        try:
            # Step 1: Setup and process case
            self.logger.info("Starting RedGEM workflow...")
            self._process_case()
            save_workspace(self.config, "Step 1: Setup")

            # Step 2: Prepare model for adjacency matrix
            self._prepare_model_for_adjmat()
            save_workspace(self.config, "Step 2: Model Preparation")

            # Step 3: Compute adjacency matrices
            self._compute_adjacency_matrices()
            save_workspace(self.config, "Step 3: Adjacency Matrices")

            # Step 4: Extract pathways
            self._extract_pathways()
            save_workspace(self.config, "Step 4: Pathway Extraction")

            # Step 5: Build reduced model with lumped reactions
            self._build_reduced_model()
            save_workspace(self.config, "Step 5: Reduced Model Construction")

            # Step 6: Add lumped reactions
            if self.config.get("perform_lumpgem", "yes").lower() == "yes":
                self._add_lumped_reactions()
                save_workspace(self.config, "Step 6: Add Lumped Reactions")

            # Step 7: Post-processing
            if self.config.get("perform_post_processing", "no").lower() != "no":
                self._post_process_model()
                save_workspace(self.config, "Step 7: Post-Processing")

            # Save final model
            self._save_final_model()
            self.logger.info("RedGEM workflow completed successfully!")

        except Exception as e:
            self.logger.error(f"Error in RedGEM workflow: {str(e)}", exc_info=True)
            raise

    def _process_case(self) -> None:
        """Process the specific organism case and load the GEM."""
        self.logger.info("Processing organism case and loading GEM...")

        # Check for existing checkpoint
        checkpoint_file = self.checkpoint_dir / "processed_case.pkl"
        processed_case = load_checkpoint(checkpoint_file)

        if processed_case is None:
            self.logger.info("No checkpoint found. Processing case from scratch...")
            processed_case = process_case(self.config)
            save_checkpoint(processed_case, checkpoint_file)
        else:
            self.logger.info("Loaded processed case from checkpoint.")

        # Unpack the processed case
        (
            self.original_gem,
            self.gem_model,
            self.core_subsystems,
            self.biomass_reaction_ids,
            self.met_pairs_to_remove,
            self.inorganic_met_ids,
            self.bbbs_to_exclude,
            self.extracellular_subsystem,
            self.oxphos_subsystem,
        ) = processed_case

        # Define the biomass reaction in the model
        biomass_reaction, _, self.gem_model = identify_biomass_reaction(self.gem_model)
        self.gem_model.biomass_reaction = biomass_reaction

        # Additional model processing based on configuration
        if self.config.get("prevent_bbb_uptake", "no").lower() == "yes":
            self.gem_model = prevent_bbb_uptake(self.gem_model)

        if self.config.get("add_etc_as_subsystem", "no").lower() == "yes":
            self.gem_model = add_etc_subsystem(
                self.gem_model, ox_phos_subsystem=self.oxphos_subsystem
            )

        if self.extracellular_subsystem:
            self.gem_model = add_extracellular_subsystem(
                self.gem_model, self.extracellular_subsystem
            )

    def _prepare_model_for_adjmat(self) -> None:
        """Prepare the model for adjacency matrix calculation."""
        self.logger.info("Preparing model for adjacency matrix calculation...")

        # Check for existing checkpoint
        checkpoint_file = self.checkpoint_dir / "gem_for_adjmat.pkl"
        self.gem_for_adjmat = load_checkpoint(checkpoint_file)

        if self.gem_for_adjmat is None:
            self.logger.info("No checkpoint found. Preparing model from scratch...")
            self.gem_for_adjmat = self.gem_model.copy()

            # Find and remove biomass reactions
            biomass_rxns = [
                rxn.id
                for rxn in self.gem_for_adjmat.reactions
                if rxn.id in self.biomass_reaction_ids
            ]
            self.gem_for_adjmat.remove_reactions(biomass_rxns)

            # Remove thermodynamic fields if needed
            if self.config.get("impose_thermodynamics", "no").lower() == "no":
                self.gem_for_adjmat = remove_thermo_fields(self.gem_for_adjmat)

            # Create stoichiometric matrix as DataFrame
            met_ids = [met.id for met in self.gem_for_adjmat.metabolites]
            rxn_ids = [rxn.id for rxn in self.gem_for_adjmat.reactions]
            s_matrix = pd.DataFrame(0, index=met_ids, columns=rxn_ids)

            # Fill the stoichiometric matrix
            for met_id in met_ids:
                met = self.gem_for_adjmat.metabolites.get_by_id(met_id)
                for rxn in met.reactions:
                    s_matrix.loc[met_id, rxn.id] = rxn.metabolites[met]

            self.gem_for_adjmat.s_matrix = s_matrix

            # Remove cofactors and small inorganic metabolites from S matrix
            self.gem_for_adjmat.s_matrix = remove_met_pairs_from_S(
                self.gem_for_adjmat, self.met_pairs_to_remove, self.inorganic_met_ids
            )

            save_checkpoint(self.gem_for_adjmat, checkpoint_file)
        else:
            self.logger.info("Loaded GEM for adjacency matrix from checkpoint.")

    def _compute_adjacency_matrices(self) -> None:
        """Compute adjacency matrices for pathway analysis."""
        self.logger.info("Computing adjacency matrices...")

        # Check for existing checkpoint
        adjacency_file = self.checkpoint_dir / "adjacency_matrices.pkl"
        adjacency_data = load_checkpoint(adjacency_file)

        if adjacency_data is None:
            self.logger.info(
                "No checkpoint found. Computing adjacency matrices from scratch..."
            )
            _, self.dir_adj_mat_wn, self.dir_adj_mat_c = adjacent_from_stoich(
                self.gem_for_adjmat
            )
            adjacency_data = (self.dir_adj_mat_wn, self.dir_adj_mat_c)
            save_checkpoint(adjacency_data, adjacency_file)
        else:
            self.logger.info("Loaded adjacency matrices from checkpoint.")
            self.dir_adj_mat_wn, self.dir_adj_mat_c = adjacency_data

        # Compute pathway matrices for length L
        l_matrices_file = self.checkpoint_dir / "l_matrices.pkl"
        l_matrices = load_checkpoint(l_matrices_file)

        L = int(self.config.get("l", 10))

        if l_matrices is None:
            self.logger.info(f"Computing pathway matrices for length L={L}...")

            # Initialize pathway matrices
            num_mets = len(self.dir_adj_mat_c.index)
            self.l_dir_adj_mat_wn = np.zeros((num_mets, num_mets, L), dtype=int)
            self.l_dir_adj_mat_c = {}
            self.l_dir_adj_mat_mets = {}

            # Compute all paths up to length L
            self._compute_all_paths(L)

            # Save the matrices
            l_matrices = (
                self.l_dir_adj_mat_wn,
                self.l_dir_adj_mat_c,
                self.l_dir_adj_mat_mets,
            )
            save_checkpoint(l_matrices, l_matrices_file)
        else:
            self.logger.info("Loaded path matrices from checkpoint.")
            self.l_dir_adj_mat_wn, self.l_dir_adj_mat_c, self.l_dir_adj_mat_mets = (
                l_matrices
            )

    def _compute_all_paths(self, L: int) -> None:
        """
        Compute all paths up to length L in the network.

        Args:
            L (int): Maximum path length to compute.
        """
        self.logger.info(f"Computing all paths up to length {L}...")

        # Get metabolite IDs
        metabolites = self.dir_adj_mat_c.index.tolist()
        num_mets = len(metabolites)

        # Initialize L=1 with direct adjacency matrices
        self.l_dir_adj_mat_wn[:, :, 0] = self.dir_adj_mat_wn

        for i, met_i in enumerate(metabolites):
            for j, met_j in enumerate(metabolites):
                if self.dir_adj_mat_wn[i, j] > 0:
                    self.l_dir_adj_mat_c[(i, j, 0)] = self.dir_adj_mat_c.at[
                        met_i, met_j
                    ]
                    self.l_dir_adj_mat_mets[(i, j, 0)] = [
                        [i, j] for _ in range(len(self.dir_adj_mat_c.at[met_i, met_j]))
                    ]
                else:
                    self.l_dir_adj_mat_c[(i, j, 0)] = []
                    self.l_dir_adj_mat_mets[(i, j, 0)] = []

        # Find L=1 connections for faster lookup
        r1, c1 = np.nonzero(self.dir_adj_mat_wn)
        connection_map = {i: np.unique(c1[r1 == i]) for i in range(num_mets)}

        # Compute paths for L > 1
        for l in range(1, L):
            self.logger.info(f"Computing paths of length {l+1}...")

            # Get connections from previous level
            if l == 1:
                prev_connections = list(zip(r1, c1))
            else:
                prev_r, prev_c = np.nonzero(self.l_dir_adj_mat_wn[:, :, l - 1])
                prev_connections = list(zip(prev_r, prev_c))

            for start_met, mid_met in prev_connections:
                # Get previous paths and reactions
                prev_paths = self.l_dir_adj_mat_mets.get(
                    (start_met, mid_met, l - 1), []
                )
                prev_reactions = self.l_dir_adj_mat_c.get(
                    (start_met, mid_met, l - 1), []
                )

                if not prev_paths:
                    continue

                # Get next connections
                next_mets = connection_map.get(mid_met, [])

                # Filter out mets that would create cycles
                unique_prev_mets = set()
                for path in prev_paths:
                    unique_prev_mets.update(path)

                next_mets = [
                    m for m in next_mets if m not in unique_prev_mets and m != start_met
                ]

                for end_met in next_mets:
                    # Get connecting reactions for this step
                    one_step_reactions = self.l_dir_adj_mat_c.get(
                        (mid_met, end_met, 0), []
                    )

                    if not one_step_reactions:
                        continue

                    # Build new paths
                    new_paths = []
                    new_reactions = []

                    for i, prev_path in enumerate(prev_paths):
                        for one_step_reaction in one_step_reactions:
                            # Create new path by extending previous path
                            new_path = prev_path.copy()
                            new_path.append(end_met)
                            new_paths.append(new_path)

                            # Create new reaction sequence
                            if isinstance(prev_reactions[i], list):
                                new_reaction = prev_reactions[i].copy()
                                new_reaction.append(one_step_reaction)
                            else:
                                new_reaction = [prev_reactions[i], one_step_reaction]

                            new_reactions.append(new_reaction)

                    # Store the new paths and reactions
                    if (start_met, end_met, l) in self.l_dir_adj_mat_c:
                        self.l_dir_adj_mat_c[(start_met, end_met, l)].extend(
                            new_reactions
                        )
                        self.l_dir_adj_mat_mets[(start_met, end_met, l)].extend(
                            new_paths
                        )
                    else:
                        self.l_dir_adj_mat_c[(start_met, end_met, l)] = new_reactions
                        self.l_dir_adj_mat_mets[(start_met, end_met, l)] = new_paths

                    # Update the weighted adjacency matrix
                    self.l_dir_adj_mat_wn[start_met, end_met, l] = len(
                        self.l_dir_adj_mat_c[(start_met, end_met, l)]
                    )

        # Clean up paths that have duplicates in the middle
        self._clean_paths(L)

    def _clean_paths(self, L: int) -> None:
        """
        Clean paths by removing those with duplicates.

        Args:
            L (int): Maximum path length.
        """
        self.logger.info("Cleaning paths to remove duplicates...")

        for l in range(1, L):
            nonzero_indices = np.nonzero(self.l_dir_adj_mat_wn[:, :, l])

            for start_met, end_met in zip(nonzero_indices[0], nonzero_indices[1]):
                paths = self.l_dir_adj_mat_mets.get((start_met, end_met, l), [])
                reactions = self.l_dir_adj_mat_c.get((start_met, end_met, l), [])

                if not paths:
                    continue

                # Find paths with duplicates
                valid_indices = []
                for i, path in enumerate(paths):
                    if len(set(path)) == len(path):  # No duplicates
                        valid_indices.append(i)

                # If some paths need to be removed
                if len(valid_indices) < len(paths):
                    # Keep only the valid paths and reactions
                    new_paths = [paths[i] for i in valid_indices]
                    new_reactions = [reactions[i] for i in valid_indices]

                    self.l_dir_adj_mat_mets[(start_met, end_met, l)] = new_paths
                    self.l_dir_adj_mat_c[(start_met, end_met, l)] = new_reactions

                    # Update the weighted adjacency matrix
                    if new_paths:
                        self.l_dir_adj_mat_wn[start_met, end_met, l] = len(new_paths)
                    else:
                        self.l_dir_adj_mat_wn[start_met, end_met, l] = 0

    def _extract_pathways(self) -> None:
        """Extract pathways connecting subsystems."""
        self.logger.info("Extracting pathways connecting subsystems...")

        # Check for existing checkpoint
        pathways_file = self.checkpoint_dir / "pathways.pkl"
        pathways_data = load_checkpoint(pathways_file)

        if pathways_data is None:
            self.logger.info("No checkpoint found. Extracting pathways from scratch...")

            # Make sure all necessary subsystems are in the list
            core_ss = list(set(self.core_subsystems))

            # Add ETC and ExtraCell if specified
            if self.config.get("add_etc_as_subsystem", "no").lower() == "yes":
                core_ss.append("ETC_Rxns")

            if self.extracellular_subsystem:
                core_ss.append("ExtraCell")

            # Extract pathways with the parameters
            D = int(self.config.get("d", 4))
            start_from_min = self.config.get("start_from_min", "yes")
            only_connect_exclusive_mets = self.config.get(
                "only_connect_exclusive_mets", "yes"
            )
            connect_intra_subsystem = self.config.get(
                "connect_intracellular_subsystems", "no"
            )
            apply_shortest_distance = self.config.get(
                "apply_shortest_distance_of_subsystems", "bothways"
            )
            throw_error_on_d_violation = self.config.get(
                "throw_error_on_d_violation", "error"
            )

            (
                self.selected_paths,
                self.selected_paths_external,
                self.connecting_reactions,
                self.rxns_ss,
                self.mets_ss,
                self.other_reactions,
            ) = extract_adj_data(
                self.gem_for_adjmat,
                core_ss,
                self.l_dir_adj_mat_wn,
                self.l_dir_adj_mat_c,
                self.l_dir_adj_mat_mets,
                D,
                start_from_min,
                only_connect_exclusive_mets,
                connect_intra_subsystem,
                apply_shortest_distance,
                throw_error_on_d_violation,
            )

            pathways_data = (
                self.selected_paths,
                self.selected_paths_external,
                self.connecting_reactions,
                self.rxns_ss,
                self.mets_ss,
                self.other_reactions,
            )
            save_checkpoint(pathways_data, pathways_file)
        else:
            self.logger.info("Loaded pathways from checkpoint.")
            (
                self.selected_paths,
                self.selected_paths_external,
                self.connecting_reactions,
                self.rxns_ss,
                self.mets_ss,
                self.other_reactions,
            ) = pathways_data

    def _build_reduced_model(self) -> None:
        """Build the reduced model with core and connecting reactions."""
        self.logger.info("Building reduced model...")

        # Get core subsystems
        core_ss = list(set(self.core_subsystems))
        if self.config.get("add_etc_as_subsystem", "no").lower() == "yes":
            core_ss.append("ETC_Rxns")
        if self.extracellular_subsystem:
            core_ss.append("ExtraCell")

        # Get all reactions in core subsystems
        core_rxns = set()
        for subsystem in core_ss:
            for rxn in self.gem_model.reactions:
                if rxn.subsystem == subsystem:
                    core_rxns.add(rxn.id)

        # Combine core and connecting reactions
        all_core_rxns = list(set(list(self.connecting_reactions) + list(core_rxns)))

        # Create initial submodel with core and connecting reactions
        submodel = self.gem_model.copy()
        reactions_to_remove = [
            rxn for rxn in submodel.reactions if rxn.id not in all_core_rxns
        ]
        submodel.remove_reactions(reactions_to_remove)

        # Check for reactions that use only core metabolites and add them to the core
        additional_core_rxns = []
        submodel_met_ids = [met.id for met in submodel.metabolites]

        for rxn in self.gem_model.reactions:
            if rxn.id in all_core_rxns:
                continue

            # Check if all metabolites in the reaction are in the submodel
            mets_in_rxn = [met.id for met in rxn.metabolites]
            if all(met_id in submodel_met_ids for met_id in mets_in_rxn):
                additional_core_rxns.append(rxn.id)

        # If additional reactions found, rebuild the submodel with them included
        if additional_core_rxns:
            all_core_rxns = list(set(all_core_rxns + additional_core_rxns))
            submodel = self.gem_model.copy()
            reactions_to_remove = [
                rxn for rxn in submodel.reactions if rxn.id not in all_core_rxns
            ]
            submodel.remove_reactions(reactions_to_remove)

        # Clean up unused metabolites and genes
        metabolites_to_remove = [
            met for met in submodel.metabolites if len(met.reactions) == 0
        ]
        submodel.remove_metabolites(metabolites_to_remove)

        # Remove unused genes (must be done after reaction removal)
        genes_to_remove = []
        for gene in submodel.genes:
            if len(gene.reactions) == 0:
                genes_to_remove.append(gene)

        for gene in genes_to_remove:
            submodel.genes.remove(gene)

        self.red_model = submodel

        # Save information about ATP synthase and ETC reactions
        self._identify_special_reaction_groups()

    def _identify_special_reaction_groups(self) -> None:
        """Identify special reaction groups like ATP synthase and ETC reactions."""
        self.logger.info("Identifying special reaction groups...")

        # Identify non-core reactions
        core_rxns = set()
        for subsystem in self.core_subsystems:
            for rxn in self.gem_model.reactions:
                if rxn.subsystem == subsystem:
                    core_rxns.add(rxn.id)

        non_core_rxns = [
            rxn for rxn in self.gem_model.reactions if rxn.id not in core_rxns
        ]

        # Identify ATP synthase reactions based on metabolites
        atp_synth_rxn_ids = []
        SEEDIDs_atpSynth = {"cpd00008", "cpd00009", "cpd00067", "cpd00002", "cpd00001"}

        for rxn in self.gem_model.reactions:
            seed_ids_rxn = get_seed_ids_for_rxn(rxn)
            if seed_ids_rxn.symmetric_difference(SEEDIDs_atpSynth) == set():
                atp_synth_rxn_ids.append(rxn.id)

        # Identify ETC reactions
        etc_rxn_ids = [
            rxn.id for rxn in self.gem_model.reactions if rxn.subsystem == "ETC_Rxns"
        ]

        # Store for later use in add_biomass
        self.atp_synth_rxn_names = atp_synth_rxn_ids
        self.etc_rxn_names = etc_rxn_ids
        self.rxn_names_prev_therm_relax = etc_rxn_ids + atp_synth_rxn_ids
        self.non_core_rxn_ids = [rxn.id for rxn in non_core_rxns]

    def _add_lumped_reactions(self) -> None:
        """Add lumped reactions for biomass building blocks."""
        self.logger.info("Adding lumped reactions for biomass building blocks...")

        # Check if we should skip lumping
        if self.config.get("perform_lumpgem", "yes").lower() != "yes":
            self.logger.info("Skipping LumpGEM as per configuration.")
            return

        # Execute the add_biomass function
        lumping_results = add_biomass(
            model=self.gem_model,
            non_core_rxn_ids=self.non_core_rxn_ids,
            thermo_dir_path=self.config.get("thermo_data_path", ""),
            biomass_building_blocks_to_exclude=self.bbbs_to_exclude,
            oxygen_metabolite_id="o2_e",  # TODO: Make this configurable
            aerobic_anaerobic=self.config.get("aerobic_anaerobic", "aerobic"),
            organism=self.config.get("organism", ""),
            num_of_lumped=self.config.get("num_of_lumped", "Smin"),
            gem_name=self.config.get("gem_name", ""),
            rxn_names_prev_therm_relax=self.rxn_names_prev_therm_relax,
            biomass_rxn_names=self.biomass_reaction_ids,
            atp_synth_rxn_names=self.atp_synth_rxn_names,
            add_gam=self.config.get("add_gam", "yes").lower() == "yes",
            percent_mu_max_lumping=int(
                self.config.get("percent_of_mu_max_for_lumping", 90)
            ),
            impose_thermodynamics=self.config.get("impose_thermodynamics", "no"),
            output_path=self.config.get("output_path", ""),
        )

        # Store the lumping results
        (
            self.active_rxns,
            self.lumped_rxn_formulas,
            self.bbb_names,
            self.dps_all,
            self.id_ncntner,
            self.relaxed_dgo_vars_values,
        ) = lumping_results

    def _post_process_model(self) -> None:
        """Post-process the reduced model."""
        self.logger.info("Post-processing the reduced model...")

        # Post-processing options
        post_processing = self.config.get("perform_post_processing", "no").lower()

        if post_processing == "no":
            self.logger.info("No post-processing requested.")
            return

        # Remove blocked reactions (FVA and TFA)
        if post_processing == "pp_removeblockrxns":
            self._remove_blocked_reactions()

        # Prepare for MCA
        elif post_processing == "pp_formca":
            self._prepare_for_mca()

        # Custom post-processing
        else:
            self.logger.info(f"Custom post-processing: {post_processing}")
            # Implement custom post-processing steps here

    def _remove_blocked_reactions(self) -> None:
        """Remove blocked reactions using FVA and TFA."""
        self.logger.info("Removing blocked reactions...")

        # First, set growth bounds to match the original GEM
        original_biomass_rxn = None
        for rxn_id in self.biomass_reaction_ids:
            try:
                original_biomass_rxn = self.original_gem.reactions.get_by_id(rxn_id)
                break
            except KeyError:
                continue

        if original_biomass_rxn is None:
            self.logger.warning(
                "Original biomass reaction not found. Using default bounds."
            )
            orig_growth_lb = 0
            orig_growth_ub = 100
        else:
            orig_growth_lb = original_biomass_rxn.lower_bound
            orig_growth_ub = min(original_biomass_rxn.upper_bound, 100)

        # Apply the bounds to the reduced model
        biomass_rxn = self.red_model.biomass_reaction
        biomass_rxn.lower_bound = orig_growth_lb
        biomass_rxn.upper_bound = orig_growth_ub

        # Set minimal growth requirement (10% of max)
        solution = self.red_model.optimize()
        if solution.status == "optimal":
            min_growth = solution.objective_value * 0.1
            biomass_rxn.lower_bound = min_growth

        try:
            fva_result = flux_variability_analysis(self.red_model)
            blocked_rxns = [
                rxn
                for rxn, result in fva_result.iterrows()
                if abs(result["minimum"]) < 1e-6 and abs(result["maximum"]) < 1e-6
            ]

            self.logger.info(f"FVA identified {len(blocked_rxns)} blocked reactions.")

            # Remove blocked reactions one by one, checking feasibility
            for rxn_id in blocked_rxns:
                model_copy = self.red_model.copy()
                model_copy.reactions.get_by_id(rxn_id).remove_from_model()
                solution = model_copy.optimize()

                if (
                    solution.status == "optimal"
                    and solution.objective_value >= min_growth
                ):
                    self.logger.info(f"Removing blocked reaction: {rxn_id}")
                    self.red_model.reactions.get_by_id(rxn_id).remove_from_model()
                else:
                    self.logger.info(
                        f"Keeping reaction {rxn_id} despite being blocked (removal causes infeasibility)"
                    )

        except Exception as e:
            self.logger.error(f"Error during FVA: {e}")

        # If thermodynamics are imposed, run TFA as well
        if self.config.get("impose_thermodynamics", "no").lower() == "yes":
            self.logger.info(
                "Running thermodynamic-based analysis to identify blocked reactions..."
            )
            # TFA implementation would go here
            # This would need to use pytfa or a similar library

    def _prepare_for_mca(self) -> None:
        """Prepare the model for Metabolic Control Analysis."""
        self.logger.info("Preparing model for MCA...")
        # MCA preparation steps would go here

    def _save_final_model(self) -> None:
        """Save the final reduced model."""
        self.logger.info("Saving final reduced model...")

        # Create model name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        red_model_name = self.config.get("red_model_name", "RedModel")
        model_name = f"{red_model_name}_{timestamp}"

        # Add metadata to the model
        self.red_model.id = model_name
        self.red_model.name = f"RedGEM model from {self.config.get('gem_name', '')}"

        # Add reduction properties
        self.red_model.redgem_config = self.config
        self.red_model.redgem_version = "1.0.0"  # Update with actual version
        self.red_model.redgem_date = datetime.datetime.now().isoformat()

        # Save in the models directory
        model_path = os.path.join(
            self.paths["user_output_models"], f"{model_name}.json"
        )

        try:
            # Save as JSON (can be easily loaded by COBRApy)
            from cobra.io import save_json_model

            save_json_model(self.red_model, model_path)
            self.logger.info(f"Model saved to {model_path}")

            # Optionally save in MATLAB format if available
            try:
                from cobra.io import save_matlab_model

                matlab_path = os.path.join(
                    self.paths["user_output_models"], f"{model_name}.mat"
                )
                save_matlab_model(self.red_model, matlab_path)
                self.logger.info(f"Model also saved in MATLAB format to {matlab_path}")
            except ImportError:
                self.logger.warning(
                    "MATLAB save functionality not available. Skipping MATLAB format."
                )

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

        # Save additional model data separately (if needed)
        if hasattr(self, "info_LMDP") and self.info_LMDP:
            import pickle

            info_path = os.path.join(
                self.paths["user_output_models"], f"{model_name}_info_LMDP.pkl"
            )
            with open(info_path, "wb") as f:
                pickle.dump(self.info_LMDP, f)
            self.logger.info(f"LMDP info saved to {info_path}")


def run_redgem(config_file=None, config_dict=None):
    """
    Run the RedGEM workflow with the provided configuration.

    Args:
        config_file (str, optional): Path to a configuration file.
        config_dict (dict, optional): Dictionary containing configuration parameters.

    Returns:
        RedGEM: The RedGEM instance with the completed workflow.

    Raises:
        ValueError: If neither config_file nor config_dict is provided.
    """
    if config_file is None and config_dict is None:
        raise ValueError("Either config_file or config_dict must be provided")

    # Load configuration from file if provided
    if config_file is not None:
        import yaml

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = config_dict

    # Create and run RedGEM
    redgem = RedGEM(config)
    redgem.run()

    return redgem


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RedGEMPy: Reduce genome-scale metabolic models"
    )
    parser.add_argument(
        "-c", "--config", help="Path to configuration file", required=True
    )
    args = parser.parse_args()

    # Run RedGEM with the provided configuration
    run_redgem(config_file=args.config)
