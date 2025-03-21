# -*- coding: utf-8 -*-
"""
Thermodynamic cobra_model class and methods definition for pyTFA
"""

import re
from math import log
import pandas as pd
from cobra import Model

from pytfa.thermo.metabolite import MetaboliteThermo
from pytfa.thermo.reaction import calcDGtpt_rhs, calcDGR_cues, get_debye_huckel_b
from pytfa.thermo.utils import (
    check_reaction_balance,
    check_transport_reaction,
    find_transported_mets,
)
from pytfa.optim.constraints import (
    SimultaneousUse,
    NegativeDeltaG,
    BackwardDeltaGCoupling,
    ForwardDeltaGCoupling,
    BackwardDirectionCoupling,
    ForwardDirectionCoupling,
    DisplacementCoupling,
)
from pytfa.optim.variables import (
    ThermoDisplacement,
    DeltaGstd,
    DeltaG,
    ForwardUseVariable,
    BackwardUseVariable,
    LogConcentration,
)
from pytfa.utils import numerics
from pytfa.utils.logger import get_bistream_logger

from libs.pytfa.core.model import LCSBModel
from libs.pytfa.thermo import std

BIGM = numerics.BIGM
BIGM_THERMO = numerics.BIGM_THERMO
BIGM_DG = numerics.BIGM_DG
BIGM_P = numerics.BIGM_P
EPSILON = numerics.EPSILON
MAX_STOICH = 10


class ThermoModel(LCSBModel, Model):
    """
    A class representing a cobra_model with thermodynamics information.

    This class uses:
      - self.compartments_thermo to store thermodynamic data (pH, ionicStr, c_min, c_max)
      - self.compartments for the original COBRA compartments (mapping IDs -> names)
    """

    def __init__(
        self,
        thermo_data=None,
        model=Model(),
        name=None,
        temperature=std.TEMPERATURE_0,
        min_ph=std.MIN_PH,
        max_ph=std.MAX_PH,
    ):
        """
        :param float temperature: the temperature (K) at which to perform the calculations
        :param dict thermo_data: The thermodynamic database (can be None or empty)
        """
        LCSBModel.__init__(self, model, name)

        self.logger = get_bistream_logger("ME model" + str(self.name))

        self.TEMPERATURE = temperature
        # If thermo_data is None or empty, store an empty dict
        self.thermo_data = thermo_data if thermo_data else {}
        self.parent = model

        # We keep the original COBRA compartments in self.compartments
        # We store thermodynamic compartments in self.compartments_thermo
        self.compartments_thermo = {}  # assigned externally by apply_compartment_data

        # CONSTANTS
        self.MAX_pH = max_ph
        self.MIN_pH = min_ph

        self._init_thermo()

        self.logger.info(
            "# Model initialized with units {} and temperature {} K".format(
                self.thermo_unit, self.TEMPERATURE
            )
        )

    def _init_thermo(self):
        """
        Initializes thermodynamic data, handling cases where thermo_data is missing or empty.
        """
        # If thermo_data is empty, these get safe defaults
        self.thermo_unit = self.thermo_data.get("units", "kJ/mol")
        self.reaction_cues_data = self.thermo_data.get("cues", [])
        self.compounds_data = self.thermo_data.get("metabolites", {})

        self.Debye_Huckel_B = get_debye_huckel_b(self.TEMPERATURE)
        self.logger = get_bistream_logger("thermomodel_" + str(self.name))

        # Compute internal values to adapt to the thermo_unit provided
        if self.thermo_unit == "kJ/mol":
            self.GAS_CONSTANT = 8.314472 / 1000  # kJ/(K mol)
            self.Adjustment = 1
        else:
            self.GAS_CONSTANT = 1.9858775 / 1000  # Kcal/(K mol)
            self.Adjustment = 4.184

        self.RT = self.GAS_CONSTANT * self.TEMPERATURE

    def normalize_reactions(self):
        """
        Find reactions with large stoichiometry and normalize them
        """
        self.logger.info("# Model normalization")

        for rxn in self.reactions:
            metabolites = rxn.metabolites
            max_stoichiometry = max(metabolites.values())
            if max_stoichiometry > MAX_STOICH:
                new_mets = {
                    m: -coeff + coeff / max_stoichiometry
                    for m, coeff in metabolites.items()
                }
                rxn.add_metabolites(new_mets)

    def _prepare_metabolite(self, met):
        """
        Prepares metabolite thermodynamic data using compartments_thermo if available.
        If compartments_thermo is empty or the specific compartment isn't found,
        we skip or assign defaults.
        """
        # If compartments_thermo is empty, we cannot set pH, ionicStr, etc.
        # We'll skip or assign default.
        if met.compartment not in self.compartments_thermo:
            self.logger.debug(
                f"Compartment '{met.compartment}' not in compartments_thermo, assigning defaults"
            )
            self.compartments_thermo[met.compartment] = {
                "pH": 7.0,
                "ionicStr": 0.1,
                "c_min": 1e-6,
                "c_max": 0.02,
            }

        cdata = self.compartments_thermo[met.compartment]
        pH = cdata.get("pH", 7.0)
        ionicStr = cdata.get("ionicStr", 0.1)

        # Build the MetaboliteThermo if we have annotation
        metData = None
        if "seed_id" in met.annotation:
            seed_id = met.annotation["seed_id"]
            if seed_id in self.compounds_data:
                metData = self.compounds_data[seed_id]
                # Possibly override formula
                if "formula" in metData:
                    met.formula = metData["formula"]
            else:
                self.logger.debug(
                    f"Metabolite {met.id} not present in self.compounds_data"
                )

        met.thermo = MetaboliteThermo(
            metData,
            pH,
            ionicStr,
            self.TEMPERATURE,
            self.MIN_pH,
            self.MAX_pH,
            self.Debye_Huckel_B,
            self.thermo_unit,
        )

    def _prepare_reaction(self, reaction, null_error_override=2):
        """
        Checks reaction for transport, balances, determines deltaG, etc.
        Skips or sets large defaults if not feasible.
        """
        DeltaGrxn = 0
        DeltaGRerr = 0

        # Drain or not
        NotDrain = len(reaction.metabolites) > 1

        reaction.thermo = {"isTrans": False, "computed": False}

        # Determine reaction compartment from first metabolite
        reaction.compartment = None
        for met in reaction.metabolites:
            if reaction.compartment is None:
                reaction.compartment = met.compartment
            elif met.compartment != reaction.compartment:
                reaction.compartment = "c"

        # Check if balanced
        # We need self._proton_of for proton
        if not hasattr(self, "_proton_of"):
            # If we never identified a proton, skip
            self._proton_of = {}

        balanceResult = check_reaction_balance(
            reaction, self._proton_of.get(reaction.compartment, None)
        )

        # Mark as transport if relevant
        reaction.thermo["isTrans"] = check_transport_reaction(reaction)

        # Ensure all mets have valid deltaGf
        correctThermoValues = True
        for met in reaction.metabolites:
            if getattr(met, "thermo", None) is None:
                # If not prepared => skip
                correctThermoValues = False
                break
            if met.thermo.deltaGf_std > 0.9 * BIGM_DG:
                correctThermoValues = False
                break

        # If single-met, unbalanced, or big => skip
        if (
            not NotDrain
            or not correctThermoValues
            or len(reaction.metabolites) >= 100
            or balanceResult in ["missing atoms", "drain flux"]
        ):
            self.logger.debug(f"{reaction.id}: thermo constraint NOT created.")
            reaction.thermo["deltaGR"] = BIGM_DG
            reaction.thermo["deltaGRerr"] = BIGM_DG
            return

        # Mark computed
        reaction.thermo["computed"] = True
        self.logger.debug(f"{reaction.id}: thermo constraint created.")

        # If transport-based
        if reaction.thermo["isTrans"]:
            rhs, breakdown = calcDGtpt_rhs(
                reaction, self.compartments_thermo, self.thermo_unit
            )
            reaction.thermo["deltaGR"] = rhs
            reaction.thermo["deltaGrxn"] = breakdown["sum_deltaGFis"]
        else:
            # Summation of deltaGf
            for met, coeff in reaction.metabolites.items():
                if met.formula == "H" and met.annotation.get("seed_id") == "cpd00067":
                    # skip protons
                    continue
                DeltaGrxn += coeff * met.thermo.deltaGf_tr
                DeltaGRerr += abs(coeff * met.thermo.deltaGf_err)

            reaction.thermo["deltaGR"] = DeltaGrxn

        # Reaction cues
        if self.reaction_cues_data:
            _, cues_err, _, _ = calcDGR_cues(reaction, self.reaction_cues_data)
            if cues_err != 0:
                DeltaGRerr = cues_err
        else:
            self.logger.debug(
                f"No reaction cues data found for {reaction.id}; skipping calcDGR_cues"
            )

        # If error is zero, override
        if DeltaGRerr == 0 and null_error_override:
            DeltaGRerr = null_error_override
        reaction.thermo["deltaGRerr"] = DeltaGRerr

    def prepare(self, null_error_override=2):
        """
        Prepares a COBRA toolbox model for TFBA:
          1) Assigns metabolite thermodynamics
          2) Identifies proton for each compartment
          3) Assigns reaction thermodynamics
        """

        self.logger.info("# Model preparation starting...")

        # Prepare all metabolites
        for met in self.metabolites:
            self._prepare_metabolite(met)

        # Identify protons
        self._proton_of = {}
        for met in self.metabolites:
            if met.formula == "H" or (
                "seed_id" in met.annotation and met.annotation["seed_id"] == "cpd00067"
            ):
                self._proton_of[met.compartment] = met

        if not self._proton_of:
            raise Exception("Cannot find proton (cpd00067/H) in the model.")

        # Prepare each reaction
        for rxn in self.reactions:
            self._prepare_reaction(rxn, null_error_override)

        self.logger.info("# Model preparation done.")

    def _convert_metabolite(self, met, add_potentials, verbose):
        """
        Conditionally creates the LogConcentration & potential variables
        ONLY if compartments_thermo and thermo_data are non-empty.
        Otherwise, skip.
        """

        # Always check if we have ANY thermo data or compartments_thermo
        if not self.thermo_data or not self.compartments_thermo:
            if verbose:
                self.logger.debug(
                    f"Skipping thermo variable creation for {met.id}; no compartments_thermo or thermo_data."
                )
            return

        # Ensure the compartment entry exists
        if met.compartment not in self.compartments_thermo:
            if verbose:
                self.logger.debug(
                    f"Skipping {met.id}; compartment '{met.compartment}' not in compartments_thermo"
                )
            return

        cdata = self.compartments_thermo[met.compartment]

        pH = cdata.setdefault("pH", 7.0)
        ionicStr = cdata.setdefault("ionicStr", 0.1)
        c_min = cdata.setdefault("c_min", 1e-6)
        c_max = cdata.setdefault("c_max", 0.02)

        metformula = met.formula
        metDeltaGF = getattr(met.thermo, "deltaGf_tr", BIGM_DG)

        # Build log concentration bounds
        lb = log(c_min)
        ub = log(c_max)

        LC = None
        if metformula == "H2O":
            # water => log(1)
            LC = self.add_variable(LogConcentration, met, lb=0, ub=0)

        elif metformula == "H":
            # proton => log(10^-pH)
            LC = self.add_variable(
                LogConcentration, met, lb=log(10**-pH), ub=log(10**-pH)
            )

        elif "seed_id" in met.annotation and met.annotation["seed_id"] == "cpd11416":
            # skip biomass metabolite
            pass

        elif metDeltaGF < 1e6:
            if verbose:
                self.logger.debug(f"generating thermo vars for {met.id}")
            LC = self.add_variable(LogConcentration, met, lb=lb, ub=ub)

            if add_potentials:
                P = self.add_variable(f"P_{met.id}", -BIGM_P, BIGM_P)
                self.P_vars[met] = P
                expr = P - self.RT * LC
                self.add_constraint(
                    LogConcentration, f"P_{met.id}", expr, metDeltaGF, metDeltaGF
                )
        else:
            self.logger.debug(f"NOT generating thermo vars for {met.id}")

        if LC is not None:
            self.LC_vars[met] = LC

    def _convert_reaction(self, rxn, add_potentials, add_displacement, verbose):
        """
        Creates minimal constraints (FU_rxn, BU_rxn, SimultaneousUse, flux coupling)
        for ALL reactions. Then, if compartments_thermo and thermo_data are non-empty
        AND rxn.thermo["computed"] is True, we create full thermo constraints
        (DeltaG, NegativeDeltaG, etc.).
        """

        # Part A: Always create the directionality constraints (FU/BU + usage constraints)
        FU_rxn = self.add_variable(ForwardUseVariable, rxn)
        BU_rxn = self.add_variable(BackwardUseVariable, rxn)

        # Prevent simultaneous use
        CLHS = FU_rxn + BU_rxn
        self.add_constraint(SimultaneousUse, rxn, CLHS, ub=1)

        # Link flux to direction usage
        F_rxn = rxn.forward_variable
        R_rxn = rxn.reverse_variable

        # Forward flux => F_rxn - M * FU_rxn <= 0
        CLHS = F_rxn - FU_rxn * BIGM
        self.add_constraint(ForwardDirectionCoupling, rxn, CLHS, ub=0)

        # Reverse flux => R_rxn - M * BU_rxn <= 0
        CLHS = R_rxn - BU_rxn * BIGM
        self.add_constraint(BackwardDirectionCoupling, rxn, CLHS, ub=0)

        # Part B: Only add full thermodynamic constraints if:
        #  1) rxn.thermo["computed"] is True
        #  2) compartments_thermo is not empty
        #  3) self.thermo_data is not empty
        if (
            rxn.thermo.get("computed", False)
            and self.compartments_thermo
            and self.thermo_data
        ):
            # Check if this is water transport
            H2OtRxns = False
            if rxn.thermo["isTrans"] and len(rxn.reactants) == 1:
                try:
                    if rxn.reactants[0].annotation["seed_id"] == "cpd00001":
                        H2OtRxns = True
                except KeyError:
                    pass

            if H2OtRxns:
                if verbose:
                    self.logger.debug(
                        f"Skipping full thermo constraints for water transport {rxn.id}"
                    )
                return

            if verbose:
                self.logger.debug(f"generating full thermo constraint for {rxn.id}")

            # DeltaG variable
            DGR = self.add_variable(DeltaG, rxn, lb=-BIGM_THERMO, ub=BIGM_THERMO)

            # DeltaGstd variable
            RxnDGerr = rxn.thermo["deltaGRerr"]
            DGoR = self.add_variable(
                DeltaGstd,
                rxn,
                lb=rxn.thermo["deltaGR"] - RxnDGerr,
                ub=rxn.thermo["deltaGR"] + RxnDGerr,
            )

            LC_TransMet = 0
            LC_ChemMet = 0
            P_expr = 0
            RT = self.RT

            if rxn.thermo["isTrans"]:
                transportedMets = find_transported_mets(rxn)
                chem_stoich = rxn.metabolites.copy()

                # handle transport
                for seed_id, trans in transportedMets.items():
                    for ttype in ["reactant", "product"]:
                        if trans[ttype].formula != "H":
                            LC_TransMet += (
                                self.LC_vars[trans[ttype]]
                                * RT
                                * trans["coeff"]
                                * (-1 if ttype == "reactant" else 1)
                            )
                        chem_stoich[trans[ttype]] += trans["coeff"] * (
                            1 if ttype == "reactant" else -1
                        )

                # Filter zero stoich
                chem_stoich = {m: s for m, s in chem_stoich.items() if s != 0}
                for m, coeff in chem_stoich.items():
                    if m.formula not in ["H", "H2O"]:
                        LC_ChemMet += self.LC_vars[m] * RT * coeff
            else:
                # Standard chemical reaction
                if add_potentials:
                    RHS_DG = 0
                    for met, coeff in rxn.metabolites.items():
                        if met.formula == "H2O":
                            RHS_DG += coeff * met.thermo.deltaGf_tr
                        elif met.formula != "H":
                            P_expr += self.P_vars[met] * coeff
                else:
                    RHS_DG = rxn.thermo["deltaGR"]
                    for m, coeff in rxn.metabolites.items():
                        if m.formula not in ["H", "H2O"]:
                            LC_ChemMet += self.LC_vars[m] * RT * coeff

            # NegativeDeltaG constraint: -DGR + DGoR + (LC_TransMet + LC_ChemMet) = 0
            CLHS = DGoR - DGR + LC_TransMet + LC_ChemMet
            self.add_constraint(NegativeDeltaG, rxn, CLHS, lb=0, ub=0)

            # Optional displacement
            if add_displacement:
                lngamma = self.add_variable(
                    ThermoDisplacement, rxn, lb=-BIGM_P, ub=BIGM_P
                )
                expr = lngamma - (1 / RT) * DGR
                self.add_constraint(DisplacementCoupling, rxn, expr, lb=0, ub=0)

            # Coupling DeltaG with direction usage:
            # Forward: DGR + 1000 * FU_rxn <= 1000 - epsilon
            CLHS = DGR + FU_rxn * BIGM_THERMO
            self.add_constraint(
                ForwardDeltaGCoupling, rxn, CLHS, ub=BIGM_THERMO - EPSILON
            )

            # Backward: 1000 * BU_rxn - DGR <= 1000 - epsilon
            CLHS = BU_rxn * BIGM_THERMO - DGR
            self.add_constraint(
                BackwardDeltaGCoupling, rxn, CLHS, ub=BIGM_THERMO - EPSILON
            )
        else:
            if verbose:
                self.logger.debug(
                    f"Skipping full thermo constraints for {rxn.id}, only directionality constraints."
                )

    def convert(self, add_potentials=False, add_displacement=False, verbose=True):
        """
        Converts a cobra_model into a tFBA-ready cobra_model by adding the
        necessary thermodynamic constraints.

        If self.compartments_thermo or self.thermo_data is empty, the code will
        skip full thermodynamic constraints (DeltaG, NegativeDeltaG, etc.) and
        only create directionality constraints (FU, BU, SimultaneousUse).

        :warning: Must run prepare() first.
        """
        self.logger.info("# Model conversion starting...")

        bigM = BIGM
        for rxn in self.reactions:
            if rxn.lower_bound < -bigM - EPSILON or rxn.upper_bound > bigM + EPSILON:
                raise Exception("flux bounds too wide or big M not big enough")
            rxn.lower_bound = max(rxn.lower_bound, -bigM)
            rxn.upper_bound = min(rxn.upper_bound, bigM)

        # Check if reaction.thermo is present
        try:
            for rxn in self.reactions:
                if "isTrans" not in rxn.thermo:
                    rxn.thermo["isTrans"] = check_transport_reaction(rxn)
        except:
            raise Exception("Reaction thermo data missing. Please run prepare() first.")

        # Clean bracket-like chars in IDs
        replacements = {"_": re.compile(r"[\[\(]"), "": re.compile(r"[\]\)]")}
        for items in [self.metabolites, self.reactions]:
            for item in items:
                for rep, pattern in replacements.items():
                    item.name = re.sub(pattern, rep, item.name)

        self.LC_vars = {}
        self.P_vars = {}

        # Convert each metabolite
        for met in self.metabolites:
            self._convert_metabolite(met, add_potentials, verbose)

        # Convert each reaction
        for rxn in self.reactions:
            self._convert_reaction(rxn, add_potentials, add_displacement, verbose)

        # If no objective, warn
        if len(self.objective.variables) == 0:
            self.logger.warning("Objective not found")

        self.logger.info("# Model conversion done.")
        self.logger.info("# Updating cobra_model variables...")
        self.repair()
        self.logger.info("# cobra_model variables are up-to-date")

    def print_info(self, specific=False):
        """
        Print information and counts for the cobra_model
        """
        if not specific:
            LCSBModel.print_info(self)

        n_mets = len(self.metabolites)
        n_rxns = len(self.reactions)
        n_mets_thermo = len(
            [x for x in self.metabolites if hasattr(x, "thermo") and x.thermo.get("id")]
        )
        n_rxns_thermo = len(
            [
                x
                for x in self.reactions
                if x.id and hasattr(x, "thermo") and x.thermo.get("computed")
            ]
        )

        info = pd.DataFrame(columns=["value"])
        info.loc["num metabolites(thermo)"] = n_mets_thermo
        info.loc["num reactions(thermo)"] = n_rxns_thermo
        if n_mets > 0:
            info.loc["pct metabolites(thermo)"] = n_mets_thermo / n_mets * 100
        else:
            info.loc["pct metabolites(thermo)"] = 0

        if n_rxns > 0:
            info.loc["pct reactions(thermo)"] = n_rxns_thermo / n_rxns * 100
        else:
            info.loc["pct reactions(thermo)"] = 0
        info.index.name = "key"

        print(info)

    def __deepcopy__(self, memo):
        """
        Use the model's copy method instead of python's built-in.
        """
        return self.copy()

    def copy(self):
        """
        Creates a new ThermoModel from the current one (deep copy).
        """
        from ..io.dict import model_from_dict, model_to_dict
        from ..optim.utils import copy_solver_configuration

        dictmodel = model_to_dict(self)
        new_model = model_from_dict(dictmodel)
        copy_solver_configuration(self, new_model)
        return new_model
